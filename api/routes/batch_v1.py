## api/routes/batch_v1.py
"""
Production Batch Operations API für MLX Vector Database
Optimiert für High-Throughput Workloads mit Streaming Support
"""

import asyncio
import json
import time
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
import numpy as np
import mlx.core as mx
from pathlib import Path
import tempfile
import aiofiles

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from sse_starlette.sse import EventSourceResponse

from security.auth import verify_api_key
from service.optimized_vector_store import MLXVectorStoreOptimized
from api.routes.vectors import store_manager

logger = logging.getLogger("mlx_batch_api")
router = APIRouter(prefix="/v1/batch", tags=["batch-v1"])

# =================== BATCH MODELS ===================

class BatchVectorAddRequest(BaseModel):
    """Request for batch vector addition"""
    user_id: str
    model_id: str
    vectors: List[List[float]] = Field(..., min_items=1, max_items=10000)
    metadata: List[Dict[str, Any]]
    batch_size: Optional[int] = Field(default=1000, ge=100, le=5000)
    enable_streaming: bool = False
    
    @validator('vectors', 'metadata')
    def validate_equal_length(cls, v, values):
        if 'vectors' in values and len(v) != len(values['vectors']):
            raise ValueError('Vectors and metadata must have equal length')
        return v

class BatchQueryRequest(BaseModel):
    """Request for batch queries"""
    user_id: str
    model_id: str
    queries: List[List[float]] = Field(..., min_items=1, max_items=1000)
    k: int = Field(default=10, ge=1, le=100)
    batch_size: Optional[int] = Field(default=100, ge=10, le=500)
    enable_streaming: bool = False
    filter_metadata: Optional[Dict[str, Any]] = None

class BulkUploadRequest(BaseModel):
    """Request for bulk file upload"""
    user_id: str
    model_id: str
    file_format: str = Field(..., regex="^(npz|jsonl|csv)$")
    chunk_size: int = Field(default=5000, ge=1000, le=50000)
    enable_progress_stream: bool = True

class BatchProgress(BaseModel):
    """Progress update for batch operations"""
    operation_id: str
    operation_type: str
    progress_percent: float
    items_processed: int
    total_items: int
    current_batch: int
    total_batches: int
    elapsed_time_ms: float
    estimated_remaining_ms: Optional[float] = None
    errors: List[str] = []

class BatchResult(BaseModel):
    """Final result for batch operations"""
    operation_id: str
    success: bool
    total_processed: int
    total_errors: int
    processing_time_ms: float
    throughput_items_per_sec: float
    memory_peak_mb: float
    summary: Dict[str, Any]

# =================== BATCH OPERATION MANAGER ===================

class BatchOperationManager:
    """Manages long-running batch operations"""
    
    def __init__(self):
        self._operations: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def start_operation(self, operation_id: str, operation_type: str, 
                            total_items: int) -> None:
        """Start tracking a batch operation"""
        async with self._lock:
            self._operations[operation_id] = {
                'operation_type': operation_type,
                'total_items': total_items,
                'items_processed': 0,
                'start_time': time.time(),
                'status': 'running',
                'errors': [],
                'current_batch': 0,
                'total_batches': 0
            }
    
    async def update_progress(self, operation_id: str, items_processed: int,
                            current_batch: int = None, errors: List[str] = None) -> None:
        """Update operation progress"""
        async with self._lock:
            if operation_id in self._operations:
                op = self._operations[operation_id]
                op['items_processed'] = items_processed
                if current_batch is not None:
                    op['current_batch'] = current_batch
                if errors:
                    op['errors'].extend(errors)
    
    async def complete_operation(self, operation_id: str, success: bool = True) -> None:
        """Mark operation as complete"""
        async with self._lock:
            if operation_id in self._operations:
                self._operations[operation_id]['status'] = 'completed' if success else 'failed'
                self._operations[operation_id]['end_time'] = time.time()
    
    async def get_progress(self, operation_id: str) -> Optional[BatchProgress]:
        """Get current progress for operation"""
        async with self._lock:
            if operation_id not in self._operations:
                return None
            
            op = self._operations[operation_id]
            elapsed_time = (time.time() - op['start_time']) * 1000
            progress_percent = (op['items_processed'] / op['total_items']) * 100
            
            # Estimate remaining time
            estimated_remaining = None
            if op['items_processed'] > 0:
                rate = op['items_processed'] / (elapsed_time / 1000)
                remaining_items = op['total_items'] - op['items_processed']
                estimated_remaining = (remaining_items / rate) * 1000 if rate > 0 else None
            
            return BatchProgress(
                operation_id=operation_id,
                operation_type=op['operation_type'],
                progress_percent=progress_percent,
                items_processed=op['items_processed'],
                total_items=op['total_items'],
                current_batch=op['current_batch'],
                total_batches=op['total_batches'],
                elapsed_time_ms=elapsed_time,
                estimated_remaining_ms=estimated_remaining,
                errors=op['errors'][-10:]  # Last 10 errors
            )

# Global batch operation manager
batch_manager = BatchOperationManager()

# =================== STREAMING UTILITIES ===================

async def stream_batch_progress(operation_id: str) -> AsyncGenerator[str, None]:
    """Stream batch operation progress as SSE"""
    while True:
        progress = await batch_manager.get_progress(operation_id)
        if not progress:
            break
        
        # Send progress update
        yield f"data: {progress.json()}\n\n"
        
        # Check if operation completed
        if progress.progress_percent >= 100:
            break
        
        await asyncio.sleep(0.5)  # Update every 500ms

async def process_batch_with_streaming(
    operation_id: str,
    items: List[Any],
    batch_size: int,
    process_func: callable,
    **kwargs
) -> Dict[str, Any]:
    """Process batch items with progress streaming"""
    
    total_items = len(items)
    total_batches = (total_items + batch_size - 1) // batch_size
    
    await batch_manager.start_operation(operation_id, "batch_processing", total_items)
    
    results = []
    errors = []
    start_time = time.time()
    
    try:
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = items[batch_start:batch_end]
            
            try:
                # Process batch
                batch_results = await process_func(batch_items, **kwargs)
                results.extend(batch_results)
                
                # Update progress
                await batch_manager.update_progress(
                    operation_id,
                    batch_end,
                    batch_idx + 1
                )
                
            except Exception as e:
                error_msg = f"Batch {batch_idx + 1} failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        await batch_manager.complete_operation(operation_id, success=len(errors) == 0)
        
        processing_time = (time.time() - start_time) * 1000
        throughput = total_items / (processing_time / 1000) if processing_time > 0 else 0
        
        return {
            'operation_id': operation_id,
            'success': len(errors) == 0,
            'total_processed': len(results),
            'total_errors': len(errors),
            'processing_time_ms': processing_time,
            'throughput_items_per_sec': throughput,
            'results': results,
            'errors': errors
        }
        
    except Exception as e:
        await batch_manager.complete_operation(operation_id, success=False)
        raise

# =================== BATCH ENDPOINTS ===================

@router.post("/vectors/add")
async def batch_add_vectors(
    request: BatchVectorAddRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Batch add vectors with optional streaming progress"""
    
    operation_id = f"add_vectors_{int(time.time() * 1000)}"
    
    async def process_vector_batch(batch_vectors: List[List[float]], 
                                 batch_metadata: List[Dict]) -> List[Dict]:
        """Process a batch of vectors"""
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        # Convert to numpy for MLX
        np_vectors = np.array(batch_vectors, dtype=np.float32)
        
        # Add vectors to store
        result = store.add_vectors(np_vectors, batch_metadata)
        return [result]
    
    if request.enable_streaming:
        # Start background processing
        background_tasks.add_task(
            process_batch_with_streaming,
            operation_id,
            list(zip(request.vectors, request.metadata)),
            request.batch_size,
            lambda batch_items: process_vector_batch(
                [item[0] for item in batch_items],
                [item[1] for item in batch_items]
            )
        )
        
        return {
            "operation_id": operation_id,
            "streaming_enabled": True,
            "progress_url": f"/v1/batch/progress/{operation_id}",
            "stream_url": f"/v1/batch/stream/{operation_id}"
        }
    
    else:
        # Synchronous batch processing
        result = await process_batch_with_streaming(
            operation_id,
            list(zip(request.vectors, request.metadata)),
            request.batch_size,
            lambda batch_items: process_vector_batch(
                [item[0] for item in batch_items],
                [item[1] for item in batch_items]
            )
        )
        
        return result

@router.post("/vectors/query")
async def batch_query_vectors(
    request: BatchQueryRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Batch query vectors with optional streaming results"""
    
    operation_id = f"query_vectors_{int(time.time() * 1000)}"
    
    async def process_query_batch(batch_queries: List[List[float]]) -> List[Dict]:
        """Process a batch of queries"""
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        batch_results = []
        for query_vector in batch_queries:
            indices, distances, metadata = store.query(
                query_vector, 
                k=request.k,
                filter_metadata=request.filter_metadata
            )
            
            # Format results
            query_results = []
            for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata)):
                query_results.append({
                    'index': idx,
                    'distance': dist,
                    'similarity_score': max(0, 1.0 - dist) if store.config.metric == "cosine" else -dist,
                    'metadata': meta,
                    'rank': i + 1
                })
            
            batch_results.append({
                'query_index': len(batch_results),
                'results': query_results
            })
        
        return batch_results
    
    if request.enable_streaming:
        # Start background processing
        background_tasks.add_task(
            process_batch_with_streaming,
            operation_id,
            request.queries,
            request.batch_size,
            process_query_batch
        )
        
        return {
            "operation_id": operation_id,
            "streaming_enabled": True,
            "progress_url": f"/v1/batch/progress/{operation_id}",
            "stream_url": f"/v1/batch/stream/{operation_id}"
        }
    
    else:
        # Synchronous batch processing
        result = await process_batch_with_streaming(
            operation_id,
            request.queries,
            request.batch_size,
            process_query_batch
        )
        
        return result

@router.post("/upload")
async def bulk_upload_vectors(
    request: BulkUploadRequest,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Bulk upload vectors from file with progress streaming"""
    
    operation_id = f"bulk_upload_{int(time.time() * 1000)}"
    
    # Save uploaded file temporarily
    temp_dir = Path(tempfile.gettempdir()) / "mlx_uploads"
    temp_dir.mkdir(exist_ok=True)
    
    temp_file_path = temp_dir / f"{operation_id}_{file.filename}"
    
    async with aiofiles.open(temp_file_path, 'wb') as temp_file:
        content = await file.read()
        await temp_file.write(content)
    
    async def process_bulk_upload():
        """Process bulk upload in background"""
        try:
            store = await store_manager.get_store(request.user_id, request.model_id)
            
            if request.file_format == "npz":
                data = np.load(temp_file_path)
                vectors = data['vectors']
                metadata = [{'id': f'bulk_{i}'} for i in range(len(vectors))]
                
            elif request.file_format == "jsonl":
                vectors = []
                metadata = []
                
                async with aiofiles.open(temp_file_path, 'r') as f:
                    async for line in f:
                        item = json.loads(line.strip())
                        vectors.append(item['vector'])
                        metadata.append(item.get('metadata', {}))
                
                vectors = np.array(vectors, dtype=np.float32)
            
            # Process in chunks
            chunk_size = request.chunk_size
            total_vectors = len(vectors)
            
            await batch_manager.start_operation(operation_id, "bulk_upload", total_vectors)
            
            for i in range(0, total_vectors, chunk_size):
                chunk_vectors = vectors[i:i+chunk_size]
                chunk_metadata = metadata[i:i+chunk_size]
                
                store.add_vectors(chunk_vectors, chunk_metadata)
                
                await batch_manager.update_progress(
                    operation_id,
                    min(i + chunk_size, total_vectors),
                    (i // chunk_size) + 1
                )
            
            await batch_manager.complete_operation(operation_id, success=True)
            
        except Exception as e:
            logger.error(f"Bulk upload failed: {e}")
            await batch_manager.complete_operation(operation_id, success=False)
        
        finally:
            # Cleanup temp file
            try:
                temp_file_path.unlink()
            except:
                pass
    
    # Start background processing
    background_tasks.add_task(process_bulk_upload)
    
    return {
        "operation_id": operation_id,
        "file_size_mb": len(content) / (1024 * 1024),
        "estimated_vectors": len(content) // (4 * 384) if request.file_format == "npz" else "unknown",
        "progress_url": f"/v1/batch/progress/{operation_id}",
        "stream_url": f"/v1/batch/stream/{operation_id}"
    }

# =================== PROGRESS & STREAMING ENDPOINTS ===================

@router.get("/progress/{operation_id}")
async def get_batch_progress(
    operation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get current progress for batch operation"""
    
    progress = await batch_manager.get_progress(operation_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return progress

@router.get("/stream/{operation_id}")
async def stream_batch_progress_sse(
    operation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Stream batch operation progress via Server-Sent Events"""
    
    # Check if operation exists
    progress = await batch_manager.get_progress(operation_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Operation not found")
    
    return EventSourceResponse(stream_batch_progress(operation_id))

@router.get("/operations")
async def list_batch_operations(
    user_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """List all batch operations (optionally filtered by user)"""
    
    # This would be enhanced to filter by user in production
    operations = []
    
    async with batch_manager._lock:
        for op_id, op_data in batch_manager._operations.items():
            if user_id and not op_id.startswith(f"operation_{user_id}"):
                continue
                
            operations.append({
                'operation_id': op_id,
                'operation_type': op_data['operation_type'],
                'status': op_data['status'],
                'progress_percent': (op_data['items_processed'] / op_data['total_items']) * 100,
                'items_processed': op_data['items_processed'],
                'total_items': op_data['total_items'],
                'start_time': op_data['start_time'],
                'errors_count': len(op_data['errors'])
            })
    
    return {
        'operations': operations,
        'total_count': len(operations)
    }

# =================== PERFORMANCE BENCHMARKING ===================

@router.post("/benchmark")
async def run_batch_benchmark(
    num_vectors: int = 10000,
    dimension: int = 384,
    batch_sizes: List[int] = [100, 500, 1000, 2000],
    api_key: str = Depends(verify_api_key)
):
    """Run batch performance benchmark"""
    
    benchmark_results = {}
    test_user = "benchmark_batch_user"
    test_model = "benchmark_batch_model"
    
    # Generate test data
    test_vectors = np.random.rand(num_vectors, dimension).astype(np.float32).tolist()
    test_metadata = [{"id": f"benchmark_{i}"} for i in range(num_vectors)]
    
    for batch_size in batch_sizes:
        logger.info(f"Benchmarking batch size: {batch_size}")
        
        # Clean store
        try:
            await store_manager.delete_store(test_user, test_model)
        except:
            pass
        
        # Create fresh store
        await store_manager.create_store(test_user, test_model)
        
        # Benchmark batch add
        start_time = time.time()
        
        operation_id = f"benchmark_{batch_size}_{int(time.time())}"
        result = await process_batch_with_streaming(
            operation_id,
            list(zip(test_vectors, test_metadata)),
            batch_size,
            lambda batch_items: process_vector_batch(
                [item[0] for item in batch_items],
                [item[1] for item in batch_items]
            )
        )
        
        benchmark_time = time.time() - start_time
        throughput = num_vectors / benchmark_time
        
        benchmark_results[f"batch_size_{batch_size}"] = {
            'processing_time_sec': benchmark_time,
            'throughput_vectors_per_sec': throughput,
            'success': result['success'],
            'errors': len(result['errors'])
        }
        
        # Cleanup
        try:
            await store_manager.delete_store(test_user, test_model)
        except:
            pass
    
    # Find optimal batch size
    best_batch_size = max(
        benchmark_results.keys(),
        key=lambda k: benchmark_results[k]['throughput_vectors_per_sec']
    )
    
    return {
        'benchmark_results': benchmark_results,
        'optimal_batch_size': int(best_batch_size.split('_')[2]),
        'test_parameters': {
            'num_vectors': num_vectors,
            'dimension': dimension,
            'batch_sizes_tested': batch_sizes
        },
        'recommendations': {
            'small_datasets': "Use batch_size=100-500 for <1K vectors",
            'medium_datasets': "Use batch_size=1000-2000 for 1K-10K vectors",
            'large_datasets': "Use batch_size=2000+ for >10K vectors"
        }
    }

# =================== WEBHOOK NOTIFICATIONS ===================

@dataclass
class WebhookConfig:
    """Webhook configuration for batch completion notifications"""
    url: str
    headers: Dict[str, str] = None
    retry_attempts: int = 3
    timeout_seconds: int = 30

async def send_webhook_notification(webhook_config: WebhookConfig, 
                                  operation_result: Dict[str, Any]) -> bool:
    """Send webhook notification when batch operation completes"""
    import httpx
    
    headers = webhook_config.headers or {}
    headers.setdefault("Content-Type", "application/json")
    headers.setdefault("User-Agent", "MLX-VectorDB-Webhook/1.0")
    
    payload = {
        "event": "batch_operation_completed",
        "timestamp": time.time(),
        "operation_result": operation_result
    }
    
    async with httpx.AsyncClient() as client:
        for attempt in range(webhook_config.retry_attempts):
            try:
                response = await client.post(
                    webhook_config.url,
                    json=payload,
                    headers=headers,
                    timeout=webhook_config.timeout_seconds
                )
                
                if response.status_code == 200:
                    logger.info(f"Webhook notification sent successfully to {webhook_config.url}")
                    return True
                else:
                    logger.warning(f"Webhook returned {response.status_code}: {response.text}")
                    
            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} failed: {e}")
                
                if attempt < webhook_config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"All webhook attempts failed for {webhook_config.url}")
    return False