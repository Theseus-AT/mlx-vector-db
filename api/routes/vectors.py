"""
Optimized FastAPI Vector Operations
Integrated with MLX Vector Store for maximum performance

Key Optimizations:
- Async batch processing
- Connection pooling for stores
- Response streaming for large results
- Automatic MLX kernel warmup
- Memory-efficient request handling
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
import asyncio
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from service.vector_store import MLXVectorStore, VectorStoreConfig, benchmark_vector_store
from security.auth import verify_api_key
from service.models import VectorQuery, VectorAddRequest, BatchQueryRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vectors", tags=["vectors"])

# Global store manager for connection pooling
class VectorStoreManager:
    """Manages MLX Vector Store instances with connection pooling"""
    
    def __init__(self):
        self._stores: Dict[str, MLXVectorStore] = {}
        self._configs: Dict[str, VectorStoreConfig] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)  # CPU-bound ops
        
    def get_store_key(self, user_id: str, model_id: str) -> str:
        return f"{user_id}_{model_id}"
    
    async def get_store(self, user_id: str, model_id: str, 
                       config: Optional[VectorStoreConfig] = None) -> MLXVectorStore:
        """Get or create vector store with async initialization"""
        store_key = self.get_store_key(user_id, model_id)
        
        if store_key not in self._stores:
            if config is None:
                config = VectorStoreConfig()  # Default config
            
            store_path = f"~/.team_mind_data/vector_stores/{user_id}/{model_id}"
            
            # Initialize store in thread pool (I/O bound)
            loop = asyncio.get_event_loop()
            store = await loop.run_in_executor(
                self._executor, 
                lambda: MLXVectorStore(store_path, config)
            )
            
            self._stores[store_key] = store
            self._configs[store_key] = config
            
            logger.info(f"✅ Initialized MLX store for {user_id}/{model_id}")
        
        return self._stores[store_key]
    
    async def warmup_all_stores(self):
        """Warm up all active stores for optimal performance"""
        for store_key, store in self._stores.items():
            logger.info(f"🔥 Warming up store: {store_key}")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, store._warmup_kernels)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics for all stores"""
        total_vectors = 0
        total_memory = 0.0
        store_count = len(self._stores)
        
        for store in self._stores.values():
            stats = store.get_stats()
            total_vectors += stats['vector_count']
            total_memory += stats['memory_usage_mb']
        
        return {
            'total_stores': store_count,
            'total_vectors': total_vectors,
            'total_memory_mb': total_memory,
            'mlx_optimized': True,
            'unified_memory': True
        }

# Global store manager instance
store_manager = VectorStoreManager()


# Pydantic models for request/response
class VectorAddResponse(BaseModel):
    success: bool
    vectors_added: int
    total_vectors: int
    processing_time_ms: float
    store_stats: Dict[str, Any]


class VectorQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_time_ms: float
    total_vectors_searched: int
    metadata_filter_applied: bool


class BatchQueryResponse(BaseModel):
    results: List[List[Dict[str, Any]]]
    total_queries: int
    avg_query_time_ms: float
    total_processing_time_ms: float


class StoreStatsResponse(BaseModel):
    store_stats: Dict[str, Any]
    performance_metrics: Dict[str, float]
    mlx_info: Dict[str, Any]


# Dependency injection
async def get_vector_store(user_id: str, model_id: str) -> MLXVectorStore:
    """Dependency to get vector store instance"""
    return await store_manager.get_store(user_id, model_id)


@router.post("/add", response_model=VectorAddResponse)
async def add_vectors(
    request: VectorAddRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Add vectors to the store with MLX optimization
    Supports both single and batch vector addition
    """
    start_time = time.time()
    
    try:
        # Get or create store
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        # Validate input
        if not request.vectors or not request.metadata:
            raise HTTPException(status_code=400, detail="Vectors and metadata required")
        
        if len(request.vectors) != len(request.metadata):
            raise HTTPException(
                status_code=400, 
                detail="Number of vectors must match number of metadata entries"
            )
        
        # Convert to numpy for MLX optimization
        vectors_np = np.array(request.vectors, dtype=np.float32)
        
        # Add vectors in thread pool (CPU-bound operation)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            store_manager._executor,
            lambda: store.add_vectors(vectors_np, request.metadata)
        )
        
        # Schedule background optimization
        background_tasks.add_task(optimize_store_background, store)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VectorAddResponse(
            success=True,
            vectors_added=len(request.vectors),
            total_vectors=store.get_stats()['vector_count'],
            processing_time_ms=processing_time,
            store_stats=store.get_stats()
        )
        
    except Exception as e:
        logger.error(f"Error adding vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add vectors: {str(e)}")


# api/routes/vectors.py - Ersetze die query Funktion (ca. Zeile 190-250)

@router.post("/query", response_model=VectorQueryResponse)  
async def query_vectors(
    request: VectorQuery,
    api_key: str = Depends(verify_api_key)
):
    """
    High-performance vector similarity search using MLX
    """
    start_time = time.time()
    
    try:
        # Get store
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        # Validate query vector
        if not request.query:
            raise HTTPException(status_code=400, detail="Query vector required")
        
        # Perform similarity search in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            store_manager._executor,
            lambda: store.query(
                request.query, 
                k=request.k,
                filter_metadata=request.filter_metadata
            )
        )
        
        # Format results - handle new return format (indices, distances, metadata)
        formatted_results = []
        if isinstance(results, tuple) and len(results) == 3:
            indices, distances, metadata_list = results
            for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata_list)):
                formatted_results.append({
                    "metadata": meta,
                    "similarity_score": float(1.0 - dist) if store.config.metric == "cosine" else float(-dist),
                    "rank": i + 1
                })
        else:
            # Legacy format compatibility
            for metadata, score in results:
                formatted_results.append({
                    "metadata": metadata,
                    "similarity_score": float(score),
                    "rank": len(formatted_results) + 1
                })
        
        query_time = (time.time() - start_time) * 1000
        
        return VectorQueryResponse(
            results=formatted_results,
            query_time_ms=query_time,
            total_vectors_searched=store.get_stats()['vector_count'],
            metadata_filter_applied=request.filter_metadata is not None
        )
        
    except Exception as e:
        logger.error(f"Error querying vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.post("/batch_query", response_model=BatchQueryResponse)
async def batch_query_vectors(
    request: BatchQueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Optimized batch query processing for maximum throughput
    """
    start_time = time.time()
    
    try:
        # Get store
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        if not request.queries:
            raise HTTPException(status_code=400, detail="Query vectors required")
        
        # Process batch queries
        loop = asyncio.get_event_loop()
        batch_results = await loop.run_in_executor(
            store_manager._executor,
            lambda: store.batch_query(request.queries, k=request.k)
        )
        
        # Format results
        formatted_batch_results = []
        total_query_time = 0
        
        for query_results in batch_results:
            formatted_query_results = []
            for metadata, score in query_results:
                formatted_query_results.append({
                    "metadata": metadata,
                    "similarity_score": float(score)
                })
            formatted_batch_results.append(formatted_query_results)
        
        total_processing_time = (time.time() - start_time) * 1000
        avg_query_time = total_processing_time / len(request.queries)
        
        return BatchQueryResponse(
            results=formatted_batch_results,
            total_queries=len(request.queries),
            avg_query_time_ms=avg_query_time,
            total_processing_time_ms=total_processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch query failed: {str(e)}")


@router.get("/count")
async def get_vector_count(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get vector count for a specific store"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        return {"count": store.get_stats()['vector_count']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StoreStatsResponse)
async def get_store_stats(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed store statistics and performance metrics"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        stats = store.get_stats()
        
        # Additional performance metrics
        performance_metrics = {
            "expected_qps": 1000,  # Based on MLX optimizations
            "memory_efficiency": "unified_memory",
            "acceleration": "metal_kernels",
            "compilation": "jit_enabled"
        }
        
        mlx_info = {
            "mlx_version": "0.25.2",
            "device": stats.get('mlx_device', 'unknown'),
            "unified_memory": stats.get('unified_memory', True),
            "metal_available": True
        }
        
        return StoreStatsResponse(
            store_stats=stats,
            performance_metrics=performance_metrics,
            mlx_info=mlx_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream_query")
async def stream_query_results(
    request: VectorQuery,
    api_key: str = Depends(verify_api_key)
):
    """
    Stream query results for large result sets
    Useful for real-time applications
    """
    async def generate_results():
        try:
            store = await store_manager.get_store(request.user_id, request.model_id)
            
            # Get results
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                store_manager._executor,
                lambda: store.query(request.query, k=request.k)
            )
            
            # Stream results one by one
            for i, (metadata, score) in enumerate(results):
                result = {
                    "metadata": metadata,
                    "similarity_score": float(score),
                    "rank": i + 1
                }
                yield f"data: {json.dumps(result)}\n\n"
                
                # Small delay for real-time feel
                await asyncio.sleep(0.01)
                
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
            
        except Exception as e:
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_results(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )


# Background tasks
async def optimize_store_background(store: MLXVectorStore):
    """Background task to optimize store performance"""
    try:
        await asyncio.sleep(1)  # Small delay to not interfere with request
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, store.optimize)
        logger.info("✅ Store optimization completed")
    except Exception as e:
        logger.error(f"Store optimization failed: {e}")


# Startup event handlers
@router.on_event("startup")
async def startup_warmup():
    """Warm up MLX kernels on application startup"""
    logger.info("🚀 Starting MLX Vector API...")
    await store_manager.warmup_all_stores()
    logger.info("✅ MLX Vector API ready")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for the vector service"""
    try:
        global_stats = store_manager.get_stats()
        return {
            "status": "healthy",
            "mlx_optimized": True,
            "stores_active": global_stats['total_stores'],
            "total_vectors": global_stats['total_vectors'],
            "memory_usage_mb": global_stats['total_memory_mb']
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }


# Performance testing endpoint
@router.post("/benchmark")
async def run_benchmark(
    user_id: str,
    model_id: str,
    num_vectors: int = 1000,
    num_queries: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """Run performance benchmark on the vector store"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        
        # Run benchmark in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            store_manager._executor,
            lambda: benchmark_vector_store(store, num_vectors, num_queries)
        )
        
        return {
            "benchmark_results": results,
            "mlx_optimized": True,
            "performance_target": "1000+ QPS"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")