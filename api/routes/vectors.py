"""
Optimized FastAPI Vector Operations
Integrated with MLX Vector Store for maximum performance
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

from service.optimized_vector_store import MLXVectorStoreConfig, MLXVectorStore
from security.auth import verify_api_key
from service.models import VectorQuery, VectorAddRequest, BatchQueryRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vectors", tags=["vectors"])

# Global store manager for connection pooling
class VectorStoreManager:
    """Manages MLX Vector Store instances with connection pooling"""
    
    def __init__(self):
        self._stores: Dict[str, MLXVectorStore] = {}
        self._configs: Dict[str, MLXVectorStoreConfig] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        
    def get_store_key(self, user_id: str, model_id: str) -> str:
        return f"{user_id}_{model_id}"
    
    async def get_store(self, user_id: str, model_id: str, 
                       config: Optional[MLXVectorStoreConfig] = None) -> MLXVectorStore:
        """Get or create vector store with async initialization"""
        store_key = self.get_store_key(user_id, model_id)
        
        if store_key not in self._stores:
            if config is None:
                config = MLXVectorStoreConfig()
            
            store_path = f"~/.team_mind_data/vector_stores/{user_id}/{model_id}"
            
            # Initialize store in thread pool
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
            total_vectors += stats.get('vector_count', 0)
            total_memory += stats.get('memory_usage_mb', 0)
        
        return {
            'total_stores': store_count,
            'total_vectors': total_vectors,
            'total_memory_mb': total_memory,
            'mlx_optimized': True,
            'unified_memory': True
        }

# Global store manager instance
store_manager = VectorStoreManager()

# Pydantic models
class VectorAddResponse(BaseModel):
    success: bool
    vectors_added: int
    total_vectors: int
    processing_time_ms: float

class VectorQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_time_ms: float
    total_vectors_searched: int

class BatchQueryResponse(BaseModel):
    results: List[List[Dict[str, Any]]]
    total_queries: int
    avg_query_time_ms: float


@router.post("/add", response_model=VectorAddResponse)
async def add_vectors(
    request: VectorAddRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Add vectors to the store with MLX optimization"""
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
        
        # Add vectors in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            store_manager._executor,
            lambda: store.add_vectors(vectors_np, request.metadata)
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return VectorAddResponse(
            success=True,
            vectors_added=len(request.vectors),
            total_vectors=store.get_stats().get('vector_count', 0),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error adding vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add vectors: {str(e)}")


@router.post("/query", response_model=VectorQueryResponse)  
async def query_vectors(
    request: VectorQuery,
    api_key: str = Depends(verify_api_key)
):
    """High-performance vector similarity search using MLX"""
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
        
        # Format results
        formatted_results = []
        if isinstance(results, tuple) and len(results) == 3:
            indices, distances, metadata_list = results
            for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata_list)):
                similarity_score = max(0, 1.0 - dist) if hasattr(store.config, 'metric') and store.config.metric == "cosine" else -dist
                formatted_results.append({
                    "metadata": meta,
                    "similarity_score": float(similarity_score),
                    "rank": i + 1
                })
        else:
            # Legacy format compatibility
            for i, (metadata, score) in enumerate(results):
                formatted_results.append({
                    "metadata": metadata,
                    "similarity_score": float(score),
                    "rank": i + 1
                })
        
        query_time = (time.time() - start_time) * 1000
        
        return VectorQueryResponse(
            results=formatted_results,
            query_time_ms=query_time,
            total_vectors_searched=store.get_stats().get('vector_count', 0)
        )
        
    except Exception as e:
        logger.error(f"Error querying vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/batch_query", response_model=BatchQueryResponse)
async def batch_query_vectors(
    request: BatchQueryRequest,
    api_key: str = Depends(verify_api_key)
):
    """Optimized batch query processing for maximum throughput"""
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
        for query_results in batch_results:
            formatted_query_results = []
            # Handle different return formats
            if isinstance(query_results, tuple) and len(query_results) == 3:
                indices, distances, metadata_list = query_results
                for i, (idx, dist, meta) in enumerate(zip(indices, distances, metadata_list)):
                    similarity_score = max(0, 1.0 - dist) if hasattr(store.config, 'metric') and store.config.metric == "cosine" else -dist
                    formatted_query_results.append({
                        "metadata": meta,
                        "similarity_score": float(similarity_score),
                        "rank": i + 1
                    })
            else:
                # Legacy format
                for i, (metadata, score) in enumerate(query_results):
                    formatted_query_results.append({
                        "metadata": metadata,
                        "similarity_score": float(score),
                        "rank": i + 1
                    })
            formatted_batch_results.append(formatted_query_results)
        
        total_processing_time = (time.time() - start_time) * 1000
        avg_query_time = total_processing_time / len(request.queries)
        
        return BatchQueryResponse(
            results=formatted_batch_results,
            total_queries=len(request.queries),
            avg_query_time_ms=avg_query_time
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
        stats = store.get_stats()
        return {"count": stats.get('vector_count', 0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_store_stats(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed store statistics"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        stats = store.get_stats()
        
        return {
            "store_stats": stats,
            "performance_info": {
                "mlx_optimized": True,
                "expected_qps": "800-1500",
                "target_latency": "<10ms"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# Background tasks
async def optimize_store_background(store: MLXVectorStore):
    """Background task to optimize store performance"""
    try:
        await asyncio.sleep(1)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, store.optimize)
        logger.info("✅ Store optimization completed")
    except Exception as e:
        logger.error(f"Store optimization failed: {e}")


# Startup warmup
@router.on_event("startup")
async def startup_warmup():
    """Warm up MLX kernels on application startup"""
    logger.info("🚀 Starting MLX Vector API...")
    await store_manager.warmup_all_stores()
    logger.info("✅ MLX Vector API ready")


def benchmark_vector_store(store: MLXVectorStore, num_vectors: int = 1000, num_queries: int = 100) -> Dict[str, Any]:
    """Simple benchmark function"""
    try:
        # Generate test data
        test_vectors = np.random.rand(num_vectors, store.config.dimension).astype(np.float32)
        test_metadata = [{"id": f"bench_{i}"} for i in range(num_vectors)]
        
        # Benchmark add
        start_time = time.time()
        store.add_vectors(test_vectors, test_metadata)
        add_time = time.time() - start_time
        
        # Benchmark queries
        query_times = []
        for i in range(min(num_queries, num_vectors)):
            start_time = time.time()
            store.query(test_vectors[i], k=10)
            query_times.append(time.time() - start_time)
        
        avg_query_time = sum(query_times) / len(query_times) if query_times else 0
        
        return {
            "add_time_seconds": add_time,
            "avg_query_time_seconds": avg_query_time,
            "vectors_per_second": num_vectors / add_time if add_time > 0 else 0,
            "queries_per_second": 1 / avg_query_time if avg_query_time > 0 else 0
        }
        
    except Exception as e:
        return {"error": str(e)}


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