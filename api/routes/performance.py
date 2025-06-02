# api/routes/performance.py
"""
Performance monitoring and optimization endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import time
import numpy as np
import mlx.core as mx
import logging

logger = logging.getLogger("mlx_vector_db.performance")

from security.auth import verify_api_key, get_client_identifier
from performance.optimized_vector_store import PerformantVectorStore
from performance.vector_cache import get_global_cache
from performance.mlx_optimized import performance_monitor, warmup_compiled_functions
from service.vector_store import get_store_path, store_exists

router = APIRouter(prefix="/performance", tags=["performance"])

class PerformanceTestRequest(BaseModel):
    user_id: str
    model_id: str
    test_size: int = 1000
    query_count: int = 100
    vector_dim: int = 384

@router.get("/stats")
async def get_performance_stats(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Get comprehensive performance statistics"""
    try:
        cache = get_global_cache()
        cache_stats = cache.get_detailed_stats()
        perf_stats = performance_monitor.get_stats()
        
        return {
            "cache": cache_stats,
            "compiled_functions": perf_stats,
            "system_info": {
                "mlx_available": True,
                "cache_enabled": True,
                "hnsw_enabled": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {e}")

@router.get("/cache/stats")
async def get_cache_stats(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed cache statistics"""
    cache = get_global_cache()
    return cache.get_detailed_stats()

@router.post("/cache/clear")
async def clear_cache(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Clear the global cache"""
    cache = get_global_cache()
    cache.clear()
    
    client_id = get_client_identifier(request)
    logger.info(f"Cache cleared by {client_id}")
    
    return {"status": "cache_cleared"}

@router.post("/warmup")
async def warmup_functions(
    request: Request,
    dimension: int = Query(384, ge=1, le=2048),
    api_key: str = Depends(verify_api_key)
):
    """Warm up compiled functions for better performance"""
    try:
        start_time = time.time()
        warmup_compiled_functions(dimension=dimension)
        warmup_time = time.time() - start_time
        
        return {
            "status": "warmup_completed",
            "dimension": dimension,
            "warmup_time_seconds": warmup_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}")

@router.post("/benchmark")
async def run_performance_benchmark(
    req: PerformanceTestRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Run performance benchmark comparing optimized vs basic operations"""
    try:
        client_id = get_client_identifier(request)
        logger.info(f"Performance benchmark requested by {client_id}")
        
        # Ensure store exists
        if not store_exists(req.user_id, req.model_id):
            raise HTTPException(status_code=404, detail="Store not found")
        
        # Create performant vector store
        base_path = get_store_path(req.user_id, req.model_id).parent.parent
        perf_store = PerformantVectorStore(req.user_id, req.model_id, base_path)
        
        # Generate test data
        test_vectors = mx.random.normal((req.test_size, req.vector_dim))
        test_metadata = [{"id": f"test_{i}", "benchmark": True} for i in range(req.test_size)]
        
        # Generate query vectors
        query_vectors = mx.random.normal((req.query_count, req.vector_dim))
        
        results = {}
        
        # Benchmark 1: Vector Addition
        start_time = time.time()
        perf_store.add_vectors_optimized(test_vectors, test_metadata)
        add_time_optimized = time.time() - start_time
        
        results["vector_addition"] = {
            "optimized_time": add_time_optimized,
            "vectors_added": req.test_size,
            "vectors_per_second": req.test_size / add_time_optimized
        }
        
        # Benchmark 2: Single Query Performance
        single_query = query_vectors[0]
        
        # Optimized query
        start_time = time.time()
        optimized_results = perf_store.query_vectors_optimized(single_query, k=10)
        single_query_time_optimized = time.time() - start_time
        
        # Basic query (fallback without HNSW)
        start_time = time.time()
        basic_results = perf_store.query_vectors_optimized(single_query, k=10, use_hnsw=False)
        single_query_time_basic = time.time() - start_time
        
        results["single_query"] = {
            "optimized_time": single_query_time_optimized,
            "basic_time": single_query_time_basic,
            "speedup_factor": single_query_time_basic / single_query_time_optimized if single_query_time_optimized > 0 else 0,
            "results_count": len(optimized_results),
            "queries_per_second_optimized": 1.0 / single_query_time_optimized if single_query_time_optimized > 0 else 0,
            "queries_per_second_basic": 1.0 / single_query_time_basic if single_query_time_basic > 0 else 0
        }
        
        # Benchmark 3: Batch Query Performance
        batch_size = min(50, req.query_count)
        batch_queries = query_vectors[:batch_size]
        
        start_time = time.time()
        batch_results = perf_store.batch_query_vectors_optimized(batch_queries, k=10)
        batch_time = time.time() - start_time
        
        results["batch_query"] = {
            "batch_size": batch_size,
            "total_time": batch_time,
            "time_per_query": batch_time / batch_size,
            "queries_per_second": batch_size / batch_time if batch_time > 0 else 0,
            "total_results": sum(len(r) for r in batch_results)
        }
        
        # Benchmark 4: Cache Performance
        cache = get_global_cache()
        cache_stats_before = cache.get_stats()
        
        # Run queries to test cache
        for i in range(min(10, req.query_count)):
            perf_store.query_vectors_optimized(query_vectors[i], k=5)
        
        cache_stats_after = cache.get_stats()
        
        results["cache_performance"] = {
            "hits_before": cache_stats_before["hits"],
            "hits_after": cache_stats_after["hits"],
            "cache_hits_gained": cache_stats_after["hits"] - cache_stats_before["hits"],
            "hit_rate_percent": cache_stats_after["hit_rate_percent"],
            "memory_usage_gb": cache_stats_after["memory_usage_gb"]
        }
        
        # Overall performance summary
        total_benchmark_time = time.time() - start_time
        
        results["summary"] = {
            "total_benchmark_time": total_benchmark_time,
            "test_vectors": req.test_size,
            "test_queries": req.query_count,
            "vector_dimension": req.vector_dim,
            "optimizations_active": {
                "hnsw_index": perf_store.use_hnsw,
                "vector_cache": perf_store.use_cache,
                "compiled_functions": True
            },
            "performance_improvement": {
                "query_speedup": results["single_query"]["speedup_factor"],
                "estimated_capacity": f"{results['single_query']['queries_per_second_optimized']:.1f} QPS"
            }
        }
        
        logger.info(f"Benchmark completed: {results['single_query']['speedup_factor']:.1f}x speedup achieved")
        
        return results
        
    except Exception as e:
        logger.exception("Benchmark failed")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {e}")

@router.post("/optimize")
async def optimize_store(
    request: Request,
    user_id: str = Query(...),
    model_id: str = Query(...),
    force_rebuild_index: bool = Query(False),
    api_key: str = Depends(verify_api_key)
):
    """Optimize a specific vector store (rebuild index, warm cache, etc.)"""
    try:
        if not store_exists(user_id, model_id):
            raise HTTPException(status_code=404, detail="Store not found")
        
        base_path = get_store_path(user_id, model_id).parent.parent
        perf_store = PerformantVectorStore(user_id, model_id, base_path)
        
        optimization_results = {}
        
        # Force index rebuild if requested
        if force_rebuild_index:
            perf_store.index_needs_rebuild = True
            perf_store.hnsw_index = None
        
        # Load data to trigger index build/cache population
        start_time = time.time()
        vectors, metadata = perf_store._load_vectors_and_metadata()
        load_time = time.time() - start_time
        
        optimization_results["data_loading"] = {
            "time_seconds": load_time,
            "vector_count": vectors.shape[0],
            "cached": perf_store.cache.get(user_id, model_id) is not None
        }
        
        # Build/rebuild index if needed
        if vectors.shape[0] > 100:
            start_time = time.time()
            index = perf_store._load_or_build_index(vectors)
            index_time = time.time() - start_time
            
            optimization_results["index_optimization"] = {
                "time_seconds": index_time,
                "index_built": index is not None,
                "index_nodes": index.node_count if index else 0,
                "index_stats": index.get_stats() if index else {}
            }
        
        # Get final performance stats
        optimization_results["final_stats"] = perf_store.get_performance_stats()
        
        client_id = get_client_identifier(request)
        logger.info(f"Store {user_id}/{model_id} optimized by {client_id}")
        
        return {
            "status": "optimization_completed",
            "user_id": user_id,
            "model_id": model_id,
            "optimization_results": optimization_results
        }
        
    except Exception as e:
        logger.exception(f"Optimization failed for {user_id}/{model_id}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

@router.get("/health")
async def performance_health():
    """Performance subsystem health check"""
    try:
        # Test basic MLX operations
        test_vector = mx.random.normal((100, 384))
        mx.eval(test_vector)
        
        # Test cache
        cache = get_global_cache()
        cache_stats = cache.get_stats()
        
        # Test compiled functions
        from performance.mlx_optimized import compute_cosine_similarity_single
        test_query = mx.random.normal((384,))
        test_db = mx.random.normal((10, 384))
        scores = compute_cosine_similarity_single(test_query, test_db)
        mx.eval(scores)
        
        return {
            "status": "healthy",
            "mlx_operations": "working",
            "cache_status": "active",
            "cache_entries": cache_stats["entries"],
            "compiled_functions": "working"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }