"""
Updated API routes to integrate HNSW functionality
Shows how to modify existing routes to leverage the new index
"""

# admin_routes.py - Updated admin routes with HNSW management
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import mlx.core as mx

from security.auth import (
    verify_admin_api_key, 
    validate_safe_identifier,
    validate_file_upload,
    get_client_identifier
)
from service.vector_store import VectorStore, VectorStoreConfig
from performance.hnsw_index import HNSWConfig

router = APIRouter(prefix="/admin", tags=["admin"])

class StoreConfig(BaseModel):
    user_id: str
    model_id: str
    enable_hnsw: bool = True
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    auto_index_threshold: int = 1000
    cache_enabled: bool = True

class OptimizeRequest(BaseModel):
    user_id: str
    model_id: str
    rebuild_index: bool = False
    optimize_memory: bool = True

def get_vector_store(user_id: str, model_id: str) -> VectorStore:
    """Helper to get vector store instance"""
    store_path = Path(f"~/.team_mind_data/vector_stores/{user_id}/{model_id}").expanduser()
    
    # Load with HNSW enabled by default
    config = VectorStoreConfig(enable_hnsw=True)
    return VectorStore(store_path, config)

@router.post("/create_store_advanced")
async def create_store_advanced(config: StoreConfig, api_key: str = Depends(get_api_key)):
    """Create a new vector store with advanced configuration"""
    store_path = Path(f"~/.team_mind_data/vector_stores/{config.user_id}/{config.model_id}").expanduser()
    
    if store_path.exists():
        raise HTTPException(status_code=400, detail="Store already exists")
    
    # Create custom configuration
    hnsw_config = HNSWConfig(
        M=config.hnsw_m,
        ef_construction=config.hnsw_ef_construction,
        ef_search=config.hnsw_ef_search
    )
    
    store_config = VectorStoreConfig(
        enable_hnsw=config.enable_hnsw,
        hnsw_config=hnsw_config,
        auto_index_threshold=config.auto_index_threshold,
        cache_enabled=config.cache_enabled
    )
    
    # Create store
    store = VectorStore(store_path, store_config)
    
    return {
        "status": "created",
        "path": str(store_path),
        "config": {
            "hnsw_enabled": config.enable_hnsw,
            "auto_index_threshold": config.auto_index_threshold
        }
    }

@router.post("/optimize_store")
async def optimize_store(request: OptimizeRequest, api_key: str = Depends(get_api_key)):
    """Optimize vector store for production use"""
    try:
        store = get_vector_store(request.user_id, request.model_id)
        
        # Rebuild index if requested
        if request.rebuild_index:
            store.rebuild_index(force=True)
        
        # Optimize memory and performance
        if request.optimize_memory:
            store.optimize()
        
        # Get updated stats
        stats = store.get_stats()
        
        return {
            "status": "optimized",
            "stats": {
                "total_vectors": stats["total_vectors"],
                "has_hnsw_index": stats["has_hnsw_index"],
                "storage_size_mb": stats["storage_size_mb"],
                "index_stats": stats.get("hnsw_stats", {})
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/store_stats/{user_id}/{model_id}")
async def get_store_stats(user_id: str, model_id: str, api_key: str = Depends(get_api_key)):
    """Get detailed statistics about a vector store"""
    try:
        store = get_vector_store(user_id, model_id)
        stats = store.get_stats()
        
        return {
            "user_id": user_id,
            "model_id": model_id,
            "stats": stats
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Store not found")

@router.post("/rebuild_index")
async def rebuild_index(user_id: str, model_id: str, force: bool = False, 
                       api_key: str = Depends(get_api_key)):
    """Rebuild HNSW index for a store"""
    try:
        store = get_vector_store(user_id, model_id)
        store.rebuild_index(force=force)
        
        return {
            "status": "index_rebuilt",
            "stats": store.get_stats()["hnsw_stats"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# performance_routes.py - Updated performance routes with HNSW metrics
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import time
import mlx.core as mx
import numpy as np

router = APIRouter(prefix="/performance", tags=["performance"])

class BenchmarkRequest(BaseModel):
    user_id: str
    model_id: str
    n_queries: int = 1000
    k: int = 10
    compare_methods: bool = True

class QueryBenchmarkResult(BaseModel):
    method: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queries_per_second: float

@router.post("/benchmark_hnsw")
async def benchmark_hnsw(request: BenchmarkRequest, api_key: str = Depends(get_api_key)):
    """Benchmark HNSW performance vs brute force"""
    try:
        store = get_vector_store(request.user_id, request.model_id)
        
        # Check if store has vectors
        stats = store.get_stats()
        if stats["total_vectors"] == 0:
            raise HTTPException(status_code=400, detail="Store is empty")
        
        dim = stats["vector_dimension"]
        results = []
        
        # Generate random queries
        queries = mx.random.normal((request.n_queries, dim))
        
        # Benchmark HNSW if available
        if stats["has_hnsw_index"]:
            hnsw_times = []
            for i in range(request.n_queries):
                start = time.time()
                store.query(queries[i], k=request.k, use_hnsw=True)
                hnsw_times.append(time.time() - start)
            
            hnsw_times_ms = np.array(hnsw_times) * 1000
            results.append(QueryBenchmarkResult(
                method="hnsw",
                avg_latency_ms=float(np.mean(hnsw_times_ms)),
                p95_latency_ms=float(np.percentile(hnsw_times_ms, 95)),
                p99_latency_ms=float(np.percentile(hnsw_times_ms, 99)),
                queries_per_second=float(request.n_queries / sum(hnsw_times))
            ))
        
        # Benchmark brute force if requested
        if request.compare_methods:
            # Limit brute force queries for large datasets
            n_bf_queries = min(100, request.n_queries)
            bf_times = []
            
            for i in range(n_bf_queries):
                start = time.time()
                store.query(queries[i], k=request.k, use_hnsw=False)
                bf_times.append(time.time() - start)
            
            bf_times_ms = np.array(bf_times) * 1000
            results.append(QueryBenchmarkResult(
                method="brute_force",
                avg_latency_ms=float(np.mean(bf_times_ms)),
                p95_latency_ms=float(np.percentile(bf_times_ms, 95)),
                p99_latency_ms=float(np.percentile(bf_times_ms, 99)),
                queries_per_second=float(n_bf_queries / sum(bf_times))
            ))
        
        # Calculate speedup if both methods tested
        speedup = None
        if len(results) == 2:
            speedup = results[1].avg_latency_ms / results[0].avg_latency_ms
        
        return {
            "results": results,
            "speedup": speedup,
            "test_config": {
                "n_queries": request.n_queries,
                "k": request.k,
                "vector_dimension": dim,
                "total_vectors": stats["total_vectors"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/index_health/{user_id}/{model_id}")
async def get_index_health(user_id: str, model_id: str, api_key: str = Depends(get_api_key)):
    """Get HNSW index health and recommendations"""
    try:
        store = get_vector_store(user_id, model_id)
        stats = store.get_stats()
        
        recommendations = []
        health_score = 100
        
        # Check if index exists
        if not stats["has_hnsw_index"] and stats["total_vectors"] > 1000:
            recommendations.append("Consider building HNSW index for better performance")
            health_score -= 20
        
        # Check query performance
        if stats["metrics"]["avg_query_time"] > 0.01:  # 10ms
            recommendations.append("Query performance is slow, consider rebuilding index")
            health_score -= 15
        
        # Check cache efficiency
        cache_hit_rate = 0
        if stats["metrics"]["total_queries"] > 0:
            cache_hit_rate = stats["metrics"]["cache_hits"] / stats["metrics"]["total_queries"]
        
        if cache_hit_rate < 0.3 and stats["metrics"]["total_queries"] > 100:
            recommendations.append("Low cache hit rate, consider increasing cache size")
            health_score -= 10
        
        # Check index stats if available
        if "hnsw_stats" in stats and stats["hnsw_stats"]:
            hnsw = stats["hnsw_stats"]
            total_edges = hnsw.get("total_edges", 0)
            total_nodes = hnsw.get("total_nodes", 1)
            avg_degree = total_edges / total_nodes if total_nodes > 0 else 0
            
            if avg_degree < 10:
                recommendations.append("Index connectivity is low, consider increasing M parameter")
                health_score -= 10
        
        return {
            "health_score": max(0, health_score),
            "status": "healthy" if health_score >= 80 else "needs_attention" if health_score >= 60 else "unhealthy",
            "recommendations": recommendations,
            "metrics": {
                "has_index": stats["has_hnsw_index"],
                "avg_query_time_ms": stats["metrics"]["avg_query_time"] * 1000,
                "cache_hit_rate": cache_hit_rate,
                "total_queries": stats["metrics"]["total_queries"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# vectors_routes.py - Updated vector routes with HNSW query options
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import mlx.core as mx

router = APIRouter(prefix="/vectors", tags=["vectors"])

class QueryRequest(BaseModel):
    user_id: str
    model_id: str
    query: List[float]
    k: int = 10
    use_hnsw: Optional[bool] = None  # None = auto, True = force HNSW, False = force brute force
    ef_search: Optional[int] = None  # Override ef_search for this query

class BatchQueryRequest(BaseModel):
    user_id: str
    model_id: str
    queries: List[List[float]]
    k: int = 10
    use_hnsw: Optional[bool] = None

@router.post("/query_advanced")
async def query_advanced(request: QueryRequest, api_key: str = Depends(get_api_key)):
    """Advanced query with HNSW options"""
    try:
        store = get_vector_store(request.user_id, request.model_id)
        
        # Convert query to MLX array
        query_vector = mx.array(request.query)
        
        # Override ef_search if specified
        if request.ef_search and hasattr(store, 'hnsw_index') and store.hnsw_index:
            original_ef = store.hnsw_index.config.ef_search
            store.hnsw_index.config.ef_search = request.ef_search
        
        # Perform query
        start_time = time.time()
        indices, distances, metadata = store.query(
            query_vector, 
            k=request.k, 
            use_hnsw=request.use_hnsw
        )
        query_time = time.time() - start_time
        
        # Restore original ef_search
        if request.ef_search and hasattr(store, 'hnsw_index') and store.hnsw_index:
            store.hnsw_index.config.ef_search = original_ef
        
        # Determine which method was used
        stats = store.get_stats()
        method_used = "hnsw" if stats["has_hnsw_index"] and request.use_hnsw != False else "brute_force"
        
        return {
            "indices": indices,
            "distances": distances,
            "metadata": metadata,
            "query_stats": {
                "method": method_used,
                "query_time_ms": query_time * 1000,
                "k": request.k,
                "ef_search": request.ef_search
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_query_optimized")
async def batch_query_optimized(request: BatchQueryRequest, api_key: str = Depends(get_api_key)):
    """Optimized batch query using HNSW"""
    try:
        store = get_vector_store(request.user_id, request.model_id)
        
        # Convert queries to MLX array
        query_vectors = mx.array(request.queries)
        
        # Perform batch query
        start_time = time.time()
        
        # Use batch search if HNSW is available
        if store.hnsw_index and request.use_hnsw != False:
            indices, distances = store.hnsw_index.batch_search(query_vectors, request.k)
            # Get metadata
            all_metadata = []
            for idx_list in indices.tolist():
                metadata = [store.metadata[i] for i in idx_list]
                all_metadata.append(metadata)
        else:
            # Fall back to sequential queries
            all_indices, all_distances, all_metadata = store.batch_query(query_vectors, request.k)
        
        batch_time = time.time() - start_time
        
        return {
            "results": [
                {
                    "indices": idx,
                    "distances": dist,
                    "metadata": meta
                }
                for idx, dist, meta in zip(
                    indices.tolist() if store.hnsw_index else all_indices,
                    distances.tolist() if store.hnsw_index else all_distances,
                    all_metadata
                )
            ],
            "batch_stats": {
                "total_queries": len(request.queries),
                "total_time_ms": batch_time * 1000,
                "avg_time_per_query_ms": (batch_time * 1000) / len(request.queries)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example of how to update main.py to include new routes
"""
# In main.py, add these routes:

from admin_routes import router as admin_router
from performance_routes import router as performance_router
from vectors_routes import router as vectors_router

app.include_router(admin_router)
app.include_router(performance_router)
app.include_router(vectors_router)

# Add startup event to log HNSW status
@app.on_event("startup")
async def startup_event():
    logger.info("MLX Vector DB starting with HNSW support...")
    logger.info(f"MLX version: {mx.__version__}")
    logger.info(f"Metal available: {mx.metal.is_available()}")
"""
