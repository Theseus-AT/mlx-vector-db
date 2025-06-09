#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

# api/routes/performance.py

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import time
import numpy as np
import logging

from service.models import BenchmarkRequest, BenchmarkResponse
# KORREKTUR 2: `verify_api_key` importieren
from security.auth import verify_api_key
from api.routes.vectors import store_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/performance", tags=["performance"])


class SimpleHealthCheck(BaseModel):
    status: str
    mlx_available: bool
    uptime_seconds: float


@router.get("/health", response_model=SimpleHealthCheck)
# KORREKTUR 2: Authentifizierung hinzugefügt
async def health_check(api_key: str = Depends(verify_api_key)):
    """Basic health check for performance monitoring, now with authentication."""
    try:
        import mlx.core as mx
        test_array = mx.random.normal((10, 10))
        mx.eval(test_array)
        mlx_healthy = True
    except Exception as e:
        logger.error(f"MLX health check failed: {e}")
        mlx_healthy = False
    
    # In einer echten App würde man hier eine globale Startzeit verwenden
    return SimpleHealthCheck(
        status="healthy" if mlx_healthy else "degraded",
        mlx_available=mlx_healthy,
        uptime_seconds=time.time() 
    )

# ... der Rest der Datei bleibt unverändert ...


class WarmupRequest(BaseModel):
    """Warmup request"""
    user_id: str
    model_id: str


def benchmark_vector_store(store, num_vectors: int = 1000, num_queries: int = 100) -> Dict[str, Any]:
    """Simple benchmark function for vector store performance"""
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


@router.get("/health", response_model=SimpleHealthCheck)
async def health_check():
    """Basic health check for performance monitoring"""
    try:
        import mlx.core as mx
        
        # Simple MLX test
        test_array = mx.random.normal((10, 10))
        mx.eval(test_array)
        mlx_healthy = True
        
    except Exception as e:
        logger.error(f"MLX health check failed: {e}")
        mlx_healthy = False
    
    return SimpleHealthCheck(
        status="healthy" if mlx_healthy else "degraded",
        mlx_available=mlx_healthy,
        uptime_seconds=time.time()  # Simplified uptime
    )


@router.post("/warmup")
async def warmup_kernels(
    request: WarmupRequest,
    api_key: str = Depends(verify_api_key)
):
    """Warm up MLX kernels for a specific store"""
    try:
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        # Warmup in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            store_manager._executor,
            store._warmup_kernels
        )
        
        return {"success": True, "message": "Kernels warmed up"}
        
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    request: BenchmarkRequest,
    api_key: str = Depends(verify_api_key)
):
    """Run performance benchmark"""
    try:
        store = await store_manager.get_store(request.user_id, request.model_id)
        
        # Run benchmark in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            store_manager._executor,
            lambda: benchmark_vector_store(
                store, 
                request.num_vectors, 
                request.num_queries
            )
        )
        
        return BenchmarkResponse(
            benchmark_results=results,
            mlx_optimized=True,
            performance_target="1000+ QPS"
        )
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize")
async def optimize_store(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Optimize store performance"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        
        # Optimize in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            store_manager._executor,
            store.optimize
        )
        
        return {"success": True, "message": "Store optimized"}
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_performance_stats(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Get basic performance statistics"""
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
        logger.error(f"Stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))