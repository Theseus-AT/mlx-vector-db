#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

#
"""
Monitoring API for MLX Vector Database
Essential monitoring and health check endpoints

Focus: Core monitoring without complex metrics
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import time
import logging
import psutil
import os

from security.auth import verify_api_key
from api.routes.vectors import store_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


class SystemHealth(BaseModel):
    """System health response"""
    status: str
    mlx_available: bool
    stores_active: int
    total_vectors: int
    memory_usage_mb: float
    uptime_seconds: float


class SystemMetrics(BaseModel):
    """Basic system metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    process_memory_mb: float


@router.get("/health", response_model=SystemHealth)
async def health_check():
    """Comprehensive health check"""
    try:
        # Test MLX
        import mlx.core as mx
        test_array = mx.random.normal((5, 5))
        mx.eval(test_array)
        mlx_healthy = True
    except Exception:
        mlx_healthy = False
    
    # Get store stats
    store_stats = store_manager.get_stats()
    
    return SystemHealth(
        status="healthy" if mlx_healthy else "degraded",
        mlx_available=mlx_healthy,
        stores_active=store_stats.get("total_stores", 0),
        total_vectors=store_stats.get("total_vectors", 0),
        memory_usage_mb=store_stats.get("total_memory_mb", 0.0),
        uptime_seconds=time.time()  # Simplified
    )


@router.get("/metrics", response_model=SystemMetrics)
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """Get system metrics"""
    try:
        # Get current process
        process = psutil.Process()
        
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent,
            process_memory_mb=process.memory_info().rss / (1024 * 1024)
        )
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_usage_percent=0.0,
            process_memory_mb=0.0
        )


@router.get("/status")
async def detailed_status(api_key: str = Depends(verify_api_key)):
    """Detailed system status"""
    try:
        import mlx.core as mx
        import platform
        
        # Test MLX device info
        try:
            device_info = str(mx.default_device())
            mlx_working = True
        except Exception:
            device_info = "unknown"
            mlx_working = False
        
        store_stats = store_manager.get_stats()
        
        return {
            "system": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "mlx": {
                "available": mlx_working,
                "device": device_info,
                "version": "0.25.2"
            },
            "stores": store_stats,
            "environment": os.getenv("ENVIRONMENT", "development")
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {"error": str(e)}


@router.get("/stores")
async def get_store_status(api_key: str = Depends(verify_api_key)):
    """Get status of all active stores"""
    try:
        stores_status = []
        
        for store_key, store in store_manager._stores.items():
            user_id, model_id = store_key.split("_", 1)
            stats = store.get_stats()
            
            stores_status.append({
                "user_id": user_id,
                "model_id": model_id,
                "vector_count": stats['vector_count'],
                "memory_mb": stats['memory_usage_mb'],
                "dimension": stats['dimension']
            })
        
        return {
            "stores": stores_status,
            "total_stores": len(stores_status)
        }
        
    except Exception as e:
        logger.error(f"Store status failed: {e}")
        return {"error": str(e)}