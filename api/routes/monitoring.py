# api/routes/monitoring.py
"""
Monitoring, metrics, and health check endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Response, Query, Request
from fastapi.responses import PlainTextResponse
from typing import Dict, Any, Optional
import time
import logging

from security.auth import verify_api_key, get_client_identifier
from monitoring.metrics import (
    metrics_registry, 
    health_checker,
    record_request,
    record_error
)

logger = logging.getLogger("mlx_vector_db.monitoring")

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/health")
async def health_check():
    """
    Basic health check endpoint (no auth required)
    Returns 200 if service is healthy, 503 if unhealthy
    """
    try:
        health_result = health_checker.run_all_checks()
        
        if health_result["overall_status"] == "unhealthy":
            return Response(
                content=f"Service unhealthy: {health_result}",
                status_code=503,
                media_type="application/json"
            )
        
        return {
            "status": health_result["overall_status"],
            "timestamp": health_result["timestamp"],
            "service": "mlx-vector-db",
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response(
            content=f"Health check error: {e}",
            status_code=503,
            media_type="text/plain"
        )

@router.get("/health/detailed")
async def detailed_health_check(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Detailed health check with all component status (requires auth)
    """
    try:
        client_id = get_client_identifier(request)
        logger.info(f"Detailed health check requested by {client_id}")
        
        health_result = health_checker.run_all_checks()
        
        # Add additional system info
        import psutil
        import mlx.core as mx
        
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
            "mlx_version": getattr(mx, '__version__', 'unknown'),
            "python_version": f"{psutil.version_info}",
        }
        
        health_result["system_info"] = system_info
        
        return health_result
        
    except Exception as e:
        logger.exception("Detailed health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.get("/metrics")
async def get_metrics(
    request: Request,
    format: str = Query("json", regex="^(json|prometheus)$"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get application metrics in JSON or Prometheus format
    """
    try:
        client_id = get_client_identifier(request)
        logger.debug(f"Metrics requested by {client_id} in {format} format")
        
        if format == "prometheus":
            prometheus_text = metrics_registry.get_prometheus_format()
            return PlainTextResponse(
                content=prometheus_text,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        else:
            # JSON format
            metrics_data = metrics_registry.get_all_metrics()
            
            # Add metadata
            return {
                "metrics": metrics_data,
                "timestamp": time.time(),
                "format": "json",
                "service": "mlx-vector-db"
            }
            
    except Exception as e:
        logger.exception("Failed to get metrics")
        raise HTTPException(status_code=500, detail=f"Metrics error: {e}")

@router.get("/metrics/summary")
async def get_metrics_summary(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Get high-level metrics summary for dashboards
    """
    try:
        metrics_data = metrics_registry.get_all_metrics()
        
        # Extract key metrics for summary
        summary = {
            "requests": {
                "total": metrics_data.get("http_requests_total", {}).get("value", 0),
                "avg_duration": 0
            },
            "vector_operations": {
                "queries_total": metrics_data.get("vector_queries_total", {}).get("value", 0),
                "additions_total": metrics_data.get("vector_additions_total", {}).get("value", 0),
                "avg_query_duration": 0
            },
            "cache": {
                "hits": metrics_data.get("cache_hits_total", {}).get("value", 0),
                "misses": metrics_data.get("cache_misses_total", {}).get("value", 0),
                "memory_usage_mb": (metrics_data.get("cache_memory_usage_bytes", {}).get("value", 0) or 0) / (1024**2)
            },
            "system": {
                "cpu_usage_percent": metrics_data.get("system_cpu_usage_percent", {}).get("value", 0),
                "memory_usage_gb": (metrics_data.get("system_memory_usage_bytes", {}).get("value", 0) or 0) / (1024**3),
                "disk_usage_gb": (metrics_data.get("system_disk_usage_bytes", {}).get("value", 0) or 0) / (1024**3)
            },
            "errors": {
                "total": metrics_data.get("errors_total", {}).get("value", 0)
            }
        }
        
        # Calculate derived metrics
        cache_total = summary["cache"]["hits"] + summary["cache"]["misses"]
        if cache_total > 0:
            summary["cache"]["hit_rate_percent"] = (summary["cache"]["hits"] / cache_total) * 100
        else:
            summary["cache"]["hit_rate_percent"] = 0
        
        # Add histogram statistics if available
        if "http_request_duration_seconds" in metrics_data:
            duration_stats = metrics_data["http_request_duration_seconds"].get("stats", {})
            summary["requests"]["avg_duration"] = duration_stats.get("mean", 0)
        
        if "vector_query_duration_seconds" in metrics_data:
            query_stats = metrics_data["vector_query_duration_seconds"].get("stats", {})
            summary["vector_operations"]["avg_query_duration"] = query_stats.get("mean", 0)
        
        return {
            "summary": summary,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.exception("Failed to get metrics summary")
        raise HTTPException(status_code=500, detail=f"Metrics summary error: {e}")

@router.get("/status")
async def service_status(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Get comprehensive service status including performance indicators
    """
    try:
        # Get health status
        health_result = health_checker.run_all_checks()
        
        # Get metrics summary
        metrics_data = metrics_registry.get_all_metrics()
        
        # Performance indicators
        performance_indicators = {
            "response_time_ok": True,
            "cache_performance_ok": True,
            "error_rate_ok": True,
            "resource_usage_ok": True
        }
        
        # Check response time
        if "http_request_duration_seconds" in metrics_data:
            stats = metrics_data["http_request_duration_seconds"].get("stats", {})
            avg_duration = stats.get("mean", 0)
            if avg_duration > 1.0:  # More than 1 second average
                performance_indicators["response_time_ok"] = False
        
        # Check cache performance
        cache_hits = metrics_data.get("cache_hits_total", {}).get("value", 0) or 0
        cache_misses = metrics_data.get("cache_misses_total", {}).get("value", 0) or 0
        if cache_hits + cache_misses > 100:  # Only check if we have enough samples
            hit_rate = cache_hits / (cache_hits + cache_misses)
            if hit_rate < 0.3:  # Less than 30% hit rate
                performance_indicators["cache_performance_ok"] = False
        
        # Check error rate
        total_requests = metrics_data.get("http_requests_total", {}).get("value", 0) or 0
        total_errors = metrics_data.get("errors_total", {}).get("value", 0) or 0
        if total_requests > 100:  # Only check if we have enough samples
            error_rate = total_errors / total_requests
            if error_rate > 0.05:  # More than 5% error rate
                performance_indicators["error_rate_ok"] = False
        
        # Check resource usage
        cpu_usage = metrics_data.get("system_cpu_usage_percent", {}).get("value", 0) or 0
        memory_usage = metrics_data.get("system_memory_usage_bytes", {}).get("value", 0) or 0
        total_memory = 8 * 1024**3  # Assume 8GB, could be made dynamic
        memory_percent = (memory_usage / total_memory) * 100
        
        if cpu_usage > 90 or memory_percent > 90:
            performance_indicators["resource_usage_ok"] = False
        
        # Overall status
        overall_healthy = (
            health_result["overall_status"] in ["healthy", "warning"] and
            all(performance_indicators.values())
        )
        
        return {
            "service": "mlx-vector-db",
            "version": "1.0.0",
            "status": "healthy" if overall_healthy else "degraded",
            "uptime_seconds": time.time(),  # Simplified uptime
            "health": health_result,
            "performance_indicators": performance_indicators,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.exception("Failed to get service status")
        raise HTTPException(status_code=500, detail=f"Status error: {e}")

@router.post("/alerts/test")
async def test_alert(
    request: Request,
    alert_type: str = Query("test", regex="^(test|high_cpu|low_disk|error_spike)$"),
    api_key: str = Depends(verify_api_key)
):
    """
    Test alert system (for development/testing)
    """
    client_id = get_client_identifier(request)
    logger.warning(f"Test alert '{alert_type}' triggered by {client_id}")
    
    alerts = {
        "test": "This is a test alert",
        "high_cpu": "CPU usage above 90%",
        "low_disk": "Disk space below 10%",
        "error_spike": "Error rate above 5%"
    }
    
    # Record test error for demonstration
    record_error("test_alert", "monitoring")
    
    return {
        "alert_type": alert_type,
        "message": alerts[alert_type],
        "severity": "warning",
        "timestamp": time.time(),
        "triggered_by": client_id
    }

# Middleware integration for automatic metrics collection
async def metrics_middleware(request: Request, call_next):
    """Middleware to automatically collect request metrics"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record request metrics
        record_request(
            duration=duration,
            status_code=response.status_code,
            endpoint=request.url.path
        )
        
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error
        record_error(
            error_type=type(e).__name__,
            endpoint=request.url.path
        )
        
        # Still record request (with error status)
        record_request(
            duration=duration,
            status_code=500,
            endpoint=request.url.path
        )
        
        raise