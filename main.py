"""
MLX Vector Database - FastAPI Application
Korrigierte, produktionsreife Version
"""

import os
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import mlx.core as mx

# Import our modules
from api.routes.vectors import router as vectors_router, store_manager
from api.routes.admin import router as admin_router
from api.routes.performance import router as performance_router
from api.routes.monitoring import router as monitoring_router
from security.auth import verify_api_key
from service.models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global performance metrics
performance_metrics = {
    "startup_time": 0.0,
    "total_requests": 0,
    "total_errors": 0,
    "avg_response_time": 0.0,
    "mlx_warmup_time": 0.0
}

def create_error_response(message: str, error_code: str = "ERROR") -> Dict[str, Any]:
    """Create a standardized error response"""
    return {
        "error": message,
        "code": error_code,
        "timestamp": time.time()
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events"""
    # Startup
    startup_start = time.time()
    logger.info("🚀 Starting MLX Vector Database...")
    
    try:
        # Initialize MLX
        logger.info("🔧 Initializing MLX framework...")
        mlx_start = time.time()
        
        # Test MLX availability
        test_array = mx.random.normal((10, 10))
        mx.eval(test_array)
        
        mlx_warmup_time = time.time() - mlx_start
        performance_metrics["mlx_warmup_time"] = mlx_warmup_time
        
        logger.info(f"✅ MLX initialized in {mlx_warmup_time:.2f}s")
        logger.info(f"📱 MLX Device: {mx.default_device()}")
        
        # Warm up vector stores
        logger.info("🔥 Warming up vector store kernels...")
        await store_manager.warmup_all_stores()
        
        # Set startup metrics
        total_startup_time = time.time() - startup_start
        performance_metrics["startup_time"] = total_startup_time
        
        logger.info(f"🎯 MLX Vector Database ready in {total_startup_time:.2f}s")
        logger.info("📊 Performance targets: 1000+ QPS, <10ms latency")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MLX Vector Database...")
    # Cleanup resources if needed
    logger.info("✅ Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="MLX Vector Database",
    description="High-performance vector database optimized for Apple Silicon with MLX",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request tracking middleware
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics for monitoring"""
    start_time = time.time()
    performance_metrics["total_requests"] += 1
    
    try:
        response = await call_next(request)
        
        # Update performance metrics
        response_time = time.time() - start_time
        current_avg = performance_metrics["avg_response_time"]
        total_requests = performance_metrics["total_requests"]
        
        # Rolling average calculation
        performance_metrics["avg_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-MLX-Optimized"] = "true"
        
        return response
        
    except Exception as e:
        performance_metrics["total_errors"] += 1
        logger.error(f"Request failed: {e}")
        
        # Return structured error response
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                message="Internal server error",
                error_code="INTERNAL_ERROR"
            )
        )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        )
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content=create_error_response(
            message=str(exc),
            error_code="VALIDATION_ERROR"
        )
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            message="An unexpected error occurred",
            error_code="UNEXPECTED_ERROR"
        )
    )

# Include API routers
app.include_router(vectors_router)
app.include_router(admin_router)
app.include_router(performance_router)
app.include_router(monitoring_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "MLX Vector Database",
        "version": "1.0.0",
        "mlx_version": "0.25.2",
        "status": "operational",
        "documentation": "/docs",
        "performance_target": "1000+ QPS",
        "optimization": "apple_silicon_native",
        "features": [
            "MLX 0.25.2 Native Operations",
            "Metal GPU Acceleration", 
            "Unified Memory",
            "JIT Compilation",
            "Zero-Copy Operations",
            "Async Request Handling"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    try:
        # Test MLX functionality
        test_array = mx.random.normal((5, 5))
        mx.eval(test_array)
        mlx_healthy = True
        mlx_device = str(mx.default_device())
    except Exception:
        mlx_healthy = False
        mlx_device = "unknown"
    
    # Get store manager stats
    store_stats = store_manager.get_stats()
    
    health_status = {
        "status": "healthy" if mlx_healthy else "degraded",
        "mlx_available": mlx_healthy,
        "mlx_device": mlx_device,
        "stores_active": store_stats.get("total_stores", 0),
        "total_vectors": store_stats.get("total_vectors", 0),
        "memory_usage_mb": store_stats.get("total_memory_mb", 0.0),
        "uptime_seconds": time.time() - performance_metrics.get("startup_time", time.time()),
        "performance_metrics": {
            "total_requests": performance_metrics["total_requests"],
            "total_errors": performance_metrics["total_errors"],
            "avg_response_time_ms": performance_metrics["avg_response_time"] * 1000,
            "error_rate": (
                performance_metrics["total_errors"] / max(performance_metrics["total_requests"], 1)
            )
        }
    }
    
    # Return appropriate status code
    status_code = 200 if mlx_healthy else 503
    return JSONResponse(content=health_status, status_code=status_code)

# System information endpoint
@app.get("/system/info")
async def system_info():
    """Detailed system information"""
    try:
        import platform
        import psutil
        
        return {
            "system": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "architecture": platform.architecture()[0]
            },
            "mlx": {
                "version": "0.25.2",
                "device": str(mx.default_device()),
                "unified_memory": True,
                "metal_available": True
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_percent": psutil.virtual_memory().percent
            },
            "performance": performance_metrics,
            "store_manager": store_manager.get_stats()
        }
    except Exception as e:
        logger.error(f"System info collection failed: {e}")
        return {"error": str(e)}

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current application configuration"""
    return {
        "mlx_optimization": True,
        "metal_acceleration": True,
        "jit_compilation": True,
        "unified_memory": True,
        "performance_targets": {
            "qps": "1000+",
            "latency_ms": "<10",
            "throughput": "high"
        },
        "features": {
            "async_processing": True,
            "batch_operations": True,
            "streaming_responses": True,
            "connection_pooling": True,
            "automatic_optimization": True
        }
    }

# Debug endpoints (only in development)
if os.getenv("ENVIRONMENT") == "development":
    
    @app.get("/debug/routes")
    async def debug_routes():
        """List all available routes"""
        routes = []
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                routes.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": getattr(route, 'name', 'unnamed')
                })
        return {"routes": routes}
    
    @app.get("/debug/mlx")
    async def debug_mlx():
        """Debug MLX functionality"""
        try:
            # Test various MLX operations
            test_data = mx.random.normal((100, 384))
            normalized = test_data / mx.linalg.norm(test_data, axis=1, keepdims=True)
            similarity = mx.matmul(normalized, normalized.T)
            mx.eval(similarity)
            
            return {
                "mlx_working": True,
                "test_shape": test_data.shape,
                "device": str(mx.default_device()),
                "operations_tested": ["random", "norm", "matmul", "eval"]
            }
        except Exception as e:
            return {
                "mlx_working": False,
                "error": str(e)
            }

# Production configuration
def create_production_app():
    """Create production-optimized FastAPI application"""
    # Production-specific configurations
    app.docs_url = None  # Disable docs in production
    app.redoc_url = None
    app.openapi_url = None
    
    return app

# Development server configuration
def run_development_server():
    """Run development server with hot reload"""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"],
        log_level="info",
        access_log=True
    )

# Production server configuration  
def run_production_server():
    """Run production server with optimized settings"""
    uvicorn.run(
        "main:create_production_app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for MLX (GPU sharing issues)
        log_level="warning",
        access_log=False,
        server_header=False,
        date_header=False
    )

def main():
    """Main entry point when running as script"""
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "production":
            logger.info("🚀 Starting production server...")
            run_production_server()
        elif sys.argv[1] == "development":
            logger.info("🛠️ Starting development server...")
            run_development_server()
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python main.py [production|development]")
            sys.exit(1)
    else:
        # Default based on environment
        env = os.getenv("ENVIRONMENT", "development")
        
        if env == "production":
            logger.info("🚀 Starting production server...")
            run_production_server()
        else:
            logger.info("🛠️ Starting development server...")
            run_development_server()

if __name__ == "__main__":
    main()