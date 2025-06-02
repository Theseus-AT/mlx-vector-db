# main_simple.py
"""
MLX Vector Database - Simplified Version (working guaranteed)
"""
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mlx_vector_db")

# Create FastAPI app
app = FastAPI(
    title="MLX Vector Database",
    version="1.0.0",
    description="High-performance vector database optimized for Apple Silicon",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register available routers
try:
    from api.routes.admin import router as admin_router
    app.include_router(admin_router)
    logger.info("✅ Admin routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ Admin routes not available: {e}")

try:
    from api.routes.vectors import router as vectors_router
    app.include_router(vectors_router)
    logger.info("✅ Vector routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ Vector routes not available: {e}")

try:
    from api.routes.performance import router as performance_router
    app.include_router(performance_router)
    logger.info("✅ Performance routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ Performance routes not available: {e}")

try:
    from api.routes.monitoring import router as monitoring_router
    app.include_router(monitoring_router)
    logger.info("✅ Monitoring routes loaded")
except ImportError as e:
    logger.warning(f"⚠️ Monitoring routes not available: {e}")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "MLX Vector Database",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "mlx-vector-db",
        "timestamp": time.time()
    }

@app.get("/debug/routes")
def debug_routes():
    """Debug endpoint to see available routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unknown')
            })
    return {"routes": routes}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting MLX Vector Database on {host}:{port}")
    logger.info("Access API documentation at: http://localhost:8000/docs")
    logger.info("Debug routes at: http://localhost:8000/debug/routes")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info",
        access_log=True
    )