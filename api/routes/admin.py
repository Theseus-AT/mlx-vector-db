from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, Optional
import logging

from service.optimized_vector_store import MLXVectorStoreConfig
from service.models import CreateStoreRequest
from security.auth import verify_admin_key
from api.routes.vectors import store_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/create_store")
async def create_store(
    request: CreateStoreRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_admin_key) 
):
    """
    Create a new vector store with specified configuration
    """
    try:
        # Konfiguration aus dem Request oder Standard verwenden
        config = MLXVectorStoreConfig(
            dimension=request.dimension, 
            metric=request.metric.value if hasattr(request.metric, 'value') else request.metric
        )
        
        store_key = store_manager.get_store_key(request.user_id, request.model_id)
        if store_key in store_manager._stores:
            raise HTTPException(
                status_code=409, 
                detail=f"Store already exists for {request.user_id}/{request.model_id}"
            )
        
        store = await store_manager.get_store(
            request.user_id, 
            request.model_id, 
            config
        )
        
        logger.info(f"✅ Created store: {request.user_id}/{request.model_id}")
        
        return {
            "status": "success",
            "data": {
                "store_created": True,
                "user_id": request.user_id,
                "model_id": request.model_id,
                "config": {
                    "dimension": config.dimension,
                    "metric": config.metric
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/store")
async def delete_store(
    user_id: str,
    model_id: str,
    force: bool = False,
    api_key: str = Depends(verify_admin_key)
):
    """Delete a vector store"""
    try:
        store_key = store_manager.get_store_key(user_id, model_id)
        if store_key not in store_manager._stores:
            raise HTTPException(status_code=404, detail="Store not found")
        
        store = store_manager._stores[store_key]
        vector_count = getattr(store, '_vector_count', 0)
        
        # Safety check
        if vector_count > 0 and not force:
            raise HTTPException(
                status_code=400, 
                detail=f"Store contains {vector_count} vectors. Use force=True to delete."
            )
        
        # Clear store data
        store.clear()
        
        # Remove from manager
        del store_manager._stores[store_key]
        
        logger.info(f"🗑️ Deleted store: {user_id}/{model_id}")
        
        return {
            "status": "success", 
            "message": f"Store {user_id}/{model_id} deleted.",
            "vectors_removed": vector_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete store: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/store/stats")
async def get_store_stats(
    user_id: str,
    model_id: str,
    api_key: str = Depends(verify_admin_key)
):
    """Get statistics for a specific store"""
    try:
        store = await store_manager.get_store(user_id, model_id)
        stats = store.get_stats()
        
        return {
            "user_id": user_id,
            "model_id": model_id,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get store stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list_stores")
async def list_stores(
    user_id: Optional[str] = None,
    api_key: str = Depends(verify_admin_key)
):
    """List all stores, optionally filtered by user"""
    try:
        stores_info = []
        
        for store_key, store in store_manager._stores.items():
            # Parse store key to get user_id and model_id
            if "_" in store_key:
                key_user_id, key_model_id = store_key.split("_", 1)
                
                # Filter by user_id if specified
                if user_id and key_user_id != user_id:
                    continue
                
                stats = store.get_stats()
                stores_info.append({
                    "user_id": key_user_id,
                    "model_id": key_model_id,
                    "vector_count": stats.get("vector_count", 0),
                    "dimension": stats.get("dimension", 0),
                    "metric": stats.get("metric", "unknown")
                })
        
        return {
            "stores": stores_info,
            "total_count": len(stores_info)
        }
        
    except Exception as e:
        logger.error(f"Failed to list stores: {e}")
        raise HTTPException(status_code=500, detail=str(e))