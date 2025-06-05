from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Optional, Any
import logging

from service.optimized_vector_store import MLXVectorStoreConfig
from service.models import CreateStoreRequest
from security.auth import verify_admin_key # KORRIGIERT: Admin-Verifizierung importiert
from api.routes.vectors import store_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/create_store")
async def create_store(
    request: CreateStoreRequest,
    background_tasks: BackgroundTasks,
    # KORRIGIERT: Benötigt Admin-Rechte, daher verify_admin_key
    api_key: str = Depends(verify_admin_key) 
):
    """
    Create a new vector store with specified configuration
    """
    try:
        # Konfiguration aus dem Request oder Standard verwenden
        config = MLXVectorStoreConfig(dimension=request.dimension, metric=request.metric)
        
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
                "config": config.__dict__
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create store: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ... (Rest der admin.py Datei bleibt unverändert) ...
# Fügen Sie hier die restlichen Routen aus Ihrer Originaldatei ein.
# Zum Beispiel die delete_store, list_stores etc.

@router.delete("/store")
async def delete_store(
    user_id: str,
    model_id: str,
    force: bool = False,
    api_key: str = Depends(verify_admin_key)
):
    # Ihre bestehende Implementierung für delete_store
    store_key = store_manager.get_store_key(user_id, model_id)
    if store_key not in store_manager._stores:
        raise HTTPException(status_code=404, detail="Store not found")
    
    # ... Rest der Logik ...
    del store_manager._stores[store_key]
    # ... Dateisystembereinigung ...

    return {"status": "success", "message": f"Store {user_id}/{model_id} deleted."}

# ... Weitere Admin-Routen ...