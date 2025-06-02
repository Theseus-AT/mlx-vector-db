# api/routes/admin.py - Secured Version
"""
Secured Admin routes for MLX Vector Database
"""
from fastapi import UploadFile, File, APIRouter, HTTPException, Query, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Dict, Optional
import numpy as np
import logging
import os
import tempfile
import zipfile
import shutil
import json

# Security imports
from security.auth import (
    verify_admin_api_key, 
    validate_safe_identifier,
    validate_file_upload,
    get_client_identifier
)

# Service Funktionen importieren
from service.vector_store import (
    list_users, list_models, count_vectors, delete_store,
    create_store, add_vectors, store_exists, get_store_path
)
import mlx.core as mx

# Logging Setup
logger = logging.getLogger("mlx_vector_db.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

# --- Secured Pydantic Models with Validation (Pydantic V2) ---
from pydantic import field_validator, model_validator

class StoreRequest(BaseModel):
    user_id: str
    model_id: str
    
    @field_validator('user_id', 'model_id')
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        return validate_safe_identifier(v)

class AdminAddVectorsRequest(BaseModel):
    user_id: str
    model_id: str
    vectors: List[List[float]]
    metadata: List[Dict]
    
    @field_validator('user_id', 'model_id')
    @classmethod
    def validate_identifiers(cls, v: str) -> str:
        return validate_safe_identifier(v)
    
    @field_validator('vectors')
    @classmethod
    def validate_vectors(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) > 10000:  # Limit batch size
            raise ValueError("Too many vectors in single request (max 10,000)")
        return v
    
    @model_validator(mode='after')
    def validate_metadata_count(self) -> 'AdminAddVectorsRequest':
        if len(self.metadata) != len(self.vectors):
            raise ValueError("Metadata count must match vector count")
        return self

# --- Secured Endpunkte ---

@router.get("/stats")
async def get_vector_store_stats(
    request: Request,
    api_key: str = Depends(verify_admin_api_key)
):
    """Aggregierte Übersicht aller User/Model-Stores (Admin only)."""
    client_id = get_client_identifier(request)
    logger.info(f"Stats request from {client_id}")
    
    results = []
    try:
        for user_id in list_users():
            user_data = {"user_id": user_id, "models": []}
            for model_id in list_models(user_id):
                counts = count_vectors(user_id, model_id)
                user_data["models"].append({
                    "model_id": model_id,
                    "vectors": counts.get("vectors", -1),
                    "metadata": counts.get("metadata", -1)
                })
            if user_data["models"]:
                results.append(user_data)
        return results
    except Exception as e:
        logger.exception("Error retrieving general stats.")
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving stats: {e}")

@router.get("/store/stats")
async def get_specific_store_stats(
    request: Request,
    user_id: str = Query(...),
    model_id: str = Query(...),
    api_key: str = Depends(verify_admin_api_key)
):
    """Gibt die Stats für einen spezifischen Store zurück (Admin only)."""
    # Validate identifiers
    user_id = validate_safe_identifier(user_id, "user_id")
    model_id = validate_safe_identifier(model_id, "model_id")
    
    client_id = get_client_identifier(request)
    logger.info(f"Store stats request for {user_id}/{model_id} from {client_id}")
    
    if not store_exists(user_id, model_id):
        raise HTTPException(status_code=404, detail="Store not found.")
    
    try:
        counts = count_vectors(user_id, model_id)
        return {
            "user_id": user_id,
            "model_id": model_id,
            "vectors": counts.get("vectors", -1),
            "metadata": counts.get("metadata", -1)
        }
    except Exception as e:
        logger.exception(f"Error getting stats for {user_id}/{model_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error getting stats: {e}")

@router.delete("/store")
async def delete_vector_store_endpoint(
    req: StoreRequest,
    request: Request,
    api_key: str = Depends(verify_admin_api_key)
):
    """Löscht den gesamten Vektorstore (Admin only)."""
    client_id = get_client_identifier(request)
    logger.warning(f"Store deletion request for {req.user_id}/{req.model_id} from {client_id}")
    
    if not store_exists(req.user_id, req.model_id):
        raise HTTPException(status_code=404, detail="Store not found.")
    
    try:
        delete_store(req.user_id, req.model_id)
        logger.warning(f"Store {req.user_id}/{req.model_id} deleted by {client_id}")
        return {"status": "deleted", "user_id": req.user_id, "model_id": req.model_id}
    except Exception as e:
        logger.exception(f"Failed to delete store via API for {req.user_id}/{req.model_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error during delete: {str(e)}")

@router.post("/create_store")
async def create_vector_store_endpoint(
    req: StoreRequest,
    request: Request,
    api_key: str = Depends(verify_admin_api_key)
):
    """Erstellt einen neuen Store (Admin only)."""
    client_id = get_client_identifier(request)
    logger.info(f"Store creation request for {req.user_id}/{req.model_id} from {client_id}")
    
    if store_exists(req.user_id, req.model_id):
        raise HTTPException(status_code=409, detail="Store already exists.")
    
    try:
        create_store(req.user_id, req.model_id)
        logger.info(f"Store {req.user_id}/{req.model_id} created by {client_id}")
        return {"status": "created", "user_id": req.user_id, "model_id": req.model_id}
    except Exception as e:
        logger.exception(f"Failed to create store via API for {req.user_id}/{req.model_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error during create: {str(e)}")

@router.post("/add_test_vectors")
async def add_test_vectors(
    req: AdminAddVectorsRequest,
    request: Request,
    api_key: str = Depends(verify_admin_api_key)
):
    """Fügt Test-Vektoren hinzu (Admin only)."""
    client_id = get_client_identifier(request)
    logger.info(f"Adding {len(req.vectors)} test vectors to {req.user_id}/{req.model_id} from {client_id}")
    
    if not store_exists(req.user_id, req.model_id):
        raise HTTPException(status_code=404, detail="Store not found. Create it first.")
    
    try:
        vectors_np = np.array(req.vectors, dtype=np.float32)
        add_vectors(req.user_id, req.model_id, vectors_np, req.metadata)
        logger.info(f"Successfully added {len(req.vectors)} test vectors to {req.user_id}/{req.model_id}")
        return {"status": "vectors added", "user_id": req.user_id, "model_id": req.model_id, "count": len(req.vectors)}
    except Exception as e:
        logger.exception(f"Error adding test vectors for {req.user_id}/{req.model_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error adding vectors: {str(e)}")

@router.get("/export_zip")
async def export_vector_store_zip(
    background_tasks: BackgroundTasks,
    request: Request,
    user_id: str = Query(...),
    model_id: str = Query(...),
    api_key: str = Depends(verify_admin_api_key)
):
    """Exportiert den Store als ZIP-Archiv (Admin only)."""
    # Validate identifiers
    user_id = validate_safe_identifier(user_id, "user_id")
    model_id = validate_safe_identifier(model_id, "model_id")
    
    client_id = get_client_identifier(request)
    logger.info(f"Export request for {user_id}/{model_id} from {client_id}")
    
    path = get_store_path(user_id, model_id)
    vector_file = path / "vectors.npz"
    metadata_file = path / "metadata.jsonl"
    
    if not vector_file.is_file():
        raise HTTPException(status_code=404, detail="Vector file not found")
    if not metadata_file.is_file():
        raise HTTPException(status_code=404, detail="Metadata file not found")

    tmp_zip_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip_file:
            tmp_zip_path = tmp_zip_file.name
            with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(vector_file, arcname="vectors.npz")
                zipf.write(metadata_file, arcname="metadata.jsonl")
            
        background_tasks.add_task(os.remove, tmp_zip_path)
        
        logger.info(f"Export completed for {user_id}/{model_id} by {client_id}")
        return FileResponse(
            path=tmp_zip_path,
            filename=f"store_export_{user_id}_{model_id}.zip",
            media_type="application/zip"
        )
    except Exception as e:
        logger.exception(f"Failed to create ZIP export for {user_id}/{model_id}")
        if tmp_zip_path and os.path.exists(tmp_zip_path):
            try: os.remove(tmp_zip_path)
            except OSError: pass
        raise HTTPException(status_code=500, detail=f"Error creating ZIP export: {str(e)}")

@router.post("/import_zip")
async def import_vector_store_from_zip(
    request: Request,
    user_id: str = Query(...),
    model_id: str = Query(...),
    overwrite: bool = Query(False),
    file: UploadFile = File(...),
    api_key: str = Depends(verify_admin_api_key)
):
    """Importiert einen Store aus einer ZIP-Datei (Admin only)."""
    # Validate identifiers
    user_id = validate_safe_identifier(user_id, "user_id")
    model_id = validate_safe_identifier(model_id, "model_id")
    
    client_id = get_client_identifier(request)
    logger.info(f"Import request for {user_id}/{model_id} from {client_id}")
    
    # Validate file upload
    file_size = 0
    if hasattr(file.file, 'seek'):
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
    
    validate_file_upload(file_size, file.content_type or "")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Save uploaded file
            zip_path = os.path.join(tmpdir, file.filename or "import.zip")
            content = await file.read()
            
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded ZIP file is empty.")
            
            with open(zip_path, "wb") as f:
                f.write(content)
            
            # Extract and validate ZIP
            try:
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    expected_files = {"vectors.npz", "metadata.jsonl"}
                    found_files = set(zipf.namelist())
                    
                    if not expected_files.issubset(found_files):
                        missing = expected_files - found_files
                        raise HTTPException(status_code=400, detail=f"ZIP missing files: {missing}")
                    
                    zipf.extractall(tmpdir)
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Invalid ZIP file format.")
            
            # Handle existing store
            target_store_exists = store_exists(user_id, model_id)
            if overwrite and target_store_exists:
                logger.warning(f"Overwriting existing store: {user_id}/{model_id}")
                delete_store(user_id, model_id)
                target_store_exists = False
            elif not overwrite and target_store_exists:
                raise HTTPException(status_code=409, detail="Store already exists. Set overwrite=true to replace.")
            
            if not target_store_exists:
                create_store(user_id, model_id)
            
            # Load and validate data
            vec_path = os.path.join(tmpdir, "vectors.npz")
            meta_path = os.path.join(tmpdir, "metadata.jsonl")
            
            loaded = mx.load(vec_path)
            if "vectors" not in loaded:
                raise ValueError("vectors.npz missing 'vectors' key")
            
            vectors = loaded["vectors"]
            
            metadata = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        metadata.append(json.loads(line))
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON in metadata.jsonl at line {i+1}")
            
            # Consistency check
            if vectors.shape[0] != len(metadata):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Data inconsistency: {vectors.shape[0]} vectors but {len(metadata)} metadata entries"
                )
            
            # Import the data
            add_vectors(user_id, model_id, vectors, metadata)
            
            logger.info(f"Successfully imported {len(metadata)} vectors to {user_id}/{model_id} by {client_id}")
            return {"status": "imported", "user_id": user_id, "model_id": model_id, "count": len(metadata)}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error during ZIP import for {user_id}/{model_id}")
            raise HTTPException(status_code=500, detail=f"Failed to import ZIP: {str(e)}")

# Health check endpoint (no auth required)
@router.get("/health")
async def admin_health():
    """Admin service health check."""
    return {"status": "healthy", "service": "admin"}