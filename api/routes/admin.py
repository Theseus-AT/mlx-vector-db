# api/routes/admin.py

# Stelle sicher, dass BackgroundTasks importiert wird
from fastapi import UploadFile, File, APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import logging
import os
import tempfile
import zipfile
import shutil
import json

# Service Funktionen importieren
from service.vector_store import (
    list_users, list_models, count_vectors, delete_store,
    create_store, add_vectors, store_exists, get_store_path
)
# Importiere mlx nur dort, wo es wirklich gebraucht wird (z.B. für mx.load)
import mlx.core as mx

# Logging Setup
logger = logging.getLogger("mlx_vector_db") # Gleicher Logger-Name wie im Service
# Grundlegende Konfiguration, falls noch nicht geschehen (z.B. in main.py)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.DEBUG)
logger.setLevel(logging.DEBUG) # Stelle sicher, dass DEBUG-Logs angezeigt werden


router = APIRouter(prefix="/admin", tags=["admin"])

# --- Pydantic Models ---
class StoreRequest(BaseModel):
    user_id: str
    model_id: str

class AdminAddVectorsRequest(BaseModel):
    user_id: str
    model_id: str
    vectors: List[List[float]]
    metadata: List[Dict]

# --- Endpunkte ---

@router.get("/stats")
def get_vector_store_stats():
    """Aggregierte Übersicht aller User/Model-Stores."""
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
            # Nur User hinzufügen, wenn sie auch Modelle haben (optional)
            if user_data["models"]:
                results.append(user_data)
        return results
    except Exception as e:
        logger.exception("Error retrieving general stats.")
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving stats: {e}")

@router.get("/store/stats")
def get_specific_store_stats(user_id: str = Query(...), model_id: str = Query(...)):
    """Gibt die Stats für einen spezifischen Store zurück."""
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
def delete_vector_store_endpoint(req: StoreRequest):
    """Löscht den gesamten Vektorstore (erwartet JSON Body)."""
    if not store_exists(req.user_id, req.model_id):
        raise HTTPException(status_code=404, detail="Store not found.")
    try:
        logger.info(f"Attempting API delete for store: {req.user_id}/{req.model_id}")
        delete_store(req.user_id, req.model_id)
        logger.info(f"API delete successful for store: {req.user_id}/{req.model_id}")
        return {"status": "deleted", "user_id": req.user_id, "model_id": req.model_id}
    except Exception as e:
        logger.exception(f"Failed to delete store via API for {req.user_id}/{req.model_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error during delete: {str(e)}")


@router.post("/create_store")
def create_vector_store_endpoint(req: StoreRequest):
    """Erstellt einen neuen Store."""
    if store_exists(req.user_id, req.model_id):
        raise HTTPException(status_code=409, detail="Store already exists.")
    try:
        logger.info(f"Attempting API create for store: {req.user_id}/{req.model_id}")
        create_store(req.user_id, req.model_id)
        logger.info(f"API create successful for store: {req.user_id}/{req.model_id}")
        return {"status": "created", "user_id": req.user_id, "model_id": req.model_id}
    except Exception as e:
        logger.exception(f"Failed to create store via API for {req.user_id}/{req.model_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error during create: {str(e)}")


@router.post("/add_test_vectors")
def add_test_vectors(req: AdminAddVectorsRequest):
    """Fügt Test-Vektoren hinzu (konvertiert Liste zu NumPy)."""
    if not store_exists(req.user_id, req.model_id):
         logger.error(f"Store {req.user_id}/{req.model_id} not found in add_test_vectors.")
         raise HTTPException(status_code=404, detail="Store not found. Create it first.")
    try:
        logger.debug(f"Received {len(req.vectors)} vectors as list for {req.user_id}/{req.model_id}.")
        vectors_np = np.array(req.vectors, dtype=np.float32)
        logger.debug(f"Converted vectors to NumPy array with shape: {vectors_np.shape}")
        add_vectors(req.user_id, req.model_id, vectors_np, req.metadata)
        logger.info(f"Successfully added {len(req.vectors)} test vectors via API to {req.user_id}/{req.model_id}")
        return {"status": "vectors added", "user_id": req.user_id, "model_id": req.model_id, "count": len(req.vectors)}
    except TypeError as te:
        logger.error(f"TypeError in add_test_vectors API: {te}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid input data type: {te}")
    except ValueError as ve:
        logger.error(f"ValueError in add_test_vectors API: {ve}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid input data value: {ve}")
    except Exception as e:
        logger.exception(f"Unexpected error in add_test_vectors API for {req.user_id}/{req.model_id}")
        raise HTTPException(status_code=500, detail=f"Internal server error adding vectors: {str(e)}")


@router.get("/preview_metadata")
def preview_metadata(
    user_id: str = Query(...),
    model_id: str = Query(...),
    limit: int = Query(5, ge=1, le=100)
):
    """Gibt eine Vorschau der Metadaten zurück."""
    path = get_store_path(user_id, model_id)
    metadata_path = path / "metadata.jsonl"
    if not metadata_path.is_file():
        raise HTTPException(status_code=404, detail="Metadata file not found for this store.")
    preview_lines = []
    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit: break
                try: preview_lines.append(json.loads(line))
                except json.JSONDecodeError: logger.warning(f"Invalid JSON in {metadata_path} at line {i+1}"); preview_lines.append({"error": "invalid JSON", "line": i+1})
        return {"user_id": user_id, "model_id": model_id, "metadata_preview": preview_lines}
    except Exception as e:
        logger.exception(f"Error reading metadata file {metadata_path}")
        raise HTTPException(status_code=500, detail=f"Error reading metadata: {str(e)}")


@router.get("/export_zip")
def export_vector_store_zip(
    background_tasks: BackgroundTasks, # Abhängigkeit zuerst
    user_id: str = Query(...),
    model_id: str = Query(...)
):
    """Exportiert den Store als ZIP-Archiv."""
    path = get_store_path(user_id, model_id)
    vector_file = path / "vectors.npz"
    metadata_file = path / "metadata.jsonl"
    if not vector_file.is_file(): raise HTTPException(status_code=404, detail=f"Vector file not found: {vector_file.name}")
    if not metadata_file.is_file(): raise HTTPException(status_code=404, detail=f"Metadata file not found: {metadata_file.name}")

    tmp_zip_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip_file:
            tmp_zip_path = tmp_zip_file.name
            with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(vector_file, arcname="vectors.npz")
                zipf.write(metadata_file, arcname="metadata.jsonl")
            logger.info(f"Store {user_id}/{model_id} zipped to {tmp_zip_path}")

        background_tasks.add_task(os.remove, tmp_zip_path) # Aufgabe zum Löschen hinzufügen

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
def import_vector_store_from_zip(
    user_id: str = Query(...),
    model_id: str = Query(...),
    overwrite: bool = Query(False, description="Wenn True, wird ein bestehender Store überschrieben."),
    file: UploadFile = File(..., description="ZIP-Datei mit 'vectors.npz' und 'metadata.jsonl'")
):
    """Importiert einen Store aus einer ZIP-Datei."""
    if file.content_type != "application/zip":
        raise HTTPException(status_code=400, detail="Only ZIP files are supported.")

    with tempfile.TemporaryDirectory() as tmpdir:
        try: # Haupt-Try-Block für die Importlogik
            zip_path = os.path.join(tmpdir, file.filename if file.filename else "import.zip")
            logger.debug(f"Saving uploaded zip to temporary path: {zip_path}")
            content = file.file.read();
            if not content: raise HTTPException(status_code=400, detail="Uploaded ZIP file is empty.")
            with open(zip_path, "wb") as f: f.write(content)
            logger.debug(f"Zip file saved to {zip_path}")

            try: # Innerer Try für ZIP-Extraktion
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    expected_files = {"vectors.npz", "metadata.jsonl"}
                    found_files = set(zipf.namelist())
                    if not expected_files.issubset(found_files): missing = expected_files - found_files; raise HTTPException(status_code=400, detail=f"ZIP must contain {expected_files}. Missing: {missing}")
                    zipf.extract("vectors.npz", tmpdir)
                    zipf.extract("metadata.jsonl", tmpdir)
                    logger.debug(f"Extracted required files to {tmpdir}")
            except zipfile.BadZipFile: logger.error(f"Invalid ZIP file uploaded: {file.filename}"); raise HTTPException(status_code=400, detail="Invalid ZIP file format.")

            vec_path = os.path.join(tmpdir, "vectors.npz"); meta_path = os.path.join(tmpdir, "metadata.jsonl")

            target_store_exists = store_exists(user_id, model_id)
            if overwrite and target_store_exists:
                logger.warning(f"Overwrite=True. Deleting existing store: {user_id}/{model_id}")
                try: delete_store(user_id, model_id); target_store_exists = False
                except Exception as del_err: raise HTTPException(status_code=500, detail=f"Failed to delete existing store: {del_err}")
            elif not overwrite and target_store_exists: raise HTTPException(status_code=409, detail="Store already exists. Set overwrite=true to replace.")

            if not target_store_exists:
                try: create_store(user_id, model_id)
                except Exception as create_err: raise HTTPException(status_code=500, detail=f"Failed to create store structure: {create_err}")

            # Lade Vektoren
            try:
                loaded = mx.load(vec_path)
                if "vectors" not in loaded: raise ValueError("vectors.npz does not contain 'vectors' key.")
                vectors = loaded["vectors"]
                # --- log_tensor_debug wurde HIER ENTFERNT ---
                logger.debug(f"Successfully loaded vectors from zip with shape: {vectors.shape}") # Alternative Log-Ausgabe
            except Exception as load_err:
                 logger.error(f"Could not load vectors from extracted {vec_path}: {load_err}", exc_info=True)
                 # Gebe den ursprünglichen Fehler weiter für mehr Details
                 raise HTTPException(status_code=400, detail=f"Could not load vectors from ZIP: {load_err}")

            # Lade Metadaten
            metadata = []
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        try: metadata.append(json.loads(line))
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Invalid JSON in metadata.jsonl at line {i+1}: {json_err}")
                            # Gib einen klareren Fehler zurück
                            raise ValueError(f"Invalid JSON in metadata.jsonl at line {i+1}")
            except Exception as meta_err:
                 logger.error(f"Could not read metadata from extracted {meta_path}: {meta_err}", exc_info=True)
                 raise HTTPException(status_code=400, detail=f"Could not read metadata from ZIP: {meta_err}")

            # Konsistenzprüfung
            if vectors.shape[0] != len(metadata):
                logger.error(f"Mismatch after import: Vectors {vectors.shape[0]}, Metadata {len(metadata)}")
                raise HTTPException(status_code=400, detail=f"Data inconsistency in ZIP: Found {vectors.shape[0]} vectors but {len(metadata)} metadata entries.")

            # Füge die geladenen Daten hinzu
            logger.info(f"Adding {len(metadata)} vectors from import to {user_id}/{model_id}")
            add_vectors(user_id, model_id, vectors, metadata) # Übergibt mx.array

            # Erfolgreiche Rückgabe
            return {"status": "imported", "user_id": user_id, "model_id": model_id, "count": len(metadata)}

        # --- Exceptions für den Haupt-Try-Block ---
        except HTTPException as http_exc:
            raise http_exc # HTTPExceptions weiterleiten
        except Exception as e:
            logger.exception(f"General error during ZIP import process for {user_id}/{model_id}")
            raise HTTPException(status_code=500, detail=f"Failed to import ZIP: {str(e)}")