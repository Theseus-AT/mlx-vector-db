from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
# In api/routes/vectors.py (Beispielhafte Anpassung)

from fastapi import APIRouter, Depends, HTTPException, status # status hinzugefügt
# ... andere Importe ...
from security.auth import get_current_user_payload # Für JWT-Payload
from security.rbac import require_permission, Permission, Role # Für RBAC

# ... (bestehende Pydantic Modelle) ...

# Beispiel: Anpassung der Route /vectors/query
@router.post("/query", response_model=QueryResultsResponse)
@require_permission(Permission.QUERY_VECTORS) # RBAC-Schutz
async def query_vector_data(
    request: VectorQueryRequest,
    current_user_payload: Dict[str, Any] = Depends(get_current_user_payload) # JWT Auth
):
    # Multi-Tenancy Check: Sicherstellen, dass der User (aus JWT) auf den angefragten Store zugreifen darf.
    # Diese Logik hängt davon ab, wie Sie User-Berechtigungen auf Stores verwalten.
    # Einfaches Beispiel: User darf nur auf Stores mit seiner eigenen User-ID zugreifen.
    jwt_user_id = current_user_payload.get("sub")
    if request.user_id != jwt_user_id and Role.ADMIN.value not in current_user_payload.get("roles", []):
        # Wenn der angeforderte user_id nicht der des Tokens ist UND der User kein Admin ist
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User {jwt_user_id} is not authorized to access data for user {request.user_id}"
        )

    # Bestehende Logik zur Vektorabfrage
    arr = np.array(request.query, dtype=np.float32) # Bleibt, da FastAPI JSON-Listen liefert
    # Hier könnte eine Konvertierung zu mx.array erfolgen, wenn query_vectors dies erwartet.
    # query_mx = mx.array(request.query, dtype=mx.float32)
    
    results = query_vectors(
        request.user_id,
        request.model_id,
        arr, # oder query_mx
        k=request.k,
        filter_metadata=request.filter_metadata
    )
    return {"results": results}

from fastapi.responses import StreamingResponse
import json
from service.vector_store import (
    add_vectors,
    query_vectors,
    batch_query,
    stream_query,
    store_exists,
    create_store,
    count_vectors,
    delete_vectors
)
class VectorBatchQueryRequest(BaseModel):
    user_id: str
    model_id: str
    queries: List[List[float]]
    k: Optional[int] = 5
    filter_metadata: Optional[Dict] = None

class VectorStreamQueryRequest(BaseModel):
    user_id: str
    model_id: str
    queries: List[List[float]]
    k: Optional[int] = 5
    filter_metadata: Optional[Dict] = None

router = APIRouter(prefix="/vectors", tags=["vector_store"])

class VectorAddRequest(BaseModel):
    user_id: str
    model_id: str
    vectors: List[List[float]]
    metadata: List[Dict]

class VectorQueryRequest(BaseModel):
    user_id: str
    model_id: str
    query: List[float]
    k: Optional[int] = 5
    filter_metadata: Optional[Dict] = None

class StoreCreateRequest(BaseModel):
    user_id: str
    model_id: str

class StoreDeleteRequest(BaseModel):
    user_id: str
    model_id: str
    filter_metadata: Dict

class StatusResponse(BaseModel):
    status: str

class QueryResultsResponse(BaseModel):
    results: List[Dict]

class DeleteResponse(BaseModel):
    deleted: int

@router.post("/add", response_model=StatusResponse)
def add_vector_data(request: VectorAddRequest):
    arr = np.array(request.vectors, dtype=np.float32)
    add_vectors(request.user_id, request.model_id, arr, request.metadata)
    return {"status": "ok"}

@router.post("/query", response_model=QueryResultsResponse)
def query_vector_data(request: VectorQueryRequest):
    arr = np.array(request.query, dtype=np.float32)
    results = query_vectors(request.user_id, request.model_id, arr, k=request.k, filter_metadata=request.filter_metadata)
    return {"results": results}

@router.post("/create", response_model=StatusResponse)
def create_store_route(request: StoreCreateRequest):
    try:
        create_store(request.user_id, request.model_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/count")
def count_vectors_route(user_id: str, model_id: str):
    if not store_exists(user_id, model_id):
        raise HTTPException(status_code=404, detail="Store not found")
    return count_vectors(user_id, model_id)

@router.post("/delete", response_model=DeleteResponse)
def delete_vectors_route(request: StoreDeleteRequest):
    if not store_exists(request.user_id, request.model_id):
        raise HTTPException(status_code=404, detail="Store not found")
    try:
        deleted = delete_vectors(request.user_id, request.model_id, request.filter_metadata)
        return {"deleted": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch_query", response_model=QueryResultsResponse)
def batch_query_route(request: VectorBatchQueryRequest):
    arr = np.array(request.queries, dtype=np.float32)
    results = batch_query(request.user_id, request.model_id, arr, k=request.k, filter_metadata=request.filter_metadata)
    return {"results": results}

@router.post("/stream_query")
def stream_query_route(request: VectorStreamQueryRequest):
    arr = np.array(request.queries, dtype=np.float32)
    def result_generator():
        for result in stream_query(request.user_id, request.model_id, arr, k=request.k, filter_metadata=request.filter_metadata):
            yield json.dumps(result) + "\n"
    return StreamingResponse(result_generator(), media_type="application/x-ndjson")


# New routes
@router.get("/users")
def list_all_users():
    """Gibt alle Nutzer zurück, für die ein Vektorstore existiert."""
    from service.vector_store import list_users
    return list_users()

from fastapi import Query

@router.get("/models")
def list_models_for_user(
    user_id: str = Query(..., description="User-ID, dessen Modelle aufgelistet werden sollen")
):
    from service.vector_store import list_models
    return list_models(user_id)

@router.get("/ping", response_model=StatusResponse)
def ping():
    """Einfacher Healthcheck für Verfügbarkeitsprüfung."""
    return {"status": "pong"}