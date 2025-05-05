from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

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