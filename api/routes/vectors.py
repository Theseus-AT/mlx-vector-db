# api/routes/vectors.py
# Überarbeitete Version, die die neue VectorStore-Klasse verwendet
# und Authentifizierung/Autorisierung integriert.

from fastapi import APIRouter, Depends, HTTPException, status, Query as FastAPIQuery # Query umbenannt
from pydantic import BaseModel, Field # Field für Validierung
from typing import List, Dict, Optional, Any, Union
import mlx.core as mx # Für die Konvertierung von Request-Daten
import numpy as np # Als Zwischenschritt für Listen -> mx.array

# Importiere die VectorStore-Klasse und Config
# Passen Sie den Pfad an Ihre Projektstruktur an.
# z.B. from service.vector_store import VectorStore, VectorStoreConfig, get_store
# Wenn vector_store.py jetzt im Root liegt:
from vector_store import VectorStore, VectorStoreConfig, get_store # get_store für Legacy-Wrapper oder einfache Instanzverwaltung

# Security-Importe
from security.auth import get_current_user_payload # Für JWT-Payload
from security.rbac import require_permission, Permission, Role # Für RBAC

# Globale Konfiguration laden (für Basispfad etc.)
from config.settings import get_config
config_manager = get_config()

router = APIRouter() # Prefix und Tags werden beim Include in main.py gesetzt

# --- Pydantic Modelle für Requests/Responses (bleiben größtenteils wie in Ihrer Datei) ---
class VectorAddRequest(BaseModel):
    user_id: str
    model_id: str
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]] # Metadaten können beliebige JSON-kompatible Werte haben
    # Optional: Dimension, falls sie nicht aus dem ersten Vektor abgeleitet werden soll
    # dimension: Optional[int] = None

class VectorQueryRequest(BaseModel):
    user_id: str
    model_id: str
    query: List[float]
    k: int = Field(10, gt=0, le=1000) # K mit Validierung
    filter_metadata: Optional[Dict[str, Any]] = None
    use_hnsw: Optional[bool] = None # Erlaube explizite Wahl der Suchmethode

class VectorBatchQueryRequest(BaseModel):
    user_id: str
    model_id: str
    queries: List[List[float]]
    k: int = Field(10, gt=0, le=1000)
    filter_metadata: Optional[Dict[str, Any]] = None
    use_hnsw: Optional[bool] = None

class VectorDeleteRequest(BaseModel): # Umbenannt von StoreDeleteRequest für Klarheit
    user_id: str
    model_id: str
    # Löschen anhand von IDs oder Filter
    vector_ids_to_delete: Optional[List[str]] = None # Annahme: Metadaten enthalten eine 'id'
    filter_metadata_to_delete: Optional[Dict[str, Any]] = None

    # @model_validator(mode='after') # Pydantic v2
    # def check_one_delete_option(cls, values):
    #     if not (values.get("vector_ids_to_delete") or values.get("filter_metadata_to_delete")):
    #         raise ValueError("Entweder 'vector_ids_to_delete' oder 'filter_metadata_to_delete' muss angegeben werden.")
    #     if values.get("vector_ids_to_delete") and values.get("filter_metadata_to_delete"):
    #         raise ValueError("Nur eine von 'vector_ids_to_delete' oder 'filter_metadata_to_delete' darf angegeben werden.")
    #     return values


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    count: Optional[int] = None

class QueryResultItemResponse(BaseModel): # Detailliertere Antwort für Query-Items
    id: Optional[str] = None # Aus Metadaten
    distance: float
    similarity_score: float # Berechnet aus Distanz
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    results: List[QueryResultItemResponse]
    query_time_ms: Optional[float] = None # Performance-Info

class BatchQueryResponse(BaseModel):
    batch_results: List[QueryResponse] # Eine QueryResponse pro Eingabe-Query

class CountResponse(BaseModel):
    vectors: int
    metadata: int
    dimension: Optional[int] = None
    # Ggf. weitere Stats aus VectorStore.get_stats()

class DeleteResponse(BaseModel):
    deleted_count: int
    message: Optional[str] = None


# --- Dependency für VectorStore Instanz ---
def get_vector_store_dependency(user_id: str, model_id: str) -> VectorStore:
    """
    FastAPI Dependency, die eine VectorStore-Instanz für den gegebenen User/Modell bereitstellt.
    Verwendet die get_store Hilfsfunktion (oder eine ähnliche Factory/Registry).
    """
    # Hier könnte die `VectorStoreConfig` dynamisch aus der Hauptkonfiguration
    # (config_manager) basierend auf user_id/model_id oder globalen Settings erstellt werden.
    # Fürs Erste verwenden wir die Default-Config von VectorStore, wenn keine spezifische übergeben wird.
    
    # Erstelle eine spezifische HNSWConfig basierend auf globalen Einstellungen
    hnsw_perf_config = config_manager.performance # Globale PerformanceConfig
    vs_config_obj = VectorStoreConfig(
        enable_hnsw=hnsw_perf_config.enable_hnsw,
        hnsw_m=hnsw_perf_config.hnsw_m,
        hnsw_ef_construction=hnsw_perf_config.hnsw_ef_construction,
        hnsw_ef_search=hnsw_perf_config.hnsw_ef_search,
        hnsw_metric=hnsw_perf_config.hnsw_metric,
        hnsw_num_threads=getattr(hnsw_perf_config, 'hnsw_num_threads', os.cpu_count() or 1), # aus HNSWConfig default nehmen
        auto_index_threshold=hnsw_perf_config.auto_index_threshold,
        cache_enabled=getattr(hnsw_perf_config, 'cache_enabled', True), # Sie hatten es in VectorStoreConfig, nicht PerformanceConfig
        query_cache_max_size=hnsw_perf_config.query_result_cache_max_size,
        query_cache_ttl_seconds=hnsw_perf_config.query_result_cache_ttl_seconds
    )
    
    # Verwende die get_store Funktion, die eine Instanz pro Pfad verwaltet.
    # Der Basispfad kommt aus der globalen Konfiguration.
    store = get_store(user_id, model_id, config=vs_config_obj) # get_store aus Ihrer vector_store.py
    if not isinstance(store, VectorStore): # Sicherheitshalber prüfen
        raise HTTPException(status_code=500, detail="Interner Fehler: VectorStore-Instanz konnte nicht korrekt abgerufen werden.")
    return store

# --- API Endpunkte ---

@router.post("/add", response_model=StatusResponse)
@require_permission(Permission.ADD_VECTORS)
async def add_vector_data_route(
    request_data: VectorAddRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_payload), # JWT Auth
    store: VectorStore = Depends(lambda request_data: get_vector_store_dependency(request_data.user_id, request_data.model_id)) # Trick für Dependency
):
    # Multi-Tenancy Check (Beispiel)
    jwt_user_id = current_user.get("sub")
    if request_data.user_id != jwt_user_id and Role.ADMIN.value not in current_user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nicht autorisiert für diesen User-Store.")

    try:
        # Konvertiere Listen zu mx.array
        # Die Dimension wird vom ersten Vektor abgeleitet oder muss konsistent sein.
        if not request_data.vectors:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vektorliste darf nicht leer sein.")
        
        # Vektoren als mx.array konvertieren
        vectors_mx = mx.array(np.array(request_data.vectors, dtype=np.float32)) # Sicherstellen, dass es float32 ist
        
        store.add_vectors(vectors_mx, request_data.metadata)
        return StatusResponse(status="ok", message=f"{vectors_mx.shape[0]} Vektoren hinzugefügt.")
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Fehler beim Hinzufügen von Vektoren zu {request_data.user_id}/{request_data.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Interner Fehler: {str(e)}")


@router.post("/query", response_model=QueryResponse)
@require_permission(Permission.QUERY_VECTORS)
async def query_vector_data_route( # Umbenannt, um Konflikt mit globaler Funktion zu vermeiden
    request_data: VectorQueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_payload),
    store: VectorStore = Depends(lambda request_data: get_vector_store_dependency(request_data.user_id, request_data.model_id))
):
    jwt_user_id = current_user.get("sub")
    if request_data.user_id != jwt_user_id and Role.ADMIN.value not in current_user.get("roles", []):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nicht autorisiert für diesen User-Store.")

    try:
        query_mx = mx.array(request_data.query, dtype=mx.float32) # Konvertiere zu mx.array
        
        query_start_time = time.perf_counter()
        indices, distances, metadatas = store.query(
            query_mx,
            k=request_data.k,
            use_hnsw=request_data.use_hnsw,
            metadata_filter=request_data.filter_metadata
        )
        query_time_ms = (time.perf_counter() - query_start_time) * 1000

        results_response = []
        for i in range(len(indices)):
            meta = metadatas[i].copy()
            doc_id = meta.get("id") # Versuche 'id' aus Metadaten zu holen
            # Distanz zu Ähnlichkeit (Beispiel für Cosine, anpassen falls L2 primär ist)
            # Ihre HNSWIndex.search gibt Distanzen zurück.
            similarity = 0.0
            if store.hnsw_index and store.hnsw_index.config.metric == 'cosine':
                similarity = 1.0 - distances[i]
            elif store.config.hnsw_config.metric == 'cosine': # Fallback auf Store-Config
                 similarity = 1.0 - distances[i]
            else: # L2, hier ist -Distanz ein Score (größer ist besser)
                similarity = -distances[i]


            results_response.append(QueryResultItemResponse(
                id=str(doc_id) if doc_id is not None else f"index_{indices[i]}",
                distance=distances[i],
                similarity_score=similarity,
                metadata=meta
            ))
        
        return QueryResponse(results=results_response, query_time_ms=round(query_time_ms,3))
    except Exception as e:
        logger.error(f"Fehler bei Query für {request_data.user_id}/{request_data.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Interner Query-Fehler: {str(e)}")


@router.post("/batch_query", response_model=BatchQueryResponse)
@require_permission(Permission.QUERY_VECTORS)
async def batch_query_data_route(
    request_data: VectorBatchQueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_payload),
    store: VectorStore = Depends(lambda request_data: get_vector_store_dependency(request_data.user_id, request_data.model_id))
):
    jwt_user_id = current_user.get("sub")
    if request_data.user_id != jwt_user_id and Role.ADMIN.value not in current_user.get("roles",[]):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nicht autorisiert für diesen User-Store.")

    try:
        if not request_data.queries:
            return BatchQueryResponse(batch_results=[])

        queries_mx = mx.array(np.array(request_data.queries, dtype=np.float32))
        
        query_start_time = time.perf_counter()
        # Die store.batch_query Methode gibt bereits List[List[indices]], List[List[distances]], List[List[metadata]] zurück
        all_indices_list, all_distances_list, all_metadata_list = store.batch_query(
            queries_mx,
            k=request_data.k,
            # use_hnsw wird von store.batch_query intern gehandhabt, falls implementiert
            metadata_filter=request_data.filter_metadata
        )
        batch_query_time_ms = (time.perf_counter() - query_start_time) * 1000
        
        batch_responses = []
        for i in range(len(all_indices_list)):
            single_query_results = []
            for j in range(len(all_indices_list[i])):
                meta = all_metadata_list[i][j].copy()
                doc_id = meta.get("id")
                distance = all_distances_list[i][j]
                similarity = 0.0
                if store.hnsw_index and store.hnsw_index.config.metric == 'cosine':
                    similarity = 1.0 - distance
                elif store.config.hnsw_config.metric == 'cosine':
                     similarity = 1.0 - distance
                else:
                    similarity = -distance

                single_query_results.append(QueryResultItemResponse(
                    id=str(doc_id) if doc_id is not None else f"index_{all_indices_list[i][j]}",
                    distance=distance,
                    similarity_score=similarity,
                    metadata=meta
                ))
            batch_responses.append(QueryResponse(results=single_query_results, query_time_ms=None)) # Zeit pro Sub-Query nicht einfach messbar hier

        logger.info(f"Batch Query für {request_data.user_id}/{request_data.model_id} ({queries_mx.shape[0]} Queries) in {batch_query_time_ms:.2f}ms verarbeitet.")
        return BatchQueryResponse(batch_results=batch_responses)

    except Exception as e:
        logger.error(f"Fehler bei Batch-Query für {request_data.user_id}/{request_data.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Interner Batch-Query-Fehler: {str(e)}")


@router.get("/count", response_model=CountResponse)
@require_permission(Permission.COUNT_VECTORS)
async def count_vectors_data_route( # Umbenannt
    user_id: str = FastAPIQuery(...), model_id: str = FastAPIQuery(...),
    current_user: Dict[str, Any] = Depends(get_current_user_payload),
    # store: VectorStore = Depends(get_vector_store_dependency) # Geht nicht direkt mit Query-Params
):
    # Manuelle Instanziierung für Routen mit Query-Parametern für user_id/model_id
    store = get_vector_store_dependency(user_id, model_id)

    jwt_user_id = current_user.get("sub")
    if user_id != jwt_user_id and Role.ADMIN.value not in current_user.get("roles",[]):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nicht autorisiert für diesen User-Store.")
    
    # Ihre alte /count Route prüfte store_exists. VectorStore._load macht das implizit.
    # Wenn der Store nicht existiert, sind vectors/metadata leer oder None.
    try:
        stats = store.get_stats() # Verwendet die get_stats Methode von VectorStore
        return CountResponse(
            vectors=stats.get('total_vectors', 0),
            metadata=stats.get('metadata_count', 0),
            dimension=stats.get('vector_dimension')
        )
    except FileNotFoundError: # Falls get_store eine Exception wirft, wenn Pfad nicht existiert
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Store nicht gefunden.")
    except Exception as e:
        logger.error(f"Fehler beim Zählen der Vektoren für {user_id}/{model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/delete", response_model=DeleteResponse)
@require_permission(Permission.DELETE_VECTORS)
async def delete_vectors_data_route( # Umbenannt
    request_data: VectorDeleteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_payload),
    store: VectorStore = Depends(lambda request_data: get_vector_store_dependency(request_data.user_id, request_data.model_id))
):
    jwt_user_id = current_user.get("sub")
    if request_data.user_id != jwt_user_id and Role.ADMIN.value not in current_user.get("roles",[]):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nicht autorisiert für diesen User-Store.")

    if not (request_data.vector_ids_to_delete or request_data.filter_metadata_to_delete):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="vector_ids_to_delete oder filter_metadata_to_delete erforderlich.")
    if request_data.vector_ids_to_delete and request_data.filter_metadata_to_delete:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nur eine von vector_ids_to_delete oder filter_metadata_to_delete erlaubt.")

    try:
        indices_to_delete_list: List[int] = []
        if request_data.vector_ids_to_delete:
            # Konvertiere Dokument-IDs (Strings) zu internen numerischen Indizes
            # Dies erfordert, dass die Metadaten eine eindeutige 'id' enthalten.
            if store.vectors is None: # Kein Store oder leer
                 return DeleteResponse(deleted_count=0, message="Store ist leer oder nicht initialisiert.")

            id_to_idx_map = {str(meta.get("id")): i for i, meta in enumerate(store.metadata) if meta.get("id") is not None}
            for doc_id_str in request_data.vector_ids_to_delete:
                if doc_id_str in id_to_idx_map:
                    indices_to_delete_list.append(id_to_idx_map[doc_id_str])
                else:
                    logger.warning(f"Zu löschende Vektor-ID '{doc_id_str}' nicht im Store {request_data.user_id}/{request_data.model_id} gefunden.")
        
        elif request_data.filter_metadata_to_delete:
            if store.vectors is None: return DeleteResponse(deleted_count=0, message="Store ist leer.")
            for i, meta in enumerate(store.metadata):
                match = all(meta.get(key) == value for key, value in request_data.filter_metadata_to_delete.items())
                if match:
                    indices_to_delete_list.append(i)
        
        if not indices_to_delete_list:
            return DeleteResponse(deleted_count=0, message="Keine Vektoren entsprachen den Löschkriterien.")

        deleted_count = store.delete_vectors(indices_to_delete_list)
        return DeleteResponse(deleted_count=deleted_count, message=f"{deleted_count} Vektoren gelöscht.")
    except Exception as e:
        logger.error(f"Fehler beim Löschen von Vektoren aus {request_data.user_id}/{request_data.model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Interner Fehler beim Löschen: {str(e)}")


# Die Endpunkte /stream_query, /users, /models, /ping können ähnlich angepasst werden,
# um die VectorStore-Instanz zu verwenden und ggf. durch RBAC zu schützen.
# Für /users und /models (die Store-übergreifend sind) wird keine einzelne VectorStore-Instanz benötigt,
# sondern die globalen list_users/list_models Funktionen aus der (alten) service.vector_store.py
# oder eine neue Manager-Klasse, die alle Stores kennt.

# Beibehalten der alten list_users/list_models für den Moment, bis Store Management überarbeitet ist
@router.get("/users", summary="List all users with vector stores")
# @require_permission(Permission.LIST_STORES) # Beispiel: Admin-Berechtigung
async def list_all_users_route(current_user: Optional[Dict[str, Any]] = Depends(get_current_user_payload if config_manager.security.enable_auth else lambda: None)):
    # Diese Route sollte idealerweise Admin-geschützt sein.
    # Wenn RBAC aktiv ist und der User kein Admin ist, wird require_permission() den Zugriff verweigern.
    # Hier eine zusätzliche manuelle Prüfung für den Fall, dass RBAC nicht für alle Routen genutzt wird:
    if config_manager.security.enable_auth and (not current_user or Role.ADMIN.value not in current_user.get("roles",[])):
         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin-Zugriff erforderlich.")
    
    from vector_store import _get_store_path # Importiere die Hilfsfunktion
    # Diese Logik muss angepasst werden, da list_users nicht mehr direkt existiert.
    # Es muss das Basisverzeichnis der Stores durchsucht werden.
    base_data_path = Path(config_manager.storage.base_path).expanduser()
    users = set()
    if base_data_path.is_dir():
        for user_dir_candidate in base_data_path.iterdir():
            if user_dir_candidate.is_dir() and user_dir_candidate.name.startswith("user_"):
                users.add(user_dir_candidate.name[len("user_"):])
    return list(users)


@router.get("/models", summary="List models for a specific user")
# @require_permission(Permission.LIST_STORES) # Oder eine spezifischere User-Berechtigung für eigene Stores
async def list_models_for_user_route(
    user_id: str = FastAPIQuery(..., description="User-ID, dessen Modelle aufgelistet werden sollen"),
    current_user: Dict[str, Any] = Depends(get_current_user_payload  if config_manager.security.enable_auth else lambda: None)
):
    jwt_user_id = current_user.get("sub") if current_user else None
    if config_manager.security.enable_auth and user_id != jwt_user_id and (not current_user or Role.ADMIN.value not in current_user.get("roles",[])):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Nicht autorisiert für diesen User.")

    from vector_store import _get_store_path
    user_path = Path(config_manager.storage.base_path).expanduser() / f"user_{user_id}"
    models = []
    if user_path.is_dir():
        for model_dir_candidate in user_path.iterdir():
            if model_dir_candidate.is_dir(): # Jedes Unterverzeichnis ist ein Modell
                models.append(model_dir_candidate.name)
    return models