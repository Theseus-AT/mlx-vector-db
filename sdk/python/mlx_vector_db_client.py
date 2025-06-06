# Neue Datei: sdk/python/mlx_vector_db_client.py
# Implementiert einen asynchronen Python-Client für die MLX Vector DB REST API.
#
# MLX Specificity: Der Client sollte idealerweise mx.array als Eingabe für Vektoren
#                  akzeptieren und optional als Ausgabe anbieten. Für die JSON-Serialisierung
#                  über HTTP müssen mx.arrays jedoch in Listen konvertiert werden, wie
#                  im Plan korrekt vorgesehen.
# LLM Anbindung: Ein benutzerfreundliches SDK vereinfacht die Integration der Vektor-DB
#                in Python-basierte LLM-Workflows, RAG-Pipelines und Agentensysteme.
#                Asynchrone Aufrufe mit httpx sind vorteilhaft für I/O-lastige Operationen.

from typing import List, Dict, Any, Optional, Tuple, Union # Union hinzugefügt
import httpx # Muss in requirements.txt (oder setup.py des SDKs) sein
from dataclasses import dataclass, field # field hinzugefügt
import logging

# Importiere mx, wenn es für Typ-Annotationen oder Konvertierungen verwendet wird.
try:
    import mlx.core as mx
    # import numpy as np # Für die Konvertierung mx -> list, falls mx.tolist() nicht ausreicht
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None # Definiere mx als None, damit Typ-Hints funktionieren, wenn MLX nicht installiert ist

logger = logging.getLogger("mlx_vector_db_client")

@dataclass
class QueryResultItemSDK: # Angepasst an QueryResultItem in gRPC für Konsistenz
    id: Optional[str] = None # ID des Dokuments, falls in Metadaten
    distance: Optional[float] = None # Distanz oder
    similarity_score: Optional[float] = None # Ähnlichkeit
    metadata: Dict[str, Any] = field(default_factory=dict)
    # vector: Optional[List[float]] = None # Optional den Vektor zurückgeben

@dataclass
class QueryResponseSDK:
    results: List[QueryResultItemSDK]
    # query_id: Optional[str] = None # Falls vom Server zurückgegeben

@dataclass
class StoreStatsSDK:
    vectors: int
    metadata: int
    # Ggf. weitere Stats

class MLXVectorDBClientError(Exception):
    """Basis-Exception für Client-Fehler."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class MLXVectorDBClient:
    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 jwt_token: Optional[str] = None, # Hinzugefügt für JWT-Auth
                 timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        
        if api_key and jwt_token:
            logger.warning("Both API key and JWT token provided. Preferring JWT token.")
            self.headers["Authorization"] = f"Bearer {jwt_token}"
        elif jwt_token:
            self.headers["Authorization"] = f"Bearer {jwt_token}"
        elif api_key:
            # Der Plan verwendet X-API-Key, aber FastAPI's HTTPBearer erwartet "Authorization: Bearer <key>"
            # Wenn Ihr Server X-API-Key unterstützt, ist das ok.
            # Für Konsistenz mit JWT ist "Authorization: Bearer" oft besser.
            # Wir bleiben hier erstmal beim Plan:
            self.headers["X-API-Key"] = api_key
            # Alternativ für Bearer Token mit API Key (falls Server das so erwartet):
            # self.headers["Authorization"] = f"Bearer {api_key}"


        self.client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"MLXVectorDBClient initialized for base_url: {self.base_url}")

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = await self.client.request(method, url, headers=self.headers, **kwargs)
            response.raise_for_status() # Wirft HTTPStatusError für 4xx/5xx Antworten
            if response.status_code == 204: # No Content
                return {}
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {method.upper()} {url}: {e.response.text[:200]}")
            raise MLXVectorDBClientError(
                message=f"API request failed: {e.response.status_code} - {e.response.text}",
                status_code=e.response.status_code,
                response_data=e.response.text
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Request error for {method.upper()} {url}: {e}")
            raise MLXVectorDBClientError(f"Request failed: {e}") from e

    # --- Admin Endpoints ---
    async def create_store(self, user_id: str, model_id: str) -> Dict[str, Any]:
        """Erstellt einen neuen Vektor-Store."""
        # Nutzt den /admin/create_store Endpunkt
        return await self._request(
            "POST",
            "/admin/create_store",
            json={"user_id": user_id, "model_id": model_id}
        )

    async def delete_store(self, user_id: str, model_id: str) -> Dict[str, Any]:
        """Löscht einen Vektor-Store."""
        # Nutzt den /admin/store Endpunkt mit DELETE Methode
        # Annahme: Der Server erwartet user_id und model_id im Body für DELETE,
        # was unüblich ist. Üblicher wären Query-Parameter oder Pfadparameter.
        # Gemäß Ihrer admin.py Route ist es ein Body:
        return await self._request(
            "DELETE",
            "/admin/store",
            json={"user_id": user_id, "model_id": model_id}
        )

    async def get_store_stats(self, user_id: str, model_id: str) -> StoreStatsSDK:
        """Holt Statistiken für einen spezifischen Store."""
        # Nutzt /admin/store/stats
        response_data = await self._request(
            "GET",
            f"/admin/store/stats?user_id={user_id}&model_id={model_id}"
        )
        return StoreStatsSDK(
            vectors=response_data.get("vectors", 0),
            metadata=response_data.get("metadata", 0)
        )

    # --- Vector Endpoints ---
    async def add_vectors(self, user_id: str, model_id: str,
                         vectors: Union["mx.array", List[List[float]]], # Akzeptiert mx.array oder Listen
                         metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fügt Vektoren zu einem Store hinzu."""
        vectors_list: List[List[float]]
        if MLX_AVAILABLE and isinstance(vectors, mx.array):
            # Konvertiere MLX Array zu Liste für JSON-Serialisierung
            # mx.eval(vectors) # Sicherstellen, dass Berechnungen abgeschlossen sind
            # vectors_list = np.array(vectors).tolist() # Über Numpy
            vectors_list = vectors.tolist() # Direkte Konvertierung, falls performant
        elif isinstance(vectors, list):
            vectors_list = vectors
        else:
            raise TypeError("vectors must be an mlx.array or a list of lists of floats.")

        return await self._request(
            "POST",
            "/vectors/add", # Ihr Endpunkt aus vectors.py
            json={
                "user_id": user_id,
                "model_id": model_id,
                "vectors": vectors_list,
                "metadata": metadata
            }
        )

    async def query(self, user_id: str, model_id: str,
                   query_vector: Union["mx.array", List[float]], # Akzeptiert mx.array oder Liste
                   k: int = 10,
                   filter_metadata: Optional[Dict[str, Any]] = None) -> QueryResponseSDK:
        """Fragt ähnliche Vektoren ab."""
        query_vector_list: List[float]
        if MLX_AVAILABLE and isinstance(query_vector, mx.array):
            # mx.eval(query_vector)
            # query_vector_list = np.array(query_vector).tolist()
            query_vector_list = query_vector.tolist()
        elif isinstance(query_vector, list):
            query_vector_list = query_vector
        else:
            raise TypeError("query_vector must be an mlx.array or a list of floats.")

        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "query": query_vector_list,
            "k": k
        }
        if filter_metadata:
            payload["filter_metadata"] = filter_metadata
            
        response_data = await self._request("POST", "/vectors/query", json=payload)
        
        # Umwandlung der Server-Antwort in SDK-spezifische Datenklassen
        # Ihre API /vectors/query gibt {"results": [...]} zurück.
        # Jedes Item in "results" ist ein Dict mit 'id', 'similarity_score', 'metadata'.
        sdk_results = []
        for item in response_data.get("results", []):
            sdk_results.append(QueryResultItemSDK(
                id=item.get("id"), # Annahme: 'id' ist in den Metadaten
                similarity_score=item.get("similarity_score"),
                metadata=item # oder item.get("metadata", {}) falls verschachtelt
            ))
        return QueryResponseSDK(results=sdk_results)

    async def batch_query(self, user_id: str, model_id: str,
                         query_vectors: Union["mx.array", List[List[float]]],
                         k: int = 10,
                         filter_metadata: Optional[Dict[str, Any]] = None) -> List[QueryResponseSDK]:
        """Führt Batch-Abfragen aus."""
        queries_list: List[List[float]]
        if MLX_AVAILABLE and isinstance(query_vectors, mx.array):
            # mx.eval(query_vectors)
            # queries_list = np.array(query_vectors).tolist()
            queries_list = query_vectors.tolist()
        elif isinstance(query_vectors, list):
            queries_list = query_vectors
        else:
            raise TypeError("query_vectors must be an mlx.array or a list of lists of floats.")

        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "queries": queries_list,
            "k": k
        }
        if filter_metadata:
            payload["filter_metadata"] = filter_metadata

        # Der Server-Endpunkt /vectors/batch_query gibt {"results": [[item1, item2], [item3, item4]]} zurück
        # wobei jede innere Liste die Ergebnisse für eine Query im Batch ist.
        # Wir müssen dies entsprechend parsen.
        # Ihre aktuelle /batch_query Route gibt aber {"results": List[Dict]} zurück, was alle Ergebnisse
        # für alle Batch-Queries in einer flachen Liste zu sein scheint.
        # Der Plan war: "returns (stream QueryResponse)" für gRPC, aber für REST ist eine Liste von Listen üblicher.
        # Ich passe mich hier an Ihre /vectors/batch_query Route an, die eine flache Liste von Ergebnissen für alle Queries zurückgibt.
        # Dies bedeutet, der Client kann nicht direkt zuordnen, welches Ergebnis zu welcher Batch-Query gehört,
        # es sei denn, der Server fügt eine query_id oder ähnliches hinzu.
        # Für dieses Beispiel nehmen wir an, der Server gibt eine Liste von QueryResponseSDK-ähnlichen Strukturen zurück.
        # Wenn der Server eine Liste von Listen von Ergebnissen zurückgibt:
        # response_data = await self._request("POST", "/vectors/batch_query", json=payload)
        # all_query_responses = []
        # for single_query_results_list in response_data.get("results", []): # Annahme: [[res1, res2], [res3, res4]]
        #     sdk_results = []
        #     for item in single_query_results_list:
        #         sdk_results.append(QueryResultItemSDK(
        #             id=item.get("id"),
        #             similarity_score=item.get("similarity_score"),
        #             metadata=item
        #         ))
        #     all_query_responses.append(QueryResponseSDK(results=sdk_results))
        # return all_query_responses
        
        # Anpassung an Ihre aktuelle /vectors/batch_query, die eine flache Liste zurückgibt:
        # Dies ist nicht ideal für Batch-Queries, da die Zuordnung verloren geht.
        # Der Server sollte idealerweise eine Struktur zurückgeben, die die Batch-Ergebnisse gruppiert.
        # Wir nehmen an, die Route gibt eine Liste von Listen zurück:
        response_container = await self._request("POST", "/vectors/batch_query", json=payload)
        # Die /batch_query Route in vectors.py gibt {"results": results} zurück,
        # wobei 'results' eine Liste von Listen von Dictionaries ist (List[List[Dict[str, Any]]])
        # Jede innere Liste ist das Ergebnis für eine Query.
        
        batch_responses: List[QueryResponseSDK] = []
        server_results_list_of_lists = response_container.get("results", [])

        for single_query_result_list in server_results_list_of_lists:
            sdk_items = []
            for item_dict in single_query_result_list:
                sdk_items.append(QueryResultItemSDK(
                    id=item_dict.get("id"),
                    similarity_score=item_dict.get("similarity_score"),
                    metadata=item_dict # oder item_dict.get("metadata")
                ))
            batch_responses.append(QueryResponseSDK(results=sdk_items))
        return batch_responses


    async def count_vectors(self, user_id: str, model_id: str) -> StoreStatsSDK:
        """Zählt Vektoren und Metadaten in einem Store."""
        response_data = await self._request(
            "GET",
            f"/vectors/count?user_id={user_id}&model_id={model_id}"
        )
        return StoreStatsSDK(
            vectors=response_data.get("vectors", 0),
            metadata=response_data.get("metadata", 0)
        )

    async def delete_vectors_by_metadata(self, user_id: str, model_id: str, filter_metadata: Dict[str, Any]) -> int:
        """Löscht Vektoren basierend auf Metadaten-Filter."""
        response_data = await self._request(
            "POST", # Ihre Route ist POST
            "/vectors/delete",
            json={
                "user_id": user_id,
                "model_id": model_id,
                "filter_metadata": filter_metadata
            }
        )
        return response_data.get("deleted", 0)

    # --- Health & Config ---
    async def health_check(self) -> Dict[str, Any]:
        """Führt einen einfachen Health Check durch."""
        # Nutzt /health (nicht /monitoring/health, da das für detailliertere Infos ist)
        return await self._request("GET", "/health")
        
    async def get_config(self) -> Dict[str, Any]:
        """Ruft die Server-Konfiguration ab."""
        # Nutzt /config (muss in Ihren Routen existieren)
        return await self._request("GET", "/config") # Annahme: /config Endpunkt existiert


    async def close(self):
        """Schließt die HTTP-Client-Session."""
        await self.client.aclose()
        logger.info("MLXVectorDBClient closed.")

# Beispielverwendung (asynchron):
# async def main():
#     client = MLXVectorDBClient(base_url="http://localhost:8000", api_key="your_api_key_or_jwt")
#
#     try:
#         # Store erstellen (Admin-Funktion, benötigt Admin-Rechte/Key)
#         # print(await client.create_store("sdk_user", "sdk_model"))
#
#         # Vektoren hinzufügen
#         if MLX_AVAILABLE:
#             vectors_mx = mx.random.normal((10, 128))
#             metadata_list = [{"id": f"sdk_vec_{i}"} for i in range(10)]
#             print(await client.add_vectors("sdk_user", "sdk_model", vectors_mx, metadata_list))
#
#             # Query
#             query_vec_mx = vectors_mx[0]
#             query_result = await client.query("sdk_user", "sdk_model", query_vec_mx, k=3)
#             print("Query Results:", query_result)
#
#             # Batch Query
#             batch_query_vecs_mx = vectors_mx[1:4]
#             batch_results = await client.batch_query("sdk_user", "sdk_model", batch_query_vecs_mx, k=2)
#             for i, res_sdk in enumerate(batch_results):
#                 print(f"Batch Query {i} Results: {res_sdk}")
#
#         # Stats
#         stats = await client.count_vectors("sdk_user", "sdk_model")
#         print("Store Stats:", stats)
#
#     except MLXVectorDBClientError as e:
#         print(f"Client Error: {e.message}, Status: {e.status_code}, Response: {e.response_data}")
#     finally:
#         await client.close()
#
# if __name__ == "__main__":
#     asyncio.run(main())