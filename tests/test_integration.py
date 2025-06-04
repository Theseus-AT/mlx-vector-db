# Neue Datei: tests/test_integration.py
# Testet den gesamten Workflow über API-Endpunkte.

# MLX Specificity: Indirekt, da die API-Endpunkte intern MLX-Operationen verwenden.
#                  Die Tests stellen sicher, dass die Serialisierung/Deserialisierung
#                  von Vektoren (die als mx.array intern existieren könnten) über die
#                  API korrekt funktioniert.
# LLM Anbindung: Simulation typischer RAG-Abläufe (Store erstellen, Wissen indizieren,
#                Abfragen stellen) auf API-Ebene.

import pytest
import httpx # Für asynchrone HTTP-Anfragen, wie im Plan für SDK
import numpy as np
import os
import time

# Annahme: Die FastAPI App Instanz ist importierbar oder wird über ein Fixture bereitgestellt.
# Für Tests direkt gegen die laufende App (wie im Plan) ist eine Basis-URL nötig.
# Wenn Tests gegen die ASGI-App laufen, kann `app` direkt übergeben werden.

# BASE_URL = "http://localhost:8000" # Für Tests gegen laufenden Server
# Für Tests mit TestClient von FastAPI:
from fastapi.testclient import TestClient
from main import app # Annahme: Ihre FastAPI-App-Instanz ist in main.py

# Verwenden Sie den TestClient für Integrationstests ohne laufenden Server
# client = TestClient(app) # Synchrone Variante

# API Key für Tests (aus Umgebungsvariable oder Default)
TEST_API_KEY = os.getenv("VECTOR_DB_API_KEY_TEST", "test-dev-key-if-none-set")
# Wichtig: Wenn Authentifizierung aktiv ist, muss dieser Key am Server bekannt sein.
# Für Tests könnte man die Authentifizierung mocken oder einen dedizierten Test-Key verwenden.
# Für den Moment nehmen wir an, dass der `verify_api_key` mit diesem Schlüssel funktioniert.
# Wenn Sie JWT verwenden, müssten Sie einen Token generieren und verwenden.

# Helferfunktion, um zu warten, bis der Server bereit ist (nur für Tests gegen laufenden Server)
# def wait_for_server(base_url, timeout=30):
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         try:
#             response = httpx.get(f"{base_url}/health")
#             if response.status_code == 200:
#                 print("Server is ready.")
#                 return
#         except httpx.RequestError:
#             time.sleep(0.5)
#     raise TimeoutError("Server did not become ready in time.")


# @pytest.fixture(scope="session", autouse=True)
# def ensure_server_is_ready_for_integration_tests():
#     # Diese Fixture würde nur laufen, wenn man gegen einen externen Server testet.
#     # wait_for_server(BASE_URL)
#     pass


@pytest.mark.asyncio # Markiert Test als asynchron
async def test_full_workflow_integration(async_client: httpx.AsyncClient): # async_client Fixture von pytest-httpx
    """
    Testet den kompletten Workflow: Store erstellen, Vektoren hinzufügen, Abfragen.
    Verwendet pytest-httpx für einen asynchronen Client gegen die ASGI-App.
    """
    user_id = "integration_test_user"
    model_id = "integration_test_model"
    headers = {"X-API-Key": TEST_API_KEY, "Content-Type": "application/json"}

    # Vorbereitung: Sicherstellen, dass der Store nicht existiert (optional, für saubere Tests)
    # Dies kann über einen Admin-Endpunkt erfolgen, falls vorhanden und gewünscht.
    # Für diesen Test gehen wir davon aus, dass der Store zu Beginn nicht existiert oder
    # dass ein Fehler bei der Erstellung (409 Conflict) korrekt behandelt wird.
    # Man könnte auch versuchen, ihn zu löschen, falls er existiert.
    # try:
    #     await async_client.delete(f"/admin/store", json={"user_id": user_id, "model_id": model_id}, headers=headers)
    # except httpx.HTTPStatusError as e:
    #     if e.response.status_code != 404: # Ok, wenn nicht gefunden
    #         raise

    # 1. Store erstellen (Admin-Endpunkt)
    print("Integration Test: Creating store...")
    response = await async_client.post(
        "/admin/create_store", # Ihre Route aus admin.py
        json={"user_id": user_id, "model_id": model_id},
        headers=headers
    )
    assert response.status_code == 200, f"Failed to create store: {response.text}"
    create_data = response.json()
    assert create_data["status"] == "created"
    assert create_data["user_id"] == user_id
    assert create_data["model_id"] == model_id
    print(f"Store {user_id}/{model_id} created.")

    # 2. Vektoren hinzufügen (Kern-Endpunkt oder Admin-Test-Endpunkt)
    # Der Plan verwendet /vectors/add, Ihre admin.py hat /admin/add_test_vectors.
    # Wir verwenden hier /vectors/add, da es der Standardweg ist.
    print("Integration Test: Adding vectors...")
    num_vectors = 100
    dimension = 128 # Muss mit der Erwartung des Servers übereinstimmen
    
    # Generiere Vektoren mit NumPy, dann konvertiere zu Liste für JSON
    np_vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
    vectors_list = np_vectors.tolist()
    metadata = [{"doc_id": f"doc_{i}", "content_hash": f"hash_{i}"} for i in range(num_vectors)]
    
    response = await async_client.post(
        "/vectors/add", # Ihre Route aus vectors.py
        json={
            "user_id": user_id,
            "model_id": model_id,
            "vectors": vectors_list,
            "metadata": metadata
        },
        headers=headers
    )
    assert response.status_code == 200, f"Failed to add vectors: {response.text}"
    add_data = response.json()
    assert add_data["status"] == "ok"
    print(f"{num_vectors} vectors added.")

    # 2.1. Anzahl der Vektoren prüfen
    print("Integration Test: Counting vectors...")
    response = await async_client.get(
        f"/vectors/count?user_id={user_id}&model_id={model_id}",
        headers=headers
    )
    assert response.status_code == 200, f"Failed to count vectors: {response.text}"
    count_data = response.json()
    assert count_data["vectors"] == num_vectors
    assert count_data["metadata"] == num_vectors
    print(f"Vector count verified: {count_data['vectors']}")

    # 3. Vektoren abfragen
    print("Integration Test: Querying vectors...")
    query_vector_list = vectors_list[0] # Nimm den ersten hinzugefügten Vektor als Query
    k_neighbors = 5
    
    response = await async_client.post(
        "/vectors/query", # Ihre Route aus vectors.py
        json={
            "user_id": user_id,
            "model_id": model_id,
            "query": query_vector_list,
            "k": k_neighbors
        },
        headers=headers
    )
    assert response.status_code == 200, f"Failed to query vectors: {response.text}"
    query_response_data = response.json()
    
    # Die Antwortstruktur Ihrer /vectors/query Route ist {"results": List[Dict]}
    results = query_response_data.get("results", [])
    assert len(results) == k_neighbors
    
    # Das erste Ergebnis sollte der Query-Vektor selbst sein (oder ihm sehr ähnlich)
    # Annahme: Metadaten enthalten 'doc_id'
    assert results[0]["metadata"]["doc_id"] == "doc_0"
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > 0.999 # Sehr hohe Ähnlichkeit
    print(f"Query successful, {len(results)} results returned. Top score: {results[0]['similarity_score']:.4f}")

    # 4. Abfrage mit Metadaten-Filter
    print("Integration Test: Querying with metadata filter...")
    # Suche nach einem Vektor, der sicher existiert und den Filterkriterien entspricht
    query_vector_for_filter = vectors_list[10] # Ein anderer Vektor
    metadata_filter = {"content_hash": "hash_10"} # Filter nach dem Hash des 10. Dokuments
    
    response = await async_client.post(
        "/vectors/query",
        json={
            "user_id": user_id,
            "model_id": model_id,
            "query": query_vector_for_filter,
            "k": 1,
            "filter_metadata": metadata_filter
        },
        headers=headers
    )
    assert response.status_code == 200, f"Failed to query with filter: {response.text}"
    filter_query_data = response.json()
    filtered_results = filter_query_data.get("results", [])
    assert len(filtered_results) == 1
    assert filtered_results[0]["metadata"]["doc_id"] == "doc_10"
    assert filtered_results[0]["metadata"]["content_hash"] == "hash_10"
    print("Query with metadata filter successful.")

    # 5. Vektoren löschen (basierend auf Metadaten)
    print("Integration Test: Deleting vectors by metadata...")
    delete_filter = {"doc_id": "doc_0"} # Lösche den ersten Vektor
    response = await async_client.post(
        "/vectors/delete", # Ihre Route aus vectors.py
        json={
            "user_id": user_id,
            "model_id": model_id,
            "filter_metadata": delete_filter
        },
        headers=headers
    )
    assert response.status_code == 200, f"Failed to delete vectors: {response.text}"
    delete_data = response.json()
    assert delete_data["deleted"] == 1
    print("Vector deletion successful.")

    # 5.1. Anzahl nach dem Löschen prüfen
    response = await async_client.get(
        f"/vectors/count?user_id={user_id}&model_id={model_id}",
        headers=headers
    )
    assert response.status_code == 200
    count_after_delete = response.json()
    assert count_after_delete["vectors"] == num_vectors - 1
    print(f"Vector count after deletion verified: {count_after_delete['vectors']}")

    # Aufräumen: Store am Ende des Tests löschen (Admin-Endpunkt)
    print("Integration Test: Cleaning up store...")
    response = await async_client.delete(
        f"/admin/store",
        json={"user_id": user_id, "model_id": model_id},
        headers=headers
    )
    assert response.status_code == 200, f"Failed to delete store during cleanup: {response.text}"
    print(f"Store {user_id}/{model_id} deleted for cleanup.")


# Fixture für den asynchronen HTTP-Client (benötigt pytest-httpx)
# Dies wird in Ihrer conftest.py oder am Anfang der Testdatei platziert.
@pytest.fixture
async def async_client():
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        yield client