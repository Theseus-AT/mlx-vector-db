#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

# Neue Datei: tests/test_integration.py
# Testet den gesamten Workflow über API-Endpunkte.
#
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
# Verwenden wir die Keys aus dem auth Modul für Konsistenz
from security.auth import get_api_key, get_admin_key
TEST_API_KEY = get_api_key()
TEST_ADMIN_KEY = get_admin_key()

@pytest.mark.asyncio # Markiert Test als asynchron
async def test_full_workflow_integration(async_client: httpx.AsyncClient): # async_client Fixture von pytest-httpx
    """
    Testet den kompletten Workflow: Store erstellen, Vektoren hinzufügen, Abfragen.
    Verwendet pytest-httpx für einen asynchronen Client gegen die ASGI-App.
    """
    user_id = "integration_test_user"
    model_id = "integration_test_model"
    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    admin_headers = {"Authorization": f"Bearer {TEST_ADMIN_KEY}"}

    # Vorbereitung: Sicherstellen, dass der Store nicht existiert
    try:
        # KORRIGIERT: Verwende `params` für DELETE
        delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
        await async_client.delete("/admin/store", params=delete_params, headers=admin_headers)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404: # OK, wenn Store nicht gefunden wurde
            raise

    # 1. Store erstellen
    print("Integration Test: Creating store...")
    response = await async_client.post(
        "/admin/create_store",
        json={"user_id": user_id, "model_id": model_id},
        headers=admin_headers # Admin-Key für Store-Erstellung
    )
    assert response.status_code == 200, f"Failed to create store: {response.text}"
    create_data = response.json()
    assert create_data["data"]["store_created"] is True
    assert create_data["data"]["user_id"] == user_id
    print(f"Store {user_id}/{model_id} created.")

    # 2. Vektoren hinzufügen
    print("Integration Test: Adding vectors...")
    num_vectors = 100
    dimension = 384
    np_vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
    vectors_list = np_vectors.tolist()
    metadata = [{"id": f"doc_{i}", "content_hash": f"hash_{i}"} for i in range(num_vectors)]
    
    response = await async_client.post(
        "/vectors/add",
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
    assert add_data["vectors_added"] == num_vectors
    print(f"{num_vectors} vectors added.")

    # 2.1. Anzahl der Vektoren prüfen
    print("Integration Test: Counting vectors...")
    response = await async_client.get(
        f"/vectors/count?user_id={user_id}&model_id={model_id}",
        headers=headers
    )
    assert response.status_code == 200, f"Failed to count vectors: {response.text}"
    count_data = response.json()
    assert count_data["count"] == num_vectors
    print(f"Vector count verified: {count_data['count']}")

    # 3. Vektoren abfragen
    print("Integration Test: Querying vectors...")
    query_vector_list = vectors_list[0]
    k_neighbors = 5
    
    response = await async_client.post(
        "/vectors/query",
        json={
            "user_id": user_id,
            "model_id": model_id,
            "query": query_vector_list,
            "k": k_neighbors,
            "filter_metadata": None # explizit
        },
        headers=headers
    )
    assert response.status_code == 200, f"Failed to query vectors: {response.text}"
    query_response_data = response.json()
    
    results = query_response_data.get("results", [])
    assert len(results) == k_neighbors
    assert results[0]["metadata"]["id"] == "doc_0"
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > 0.999
    print(f"Query successful, {len(results)} results returned. Top score: {results[0]['similarity_score']:.4f}")

    # 4. Abfrage mit Metadaten-Filter
    print("Integration Test: Querying with metadata filter...")
    query_vector_for_filter = vectors_list[10]
    metadata_filter = {"content_hash": "hash_10"}
    
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
    assert filtered_results[0]["metadata"]["id"] == "doc_10"
    assert filtered_results[0]["metadata"]["content_hash"] == "hash_10"
    print("Query with metadata filter successful.")

    # Aufräumen: Store am Ende des Tests löschen
    print("Integration Test: Cleaning up store...")
    # KORRIGIERT: Verwende `params` statt `json`
    delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
    response = await async_client.delete(
        "/admin/store",
        params=delete_params,
        headers=admin_headers
    )
    assert response.status_code == 200, f"Failed to delete store during cleanup: {response.text}"
    print(f"Store {user_id}/{model_id} deleted for cleanup.")


@pytest.fixture
async def async_client():
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        yield client