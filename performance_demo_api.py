# performance_demo_api.py
# (Basiert auf Ihrer Datei theseus-at/mlx-vector-db/mlx-vector-db-12319040886f6c23f84935346248ef50cf06157f/performance_demo.py)
"""
Performance Demo für MLX Vector Database via API-Endpunkte.
Zeigt die Performance-Verbesserungen durch MLX, Caching und optimierte Operationen über die API.
"""
import requests
import time
import numpy as np
import json
import os
import logging # Logging hinzugefügt
# Am Anfang von performance_demo_api.py hinzufügen:
from typing import Optional
logger = logging.getLogger("mlx_vector_db.perf_demo_api")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


BASE_URL = os.getenv("VECTOR_DB_BASE_URL", "http://localhost:8000")
# API_KEY sollte idealerweise als Parameter übergeben oder sicher gehandhabt werden.
# Für dieses Demo-Skript holen wir es aus der Umgebung oder fragen danach.
API_KEY_ENV_VAR = "VECTOR_DB_API_KEY"


def wait_for_server(base_url_to_check: str, timeout_seconds: int = 30):
    """Wartet, bis der Server unter der angegebenen URL erreichbar ist."""
    logger.info(f"🔍 Überprüfe Server-Verfügbarkeit unter {base_url_to_check}...")
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            logger.error(f"❌ Server nicht erreichbar nach {timeout_seconds} Sekunden.")
            raise TimeoutError(f"Server {base_url_to_check} nicht erreichbar nach {timeout_seconds}s")
        try:
            # Verwende einen einfachen Health-Endpunkt, falls vorhanden
            response = requests.get(f"{base_url_to_check}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ Server ist bereit!")
                return
        except requests.exceptions.RequestException:
            pass # Einfach weiter versuchen
        time.sleep(1)

def get_api_key_for_demo() -> Optional[str]:
    """Holt den API-Key für die Demo."""
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        logger.warning(f"{API_KEY_ENV_VAR} nicht in Umgebungsvariablen gefunden.")
        # In einer automatisierten Umgebung sollte hier kein input() sein.
        # Für manuelle Demo-Ausführung ist es ok.
        # api_key = input("API Key eingeben (aus .env oder Konfiguration): ").strip()
        # if not api_key:
        #     logger.warning("Kein API-Key angegeben. Versuche Demos ohne Authentifizierung (kann fehlschlagen).")
        #     return None
        logger.warning("Kein API-Key in Umgebungsvariablen. Authentifizierte Endpunkte könnten fehlschlagen.")
        return None # Besser None zurückgeben, wenn nicht gesetzt
    return api_key

def create_test_store_api(user_id: str, model_id: str, api_key: Optional[str]):
    """Erstellt einen Test-Store über die API."""
    headers = {"Content-Type": "application/json"}
    if api_key: headers["X-API-Key"] = api_key
    
    admin_url_base = f"{BASE_URL}/admin" # Annahme: Admin Routen haben /admin Präfix

    # Store löschen, falls vorhanden (für sauberen Testlauf)
    try:
        delete_payload = {"user_id": user_id, "model_id": model_id}
        # Die Admin-Route zum Löschen ist /admin/store (DELETE)
        requests.delete(f"{admin_url_base}/store", json=delete_payload, headers=headers, timeout=10)
        logger.info(f"Vorheriger Store {user_id}/{model_id} (falls existent) gelöscht.")
    except Exception:
        pass # Ignoriere Fehler, wenn Store nicht existiert
    
    # Store erstellen
    create_payload = {"user_id": user_id, "model_id": model_id}
    response = requests.post(f"{admin_url_base}/create_store", json=create_payload, headers=headers, timeout=10)
    response.raise_for_status() # Wirft Fehler bei nicht-2xx Status
    
    logger.info(f"✅ Test-Store über API erstellt: {user_id}/{model_id}")


def run_performance_demo(): # Dies ist die Hauptfunktion für diese Demo
    """Führt die API-Performance-Demonstration durch."""
    logger.info("🚀 MLX Vector Database API Performance Demo")
    logger.info(f"🔥 Ziel-Server: {BASE_URL}")
    logger.info("=" * 60)
    
    try:
        wait_for_server(BASE_URL)
    except TimeoutError:
        logger.error("Demo abgebrochen, da Server nicht erreichbar ist.")
        return

    api_key = get_api_key_for_demo()
    headers = {"Content-Type": "application/json"}
    if api_key: headers["X-API-Key"] = api_key
    
    # Performance-Endpunkte (Annahme: /performance Präfix)
    perf_url_base = f"{BASE_URL}/performance"

    logger.info("\n1️⃣ Performance Health Check (API)")
    try:
        response = requests.get(f"{perf_url_base}/health", headers=headers, timeout=5)
        response.raise_for_status()
        health = response.json()
        logger.info(f"   Status: {health.get('status')}")
        logger.info(f"   MLX Version: {health.get('mlx_version', 'unbekannt')}")
        logger.info(f"   Cache Status: {health.get('cache_status', 'unbekannt')}")
    except Exception as e:
        logger.error(f"   ❌ Health Check Fehler: {e}")
        return # Demo hier abbrechen, wenn Basis-Checks fehlschlagen

    logger.info("\n2️⃣ Aufwärmen der MLX-Funktionen über API...")
    try:
        # Der Warmup-Endpunkt erwartet `dimension` als Query-Parameter
        response = requests.post(f"{perf_url_base}/warmup?dimension=384", headers=headers, timeout=60) # Längerer Timeout für Warmup
        response.raise_for_status()
        warmup_data = response.json()
        logger.info(f"   ✅ Aufwärmen abgeschlossen in {warmup_data.get('warmup_time_seconds', 0):.3f}s")
    except Exception as e:
        logger.warning(f"   ⚠️ Fehler beim Aufwärmen: {e}")

    user_id_perf = "api_perf_user"
    model_id_perf = "api_perf_model"
    vector_dim_perf = 384

    logger.info("\n3️⃣ Testumgebung über API einrichten...")
    try:
        create_test_store_api(user_id_perf, model_id_perf, api_key)
    except Exception as e:
        logger.error(f"   ❌ Fehler beim Erstellen des Test-Stores über API: {e}")
        return

    logger.info("\n4️⃣ Test-Vektoren über API hinzufügen...")
    # Ihre Admin-Route /admin/add_test_vectors wird hier verwendet
    vector_count_add = 5000
    try:
        vectors_np = np.random.rand(vector_count_add, vector_dim_perf).astype(np.float32)
        metadata_add = [{"id": f"api_vec_{i}"} for i in range(vector_count_add)]
        add_payload = {
            "user_id": user_id_perf, "model_id": model_id_perf,
            "vectors": vectors_np.tolist(), "metadata": metadata_add
        }
        start_add_api = time.time()
        # Endpunkt /admin/add_test_vectors
        response = requests.post(f"{BASE_URL}/admin/add_test_vectors", json=add_payload, headers=headers, timeout=60)
        response.raise_for_status()
        add_time_api = time.time() - start_add_api
        logger.info(f"   ✅ {vector_count_add} Vektoren hinzugefügt in {add_time_api:.3f}s "
                    f"({vector_count_add/add_time_api:.1f} Vektoren/s über API)")
    except Exception as e:
        logger.error(f"   ❌ Fehler beim Hinzufügen der Vektoren über API: {e}")
        return

    logger.info("\n5️⃣ API Performance Benchmark ausführen...")
    # Endpunkt /performance/benchmark
    benchmark_payload = {
        "user_id": user_id_perf, "model_id": model_id_perf,
        "test_size": 1000, "query_count": 100, "vector_dim": vector_dim_perf
    }
    try:
        response = requests.post(f"{perf_url_base}/benchmark", json=benchmark_payload, headers=headers, timeout=120) # Längerer Timeout
        response.raise_for_status()
        results = response.json()
        logger.info("\n📊 API BENCHMARK ERGEBNISSE:")
        logger.info(json.dumps(results, indent=2)) # Gibt die vollen Ergebnisse aus
        
        # Wichtige Metriken hervorheben
        summary = results.get("summary", {})
        improvement = summary.get("performance_improvement", {})
        logger.info(f"   🚀 Gesamt-Speedup (Query): {improvement.get('query_speedup', 'N/A'):.1f}x")
        logger.info(f"   📈 Geschätzte Kapazität: {improvement.get('estimated_capacity', 'N/A')}")

    except Exception as e:
        logger.error(f"   ❌ API Benchmark Fehler: {e}")

    # Optional: Store Optimierung über API testen, falls implementiert und gewünscht
    # logger.info("\n6️⃣ Store Optimierung über API...")
    # try:
    #     response = requests.post(f"{perf_url_base}/optimize?user_id={user_id_perf}&model_id={model_id_perf}", headers=headers, timeout=60)
    # ...

    logger.info("\n🧹 Aufräumen des Test-Stores über API...")
    try:
        delete_payload = {"user_id": user_id_perf, "model_id": model_id_perf}
        requests.delete(f"{BASE_URL}/admin/store", json=delete_payload, headers=headers, timeout=10)
        logger.info(f"   ✅ Test-Store {user_id_perf}/{model_id_perf} gelöscht.")
    except Exception as e:
        logger.warning(f"   ⚠️ Fehler beim Aufräumen des Test-Stores: {e}")

    logger.info(f"\n🎉 API Performance Demo abgeschlossen!")

if __name__ == "__main__":
    run_performance_demo()