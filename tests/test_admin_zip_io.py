import os
import tempfile
import zipfile
import numpy as np
import requests
import time

BASE_URL = "http://localhost:8000"

def wait_for_server():
    """Wait for the FastAPI server to be ready."""
    print("Waiting for server...")
    start_time = time.time()
    while True:
        if time.time() - start_time > 30: # Timeout nach 30s
             raise TimeoutError("Server did not start within 30 seconds.")
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                print("Server is up!")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

wait_for_server()

def test_export_import_cycle():
    user_id = "ziptest_user"
    model_id = "ziptest_model"
    admin_base = f"{BASE_URL}/admin" # Basis für Admin-Routen
    admin_store_url = f"{admin_base}/store"
    admin_create_url = f"{admin_base}/create_store"
    admin_add_vec_url = f"{admin_base}/add_test_vectors"
    admin_export_url = f"{admin_base}/export_zip"
    admin_import_url = f"{admin_base}/import_zip"
    # === GEÄNDERT: Spezifischen Stats-Endpunkt verwenden ===
    admin_specific_stats_url = f"{admin_base}/store/stats"
    # =====================================================

    # === Cleanup (wie zuletzt, sollte funktionieren) ===
    print(f"\n[CLEANUP] Attempting to delete store {user_id}/{model_id} via API if it exists...")
    delete_payload = {"user_id": user_id, "model_id": model_id}
    try:
        r_delete = requests.delete(admin_store_url, json=delete_payload, timeout=10)
        if r_delete.status_code not in [200, 404]:
             raise Exception(f"Pre-test cleanup failed! Status: {r_delete.status_code}, Response: {r_delete.text}")
        else:
            print(f"[CLEANUP OK] Store deleted or did not exist (Status: {r_delete.status_code})")
            time.sleep(0.2)
    except requests.exceptions.RequestException as e:
        raise Exception(f"Pre-test cleanup failed due to connection error: {e}")
    # === ENDE Cleanup ===

    # 1. Store anlegen
    print(f"[TEST] Creating store {user_id}/{model_id}...")
    try:
        r_create = requests.post(admin_create_url, json={"user_id": user_id, "model_id": model_id}, timeout=10)
        if r_create.status_code != 200: raise Exception(f"Failed to create store. Status: {r_create.status_code}, Response: {r_create.text}")
        print(f"[TEST] Store created successfully.")
    except requests.exceptions.RequestException as e: raise Exception(f"Failed to create store due to connection error: {e}")

    # 2. Vektoren hinzufügen
    print(f"[TEST] Adding vectors to {user_id}/{model_id}...")
    vecs = np.random.rand(3, 128).astype(np.float32)
    meta = [{"id": f"v{i}", "tag": "zip"} for i in range(3)]
    add_payload = {"user_id": user_id, "model_id": model_id, "vectors": vecs.tolist(), "metadata": meta }
    try:
        r_add = requests.post(admin_add_vec_url, json=add_payload, timeout=10)
        if r_add.status_code != 200: raise Exception(f"Failed to add vectors. Status: {r_add.status_code}, Response: {r_add.text}")
        print(f"[TEST] Vectors added successfully.")
    except requests.exceptions.RequestException as e: raise Exception(f"Failed to add vectors due to connection error: {e}")

    # 3. Exportieren
    print(f"[TEST] Exporting store {user_id}/{model_id}...")
    tmp_zip_path = None
    try:
        export_params={"user_id": user_id, "model_id": model_id}
        r_export = requests.get(admin_export_url, params=export_params, timeout=10)
        if r_export.status_code != 200: raise Exception(f"Failed to export zip. Status: {r_export.status_code}, Response: {r_export.text}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            tmp_zip_path = tmp_zip.name
            tmp_zip.write(r_export.content)
        print(f"[TEST] Store exported successfully to {tmp_zip_path}")
        try:
            with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
                print(f"[TEST] Contents of exported zip: {zip_ref.namelist()}")
                assert "vectors.npz" in zip_ref.namelist(); assert "metadata.jsonl" in zip_ref.namelist()
        except zipfile.BadZipFile: raise Exception("Exported file is not a valid ZIP file.")

        # 4. Löschen des Stores für den Import-Test
        print(f"[TEST] Deleting store {user_id}/{model_id} before import...")
        r_delete_during = requests.delete(admin_store_url, json=delete_payload, timeout=10)
        if r_delete_during.status_code != 200: raise Exception(f"Failed to delete store during test! Status: {r_delete_during.status_code}, Response: {r_delete_during.text}")
        print(f"[TEST] Store deleted successfully before import.")
        time.sleep(0.2)

        # 5. Importieren
        print(f"[TEST] Importing store {user_id}/{model_id} from {tmp_zip_path}...")
        with open(tmp_zip_path, "rb") as f:
            files_data = {"file": ("import.zip", f, "application/zip")}
            import_params = {"user_id": user_id, "model_id": model_id, "overwrite": "false"} # overwrite als String boolean senden?
            r_import = requests.post(admin_import_url, params=import_params, files=files_data, timeout=30)
            if r_import.status_code != 200: raise Exception(f"Failed to import zip. Status: {r_import.status_code}, Response: {r_import.text}")
            import_result = r_import.json(); print(f"[TEST] Import successful: {import_result}"); assert import_result.get("count") == 3

        # === GEÄNDERT: Spezifischen Stats-Endpunkt verwenden ===
        # 6. Verifizieren nach Import
        print(f"[TEST] Verifying store {user_id}/{model_id} stats after import...")
        stats_params = {"user_id": user_id, "model_id": model_id}
        r_stats = requests.get(admin_specific_stats_url, params=stats_params, timeout=10) # Nutze den neuen URL

        # Jetzt sollten wir direkt das erwartete Dictionary bekommen
        if r_stats.status_code != 200:
            raise Exception(f"Failed to get specific stats after import! Status: {r_stats.status_code}, Response: {r_stats.text}")

        stats = r_stats.json() # Das sollte jetzt das Dictionary sein
        print(f"[TEST] Stats after import: {stats}")
        # Die Asserts sollten jetzt funktionieren
        assert isinstance(stats, dict), f"Expected stats to be a dict, but got {type(stats)}" # Zusätzlicher Check
        assert stats.get("vectors") == 3
        assert stats.get("metadata") == 3
        print("[TEST] Verification successful!")
        # === ENDE ÄNDERUNG ===

    finally:
        # Cleanup für tmp_zip_path
        if tmp_zip_path and os.path.exists(tmp_zip_path):
            try: os.remove(tmp_zip_path); print(f"[CLEANUP] Temporary zip file deleted: {tmp_zip_path}")
            except OSError as e: print(f"[CLEANUP ERROR] Failed to delete temporary zip file {tmp_zip_path}: {e}")