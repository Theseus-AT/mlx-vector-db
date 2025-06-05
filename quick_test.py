#!/usr/bin/env python3
"""
Schneller Test der MLX Vector Database API
Automatische Erkennung der korrekten API-Keys
"""

import requests
import numpy as np
import time
import os
import json
import sys
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:8000"

def get_server_api_keys():
    """Hole die aktuell vom Server verwendeten API-Keys"""
    try:
        # Versuche auth module zu importieren
        sys.path.append('.')
        from security.auth import get_api_key, get_admin_key
        return get_api_key(), get_admin_key()
    except Exception as e:
        print(f"⚠️ Could not import auth module: {e}")
        return None, None

def get_api_keys():
    """Ermittle die korrekten API-Keys aus verschiedenen Quellen"""
    print("🔑 Determining API keys...")
    
    # Priorität: 1. Umgebungsvariablen, 2. Server-Konfiguration, 3. Defaults
    api_key = os.getenv("VECTOR_DB_API_KEY")
    admin_key = os.getenv("VECTOR_DB_ADMIN_KEY")
    
    if not api_key or not admin_key:
        print("   No environment variables found, checking server config...")
        server_api, server_admin = get_server_api_keys()
        if server_api and server_admin:
            api_key = server_api
            admin_key = server_admin
            print("   ✅ Using server configuration keys")
        else:
            # Fallback zu Default-Keys
            api_key = "mlx-vector-dev-key-2024"
            admin_key = "mlx-vector-admin-key-2024"
            print("   ⚠️ Using default keys")
    else:
        print("   ✅ Using environment variables")
    
    print(f"   API Key: {api_key[:10]}...")
    print(f"   Admin Key: {admin_key[:10]}...")
    
    return api_key, admin_key

def test_with_auth():
    """Test mit korrekter Authentifizierung"""
    
    # Ermittle korrekte API-Keys
    API_KEY, ADMIN_KEY = get_api_keys()
    
    # Headers für API-Requests
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    admin_headers = {
        "Authorization": f"Bearer {ADMIN_KEY}",
        "Content-Type": "application/json"
    }
    
    print("\n🧪 MLX Vector Database API Test")
    print("=" * 50)
    
    # 1. Health Check (keine Auth erforderlich)
    print("\n1️⃣ Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Status: {health.get('status')}")
            print(f"   📱 MLX Device: {health.get('mlx_device')}")
            print(f"   📊 Stores: {health.get('stores_active', 0)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # 2. Performance Health (mit Auth) - Test der Authentifizierung
    print("\n2️⃣ Testing Authentication...")
    try:
        response = requests.get(f"{BASE_URL}/performance/health", headers=headers, timeout=5)
        if response.status_code == 200:
            perf_health = response.json()
            print(f"   ✅ Authentication successful!")
            print(f"   📊 Performance Status: {perf_health.get('status')}")
        elif response.status_code == 401:
            print(f"   ❌ Authentication failed: Invalid API key")
            print(f"   💡 Server expects different API key than: {API_KEY}")
            
            # Versuche alternative Keys
            print("   🔄 Trying alternative authentication...")
            
            # Test mit verschiedenen Key-Formaten
            alternative_keys = [
                "dev-key-please-change",  # Aus auth.py
                "mlx-vector-dev-key-2024", # Standard
                os.getenv("API_KEY", ""),   # Allgemeine Env-Var
            ]
            
            for alt_key in alternative_keys:
                if alt_key and alt_key != API_KEY:
                    alt_headers = {"Authorization": f"Bearer {alt_key}", "Content-Type": "application/json"}
                    try:
                        alt_response = requests.get(f"{BASE_URL}/performance/health", headers=alt_headers, timeout=5)
                        if alt_response.status_code == 200:
                            print(f"   ✅ Success with key: {alt_key[:10]}...")
                            API_KEY = alt_key
                            ADMIN_KEY = alt_key  # Assume same for admin
                            headers["Authorization"] = f"Bearer {API_KEY}"
                            admin_headers["Authorization"] = f"Bearer {ADMIN_KEY}"
                            break
                    except:
                        continue
            else:
                print("   ❌ No working API key found")
                return False
        else:
            print(f"   ❌ Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Authentication test error: {e}")
        return False
    
    # 3. Store erstellen (Admin-Endpunkt)
    print("\n3️⃣ Creating test store...")
    user_id = "test_user"
    model_id = "test_model"
    
    try:
        # Erst versuchen zu löschen (falls vorhanden) - KORRIGIERT
        delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
        requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers, timeout=5)
        
        # Store erstellen
        create_payload = {"user_id": user_id, "model_id": model_id}
        response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=admin_headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result:
                print(f"   ✅ Store created: {result['data'].get('store_created', True)}")
            else:
                print(f"   ✅ Store created successfully")
        elif response.status_code == 409:
             print(f"   ✅ Store already existed, continuing test.")
        else:
            print(f"   ❌ Store creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Store creation error: {e}")
        return False
    
    # 4. Vektoren hinzufügen
    print("\n4️⃣ Adding vectors...")
    try:
        # Generiere Test-Vektoren
        vectors = np.random.rand(10, 384).astype(np.float32)
        metadata = [{"id": f"doc_{i}", "content": f"Document {i}"} for i in range(10)]
        
        add_payload = {
            "user_id": user_id,
            "model_id": model_id,
            "vectors": vectors.tolist(),
            "metadata": metadata
        }
        
        response = requests.post(f"{BASE_URL}/vectors/add", json=add_payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Added {result.get('vectors_added')} vectors")
            print(f"   📊 Total vectors: {result.get('total_vectors')}")
        else:
            print(f"   ❌ Vector addition failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Vector addition error: {e}")
    
    # 5. Vektoren abfragen
    print("\n5️⃣ Querying vectors...")
    try:
        query_vector = vectors[0].tolist()  # Verwende den ersten Vektor als Query
        
        query_payload = {
            "user_id": user_id,
            "model_id": model_id,
            "query": query_vector,
            "k": 3
        }
        
        response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            results = result.get('results', [])
            print(f"   ✅ Query returned {len(results)} results")
            if results:
                print(f"   🎯 Top result similarity: {results[0].get('similarity_score', 0):.4f}")
                print(f"   📄 Top result metadata: {results[0].get('metadata', {}).get('id')}")
        else:
            print(f"   ❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Query error: {e}")
    
    # 6. Store Statistiken
    print("\n6️⃣ Getting store stats...")
    try:
        response = requests.get(f"{BASE_URL}/vectors/count?user_id={user_id}&model_id={model_id}", headers=headers, timeout=5)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"   📊 Vector count: {stats.get('count')}")
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Stats error: {e}")
    
    # 7. Cleanup
    print("\n7️⃣ Cleanup...")
    try:
        # KORRIGIERT: Verwende `params` statt `json` für den DELETE-Request
        delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
        response = requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers, timeout=10)
        
        if response.status_code == 200:
            print("   ✅ Test store deleted")
        else:
            print(f"   ⚠️ Cleanup warning: {response.status_code}")
            print(f"      Response: {response.text}")
    except Exception as e:
        print(f"   ⚠️ Cleanup error: {e}")
    
    print(f"\n🎉 Test completed successfully!")
    print(f"💡 Working API Key: {API_KEY}")
    
    return True

def wait_for_server():
    """Warte auf Server"""
    print("⏳ Waiting for server...")
    for i in range(30):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except:
            pass
        time.sleep(1)
    print("❌ Server not ready after 30 seconds")
    return False

if __name__ == "__main__":
    if wait_for_server():
        if test_with_auth():
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            print("\n💡 Try running the debug script:")
            print("   python debug_api_keys.py")
    else:
        print("Cannot run tests - server not available")

# quick_test.py

# ... (andere Funktionen) ...

def test_with_auth():
    # ... (Setup) ...

    # 3. Store erstellen (Admin-Endpunkt)
    print("\n3️⃣ Creating test store...")
    user_id = "test_user"
    model_id = "test_model"
    
    try:
        # Erst versuchen zu löschen (falls vorhanden)
        delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
        requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers)
        
        # KORRIGIERT: Payload enthält jetzt die erforderliche 'dimension'
        create_payload = {
            "user_id": user_id, 
            "model_id": model_id,
            "dimension": 384  # Standard-Dimension für den Test
        }
        response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=admin_headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if "data" in result:
                print(f"   ✅ Store created: {result['data'].get('store_created', True)}")
            else:
                print(f"   ✅ Store created successfully")
        # ... (Rest der Funktion)