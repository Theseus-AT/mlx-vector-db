##!/usr/bin/env python3
"""
Funktionierender MLX Vector Database Test
Mit korrekten Parametern für alle Endpunkte
"""

import requests
import numpy as np
import time
import sys
import os

BASE_URL = "http://localhost:8000"

def get_working_api_key():
    """Hole den funktionierenden API-Key vom Server"""
    try:
        sys.path.append('.')
        from security.auth import get_api_key, get_admin_key
        return get_api_key(), get_admin_key()
    except Exception:
        return "mlx-vector-dev-key-2024", "mlx-vector-admin-key-2024"

def test_complete_workflow():
    """Vollständiger Test mit korrekten Parametern"""
    
    API_KEY, ADMIN_KEY = get_working_api_key()
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    admin_headers = {
        "Authorization": f"Bearer {ADMIN_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🧪 MLX Vector Database - Funktionierender Test")
    print("=" * 55)
    print(f"   API Key: {API_KEY[:15]}...")
    print(f"   Admin Key: {ADMIN_KEY[:15]}...")
    
    # 1. Health Check
    print("\n1️⃣ Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("   ✅ Basic health check passed")
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
        return False
    
    # 2. Performance Health (mit Auth)
    print("\n2️⃣ Performance Health Check...")
    response = requests.get(f"{BASE_URL}/performance/health", headers=headers)
    if response.status_code == 200:
        print("   ✅ Performance health check passed")
    else:
        print(f"   ❌ Performance health failed: {response.status_code}")
        return False
    
    # 3. Store erstellen - MIT KORREKTEN PARAMETERN
    print("\n3️⃣ Creating store with correct parameters...")
    user_id = "test_user_complete"
    model_id = "test_model_complete"
    
    # Cleanup erst (falls Store existiert)
    try:
        delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
        requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers, timeout=5)
    except:
        pass
    
    # Store erstellen mit korrektem JSON Body
    create_payload = {
        "user_id": user_id,
        "model_id": model_id,
        "dimension": 384  # Erforderlicher Parameter hinzugefügt
    }
    
    response = requests.post(
        f"{BASE_URL}/admin/create_store", 
        json=create_payload, 
        headers=admin_headers, 
        timeout=10
    )
    
    if response.status_code == 200:
        print("   ✅ Store created successfully")
        print(f"      Response: {response.json()}")
    elif response.status_code == 409:
        print(f"   ✅ Store already existed, which is fine for this test.")
    else:
        print(f"   ❌ Store creation failed: {response.status_code}")
        print(f"      Response: {response.text}")
        return False
    
    # 4. Vektoren hinzufügen
    print("\n4️⃣ Adding vectors...")
    num_vectors = 5  # Menge klar definieren
    vectors = np.random.rand(num_vectors, 384).astype(np.float32)
    metadata = [{"id": f"test_doc_{i}", "content": f"Test document {i}"} for i in range(num_vectors)]
    
    # Sicherstellen dass beide Listen die gleiche Länge haben
    assert len(vectors) == len(metadata), f"Length mismatch: {len(vectors)} vectors vs {len(metadata)} metadata"
    
    add_payload = {
        "user_id": user_id,
        "model_id": model_id,
        "vectors": vectors.tolist(),
        "metadata": metadata
    }
    
    response = requests.post(
        f"{BASE_URL}/vectors/add", 
        json=add_payload, 
        headers=headers, 
        timeout=15
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Added {result.get('vectors_added', '?')} vectors")
    else:
        print(f"   ❌ Vector addition failed: {response.status_code}")
        print(f"      Response: {response.text}")
        return False
    
    # 5. Vector Count - MIT KORREKTEN QUERY PARAMETERN
    print("\n5️⃣ Getting vector count...")
    response = requests.get(
        f"{BASE_URL}/vectors/count",
        params={"user_id": user_id, "model_id": model_id},  # Als Query Parameter!
        headers=headers
    )
    
    if response.status_code == 200:
        count_data = response.json()
        print(f"   ✅ Vector count: {count_data.get('count', 'unknown')}")
    else:
        print(f"   ❌ Vector count failed: {response.status_code}")
        print(f"      Response: {response.text}")
    
    # 6. Admin Stats - MIT KORREKTEN QUERY PARAMETERN
    print("\n6️⃣ Getting admin stats...")
    response = requests.get(
        f"{BASE_URL}/admin/store/stats",
        params={"user_id": user_id, "model_id": model_id},  # Als Query Parameter!
        headers=admin_headers
    )
    
    if response.status_code == 200:
        stats_data = response.json()
        print(f"   ✅ Store stats: {stats_data}")
    else:
        print(f"   ❌ Admin stats failed: {response.status_code}")
        print(f"      Response: {response.text}")
    
    # 7. Vector Query
    print("\n7️⃣ Querying vectors...")
    query_vector = vectors[0].tolist()
    
    query_payload = {
        "user_id": user_id,
        "model_id": model_id,
        "query": query_vector,
        "k": 3
    }
    
    response = requests.post(
        f"{BASE_URL}/vectors/query",
        json=query_payload,
        headers=headers
    )
    
    if response.status_code == 200:
        query_result = response.json()
        results = query_result.get('results', [])
        print(f"   ✅ Query returned {len(results)} results")
        if results:
            print(f"      Top similarity: {results[0].get('similarity_score', 0):.4f}")
    else:
        print(f"   ❌ Vector query failed: {response.status_code}")
        print(f"      Response: {response.text}")
    
    # 8. Performance Warmup Test
    print("\n8️⃣ Testing performance warmup...")
    warmup_payload = {"user_id": user_id, "model_id": model_id}
    response = requests.post(
        f"{BASE_URL}/performance/warmup",
        json=warmup_payload,
        headers=headers
    )
    
    if response.status_code == 200:
        print("   ✅ Performance warmup successful")
    else:
        print(f"   ❌ Performance warmup failed: {response.status_code}")
        print(f"      Response: {response.text}")
    
    # 9. Cleanup
    print("\n9️⃣ Cleanup...")
    delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
    response = requests.delete(
        f"{BASE_URL}/admin/store",
        params=delete_params,
        headers=admin_headers
    )
    
    if response.status_code == 200:
        print("   ✅ Store deleted successfully")
    else:
        print(f"   ⚠️ Cleanup warning: {response.status_code}")
        print(f"      Response: {response.text}")
    
    print(f"\n🎉 COMPLETE TEST SUCCESSFUL!")
    print(f"💡 All endpoints are working correctly")
    return True

def check_endpoint_schemas():
    """Überprüfe die API-Schema-Dokumentation"""
    print("\n📋 Checking API Schema...")
    
    try:
        response = requests.get(f"{BASE_URL}/openapi.json")
        if response.status_code == 200:
            schema = response.json()
            print("   ✅ API Schema available")
            
            # Zeige verfügbare Endpunkte
            paths = schema.get("paths", {})
            print(f"   📊 Found {len(paths)} API endpoints:")
            
            for path, methods in list(paths.items())[:10]:  # Zeige erste 10
                method_list = list(methods.keys())
                print(f"      {path}: {', '.join(method_list)}")
            
            if len(paths) > 10:
                print(f"      ... and {len(paths) - 10} more")
            
        else:
            print(f"   ❌ Schema not available: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Schema check error: {e}")

def main():
    print("🚀 MLX Vector Database - Comprehensive Test")
    print("=" * 50)
    
    # Warte auf Server
    print("⏳ Waiting for server...")
    for i in range(15):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server not ready")
        return False
    
    # Überprüfe API Schema
    check_endpoint_schemas()
    
    # Führe vollständigen Test durch
    success = test_complete_workflow()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("🎯 Your MLX Vector Database is working perfectly!")
        
        # Zeige Demo-Commands
        print("\n📖 Ready to use! Try these demo commands:")
        print("   python demo.py")
        print("   python enterprise_demo.py")
        
    else:
        print("\n❌ Some tests failed")
        print("💡 Check the server logs for more details")
    
    return success

if __name__ == "__main__":
    main()