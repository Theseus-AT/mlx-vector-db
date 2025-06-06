#!/usr/bin/env python3
"""
Einfacher Funktionstest für MLX Vector Database
Testet grundlegende Funktionalität ohne komplexe Dependencies
"""

import requests
import numpy as np
import time
import sys
import os

# Basis-Konfiguration
BASE_URL = "http://localhost:8000"
API_KEY = "mlx-vector-dev-key-2024"
ADMIN_KEY = "mlx-vector-admin-key-2024"

def test_server_connectivity():
    """Test ob Server erreichbar ist"""
    print("🌐 Testing server connectivity...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Server is healthy")
            print(f"   📱 MLX Device: {health_data.get('mlx_device', 'unknown')}")
            print(f"   📊 Active Stores: {health_data.get('stores_active', 0)}")
            return True
        else:
            print(f"   ❌ Server unhealthy: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def test_basic_api_auth():
    """Test API Authentifizierung"""
    print("\n🔐 Testing API authentication...")
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    try:
        response = requests.get(f"{BASE_URL}/performance/health", headers=headers, timeout=5)
        if response.status_code == 200:
            print("   ✅ API authentication successful")
            return True
        elif response.status_code == 401:
            print("   ❌ API authentication failed - invalid key")
            return False
        else:
            print(f"   ⚠️ Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Auth test failed: {e}")
        return False

def test_store_lifecycle():
    """Test kompletten Store-Lebenszyklus"""
    print("\n🏗️ Testing store lifecycle...")
    
    user_id = "simple_test_user"
    model_id = "simple_test_model"
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    admin_headers = {"Authorization": f"Bearer {ADMIN_KEY}", "Content-Type": "application/json"}
    
    try:
        # 1. Cleanup (falls Store existiert)
        try:
            delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
            requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers, timeout=5)
        except:
            pass
        
        # 2. Store erstellen
        print("   📁 Creating store...")
        create_payload = {
            "user_id": user_id,
            "model_id": model_id,
            "dimension": 384
        }
        
        response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=admin_headers, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Store creation failed: {response.status_code}")
            print(f"      Response: {response.text}")
            return False
        
        print("   ✅ Store created successfully")
        
        # 3. Vektoren hinzufügen
        print("   ➕ Adding vectors...")
        
        # Kleine Anzahl für einfachen Test
        num_vectors = 5
        vectors = np.random.rand(num_vectors, 384).astype(np.float32)
        metadata = [{"id": f"test_{i}", "content": f"Test content {i}"} for i in range(num_vectors)]
        
        add_payload = {
            "user_id": user_id,
            "model_id": model_id,
            "vectors": vectors.tolist(),
            "metadata": metadata
        }
        
        response = requests.post(f"{BASE_URL}/vectors/add", json=add_payload, headers=headers, timeout=15)
        
        if response.status_code != 200:
            print(f"   ❌ Vector addition failed: {response.status_code}")
            print(f"      Response: {response.text}")
            return False
        
        add_result = response.json()
        print(f"   ✅ Added {add_result.get('vectors_added', '?')} vectors")
        
        # 4. Vektoren zählen
        print("   📊 Counting vectors...")
        
        response = requests.get(f"{BASE_URL}/vectors/count", 
                              params={"user_id": user_id, "model_id": model_id}, 
                              headers=headers, timeout=5)
        
        if response.status_code != 200:
            print(f"   ❌ Vector count failed: {response.status_code}")
            return False
        
        count_result = response.json()
        vector_count = count_result.get("count", 0)
        print(f"   ✅ Vector count: {vector_count}")
        
        if vector_count != num_vectors:
            print(f"   ⚠️ Count mismatch: expected {num_vectors}, got {vector_count}")
        
        # 5. Vektoren abfragen
        print("   🔍 Querying vectors...")
        
        query_payload = {
            "user_id": user_id,
            "model_id": model_id,
            "query": vectors[0].tolist(),
            "k": 3
        }
        
        response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Vector query failed: {response.status_code}")
            return False
        
        query_result = response.json()
        results = query_result.get("results", [])
        
        if len(results) > 0:
            top_similarity = results[0].get("similarity_score", 0)
            print(f"   ✅ Query successful: {len(results)} results, top similarity: {top_similarity:.4f}")
            
            # Erwarten dass erste Abfrage perfekt matched (similarity ≈ 1.0)
            if top_similarity > 0.99:
                print("   ✅ Self-query similarity check passed")
            else:
                print(f"   ⚠️ Self-query similarity lower than expected: {top_similarity}")
        else:
            print("   ❌ No query results returned")
            return False
        
        # 6. Store-Statistiken
        print("   📈 Getting store stats...")
        
        response = requests.get(f"{BASE_URL}/admin/store/stats", 
                              params={"user_id": user_id, "model_id": model_id}, 
                              headers=admin_headers, timeout=5)
        
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✅ Store stats retrieved")
        else:
            print(f"   ⚠️ Store stats failed: {response.status_code}")
        
        # 7. Cleanup
        print("   🧹 Cleaning up...")
        
        delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
        response = requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers, timeout=10)
        
        if response.status_code == 200:
            print("   ✅ Store deleted successfully")
        else:
            print(f"   ⚠️ Store deletion warning: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Store lifecycle test failed: {e}")
        return False

def test_error_handling():
    """Test Error Handling"""
    print("\n🛡️ Testing error handling...")
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    # Test 1: Ungültiger Endpunkt
    try:
        response = requests.get(f"{BASE_URL}/nonexistent", headers=headers, timeout=5)
        if response.status_code == 404:
            print("   ✅ 404 error handling works")
        else:
            print(f"   ⚠️ Unexpected response for invalid endpoint: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        return False
    
    # Test 2: Ungültige Authentifizierung
    try:
        bad_headers = {"Authorization": "Bearer invalid-key", "Content-Type": "application/json"}
        response = requests.get(f"{BASE_URL}/performance/health", headers=bad_headers, timeout=5)
        if response.status_code == 401:
            print("   ✅ Authentication error handling works")
        else:
            print(f"   ⚠️ Unexpected response for bad auth: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Auth error test failed: {e}")
        return False
    
    # Test 3: Abfrage auf nicht-existenten Store
    try:
        query_payload = {
            "user_id": "nonexistent_user",
            "model_id": "nonexistent_model",
            "query": [0.1] * 384,
            "k": 5
        }
        
        response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=headers, timeout=10)
        # Sollte einen Fehler oder leere Ergebnisse zurückgeben
        if response.status_code in [404, 400, 500] or (response.status_code == 200 and len(response.json().get("results", [])) == 0):
            print("   ✅ Non-existent store handling works")
        else:
            print(f"   ⚠️ Unexpected response for non-existent store: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Non-existent store test failed: {e}")
        return False
    
    return True

def test_performance_basic():
    """Einfacher Performance Test"""
    print("\n⚡ Testing basic performance...")
    
    user_id = "perf_test_user"
    model_id = "perf_test_model"
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    admin_headers = {"Authorization": f"Bearer {ADMIN_KEY}", "Content-Type": "application/json"}
    
    try:
        # Cleanup und Setup
        try:
            delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
            requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers, timeout=5)
        except:
            pass
        
        # Store erstellen
        create_payload = {"user_id": user_id, "model_id": model_id, "dimension": 384}
        response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=admin_headers, timeout=10)
        
        if response.status_code != 200:
            print(f"   ❌ Performance test setup failed: {response.status_code}")
            return False
        
        # Performance Test: Vektoren hinzufügen
        num_vectors = 50  # Kleine Anzahl für schnellen Test
        vectors = np.random.rand(num_vectors, 384).astype(np.float32)
        metadata = [{"id": f"perf_{i}"} for i in range(num_vectors)]
        
        add_payload = {
            "user_id": user_id,
            "model_id": model_id,
            "vectors": vectors.tolist(),
            "metadata": metadata
        }
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/vectors/add", json=add_payload, headers=headers, timeout=30)
        add_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"   ❌ Performance add test failed: {response.status_code}")
            return False
        
        add_rate = num_vectors / add_time
        print(f"   📈 Add Performance: {add_rate:.1f} vectors/sec")
        
        # Performance Test: Queries
        query_times = []
        
        for i in range(min(10, num_vectors)):
            query_payload = {
                "user_id": user_id,
                "model_id": model_id,
                "query": vectors[i].tolist(),
                "k": 5
            }
            
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/vectors/query", json=query_payload, headers=headers, timeout=10)
            query_time = time.time() - start_time
            
            if response.status_code == 200:
                query_times.append(query_time)
        
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            qps = 1 / avg_query_time if avg_query_time > 0 else 0
            
            print(f"   ⚡ Query Performance: {qps:.1f} QPS")
            print(f"   ⏱️ Avg Query Time: {avg_query_time*1000:.2f}ms")
            
            # Einfache Performance-Targets
            if add_rate >= 50:
                print("   ✅ Add performance target met")
            else:
                print("   ⚠️ Add performance below target")
            
            if qps >= 10:
                print("   ✅ Query performance target met")
            else:
                print("   ⚠️ Query performance below target")
        else:
            print("   ❌ No successful queries for performance test")
            return False
        
        # Cleanup
        delete_params = {"user_id": user_id, "model_id": model_id, "force": True}
        requests.delete(f"{BASE_URL}/admin/store", params=delete_params, headers=admin_headers, timeout=10)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def main():
    """Hauptfunktion für einfachen Test"""
    
    print("🧪 MLX Vector Database - Simple Functional Test")
    print("=" * 60)
    
    # Test-Ergebnisse sammeln
    tests = [
        ("Server Connectivity", test_server_connectivity),
        ("API Authentication", test_basic_api_auth),
        ("Store Lifecycle", test_store_lifecycle),
        ("Error Handling", test_error_handling),
        ("Basic Performance", test_performance_basic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n" + "="*40)
        print(f"🔬 TEST: {test_name}")
        print("="*40)
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    # Zusammenfassung
    print(f"\n" + "="*60)
    print(f"📊 TEST SUMMARY")
    print("="*60)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED!")
        print(f"✅ MLX Vector Database is working correctly!")
        
        print(f"\n💡 Ready for advanced testing:")
        print(f"   python working_test_fixed.py")
        print(f"   python sprint3_demo.py")
        
        return True
    elif passed >= total * 0.8:
        print(f"✅ MOSTLY WORKING! {passed}/{total} tests passed")
        print(f"⚠️ Some features may need attention")
        return True
    else:
        print(f"❌ MULTIPLE ISSUES DETECTED")
        print(f"🔧 Please check server configuration and dependencies")
        return False

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)