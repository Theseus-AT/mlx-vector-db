#!/usr/bin/env python3
"""
Spezifischer Test für Admin-Authentifizierung
Überprüft die verschiedenen Auth-Methoden für Admin-Endpunkte
"""

import requests
import time
import sys
import os

BASE_URL = "http://localhost:8000"

def get_all_possible_keys():
    """Sammle alle möglichen API-Keys aus verschiedenen Quellen"""
    keys = []
    
    # 1. Umgebungsvariablen
    env_api = os.getenv("VECTOR_DB_API_KEY")
    env_admin = os.getenv("VECTOR_DB_ADMIN_KEY")
    if env_api: keys.append(("ENV_API", env_api))
    if env_admin: keys.append(("ENV_ADMIN", env_admin))
    
    # 2. Server-Konfiguration
    try:
        sys.path.append('.')
        from security.auth import get_api_key, get_admin_key
        server_api = get_api_key()
        server_admin = get_admin_key()
        keys.append(("SERVER_API", server_api))
        keys.append(("SERVER_ADMIN", server_admin))
    except Exception as e:
        print(f"⚠️ Could not get server keys: {e}")
    
    # 3. Default-Keys (aus auth.py)
    default_keys = [
        ("DEFAULT_API", "mlx-vector-dev-key-2024"),
        ("DEFAULT_ADMIN", "mlx-vector-admin-key-2024"),
        ("FALLBACK_API", "dev-key-please-change"),
        ("FALLBACK_ADMIN", "dev-admin-key-please-change"),
    ]
    keys.extend(default_keys)
    
    # Duplikate entfernen aber Quelle behalten
    unique_keys = []
    seen_values = set()
    for source, key in keys:
        if key not in seen_values:
            unique_keys.append((source, key))
            seen_values.add(key)
    
    return unique_keys

def test_endpoint_with_key(endpoint, method="GET", json_data=None, key_name="", key_value=""):
    """Teste einen Endpunkt mit einem spezifischen Key"""
    headers = {
        "Authorization": f"Bearer {key_value}",
        "Content-Type": "application/json"
    }
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", headers=headers, json=json_data, timeout=5)
        elif method == "DELETE":
            response = requests.delete(f"{BASE_URL}{endpoint}", headers=headers, json=json_data, timeout=5)
        
        return response.status_code, response.text[:100]
    except Exception as e:
        return None, str(e)

def test_all_auth_combinations():
    """Teste alle Auth-Kombinationen für verschiedene Endpunkte"""
    print("🔐 Testing all authentication combinations...")
    print("=" * 60)
    
    # Alle verfügbaren Keys
    all_keys = get_all_possible_keys()
    
    print(f"📋 Found {len(all_keys)} unique keys:")
    for source, key in all_keys:
        print(f"   {source}: {key[:15]}...")
    
    # Test-Endpunkte
    test_endpoints = [
        ("Performance (API)", "/performance/health", "GET", None),
        ("Admin Create", "/admin/create_store", "POST", {"user_id": "test", "model_id": "test"}),
        ("Admin Stats", "/admin/store/stats", "GET", None),
        ("Vectors Count", "/vectors/count", "GET", None),
    ]
    
    print(f"\n🧪 Testing {len(test_endpoints)} endpoints with {len(all_keys)} keys...")
    print("=" * 60)
    
    results = {}
    
    for endpoint_name, endpoint, method, data in test_endpoints:
        print(f"\n📍 Testing {endpoint_name} ({method} {endpoint})")
        print("-" * 40)
        
        endpoint_results = []
        
        for key_source, key_value in all_keys:
            status, response = test_endpoint_with_key(endpoint, method, data, key_source, key_value)
            
            if status == 200:
                print(f"   ✅ {key_source}: SUCCESS (200)")
                endpoint_results.append((key_source, key_value, "SUCCESS"))
            elif status == 401:
                print(f"   ❌ {key_source}: UNAUTHORIZED (401)")
                endpoint_results.append((key_source, key_value, "UNAUTHORIZED"))
            elif status == 403:
                print(f"   🚫 {key_source}: FORBIDDEN (403)")
                endpoint_results.append((key_source, key_value, "FORBIDDEN"))
            elif status is None:
                print(f"   ⚠️ {key_source}: ERROR ({response})")
                endpoint_results.append((key_source, key_value, "ERROR"))
            else:
                print(f"   ❓ {key_source}: OTHER ({status})")
                endpoint_results.append((key_source, key_value, f"HTTP_{status}"))
        
        results[endpoint_name] = endpoint_results
    
    return results

def analyze_results(results):
    """Analysiere die Test-Ergebnisse"""
    print(f"\n📊 ANALYSIS")
    print("=" * 30)
    
    # Finde funktionierende Keys für jeden Endpunkt
    working_combinations = {}
    
    for endpoint, endpoint_results in results.items():
        working_keys = [
            (key_source, key_value) 
            for key_source, key_value, status in endpoint_results 
            if status == "SUCCESS"
        ]
        working_combinations[endpoint] = working_keys
        
        if working_keys:
            print(f"\n✅ {endpoint}:")
            for key_source, key_value in working_keys:
                print(f"   Works with: {key_source} ({key_value[:15]}...)")
        else:
            print(f"\n❌ {endpoint}: No working keys found")
    
    # Globale Empfehlung
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 20)
    
    # Finde Keys, die für die meisten Endpunkte funktionieren
    key_success_count = {}
    for endpoint, endpoint_results in results.items():
        for key_source, key_value, status in endpoint_results:
            if status == "SUCCESS":
                if key_value not in key_success_count:
                    key_success_count[key_value] = []
                key_success_count[key_value].append(endpoint)
    
    if key_success_count:
        best_key = max(key_success_count.items(), key=lambda x: len(x[1]))
        print(f"🏆 Best overall key: {best_key[0][:20]}...")
        print(f"   Works for: {', '.join(best_key[1])}")
        
        print(f"\n🔧 Quick fix:")
        print(f"   export VECTOR_DB_API_KEY='{best_key[0]}'")
        print(f"   export VECTOR_DB_ADMIN_KEY='{best_key[0]}'")
    else:
        print("❌ No working keys found for any endpoint!")
        print("   Check server logs for authentication errors")

def main():
    print("🔐 MLX Vector DB Admin Authentication Tester")
    print("=" * 50)
    
    # Warte auf Server
    print("⏳ Waiting for server...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                break
        except:
            time.sleep(1)
    else:
        print("❌ Server not accessible")
        return
    
    # Führe Tests durch
    results = test_all_auth_combinations()
    
    # Analysiere Ergebnisse
    analyze_results(results)

if __name__ == "__main__":
    main()