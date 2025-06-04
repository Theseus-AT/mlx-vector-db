#!/usr/bin/env python3
"""
Debug-Skript zum Überprüfen der API-Key-Konfiguration
Zeigt die aktuell konfigurierten API-Keys an
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Überprüfe .env Datei"""
    env_file = Path(".env")
    print("🔍 Checking .env file...")
    
    if env_file.exists():
        print(f"   ✅ .env file exists: {env_file.absolute()}")
        with open(env_file, 'r') as f:
            content = f.read()
            print("   📄 Content:")
            for line in content.split('\n'):
                if line.strip() and not line.startswith('#'):
                    key, _, value = line.partition('=')
                    if 'KEY' in key.upper():
                        print(f"      {key}={value[:10]}...")
                    else:
                        print(f"      {key}={value}")
    else:
        print("   ❌ .env file not found")
        return False
    
    return True

def check_environment_vars():
    """Überprüfe Umgebungsvariablen"""
    print("\n🔍 Checking environment variables...")
    
    api_key = os.getenv("VECTOR_DB_API_KEY")
    admin_key = os.getenv("VECTOR_DB_ADMIN_KEY")
    
    print(f"   VECTOR_DB_API_KEY: {api_key[:10] + '...' if api_key else 'NOT SET'}")
    print(f"   VECTOR_DB_ADMIN_KEY: {admin_key[:10] + '...' if admin_key else 'NOT SET'}")
    
    return api_key is not None and admin_key is not None

def check_auth_module():
    """Überprüfe auth.py Konfiguration"""
    print("\n🔍 Checking auth.py configuration...")
    
    try:
        sys.path.append('.')
        from security.auth import get_api_key, get_admin_key
        
        current_api_key = get_api_key()
        current_admin_key = get_admin_key()
        
        print(f"   Current API Key: {current_api_key[:10]}...")
        print(f"   Current Admin Key: {current_admin_key[:10]}...")
        
        return current_api_key, current_admin_key
        
    except Exception as e:
        print(f"   ❌ Error importing auth module: {e}")
        return None, None

def test_api_key_match():
    """Teste ob API-Keys übereinstimmen"""
    print("\n🧪 Testing API key match...")
    
    # Keys aus verschiedenen Quellen
    env_api_key = os.getenv("VECTOR_DB_API_KEY", "mlx-vector-dev-key-2024")
    
    try:
        from security.auth import get_api_key
        auth_api_key = get_api_key()
        
        print(f"   Test script uses: {env_api_key}")
        print(f"   Server uses: {auth_api_key}")
        
        if env_api_key == auth_api_key:
            print("   ✅ API keys match!")
            return True
        else:
            print("   ❌ API keys do NOT match!")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking keys: {e}")
        return False

def create_env_file():
    """Erstelle .env Datei mit korrekten Keys"""
    print("\n🔧 Creating .env file...")
    
    env_content = """# MLX Vector Database Environment Configuration
VECTOR_DB_API_KEY=mlx-vector-dev-key-2024
VECTOR_DB_ADMIN_KEY=mlx-vector-admin-key-2024
ENVIRONMENT=development
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("   ✅ .env file created successfully!")
        return True
    except Exception as e:
        print(f"   ❌ Error creating .env file: {e}")
        return False

def main():
    print("🔐 MLX Vector DB API Key Debug Tool")
    print("=" * 50)
    
    # 1. Überprüfe .env Datei
    env_exists = check_env_file()
    
    # 2. Überprüfe Umgebungsvariablen
    env_vars_set = check_environment_vars()
    
    # 3. Überprüfe auth.py
    api_key, admin_key = check_auth_module()
    
    # 4. Teste Übereinstimmung
    keys_match = test_api_key_match()
    
    print("\n📋 SUMMARY:")
    print("=" * 30)
    
    if not env_exists:
        print("❌ No .env file found")
        if input("\nCreate .env file? (y/n): ").lower() == 'y':
            create_env_file()
            print("\n🔄 Please restart the server after creating .env file:")
            print("   python main.py")
    elif not keys_match:
        print("❌ API keys don't match between test script and server")
        print("\n💡 Solutions:")
        print("1. Restart the server to load .env file:")
        print("   python main.py")
        print("2. Or set environment variables:")
        print("   export VECTOR_DB_API_KEY=mlx-vector-dev-key-2024")
        print("   export VECTOR_DB_ADMIN_KEY=mlx-vector-admin-key-2024")
    else:
        print("✅ Everything looks good!")
        
        # Zeige Test-Command
        if api_key:
            print(f"\n🧪 Test command:")
            print(f'curl -H "Authorization: Bearer {api_key}" http://localhost:8000/performance/health')

if __name__ == "__main__":
    main()