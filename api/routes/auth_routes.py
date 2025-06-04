# Neue Datei: api/routes/auth_routes.py
# Dieser Router behandelt die Token-Ausgabe.

# MLX Specificity: Dieser Teil ist primär für die Anwendungslogik (Authentifizierung),
#                  nicht direkt MLX-spezifisch. Die Tokens schützen aber den Zugriff
#                  auf MLX-basierte Ressourcen.
# LLM Anbindung: Agenten benötigen einen sicheren Weg, um Tokens zu erhalten und
#                Zugriff auf die Vektor-Datenbank zu bekommen.

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm # Standardformular für Benutzername/Passwort
from typing import Dict, Any

# Importe aus Ihrem security-Modul
from security.auth import jwt_auth_handler, JWTAuth # JWTAuth für Typ-Hinting
from security.rbac import Role # Import Role für Beispiel

# (Optional) Import für eine Benutzerdatenbank/-verwaltung, falls vorhanden.
# Für dieses Beispiel verwenden wir eine Dummy-Benutzerprüfung.
# from your_user_management_module import authenticate_user, get_user_roles

router = APIRouter(prefix="/auth", tags=["authentication"])

# Dummy-Benutzerdatenbank und -funktionen für das Beispiel
# In einer Produktionsumgebung würden Sie hier eine echte Benutzerdatenbank/-prüfung integrieren.
DUMMY_USERS_DB = {
    "testuser": {
        "password_hash": hashlib.sha256("testpassword".encode()).hexdigest(), # Passwörter immer hashen!
        "roles": [Role.USER.value, Role.READONLY.value], # Rollen als Strings speichern
        "user_id": "user123"
    },
    "adminuser": {
        "password_hash": hashlib.sha256("adminpassword".encode()).hexdigest(),
        "roles": [Role.ADMIN.value],
        "user_id": "admin001"
    },
    "agent007": { # Beispiel für einen Agenten
        "client_secret_hash": hashlib.sha256("supersecretagentkey".encode()).hexdigest(), # Agenten könnten Client-ID/Secret verwenden
        "roles": [Role.AGENT.value],
        "user_id": "agent007_id" # Eindeutige ID für den Agenten
    }
}

def dummy_authenticate_user(username_or_client_id: str, password_or_secret: str) -> Optional[Dict[str, Any]]:
    """
    Dummy-Authentifizierungsfunktion. Ersetzbar durch Ihre echte Benutzerauthentifizierung.
    Gibt Benutzerdaten zurück, wenn erfolgreich, sonst None.
    """
    user_data = DUMMY_USERS_DB.get(username_or_client_id)
    if not user_data:
        return None

    # Unterscheidung zwischen Passwort-basierter und Client-Secret-basierter Auth
    if "password_hash" in user_data:
        expected_hash = user_data["password_hash"]
        provided_hash = hashlib.sha256(password_or_secret.encode()).hexdigest()
    elif "client_secret_hash" in user_data:
        expected_hash = user_data["client_secret_hash"]
        provided_hash = hashlib.sha256(password_or_secret.encode()).hexdigest()
    else:
        return None # Kein bekannter Auth-Mechanismus für diesen User

    if secrets.compare_digest(provided_hash, expected_hash):
        return {
            "sub": user_data["user_id"], # 'sub' (Subject) ist Standard-Claim für User-ID im JWT
            "username": username_or_client_id,
            "roles": user_data["roles"]
        }
    return None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int # In Sekunden
    user_info: Dict[str, Any] # Optionale User-Infos

@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends()
    # jwt_handler: JWTAuth = Depends(lambda: jwt_auth_handler) # Alternative Injektion
):
    """
    Nimmt Benutzername und Passwort entgegen (Standard OAuth2 Formular)
    und gibt einen Access Token zurück.
    Für Agenten könnte `username` die Client-ID und `password` das Client-Secret sein.
    """
    user_identity_data = dummy_authenticate_user(form_data.username, form_data.password)
    if not user_identity_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Daten für den Token-Payload vorbereiten
    # 'sub' ist der Subject Claim, typischerweise die User-ID.
    # 'roles' wird vom RBAC-System verwendet.
    token_payload_data = {
        "sub": user_identity_data["sub"],
        "username": user_identity_data["username"], # Optional, für Anzeige etc.
        "roles": user_identity_data["roles"]
        # Hier können weitere Claims hinzugefügt werden, die für Ihre Anwendung relevant sind.
    }
    
    access_token = jwt_auth_handler.create_access_token(data=token_payload_data)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": jwt_auth_handler.access_token_expire_minutes * 60,
        "user_info": { # Nur sichere, nicht-sensitive Infos zurückgeben
            "username": user_identity_data["username"],
            "roles": user_identity_data["roles"]
        }
    }