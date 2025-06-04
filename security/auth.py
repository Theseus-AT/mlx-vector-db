# security/auth.py
# Erweiterung um JWT-Authentifizierung gemäß Produktionsplan.
# Die bestehende API-Key-Authentifizierung kann parallel existieren oder
# für bestimmte (z.B. interne/Legacy) Zwecke beibehalten werden.
# Für neue, User-bezogene Endpunkte sollte JWT bevorzugt werden.

# MLX Specificity: Die Authentifizierung selbst ist MLX-agnostisch, aber sie
#                  schützt den Zugriff auf MLX-basierte Daten und Operationen.
# LLM Anbindung: Multiuser-Sicherheit ist essentiell, wenn LLM-Agenten mit
#                individuellen Rechten auf die Vektor-DB zugreifen.

import os
import re
import hashlib
import secrets
from typing import Optional, Dict, Any
from fastapi import HTTPException, Header, Depends, Request, status # status hinzugefügt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import logging

# JWT-spezifische Imports gemäß Plan
from jose import JWTError, jwt
from datetime import datetime, timedelta

logger = logging.getLogger("mlx_vector_db.auth")

# --- Bestehende Ratenbegrenzung (RateLimiter, _rate_limit_storage) ---
# Der Code für RateLimiter und _rate_limit_storage bleibt unverändert hier.
# (Code aus theseus-at/mlx-vector-db/mlx-vector-db-12319040886f6c23f84935346248ef50cf06157f/security/auth.py einfügen)
_rate_limit_storage: Dict[str, Dict[str, Any]] = {}

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        
        if identifier not in _rate_limit_storage:
            _rate_limit_storage[identifier] = {
                "requests": [],
                "blocked_until": 0
            }
        
        user_data = _rate_limit_storage[identifier]
        
        # Check if user is currently blocked
        if now < user_data["blocked_until"]:
            return False
        
        # Clean old requests outside window
        user_data["requests"] = [
            req_time for req_time in user_data["requests"] 
            if now - req_time < self.window_seconds
        ]
        
        # Check rate limit
        if len(user_data["requests"]) >= self.max_requests:
            # Block for 1 hour
            user_data["blocked_until"] = now + 3600
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Add current request
        user_data["requests"].append(now)
        return True

# Globale Ratenbegrenzer-Instanzen
api_rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)
admin_rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)
# --- Ende Ratenbegrenzung ---


# --- Bestehende API-Key Verifizierung ---
# Der Code für security_scheme, validate_api_key, get_client_identifier,
# verify_api_key, verify_admin_api_key, validate_safe_identifier,
# validate_file_upload, ensure_api_key bleibt hier.
# (Code aus theseus-at/mlx-vector-db/mlx-vector-db-12319040886f6c23f84935346248ef50cf06157f/security/auth.py einfügen)
security_scheme = HTTPBearer(auto_error=False)

def validate_api_key(api_key: str) -> bool:
    """Validate API key securely"""
    expected_key = os.getenv("VECTOR_DB_API_KEY")
    
    if not expected_key or not api_key:
        return False
    
    # Use constant-time comparison to prevent timing attacks
    return secrets.compare_digest(api_key, expected_key)

def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting"""
    # Try to get real IP from headers (for reverse proxy setups)
    real_ip = request.headers.get("X-Real-IP")
    forwarded_for = request.headers.get("X-Forwarded-For")
    
    if real_ip:
        return real_ip
    elif forwarded_for:
        # Get first IP from comma-separated list
        return forwarded_for.split(",")[0].strip()
    else:
        return request.client.host if request.client else "unknown"

async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    x_api_key: Optional[str] = Header(None)
) -> str:
    """Verify API key from Authorization header or X-API-Key header"""
    
    # Get API key from different sources
    api_key_value = None # Umbenannt, um Konflikt mit Modulnamen zu vermeiden
    if credentials:
        api_key_value = credentials.credentials
    elif x_api_key:
        api_key_value = x_api_key
    
    if not api_key_value:
        logger.warning(f"Missing API key from {get_client_identifier(request)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="API key required. Use Authorization: Bearer <key> or X-API-Key header"
        )
    
    if not validate_api_key(api_key_value):
        logger.warning(f"Invalid API key attempt from {get_client_identifier(request)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    
    # Rate limiting
    client_id = get_client_identifier(request)
    if not api_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
    
    return api_key_value

async def verify_admin_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    x_api_key: Optional[str] = Header(None)
) -> str:
    """Verify API key for admin operations with stricter rate limiting"""
    
    # First verify basic API key
    api_key_value = await verify_api_key(request, credentials, x_api_key) # Nutzt umbenannte Variable
    
    # Additional rate limiting for admin operations
    client_id = get_client_identifier(request)
    if not admin_rate_limiter.is_allowed(client_id):
        logger.warning(f"Admin rate limit exceeded for {client_id}")
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Admin rate limit exceeded")
    
    logger.info(f"Admin operation authorized for {client_id}")
    return api_key_value

def validate_safe_identifier(value: str, field_name: str = "identifier") -> str:
    """Validate that user_id/model_id are safe"""
    if not value:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{field_name} cannot be empty")
    
    if len(value) > 50:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{field_name} too long (max 50 chars)")
    
    # Only allow alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"{field_name} contains invalid characters. Only a-z, A-Z, 0-9, _, - allowed"
        )
    
    # Prevent path traversal attempts
    dangerous_patterns = ['..', '/', '\\', 'CON', 'PRN', 'AUX', 'NUL']
    value_upper = value.upper()
    for pattern in dangerous_patterns:
        if pattern in value_upper:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"{field_name} contains prohibited pattern: {pattern}"
            )
    
    return value

def validate_file_upload(file_size: int, content_type: Optional[str], max_size_mb: int = 100) -> None: # content_type optional gemacht
    """Validate file upload security"""
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
            detail=f"File too large. Maximum size: {max_size_mb}MB"
        )
    
    allowed_types = ["application/zip", "application/x-zip-compressed"]
    if content_type and content_type not in allowed_types: # Check ob content_type existiert
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid file type: {content_type}. Only ZIP files allowed"
        )

# Optional: Create API key if not exists
def ensure_api_key():
    """Ensure API key exists in environment"""
    api_key_value = os.getenv("VECTOR_DB_API_KEY") # Umbenannt
    if not api_key_value or api_key_value == "dev-key-change-in-production":
        new_key = secrets.token_urlsafe(32)
        logger.warning(
            f"No secure API key found. Generated new key: {new_key}\n"
            f"Add this to your .env file: VECTOR_DB_API_KEY={new_key}"
        )
        # Wichtig: In einer realen Anwendung sollte der Key nicht nur geloggt,
        # sondern sicher gespeichert und dem Admin mitgeteilt werden.
        # Für Entwicklungszwecke kann man ihn setzen:
        # os.environ["VECTOR_DB_API_KEY"] = new_key
        return new_key
    return api_key_value

# --- Ende API-Key Verifizierung ---


# --- NEU: JWT Authentication gemäß Plan ---
# Security scheme für JWT
jwt_bearer_scheme = HTTPBearer(auto_error=True) # auto_error=True, um direkt 401 zu werfen

class JWTAuth:
    def __init__(self):
        # Geheime Schlüssel und Algorithmus aus Umgebungsvariablen oder Konfiguration laden
        # Fallback auf Default-Werte nur für Entwicklung, in Produktion MUSS dies gesetzt sein.
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-default-super-secret-key-for-dev-only")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

        if self.secret_key == "your-default-super-secret-key-for-dev-only":
            logger.warning("SECURITY WARNING: Using default JWT_SECRET_KEY. This is insecure and should ONLY be used for development.")

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Erstellt einen neuen Access Token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        # Standard-Claims wie 'sub' (subject, z.B. user_id) sollten in 'data' enthalten sein.
        # z.B. data = {"sub": user_id, "roles": ["user"]}
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token_and_get_payload(self, token: str) -> Dict[str, Any]:
        """
        Verifiziert den Token und gibt die Payload zurück.
        Wirft HTTPException bei Fehlern.
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            # Hier könnten weitere Prüfungen stattfinden, z.B. ob der Token in einer Blacklist ist.
            username: Optional[str] = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials: Missing 'sub' (subject) in token",
                )
            return payload
        except JWTError as e:
            logger.error(f"JWT Error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Could not validate credentials: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}, # Wichtig für korrekte Client-Reaktion
            )

# Globale Instanz für JWT Auth
jwt_auth_handler = JWTAuth()

# Dependency für FastAPI, um den aktuellen User (Payload) aus dem Token zu erhalten
async def get_current_user_payload(credentials: HTTPAuthorizationCredentials = Depends(jwt_bearer_scheme)) -> Dict[str, Any]:
    """
    Dependency, die den JWT Token aus dem Authorization Header validiert
    und die Payload zurückgibt.
    """
    token = credentials.credentials
    return jwt_auth_handler.verify_token_and_get_payload(token)

# Beispiel für eine geschützte Route (später in den API-Routen verwenden)
# @router.get("/users/me")
# async def read_users_me(current_user: Dict[str, Any] = Depends(get_current_user_payload)):
#     return current_user

# Initialisierung beim Start (z.B. in main.py oder config)
# ensure_api_key() # Stellt sicher, dass ein API Key existiert (für die API-Key basierte Auth)
# jwt_auth_handler = JWTAuth() # Initialisiert den JWT Handler

# Wichtig: Es muss ein Endpunkt zum Erstellen von Tokens geben (z.B. /auth/token),
# der Benutzername/Passwort prüft und dann jwt_auth_handler.create_access_token aufruft.
# Dieser Endpunkt ist nicht Teil dieser Datei, sondern wäre z.B. in api/routes/auth.py.