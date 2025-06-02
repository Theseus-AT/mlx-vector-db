# security/auth.py
"""
Authentication and authorization for MLX Vector Database
"""
import os
import re
import hashlib
import secrets
from typing import Optional, Dict, Any
from fastapi import HTTPException, Header, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import logging

logger = logging.getLogger("mlx_vector_db.auth")

# Rate limiting storage (in production use Redis)
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

# Global rate limiter instances
api_rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)  # 1000 req/hour
admin_rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)  # 100 req/hour for admin

# Security scheme
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
    api_key = None
    if credentials:
        api_key = credentials.credentials
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key:
        logger.warning(f"Missing API key from {get_client_identifier(request)}")
        raise HTTPException(
            status_code=401, 
            detail="API key required. Use Authorization: Bearer <key> or X-API-Key header"
        )
    
    if not validate_api_key(api_key):
        logger.warning(f"Invalid API key attempt from {get_client_identifier(request)}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Rate limiting
    client_id = get_client_identifier(request)
    if not api_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    return api_key

async def verify_admin_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    x_api_key: Optional[str] = Header(None)
) -> str:
    """Verify API key for admin operations with stricter rate limiting"""
    
    # First verify basic API key
    api_key = await verify_api_key(request, credentials, x_api_key)
    
    # Additional rate limiting for admin operations
    client_id = get_client_identifier(request)
    if not admin_rate_limiter.is_allowed(client_id):
        logger.warning(f"Admin rate limit exceeded for {client_id}")
        raise HTTPException(status_code=429, detail="Admin rate limit exceeded")
    
    logger.info(f"Admin operation authorized for {client_id}")
    return api_key

def validate_safe_identifier(value: str, field_name: str = "identifier") -> str:
    """Validate that user_id/model_id are safe"""
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    
    if len(value) > 50:
        raise HTTPException(status_code=400, detail=f"{field_name} too long (max 50 chars)")
    
    # Only allow alphanumeric, underscore, hyphen
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise HTTPException(
            status_code=400, 
            detail=f"{field_name} contains invalid characters. Only a-z, A-Z, 0-9, _, - allowed"
        )
    
    # Prevent path traversal attempts
    dangerous_patterns = ['..', '/', '\\', 'CON', 'PRN', 'AUX', 'NUL']
    value_upper = value.upper()
    for pattern in dangerous_patterns:
        if pattern in value_upper:
            raise HTTPException(
                status_code=400, 
                detail=f"{field_name} contains prohibited pattern: {pattern}"
            )
    
    return value

def validate_file_upload(file_size: int, content_type: str, max_size_mb: int = 100) -> None:
    """Validate file upload security"""
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {max_size_mb}MB"
        )
    
    allowed_types = ["application/zip", "application/x-zip-compressed"]
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {content_type}. Only ZIP files allowed"
        )

# Optional: Create API key if not exists
def ensure_api_key():
    """Ensure API key exists in environment"""
    api_key = os.getenv("VECTOR_DB_API_KEY")
    if not api_key or api_key == "dev-key-change-in-production":
        new_key = secrets.token_urlsafe(32)
        logger.warning(
            f"No secure API key found. Generated new key: {new_key}\n"
            f"Add this to your .env file: VECTOR_DB_API_KEY={new_key}"
        )
        return new_key
    return api_key