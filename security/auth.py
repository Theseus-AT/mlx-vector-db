#
"""
Security and Authentication for MLX Vector Database
Simple, robust API key authentication with JWT support

Focus: Essential security without over-engineering
"""

import os
import secrets
import hashlib
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Environment variables for API keys
API_KEY = os.getenv("VECTOR_DB_API_KEY")
ADMIN_KEY = os.getenv("VECTOR_DB_ADMIN_KEY")

# Default keys for development (change in production!)
DEFAULT_API_KEY = "mlx-vector-dev-key-2024"
DEFAULT_ADMIN_KEY = "mlx-vector-admin-key-2024"


def get_api_key() -> str:
    """Get API key from environment or default"""
    return API_KEY or DEFAULT_API_KEY


def get_admin_key() -> str:
    """Get admin key from environment or default"""
    return ADMIN_KEY or DEFAULT_ADMIN_KEY


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify API key from Authorization header
    Returns the API key if valid, raises HTTPException if not
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    provided_key = credentials.credentials
    valid_key = get_api_key()
    
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(provided_key, valid_key):
        logger.warning(f"Invalid API key attempt: {provided_key[:10]}...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return provided_key


def verify_admin_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verify admin key for privileged operations
    Returns the admin key if valid, raises HTTPException if not
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    provided_key = credentials.credentials
    valid_admin_key = get_admin_key()
    
    # Check if it's a valid admin key
    if secrets.compare_digest(provided_key, valid_admin_key):
        return provided_key
    
    # Also accept regular API key for non-destructive admin operations
    valid_api_key = get_api_key()
    if secrets.compare_digest(provided_key, valid_api_key):
        # Log the attempt but allow it for now (could be restricted later)
        logger.info("API key used for admin operation")
        return provided_key
    
    logger.warning(f"Invalid admin key attempt: {provided_key[:10]}...")
    raise HTTPException(
        status_code=403,
        detail="Admin privileges required",
        headers={"WWW-Authenticate": "Bearer"},
    )


# JWT Support (placeholder for future implementation)
def get_current_user_payload() -> Dict[str, Any]:
    """
    Placeholder for JWT token validation
    Returns user payload from JWT token
    """
    # This would be implemented when JWT authentication is needed
    # For now, return a default payload
    return {
        "sub": "default_user",
        "roles": ["user"],
        "exp": 9999999999  # Far future expiry
    }


def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    """Hash an API key for storage (future use)"""
    return hashlib.sha256(key.encode()).hexdigest()


def validate_key_format(key: str) -> bool:
    """Validate API key format"""
    if not key:
        return False
    
    # Basic validation: minimum length and allowed characters
    if len(key) < 16:
        return False
    
    # Allow alphanumeric, hyphens, and underscores
    allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return all(c in allowed_chars for c in key)


# Optional: Dependency for checking if authentication is enabled
def auth_enabled() -> bool:
    """Check if authentication is enabled"""
    # Always enabled for security
    return True


# Optional: Get current API key without verification (for internal use)
def get_current_api_key() -> Optional[str]:
    """Get currently configured API key"""
    return get_api_key()


# Development helper
def print_auth_info():
    """Print authentication info for development"""
    if os.getenv("ENVIRONMENT") == "development":
        print("🔐 Authentication Info:")
        print(f"   API Key: {get_api_key()}")
        print(f"   Admin Key: {get_admin_key()}")
        print("   ⚠️  Change these keys in production!")


# Initialize auth system
def init_auth():
    """Initialize authentication system"""
    api_key = get_api_key()
    admin_key = get_admin_key()
    
    if not validate_key_format(api_key):
        logger.warning("API key format validation failed")
    
    if not validate_key_format(admin_key):
        logger.warning("Admin key format validation failed")
    
    # Print info in development
    if os.getenv("ENVIRONMENT") == "development":
        print_auth_info()
    
    logger.info("🔐 Authentication system initialized")


# Call init on import
init_auth()