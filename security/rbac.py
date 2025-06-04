# Neue Datei: security/rbac.py
# Implementiert Role-Based Access Control (RBAC) gemäß Produktionsplan.

# MLX Specificity: RBAC selbst ist generisch, kontrolliert aber den Zugriff
#                  auf MLX-bezogene Ressourcen und Operationen.
# LLM Anbindung: Definiert, welche Aktionen LLM-Agenten (basierend auf ihrer Rolle)
#                in der Vektor-DB ausführen dürfen (z.B. nur lesen, schreiben, Stores verwalten).

from enum import Enum
from typing import Dict, List, Set, Any # Any hinzugefügt
from functools import wraps
from fastapi import HTTPException, Depends, status # status hinzugefügt

# Import der get_current_user_payload Funktion aus security.auth
# Dies setzt voraus, dass security.auth.py JWT implementiert hat.
from .auth import get_current_user_payload # Relative Import

class Role(str, Enum): # Von str erben für einfache Verwendung in Pydantic etc.
    ADMIN = "admin"
    USER = "user"
    AGENT = "agent" # Beispiel für eine Rolle für LLM-Agenten
    READONLY = "readonly"
    # Weitere Rollen nach Bedarf

class Permission(str, Enum): # Von str erben
    # Store Management
    CREATE_STORE = "store:create"
    DELETE_STORE = "store:delete"
    LIST_STORES = "store:list" # Hinzugefügt für Admin-Übersichten
    GET_STORE_STATS = "store:get_stats" # Hinzugefügt
    
    # Vector Operations
    ADD_VECTORS = "vector:add"
    QUERY_VECTORS = "vector:query"
    DELETE_VECTORS = "vector:delete" # Hinzugefügt
    COUNT_VECTORS = "vector:count" # Hinzugefügt
    
    # Admin & Monitoring
    VIEW_METRICS = "monitoring:view_metrics"
    VIEW_HEALTH = "monitoring:view_health" # Unterscheidung zu allgemeinen Health-Checks
    MANAGE_USERS = "admin:manage_users" # Beispiel für User-Management
    MANAGE_ROLES = "admin:manage_roles" # Beispiel für Rollen-Management
    PERFORM_BACKUP = "admin:perform_backup" # Hinzugefügt
    PERFORM_RESTORE = "admin:perform_restore" # Hinzugefügt
    
    # Weitere Berechtigungen nach Bedarf

# Rollen-Berechtigungs-Mapping
# Dies sollte idealerweise flexibler gestaltet sein, z.B. aus einer DB oder Konfigurationsdatei geladen werden.
# Für den Anfang ist ein Hardcoding in Ordnung.
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.CREATE_STORE, Permission.DELETE_STORE, Permission.LIST_STORES, Permission.GET_STORE_STATS,
        Permission.ADD_VECTORS, Permission.QUERY_VECTORS, Permission.DELETE_VECTORS, Permission.COUNT_VECTORS,
        Permission.VIEW_METRICS, Permission.VIEW_HEALTH,
        Permission.MANAGE_USERS, Permission.MANAGE_ROLES,
        Permission.PERFORM_BACKUP, Permission.PERFORM_RESTORE,
    },
    Role.USER: { # Regulärer Benutzer, der eigene Stores verwalten und nutzen kann
        Permission.CREATE_STORE, # Erlaubt Usern, eigene Stores zu erstellen (Multi-Tenant)
        Permission.DELETE_STORE, # Nur eigene Stores
        Permission.LIST_STORES,  # Nur eigene Stores
        Permission.GET_STORE_STATS, # Nur eigene Stores
        Permission.ADD_VECTORS,    # In eigene Stores
        Permission.QUERY_VECTORS,  # Aus eigenen Stores
        Permission.DELETE_VECTORS, # Aus eigenen Stores
        Permission.COUNT_VECTORS,  # Eigene Stores
        Permission.VIEW_METRICS,   # Nur eigene Store-Metriken (falls implementiert)
    },
    Role.AGENT: { # LLM-Agent mit eingeschränkten Rechten, typischerweise für einen bestimmten User/Store
        Permission.ADD_VECTORS,    # Z.B. um neues Wissen zu indizieren
        Permission.QUERY_VECTORS,  # Hauptfunktion für RAG
        Permission.VIEW_METRICS,   # Um Performance zu überwachen
    },
    Role.READONLY: {
        Permission.QUERY_VECTORS,
        Permission.COUNT_VECTORS,
        Permission.LIST_STORES, # Ggf. nur öffentliche oder zugewiesene Stores
        Permission.GET_STORE_STATS,
        Permission.VIEW_METRICS,
    },
}

def user_has_permission(user_payload: Dict[str, Any], permission: Permission) -> bool:
    """
    Prüft, ob der Benutzer (basierend auf seiner JWT-Payload) die angeforderte Berechtigung hat.
    Die Payload sollte die Rolle(n) des Benutzers enthalten, z.B. user_payload.get("roles", [])
    """
    user_roles = user_payload.get("roles", []) # Annahme: 'roles' ist eine Liste von Rollen-Strings in der Payload
    if not user_roles:
        return False

    for role_str in user_roles:
        try:
            role = Role(role_str) # Konvertiere String zu Role Enum
            if role in ROLE_PERMISSIONS and permission in ROLE_PERMISSIONS[role]:
                return True
        except ValueError: # Ungültiger Rollen-String in Payload
            logger.warning(f"Invalid role string '{role_str}' in user payload.")
            continue
    return False

def require_permission(permission: Permission):
    """
    Decorator für FastAPI-Routen, der prüft, ob der authentifizierte Benutzer
    die erforderliche Berechtigung hat.
    Setzt voraus, dass `get_current_user_payload` als Dependency verwendet wird.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Hole current_user (Payload) aus den kwargs, die von Depends(get_current_user_payload) bereitgestellt werden
            user_payload: Optional[Dict[str, Any]] = kwargs.get('current_user_payload') # Name anpassen, falls anders in Depends
            
            # Wenn die Dependency anders heißt, muss der Key hier angepasst werden.
            # Alternativ: Direkte Dependency auf get_current_user_payload hier einfügen.
            # async def wrapper(current_user_payload: Dict[str, Any] = Depends(get_current_user_payload), *args, **kwargs):
            
            if not user_payload: # Sollte durch HTTPBearer in get_current_user_payload abgefangen werden
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated"
                )

            if not user_has_permission(user_payload, permission):
                logger.warning(f"User {user_payload.get('sub')} with roles {user_payload.get('roles')} denied permission {permission} for {func.__name__}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Requires: {permission.value}"
                )
            
            logger.debug(f"User {user_payload.get('sub')} granted permission {permission} for {func.__name__}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Beispiel für die Anwendung auf einer Route (in den API-Routen-Dateien):
# from .security.rbac import require_permission, Permission, Role # (Annahme: relative Pfade)
# from .security.auth import get_current_user_payload
#
# @router.post("/admin/some_admin_action")
# @require_permission(Permission.MANAGE_USERS)
# async def admin_action(current_user_payload: Dict[str, Any] = Depends(get_current_user_payload)):
#     # Logik hier...
#     # current_user_payload enthält {'sub': 'user_id', 'roles': ['admin'], ...}
#     return {"message": "Admin action successful", "user": current_user_payload.get("sub")}