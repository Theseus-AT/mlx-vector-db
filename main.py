# main.py
"""
MLX Vector Database - Hauptanwendungsserver
Integriert Konfiguration, Logging, Tracing, Security und optionale Hintergrunddienste.
"""
import os
import time
import logging # Wird durch setup_logging() aus core.logging_config konfiguriert
from contextlib import asynccontextmanager
import secrets # Für Request ID in Middleware

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- Konfigurationsmanagement ---
# Versuche, die zentrale Konfiguration zu laden
try:
    from config.settings import get_config, ConfigManager
    config: ConfigManager = get_config()
    # Umgebungsvariablen für Logging setzen, damit setup_logging sie verwenden kann
    os.environ["LOG_LEVEL"] = config.server.log_level.upper()
    # Annahme: APP_LOG_LEVEL wird auch über config.server.log_level gesteuert oder spezifischer in config.monitoring gesetzt
    os.environ["APP_LOG_LEVEL"] = os.getenv("APP_LOG_LEVEL", config.server.log_level.upper())
    os.environ["LOG_HANDLER"] = os.getenv("LOG_HANDLER", "json") # Standard auf json, falls nicht anders gesetzt
except ImportError:
    print("KRITISCH: config.settings.py nicht gefunden. Abbruch.")
    exit(1) # Ohne Konfiguration kann die App nicht sinnvoll starten
except Exception as e:
    print(f"KRITISCH: Fehler beim Laden der Konfiguration aus config.settings: {e}. Abbruch.")
    exit(1)

# --- Strukturiertes Logging ---
# Annahme: logging_config.py ist in einem 'core' Verzeichnis oder direkt im Root
try:
    from core.logging_config import setup_logging, get_logger_with_extra
    setup_logging() # Verwendet die interne LOGGING_CONFIG und Umgebungsvariablen
    logger = get_logger_with_extra("mlx_vector_db.main")
except ImportError as e:
    print(f"WARNUNG: core.logging_config.py nicht gefunden. Strukturiertes Logging nicht verfügbar: {e}. Fallback auf Basis-Logging.")
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("mlx_vector_db.main") # type: ignore
    # Fallback für get_logger_with_extra, falls nicht geladen
    def get_logger_with_extra(name: str) -> logging.Logger: # type: ignore
        return logging.getLogger(name)

# --- OpenTelemetry Tracing ---
try:
    from telemetry.tracing_config import setup_tracing as setup_opentelemetry_tracing
except ImportError:
    logger.warning("telemetry.tracing_config.py nicht gefunden, OpenTelemetry Tracing ist deaktiviert.")
    def setup_opentelemetry_tracing(app, service_name, enable_console_exporter=False): pass # Dummy

# --- Security Initialisierung ---
try:
    from security.auth import ensure_api_key, jwt_auth_handler, JWTAuth # JWTAuth für Typ-Hinting
    # API-Key sicherstellen (wird geloggt, wenn neu generiert oder unsicher)
    # `ensure_api_key` sollte idealerweise den Key aus config.security.api_key prüfen und ggf. aktualisieren
    # oder direkt in ensure_api_key auf os.getenv("VECTOR_DB_API_KEY") zugreifen.
    # Fürs Erste nehmen wir an, ensure_api_key handhabt das.
    config.security.api_key = ensure_api_key() # ensure_api_key liest VECTOR_DB_API_KEY selbst

    # JWT Handler Instanz (jwt_auth_handler) wird in security.auth.py global erstellt und initialisiert
    # JWT_SECRET_KEY etc. werden dort via os.getenv aus der .env oder Umgebung geladen
    if jwt_auth_handler.secret_key == "your-default-super-secret-key-for-dev-only": # Zugriff auf Attribut der Instanz
         logger.warning("DEFAULT JWT SECRET KEY IN USE! Bitte JWT_SECRET_KEY in .env setzen für Produktion!")

except ImportError as e:
    logger.error(f"security.auth.py nicht gefunden oder fehlerhaft: {e}. Authentifizierung ist möglicherweise nicht funktionsfähig.")
    jwt_auth_handler = None # Fallback


# --- Persistenz: Write-Ahead Log (optional, basierend auf Config) ---
wal_manager = None
if hasattr(config.storage, "wal_path") and config.storage.wal_path: # Prüfen ob Attribut existiert
    try:
        from persistence.wal import WriteAheadLog
        wal_manager = WriteAheadLog(base_wal_path=config.storage.wal_path)
        logger.info(f"Write-Ahead Log Manager initialisiert für Pfad: {config.storage.wal_path}")
        # WAL Replay Logik sollte hier vor dem Start des Servers erfolgen,
        # um einen konsistenten Zustand sicherzustellen. Dies ist eine komplexe Operation.
        # entries_to_replay = wal_manager.replay_wal()
        # if entries_to_replay:
        #     logger.info(f"WAL Replay: {len(entries_to_replay)} Operationen werden wiederhergestellt...")
        #     # Hier die Logik zum Anwenden der Einträge
        #     # for entry in entries_to_replay: apply_wal_entry(entry)
        #     # wal_manager.checkpoint() # Nach erfolgreichem Replay
        # else:
        #     logger.info("WAL Replay: Kein Replay notwendig.")
    except ImportError:
        logger.warning("persistence.wal.py nicht gefunden. Write-Ahead Logging ist deaktiviert.")
    except AttributeError:
        logger.warning("Konfiguration 'storage.wal_path' nicht in settings.py gefunden. WAL deaktiviert.")
else:
    logger.info("Write-Ahead Logging ist nicht in der Konfiguration aktiviert (kein 'wal_path').")


# --- Hintergrunddienste-Instanzen (werden im Lifespan Manager gestartet) ---
cluster_manager_instance = None
vector_replicator_instance = None # Deklarieren für spätere Verwendung
auto_backup_manager_instance = None

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI): # app_instance umbenannt, um Konflikt mit globalem app zu vermeiden
    # Startup Events
    logger.info("MLX Vector DB Server startet...")
    load_dotenv() # Stellt sicher, dass .env Variablen auch für Lifespan Events verfügbar sind

    # MLX Warmup (aus config.performance)
    if config.performance.warmup_on_startup:
        logger.info("Führe MLX Warmup durch...")
        try:
            from performance.mlx_optimized import warmup_compiled_functions
            warmup_compiled_functions(
                dimension=config.mlx.warmup_dimension, # Aus MLXConfig
                n_vectors=config.mlx.warmup_vectors   # Aus MLXConfig
            )
            logger.info("MLX Warmup abgeschlossen.")
        except ImportError:
            logger.warning("performance.mlx_optimized.py nicht gefunden. MLX Warmup übersprungen.")
        except Exception as e_warmup:
            logger.error(f"Fehler beim MLX Warmup: {e_warmup}")

    # Clustering & Replication (aus config.clustering - Annahme: Muss in settings.py hinzugefügt werden)
    if hasattr(config, "clustering") and config.clustering.enable_clustering:
        try:
            from clustering.cluster_manager import ClusterManager
            global cluster_manager_instance # Um globale Variable zu modifizieren
            cluster_manager_instance = ClusterManager(redis_url=config.clustering.redis_url)
            await cluster_manager_instance.start()
            app_instance.state.cluster_manager = cluster_manager_instance
            logger.info(f"ClusterManager gestartet für Node ID: {cluster_manager_instance.node_id}")

            if hasattr(config.clustering, "enable_replication") and config.clustering.enable_replication:
                from replication.replicator import VectorReplicator
                global vector_replicator_instance
                vector_replicator_instance = VectorReplicator(
                    cluster_manager=cluster_manager_instance,
                    local_node_id=cluster_manager_instance.node_id
                )
                app_instance.state.vector_replicator = vector_replicator_instance
                logger.info("VectorReplicator initialisiert.")

        except ImportError:
            logger.error("Clustering oder Replikationsmodule nicht gefunden. Clustering deaktiviert.")
        except AttributeError:
            logger.error("Konfiguration für Clustering (z.B. enable_clustering, redis_url) fehlt in settings.py. Clustering deaktiviert.")
        except Exception as e_cluster:
            logger.error(f"Fehler beim Starten des ClusterManagers: {e_cluster}")
    else:
        logger.info("Clustering ist in der Konfiguration deaktiviert.")


    # AutoBackupManager (aus config.storage)
    if config.storage.enable_backup:
        try:
            from backup.auto_backup import AutoBackupManager
            from service.vector_store import list_users, list_models, get_store_path # Abhängigkeiten für Backup
            
            # Backup-Pfade und Intervalle müssen in StorageConfig oder einer neuen BackupConfig in settings.py definiert sein
            backup_path = getattr(config.storage, "backup_path", "./default_backups")
            backup_interval_hours = getattr(config.storage, "backup_interval_hours", 6)
            # backup_retention_days ist bereits in Ihrer StorageConfig

            global auto_backup_manager_instance
            auto_backup_manager_instance = AutoBackupManager(
                data_path_str=config.storage.base_path,
                backup_path_str=backup_path, # Verwende definierten oder Default-Pfad
                list_users_func=list_users,
                list_models_func=list_models,
                get_store_path_func=get_store_path
            )
            auto_backup_manager_instance.start(
                interval_hours=backup_interval_hours,
                retention_days=config.storage.backup_retention_days
            )
            app_instance.state.auto_backup_manager = auto_backup_manager_instance
            logger.info(f"AutoBackupManager gestartet. Backups nach: {backup_path}")
        except ImportError:
            logger.error("backup.auto_backup.py nicht gefunden. Automatische Backups deaktiviert.")
        except AttributeError as e_backup_attr:
            logger.error(f"Fehlende Konfiguration für AutoBackupManager in settings.py (z.B. backup_path, backup_interval_hours): {e_backup_attr}. Backups deaktiviert.")
        except Exception as e_backup:
            logger.error(f"Fehler beim Starten des AutoBackupManagers: {e_backup}")
    else:
        logger.info("Automatische Backups sind in der Konfiguration deaktiviert.")

    if wal_manager:
        app_instance.state.wal_manager = wal_manager

    logger.info(f"MLX Vector DB Server (Version {app_instance.version}) ist betriebsbereit auf http://{config.server.host}:{config.server.port}")
    yield # Hier läuft die Anwendung

    # Shutdown Events
    logger.info("MLX Vector DB Server fährt herunter...")
    if hasattr(app_instance.state, "auto_backup_manager") and app_instance.state.auto_backup_manager:
        logger.info("Stoppe AutoBackupManager...")
        app_instance.state.auto_backup_manager.stop()
    
    if hasattr(app_instance.state, "cluster_manager") and app_instance.state.cluster_manager:
        logger.info("Stoppe ClusterManager...")
        await app_instance.state.cluster_manager.stop()
        if hasattr(app_instance.state, "vector_replicator"): # Replicator hängt am ClusterManager
             logger.info("VectorReplicator wird implizit mit ClusterManager gestoppt.")
    
    logger.info("MLX Vector DB Server heruntergefahren.")


# --- FastAPI App Instanz ---
app = FastAPI(
    title=getattr(config, "app_title", "MLX Vector DB"), # Titel aus Config oder Default
    version=getattr(config, "app_version", "1.0.1"),    # Version aus Config oder Default
    description="High-performance vector database optimized for Apple Silicon with MLX",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# In main.py, nach app = FastAPI(...)
from core.logging_config import structured_request_logging_middleware
app.middleware("http")(structured_request_logging_middleware)

# --- Middleware Definition und Registrierung ---

# Strukturierte Request Logging Middleware (Beispiel aus logging_config.py angepasst)
async def structured_request_logging_middleware_impl(request: Request, call_next):
    request_id = secrets.token_hex(8)
    # Verwende get_logger_with_extra, falls es erfolgreich importiert wurde
    req_logger_adapter = get_logger_with_extra("mlx_vector_db.api.requests")
    
    start_time_mw = time.time() # Umbenannt, um Konflikt mit globalem time zu vermeiden
    
    # User-ID aus Token extrahieren (falls JWT/RBAC bereits integriert und Token vorhanden)
    user_id_from_token = "anonymous"
    if hasattr(request.state, "current_user_payload") and request.state.current_user_payload:
        user_id_from_token = request.state.current_user_payload.get("sub", "unknown_jwt_user")
    elif request.headers.get("x-api-key"):
        user_id_from_token = "api_key_user" # Vereinfacht für API-Key

    try:
        response = await call_next(request)
    except Exception as e:
        # Logge den Fehler hier, bevor er vom globalen Exception Handler behandelt wird
        duration_ms_mw_error = (time.time() - start_time_mw) * 1000
        error_log_details = {
            'event_type': 'http_request_error', 'request_id': request_id,
            'http_method': request.method, 'http_path': request.url.path,
            'duration_ms': round(duration_ms_mw_error, 2),
            'client_address': f"{request.client.host}:{request.client.port}",
            'user_id': user_id_from_token, 'error': str(e), 'error_type': type(e).__name__
        }
        req_logger_adapter.error(f"Request Error: {request.method} {request.url.path}", extra_fields=error_log_details, exc_info=True)
        raise # Fehler weiterwerfen, damit FastAPI ihn handhaben kann
    
    duration_ms_mw = (time.time() - start_time_mw) * 1000
    
    log_details = {
        'event_type': 'http_request', 'request_id': request_id,
        'http_method': request.method, 'http_path': request.url.path,
        'http_query_params': str(request.query_params),
        'http_status_code': response.status_code,
        'duration_ms': round(duration_ms_mw, 2),
        'client_address': f"{request.client.host}:{request.client.port}",
        'user_agent': request.headers.get("user-agent", "N/A"),
        'user_id': user_id_from_token
    }
    req_logger_adapter.info(f"{request.method} {request.url.path} - {response.status_code}", extra_fields=log_details)
    return response

app.middleware("http")(structured_request_logging_middleware_impl)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenTelemetry Tracing (nachdem App erstellt wurde und Config geladen)
if hasattr(config.monitoring, "enable_tracing") and config.monitoring.enable_tracing:
    otel_endpoint = getattr(config.monitoring, "otlp_endpoint", "http://localhost:4317")
    otel_console = getattr(config.monitoring, "otel_console_exporter", False)
    setup_opentelemetry_tracing(app, service_name=app.title, enable_console_exporter=otel_console)
    logger.info(f"OpenTelemetry Tracing konfiguriert. Endpoint: {otel_endpoint}, ConsoleExporter: {otel_console}")
else:
    logger.info("OpenTelemetry Tracing ist in der Konfiguration deaktiviert.")


# --- Globale Fehlerbehandlung ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Das Logging des Fehlers passiert jetzt idealerweise in der Middleware
    # oder hier, falls die Middleware ihn nicht fängt oder man es doppelt will.
    # logger.error(f"Unhandled exception for request {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Ein interner Serverfehler ist aufgetreten.", "error_type": type(exc).__name__},
    )

# --- API Router registrieren ---
# Annahme: Ihre Authentifizierungs-Dependencies werden in config.security definiert
# z.B. config.security.admin_auth_dependency = Depends(verify_admin_api_key_or_jwt_admin_role)
# Für den Moment verwenden wir die direkten Depends aus security.auth oder lassen sie weg, wenn Auth deaktiviert ist
# Die Dependencies für Router sollten idealerweise auch aus der Config kommen oder zentral definiert werden.

default_user_auth_dependency = []
if config.security.enable_auth and jwt_auth_handler: # Wenn Auth aktiv ist und JWT Handler existiert
    from security.auth import get_current_user_payload # Benötigt für JWT
    default_user_auth_dependency = [Depends(get_current_user_payload)]
elif config.security.enable_auth: # Fallback auf API Key, wenn kein JWT aber Auth an
    from security.auth import verify_api_key
    default_user_auth_dependency = [Depends(verify_api_key)]

default_admin_auth_dependency = []
if config.security.enable_auth: # Admin Routen immer schützen wenn Auth an
    # Hier müsste eine Dependency hin, die Admin-Rechte prüft (via API-Key oder JWT-Admin-Rolle)
    # Fürs Erste nehmen wir verify_admin_api_key als Beispiel
    from security.auth import verify_admin_api_key
    default_admin_auth_dependency = [Depends(verify_admin_api_key)]


try:
    from api.routes import admin, vectors, performance, monitoring, auth_routes
    app.include_router(admin.router, prefix="/admin", tags=["Admin"], dependencies=default_admin_auth_dependency)
    app.include_router(vectors.router, prefix="/vectors", tags=["Vectors"], dependencies=default_user_auth_dependency)
    app.include_router(performance.router, prefix="/performance", tags=["Performance"]) # Ggf. auch schützen
    app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"]) # Ggf. auch schützen
    app.include_router(auth_routes.router, prefix="/auth", tags=["Authentication"]) # Token-Endpunkt
    logger.info("API-Router erfolgreich geladen und (de-)aktivierter Authentifizierung konfiguriert.")
except ImportError as e:
    logger.error(f"Fehler beim Laden der API-Router: {e}. Die API ist möglicherweise nicht voll funktionsfähig.")


# --- Basis-Endpunkte ---
@app.get("/", tags=["General"])
async def root():
    return {
        "name": app.title,
        "version": app.version,
        "status": "running",
        "documentation": app.docs_url,
        "config_loaded": True # Da wir oben abbrechen, wenn Config nicht lädt
    }

@app.get("/health", tags=["General"])
async def health_check_endpoint():
    # Dieser könnte erweitert werden, um health_checker aus monitoring.metrics zu nutzen
    return {
        "status": "healthy",
        "version": app.version,
        "service": app.title,
        "timestamp": time.time()
    }

# --- Uvicorn Runner ---
def run_server_main(): # Umbenannt, um Konflikt mit argparse main zu vermeiden
    import uvicorn
    # Lade AUTO_RELOAD aus Umgebungsvariablen, um Hot-Reload in Entwicklung zu ermöglichen
    # In Produktion sollte dies 'false' sein.
    reload_flag = os.getenv("AUTO_RELOAD", "False").lower() == 'true'
    log_level_server = config.server.log_level.lower()

    # Warnung, wenn AUTO_RELOAD und mehrere Worker (was Uvicorn nicht direkt unterstützt)
    if reload_flag and config.server.workers > 1:
        logger.warning("AUTO_RELOAD ist aktiv, aber config.server.workers > 1. Uvicorn's --reload startet nur einen Worker.")
        
    logger.info(f"Starte FastAPI Server mit Uvicorn: Host={config.server.host}, Port={config.server.port}, LogLevel={log_level_server}, Reload={reload_flag}")
    
    uvicorn.run(
        "main:app", # Standard FastAPI App Referenz
        host=config.server.host,
        port=config.server.port,
        log_level=log_level_server,
        reload=reload_flag,
        # workers=config.server.workers # --workers ist besser als CLI-Argument für uvicorn, nicht hier
        access_log=True # Standard Uvicorn Access Logs
    )

if __name__ == "__main__":
    # Diese main() Funktion ist für das Starten via `python main.py`
    # Die main() Funktion aus Ihrer Originaldatei (für `pip install mlx-vector-db CLI`)
    # sollte idealerweise in ein separates CLI-Skript ausgelagert werden,
    # z.B. `cli.py`, das dann `run_server_main()` aufrufen kann oder andere CLI-Befehle.
    # Für den Moment rufen wir direkt den Serverstart auf.
    run_server_main()