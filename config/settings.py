# config/settings.py - Kompletter korrigierter Header
"""
Erweitertes Konfigurationsmanagement für MLX Vector Database.
Unterstützt Umgebungsvariablen, .env-Datei und optionale JSON-Konfigurationsdateien.
"""
import os
import json
import threading
from typing import Any, Dict, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging
from dotenv import load_dotenv

logger = logging.getLogger("mlx_vector_db.config")

@dataclass
class ServerConfig:
    """Server-Konfiguration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1 # Relevant, wenn Uvicorn über Gunicorn o.ä. mit Workers gestartet wird
    log_level: str = "INFO" # Wird auch für logging_config.py verwendet
    app_title: str = "MLX Vector DB"
    app_version: str = "1.0.0" # Sollte idealerweise aus pyproject.toml o.ä. stammen
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size_mb: int = 100 # In MB

@dataclass
class SecurityConfig:
    """Sicherheitskonfiguration"""
    api_key: str = "dev-key-please-change" # Standard-API-Key für Entwicklung
    internal_api_key: str = "dev-internal-key-please-change" # Für Node-zu-Node Kommunikation
    enable_auth: bool = True
    
    # JWT Einstellungen
    jwt_secret_key: str = "a-very-secret-key-for-dev-change-in-prod" # In .env setzen!
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    
    # Ratenbegrenzung
    rate_limit_requests: int = 1000
    rate_limit_window_seconds: int = 3600
    admin_rate_limit_requests: int = 100
    admin_rate_limit_window_seconds: int = 3600
    
    max_file_upload_size_mb: int = 100

    # Platzhalter für dynamische Auth-Dependencies (Strings, die in main.py aufgelöst werden)
    # default_user_auth_dependency_str: Optional[str] = None # z.B. "security.auth.get_current_user_payload"
    # default_admin_auth_dependency_str: Optional[str] = None # z.B. "security.auth.verify_admin_jwt_or_strong_apikey"


@dataclass
class PerformanceConfig:
    """Performance-Konfiguration"""
    max_vector_cache_size_gb: float = 2.0 # Angepasst für typische lokale Entwicklung
    cache_cleanup_threshold: float = 0.9
    query_result_cache_max_size: int = 1000 # Für den neuen QueryResultCache
    query_result_cache_ttl_seconds: int = 3600

    default_batch_size: int = 1000
    
    # HNSW Einstellungen (basierend auf Ihrer neuen hnsw_index.HNSWConfig)
    enable_hnsw: bool = True
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100 # Angepasst an Ihre neue HNSWConfig
    hnsw_metric: str = 'l2' # 'l2' oder 'cosine'
    auto_index_threshold: int = 100 # Kleinere Schwelle für schnellere Indexerstellung im Dev

    enable_compiled_functions: bool = True # MLX @mx.compile

@dataclass
class StorageConfig:
    """Speicherkonfiguration"""
    base_path: str = "~/.mlx_vector_db_data/vector_stores" # Geänderter Default-Pfad
    wal_path: str = "~/.mlx_vector_db_data/wal"
    backup_path: str = "~/.mlx_vector_db_data/backups"
    
    enable_wal: bool = True # WAL standardmäßig aktivieren
    enable_backup: bool = False # Automatische Backups standardmäßig aus
    backup_interval_hours: int = 6
    backup_retention_days: int = 7
    
    # Diese Felder waren in Ihrer ursprünglichen settings.py, können aber nützlich sein
    compression_enabled: bool = False # Für Backups oder Daten?
    sync_writes: bool = True # Wichtig für WAL und Datenkonsistenz

@dataclass
class MonitoringConfig:
    """Monitoring-Konfiguration"""
    enable_metrics: bool = True
    # metrics_port: int = 9090 # Wird von Prometheus direkt gescraped, kein separater Port nötig wenn via /metrics Endpunkt
    
    enable_health_checks: bool = True
    # health_check_interval: int = 30 # Intervall wird intern im HealthChecker gehandhabt

    enable_system_metrics: bool = True # Für psutil Metriken
    # system_metrics_interval: int = 10 # Intern im MetricsRegistry gehandhabt

    enable_tracing: bool = False # Tracing standardmäßig aus
    otlp_endpoint: str = "http://localhost:4317" # Standard OTLP gRPC Collector
    otlp_insecure: bool = True # Für lokale Entwicklung
    otel_console_exporter: bool = False # Für Debugging von Spans auf der Konsole

    # Die logging_config (das große Dict) wird direkt in core.logging_config.py verwaltet.
    # Hier könnten aber globale Log-Einstellungen stehen, die setup_logging() beeinflussen,
    # z.B. globaler Log-Level oder ob JSON-Logging standardmäßig aktiv ist.
    # Diese werden aktuell über config.server.log_level und ENV-Vars in setup_logging() gehandhabt.


@dataclass
class MLXConfig:
    """MLX Framework-Konfiguration"""
    # default_dtype: str = "float32" # MLX verwendet intern float32, dies ist eher informativ
    warmup_on_startup: bool = True
    warmup_dimension: int = 384
    warmup_vectors: int = 1000
    # compile_functions: bool = True # Gesteuert über PerformanceConfig.enable_compiled_functions
    # gpu_enabled: bool = True # MLX entscheidet dies automatisch

@dataclass
class ClusteringConfig:
    """Clustering-Konfiguration (NEU)"""
    enable_clustering: bool = False
    redis_url: str = "redis://localhost:6379"
    enable_replication: bool = False # Ob der Replicator genutzt werden soll
    enable_leader_election: bool = True # Ob Leader Election aktiv sein soll


class ConfigManager:
    """
    Zentrales Konfigurationsmanagement.
    Lädt aus Defaults, .env-Datei, optionaler JSON-Datei und Umgebungsvariablen.
    """
    def __init__(self, config_file_path: Optional[str] = None):
        # Erweitere config_data mit allen erwarteten Schlüsseln und deren Defaults
        self._initialize_config_data_defaults()

        if config_file_path:
            self.config_file = Path(config_file_path)
        else:
            # Versuche, CONFIG_FILE aus Umgebungsvariablen zu laden
            env_config_file = os.getenv("CONFIG_FILE")
            self.config_file = Path(env_config_file) if env_config_file else None

        self._load_env_file() # .env Datei laden (überschreibt Defaults in self.config_data)
        
        if self.config_file and self.config_file.exists() and self.config_file.is_file():
            self._load_config_from_json_file() # JSON-Datei laden (überschreibt .env und Defaults)
        
        self._load_environment_variables() # Umgebungsvariablen laden (höchste Präzedenz)
        
        # Erstelle Konfigurationsobjekte basierend auf dem finalen self.config_data
        self.server = self._create_server_config()
        self.security = self._create_security_config()
        self.performance = self._create_performance_config()
        self.storage = self._create_storage_config()
        self.monitoring = self._create_monitoring_config()
        self.mlx = self._create_mlx_config()
        self.clustering = self._create_clustering_config() # Neu
        
        self._validate_config()
        logger.info("ConfigManager: Konfiguration erfolgreich geladen und validiert.")

    def _initialize_config_data_defaults(self):
        """Initialisiert self.config_data mit allen bekannten Schlüsseln und deren Standardwerten."""
        self.config_data = {
            # Server
            "HOST": "0.0.0.0", "PORT": "8000", "WORKERS": "1", "LOG_LEVEL": "INFO",
            "APP_TITLE": "MLX Vector DB", "APP_VERSION": "1.0.1",
            "CORS_ORIGINS": "*", "MAX_REQUEST_SIZE_MB": "100",
            # Security
            "API_KEY": "dev-key-please-change", "INTERNAL_API_KEY": "dev-internal-key-please-change",
            "ENABLE_AUTH": "true", "JWT_SECRET_KEY": "a-very-secret-key-for-dev-change-in-prod",
            "JWT_ALGORITHM": "HS256", "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "30",
            "RATE_LIMIT_REQUESTS": "1000", "RATE_LIMIT_WINDOW_SECONDS": "3600",
            "ADMIN_RATE_LIMIT_REQUESTS": "100", "ADMIN_RATE_LIMIT_WINDOW_SECONDS": "3600",
            "MAX_FILE_UPLOAD_SIZE_MB": "100",
            # Performance
            "MAX_VECTOR_CACHE_SIZE_GB": "2.0", "CACHE_CLEANUP_THRESHOLD": "0.9",
            "QUERY_RESULT_CACHE_MAX_SIZE": "1000", "QUERY_RESULT_CACHE_TTL_SECONDS": "3600",
            "DEFAULT_BATCH_SIZE": "1000", "ENABLE_HNSW": "true",
            "HNSW_M": "16", "HNSW_EF_CONSTRUCTION": "200", "HNSW_EF_SEARCH": "100", "HNSW_METRIC": "l2",
            "AUTO_INDEX_THRESHOLD": "100", "ENABLE_COMPILED_FUNCTIONS": "true",
            # Storage
            "VECTOR_STORE_BASE_PATH": "~/.mlx_vector_db_data/vector_stores",
            "WAL_PATH": "~/.mlx_vector_db_data/wal", "BACKUP_PATH": "~/.mlx_vector_db_data/backups",
            "ENABLE_WAL": "true", "ENABLE_BACKUP": "false", "BACKUP_INTERVAL_HOURS": "6",
            "BACKUP_RETENTION_DAYS": "7", "COMPRESSION_ENABLED": "false", "SYNC_WRITES": "true",
            # Monitoring
            "ENABLE_METRICS": "true", "ENABLE_HEALTH_CHECKS": "true",
            "ENABLE_SYSTEM_METRICS": "true", "ENABLE_TRACING": "false",
            "OTLP_ENDPOINT": "http://localhost:4317", "OTLP_INSECURE": "true",
            "OTEL_CONSOLE_EXPORTER": "false",
            # MLX
            "MLX_WARMUP_ON_STARTUP": "true", "MLX_WARMUP_DIMENSION": "384", "MLX_WARMUP_VECTORS": "1000",
            # Clustering
            "ENABLE_CLUSTERING": "false", "REDIS_URL": "redis://localhost:6379",
            "ENABLE_REPLICATION": "false", "ENABLE_LEADER_ELECTION": "true",
        }

    def _load_env_file(self):
        env_path = Path(os.getenv("ENV_FILE", ".env"))
        if env_path.exists() and env_path.is_file():
            logger.info(f"Lade .env Datei von: {env_path.resolve()}")
            load_dotenv(dotenv_path=env_path, override=True) # override=True damit .env > Defaults
        else:
            logger.debug(f".env Datei nicht gefunden unter: {env_path.resolve()} (oder ENV_FILE nicht gesetzt).")

    def _load_config_from_json_file(self):
        try:
            with self.config_file.open('r') as f: # type: ignore
                file_config = json.load(f)
            # Überschreibe Werte in self.config_data mit denen aus der JSON-Datei
            # Hier wird eine flache Struktur erwartet oder muss geflattet werden.
            # Für Einfachheit nehmen wir an, die JSON-Datei hat flache Schlüssel wie in self.config_data.
            for key, value in file_config.items():
                if key.upper() in self.config_data:
                    self.config_data[key.upper()] = str(value) # Konvertiere zu String für Konsistenz
            logger.info(f"Konfiguration aus JSON-Datei geladen: {self.config_file}")
        except Exception as e:
            logger.error(f"Fehler beim Laden der JSON-Konfigurationsdatei {self.config_file}: {e}")

    def _load_environment_variables(self):
        """Lädt Konfiguration aus Umgebungsvariablen (höchste Präzedenz)."""
        for key in self.config_data.keys(): # Iteriere über alle bekannten Config-Keys
            env_value = os.getenv(key)
            if env_value is not None:
                self.config_data[key] = env_value # Überschreibt vorherige Werte
        logger.debug("Umgebungsvariablen auf Konfigurationswerte angewendet.")
    
    def _get_val(self, key: str, default: Optional[Any] = None, type_converter: Optional[Callable[[str], Any]] = None):
        val_str = self.config_data.get(key) # Sollte immer existieren nach _initialize_config_data_defaults
        if val_str is None: # Fallback falls Key doch fehlt
            if default is not None: return default
            raise KeyError(f"Konfigurationsschlüssel {key} nicht gefunden und kein Standardwert vorhanden.")

        if type_converter:
            try:
                return type_converter(val_str)
            except ValueError:
                logger.warning(f"Ungültiger Wert '{val_str}' für {key}. Verwende Standardwert: {default}")
                if default is not None: return default
                raise ValueError(f"Ungültiger Wert '{val_str}' für {key} und kein Standardwert zum Zurückfallen.")
        return val_str # Als String zurückgeben, wenn kein Konverter

    def _get_bool(self, key: str) -> bool:
        return self._get_val(key, type_converter=lambda x: x.lower() in ("true", "1", "yes", "on", "enabled"))

    def _get_int(self, key: str) -> int:
        return self._get_val(key, type_converter=int)

    def _get_float(self, key: str) -> float:
        return self._get_val(key, type_converter=float)

    def _get_list(self, key: str, separator: str = ",") -> List[str]:
        val_str = self._get_val(key)
        if not val_str: return []
        return [item.strip() for item in val_str.split(separator) if item.strip()]

    def _create_server_config(self) -> ServerConfig:
        return ServerConfig(
            host=self._get_val("HOST"), port=self._get_int("PORT"), workers=self._get_int("WORKERS"),
            log_level=self._get_val("LOG_LEVEL"), app_title=self._get_val("APP_TITLE"),
            app_version=self._get_val("APP_VERSION"), cors_origins=self._get_list("CORS_ORIGINS"),
            max_request_size_mb=self._get_int("MAX_REQUEST_SIZE_MB")
        )

    def _create_security_config(self) -> SecurityConfig:
        return SecurityConfig(
            api_key=self._get_val("API_KEY"), internal_api_key=self._get_val("INTERNAL_API_KEY"),
            enable_auth=self._get_bool("ENABLE_AUTH"), jwt_secret_key=self._get_val("JWT_SECRET_KEY"),
            jwt_algorithm=self._get_val("JWT_ALGORITHM"),
            jwt_access_token_expire_minutes=self._get_int("JWT_ACCESS_TOKEN_EXPIRE_MINUTES"),
            rate_limit_requests=self._get_int("RATE_LIMIT_REQUESTS"),
            rate_limit_window_seconds=self._get_int("RATE_LIMIT_WINDOW_SECONDS"),
            admin_rate_limit_requests=self._get_int("ADMIN_RATE_LIMIT_REQUESTS"),
            admin_rate_limit_window_seconds=self._get_int("ADMIN_RATE_LIMIT_WINDOW_SECONDS"),
            max_file_upload_size_mb=self._get_int("MAX_FILE_UPLOAD_SIZE_MB")
        )

    def _create_performance_config(self) -> PerformanceConfig:
        return PerformanceConfig(
            max_vector_cache_size_gb=self._get_float("MAX_VECTOR_CACHE_SIZE_GB"),
            cache_cleanup_threshold=self._get_float("CACHE_CLEANUP_THRESHOLD"),
            query_result_cache_max_size=self._get_int("QUERY_RESULT_CACHE_MAX_SIZE"),
            query_result_cache_ttl_seconds=self._get_int("QUERY_RESULT_CACHE_TTL_SECONDS"),
            default_batch_size=self._get_int("DEFAULT_BATCH_SIZE"),
            enable_hnsw=self._get_bool("ENABLE_HNSW"), hnsw_m=self._get_int("HNSW_M"),
            hnsw_ef_construction=self._get_int("HNSW_EF_CONSTRUCTION"),
            hnsw_ef_search=self._get_int("HNSW_EF_SEARCH"), hnsw_metric=self._get_val("HNSW_METRIC"),
            auto_index_threshold=self._get_int("AUTO_INDEX_THRESHOLD"),
            enable_compiled_functions=self._get_bool("ENABLE_COMPILED_FUNCTIONS")
        )

    def _create_storage_config(self) -> StorageConfig:
        return StorageConfig(
            base_path=str(Path(self._get_val("VECTOR_STORE_BASE_PATH")).expanduser()),
            wal_path=str(Path(self._get_val("WAL_PATH")).expanduser()),
            backup_path=str(Path(self._get_val("BACKUP_PATH")).expanduser()),
            enable_wal=self._get_bool("ENABLE_WAL"), enable_backup=self._get_bool("ENABLE_BACKUP"),
            backup_interval_hours=self._get_int("BACKUP_INTERVAL_HOURS"),
            backup_retention_days=self._get_int("BACKUP_RETENTION_DAYS"),
            compression_enabled=self._get_bool("COMPRESSION_ENABLED"),
            sync_writes=self._get_bool("SYNC_WRITES")
        )

    def _create_monitoring_config(self) -> MonitoringConfig:
        return MonitoringConfig(
            enable_metrics=self._get_bool("ENABLE_METRICS"),
            enable_health_checks=self._get_bool("ENABLE_HEALTH_CHECKS"),
            enable_system_metrics=self._get_bool("ENABLE_SYSTEM_METRICS"),
            enable_tracing=self._get_bool("ENABLE_TRACING"),
            otlp_endpoint=self._get_val("OTLP_ENDPOINT"),
            otlp_insecure=self._get_bool("OTLP_INSECURE"),
            otel_console_exporter=self._get_bool("OTEL_CONSOLE_EXPORTER")
        )

    def _create_mlx_config(self) -> MLXConfig:
        return MLXConfig(
            warmup_on_startup=self._get_bool("MLX_WARMUP_ON_STARTUP"),
            warmup_dimension=self._get_int("MLX_WARMUP_DIMENSION"),
            warmup_vectors=self._get_int("MLX_WARMUP_VECTORS")
        )

    def _create_clustering_config(self) -> ClusteringConfig: # NEU
        return ClusteringConfig(
            enable_clustering=self._get_bool("ENABLE_CLUSTERING"),
            redis_url=self._get_val("REDIS_URL"),
            enable_replication=self._get_bool("ENABLE_REPLICATION"),
            enable_leader_election=self._get_bool("ENABLE_LEADER_ELECTION")
        )

    def _validate_config(self):
        """Validiert geladene Konfigurationswerte."""
        errors = []
        if not (1 <= self.server.port <= 65535):
            errors.append(f"Ungültiger Server-Port: {self.server.port}")
        
        if self.security.enable_auth:
            if not self.security.api_key or self.security.api_key == "dev-key-please-change":
                logger.warning("Authentifizierung ist aktiviert, aber API_KEY ist nicht gesetzt oder unsicher. (VECTOR_DB_API_KEY)")
            if not self.security.jwt_secret_key or self.security.jwt_secret_key == "a-very-secret-key-for-dev-change-in-prod":
                logger.warning("Authentifizierung ist aktiviert, aber JWT_SECRET_KEY ist nicht gesetzt oder unsicher.")
        
        if self.performance.max_vector_cache_size_gb <= 0:
            errors.append("MAX_VECTOR_CACHE_SIZE_GB muss positiv sein.")
        if self.performance.default_batch_size <= 0:
            errors.append("DEFAULT_BATCH_SIZE muss positiv sein.")
        
        try: Path(self.storage.base_path).mkdir(parents=True, exist_ok=True)
        except Exception as e: errors.append(f"Speicherpfad {self.storage.base_path} kann nicht erstellt werden: {e}")
        if self.storage.enable_wal:
            try: Path(self.storage.wal_path).mkdir(parents=True, exist_ok=True)
            except Exception as e: errors.append(f"WAL-Pfad {self.storage.wal_path} kann nicht erstellt werden: {e}")
        if self.storage.enable_backup:
            try: Path(self.storage.backup_path).mkdir(parents=True, exist_ok=True)
            except Exception as e: errors.append(f"Backup-Pfad {self.storage.backup_path} kann nicht erstellt werden: {e}")

        if errors:
            error_msg = "Konfigurationsvalidierung fehlgeschlagen:\n" + "\n".join(f"  - {err}" for err in errors)
            logger.error(error_msg)
            # In einer Produktionsumgebung könnte man hier einen Fehler werfen, um den Start zu verhindern:
            # raise ValueError(error_msg)
            logger.warning("Server startet trotz Konfigurationswarnungen.")


    def get_all_config_as_dict(self) -> Dict[str, Any]:
        """Gibt die gesamte Konfiguration als Dictionary zurück (ohne sensible Daten)."""
        safe_security_config = {k: v for k, v in self.security.__dict__.items() if k not in ["api_key", "internal_api_key", "jwt_secret_key"]}
        return {
            "server": self.server.__dict__,
            "security_safe": safe_security_config, # Gibt nur "sichere" Security-Config zurück
            "performance": self.performance.__dict__,
            "storage": self.storage.__dict__,
            "monitoring": self.monitoring.__dict__,
            "mlx": self.mlx.__dict__,
            "clustering": self.clustering.__dict__,
        }

# Globale Konfigurationsinstanz
_config_manager_instance: Optional[ConfigManager] = None
_config_lock = threading.Lock()

def get_config() -> ConfigManager:
    """Gibt die globale Konfigurationsinstanz zurück (Singleton)."""
    global _config_manager_instance
    if _config_manager_instance is None:
        with _config_lock:
            if _config_manager_instance is None: # Double-check locking
                _config_manager_instance = ConfigManager()
    return _config_manager_instance

def reload_config() -> ConfigManager:
    """Lädt die Konfiguration neu."""
    global _config_manager_instance
    with _config_lock:
        logger.info("Lade Konfiguration neu...")
        _config_manager_instance = ConfigManager()
    return _config_manager_instance