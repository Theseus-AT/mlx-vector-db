# Neue Datei: core/logging_config.py
# Konfiguriert strukturiertes Logging für die gesamte Anwendung.

# MLX Specificity: Log-Einträge können spezifische MLX-Kontextinformationen
#                  enthalten (z.B. verwendetes Device, Array-Shapes bei Fehlern).
# LLM Anbindung: Detaillierte Logs sind wichtig, um den Datenfluss und
#                Performance-Engpässe bei LLM-Interaktionen nachzuvollziehen.

import json
import logging
import logging.config # Für komplexere Konfiguration
from datetime import datetime
from typing import Any, Dict, Optional
import sys
import os

# Konfiguration für den JSON Formatter (kann auch aus einer Datei geladen werden)
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "core.logging_config.JsonFormatter", # Referenziert die Klasse unten
            "format_keys": { # Optionale Anpassung der Schlüsselnamen im JSON-Output
                "level": "levelname",
                "message": "message",
                "timestamp": "asctime",
                "logger": "name",
                "module": "module",
                "function": "funcName",
                "line": "lineno",
                "thread_name": "threadName"
            }
        },
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console_json": {
            "class": "logging.StreamHandler",
            "level": "DEBUG", # Konsolen-Handler-Level
            "formatter": "json",
            "stream": "ext://sys.stdout",
        },
        "console_simple": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        }
        # Hier könnten weitere Handler hinzugefügt werden, z.B. für Datei-Logging
        # "file_json": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "level": "INFO",
        #     "formatter": "json",
        #     "filename": "mlx_vector_db.log.jsonl",
        #     "maxBytes": 10485760,  # 10MB
        #     "backupCount": 5,
        #     "encoding": "utf8"
        # }
    },
    "root": { # Root-Logger-Konfiguration
        "level": "INFO", # Globaler Standard-Log-Level, kann durch ENV überschrieben werden
        "handlers": ["console_json"], # Standardmäßig JSON auf Konsole
    },
    "loggers": { # Spezifische Logger-Konfigurationen (optional)
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console_simple"], # Uvicorn-Fehler ggf. im einfachen Format
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",
            "handlers": ["console_simple"], # Uvicorn-Access-Logs ggf. im einfachen Format
            "propagate": False,
        },
        "mlx_vector_db": { # Logger für Ihre Anwendung
            "level": "DEBUG", # Standardmäßig DEBUG für Ihre App-Logs
            "handlers": ["console_json"], # Oder ["console_json", "file_json"]
            "propagate": False, # Verhindert, dass Logs auch vom Root-Logger behandelt werden
        },
        # Logger für spezifische Module können hier feiner eingestellt werden
        "mlx_vector_db.performance": {
            "level": "INFO", # Performance-Logs z.B. nur ab INFO
            "propagate": True, # Damit sie vom mlx_vector_db Logger mitbehandelt werden
        }
    }
}


class JsonFormatter(logging.Formatter):
    """
    Benutzerdefinierter Formatter, um Log-Records als JSON-String auszugeben.
    """
    def __init__(self, fmt: Optional[Dict[str, str]] = None, format_keys: Optional[Dict[str,str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.format_keys = format_keys or {}

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {}
        
        # Standardfelder, die immer vorhanden sein sollten
        default_fields = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(), # Formatiert die Nachricht mit Argumenten
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'thread_name': record.threadName,
            'process_id': record.process,
            # 'filename': record.filename, # Kann redundant zu module/function sein
            # 'pathname': record.pathname,
        }

        for key, value in default_fields.items():
            json_key = self.format_keys.get(key, key)
            log_object[json_key] = value

        # Hinzufügen von zusätzlichen Feldern aus record.args oder record.__dict__
        # die nicht Standard sind (z.B. durch logger.info("msg", extra={...}))
        if hasattr(record, 'custom_extra_fields') and isinstance(record.custom_extra_fields, dict):
            for key, value in record.custom_extra_fields.items():
                log_object[key] = value
        
        # Behandlung von ExcInfo für Stack Traces
        if record.exc_info:
            if not record.exc_text: # Wenn kein vorformatierter Text existiert
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            log_object['exception'] = record.exc_text
        if record.stack_info:
            log_object['stack_info'] = self.formatStack(record.stack_info)
            
        return json.dumps(log_object, ensure_ascii=False)

def setup_logging():
    """Konfiguriert das Logging-System basierend auf LOGGING_CONFIG."""
    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Überschreibe Root-Level und App-Logger-Level basierend auf Umgebungsvariable
    LOGGING_CONFIG["root"]["level"] = log_level_env
    if "loggers" in LOGGING_CONFIG and "mlx_vector_db" in LOGGING_CONFIG["loggers"]:
        LOGGING_CONFIG["loggers"]["mlx_vector_db"]["level"] = os.getenv("APP_LOG_LEVEL", log_level_env).upper()

    # Handler basierend auf Umgebungsvariable (z.B. für lokale Entwicklung vs. Produktion)
    log_handler_env = os.getenv("LOG_HANDLER", "json") # "json" or "simple"
    if log_handler_env == "simple":
        LOGGING_CONFIG["root"]["handlers"] = ["console_simple"]
        if "loggers" in LOGGING_CONFIG and "mlx_vector_db" in LOGGING_CONFIG["loggers"]:
            LOGGING_CONFIG["loggers"]["mlx_vector_db"]["handlers"] = ["console_simple"]
    else: # Default to json
        LOGGING_CONFIG["root"]["handlers"] = ["console_json"]
        if "loggers" in LOGGING_CONFIG and "mlx_vector_db" in LOGGING_CONFIG["loggers"]:
            LOGGING_CONFIG["loggers"]["mlx_vector_db"]["handlers"] = ["console_json"]


    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger("mlx_vector_db.core.logging_config")
    logger.info(f"Structured logging configured. Root level: {LOGGING_CONFIG['root']['level']}. Handler: {LOGGING_CONFIG['root']['handlers']}.")

# Um Logging Aufrufe mit `extra` zu vereinfachen:
def get_logger_with_extra(name: str) -> logging.LoggerAdapter:
    """
    Gibt einen LoggerAdapter zurück, der das einfache Hinzufügen von 'extra'-Feldern ermöglicht.
    Beispiel: logger.info("Eine Nachricht", extra_fields={"key": "value"})
    """
    class CustomAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            if 'extra_fields' in kwargs:
                # Speichere die extra_fields im LogRecord unter einem bekannten Namen
                # JsonFormatter kann dann darauf zugreifen.
                if not hasattr(kwargs.get("extra", {}), 'custom_extra_fields'): # Verhindert Überschreiben falls 'extra' schon da
                    kwargs["extra"] = kwargs.get("extra", {})
                    kwargs["extra"]['custom_extra_fields'] = kwargs.pop('extra_fields')
            return msg, kwargs

    return CustomAdapter(logging.getLogger(name), {})


# Die Funktion `log_request` aus dem Plan kann als FastAPI Middleware implementiert werden:
# (Diese gehört eher in main.py oder ein Middleware-Modul)
# async def structured_request_logging_middleware(request: Request, call_next):
#     request_id = secrets.token_hex(8) # Eindeutige Request ID
#     logger = get_logger_with_extra("mlx_vector_db.api.request")
#
#     start_time = time.time()
#     response = await call_next(request)
#     duration_ms = (time.time() - start_time) * 1000
#
#     client_host = request.client.host
#     client_port = request.client.port
#
#     log_details = {
#         'event_type': 'http_request',
#         'request_id': request_id,
#         'http_method': request.method,
#         'http_path': request.url.path,
#         'http_query_params': str(request.query_params),
#         'http_status_code': response.status_code,
#         'duration_ms': round(duration_ms, 2),
#         'client_address': f"{client_host}:{client_port}",
#         'user_agent': request.headers.get("user-agent", "N/A"),
#         # User-ID, falls verfügbar (nach Authentifizierung)
#         # 'user_id': request.state.user.id if hasattr(request.state, "user") and request.state.user else None
#     }
#     logger.info(f"{request.method} {request.url.path} - {response.status_code}", extra_fields=log_details)
#     return response