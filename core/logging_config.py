# core/logging_config.py
# (Ihre bestehende logging_config.py ist bereits sehr gut)
# Die folgende Anpassung stellt sicher, dass die Middleware-Funktion klar definiert ist.

import json
import logging
import logging.config
from datetime import datetime
from typing import Any, Dict, Optional
import sys
import os
import time # Für Middleware
import secrets # Für Middleware

# --- Ihre bestehende LOGGING_CONFIG, JsonFormatter, setup_logging, get_logger_with_extra ---
# (Fügen Sie hier Ihren Code aus der hochgeladenen logging_config.py ein)
# Stellen Sie sicher, dass `JsonFormatter` korrekt referenziert wird, falls der Pfad
# "core.logging_config.JsonFormatter" in LOGGING_CONFIG ist, muss die Datei so liegen.

# Globale LOGGING_CONFIG Konstante (wie in Ihrer Datei)
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "core.logging_config.JsonFormatter", # Pfad anpassen, falls nötig
            "format_keys": {
                "level": "levelname", "message": "message", "timestamp": "asctime",
                "logger": "name", "module": "module", "function": "funcName",
                "line": "lineno", "thread_name": "threadName"
            }
        },
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console_json": {
            "class": "logging.StreamHandler", "level": "DEBUG",
            "formatter": "json", "stream": "ext://sys.stdout",
        },
        "console_simple": {
            "class": "logging.StreamHandler", "level": "INFO",
            "formatter": "simple", "stream": "ext://sys.stdout",
        }
        # Optional: File Handler
    },
    "root": {"level": "INFO", "handlers": ["console_json"]},
    "loggers": {
        "uvicorn.error": {"level": "INFO", "handlers": ["console_simple"], "propagate": False},
        "uvicorn.access": {"level": "INFO", "handlers": ["console_simple"], "propagate": False},
        "mlx_vector_db": {"level": "DEBUG", "handlers": ["console_json"], "propagate": False},
        "mlx_vector_db.performance": {"level": "INFO", "propagate": True}
    }
}

class JsonFormatter(logging.Formatter):
    def __init__(self, fmt: Optional[Dict[str, str]] = None, format_keys: Optional[Dict[str,str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.format_keys = format_keys or {}

    def format(self, record: logging.LogRecord) -> str:
        log_object: Dict[str, Any] = {}
        default_fields = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname, 'logger': record.name, 'message': record.getMessage(),
            'module': record.module, 'function': record.funcName, 'line': record.lineno,
            'thread_id': record.thread, 'thread_name': record.threadName, 'process_id': record.process,
        }
        for key, value in default_fields.items():
            json_key = self.format_keys.get(key, key)
            log_object[json_key] = value
        if hasattr(record, 'custom_extra_fields') and isinstance(record.custom_extra_fields, dict):
            for key, value in record.custom_extra_fields.items():
                log_object[key] = value
        if record.exc_info:
            if not record.exc_text: record.exc_text = self.formatException(record.exc_info)
        if record.exc_text: log_object['exception'] = record.exc_text
        if record.stack_info: log_object['stack_info'] = self.formatStack(record.stack_info)
        return json.dumps(log_object, ensure_ascii=False)

_logging_initialized = False
_logging_lock = threading.Lock()

def setup_logging():
    global _logging_initialized
    with _logging_lock:
        if _logging_initialized:
            # logging.getLogger("mlx_vector_db.core.logging_config").debug("Logging bereits initialisiert.")
            return

        log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
        app_log_level_env = os.getenv("APP_LOG_LEVEL", log_level_env).upper()
        log_handler_env = os.getenv("LOG_HANDLER", "json")

        current_config = json.loads(json.dumps(LOGGING_CONFIG)) # Tiefe Kopie

        current_config["root"]["level"] = log_level_env
        if "loggers" in current_config and "mlx_vector_db" in current_config["loggers"]:
            current_config["loggers"]["mlx_vector_db"]["level"] = app_log_level_env

        if log_handler_env == "simple":
            current_config["root"]["handlers"] = ["console_simple"]
            if "loggers" in current_config and "mlx_vector_db" in current_config["loggers"]:
                current_config["loggers"]["mlx_vector_db"]["handlers"] = ["console_simple"]
        else: # Default to json
            current_config["root"]["handlers"] = ["console_json"]
            if "loggers" in current_config and "mlx_vector_db" in current_config["loggers"]:
                current_config["loggers"]["mlx_vector_db"]["handlers"] = ["console_json"]
        
        logging.config.dictConfig(current_config)
        _logging_initialized = True
        logger = logging.getLogger("mlx_vector_db.core.logging_config") # Logger erst nach config holen
        logger.info(f"Strukturiertes Logging konfiguriert. Root-Level: {current_config['root']['level']}. App-Level: {app_log_level_env}. Handler: {current_config['root']['handlers']}.")


def get_logger_with_extra(name: str) -> logging.LoggerAdapter:
    class CustomAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            if 'extra_fields' in kwargs:
                if not hasattr(kwargs.get("extra", {}), 'custom_extra_fields'):
                    kwargs["extra"] = kwargs.get("extra", {})
                    kwargs["extra"]['custom_extra_fields'] = kwargs.pop('extra_fields')
            return msg, kwargs
    return CustomAdapter(logging.getLogger(name), {})


# --- Middleware für strukturiertes Request Logging ---
# Diese Funktion wird von main.py importiert und als Middleware verwendet.
# Sie benötigt `Request` von FastAPI und `time`, `secrets`.
async def structured_request_logging_middleware(request: Any, call_next: Any) -> Any:
    """
    FastAPI Middleware für strukturiertes Logging von HTTP Anfragen.
    Muss in main.py der FastAPI App hinzugefügt werden.
    Die Typ-Hints für Request und CallNext sind Any, um FastAPI nicht hier importieren zu müssen.
    """
    request_id = secrets.token_hex(8)
    logger_mw = get_logger_with_extra("mlx_vector_db.api.request_middleware") # Eigener Logger-Name

    start_time = time.time()
    
    user_id_from_token = "anonymous"
    # Versuch, User-ID aus request.state zu holen (wird von Auth-Middleware gesetzt)
    if hasattr(request.state, "current_user_payload") and request.state.current_user_payload:
        user_id_from_token = request.state.current_user_payload.get("sub", "unknown_jwt_user")
    elif request.headers.get("x-api-key"): # Fallback für API-Key
        user_id_from_token = "api_key_user"


    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response # Wichtig: Response zurückgeben
    except Exception as e:
        status_code = 500 # Interner Serverfehler
        logger_mw.error(
            f"Unhandled exception during request: {request.method} {request.url.path}",
            exc_info=True, # Fügt Stacktrace hinzu
            extra_fields={ # custom_extra_fields werden hier verwendet
                'event_type': 'http_request_unhandled_exception',
                'request_id': request_id,
                'http_method': request.method,
                'http_path': str(request.url.path), # Sicherstellen, dass es ein String ist
                'client_address': f"{request.client.host}:{request.client.port}" if request.client else "N/A",
                'user_id': user_id_from_token,
                'error': str(e),
                'error_type': type(e).__name__
            }
        )
        # Fehler muss weitergeworfen werden, damit FastAPI ihn korrekt behandelt
        # oder hier eine JSONResponse zurückgegeben werden.
        # Wenn eine globale Fehlerbehandlung in main.py existiert, wird diese greifen.
        raise e 
    finally:
        duration_ms = (time.time() - start_time) * 1000
        log_details = {
            'event_type': 'http_request_completed',
            'request_id': request_id,
            'http_method': request.method,
            'http_path': str(request.url.path),
            'http_query_params': str(request.query_params),
            'http_status_code': status_code if response else 500, # Status Code aus Response oder 500 bei Exception
            'duration_ms': round(duration_ms, 2),
            'client_address': f"{request.client.host}:{request.client.port}" if request.client else "N/A",
            'user_agent': request.headers.get("user-agent", "N/A"),
            'user_id': user_id_from_token
        }
        if response and status_code < 400 :
             logger_mw.info(f"{request.method} {request.url.path} - {status_code if response else 'N/A'}", extra_fields=log_details)
        elif response and status_code >=400 : # Fehlerhafte Anfragen auch loggen, aber ggf. mit anderem Level
             logger_mw.warning(f"{request.method} {request.url.path} - {status_code if response else 'N/A'}", extra_fields=log_details)
        # Wenn response None ist (Exception wurde vorher geworfen und gefangen), wurde schon geloggt.