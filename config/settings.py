# config/settings.py
"""
Advanced configuration management for MLX Vector Database
Supports environment variables, config files, and dynamic updates
"""
import os
import json
from typing import Any, Dict, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
import logging
from dotenv import load_dotenv

logger = logging.getLogger("mlx_vector_db.config")

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key: str = ""
    enable_auth: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600
    admin_rate_limit_requests: int = 100
    admin_rate_limit_window: int = 3600
    max_file_upload_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class PerformanceConfig:
    """Performance configuration"""
    max_vector_cache_size_gb: float = 4.0
    cache_cleanup_threshold: float = 0.9
    default_batch_size: int = 1000
    hnsw_max_connections: int = 16
    hnsw_max_connections_layer0: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    enable_hnsw: bool = True
    enable_cache: bool = True
    enable_compiled_functions: bool = True
    auto_index_threshold: int = 1000

@dataclass
class StorageConfig:
    """Storage configuration"""
    base_path: str = "~/.team_mind_data/vector_stores"
    enable_backup: bool = False
    backup_retention_days: int = 30
    compression_enabled: bool = False
    sync_writes: bool = True

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_system_metrics: bool = True
    system_metrics_interval: int = 10
    prometheus_enabled: bool = False

@dataclass
class MLXConfig:
    """MLX framework configuration"""
    default_dtype: str = "float32"
    warmup_on_startup: bool = True
    warmup_dimension: int = 384
    warmup_vectors: int = 1000
    compile_functions: bool = True
    gpu_enabled: bool = True

class ConfigManager:
    """
    Centralized configuration management
    Loads from environment variables, .env file, and optional config files
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file
        self.config_data = {}
        
        # Load configuration in order of precedence
        self._load_defaults()
        self._load_env_file()
        if config_file and config_file.exists():
            self._load_config_file()
        self._load_environment_variables()
        
        # Create configuration objects
        self.server = self._create_server_config()
        self.security = self._create_security_config()
        self.performance = self._create_performance_config()
        self.storage = self._create_storage_config()
        self.monitoring = self._create_monitoring_config()
        self.mlx = self._create_mlx_config()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("Configuration loaded successfully")
    
    def _load_defaults(self):
        """Load default configuration values"""
        self.config_data = {
            # Server defaults
            "HOST": "0.0.0.0",
            "PORT": "8000",
            "LOG_LEVEL": "INFO",
            "CORS_ORIGINS": "*",
            
            # Security defaults
            "ENABLE_AUTH": "true",
            "RATE_LIMIT_REQUESTS": "1000",
            "RATE_LIMIT_WINDOW": "3600",
            "MAX_FILE_UPLOAD_SIZE_MB": "100",
            
            # Performance defaults
            "MAX_VECTOR_CACHE_SIZE_GB": "4.0",
            "DEFAULT_BATCH_SIZE": "1000",
            "ENABLE_HNSW": "true",
            "ENABLE_CACHE": "true",
            "HNSW_MAX_CONNECTIONS": "16",
            "HNSW_EF_CONSTRUCTION": "200",
            "AUTO_INDEX_THRESHOLD": "1000",
            
            # Storage defaults
            "VECTOR_STORE_BASE_PATH": "~/.team_mind_data/vector_stores",
            "ENABLE_BACKUP": "false",
            "BACKUP_RETENTION_DAYS": "30",
            
            # Monitoring defaults
            "ENABLE_METRICS": "true",
            "ENABLE_HEALTH_CHECKS": "true",
            "HEALTH_CHECK_INTERVAL": "30",
            
            # MLX defaults
            "MLX_DEFAULT_DTYPE": "float32",
            "MLX_WARMUP_ON_STARTUP": "true",
            "MLX_WARMUP_DIMENSION": "384",
            "MLX_COMPILE_FUNCTIONS": "true",
        }
    
    def _load_env_file(self):
        """Load from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            logger.debug(f"Loaded .env file: {env_file}")
    
    def _load_config_file(self):
        """Load from JSON config file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
            
            # Flatten nested config for easier access
            self._flatten_dict(file_config, self.config_data)
            logger.info(f"Loaded config file: {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file {self.config_file}: {e}")
    
    def _load_environment_variables(self):
        """Load from environment variables (highest precedence)"""
        for key in self.config_data.keys():
            env_value = os.getenv(key)
            if env_value is not None:
                self.config_data[key] = env_value
    
    def _flatten_dict(self, nested_dict: Dict, target_dict: Dict, prefix: str = ""):
        """Flatten nested dictionary for config loading"""
        for key, value in nested_dict.items():
            flat_key = f"{prefix}_{key}".upper() if prefix else key.upper()
            
            if isinstance(value, dict):
                self._flatten_dict(value, target_dict, flat_key)
            else:
                target_dict[flat_key] = str(value)
    
    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from config"""
        value = self.config_data.get(key, str(default)).lower()
        return value in ("true", "1", "yes", "on", "enabled")
    
    def _get_int(self, key: str, default: int = 0) -> int:
        """Get integer value from config"""
        try:
            return int(self.config_data.get(key, default))
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value for {key}, using default: {default}")
            return default
    
    def _get_float(self, key: str, default: float = 0.0) -> float:
        """Get float value from config"""
        try:
            return float(self.config_data.get(key, default))
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value for {key}, using default: {default}")
            return default
    
    def _get_list(self, key: str, default: List[str] = None, separator: str = ",") -> List[str]:
        """Get list value from config"""
        if default is None:
            default = []
        
        value = self.config_data.get(key, "")
        if not value:
            return default
        
        if isinstance(value, str):
            return [item.strip() for item in value.split(separator) if item.strip()]
        elif isinstance(value, list):
            return value
        else:
            return default
    
    def _create_server_config(self) -> ServerConfig:
        """Create server configuration"""
        return ServerConfig(
            host=self.config_data.get("HOST", "0.0.0.0"),
            port=self._get_int("PORT", 8000),
            workers=self._get_int("WORKERS", 1),
            log_level=self.config_data.get("LOG_LEVEL", "INFO"),
            cors_origins=self._get_list("CORS_ORIGINS", ["*"]),
            max_request_size=self._get_int("MAX_REQUEST_SIZE_MB", 100) * 1024 * 1024
        )
    
    def _create_security_config(self) -> SecurityConfig:
        """Create security configuration"""
        api_key = self.config_data.get("VECTOR_DB_API_KEY", "")
        if not api_key or api_key == "dev-key-change-in-production":
            logger.warning("Using default/weak API key. Set VECTOR_DB_API_KEY in production!")
        
        return SecurityConfig(
            api_key=api_key,
            enable_auth=self._get_bool("ENABLE_AUTH", True),
            rate_limit_requests=self._get_int("RATE_LIMIT_REQUESTS", 1000),
            rate_limit_window=self._get_int("RATE_LIMIT_WINDOW", 3600),
            admin_rate_limit_requests=self._get_int("ADMIN_RATE_LIMIT_REQUESTS", 100),
            admin_rate_limit_window=self._get_int("ADMIN_RATE_LIMIT_WINDOW", 3600),
            max_file_upload_size=self._get_int("MAX_FILE_UPLOAD_SIZE_MB", 100) * 1024 * 1024
        )
    
    def _create_performance_config(self) -> PerformanceConfig:
        """Create performance configuration"""
        return PerformanceConfig(
            max_vector_cache_size_gb=self._get_float("MAX_VECTOR_CACHE_SIZE_GB", 4.0),
            cache_cleanup_threshold=self._get_float("CACHE_CLEANUP_THRESHOLD", 0.9),
            default_batch_size=self._get_int("DEFAULT_BATCH_SIZE", 1000),
            hnsw_max_connections=self._get_int("HNSW_MAX_CONNECTIONS", 16),
            hnsw_max_connections_layer0=self._get_int("HNSW_MAX_CONNECTIONS_LAYER0", 32),
            hnsw_ef_construction=self._get_int("HNSW_EF_CONSTRUCTION", 200),
            hnsw_ef_search=self._get_int("HNSW_EF_SEARCH", 50),
            enable_hnsw=self._get_bool("ENABLE_HNSW", True),
            enable_cache=self._get_bool("ENABLE_CACHE", True),
            enable_compiled_functions=self._get_bool("ENABLE_COMPILED_FUNCTIONS", True),
            auto_index_threshold=self._get_int("AUTO_INDEX_THRESHOLD", 1000)
        )
    
    def _create_storage_config(self) -> StorageConfig:
        """Create storage configuration"""
        base_path = self.config_data.get("VECTOR_STORE_BASE_PATH", "~/.team_mind_data/vector_stores")
        base_path = str(Path(base_path).expanduser())
        
        return StorageConfig(
            base_path=base_path,
            enable_backup=self._get_bool("ENABLE_BACKUP", False),
            backup_retention_days=self._get_int("BACKUP_RETENTION_DAYS", 30),
            compression_enabled=self._get_bool("COMPRESSION_ENABLED", False),
            sync_writes=self._get_bool("SYNC_WRITES", True)
        )
    
    def _create_monitoring_config(self) -> MonitoringConfig:
        """Create monitoring configuration"""
        return MonitoringConfig(
            enable_metrics=self._get_bool("ENABLE_METRICS", True),
            metrics_port=self._get_int("METRICS_PORT", 9090),
            enable_health_checks=self._get_bool("ENABLE_HEALTH_CHECKS", True),
            health_check_interval=self._get_int("HEALTH_CHECK_INTERVAL", 30),
            enable_system_metrics=self._get_bool("ENABLE_SYSTEM_METRICS", True),
            system_metrics_interval=self._get_int("SYSTEM_METRICS_INTERVAL", 10),
            prometheus_enabled=self._get_bool("PROMETHEUS_ENABLED", False)
        )
    
    def _create_mlx_config(self) -> MLXConfig:
        """Create MLX configuration"""
        return MLXConfig(
            default_dtype=self.config_data.get("MLX_DEFAULT_DTYPE", "float32"),
            warmup_on_startup=self._get_bool("MLX_WARMUP_ON_STARTUP", True),
            warmup_dimension=self._get_int("MLX_WARMUP_DIMENSION", 384),
            warmup_vectors=self._get_int("MLX_WARMUP_VECTORS", 1000),
            compile_functions=self._get_bool("MLX_COMPILE_FUNCTIONS", True),
            gpu_enabled=self._get_bool("MLX_GPU_ENABLED", True)
        )
    
    def _validate_config(self):
        """Validate configuration values"""
        errors = []
        
        # Validate server config
        if not (1 <= self.server.port <= 65535):
            errors.append(f"Invalid port: {self.server.port}")
        
        # Validate security config - RELAXED FOR DEVELOPMENT
        if self.security.enable_auth and not self.security.api_key:
            # In development, automatically disable auth if no API key
            logger.warning("No API key provided, disabling authentication for development")
            self.security.enable_auth = False
        
        # Validate performance config
        if self.performance.max_vector_cache_size_gb <= 0:
            errors.append("Cache size must be positive")
        
        if self.performance.default_batch_size <= 0:
            errors.append("Batch size must be positive")
        
        # Validate storage config
        try:
            Path(self.storage.base_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create storage path {self.storage.base_path}: {e}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            "server": self.server.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() if k != "api_key"},  # Hide API key
            "performance": self.performance.__dict__,
            "storage": self.storage.__dict__,
            "monitoring": self.monitoring.__dict__,
            "mlx": self.mlx.__dict__
        }
    
    def save_config(self, file_path: Path):
        """Save current configuration to file"""
        config_dict = self.get_all_config()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            raise
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration dynamically"""
        # This is a simplified version - in production you'd want more sophisticated updating
        logger.warning("Dynamic configuration updates not fully implemented")
        # You could implement hot-reloading of specific config sections here

# Global configuration instance
config: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    """Get global configuration instance"""
    global config
    if config is None:
        config_file = Path(os.getenv("CONFIG_FILE", "")) if os.getenv("CONFIG_FILE") else None
        config = ConfigManager(config_file)
    return config

def reload_config():
    """Reload configuration"""
    global config
    config = None
    return get_config()

# Environment-specific configuration presets
class ConfigPresets:
    """Predefined configuration presets for different environments"""
    
    @staticmethod
    def development() -> Dict[str, str]:
        """Development environment preset"""
        return {
            "LOG_LEVEL": "DEBUG",
            "ENABLE_AUTH": "false",
            "ENABLE_METRICS": "true",
            "MAX_VECTOR_CACHE_SIZE_GB": "1.0",
            "MLX_WARMUP_ON_STARTUP": "false",
            "RATE_LIMIT_REQUESTS": "10000"  # Higher limits for dev
        }
    
    @staticmethod
    def production() -> Dict[str, str]:
        """Production environment preset"""
        return {
            "LOG_LEVEL": "INFO",
            "ENABLE_AUTH": "true",
            "ENABLE_METRICS": "true",
            "ENABLE_HEALTH_CHECKS": "true",
            "PROMETHEUS_ENABLED": "true",
            "MAX_VECTOR_CACHE_SIZE_GB": "8.0",
            "MLX_WARMUP_ON_STARTUP": "true",
            "ENABLE_BACKUP": "true",
            "CORS_ORIGINS": ""  # No CORS in production
        }
    
    @staticmethod
    def testing() -> Dict[str, str]:
        """Testing environment preset"""
        return {
            "LOG_LEVEL": "WARNING",
            "ENABLE_AUTH": "false",
            "ENABLE_METRICS": "false",
            "MAX_VECTOR_CACHE_SIZE_GB": "0.5",
            "MLX_WARMUP_ON_STARTUP": "false",
            "ENABLE_HNSW": "false",  # Faster tests
            "DEFAULT_BATCH_SIZE": "100"
        }