# service/production_integration.py (FINALE KORRIGIERTE VERSION)

"""
Production-Ready MLX Vector Database - Vollständig Integriertes System
Sprint 1 Deliverable: MLX Core Optimierung + Memory Management + Error Handling
"""

import mlx.core as mx
import numpy as np
import asyncio
import logging
import time
from pathlib import Path
# KORREKTUR: Fehlende Imports aus dem 'typing'-Modul hinzugefügt
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

# Import unserer optimierten Komponenten mit relativem Pfad
from .optimized_vector_store import MLXVectorStore, MLXVectorStoreConfig
from .service_handling import MLXErrorHandler, with_error_handling, with_circuit_breaker, DegradationLevel

logger = logging.getLogger("mlx_production_system")

# =================== PRODUCTION VECTOR STORE MANAGER ===================

class ProductionVectorStoreManager:
    """Production-Ready Store Manager mit allen Optimierungen"""
    
    def __init__(self, base_path: str = "~/.mlx_vector_stores_production"):
        self.base_path = Path(base_path).expanduser()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self._stores: Dict[str, MLXVectorStore] = {}
        self._store_configs: Dict[str, MLXVectorStoreConfig] = {}
        self._lock = threading.RLock()
        self.error_handler = MLXErrorHandler()
        self._performance_metrics = {
            'total_requests': 0, 'successful_requests': 0, 'failed_requests': 0,
            'avg_response_time_ms': 0.0, 'current_qps': 0.0, 'uptime_start': time.time()
        }
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="store_manager")
        logger.info("🚀 Production Vector Store Manager initialized")
        self._load_existing_stores()

    def _load_existing_stores(self):
        logger.info(f"Lade existierende Stores von: {self.base_path}")
        # ... (restliche Logik bleibt unverändert)

    def _get_store_key(self, user_id: str, model_id: str) -> str:
        return f"{user_id}_{model_id}"

    def _get_store_path(self, user_id: str, model_id: str) -> Path:
        return self.base_path / f"{user_id}_{model_id}"

    async def create_store(self, user_id: str, model_id: str, 
                           config: Optional[MLXVectorStoreConfig] = None) -> Dict[str, Any]:
        with self._lock:
            store_key = self._get_store_key(user_id, model_id)
            if store_key in self._stores:
                raise ValueError(f"Store already exists: {user_id}/{model_id}")
            if config is None:
                config = MLXVectorStoreConfig(jit_compile=True, auto_recovery=True)
            
            store_path = self._get_store_path(user_id, model_id)
            store = MLXVectorStore(str(store_path), config)
            
            self._stores[store_key] = store
            self._store_configs[store_key] = config
            logger.info(f"✅ Created optimized store: {user_id}/{model_id}")
            return {'success': True, 'user_id': user_id, 'model_id': model_id}

    async def get_store(self, user_id: str, model_id: str) -> MLXVectorStore:
        store_key = self._get_store_key(user_id, model_id)
        if store_key not in self._stores:
            raise ValueError(f"Store not found: {user_id}/{model_id}")
        return self._stores[store_key]

    async def delete_store(self, user_id: str, model_id: str, force: bool = False) -> Dict[str, Any]:
        # ... (Logik bleibt unverändert)
        store_key = self._get_store_key(user_id, model_id)
        with self._lock:
            if store_key not in self._stores:
                 raise ValueError(f"Store not found: {user_id}/{model_id}")
            #... rest of the function
            store = self._stores.pop(store_key)
            self._store_configs.pop(store_key, None)
            
            store.clear()
            import shutil
            shutil.rmtree(store.store_path)
            
            logger.info(f"🗑️ Deleted store: {user_id}/{model_id}")
            return {'success': True}

    async def add_vectors(self, user_id: str, model_id: str,
                          vectors: Union[np.ndarray, List[List[float]]], 
                          metadata: List[Dict]) -> Dict[str, Any]:
        store = await self.get_store(user_id, model_id)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: store.add_vectors(vectors, metadata)
        )
        return result

    async def query_vectors(self, user_id: str, model_id: str,
                            query_vector: Union[np.ndarray, List[float]], 
                            k: int = 10,
                            filter_metadata: Optional[Dict] = None):
        store = await self.get_store(user_id, model_id)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            lambda: store.query(query_vector, k=k, filter_metadata=filter_metadata)
        )
        return result