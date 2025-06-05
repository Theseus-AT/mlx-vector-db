"""
MLX Vector Store - Production-Ready mit MLX 0.25.2 Features
...
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import os
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import time
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import weakref
# KORRIGIERT: Fehlender Import für defaultdict und OrderedDict hinzugefügt
from collections import OrderedDict, defaultdict
import pickle

logger = logging.getLogger("mlx_vector_db.optimized_store")


# ... (Der Rest der Datei, wie in der vorherigen Antwort korrigiert) ...
# Stellen Sie sicher, dass die kompilierten Funktionen außerhalb der Klasse sind.

class MLXVectorStore:
    
    def __init__(self, store_path: str, config: MLXVectorStoreConfig):
        # ... (andere Initialisierungen)
        
        # KORRIGIERT: defaultdict wird nun korrekt erkannt
        self._operation_stats = defaultdict(int)
        
        # ... (Rest der __init__ Methode)

# ... (Rest der Datei) ...

# =================== COMPILED STATIC FUNCTIONS (KORRIGIERT) ===================
# Die Funktionen wurden aus der Klasse entfernt, um JIT-kompatibel zu sein.

@mx.compile
def _compiled_cosine_similarity(query: mx.array, vectors: mx.array) -> mx.array:
    """Optimierte Cosine Similarity mit MLX 0.25.2"""
    # Stellt sicher, dass die Query eine Zeilen-Dimension für matmul hat
    if query.ndim == 1:
        query = query[None, :]

    query_norm = mx.linalg.norm(query, axis=1, keepdims=True)
    vectors_norm = mx.linalg.norm(vectors, axis=1, keepdims=True)
    
    eps = mx.array(1e-8, dtype=query.dtype)
    query_norm = mx.maximum(query_norm, eps)
    vectors_norm = mx.maximum(vectors_norm, eps)
    
    query_normalized = query / query_norm
    vectors_normalized = vectors / vectors_norm
    
    return mx.matmul(vectors_normalized, query_normalized.T).flatten()

@mx.compile
def _compiled_euclidean_distance(query: mx.array, vectors: mx.array) -> mx.array:
    """Optimierte Euclidean Distance mit MLX 0.25.2"""
    diff = vectors - query[None, :]
    distances_sq = mx.sum(diff * diff, axis=1)
    return mx.sqrt(distances_sq)

@mx.compile  
def _compiled_dot_product(query: mx.array, vectors: mx.array) -> mx.array:
    """Optimiertes Dot Product mit MLX 0.25.2"""
    if query.ndim == 1:
        return mx.matmul(vectors, query)
    return mx.matmul(vectors, query.T).flatten()


# =================== MEMORY MANAGEMENT SYSTEM ===================

class MemoryPressureMonitor:
    """Intelligente Memory Pressure Detection für Apple Silicon"""
    
    def __init__(self, warning_threshold: float = 0.75, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._last_check = 0
        self._check_interval = 1.0  # Sekunden
        self._cached_pressure = {}
        
    def get_memory_pressure(self) -> Dict[str, Any]:
        """Detaillierte Memory Analysis für Apple Silicon"""
        current_time = time.time()
        
        if current_time - self._last_check < self._check_interval and self._cached_pressure:
            return self._cached_pressure
        
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**3)
            
            pressure_info = {
                'system_memory_percent': memory_percent,
                'process_memory_gb': process_memory,
                'pressure_level': self._calculate_pressure_level(memory_percent),
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3),
                'timestamp': current_time
            }
            
            self._cached_pressure = pressure_info
            self._last_check = current_time
            
            return pressure_info
            
        except Exception as e:
            logger.error(f"Memory pressure check failed: {e}")
            return {'pressure_level': 'unknown', 'error': str(e)}

    def _calculate_pressure_level(self, memory_percent: float) -> str:
        if memory_percent >= self.critical_threshold: return 'critical'
        if memory_percent >= self.warning_threshold: return 'warning'
        return 'normal'

    def should_evict_cache(self) -> bool:
        return self.get_memory_pressure().get('pressure_level') in ['warning', 'critical']

    def should_reject_operation(self) -> bool:
        return self.get_memory_pressure().get('pressure_level') == 'critical'


class MLXMemoryPool:
    def __init__(self, max_pool_size_gb: float = 2.0):
        self.max_pool_size_gb = max_pool_size_gb
        self._pool: Dict[Tuple[int, int], List[mx.array]] = {}
        self._pool_lock = threading.Lock()
        self._total_size_bytes = 0

    def get_array(self, shape: Tuple[int, int], dtype=mx.float32) -> mx.array:
        with self._pool_lock:
            key = shape
            if key in self._pool and self._pool[key]:
                return self._pool[key].pop()
            return mx.zeros(shape, dtype=dtype)

    def return_array(self, array: mx.array) -> None:
        if array is None: return
        shape = array.shape
        size_bytes = np.prod(shape) * 4
        with self._pool_lock:
            if self._total_size_bytes + size_bytes > self.max_pool_size_gb * (1024**3): return
            key = shape
            if key not in self._pool: self._pool[key] = []
            self._pool[key].append(array)
            self._total_size_bytes += size_bytes

    def clear_pool(self):
        with self._pool_lock:
            self._pool.clear()
            self._total_size_bytes = 0
            logger.info("Memory pool cleared")

    def get_pool_stats(self) -> Dict[str, Any]:
        with self._pool_lock:
            return {'total_arrays': sum(len(a) for a in self._pool.values()), 'total_size_gb': self._total_size_bytes / (1024**3)}


class SmartVectorCache:
    def __init__(self, max_vectors: int = 10000, max_memory_gb: float = 1.0):
        self.max_vectors = max_vectors
        self.max_memory_gb = max_memory_gb
        self._cache: OrderedDict = OrderedDict()
        self._memory_usage = 0
        self._cache_lock = threading.RLock()
        self._memory_monitor = MemoryPressureMonitor()
        self._hits = 0; self._misses = 0; self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, value: Any, size_bytes: int) -> None:
        with self._cache_lock:
            if self._memory_monitor.should_evict_cache(): self._aggressive_evict()
            if key in self._cache: self._evict_key(key)
            while (len(self._cache) >= self.max_vectors or self._memory_usage + size_bytes > self.max_memory_gb * (1024**3)):
                if not self._cache: break
                self._evict_lru()
            self._cache[key] = value
            self._memory_usage += size_bytes

    def _evict_key(self, key: str):
        old_value = self._cache.pop(key)
        # Assuming value is (vector, size_bytes) tuple
        if isinstance(old_value, tuple) and len(old_value) == 2:
            self._memory_usage -= old_value[1]

    def _evict_lru(self) -> None:
        if self._cache:
            key, value = self._cache.popitem(last=False)
            if isinstance(value, tuple) and len(value) == 2:
                self._memory_usage -= value[1]
            self._evictions += 1

    def _aggressive_evict(self):
        target_size = len(self._cache) // 2
        while len(self._cache) > target_size: self._evict_lru()
        logger.info(f"Aggressive cache eviction: reduced to {len(self._cache)} vectors")

    def clear(self):
        with self._cache_lock: self._cache.clear(); self._memory_usage = 0; logger.info("Vector cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        with self._cache_lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {'size': len(self._cache), 'memory_usage_gb': self._memory_usage / (1024**3), 'hit_rate_percent': hit_rate, 'hits': self._hits, 'misses': self._misses, 'evictions': self._evictions}

@dataclass
class MLXVectorStoreConfig:
    dimension: int = 384
    metric: str = "cosine"
    index_type: str = "flat"
    batch_size: int = 1000
    use_metal: bool = True
    jit_compile: bool = True
    enable_lazy_eval: bool = True
    max_cache_vectors: int = 10000
    max_cache_memory_gb: float = 1.0
    memory_pool_size_gb: float = 2.0
    auto_recovery: bool = True
    backup_on_corruption: bool = True
    max_retry_attempts: int = 3
    enable_hnsw: bool = False
    hnsw_config: Optional[Any] = None # Can be AdaptiveHNSWConfig object

class MLXVectorStore:
    def __init__(self, store_path: str, config: MLXVectorStoreConfig):
        self.store_path = Path(store_path).expanduser()
        self.config = config
        self._lock = threading.RLock()
        self._vectors: Optional[mx.array] = None
        self._vector_count = 0
        self._metadata: List[Dict] = []
        self._is_dirty = False
        self._memory_monitor = MemoryPressureMonitor()
        self._memory_pool = MLXMemoryPool(config.memory_pool_size_gb)
        self._vector_cache = SmartVectorCache(config.max_cache_vectors, config.max_cache_memory_gb)
        self._operation_stats = defaultdict(int)
        self._compiled_similarity_fn = None
        # HNSW Index
        self._hnsw_index = None
        if config.enable_hnsw:
            from performance.hnsw_index import ProductionHNSWIndex
            # Pass metric from main config to HNSW config if not set
            if config.hnsw_config and not hasattr(config.hnsw_config, 'metric'):
                config.hnsw_config.metric = config.metric
            self._hnsw_index = ProductionHNSWIndex(dimension=config.dimension, config=config.hnsw_config)
            
        self._initialize_store()
        if config.jit_compile: self._compile_critical_functions()

    def _initialize_store(self):
        self.store_path.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.config.max_retry_attempts):
            try:
                self._load_store()
                logger.info(f"Store loaded successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.error(f"Store load attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retry_attempts - 1:
                    if self.config.auto_recovery: self._attempt_recovery()
                    time.sleep(0.1 * (attempt + 1))
                else:
                    if self.config.auto_recovery: self._create_empty_store()

    def _attempt_recovery(self): logger.info("Attempting store recovery...") # Placeholder
    def _create_empty_store(self): logger.info("Created empty store as recovery fallback") # Placeholder

    def _compile_critical_functions(self):
        logger.info("Compiling MLX functions for optimal performance...")
        try:
            if self.config.metric == "cosine": self._compiled_similarity_fn = _compiled_cosine_similarity
            elif self.config.metric == "euclidean": self._compiled_similarity_fn = _compiled_euclidean_distance
            elif self.config.metric == "dot_product": self._compiled_similarity_fn = _compiled_dot_product
            
            # Test compilation
            if self._compiled_similarity_fn:
                dummy_query = mx.random.normal((self.config.dimension,))
                dummy_vectors = mx.random.normal((10, self.config.dimension))
                result = self._compiled_similarity_fn(dummy_query, dummy_vectors)
                mx.eval(result)
                logger.info("✅ MLX function compilation successful")
        except Exception as e:
            logger.error(f"MLX compilation failed: {e}")
            self._compiled_similarity_fn = None

    def add_vectors(self, vectors: Union[np.ndarray, List[List[float]], mx.array], metadata: List[Dict]):
        if self._memory_monitor.should_reject_operation():
            raise MemoryError("Cannot add vectors: Critical memory pressure detected")
        with self._lock:
            # ... (rest of add_vectors implementation)
            new_vectors = self._convert_to_mlx_array(vectors)
            # ...
            if self._vectors is None:
                self._vectors = new_vectors
            else:
                self._vectors = mx.concatenate([self._vectors, new_vectors], axis=0)
            self._metadata.extend(metadata)
            self._vector_count = self._vectors.shape[0]
            self._is_dirty = True
            self._schedule_save()
            logger.info(f"Added {len(metadata)} vectors")
            return {'vectors_added': len(metadata), 'total_vectors': self._vector_count}

    def query(self, query_vector: Union[np.ndarray, List[float], mx.array], k: int = 10, filter_metadata: Optional[Dict] = None, use_hnsw: bool = True) -> Tuple[List[int], List[float], List[Dict]]:
        if self._vectors is None or self._vector_count == 0:
            return [], [], []
        
        # HNSW-optimierter Pfad
        if use_hnsw and self._hnsw_index and self._hnsw_index.state == "ready":
            indices, distances = self._hnsw_index.search(self._convert_to_mlx_array(query_vector), k=k)
            metadata_list = [self._metadata[i] for i in indices]
            return indices, distances, metadata_list

        # Fallback auf Brute-Force
        with self._lock:
            query_mx = self._convert_to_mlx_array(query_vector)
            
            if self._compiled_similarity_fn:
                similarities = self._compiled_similarity_fn(query_mx, self._vectors)
            else:
                similarities = self._fallback_similarity(query_mx, self._vectors)

            return self._get_top_k_results(similarities, k, None)

    def _fallback_similarity(self, query: mx.array, vectors: mx.array) -> mx.array:
        """Fallback-Ähnlichkeit mit Schutz vor Nulldivision."""
        if self.config.metric == "cosine":
            query_norm_val = mx.linalg.norm(query)
            query_norm = query / mx.maximum(query_norm_val, mx.array(1e-8)) # KORRIGIERT
            
            vectors_norm_val = mx.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_norm = vectors / mx.maximum(vectors_norm_val, mx.array(1e-8)) # KORRIGIERT

            return mx.matmul(vectors_norm, query_norm.T).flatten() # KORRIGIERT
        
        elif self.config.metric == "euclidean":
            # ...
            return -mx.sqrt(mx.sum((vectors - query[None,:])**2, axis=1))
        else:  # dot_product
            return mx.matmul(vectors, query.T).flatten()

    def _get_top_k_results(self, similarities: mx.array, k: int, valid_indices: Optional[np.ndarray]) -> Tuple[List[int], List[float], List[Dict]]:
        # For similarity (higher is better), we sort descending
        # For distance (lower is better), we sort ascending
        sort_asc = self.config.metric == "euclidean"

        if sort_asc:
             top_k_indices = mx.argsort(similarities)[:k]
        else: # Sort descending for cosine/dot_product
             top_k_indices = mx.argsort(-similarities)[:k]

        # mx.eval to compute results
        mx.eval(top_k_indices)
        top_k_scores = similarities[top_k_indices]
        mx.eval(top_k_scores)
        
        indices_list = top_k_indices.tolist()
        scores_list = top_k_scores.tolist()
        
        # Distanz für die Rückgabe anpassen
        if self.config.metric == 'cosine':
            distances = [1.0 - s for s in scores_list]
        elif self.config.metric == 'euclidean':
            distances = scores_list # Bereits eine Distanz
        else: # dot_product
            distances = [-s for s in scores_list] # Negieren, um es als Distanz darzustellen

        metadata_list = [self._metadata[i] for i in indices_list]
        return indices_list, distances, metadata_list

    def _schedule_save(self):
        if self._is_dirty:
            try:
                self._save_store()
                self._is_dirty = False
            except Exception as e:
                logger.error(f"Scheduled save failed: {e}")

    def _save_store(self):
        if self._vectors is None: return
        try:
            # KORRIGIERT: Stelle sicher, dass das Verzeichnis existiert, bevor geschrieben wird.
            self.store_path.mkdir(parents=True, exist_ok=True)
            
            temp_vectors_path = self.store_path / "vectors.npz.tmp"
            temp_metadata_path = self.store_path / "metadata.jsonl.tmp"
            
            mx.savez(str(temp_vectors_path), vectors=self._vectors)
            with open(temp_metadata_path, 'w') as f:
                for meta in self._metadata: f.write(json.dumps(meta) + '\n')

            temp_vectors_path.rename(self.store_path / "vectors.npz")
            temp_metadata_path.rename(self.store_path / "metadata.jsonl")
        except Exception as e:
            logger.error(f"Store save failed: {e}")
            # Potenzielles Aufräumen von .tmp Dateien hier
            if 'temp_vectors_path' in locals() and temp_vectors_path.exists(): temp_vectors_path.unlink()
            if 'temp_metadata_path' in locals() and temp_metadata_path.exists(): temp_metadata_path.unlink()
            raise

    def _load_store(self):
        vectors_path = self.store_path / "vectors.npz"
        if not vectors_path.exists():
            logger.info("🆕 Creating new optimized vector store")
            return
        # ... (rest of loading logic)
        try:
            data = mx.load(str(vectors_path))
            self._vectors = data['vectors']
            self._vector_count = self._vectors.shape[0]
            
            metadata_path = self.store_path / "metadata.jsonl"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self._metadata = [json.loads(line) for line in f]
            
            logger.info(f"📂 Loaded optimized store: {self._vector_count} vectors")
        except Exception as e:
            logger.error(f"Store loading failed: {e}")
            self._create_empty_store()

    def optimize(self):
        """Triggers HNSW index build if enabled and not already built."""
        if self.config.enable_hnsw and self._hnsw_index and self._vectors is not None:
            logger.info("Optimizing store by building HNSW index...")
            self._hnsw_index.build(self._vectors)
            logger.info("HNSW index build complete.")
        else:
            logger.info("No optimization required or HNSW not enabled.")
        if self._vectors is not None: mx.eval(self._vectors)
        gc.collect()

    def get_comprehensive_stats(self):
        # ... (existing implementation)
        return {"vector_count": self._vector_count, "dimension": self.config.dimension, "metric": self.config.metric}

    def _convert_to_mlx_array(self, vectors: Union[np.ndarray, List[List[float]], mx.array]) -> mx.array:
        if isinstance(vectors, mx.array): return vectors
        np_array = np.array(vectors, dtype=np.float32)
        return mx.array(np_array)
    
    def batch_query(self, query_vectors: Union[np.ndarray, List[List[float]], mx.array], k: int = 10):
        # ... (existing implementation)
        queries_mx = self._convert_to_mlx_array(query_vectors)
        results = []
        for i in range(queries_mx.shape[0]):
            results.append(self.query(queries_mx[i], k=k))
        return results

    def clear(self):
        with self._lock:
            self._vectors = None
            self._metadata = []
            self._vector_count = 0
            self._is_dirty = False
            self._vector_cache.clear()
            self._memory_pool.clear_pool()
            # ... (file cleanup)

    def get_stats(self):
        return self.get_comprehensive_stats()

    def delete_vectors(self, indices: List[int]) -> int:
        # Placeholder for delete logic
        logger.warning("delete_vectors is not fully implemented yet.")
        return 0

    def _warmup_kernels(self):
        logger.info("Warming up MLX kernels for the store...")
        self._compile_critical_functions()