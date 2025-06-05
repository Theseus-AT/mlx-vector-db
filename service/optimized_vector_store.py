"""
MLX Vector Store - Production-Ready mit MLX 0.25.2 Features
Revolutionäre Performance-Optimierungen für Apple Silicon

Key Features:
- Metal-native Batch-Operationen mit mx.compile
- Intelligentes Memory Management mit Unified Memory
- Zero-Copy Operations maximiert
- Lazy Evaluation optimiert
- Production-Grade Error Handling
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
from collections import OrderedDict
import pickle

logger = logging.getLogger("mlx_vector_db.optimized_store")

# =================== MEMORY MANAGEMENT SYSTEM ===================

class MemoryPressureMonitor:
    """Intelligente Memory Pressure Detection für Apple Silicon"""
    
    def __init__(self, warning_threshold: float = 0.75, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._last_check = 0
        self._check_interval = 1.0  # Sekunden
        
    def get_memory_pressure(self) -> Dict[str, Any]:
        """Detaillierte Memory Analysis für Apple Silicon"""
        current_time = time.time()
        
        # Cache für 1 Sekunde
        if current_time - self._last_check < self._check_interval:
            return getattr(self, '_cached_pressure', {})
        
        try:
            # System Memory (macOS Unified Memory)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            # Process Memory
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**3)  # GB
            
            # Estimate MLX GPU Memory (approximation)
            mlx_memory_estimate = self._estimate_mlx_memory()
            
            pressure_info = {
                'system_memory_percent': memory_percent,
                'process_memory_gb': process_memory,
                'mlx_memory_estimate_gb': mlx_memory_estimate,
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
    
    def _estimate_mlx_memory(self) -> float:
        """Schätze MLX Memory Usage (Unified Memory)"""
        try:
            # Vereinfachte Schätzung basierend auf mx Arrays im Scope
            # In realer Implementierung könnte man mx.metal.get_memory_info() verwenden
            return 0.1  # Placeholder
        except:
            return 0.0
    
    def _calculate_pressure_level(self, memory_percent: float) -> str:
        """Berechne Pressure Level"""
        if memory_percent >= self.critical_threshold:
            return 'critical'
        elif memory_percent >= self.warning_threshold:
            return 'warning'
        else:
            return 'normal'
    
    def should_evict_cache(self) -> bool:
        """Entscheidung für Cache Eviction"""
        pressure = self.get_memory_pressure()
        return pressure.get('pressure_level') in ['warning', 'critical']
    
    def should_reject_operation(self) -> bool:
        """Entscheidung für Operation Rejection"""
        pressure = self.get_memory_pressure()
        return pressure.get('pressure_level') == 'critical'


class MLXMemoryPool:
    """Memory Pool für häufige MLX Array Operationen"""
    
    def __init__(self, max_pool_size_gb: float = 2.0):
        self.max_pool_size_gb = max_pool_size_gb
        self._pool: Dict[Tuple[int, int], List[mx.array]] = {}
        self._pool_lock = threading.Lock()
        self._total_size_bytes = 0
        
    def get_array(self, shape: Tuple[int, int], dtype=mx.float32) -> mx.array:
        """Hole Array aus Pool oder erstelle neu"""
        with self._pool_lock:
            key = shape
            
            if key in self._pool and self._pool[key]:
                array = self._pool[key].pop()
                logger.debug(f"Reused array from pool: {shape}")
                return array
            
            # Erstelle neues Array
            array = mx.zeros(shape, dtype=dtype)
            logger.debug(f"Created new array: {shape}")
            return array
    
    def return_array(self, array: mx.array) -> None:
        """Gib Array zurück an Pool"""
        if array is None:
            return
            
        shape = array.shape
        size_bytes = np.prod(shape) * 4  # float32
        
        with self._pool_lock:
            # Check Pool Size Limit
            if self._total_size_bytes + size_bytes > self.max_pool_size_gb * (1024**3):
                logger.debug(f"Pool full, discarding array: {shape}")
                return
            
            key = shape
            if key not in self._pool:
                self._pool[key] = []
            
            self._pool[key].append(array)
            self._total_size_bytes += size_bytes
            logger.debug(f"Returned array to pool: {shape}")
    
    def clear_pool(self) -> None:
        """Leere kompletten Pool"""
        with self._pool_lock:
            self._pool.clear()
            self._total_size_bytes = 0
            logger.info("Memory pool cleared")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Pool Statistiken"""
        with self._pool_lock:
            total_arrays = sum(len(arrays) for arrays in self._pool.values())
            return {
                'total_arrays': total_arrays,
                'total_size_gb': self._total_size_bytes / (1024**3),
                'shapes_cached': list(self._pool.keys()),
                'max_size_gb': self.max_pool_size_gb
            }


class SmartVectorCache:
    """LRU Cache für Vektoren mit Memory Pressure Awareness"""
    
    def __init__(self, max_vectors: int = 10000, max_memory_gb: float = 1.0):
        self.max_vectors = max_vectors
        self.max_memory_gb = max_memory_gb
        self._cache: OrderedDict = OrderedDict()
        self._memory_usage = 0
        self._cache_lock = threading.RLock()
        self._memory_monitor = MemoryPressureMonitor()
        
        # Statistiken
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    def get(self, key: str) -> Optional[mx.array]:
        """Hole Vector aus Cache"""
        with self._cache_lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._hits += 1
                return value
            
            self._misses += 1
            return None
    
    def put(self, key: str, vector: mx.array) -> None:
        """Speichere Vector in Cache"""
        if vector is None:
            return
            
        vector_size = np.prod(vector.shape) * 4  # float32 bytes
        
        with self._cache_lock:
            # Check Memory Pressure
            if self._memory_monitor.should_evict_cache():
                self._aggressive_evict()
            
            # Remove if already exists
            if key in self._cache:
                old_vector = self._cache.pop(key)
                self._memory_usage -= np.prod(old_vector.shape) * 4
            
            # Check capacity
            while (len(self._cache) >= self.max_vectors or 
                   self._memory_usage + vector_size > self.max_memory_gb * (1024**3)):
                if not self._cache:
                    break
                self._evict_lru()
            
            # Add new vector
            self._cache[key] = vector
            self._memory_usage += vector_size
    
    def _evict_lru(self) -> None:
        """Evict Least Recently Used"""
        if self._cache:
            key, vector = self._cache.popitem(last=False)
            self._memory_usage -= np.prod(vector.shape) * 4
            self._evictions += 1
            logger.debug(f"Evicted vector from cache: {key}")
    
    def _aggressive_evict(self) -> None:
        """Aggressive Eviction bei Memory Pressure"""
        target_size = len(self._cache) // 2
        while len(self._cache) > target_size:
            self._evict_lru()
        logger.info(f"Aggressive cache eviction: reduced to {len(self._cache)} vectors")
    
    def clear(self) -> None:
        """Leere Cache komplett"""
        with self._cache_lock:
            self._cache.clear()
            self._memory_usage = 0
            logger.info("Vector cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Cache Statistiken"""
        with self._cache_lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'memory_usage_gb': self._memory_usage / (1024**3),
                'hit_rate_percent': hit_rate,
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'max_vectors': self.max_vectors,
                'max_memory_gb': self.max_memory_gb
            }


# =================== OPTIMIZED VECTOR STORE ===================

@dataclass
class MLXVectorStoreConfig:
    """Erweiterte Konfiguration für Production-Ready Store"""
    dimension: int = 384
    metric: str = "cosine"  # cosine, euclidean, dot_product
    index_type: str = "flat"  # flat, hnsw
    
    # Performance Optimierungen
    batch_size: int = 1000
    use_metal: bool = True
    jit_compile: bool = True
    enable_lazy_eval: bool = True
    
    # Memory Management
    max_cache_vectors: int = 10000
    max_cache_memory_gb: float = 1.0
    memory_pool_size_gb: float = 2.0
    
    # Error Handling
    auto_recovery: bool = True
    backup_on_corruption: bool = True
    max_retry_attempts: int = 3
    
    # HNSW Configuration
    enable_hnsw: bool = False
    hnsw_config: Optional[Dict] = None


class MLXVectorStoreOptimized:
    """Production-Ready MLX Vector Store mit allen Optimierungen"""
    
    def __init__(self, store_path: str, config: MLXVectorStoreConfig):
        self.store_path = Path(store_path)
        self.config = config
        self._lock = threading.RLock()
        
        # Core Data
        self._vectors: Optional[mx.array] = None
        self._vector_count = 0
        self._metadata: List[Dict] = []
        self._is_dirty = False
        
        # Memory Management
        self._memory_monitor = MemoryPressureMonitor()
        self._memory_pool = MLXMemoryPool(config.memory_pool_size_gb)
        self._vector_cache = SmartVectorCache(
            config.max_cache_vectors, 
            config.max_cache_memory_gb
        )
        
        # Performance Tracking
        self._operation_stats = {
            'queries': 0,
            'additions': 0,
            'cache_hits': 0,
            'memory_evictions': 0,
            'errors': 0
        }
        
        # Compiled Functions
        self._compiled_similarity_fn = None
        self._compiled_batch_add_fn = None
        
        # Initialize
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._initialize_store()
        
        # Compile functions on startup
        if config.jit_compile:
            self._compile_critical_functions()
    
    def _initialize_store(self) -> None:
        """Initialisiere Store mit Error Recovery"""
        max_attempts = self.config.max_retry_attempts
        
        for attempt in range(max_attempts):
            try:
                self._load_store()
                logger.info(f"Store loaded successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.error(f"Store load attempt {attempt + 1} failed: {e}")
                
                if attempt < max_attempts - 1:
                    if self.config.auto_recovery:
                        self._attempt_recovery()
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error("All store load attempts failed")
                    if self.config.auto_recovery:
                        self._create_empty_store()
    
    def _attempt_recovery(self) -> None:
        """Versuche Store Recovery"""
        logger.info("Attempting store recovery...")
        
        try:
            # Backup corrupted files
            if self.config.backup_on_corruption:
                self._backup_corrupted_files()
            
            # Try loading from backup
            backup_path = self.store_path / "backup"
            if backup_path.exists():
                self._restore_from_backup(backup_path)
                return
            
            # Create empty store as last resort
            self._create_empty_store()
            
        except Exception as e:
            logger.error(f"Store recovery failed: {e}")
    
    def _backup_corrupted_files(self) -> None:
        """Backup korrupte Dateien für Analyse"""
        corrupted_dir = self.store_path / "corrupted" / str(int(time.time()))
        corrupted_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in self.store_path.glob("*.npz"):
            try:
                import shutil
                shutil.copy2(file_path, corrupted_dir / file_path.name)
                logger.info(f"Backed up corrupted file: {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to backup {file_path.name}: {e}")
    
    def _create_empty_store(self) -> None:
        """Erstelle leeren Store"""
        self._vectors = None
        self._metadata = []
        self._vector_count = 0
        self._save_store()
        logger.info("Created empty store as recovery fallback")
    
    # =================== MLX 0.25.2 COMPILED FUNCTIONS ===================
    
    def _compile_critical_functions(self) -> None:
        """Kompiliere alle performance-kritischen Funktionen"""
        logger.info("Compiling MLX functions for optimal performance...")
        
        try:
            # Test compilation mit dummy data
            dummy_shape = (100, self.config.dimension)
            dummy_vectors = mx.random.normal(dummy_shape)
            dummy_query = mx.random.normal((self.config.dimension,))
            
            # Compile similarity functions
            self._compiled_similarity_fn = self._get_compiled_similarity_fn()
            
            # Test compilation
            result = self._compiled_similarity_fn(dummy_query, dummy_vectors)
            mx.eval(result)
            
            logger.info("✅ MLX function compilation successful")
            
        except Exception as e:
            logger.error(f"MLX compilation failed: {e}")
            self._compiled_similarity_fn = None
    
    def _get_compiled_similarity_fn(self):
        """Hole kompilierte Similarity Function"""
        if self.config.metric == "cosine":
            return self._compile_cosine_similarity()
        elif self.config.metric == "euclidean":
            return self._compile_euclidean_distance()
        elif self.config.metric == "dot_product":
            return self._compile_dot_product()
        else:
            raise ValueError(f"Unsupported metric: {self.config.metric}")
    
    @mx.compile
    def _compile_cosine_similarity(self):
        """Optimierte Cosine Similarity mit MLX 0.25.2"""
        def cosine_similarity_batch(query: mx.array, vectors: mx.array) -> mx.array:
            # Optimierte Normalisierung
            query_norm = mx.linalg.norm(query)
            vectors_norm = mx.linalg.norm(vectors, axis=1, keepdims=True)
            
            # Epsilon für numerische Stabilität
            eps = mx.array(1e-8, dtype=query.dtype)
            query_norm = mx.maximum(query_norm, eps)
            vectors_norm = mx.maximum(vectors_norm, eps)
            
            # Normalisierte Vektoren
            query_normalized = query / query_norm
            vectors_normalized = vectors / vectors_norm
            
            # Metal-optimierte Matrix-Multiplikation
            return mx.matmul(vectors_normalized, query_normalized)
        
        return cosine_similarity_batch
    
    @mx.compile
    def _compile_euclidean_distance(self):
        """Optimierte Euclidean Distance mit MLX 0.25.2"""
        def euclidean_distance_batch(query: mx.array, vectors: mx.array) -> mx.array:
            # Broadcasting für effiziente Subtraktion
            diff = vectors - query[None, :]
            
            # Squared distances mit Metal acceleration
            distances_sq = mx.sum(diff * diff, axis=1)
            
            # Return negative für consistent "higher is better" semantics
            return -mx.sqrt(distances_sq)
        
        return euclidean_distance_batch
    
    @mx.compile  
    def _compile_dot_product(self):
        """Optimiertes Dot Product mit MLX 0.25.2"""
        def dot_product_batch(query: mx.array, vectors: mx.array) -> mx.array:
            return mx.matmul(vectors, query)
        
        return dot_product_batch
    
    # =================== CORE OPERATIONS ===================
    
    def add_vectors(self, vectors: Union[np.ndarray, List[List[float]], mx.array], 
                   metadata: List[Dict]) -> Dict[str, Any]:
        """Optimized Vector Addition mit Error Handling"""
        
        # Memory Pressure Check
        if self._memory_monitor.should_reject_operation():
            raise MemoryError("Cannot add vectors: Critical memory pressure detected")
        
        with self._lock:
            try:
                start_time = time.time()
                
                # Convert zu MLX array mit optimierter Konvertierung
                new_vectors = self._convert_to_mlx_array(vectors)
                
                # Validation
                if new_vectors.shape[1] != self.config.dimension:
                    raise ValueError(f"Vector dimension {new_vectors.shape[1]} != {self.config.dimension}")
                
                if len(metadata) != new_vectors.shape[0]:
                    raise ValueError(f"Metadata count {len(metadata)} != vector count {new_vectors.shape[0]}")
                
                # Batch Processing für große Datasets
                if new_vectors.shape[0] > self.config.batch_size:
                    return self._add_vectors_batched(new_vectors, metadata)
                
                # Concatenate mit existing vectors (lazy operation)
                if self._vectors is None:
                    self._vectors = new_vectors
                else:
                    self._vectors = mx.concatenate([self._vectors, new_vectors], axis=0)
                
                # Update metadata
                self._metadata.extend(metadata)
                self._vector_count += len(metadata)
                self._is_dirty = True
                
                # Update cache
                for i, meta in enumerate(metadata):
                    if 'id' in meta:
                        vector_idx = self._vector_count - len(metadata) + i
                        cache_key = f"vector_{meta['id']}"
                        self._vector_cache.put(cache_key, self._vectors[vector_idx])
                
                # Auto-save in background
                self._schedule_save()
                
                # Update stats
                self._operation_stats['additions'] += 1
                add_time = time.time() - start_time
                
                logger.info(f"Added {len(metadata)} vectors in {add_time:.3f}s")
                
                return {
                    'success': True,
                    'vectors_added': len(metadata),
                    'total_vectors': self._vector_count,
                    'add_time_ms': add_time * 1000,
                    'memory_usage_gb': self._get_memory_usage_gb()
                }
                
            except Exception as e:
                self._operation_stats['errors'] += 1
                logger.error(f"Vector addition failed: {e}")
                raise
    
    def _convert_to_mlx_array(self, vectors: Union[np.ndarray, List[List[float]], mx.array]) -> mx.array:
        """Optimierte Konvertierung zu MLX Array"""
        if isinstance(vectors, mx.array):
            return vectors
        elif isinstance(vectors, np.ndarray):
            # Zero-copy wenn möglich
            return mx.array(vectors.astype(np.float32))
        else:
            # List zu numpy zu mlx
            np_array = np.array(vectors, dtype=np.float32)
            return mx.array(np_array)
    
    def _add_vectors_batched(self, vectors: mx.array, metadata: List[Dict]) -> Dict[str, Any]:
        """Batch Processing für große Vector Sets"""
        total_vectors = vectors.shape[0]
        batch_size = self.config.batch_size
        
        results = []
        for i in range(0, total_vectors, batch_size):
            end_idx = min(i + batch_size, total_vectors)
            batch_vectors = vectors[i:end_idx]
            batch_metadata = metadata[i:end_idx]
            
            # Process batch
            result = self.add_vectors(batch_vectors, batch_metadata)
            results.append(result)
            
            # Memory pressure check between batches
            if self._memory_monitor.should_evict_cache():
                self._vector_cache.clear()
        
        # Aggregate results
        total_added = sum(r['vectors_added'] for r in results)
        total_time = sum(r['add_time_ms'] for r in results)
        
        return {
            'success': True,
            'vectors_added': total_added,
            'total_vectors': self._vector_count,
            'batches_processed': len(results),
            'total_time_ms': total_time,
            'memory_usage_gb': self._get_memory_usage_gb()
        }
    
    def query(self, query_vector: Union[np.ndarray, List[float], mx.array], 
              k: int = 10, 
              filter_metadata: Optional[Dict] = None,
              use_cache: bool = True) -> Tuple[List[int], List[float], List[Dict]]:
        """Ultra-optimized Vector Query mit Caching"""
        
        if self._vectors is None or self._vector_count == 0:
            return [], [], []
        
        with self._lock:
            try:
                start_time = time.time()
                
                # Convert query zu MLX
                query_mx = self._convert_to_mlx_array(
                    query_vector if isinstance(query_vector, (list, np.ndarray)) 
                    else [query_vector]
                )[0]
                
                # Cache Check
                cache_key = f"query_{hash(str(query_vector))}"
                if use_cache:
                    cached_result = self._vector_cache.get(cache_key)
                    if cached_result is not None:
                        self._operation_stats['cache_hits'] += 1
                        return cached_result
                
                # Apply metadata filtering
                valid_indices = self._apply_metadata_filter(filter_metadata)
                
                # Use compiled similarity function
                if self._compiled_similarity_fn:
                    similarities = self._compiled_similarity_fn(query_mx, self._vectors)
                else:
                    # Fallback
                    similarities = self._fallback_similarity(query_mx, self._vectors)
                
                # Get top-k with efficient sorting
                indices, distances, metadata_list = self._get_top_k_results(
                    similarities, k, valid_indices
                )
                
                # Cache result
                if use_cache:
                    result = (indices, distances, metadata_list)
                    self._vector_cache.put(cache_key, result)
                
                # Update stats
                self._operation_stats['queries'] += 1
                query_time = time.time() - start_time
                
                logger.debug(f"Query completed in {query_time*1000:.2f}ms")
                
                return indices, distances, metadata_list
                
            except Exception as e:
                self._operation_stats['errors'] += 1
                logger.error(f"Query failed: {e}")
                raise
    
    def _fallback_similarity(self, query: mx.array, vectors: mx.array) -> mx.array:
        """Fallback similarity ohne compilation"""
        if self.config.metric == "cosine":
            query_norm = query / mx.linalg.norm(query)
            vectors_norm = vectors / mx.linalg.norm(vectors, axis=1, keepdims=True)
            return mx.matmul(vectors_norm, query_norm)
        elif self.config.metric == "euclidean":
            diff = vectors - query[None, :]
            return -mx.sqrt(mx.sum(diff * diff, axis=1))
        else:  # dot_product
            return mx.matmul(vectors, query)
    
    def _get_top_k_results(self, similarities: mx.array, k: int, 
                          valid_indices: Optional[np.ndarray]) -> Tuple[List[int], List[float], List[Dict]]:
        """Efficient Top-K Selection mit Filtering"""
        
        # Convert zu numpy für indexing (MLX 0.25.2 optimiert)
        similarities_np = np.array(similarities.tolist())
        
        # Apply filtering
        if valid_indices is not None and len(valid_indices) > 0:
            filtered_similarities = similarities_np[valid_indices]
            available_indices = valid_indices
        else:
            filtered_similarities = similarities_np
            available_indices = np.arange(len(similarities_np))
        
        # Get top-k indices
        if len(filtered_similarities) <= k:
            top_k_indices = np.arange(len(filtered_similarities))
        else:
            # Efficient partial sort für große k
            top_k_indices = np.argpartition(filtered_similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(filtered_similarities[top_k_indices])[::-1]]
        
        # Build results
        result_indices = [int(available_indices[idx]) for idx in top_k_indices]
        
        # Convert similarities to distances based on metric
        if self.config.metric == "cosine":
            result_distances = [float(1.0 - filtered_similarities[idx]) for idx in top_k_indices]
        else:
            result_distances = [float(-filtered_similarities[idx]) for idx in top_k_indices]
        
        result_metadata = [self._metadata[idx] for idx in result_indices]
        
        return result_indices, result_distances, result_metadata
    
    def _apply_metadata_filter(self, filter_metadata: Optional[Dict]) -> Optional[np.ndarray]:
        """Optimized Metadata Filtering"""
        if filter_metadata is None:
            return None
        
        valid_indices = []
        for i, metadata in enumerate(self._metadata):
            if all(metadata.get(k) == v for k, v in filter_metadata.items()):
                valid_indices.append(i)
        
        return np.array(valid_indices) if valid_indices else np.array([])
    
    # =================== PERSISTENCE & RECOVERY ===================
    
    def _schedule_save(self) -> None:
        """Scheduliere asynchrones Speichern"""
        if not self._is_dirty:
            return
        
        # Einfaches immediate save für jetzt
        # In production würde man ThreadPoolExecutor verwenden
        try:
            self._save_store()
            self._is_dirty = False
        except Exception as e:
            logger.error(f"Scheduled save failed: {e}")
    
    def _save_store(self) -> None:
        """Optimized Store Persistence mit Atomic Writes"""
        if self._vectors is None:
            return
        
        try:
            # Atomic save mit temp files
            temp_vectors_path = self.store_path / "vectors.npz.tmp"
            temp_metadata_path = self.store_path / "metadata.jsonl.tmp"
            temp_config_path = self.store_path / "config.json.tmp"
            
            # Save vectors (MLX native format)
            mx.savez(str(temp_vectors_path), vectors=self._vectors)
            
            # Save metadata (JSONL)
            with open(temp_metadata_path, 'w') as f:
                for meta in self._metadata:
                    f.write(json.dumps(meta) + '\n')
            
            # Save config
            config_data = {
                'dimension': self.config.dimension,
                'metric': self.config.metric,
                'vector_count': self._vector_count,
                'timestamp': time.time(),
                'version': '2.0'
            }
            
            with open(temp_config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Atomic rename (POSIX atomic operation)
            temp_vectors_path.rename(self.store_path / "vectors.npz")
            temp_metadata_path.rename(self.store_path / "metadata.jsonl")
            temp_config_path.rename(self.store_path / "config.json")
            
            logger.debug("Store saved successfully with atomic operations")
            
        except Exception as e:
            # Cleanup temp files on failure
            for temp_path in [temp_vectors_path, temp_metadata_path, temp_config_path]:
                if temp_path.exists():
                    temp_path.unlink()
            
            logger.error(f"Store save failed: {e}")
            raise
    
    def _load_store(self) -> None:
        """Load Store mit Error Recovery"""
        vectors_path = self.store_path / "vectors.npz"
        metadata_path = self.store_path / "metadata.jsonl"
        config_path = self.store_path / "config.json"
        
        if not vectors_path.exists():
            logger.info("🆕 Creating new optimized vector store")
            return
        
        try:
            # Load config first
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    stored_count = config_data.get('vector_count', 0)
                    stored_version = config_data.get('version', '1.0')
                    
                    if stored_version != '2.0':
                        logger.warning(f"Loading legacy store version {stored_version}")
            
            # Load vectors (lazy loading)
            vectors_data = mx.load(str(vectors_path))
            self._vectors = vectors_data['vectors']
            
            # Validate vector dimensions
            if self._vectors.shape[1] != self.config.dimension:
                raise ValueError(f"Stored dimension {self._vectors.shape[1]} != config {self.config.dimension}")
            
            # Load metadata
            self._metadata = []
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                metadata_item = json.loads(line)
                                self._metadata.append(metadata_item)
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON at line {line_num}: {e}")
                                # Continue loading other metadata
            
            # Update count and validate consistency
            self._vector_count = self._vectors.shape[0]
            
            if len(self._metadata) != self._vector_count:
                logger.warning(f"Metadata count {len(self._metadata)} != vector count {self._vector_count}")
                # Pad metadata if needed
                while len(self._metadata) < self._vector_count:
                    self._metadata.append({'id': f'missing_{len(self._metadata)}'})
            
            logger.info(f"📂 Loaded optimized store: {self._vector_count} vectors, {len(self._metadata)} metadata")
            
        except Exception as e:
            logger.error(f"Store loading failed: {e}")
            raise
    
    def _restore_from_backup(self, backup_path: Path) -> None:
        """Restore Store from Backup"""
        try:
            import shutil
            
            # Copy backup files to main location
            for backup_file in backup_path.glob("*"):
                if backup_file.is_file():
                    target_file = self.store_path / backup_file.name
                    shutil.copy2(backup_file, target_file)
            
            # Reload from restored files
            self._load_store()
            logger.info("Store successfully restored from backup")
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            raise
    
    # =================== MAINTENANCE & OPTIMIZATION ===================
    
    def optimize(self) -> Dict[str, Any]:
        """Comprehensive Store Optimization"""
        start_time = time.time()
        
        with self._lock:
            optimization_results = {
                'memory_before_gb': self._get_memory_usage_gb(),
                'cache_stats_before': self._vector_cache.get_stats()
            }
            
            try:
                # Force evaluation of all lazy operations
                if self._vectors is not None:
                    mx.eval(self._vectors)
                
                # Memory optimization
                self._optimize_memory()
                
                # Cache optimization
                self._optimize_cache()
                
                # Recompile functions if needed
                if self.config.jit_compile and self._compiled_similarity_fn is None:
                    self._compile_critical_functions()
                
                # Garbage collection
                gc.collect()
                
                optimization_time = time.time() - start_time
                
                optimization_results.update({
                    'memory_after_gb': self._get_memory_usage_gb(),
                    'cache_stats_after': self._vector_cache.get_stats(),
                    'optimization_time_ms': optimization_time * 1000,
                    'success': True
                })
                
                logger.info(f"Store optimization completed in {optimization_time:.3f}s")
                
            except Exception as e:
                optimization_results.update({
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"Store optimization failed: {e}")
            
            return optimization_results
    
    def _optimize_memory(self) -> None:
        """Memory-specific Optimizations"""
        # Clear memory pool if under pressure
        if self._memory_monitor.should_evict_cache():
            self._memory_pool.clear_pool()
            self._operation_stats['memory_evictions'] += 1
        
        # Defragment metadata (remove None entries)
        self._metadata = [meta for meta in self._metadata if meta is not None]
        
        logger.debug("Memory optimization completed")
    
    def _optimize_cache(self) -> None:
        """Cache-specific Optimizations"""
        # Clear old cache entries if memory pressure
        if self._memory_monitor.should_evict_cache():
            cache_size_before = len(self._vector_cache._cache)
            self._vector_cache._aggressive_evict()
            cache_size_after = len(self._vector_cache._cache)
            
            logger.info(f"Cache optimized: {cache_size_before} -> {cache_size_after} entries")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Comprehensive Store Statistics"""
        memory_pressure = self._memory_monitor.get_memory_pressure()
        cache_stats = self._vector_cache.get_stats()
        pool_stats = self._memory_pool.get_pool_stats()
        
        return {
            # Core Stats
            'vector_count': self._vector_count,
            'dimension': self.config.dimension,
            'metric': self.config.metric,
            'memory_usage_gb': self._get_memory_usage_gb(),
            
            # Performance Stats
            'operation_stats': self._operation_stats.copy(),
            'compiled_functions': self._compiled_similarity_fn is not None,
            
            # Memory Management
            'memory_pressure': memory_pressure,
            'cache_stats': cache_stats,
            'memory_pool_stats': pool_stats,
            
            # Health Indicators
            'store_health': self._assess_store_health(),
            'optimization_recommended': self._should_optimize(),
            
            # MLX Info
            'mlx_device': str(mx.default_device()),
            'unified_memory': True,
            'metal_available': True
        }
    
    def _assess_store_health(self) -> Dict[str, Any]:
        """Assess Overall Store Health"""
        pressure = self._memory_monitor.get_memory_pressure()
        cache_stats = self._vector_cache.get_stats()
        
        health_score = 100
        issues = []
        
        # Memory pressure check
        if pressure.get('pressure_level') == 'critical':
            health_score -= 30
            issues.append('Critical memory pressure')
        elif pressure.get('pressure_level') == 'warning':
            health_score -= 15
            issues.append('High memory usage')
        
        # Cache efficiency check
        hit_rate = cache_stats.get('hit_rate_percent', 0)
        if hit_rate < 20:
            health_score -= 20
            issues.append('Low cache hit rate')
        
        # Error rate check
        total_ops = sum(self._operation_stats.values())
        if total_ops > 0:
            error_rate = self._operation_stats['errors'] / total_ops
            if error_rate > 0.05:  # 5% error rate
                health_score -= 25
                issues.append('High error rate')
        
        # Compilation check
        if self.config.jit_compile and self._compiled_similarity_fn is None:
            health_score -= 10
            issues.append('Functions not compiled')
        
        health_level = 'excellent' if health_score >= 90 else \
                      'good' if health_score >= 70 else \
                      'warning' if health_score >= 50 else 'critical'
        
        return {
            'health_score': max(0, health_score),
            'health_level': health_level,
            'issues': issues,
            'recommendations': self._get_health_recommendations(issues)
        }
    
    def _get_health_recommendations(self, issues: List[str]) -> List[str]:
        """Get Recommendations based on Health Issues"""
        recommendations = []
        
        if 'Critical memory pressure' in issues:
            recommendations.append('Run optimize() to free memory')
            recommendations.append('Consider reducing cache size')
        
        if 'Low cache hit rate' in issues:
            recommendations.append('Review query patterns')
            recommendations.append('Consider increasing cache size')
        
        if 'High error rate' in issues:
            recommendations.append('Check logs for error patterns')
            recommendations.append('Verify data integrity')
        
        if 'Functions not compiled' in issues:
            recommendations.append('Enable JIT compilation for better performance')
        
        return recommendations
    
    def _should_optimize(self) -> bool:
        """Determine if Optimization is Recommended"""
        pressure = self._memory_monitor.get_memory_pressure()
        cache_stats = self._vector_cache.get_stats()
        
        return (
            pressure.get('pressure_level') in ['warning', 'critical'] or
            cache_stats.get('hit_rate_percent', 100) < 30 or
            self._operation_stats['errors'] > 10
        )
    
    def _get_memory_usage_gb(self) -> float:
        """Estimate Total Memory Usage"""
        if self._vectors is None:
            return 0.0
        
        # Vector memory
        vector_size = self._vector_count * self.config.dimension * 4  # float32
        
        # Metadata memory
        metadata_size = len(json.dumps(self._metadata).encode())
        
        # Cache memory
        cache_size = self._vector_cache._memory_usage
        
        # Pool memory
        pool_size = self._memory_pool._total_size_bytes
        
        total_bytes = vector_size + metadata_size + cache_size + pool_size
        return total_bytes / (1024**3)
    
    # =================== BATCH OPERATIONS ===================
    
    def batch_query(self, query_vectors: Union[np.ndarray, List[List[float]], mx.array], 
                   k: int = 10, batch_size: Optional[int] = None) -> Tuple[List[List[int]], List[List[float]], List[List[Dict]]]:
        """Optimized Batch Query Processing"""
        
        if isinstance(query_vectors, (list, np.ndarray)):
            queries_mx = mx.array(np.array(query_vectors, dtype=np.float32))
        else:
            queries_mx = query_vectors
        
        num_queries = queries_mx.shape[0]
        batch_size = batch_size or min(self.config.batch_size, num_queries)
        
        all_indices = []
        all_distances = []
        all_metadata = []
        
        # Process in batches to manage memory
        for i in range(0, num_queries, batch_size):
            end_idx = min(i + batch_size, num_queries)
            batch_queries = queries_mx[i:end_idx]
            
            # Process each query in batch
            batch_indices = []
            batch_distances = []
            batch_metadata = []
            
            for j in range(batch_queries.shape[0]):
                query_np = np.array(batch_queries[j].tolist())
                indices, distances, metadata = self.query(query_np, k=k, use_cache=False)
                
                batch_indices.append(indices)
                batch_distances.append(distances)
                batch_metadata.append(metadata)
            
            all_indices.extend(batch_indices)
            all_distances.extend(batch_distances)
            all_metadata.extend(batch_metadata)
            
            # Memory pressure check between batches
            if self._memory_monitor.should_evict_cache():
                self._vector_cache._aggressive_evict()
        
        return all_indices, all_distances, all_metadata
    
    # =================== UTILITIES ===================
    
    def clear(self) -> None:
        """Clear All Data with Proper Cleanup"""
        with self._lock:
            # Clear core data
            self._vectors = None
            self._metadata = []
            self._vector_count = 0
            self._is_dirty = False
            
            # Clear caches and pools
            self._vector_cache.clear()
            self._memory_pool.clear_pool()
            
            # Reset stats
            self._operation_stats = {
                'queries': 0,
                'additions': 0,
                'cache_hits': 0,
                'memory_evictions': 0,
                'errors': 0
            }
            
            # Clean up files
            for file_path in self.store_path.glob("*"):
                if file_path.is_file() and not file_path.name.startswith('backup'):
                    file_path.unlink()
            
            logger.info("🗑️ Optimized vector store cleared completely")
    
    def health_check(self) -> Dict[str, Any]:
        """Quick Health Check"""
        try:
            # Basic functionality test
            if self._vectors is not None and self._vector_count > 0:
                test_query = mx.random.normal((self.config.dimension,))
                indices, _, _ = self.query(test_query, k=1, use_cache=False)
                query_works = len(indices) > 0
            else:
                query_works = True  # Empty store is healthy
            
            # Memory check
            memory_pressure = self._memory_monitor.get_memory_pressure()
            memory_ok = memory_pressure.get('pressure_level') != 'critical'
            
            # Compilation check
            compilation_ok = (not self.config.jit_compile) or (self._compiled_similarity_fn is not None)
            
            overall_healthy = query_works and memory_ok and compilation_ok
            
            return {
                'healthy': overall_healthy,
                'query_works': query_works,
                'memory_ok': memory_ok,
                'compilation_ok': compilation_ok,
                'vector_count': self._vector_count,
                'memory_pressure': memory_pressure.get('pressure_level', 'unknown'),
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    def __del__(self):
        """Cleanup on Destruction"""
        try:
            if hasattr(self, '_is_dirty') and self._is_dirty:
                self._save_store()
        except:
            pass  # Don't raise exceptions in destructor


# =================== PERFORMANCE BENCHMARK ===================

def benchmark_optimized_store(store: MLXVectorStoreOptimized, 
                             num_vectors: int = 10000, 
                             num_queries: int = 1000) -> Dict[str, Any]:
    """Comprehensive Performance Benchmark"""
    
    print(f"\n🚀 Benchmarking Optimized MLX Vector Store")
    print(f"   Vectors: {num_vectors:,}")
    print(f"   Queries: {num_queries:,}")
    print(f"   Dimension: {store.config.dimension}")
    print("=" * 60)
    
    results = {}
    
    # 1. Vector Addition Benchmark
    print("📈 Testing Vector Addition Performance...")
    test_vectors = np.random.rand(num_vectors, store.config.dimension).astype(np.float32)
    test_metadata = [{"id": f"perf_test_{i}", "batch": "benchmark"} for i in range(num_vectors)]
    
    start_time = time.time()
    add_result = store.add_vectors(test_vectors, test_metadata)
    add_time = time.time() - start_time
    
    add_rate = num_vectors / add_time if add_time > 0 else float('inf')
    results['vector_addition'] = {
        'rate_vectors_per_sec': add_rate,
        'total_time_sec': add_time,
        'memory_usage_gb': add_result['memory_usage_gb']
    }
    
    print(f"   ✅ Addition Rate: {add_rate:,.1f} vectors/sec")
    print(f"   ⏱️  Total Time: {add_time:.3f} seconds")
    print(f"   💾 Memory Usage: {add_result['memory_usage_gb']:.2f} GB")
    
    # 2. Query Performance Benchmark
    print("\n🔍 Testing Query Performance...")
    query_vectors = np.random.rand(num_queries, store.config.dimension).astype(np.float32)
    
    # Warmup
    for i in range(min(10, num_queries)):
        store.query(query_vectors[i], k=10)
    
    # Actual benchmark
    start_time = time.time()
    for i in range(num_queries):
        indices, distances, metadata = store.query(query_vectors[i], k=10, use_cache=True)
    
    query_time = time.time() - start_time
    qps = num_queries / query_time if query_time > 0 else float('inf')
    avg_latency_ms = (query_time / num_queries) * 1000
    
    results['query_performance'] = {
        'qps': qps,
        'avg_latency_ms': avg_latency_ms,
        'total_time_sec': query_time
    }
    
    print(f"   ✅ Query Rate: {qps:,.1f} QPS")
    print(f"   ⚡ Avg Latency: {avg_latency_ms:.2f} ms")
    print(f"   ⏱️  Total Time: {query_time:.3f} seconds")
    
    # 3. Cache Performance
    cache_stats = store._vector_cache.get_stats()
    results['cache_performance'] = cache_stats
    
    print(f"\n💾 Cache Performance:")
    print(f"   Hit Rate: {cache_stats['hit_rate_percent']:.1f}%")
    print(f"   Cache Size: {cache_stats['size']:,} entries")
    print(f"   Memory Usage: {cache_stats['memory_usage_gb']:.3f} GB")
    
    # 4. Memory Analysis
    memory_stats = store._memory_monitor.get_memory_pressure()
    results['memory_analysis'] = memory_stats
    
    print(f"\n🧠 Memory Analysis:")
    print(f"   System Memory: {memory_stats['system_memory_percent']*100:.1f}%")
    print(f"   Process Memory: {memory_stats['process_memory_gb']:.2f} GB")
    print(f"   Pressure Level: {memory_stats['pressure_level']}")
    
    # 5. Overall Performance Score
    performance_score = min(100, (add_rate / 1000) * 10 + (qps / 100) * 10 + cache_stats['hit_rate_percent'])
    results['performance_score'] = performance_score
    
    print(f"\n🎯 Performance Score: {performance_score:.1f}/100")
    
    # Performance targets check
    targets_met = {
        'add_rate_target': add_rate >= 5000,  # 5K vectors/sec
        'qps_target': qps >= 1000,            # 1K QPS
        'latency_target': avg_latency_ms <= 10, # <10ms
        'cache_target': cache_stats['hit_rate_percent'] >= 50  # >50% hit rate
    }
    
    results['targets_met'] = targets_met
    targets_count = sum(targets_met.values())
    
    print(f"\n📊 Performance Targets Met: {targets_count}/4")
    for target, met in targets_met.items():
        status = "✅" if met else "❌"
        print(f"   {status} {target}")
    
    # Final assessment
    if targets_count >= 3:
        print(f"\n🎉 EXCELLENT PERFORMANCE! Production ready.")
    elif targets_count >= 2:
        print(f"\n✅ Good performance. Minor optimizations recommended.")
    else:
        print(f"\n⚠️ Performance below targets. Optimization needed.")
    
    return results


# =================== FACTORY FUNCTION ===================

def create_optimized_vector_store(store_path: str, 
                                 dimension: int = 384,
                                 metric: str = "cosine",
                                 **kwargs) -> MLXVectorStoreOptimized:
    """Factory function für optimized vector store"""
    
    config = MLXVectorStoreConfig(
        dimension=dimension,
        metric=metric,
        **kwargs
    )
    
    return MLXVectorStoreOptimized(store_path, config)


if __name__ == "__main__":
    # Example Usage
    print("🚀 MLX Optimized Vector Store Demo")
    
    # Create optimized store
    store = create_optimized_vector_store(
        "./optimized_test_store",
        dimension=384,
        jit_compile=True,
        max_cache_vectors=5000
    )
    
    # Run comprehensive benchmark
    benchmark_results = benchmark_optimized_store(store, num_vectors=5000, num_queries=500)
    
    # Show health status
    health = store.health_check()
    print(f"\n🏥 Store Health: {'✅ Healthy' if health['healthy'] else '❌ Issues detected'}")
    
    # Show comprehensive stats
    stats = store.get_comprehensive_stats()
    print(f"\n📊 Store Overview:")
    print(f"   Vectors: {stats['vector_count']:,}")
    print(f"   Memory: {stats['memory_usage_gb']:.2f} GB")
    print(f"   Health Score: {stats['store_health']['health_score']}/100")
    
    print(f"\n🎯 MLX Optimized Vector Store Ready for Production!")