"""
MLX Vector Store - Production-Ready mit aktuellen MLX APIs
Korrigierte Version basierend auf github.com/ml-explore/mlx
"""

import mlx.core as mx
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
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

logger = logging.getLogger("mlx_vector_db.optimized_store")

# Compiled MLX functions with current API
@mx.compile
def _compiled_cosine_similarity(query: mx.array, vectors: mx.array) -> mx.array:
    """Optimierte Cosine Similarity mit aktueller MLX API"""
    if query.ndim == 1:
        query = query[None, :]

    query_norm = mx.linalg.norm(query, axis=1, keepdims=True)
    vectors_norm = mx.linalg.norm(vectors, axis=1, keepdims=True)
    
    eps = 1e-8
    query_norm = mx.maximum(query_norm, eps)
    vectors_norm = mx.maximum(vectors_norm, eps)
    
    query_normalized = query / query_norm
    vectors_normalized = vectors / vectors_norm
    
    return mx.matmul(vectors_normalized, query_normalized.T).flatten()

@mx.compile
def _compiled_euclidean_distance(query: mx.array, vectors: mx.array) -> mx.array:
    """Optimierte Euclidean Distance mit aktueller MLX API"""
    if query.ndim == 1:
        query = query[None, :]
    diff = vectors - query
    distances_sq = mx.sum(diff * diff, axis=1)
    return mx.sqrt(distances_sq)

@mx.compile  
def _compiled_dot_product(query: mx.array, vectors: mx.array) -> mx.array:
    """Optimiertes Dot Product mit aktueller MLX API"""
    if query.ndim == 1:
        return mx.matmul(vectors, query)
    return mx.matmul(vectors, query.T).flatten()

@dataclass
class MLXVectorStoreConfig:
    """Configuration for MLX Vector Store"""
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
    hnsw_config: Optional[Any] = None

class MLXVectorStore:
    """Production-ready MLX Vector Store mit aktueller MLX API"""
    
    def __init__(self, store_path: str, config: Optional[MLXVectorStoreConfig] = None):
        self.store_path = Path(store_path).expanduser()
        self.config = config or MLXVectorStoreConfig()
        self._lock = threading.RLock()
        
        # Core data
        self._vectors: Optional[mx.array] = None
        self._vector_count = 0
        self._metadata: List[Dict] = []
        self._is_dirty = False
        
        # Performance tracking
        self._operation_stats = defaultdict(int)
        self._compiled_similarity_fn = None
        
        # HNSW Index (optional)
        self._hnsw_index = None
        if config and config.enable_hnsw:
            self._initialize_hnsw()
        
        # Initialize store
        self._initialize_store()
        
        # Compile functions if enabled
        if self.config.jit_compile:
            self._compile_critical_functions()
        
        logger.info(f"MLX Vector Store initialized: {self.store_path}")
    
    def _initialize_hnsw(self):
        """Initialize HNSW index if available"""
        try:
            from performance.hnsw_index import ProductionHNSWIndex, AdaptiveHNSWConfig
            
            hnsw_config = self.config.hnsw_config
            if hnsw_config is None:
                hnsw_config = AdaptiveHNSWConfig(metric=self.config.metric)
            elif not hasattr(hnsw_config, 'metric'):
                hnsw_config.metric = self.config.metric
                
            self._hnsw_index = ProductionHNSWIndex(self.config.dimension, hnsw_config)
            logger.info("HNSW index initialized")
        except ImportError:
            logger.warning("HNSW index not available")
            self._hnsw_index = None
        except Exception as e:
            logger.error(f"HNSW initialization failed: {e}")
            self._hnsw_index = None
    
    def _initialize_store(self):
        """Initialize store directory and load existing data"""
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.config.max_retry_attempts):
            try:
                self._load_store()
                logger.info(f"Store loaded successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.error(f"Store load attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retry_attempts - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    if self.config.auto_recovery:
                        self._create_empty_store()
    
    def _create_empty_store(self):
        """Create empty store as recovery fallback"""
        logger.info("Creating empty store as recovery fallback")
        self._vectors = None
        self._metadata = []
        self._vector_count = 0
        self._is_dirty = False
    
    def _compile_critical_functions(self):
        """Setup MLX compiled functions for optimal performance"""
        logger.info("Compiling MLX functions for optimal performance...")
        try:
            if self.config.metric == "cosine":
                self._compiled_similarity_fn = _compiled_cosine_similarity
            elif self.config.metric == "euclidean":
                self._compiled_similarity_fn = _compiled_euclidean_distance
            elif self.config.metric == "dot_product":
                self._compiled_similarity_fn = _compiled_dot_product
            
            if self._compiled_similarity_fn:
                # Warmup compilation
                dummy_query = mx.random.normal((self.config.dimension,))
                dummy_vectors = mx.random.normal((10, self.config.dimension))
                result = self._compiled_similarity_fn(dummy_query, dummy_vectors)
                mx.eval(result)
                logger.info("✅ MLX function compilation successful")
        except Exception as e:
            logger.error(f"MLX compilation failed: {e}")
            self._compiled_similarity_fn = None
    
    def add_vectors(self, vectors: Union[np.ndarray, List[List[float]], mx.array], 
                   metadata: List[Dict]) -> Dict[str, Any]:
        """Add vectors to the store"""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        with self._lock:
            # Convert to MLX array
            new_vectors = self._convert_to_mlx_array(vectors)
            
            # Validate dimensions
            if new_vectors.shape[1] != self.config.dimension:
                raise ValueError(f"Vector dimension {new_vectors.shape[1]} doesn't match config dimension {self.config.dimension}")
            
            # Add to store
            if self._vectors is None:
                self._vectors = new_vectors
            else:
                self._vectors = mx.concatenate([self._vectors, new_vectors], axis=0)
            
            self._metadata.extend(metadata)
            self._vector_count = self._vectors.shape[0]
            self._is_dirty = True
            
            # Force evaluation for consistency
            mx.eval(self._vectors)
            
            # Save to disk
            self._schedule_save()
            
            # Update HNSW index if available
            if self._hnsw_index and hasattr(self._hnsw_index, 'build'):
                try:
                    self._hnsw_index.build(self._vectors)
                except Exception as e:
                    logger.warning(f"HNSW index update failed: {e}")
            
            logger.info(f"Added {len(metadata)} vectors (total: {self._vector_count})")
            return {
                'vectors_added': len(metadata),
                'total_vectors': self._vector_count
            }
    
    def query(self, query_vector: Union[np.ndarray, List[float], mx.array], 
              k: int = 10, filter_metadata: Optional[Dict] = None,
              use_hnsw: bool = True, use_cache: bool = True) -> Tuple[List[int], List[float], List[Dict]]:
        """Query similar vectors"""
        if self._vectors is None or self._vector_count == 0:
            return [], [], []
        
        # Convert query to MLX array
        query_mx = self._convert_to_mlx_array(query_vector)
        if query_mx.ndim == 2:
            query_mx = query_mx.flatten()
        
        # Try HNSW first if available and ready
        if (use_hnsw and self._hnsw_index and 
            hasattr(self._hnsw_index, 'state') and 
            getattr(self._hnsw_index, 'state', None) == "ready"):
            try:
                indices, distances = self._hnsw_index.search(query_mx, k=k)
                metadata_list = [self._metadata[i] for i in indices if i < len(self._metadata)]
                return indices, distances, metadata_list
            except Exception as e:
                logger.warning(f"HNSW search failed, falling back to brute force: {e}")
        
        # Brute force search
        with self._lock:
            if self._compiled_similarity_fn:
                similarities = self._compiled_similarity_fn(query_mx, self._vectors)
            else:
                similarities = self._fallback_similarity(query_mx, self._vectors)
            
            # Force evaluation
            mx.eval(similarities)
            
            # Apply metadata filter if provided
            valid_indices = None
            if filter_metadata:
                valid_indices = self._apply_metadata_filter(filter_metadata)
                if not valid_indices:
                    return [], [], []
            
            return self._get_top_k_results(similarities, k, valid_indices)
    
    def batch_query(self, query_vectors: Union[np.ndarray, List[List[float]], mx.array], 
                   k: int = 10) -> List[Tuple[List[int], List[float], List[Dict]]]:
        """Batch query multiple vectors"""
        queries_mx = self._convert_to_mlx_array(query_vectors)
        
        results = []
        for i in range(queries_mx.shape[0]):
            result = self.query(queries_mx[i], k=k, use_hnsw=False)  # Use brute force for consistency
            results.append(result)
        
        return results
    
    def _apply_metadata_filter(self, filter_metadata: Dict) -> List[int]:
        """Apply metadata filter and return valid indices"""
        valid_indices = []
        
        for i, meta in enumerate(self._metadata):
            match = True
            for key, value in filter_metadata.items():
                if meta.get(key) != value:
                    match = False
                    break
            if match:
                valid_indices.append(i)
        
        return valid_indices
    
    def _fallback_similarity(self, query: mx.array, vectors: mx.array) -> mx.array:
        """Fallback similarity computation without compilation"""
        if self.config.metric == "cosine":
            query_norm = mx.linalg.norm(query)
            query_normalized = query / mx.maximum(query_norm, 1e-8)
            
            vectors_norm = mx.linalg.norm(vectors, axis=1, keepdims=True)
            vectors_normalized = vectors / mx.maximum(vectors_norm, 1e-8)
            
            return mx.matmul(vectors_normalized, query_normalized)
        
        elif self.config.metric == "euclidean":
            diff = vectors - query[None, :]
            distances = mx.sqrt(mx.sum(diff * diff, axis=1))
            return -distances  # Negative for sorting
        
        else:  # dot_product
            return mx.matmul(vectors, query)
    
    def _get_top_k_results(self, similarities: mx.array, k: int, 
                          valid_indices: Optional[List[int]]) -> Tuple[List[int], List[float], List[Dict]]:
        """Get top k results from similarities"""
        
        # Apply filter if provided
        if valid_indices is not None:
            if not valid_indices:
                return [], [], []
            
            # Filter similarities
            valid_similarities = similarities[valid_indices]
            
            # Sort and get top k
            if self.config.metric == "euclidean":
                sorted_local_indices = mx.argsort(valid_similarities)  # Ascending for negative distances
            else:
                sorted_local_indices = mx.argsort(-valid_similarities)  # Descending for similarities
            
            k = min(k, len(valid_indices))
            top_local_indices = sorted_local_indices[:k]
            
            # Map back to global indices
            top_indices = [valid_indices[int(i)] for i in top_local_indices.tolist()]
            top_scores = valid_similarities[top_local_indices]
        else:
            # No filter - use all vectors
            if self.config.metric == "euclidean":
                sorted_indices = mx.argsort(similarities)  # Ascending for negative distances
            else:
                sorted_indices = mx.argsort(-similarities)  # Descending for similarities
            
            k = min(k, self._vector_count)
            top_indices_mx = sorted_indices[:k]
            top_indices = [int(i) for i in top_indices_mx.tolist()]
            top_scores = similarities[top_indices_mx]
        
        # Convert scores to distances
        scores_list = top_scores.tolist()
        
        if self.config.metric == 'cosine':
            distances = [1.0 - s for s in scores_list]
        elif self.config.metric == 'euclidean':
            distances = [-s for s in scores_list]  # Convert back to positive distances
        else:  # dot_product
            distances = [-s for s in scores_list]  # Convert to distances
        
        # Get metadata
        metadata_list = [self._metadata[i] for i in top_indices if i < len(self._metadata)]
        
        return top_indices, distances, metadata_list
    
    def _convert_to_mlx_array(self, vectors: Union[np.ndarray, List[List[float]], mx.array]) -> mx.array:
        """Convert input vectors to MLX array"""
        if isinstance(vectors, mx.array):
            return vectors
        elif isinstance(vectors, np.ndarray):
            return mx.array(vectors.astype(np.float32))
        else:
            return mx.array(np.array(vectors, dtype=np.float32))
    
    def _schedule_save(self):
        """Schedule save operation"""
        if self._is_dirty:
            try:
                self._save_store()
                self._is_dirty = False
            except Exception as e:
                logger.error(f"Scheduled save failed: {e}")
    
    def _save_store(self):
        """Save store to disk using current MLX API"""
        if self._vectors is None:
            return
        
        vectors_path = None
        metadata_path = None
        temp_vectors_path = None
        temp_metadata_path = None
        
        try:
            # Ensure directory exists
            self.store_path.mkdir(parents=True, exist_ok=True)
            
            # Save vectors with atomic write
            vectors_path = self.store_path / "vectors.npz"
            temp_vectors_path = self.store_path / "vectors.npz.tmp"
            
            # Convert to numpy for saving (current MLX may not have direct npz save)
            vectors_np = np.array(self._vectors.tolist())
            np.savez(str(temp_vectors_path), vectors=vectors_np)
            temp_vectors_path.rename(vectors_path)
            
            # Save metadata with atomic write
            metadata_path = self.store_path / "metadata.jsonl"
            temp_metadata_path = self.store_path / "metadata.jsonl.tmp"
            
            with open(temp_metadata_path, 'w') as f:
                for meta in self._metadata:
                    f.write(json.dumps(meta) + '\n')
            
            temp_metadata_path.rename(metadata_path)
            
            logger.debug(f"Store saved: {self._vector_count} vectors")
            
        except Exception as e:
            logger.error(f"Store save failed: {e}")
            # Clean up temp files
            if temp_vectors_path and temp_vectors_path.exists():
                try:
                    temp_vectors_path.unlink()
                except:
                    pass
            if temp_metadata_path and temp_metadata_path.exists():
                try:
                    temp_metadata_path.unlink()
                except:
                    pass
            raise
    
    def _load_store(self):
        """Load store from disk"""
        vectors_path = self.store_path / "vectors.npz"
        metadata_path = self.store_path / "metadata.jsonl"
        
        if not vectors_path.exists():
            logger.info("Creating new vector store")
            return
        
        try:
            # Load vectors using numpy then convert to MLX
            data = np.load(str(vectors_path))
            if 'vectors' in data:
                vectors_np = data['vectors']
                self._vectors = mx.array(vectors_np)
                self._vector_count = self._vectors.shape[0]
                logger.info(f"Loaded {self._vector_count} vectors")
            
            # Load metadata
            if metadata_path.exists():
                self._metadata = []
                with open(metadata_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            self._metadata.append(json.loads(line))
                logger.info(f"Loaded {len(self._metadata)} metadata entries")
            
            # Validate consistency
            if len(self._metadata) != self._vector_count:
                logger.warning(f"Metadata count {len(self._metadata)} doesn't match vector count {self._vector_count}")
            
        except Exception as e:
            logger.error(f"Store loading failed: {e}")
            self._create_empty_store()
    
    def optimize(self):
        """Optimize store performance"""
        if self.config.enable_hnsw and self._hnsw_index and self._vectors is not None:
            try:
                logger.info("Building HNSW index for optimization...")
                self._hnsw_index.build(self._vectors)
                logger.info("HNSW optimization complete")
            except Exception as e:
                logger.warning(f"HNSW optimization failed: {e}")
        
        # Force evaluation of any pending MLX operations
        if self._vectors is not None:
            mx.eval(self._vectors)
        
        # Trigger garbage collection
        gc.collect()
        
        logger.info("Store optimization complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            'vector_count': self._vector_count,
            'dimension': self.config.dimension,
            'metric': self.config.metric,
            'memory_usage_mb': self._estimate_memory_usage(),
            'index_type': 'hnsw' if self._hnsw_index else 'flat',
            'compiled_functions': self._compiled_similarity_fn is not None
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics (alias for compatibility)"""
        stats = self.get_stats()
        stats.update({
            'store_path': str(self.store_path),
            'config': self.config.__dict__,
            'operation_stats': dict(self._operation_stats)
        })
        return stats
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if self._vectors is None:
            return 0.0
        
        # Vector memory (float32 = 4 bytes per element)
        vector_memory = self._vectors.size * 4
        
        # Metadata memory (rough estimate)
        metadata_memory = len(self._metadata) * 100  # ~100 bytes per metadata entry
        
        # Additional overhead
        overhead = 1024 * 1024  # 1MB overhead
        
        total_bytes = vector_memory + metadata_memory + overhead
        return total_bytes / (1024 * 1024)
    
    def clear(self):
        """Clear all data"""
        with self._lock:
            self._vectors = None
            self._metadata = []
            self._vector_count = 0
            self._is_dirty = False
            logger.info("Store cleared")
    
    def delete_vectors(self, indices: List[int]) -> int:
        """Delete vectors by indices (placeholder implementation)"""
        logger.warning("delete_vectors is not fully implemented yet")
        return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        issues = []
        
        # Check basic state
        if self._vectors is None:
            issues.append("No vectors loaded")
        
        # Check metadata consistency
        if len(self._metadata) != self._vector_count:
            issues.append("Metadata/vector count mismatch")
        
        # Check store path
        if not self.store_path.exists():
            issues.append("Store directory doesn't exist")
        
        return {
            'healthy': len(issues) == 0,
            'vector_count': self._vector_count,
            'memory_usage_mb': self._estimate_memory_usage(),
            'issues': issues
        }
    
    def _warmup_kernels(self):
        """Warmup MLX kernels"""
        logger.info("Warming up MLX kernels...")
        try:
            if self._vectors is not None and self._vector_count > 0:
                # Test with actual vectors
                test_size = min(10, self._vector_count)
                dummy_query = self._vectors[0]
                test_vectors = self._vectors[:test_size]
                
                if self._compiled_similarity_fn:
                    result = self._compiled_similarity_fn(dummy_query, test_vectors)
                    mx.eval(result)
            else:
                # Test with dummy data
                dummy_query = mx.random.normal((self.config.dimension,))
                dummy_vectors = mx.random.normal((10, self.config.dimension))
                
                if self._compiled_similarity_fn:
                    result = self._compiled_similarity_fn(dummy_query, dummy_vectors)
                    mx.eval(result)
            
            logger.info("Kernel warmup completed")
        except Exception as e:
            logger.warning(f"Kernel warmup failed: {e}")

# Factory function
def create_optimized_vector_store(store_path: str, dimension: int = 384, 
                                 jit_compile: bool = True, enable_hnsw: bool = False,
                                 **kwargs) -> MLXVectorStore:
    """Factory function for creating optimized vector store"""
    config = MLXVectorStoreConfig(
        dimension=dimension,
        jit_compile=jit_compile,
        enable_hnsw=enable_hnsw,
        **kwargs
    )
    return MLXVectorStore(store_path, config)