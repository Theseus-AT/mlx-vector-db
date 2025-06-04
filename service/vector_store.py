"""
MLX-Native Vector Store Core
Optimized for Apple Silicon with MLX 0.25.2

Key Optimizations:
- Zero-copy operations using unified memory
- Lazy evaluation for all array operations  
- Metal kernels for similarity search
- Native MLX serialization (NPZ format)
- JIT compilation for hot paths
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import threading
import time
from dataclasses import dataclass


@dataclass
class VectorStoreConfig:
    """Configuration for MLX Vector Store"""
    dimension: int = 384
    metric: str = "cosine"  # cosine, euclidean, dot_product
    index_type: str = "flat"  # flat, hnsw (planned)
    chunk_size: int = 1000  # for batch operations
    cache_size: int = 10000  # number of vectors to keep in memory
    use_metal: bool = True
    jit_compile: bool = True
    # HNSW support
    enable_hnsw: bool = False
    hnsw_config: Optional['HNSWConfig'] = None


class MLXVectorStore:
    """High-performance vector store optimized for Apple Silicon"""
    
    def __init__(self, store_path: str, config: VectorStoreConfig):
        self.store_path = Path(store_path)
        self.config = config
        self.lock = threading.RLock()
        
        # MLX arrays - kept in unified memory
        self._vectors: Optional[mx.array] = None
        self._vector_count = 0
        self._metadata: List[Dict] = []
        
        # Performance caches
        self._vector_cache = {}
        self._compiled_similarity_fn = None
        
        # HNSW index
        self.hnsw_index = None
        
        # Initialize storage
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._load_store()
        
        # Initialize HNSW if configured
        self._init_hnsw_if_needed()
        
        # Compile similarity functions on first use
        if self.config.jit_compile:
            self._warmup_kernels()
    
    def _init_hnsw_if_needed(self):
        """Initialize HNSW index if configured"""
        if hasattr(self.config, 'enable_hnsw') and getattr(self.config, 'enable_hnsw', False):
            try:
                from performance.hnsw_index import HNSWIndex, HNSWConfig
                hnsw_config = getattr(self.config, 'hnsw_config', HNSWConfig())
                self.hnsw_index = HNSWIndex(self.config.dimension, hnsw_config)
                print("🔍 HNSW index initialized")
            except ImportError:
                print("⚠️ HNSW module not available, using flat index")
                self.hnsw_index = None
    
    def _warmup_kernels(self):
        """Pre-compile MLX kernels for optimal performance"""
        print("🔥 Warming up MLX kernels...")
        
        # Create dummy data for compilation
        dummy_vectors = mx.random.normal((100, self.config.dimension))
        dummy_query = mx.random.normal((self.config.dimension,))
        
        # Trigger compilation by running operations
        if self.config.metric == "cosine":
            self._cosine_similarity_batch(dummy_query, dummy_vectors)
        elif self.config.metric == "euclidean":
            self._euclidean_distance_batch(dummy_query, dummy_vectors)
        
        # Force evaluation to complete compilation
        mx.eval(dummy_vectors)
        print("✅ MLX kernels ready")
    
    @mx.compile  # JIT compile for performance
    def _cosine_similarity_batch(self, query: mx.array, vectors: mx.array) -> mx.array:
        """Optimized batch cosine similarity using MLX"""
        # Normalize query vector (lazy evaluation)
        query_norm = query / mx.linalg.norm(query)
        
        # Normalize all vectors at once (vectorized)
        vectors_norm = vectors / mx.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Batch dot product (Metal accelerated)
        similarities = mx.dot(vectors_norm, query_norm)
        
        return similarities
    
    @mx.compile
    def _euclidean_distance_batch(self, query: mx.array, vectors: mx.array) -> mx.array:
        """Optimized batch euclidean distance using MLX"""
        # Broadcasting subtraction
        diff = vectors - query[None, :]  # Shape: (n_vectors, dim)
        
        # Squared distances (vectorized)
        distances = mx.sum(diff * diff, axis=1)
        
        # Return negative distances for consistent "higher is better" semantics
        return -mx.sqrt(distances)
    
    @mx.compile  
    def _dot_product_batch(self, query: mx.array, vectors: mx.array) -> mx.array:
        """Optimized batch dot product using MLX"""
        return mx.dot(vectors, query)
    
    def _get_similarity_fn(self):
        """Get the compiled similarity function"""
        if self.config.metric == "cosine":
            return self._cosine_similarity_batch
        elif self.config.metric == "euclidean":
            return self._euclidean_distance_batch
        elif self.config.metric == "dot_product":
            return self._dot_product_batch
        else:
            raise ValueError(f"Unsupported metric: {self.config.metric}")
    
    def add_vectors(self, vectors: Union[np.ndarray, List[List[float]]], 
                   metadata: List[Dict]) -> None:
        """Add vectors to the store with MLX optimization"""
        with self.lock:
            # Convert to MLX array (zero-copy if possible)
            if isinstance(vectors, np.ndarray):
                new_vectors = mx.array(vectors.astype(np.float32))
            else:
                new_vectors = mx.array(np.array(vectors, dtype=np.float32))
            
            # Validate dimensions
            if new_vectors.shape[1] != self.config.dimension:
                raise ValueError(f"Vector dimension {new_vectors.shape[1]} != {self.config.dimension}")
            
            # Concatenate with existing vectors (lazy operation)
            if self._vectors is None:
                self._vectors = new_vectors
            else:
                self._vectors = mx.concatenate([self._vectors, new_vectors], axis=0)
            
            # Update metadata
            self._metadata.extend(metadata)
            self._vector_count += len(metadata)
            
            # Update HNSW index if available
            if self.hnsw_index:
                if self.hnsw_index.n_points == 0:
                    self.hnsw_index.build(self._vectors, show_progress=False)
                else:
                    self.hnsw_index.extend_vectors(new_vectors)
            
            # Save to disk (async-friendly)
            self._save_store()
            
            print(f"✅ Added {len(metadata)} vectors. Total: {self._vector_count}")
    
    def query(self, query_vector: Union[np.ndarray, List[float]], 
              k: int = 10, 
              filter_metadata: Optional[Dict] = None,
              use_hnsw: Optional[bool] = None) -> Union[List[Tuple[Dict, float]], Tuple[List[int], List[float], List[Dict]]]:
        """High-performance similarity search - returns (indices, distances, metadata) for API compatibility"""
        if self._vectors is None or self._vector_count == 0:
            return [], [], []
        
        with self.lock:
            # Convert query to MLX array
            if isinstance(query_vector, np.ndarray):
                query_mx = mx.array(query_vector.astype(np.float32))
            else:
                query_mx = mx.array(np.array(query_vector, dtype=np.float32))
            
            # Apply metadata filtering if needed
            valid_indices = self._apply_metadata_filter(filter_metadata)
            
            # Use HNSW if available and requested
            if use_hnsw is not False and self.hnsw_index and self.hnsw_index.n_points > 0:
                indices_mx, distances_mx = self.hnsw_index.search(query_mx, k)
                indices = indices_mx.tolist()
                distances = distances_mx.tolist()
            else:
                # Get similarity function
                similarity_fn = self._get_similarity_fn()
                
                # Compute similarities (Metal accelerated)
                similarities = similarity_fn(query_mx, self._vectors)
                
                # Convert to numpy for indexing
                similarities_np = np.array(similarities)
                
                if valid_indices is not None:
                    # Mask similarities for filtered results
                    filtered_similarities = similarities_np[valid_indices]
                    filtered_indices = valid_indices
                else:
                    filtered_similarities = similarities_np
                    filtered_indices = np.arange(len(similarities_np))
                
                # Get top-k indices
                if len(filtered_similarities) <= k:
                    top_indices = np.arange(len(filtered_similarities))
                else:
                    top_indices = np.argpartition(filtered_similarities, -k)[-k:]
                    top_indices = top_indices[np.argsort(filtered_similarities[top_indices])[::-1]]
                
                indices = [int(filtered_indices[idx]) for idx in top_indices]
                # Convert similarities to distances
                if self.config.metric == "cosine":
                    distances = [float(1.0 - filtered_similarities[idx]) for idx in top_indices]
                else:
                    distances = [float(-filtered_similarities[idx]) for idx in top_indices]
            
            # Build metadata list
            metadata_list = [self._metadata[idx] for idx in indices]
            
            # Check if we should return new format (for API) or old format (for backward compatibility)
            # Return new format by default
            return indices, distances, metadata_list
    
    @property
    def dimension(self) -> int:
        """Get vector dimension"""
        return self.config.dimension

    @property
    def vectors(self) -> Optional[mx.array]:
        """Get vectors array"""
        return self._vectors

    @property
    def metadata(self) -> List[Dict]:
        """Get metadata list"""
        return self._metadata
    
    def get_metadata(self, index: int) -> Dict:
        """Get metadata for a specific index"""
        with self.lock:
            if 0 <= index < len(self._metadata):
                return self._metadata[index]
            return {}
    
    def delete_vectors(self, indices: List[int]) -> int:
        """Delete vectors by indices"""
        with self.lock:
            if not indices or self._vectors is None:
                return 0
            
            # Sortiere Indizes absteigend für korrektes Löschen
            indices_to_delete = sorted(set(indices), reverse=True)
            deleted_count = 0
            
            # Konvertiere zu numpy für einfacheres Löschen
            vectors_np = np.array(self._vectors)
            mask = np.ones(len(vectors_np), dtype=bool)
            
            for idx in indices_to_delete:
                if 0 <= idx < len(self._metadata):
                    mask[idx] = False
                    deleted_count += 1
            
            # Update vectors
            remaining_vectors = vectors_np[mask]
            if len(remaining_vectors) > 0:
                self._vectors = mx.array(remaining_vectors)
                # Update metadata
                self._metadata = [meta for i, meta in enumerate(self._metadata) if mask[i]]
            else:
                self._vectors = None
                self._metadata = []
            
            self._vector_count -= deleted_count
            
            # Save changes
            self._save_store()
            
            return deleted_count
    
    def _apply_metadata_filter(self, filter_metadata: Optional[Dict]) -> Optional[np.ndarray]:
        """Apply metadata filtering to get valid indices"""
        if filter_metadata is None:
            return None
        
        valid_indices = []
        for i, metadata in enumerate(self._metadata):
            if all(metadata.get(k) == v for k, v in filter_metadata.items()):
                valid_indices.append(i)
        
        return np.array(valid_indices) if valid_indices else np.array([])
    
    def batch_query(self, query_vectors: Union[np.ndarray, List[List[float]]], 
                   k: int = 10) -> Tuple[List[List[int]], List[List[float]], List[List[Dict]]]:
        """Batch query processing for maximum throughput"""
        if isinstance(query_vectors, list):
            query_vectors = np.array(query_vectors, dtype=np.float32)
        
        # Convert to MLX
        queries_mx = mx.array(query_vectors)
        
        all_indices = []
        all_distances = []
        all_metadata = []
        
        for i in range(queries_mx.shape[0]):
            indices, distances, metadata = self.query(np.array(queries_mx[i]), k=k)
            all_indices.append(indices)
            all_distances.append(distances)
            all_metadata.append(metadata)
        
        return all_indices, all_distances, all_metadata
    
    def _save_store(self):
        """Save vectors and metadata to disk using MLX NPZ format"""
        if self._vectors is not None:
            # Save vectors in MLX native format
            vectors_path = self.store_path / "vectors.npz"
            mx.savez(str(vectors_path), vectors=self._vectors)
            
            # Save metadata as JSONL
            metadata_path = self.store_path / "metadata.jsonl"
            with open(metadata_path, 'w') as f:
                for meta in self._metadata:
                    f.write(json.dumps(meta) + '\n')
            
            # Save config
            config_path = self.store_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'dimension': self.config.dimension,
                    'metric': self.config.metric,
                    'vector_count': self._vector_count
                }, f)
    
    def _load_store(self):
        """Load existing store from disk"""
        vectors_path = self.store_path / "vectors.npz"
        metadata_path = self.store_path / "metadata.jsonl"
        config_path = self.store_path / "config.json"
        
        if not vectors_path.exists():
            print("🆕 Creating new vector store")
            return
        
        try:
            # Load config
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    self._vector_count = config_data.get('vector_count', 0)
            
            # Load vectors (lazy loading - won't materialize until needed)
            vectors_data = mx.load(str(vectors_path))
            self._vectors = vectors_data['vectors']
            
            # Load metadata
            self._metadata = []
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            self._metadata.append(json.loads(line.strip()))
            
            print(f"📂 Loaded store with {self._vector_count} vectors")
            
        except Exception as e:
            print(f"⚠️ Error loading store: {e}")
            self._vectors = None
            self._metadata = []
            self._vector_count = 0
    
    def get_stats(self) -> Dict:
        """Get performance and storage statistics"""
        return {
            'vector_count': self._vector_count,
            'dimension': self.config.dimension,
            'metric': self.config.metric,
            'memory_usage_mb': self._get_memory_usage(),
            'index_type': self.config.index_type,
            'mlx_device': str(mx.default_device()),
            'unified_memory': True
        }
    
    def _get_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if self._vectors is None:
            return 0.0
        
        # MLX arrays use unified memory
        vector_size = self._vector_count * self.config.dimension * 4  # float32
        metadata_size = len(json.dumps(self._metadata).encode())
        
        return (vector_size + metadata_size) / (1024 * 1024)
    
    def optimize(self):
        """Optimize the vector store for better performance"""
        with self.lock:
            if self._vectors is None:
                return
            
            print("🔧 Optimizing vector store...")
            
            # Force evaluation of lazy operations
            mx.eval(self._vectors)
            
            # Rebuild HNSW index if available
            if self.hnsw_index and self._vectors is not None:
                self.hnsw_index.build(self._vectors, show_progress=False)
            
            print("✅ Optimization complete")
    
    def clear(self):
        """Clear all vectors and metadata"""
        with self.lock:
            self._vectors = None
            self._metadata = []
            self._vector_count = 0
            self.hnsw_index = None
            
            # Clean up files
            for file_path in self.store_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            
            print("🗑️ Vector store cleared")


# Performance testing utilities
def benchmark_vector_store(store: MLXVectorStore, num_vectors: int = 1000, num_queries: int = 100):
    """Benchmark the vector store performance"""
    print(f"🚀 Benchmarking with {num_vectors} vectors, {num_queries} queries")
    
    # Generate test data
    test_vectors = np.random.rand(num_vectors, store.config.dimension).astype(np.float32)
    test_metadata = [{"id": f"test_{i}", "category": f"cat_{i%10}"} for i in range(num_vectors)]
    
    # Benchmark addition
    start_time = time.time()
    store.add_vectors(test_vectors, test_metadata)
    add_time = time.time() - start_time
    add_rate = num_vectors / add_time
    
    # Benchmark queries
    query_vectors = np.random.rand(num_queries, store.config.dimension).astype(np.float32)
    
    start_time = time.time()
    for query in query_vectors:
        results = store.query(query, k=10)
    query_time = time.time() - start_time
    qps = num_queries / query_time
    
    # Print results
    print(f"📈 Performance Results:")
    print(f"   Vector Addition: {add_rate:.1f} vectors/sec")
    print(f"   Query Performance: {qps:.1f} QPS")
    print(f"   Average Query Latency: {(query_time/num_queries)*1000:.2f}ms")
    print(f"   Memory Usage: {store.get_stats()['memory_usage_mb']:.2f} MB")
    
    return {
        'add_rate': add_rate,
        'qps': qps,
        'avg_latency_ms': (query_time/num_queries) * 1000,
        'memory_mb': store.get_stats()['memory_usage_mb']
    }


if __name__ == "__main__":
    # Example usage
    config = VectorStoreConfig(
        dimension=384,
        metric="cosine",
        jit_compile=True
    )
    
    store = MLXVectorStore("./test_store", config)
    
    # Run benchmark
    results = benchmark_vector_store(store, num_vectors=10000, num_queries=1000)
    
    print("\n🎯 MLX Vector Store Ready for Production!")