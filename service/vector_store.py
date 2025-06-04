"""
Updated VectorStore with HNSW Index Integration
Maintains backward compatibility while adding high-performance search
"""

import mlx.core as mx
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Any
import threading
import time
from dataclasses import dataclass
import logging

# Import HNSW implementation
from performance.hnsw_index import HNSWIndex, HNSWConfig

logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Configuration for VectorStore"""
    enable_hnsw: bool = True
    hnsw_config: HNSWConfig = None
    auto_index_threshold: int = 1000  # Build HNSW when vectors exceed this
    cache_enabled: bool = True
    cache_size: int = 1000
    
    def __post_init__(self):
        if self.hnsw_config is None:
            self.hnsw_config = HNSWConfig()

class VectorStore:
    """
    MLX-optimized vector storage with optional HNSW indexing
    Maintains compatibility with existing API
    """
    
    def __init__(self, store_path: Path, config: VectorStoreConfig = None):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.config = config or VectorStoreConfig()
        
        # Core storage
        self.vectors: Optional[mx.array] = None
        self.metadata: List[Dict[str, Any]] = []
        
        # HNSW index
        self.hnsw_index: Optional[HNSWIndex] = None
        self._index_lock = threading.Lock()
        
        # Query cache
        self._query_cache = {} if self.config.cache_enabled else None
        self._cache_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'hnsw_queries': 0,
            'brute_force_queries': 0,
            'cache_hits': 0,
            'avg_query_time': 0
        }
        
        # Load existing data
        self._load()
        
    def _load(self):
        """Load vectors and metadata from disk"""
        vectors_path = self.store_path / "vectors.npz"
        metadata_path = self.store_path / "metadata.jsonl"
        
        # Load vectors
        if vectors_path.exists():
            data = np.load(vectors_path)
            self.vectors = mx.array(data['vectors'])
            logger.info(f"Loaded {self.vectors.shape[0]} vectors")
            
        # Load metadata
        if metadata_path.exists():
            self.metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    self.metadata.append(json.loads(line.strip()))
                    
        # Load or build HNSW index if enabled
        if self.config.enable_hnsw and self.vectors is not None:
            self._load_or_build_index()
            
    def _load_or_build_index(self):
        """Load existing HNSW index or build new one if needed"""
        index_path = self.store_path / "hnsw_index"
        
        if index_path.with_suffix('.graph.pkl').exists():
            try:
                # Load existing index
                self.hnsw_index = HNSWIndex(self.config.hnsw_config)
                self.hnsw_index.load(str(index_path))
                self.hnsw_index.vectors = self.vectors
                logger.info("Loaded existing HNSW index")
            except Exception as e:
                logger.warning(f"Failed to load HNSW index: {e}")
                self._build_index()
        else:
            # Build new index if threshold met
            if len(self.vectors) >= self.config.auto_index_threshold:
                self._build_index()
                
    def _build_index(self):
        """Build new HNSW index"""
        if self.vectors is None or len(self.vectors) == 0:
            return
            
        logger.info(f"Building HNSW index for {len(self.vectors)} vectors...")
        
        with self._index_lock:
            self.hnsw_index = HNSWIndex(self.config.hnsw_config)
            self.hnsw_index.build(self.vectors, show_progress=True)
            
            # Save index
            index_path = self.store_path / "hnsw_index"
            self.hnsw_index.save(str(index_path))
            
        logger.info("HNSW index built successfully")
        
    def add_vectors(self, vectors: mx.array, metadata: List[Dict[str, Any]]):
        """Add vectors with metadata"""
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
            
        # Initialize if empty
        if self.vectors is None:
            self.vectors = vectors
            self.metadata = metadata
        else:
            # Append to existing
            self.vectors = mx.concatenate([self.vectors, vectors], axis=0)
            self.metadata.extend(metadata)
            
        # Update HNSW index if enabled
        if self.config.enable_hnsw and self.hnsw_index is not None:
            start_idx = len(self.vectors) - len(vectors)
            with self._index_lock:
                for i in range(len(vectors)):
                    self.hnsw_index._insert(start_idx + i)
                    
        # Check if we should build index
        elif (self.config.enable_hnsw and 
              self.hnsw_index is None and 
              len(self.vectors) >= self.config.auto_index_threshold):
            self._build_index()
            
        # Save to disk
        self._save()
        
        # Clear cache
        if self._query_cache is not None:
            with self._cache_lock:
                self._query_cache.clear()
                
    def _save(self):
        """Save vectors and metadata to disk"""
        if self.vectors is not None:
            # Save vectors as NPZ
            vectors_path = self.store_path / "vectors.npz"
            np.savez_compressed(vectors_path, vectors=np.array(self.vectors))
            
        # Save metadata as JSONL
        metadata_path = self.store_path / "metadata.jsonl"
        with open(metadata_path, 'w') as f:
            for item in self.metadata:
                f.write(json.dumps(item) + '\n')
                
        # Save HNSW index if exists
        if self.hnsw_index is not None:
            index_path = self.store_path / "hnsw_index"
            self.hnsw_index.save(str(index_path))
            
    def query(self, query_vector: mx.array, k: int = 10, 
             use_hnsw: Optional[bool] = None) -> Tuple[List[int], List[float], List[Dict]]:
        """
        Query for k nearest neighbors
        
        Args:
            query_vector: Query vector
            k: Number of nearest neighbors
            use_hnsw: Force HNSW usage (None=auto, True=force, False=brute force)
            
        Returns:
            Tuple of (indices, distances, metadata)
        """
        if self.vectors is None or len(self.vectors) == 0:
            return [], [], []
            
        # Update metrics
        self.metrics['total_queries'] += 1
        start_time = time.time()
        
        # Check cache
        cache_key = None
        if self._query_cache is not None:
            cache_key = self._compute_cache_key(query_vector, k)
            with self._cache_lock:
                if cache_key in self._query_cache:
                    self.metrics['cache_hits'] += 1
                    return self._query_cache[cache_key]
                    
        # Determine search method
        if use_hnsw is None:
            use_hnsw = self.hnsw_index is not None
            
        if use_hnsw and self.hnsw_index is not None:
            # Use HNSW index
            indices, distances = self.hnsw_index.search(query_vector, k)
            self.metrics['hnsw_queries'] += 1
        else:
            # Fallback to brute force
            indices, distances = self._brute_force_search(query_vector, k)
            self.metrics['brute_force_queries'] += 1
            
        # Convert to lists
        indices_list = indices.tolist()
        distances_list = distances.tolist()
        
        # Get metadata
        metadata_list = [self.metadata[i] for i in indices_list]
        
        # Update cache
        if self._query_cache is not None and cache_key is not None:
            with self._cache_lock:
                self._query_cache[cache_key] = (indices_list, distances_list, metadata_list)
                # Evict old entries if cache is full
                if len(self._query_cache) > self.config.cache_size:
                    # Remove oldest entry (simple FIFO)
                    self._query_cache.pop(next(iter(self._query_cache)))
                    
        # Update metrics
        query_time = time.time() - start_time
        self.metrics['avg_query_time'] = (
            (self.metrics['avg_query_time'] * (self.metrics['total_queries'] - 1) + query_time) 
            / self.metrics['total_queries']
        )
        
        return indices_list, distances_list, metadata_list
        
    def _brute_force_search(self, query_vector: mx.array, k: int) -> Tuple[mx.array, mx.array]:
        """Brute force nearest neighbor search using MLX operations"""
        # Normalize query for cosine similarity
        query_norm = mx.sqrt(mx.sum(query_vector * query_vector))
        query_norm = mx.maximum(query_norm, 1e-8)
        query_normalized = query_vector / query_norm
        
        # Normalize all vectors
        norms = mx.sqrt(mx.sum(self.vectors * self.vectors, axis=1))
        norms = mx.maximum(norms, 1e-8)
        normalized_vectors = self.vectors / norms[:, None]
        
        # Compute cosine similarities
        similarities = mx.sum(normalized_vectors * query_normalized[None, :], axis=1)
        
        # Convert to distances
        distances = 1.0 - similarities
        
        # Get top k
        mx.eval(distances)
        indices = mx.argsort(distances)[:k]
        top_distances = distances[indices]
        
        return indices, top_distances
        
    def _compute_cache_key(self, query_vector: mx.array, k: int) -> str:
        """Compute cache key for query"""
        # Use hash of query vector and k
        import hashlib
        query_bytes = query_vector.tobytes()
        key_string = f"{hashlib.md5(query_bytes).hexdigest()}_{k}"
        return key_string
        
    def batch_query(self, query_vectors: mx.array, k: int = 10) -> Tuple[List[List[int]], 
                                                                        List[List[float]], 
                                                                        List[List[Dict]]]:
        """Batch query for multiple vectors"""
        all_indices = []
        all_distances = []
        all_metadata = []
        
        for i in range(query_vectors.shape[0]):
            indices, distances, metadata = self.query(query_vectors[i], k)
            all_indices.append(indices)
            all_distances.append(distances)
            all_metadata.append(metadata)
            
        return all_indices, all_distances, all_metadata
        
    def delete_vectors(self, indices: List[int]):
        """Delete vectors by indices"""
        if self.vectors is None:
            return
            
        # Create mask for vectors to keep
        mask = mx.ones(len(self.vectors), dtype=mx.bool_)
        for idx in indices:
            mask[idx] = False
            
        # Filter vectors and metadata
        self.vectors = self.vectors[mask]
        self.metadata = [m for i, m in enumerate(self.metadata) if mask[i]]
        
        # Rebuild index
        if self.hnsw_index is not None:
            self._build_index()
            
        # Clear cache
        if self._query_cache is not None:
            with self._cache_lock:
                self._query_cache.clear()
                
        # Save changes
        self._save()
        
    def update_vector(self, index: int, new_vector: mx.array, new_metadata: Dict[str, Any]):
        """Update a single vector"""
        if self.vectors is None or index >= len(self.vectors):
            raise IndexError(f"Index {index} out of range")
            
        # Update vector
        self.vectors[index] = new_vector
        self.metadata[index] = new_metadata
        
        # Rebuild index (could be optimized in future)
        if self.hnsw_index is not None:
            self._build_index()
            
        # Clear cache
        if self._query_cache is not None:
            with self._cache_lock:
                self._query_cache.clear()
                
        # Save changes
        self._save()
        
    def optimize(self):
        """Optimize the vector store for better performance"""
        logger.info("Optimizing vector store...")
        
        # Force MLX evaluation
        if self.vectors is not None:
            mx.eval(self.vectors)
            
        # Optimize HNSW index
        if self.hnsw_index is not None:
            self.hnsw_index.optimize()
            
        # Save optimized state
        self._save()
        
        logger.info("Optimization complete")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        stats = {
            'total_vectors': len(self.vectors) if self.vectors is not None else 0,
            'vector_dimension': self.vectors.shape[1] if self.vectors is not None else 0,
            'has_hnsw_index': self.hnsw_index is not None,
            'metrics': self.metrics,
            'cache_size': len(self._query_cache) if self._query_cache is not None else 0,
            'storage_size_mb': self._calculate_storage_size()
        }
        
        if self.hnsw_index is not None:
            stats['hnsw_stats'] = self.hnsw_index.get_stats()
            
        return stats
        
    def _calculate_storage_size(self) -> float:
        """Calculate total storage size in MB"""
        total_size = 0
        
        # Vectors file
        vectors_path = self.store_path / "vectors.npz"
        if vectors_path.exists():
            total_size += vectors_path.stat().st_size
            
        # Metadata file
        metadata_path = self.store_path / "metadata.jsonl"
        if metadata_path.exists():
            total_size += metadata_path.stat().st_size
            
        # HNSW index files
        for suffix in ['.graph.pkl', '.config.json']:
            index_path = self.store_path / f"hnsw_index{suffix}"
            if index_path.exists():
                total_size += index_path.stat().st_size
                
        return total_size / (1024 * 1024)  # Convert to MB
        
    def rebuild_index(self, force: bool = False):
        """Rebuild HNSW index"""
        if not self.config.enable_hnsw:
            logger.warning("HNSW indexing is disabled")
            return
            
        if self.vectors is None or len(self.vectors) == 0:
            logger.warning("No vectors to index")
            return
            
        if not force and len(self.vectors) < self.config.auto_index_threshold:
            logger.info(f"Vector count ({len(self.vectors)}) below threshold ({self.config.auto_index_threshold})")
            return
            
        self._build_index()
        
    def clear(self):
        """Clear all vectors and metadata"""
        self.vectors = None
        self.metadata = []
        self.hnsw_index = None
        
        if self._query_cache is not None:
            with self._cache_lock:
                self._query_cache.clear()
                
        # Remove files
        for file in self.store_path.glob("*"):
            if file.is_file():
                file.unlink()
                
        logger.info("Vector store cleared")


# Backward compatibility wrapper
def create_vector_store(store_path: Path, enable_hnsw: bool = True) -> VectorStore:
    """Create a vector store with optional HNSW indexing"""
    config = VectorStoreConfig(enable_hnsw=enable_hnsw)
    return VectorStore(store_path, config)


# Example usage and migration helper
if __name__ == "__main__":
    import time
    
    # Create test data
    n_vectors = 10000
    dim = 384
    
    print(f"Creating test vector store with {n_vectors} vectors...")
    
    # Create vector store with HNSW enabled
    store_path = Path("./test_vector_store")
    store = VectorStore(store_path)
    
    # Add vectors in batches
    batch_size = 1000
    for i in range(0, n_vectors, batch_size):
        vectors = mx.random.normal((batch_size, dim))
        metadata = [{"id": f"vec_{j}", "batch": i // batch_size} 
                   for j in range(i, i + batch_size)]
        
        store.add_vectors(vectors, metadata)
        print(f"Added batch {i // batch_size + 1}/{n_vectors // batch_size}")
        
    # Test queries
    print("\nTesting queries...")
    query = mx.random.normal((dim,))
    
    # Test with HNSW
    start = time.time()
    indices_hnsw, distances_hnsw, metadata_hnsw = store.query(query, k=10, use_hnsw=True)
    hnsw_time = time.time() - start
    
    # Test brute force
    start = time.time()
    indices_bf, distances_bf, metadata_bf = store.query(query, k=10, use_hnsw=False)
    bf_time = time.time() - start
    
    print(f"\nHNSW query time: {hnsw_time*1000:.2f} ms")
    print(f"Brute force query time: {bf_time*1000:.2f} ms")
    print(f"Speedup: {bf_time/hnsw_time:.2f}x")
    
    # Print stats
    print("\nVector store stats:")
    stats = store.get_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
