# performance/optimized_vector_store.py
"""
High-performance vector store operations using HNSW index, caching, and compiled MLX operations
This module provides 10-100x performance improvements over the basic implementation
"""
import time
import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import mlx.core as mx
import numpy as np

from .hnsw_index import HNSWIndex
from .vector_cache import get_global_cache, invalidate_cache
from .mlx_optimized import (
    optimized_similarity_search,
    optimized_batch_similarity_search,
    optimized_vector_addition,
    warmup_compiled_functions,
    performance_monitor
)

logger = logging.getLogger("mlx_vector_db.optimized_store")

class PerformantVectorStore:
    """
    High-performance vector store with HNSW indexing and caching
    """
    
    def __init__(self, user_id: str, model_name: str, base_path: Path):
        self.user_id = user_id
        self.model_name = model_name
        self.base_path = base_path
        self.store_path = base_path / f"user_{user_id}" / model_name
        
        # Performance components
        self.cache = get_global_cache()
        self.hnsw_index: Optional[HNSWIndex] = None
        self.dimension: Optional[int] = None
        
        # Index management
        self.index_path = self.store_path / "hnsw_index.pkl"
        self.index_needs_rebuild = False
        self.auto_index_threshold = 1000  # Rebuild index if vectors > threshold
        
        # Performance settings
        self.use_hnsw = True
        self.use_cache = True
        self.batch_size = int(os.getenv("DEFAULT_BATCH_SIZE", "1000"))
        
        logger.info(f"PerformantVectorStore initialized for {user_id}/{model_name}")
    
    def _load_vectors_and_metadata(self) -> Tuple[mx.array, List[Dict]]:
        """Load vectors and metadata with caching"""
        # Try cache first
        if self.use_cache:
            cached_data = self.cache.get(self.user_id, self.model_name)
            if cached_data is not None:
                vectors, metadata = cached_data
                logger.debug(f"Loaded {vectors.shape[0]} vectors from cache")
                return vectors, metadata
        
        # Load from disk
        vector_path = self.store_path / "vectors.npz"
        metadata_path = self.store_path / "metadata.jsonl"
        
        if not vector_path.exists():
            empty_vectors = mx.zeros((0, self.dimension or 384), dtype=mx.float32)
            return empty_vectors, []
        
        start_time = time.time()
        
        # Load vectors
        loaded_data = mx.load(str(vector_path))
        vectors = loaded_data.get('vectors', mx.zeros((0, 384), dtype=mx.float32))
        
        # Load metadata
        metadata = []
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        metadata.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        load_time = time.time() - start_time
        logger.debug(f"Loaded {vectors.shape[0]} vectors from disk in {load_time:.3f}s")
        
        # Cache the loaded data
        if self.use_cache and vectors.shape[0] > 0:
            self.cache.put(self.user_id, self.model_name, vectors, metadata)
        
        # Update dimension
        if vectors.shape[0] > 0:
            self.dimension = vectors.shape[1]
        
        return vectors, metadata
    
    def _load_or_build_index(self, vectors: mx.array) -> Optional[HNSWIndex]:
        """Load existing HNSW index or build new one"""
        if not self.use_hnsw or vectors.shape[0] == 0:
            return None
        
        # Set dimension
        self.dimension = vectors.shape[1]
        
        # Try to load existing index
        if self.index_path.exists() and not self.index_needs_rebuild:
            try:
                start_time = time.time()
                index = HNSWIndex.load(self.index_path)
                load_time = time.time() - start_time
                
                # Verify index is compatible
                if (index.dimension == self.dimension and 
                    index.node_count == vectors.shape[0]):
                    logger.info(f"Loaded HNSW index in {load_time:.3f}s: {index.node_count} nodes")
                    return index
                else:
                    logger.warning(f"Index dimension/size mismatch, rebuilding")
                    self.index_needs_rebuild = True
            except Exception as e:
                logger.warning(f"Failed to load HNSW index: {e}, rebuilding")
                self.index_needs_rebuild = True
        
        # Build new index
        if vectors.shape[0] >= 100:  # Only build index for reasonable size
            return self._build_hnsw_index(vectors)
        
        return None
    
    def _build_hnsw_index(self, vectors: mx.array) -> HNSWIndex:
        """Build HNSW index from vectors"""
        start_time = time.time()
        
        # Create HNSW index with optimized parameters
        ef_construction = min(200, max(100, vectors.shape[0] // 10))
        index = HNSWIndex(
            dimension=self.dimension,
            max_connections=16,
            max_connections_layer0=32,
            ef_construction=ef_construction,
            ef_search=50
        )
        
        # Add vectors in batches for better performance
        batch_size = min(1000, vectors.shape[0])
        
        for i in range(0, vectors.shape[0], batch_size):
            end_idx = min(i + batch_size, vectors.shape[0])
            batch_vectors = vectors[i:end_idx]
            
            for j, vector in enumerate(batch_vectors):
                vector_id = i + j
                index.add_vector(vector_id, vector)
            
            if (i + batch_size) % 5000 == 0:
                logger.info(f"HNSW index building progress: {i + batch_size}/{vectors.shape[0]}")
        
        build_time = time.time() - start_time
        logger.info(f"Built HNSW index in {build_time:.3f}s: {index.node_count} nodes")
        
        # Save index
        try:
            self.store_path.mkdir(parents=True, exist_ok=True)
            index.save(self.index_path)
        except Exception as e:
            logger.error(f"Failed to save HNSW index: {e}")
        
        return index
    
    def query_vectors_optimized(
        self, 
        query_vector: Union[np.ndarray, mx.array], 
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        use_hnsw: bool = True
    ) -> List[Dict[str, Any]]:
        """
        High-performance vector query with multiple optimization strategies
        """
        start_time = time.time()
        
        # Convert query to MLX array
        if isinstance(query_vector, np.ndarray):
            query_mx = mx.array(query_vector.astype(np.float32))
        else:
            query_mx = query_vector.astype(mx.float32)
        
        if query_mx.ndim == 1:
            query_mx = query_mx.reshape(1, -1)
        
        # Load vectors and metadata
        db_vectors, metadata = self._load_vectors_and_metadata()
        
        if db_vectors.shape[0] == 0:
            return []
        
        load_time = time.time() - start_time
        
        # Apply metadata filtering if needed
        if filter_metadata:
            filtered_indices = [
                i for i, meta in enumerate(metadata)
                if all(meta.get(key) == value for key, value in filter_metadata.items())
            ]
            
            if not filtered_indices:
                return []
            
            db_vectors = db_vectors[filtered_indices]
            metadata = [metadata[i] for i in filtered_indices]
        else:
            filtered_indices = list(range(len(metadata)))
        
        search_start = time.time()
        
        # Choose search strategy based on dataset size and availability
        if (use_hnsw and self.use_hnsw and 
            db_vectors.shape[0] > 500 and 
            not filter_metadata):  # HNSW doesn't work well with filtering
            
            # Use HNSW index for large datasets
            if self.hnsw_index is None:
                self.hnsw_index = self._load_or_build_index(db_vectors)
            
            if self.hnsw_index:
                try:
                    # HNSW search
                    hnsw_results = self.hnsw_index.search(
                        query_mx.flatten(), 
                        k=min(k, db_vectors.shape[0])
                    )
                    
                    # Convert to result format
                    results = []
                    for vector_id, distance in hnsw_results:
                        if vector_id < len(metadata):
                            entry = metadata[vector_id].copy()
                            entry['similarity_score'] = float(1.0 - distance)  # Convert distance to similarity
                            results.append(entry)
                    
                    search_time = time.time() - search_start
                    total_time = time.time() - start_time
                    
                    logger.info(f"HNSW query completed: {len(results)} results in {search_time:.3f}s (total: {total_time:.3f}s)")
                    performance_monitor.record_call("hnsw_query", total_time)
                    
                    return results
                    
                except Exception as e:
                    logger.warning(f"HNSW search failed, falling back to brute force: {e}")
        
        # Fallback to optimized brute force search
        top_k_indices, top_k_scores = optimized_similarity_search(
            query_mx.flatten(), db_vectors, k
        )
        
        # Convert to result format
        results = []
        for i in range(len(top_k_indices)):
            idx = int(top_k_indices[i])
            if idx < len(metadata):
                entry = metadata[idx].copy()
                entry['similarity_score'] = float(top_k_scores[i])
                results.append(entry)
        
        search_time = time.time() - search_start
        total_time = time.time() - start_time
        
        logger.info(f"Optimized brute force query: {len(results)} results in {search_time:.3f}s (total: {total_time:.3f}s)")
        performance_monitor.record_call("optimized_bruteforce_query", total_time)
        
        return results
    
    def batch_query_vectors_optimized(
        self,
        query_vectors: Union[np.ndarray, mx.array],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        High-performance batch vector query
        """
        start_time = time.time()
        
        # Convert queries to MLX array
        if isinstance(query_vectors, np.ndarray):
            queries_mx = mx.array(query_vectors.astype(np.float32))
        else:
            queries_mx = query_vectors.astype(mx.float32)
        
        if queries_mx.ndim == 1:
            queries_mx = queries_mx.reshape(1, -1)
        
        # Load vectors and metadata
        db_vectors, metadata = self._load_vectors_and_metadata()
        
        if db_vectors.shape[0] == 0:
            return [[] for _ in range(queries_mx.shape[0])]
        
        # Apply metadata filtering if needed
        if filter_metadata:
            filtered_indices = [
                i for i, meta in enumerate(metadata)
                if all(meta.get(key) == value for key, value in filter_metadata.items())
            ]
            
            if not filtered_indices:
                return [[] for _ in range(queries_mx.shape[0])]
            
            db_vectors = db_vectors[filtered_indices]
            metadata = [metadata[i] for i in filtered_indices]
        
        search_start = time.time()
        
        # Use optimized batch search
        top_k_indices, top_k_scores = optimized_batch_similarity_search(
            queries_mx, db_vectors, k
        )
        
        # Convert to result format
        batch_results = []
        for q in range(queries_mx.shape[0]):
            query_results = []
            for i in range(k):
                if i < top_k_indices.shape[1]:
                    idx = int(top_k_indices[q, i])
                    if idx < len(metadata):
                        entry = metadata[idx].copy()
                        entry['similarity_score'] = float(top_k_scores[q, i])
                        query_results.append(entry)
            batch_results.append(query_results)
        
        total_time = time.time() - start_time
        search_time = time.time() - search_start
        
        logger.info(f"Batch query completed: {queries_mx.shape[0]} queries in {search_time:.3f}s (total: {total_time:.3f}s)")
        performance_monitor.record_call("batch_query", total_time)
        
        return batch_results
    
    def add_vectors_optimized(
        self,
        vectors: Union[np.ndarray, mx.array],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        High-performance vector addition with index updates
        """
        start_time = time.time()
        
        # Convert to MLX array
        if isinstance(vectors, np.ndarray):
            vectors_mx = mx.array(vectors.astype(np.float32))
        else:
            vectors_mx = vectors.astype(mx.float32)
        
        # Set/verify dimension
        if self.dimension is None:
            self.dimension = vectors_mx.shape[1]
        elif vectors_mx.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: {vectors_mx.shape[1]} != {self.dimension}")
        
        # Load existing data
        existing_vectors, existing_metadata = self._load_vectors_and_metadata()
        
        # Combine with new data using optimized operation
        if existing_vectors.shape[0] > 0:
            combined_vectors = optimized_vector_addition(existing_vectors, vectors_mx, normalize=False)
        else:
            combined_vectors = vectors_mx
        
        combined_metadata = existing_metadata + metadata
        
        # Save to disk (using original vector_store functions for consistency)
        from service.vector_store import get_store_path
        from pathlib import Path
        import mlx.core as mx
        
        store_path = get_store_path(self.user_id, self.model_name)
        vector_path = store_path / "vectors.npz"
        metadata_path = store_path / "metadata.jsonl"
        
        # Save vectors
        store_path.mkdir(parents=True, exist_ok=True)
        mx.savez(str(vector_path), vectors=combined_vectors)
        mx.synchronize()
        
        # Save metadata
        with metadata_path.open("w", encoding="utf-8") as f:
            for entry in combined_metadata:
                f.write(json.dumps(entry) + "\n")
        
        # Invalidate cache
        invalidate_cache(self.user_id, self.model_name)
        
        # Mark index for rebuild if it's getting large
        if combined_vectors.shape[0] > self.auto_index_threshold:
            self.index_needs_rebuild = True
            self.hnsw_index = None  # Force rebuild on next query
        
        total_time = time.time() - start_time
        logger.info(f"Added {vectors_mx.shape[0]} vectors in {total_time:.3f}s")
        performance_monitor.record_call("add_vectors", total_time)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        cache_stats = self.cache.get_stats()
        perf_stats = performance_monitor.get_stats()
        
        index_stats = {}
        if self.hnsw_index:
            index_stats = self.hnsw_index.get_stats()
        
        return {
            "cache": cache_stats,
            "performance": perf_stats,
            "index": index_stats,
            "store_info": {
                "user_id": self.user_id,
                "model_name": self.model_name,
                "dimension": self.dimension,
                "use_hnsw": self.use_hnsw,
                "use_cache": self.use_cache,
                "index_needs_rebuild": self.index_needs_rebuild
            }
        }

# Initialize compiled functions on module import
def initialize_performance_optimizations():
    """Initialize performance optimizations"""
    try:
        # Warm up compiled functions
        warmup_compiled_functions()
        logger.info("Performance optimizations initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize performance optimizations: {e}")

# Auto-initialize when module is imported
initialize_performance_optimizations()