# performance/mlx_optimized.py
"""
MLX compiled operations for maximum performance (Simplified Version)
"""
import mlx.core as mx
import numpy as np
from typing import Tuple, List
import logging
import os

logger = logging.getLogger("mlx_vector_db.optimized")

@mx.compile
def compute_cosine_similarity_single(query_vector: mx.array, db_vectors: mx.array) -> mx.array:
    """
    Compiled single query cosine similarity computation
    """
    # Ensure query is 2D
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Compute norms using basic MLX operations
    query_norm = mx.sqrt(mx.sum(query_vector * query_vector, axis=1, keepdims=True))
    db_norms = mx.sqrt(mx.sum(db_vectors * db_vectors, axis=1, keepdims=True))
    
    # Avoid division by zero
    epsilon = 1e-10
    query_norm = mx.maximum(query_norm, epsilon)
    db_norms = mx.maximum(db_norms, epsilon)
    
    # Normalize vectors
    query_normalized = query_vector / query_norm
    db_normalized = db_vectors / db_norms
    
    # Compute cosine similarity
    similarity_scores = mx.matmul(query_normalized, db_normalized.T)
    
    return similarity_scores.flatten()

@mx.compile
def compute_cosine_similarity_batch(query_vectors: mx.array, db_vectors: mx.array) -> mx.array:
    """
    Compiled batch cosine similarity computation
    """
    # Compute norms
    query_norms = mx.sqrt(mx.sum(query_vectors * query_vectors, axis=1, keepdims=True))
    db_norms = mx.sqrt(mx.sum(db_vectors * db_vectors, axis=1, keepdims=True))
    
    # Avoid division by zero
    epsilon = 1e-10
    query_norms = mx.maximum(query_norms, epsilon)
    db_norms = mx.maximum(db_norms, epsilon)
    
    # Normalize vectors
    query_normalized = query_vectors / query_norms
    db_normalized = db_vectors / db_norms
    
    # Compute cosine similarity via matrix multiplication
    similarity_scores = mx.matmul(query_normalized, db_normalized.T)
    
    return similarity_scores

@mx.compile
def fast_top_k_indices(scores: mx.array, k: int) -> mx.array:
    """
    Compiled top-k selection
    """
    if k >= scores.shape[0]:
        return mx.argsort(-scores)
    
    # Use argsort for simplicity (MLX will optimize this)
    sorted_indices = mx.argsort(-scores)
    return sorted_indices[:k]

@mx.compile
def normalize_vectors_inplace(vectors: mx.array) -> mx.array:
    """
    Compiled vector normalization
    """
    norms = mx.sqrt(mx.sum(vectors * vectors, axis=1, keepdims=True))
    epsilon = 1e-10
    norms = mx.maximum(norms, epsilon)
    return vectors / norms

@mx.compile
def fast_vector_concatenation(existing_vectors: mx.array, new_vectors: mx.array) -> mx.array:
    """
    Compiled vector concatenation
    """
    return mx.concatenate([existing_vectors, new_vectors], axis=0)

# Higher-level optimized functions (non-compiled wrappers)

def optimized_similarity_search(query_vector: mx.array, db_vectors: mx.array, k: int = 10) -> Tuple[mx.array, mx.array]:
    """Optimized similarity search using compiled operations"""
    scores = compute_cosine_similarity_single(query_vector, db_vectors)
    top_k_indices = fast_top_k_indices(scores, k)
    top_k_scores = scores[top_k_indices]
    return top_k_indices, top_k_scores

def optimized_batch_similarity_search(query_vectors: mx.array, db_vectors: mx.array, k: int = 10) -> Tuple[mx.array, mx.array]:
    """Optimized batch similarity search"""
    scores = compute_cosine_similarity_batch(query_vectors, db_vectors)
    n_queries = scores.shape[0]
    
    # Process each query separately for simplicity
    top_k_indices_list = []
    top_k_scores_list = []
    
    for i in range(n_queries):
        query_scores = scores[i]
        query_top_k_indices = fast_top_k_indices(query_scores, k)
        query_top_k_scores = query_scores[query_top_k_indices]
        
        top_k_indices_list.append(query_top_k_indices)
        top_k_scores_list.append(query_top_k_scores)
    
    # Stack results
    top_k_indices = mx.stack(top_k_indices_list)
    top_k_scores = mx.stack(top_k_scores_list)
    
    return top_k_indices, top_k_scores

def optimized_vector_addition(existing_vectors: mx.array, new_vectors: mx.array, normalize: bool = True) -> mx.array:
    """Optimized vector addition"""
    combined = fast_vector_concatenation(existing_vectors, new_vectors)
    
    if normalize:
        combined = normalize_vectors_inplace(combined)
    
    return combined

def warmup_compiled_functions(dimension: int = 384, n_vectors: int = 1000):
    """Warm up compiled functions"""
    logger.info("Warming up compiled MLX functions...")
    
    try:
        # Create dummy data
        dummy_db_vectors = mx.random.normal((n_vectors, dimension))
        dummy_query_single = mx.random.normal((dimension,))
        dummy_query_batch = mx.random.normal((10, dimension))
        
        # Warm up functions
        _ = compute_cosine_similarity_single(dummy_query_single, dummy_db_vectors)
        _ = compute_cosine_similarity_batch(dummy_query_batch, dummy_db_vectors)
        _ = fast_top_k_indices(mx.random.normal((n_vectors,)), 10)
        _ = normalize_vectors_inplace(dummy_db_vectors)
        _ = fast_vector_concatenation(dummy_db_vectors[:500], dummy_db_vectors[500:])
        
        # Force evaluation
        mx.eval([dummy_db_vectors, dummy_query_single, dummy_query_batch])
        
        logger.info("MLX function warm-up completed successfully")
        
    except Exception as e:
        logger.error(f"Error during MLX function warm-up: {e}")
        raise

# Performance monitoring
class PerformanceMonitor:
    """Simple performance monitor"""
    
    def __init__(self):
        self.call_counts = {}
        self.total_times = {}
    
    def record_call(self, func_name: str, duration: float):
        """Record a function call duration"""
        if func_name not in self.call_counts:
            self.call_counts[func_name] = 0
            self.total_times[func_name] = 0.0
        
        self.call_counts[func_name] += 1
        self.total_times[func_name] += duration
    
    def get_stats(self) -> dict:
        """Get performance statistics"""
        stats = {}
        for func_name in self.call_counts:
            avg_time = self.total_times[func_name] / self.call_counts[func_name]
            stats[func_name] = {
                "calls": self.call_counts[func_name],
                "total_time": self.total_times[func_name],
                "avg_time": avg_time,
                "calls_per_second": 1.0 / avg_time if avg_time > 0 else 0
            }
        return stats

# Global performance monitor
performance_monitor = PerformanceMonitor()