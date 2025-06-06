#
"""
MLX compiled operations for maximum performance.
Optimized for Apple Silicon with current MLX API.
"""
import mlx.core as mx
from typing import Tuple, Dict
import logging
import threading
import time

logger = logging.getLogger("mlx_vector_db.optimized")

# Compiled MLX functions for optimal performance
@mx.compile
def compute_cosine_similarity_single(query_vector: mx.array, db_vectors: mx.array) -> mx.array:
    """
    Kompilierte Berechnung der Cosine-Ähnlichkeit für eine einzelne Abfrage.
    Gibt einen 1D-Array von Ähnlichkeits-Scores zurück.
    """
    # Stelle sicher, dass query_vector 2D ist (1, dim) für matmul
    if query_vector.ndim == 1:
        query_vector_2d = query_vector.reshape(1, -1)
    elif query_vector.ndim == 2 and query_vector.shape[0] == 1:
        query_vector_2d = query_vector
    else:
        raise ValueError(f"query_vector muss 1D oder 2D mit einer Zeile sein, erhielt Shape {query_vector.shape}")

    # Normenberechnung
    query_norm = mx.sqrt(mx.sum(mx.square(query_vector_2d), axis=1, keepdims=True))
    db_norms = mx.sqrt(mx.sum(mx.square(db_vectors), axis=1, keepdims=True))
    
    # Epsilon für numerische Stabilität
    epsilon = mx.array(1e-8, dtype=query_norm.dtype) 
    
    query_norm = mx.maximum(query_norm, epsilon)
    db_norms = mx.maximum(db_norms, epsilon)
    
    # Normalisierung
    query_normalized = query_vector_2d / query_norm
    db_normalized = db_vectors / db_norms
    
    # Cosine-Ähnlichkeit
    similarity_scores = mx.matmul(query_normalized, db_normalized.T)
    
    return similarity_scores.flatten()

@mx.compile
def compute_cosine_similarity_batch(query_vectors: mx.array, db_vectors: mx.array) -> mx.array:
    """
    Kompilierte Berechnung der Cosine-Ähnlichkeit für einen Batch von Abfragen.
    Gibt eine 2D-Matrix von Ähnlichkeits-Scores zurück (num_queries, num_db_vectors).
    """
    if query_vectors.ndim != 2:
        raise ValueError(f"query_vectors muss 2D sein, erhielt Shape {query_vectors.shape}")
    if db_vectors.ndim != 2:
        raise ValueError(f"db_vectors muss 2D sein, erhielt Shape {db_vectors.shape}")
    if query_vectors.shape[1] != db_vectors.shape[1]:
        raise ValueError(
            f"Dimension Mismatch: query_vectors Dim {query_vectors.shape[1]}, db_vectors Dim {db_vectors.shape[1]}"
        )

    # Normenberechnung
    query_norms = mx.sqrt(mx.sum(mx.square(query_vectors), axis=1, keepdims=True))
    db_norms = mx.sqrt(mx.sum(mx.square(db_vectors), axis=1, keepdims=True))
    
    epsilon = mx.array(1e-8, dtype=query_norms.dtype)
        
    query_norms = mx.maximum(query_norms, epsilon)
    db_norms = mx.maximum(db_norms, epsilon)
    
    query_normalized = query_vectors / query_norms
    db_normalized = db_vectors / db_norms
    
    similarity_scores = mx.matmul(query_normalized, db_normalized.T)
    
    return similarity_scores

@mx.compile
def fast_top_k_indices(scores: mx.array, k: int) -> mx.array:
    """
    Kompilierte Auswahl der Top-K Indizes.
    `scores` ist ein 1D-Array von Ähnlichkeits-Scores (höher ist besser).
    """
    if not isinstance(scores, mx.array) or scores.ndim != 1:
        raise ValueError("scores muss ein 1D mx.array sein.")
    if k <= 0:
        return mx.array([], dtype=mx.int32)
        
    num_scores = scores.shape[0]
    actual_k = min(k, num_scores)
    
    if actual_k == 0:
        return mx.array([], dtype=mx.int32)

    # argsort sortiert aufsteigend. Für höchste Scores negieren wir die Scores.
    return mx.argsort(-scores)[:actual_k]

@mx.compile
def normalize_vectors(vectors: mx.array) -> mx.array:
    """
    Kompilierte Vektor-Normalisierung (L2-Norm).
    Gibt ein neues Array mit normalisierten Vektoren zurück.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors muss 2D sein für Normalisierung, erhielt Shape {vectors.shape}")
    if vectors.shape[0] == 0:
        return vectors.astype(vectors.dtype)

    norms = mx.linalg.norm(vectors, axis=1, keepdims=True)
    epsilon = mx.array(1e-8, dtype=norms.dtype)
    norms = mx.maximum(norms, epsilon)
    
    return vectors / norms

@mx.compile
def fast_vector_concatenation(existing_vectors: mx.array, new_vectors: mx.array) -> mx.array:
    """Kompilierte Vektor-Konkatenation."""
    if existing_vectors.shape[0] == 0:
        return new_vectors
    if new_vectors.shape[0] == 0:
        return existing_vectors
    if existing_vectors.shape[1] != new_vectors.shape[1]:
        raise ValueError("Dimensionen der zu konkatenierenden Vektoren stimmen nicht überein.")
        
    return mx.concatenate([existing_vectors, new_vectors], axis=0)

@mx.compile
def compute_euclidean_distance(query_vector: mx.array, db_vectors: mx.array) -> mx.array:
    """Kompilierte Euclidean Distance Berechnung"""
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    diff = db_vectors - query_vector
    distances = mx.sqrt(mx.sum(mx.square(diff), axis=1))
    
    return distances

@mx.compile
def compute_dot_product(query_vector: mx.array, db_vectors: mx.array) -> mx.array:
    """Kompilierte Dot Product Berechnung"""
    if query_vector.ndim == 1:
        return mx.matmul(db_vectors, query_vector)
    else:
        return mx.matmul(db_vectors, query_vector.T).flatten()

# Performance Monitor
class PerformanceMonitor:
    """Simple performance monitor"""
    def __init__(self):
        self.call_counts: Dict[str, int] = {}
        self.total_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def record_call(self, func_name: str, duration: float):
        with self._lock:
            if func_name not in self.call_counts:
                self.call_counts[func_name] = 0
                self.total_times[func_name] = 0.0
            self.call_counts[func_name] += 1
            self.total_times[func_name] += duration

    def get_stats(self) -> dict:
        with self._lock:
            stats = {}
            for func_name in self.call_counts:
                if self.call_counts[func_name] == 0:
                    continue
                avg_time = self.total_times[func_name] / self.call_counts[func_name]
                stats[func_name] = {
                    "calls": self.call_counts[func_name],
                    "total_time_seconds": round(self.total_times[func_name], 4),
                    "avg_time_ms": round(avg_time * 1000, 4),
                    "calls_per_second": round(1.0 / avg_time if avg_time > 0 else 0, 2)
                }
            return stats
    
    def reset(self):
        with self._lock:
            self.call_counts.clear()
            self.total_times.clear()
        logger.info("PerformanceMonitor has been reset.")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# High-level optimized wrapper functions
def optimized_similarity_search(query_vector: mx.array, db_vectors: mx.array, k: int = 10) -> Tuple[mx.array, mx.array]:
    """
    Optimierte Ähnlichkeitssuche unter Verwendung kompilierter Operationen.
    Gibt (top_k_indices, top_k_scores) zurück. Scores sind Ähnlichkeits-Scores.
    """
    if query_vector.ndim == 2 and query_vector.shape[0] == 1:
        query_vector_flat = query_vector.flatten()
    elif query_vector.ndim == 1:
        query_vector_flat = query_vector
    else:
        raise ValueError(f"query_vector muss 1D oder 2D (1 Zeile) sein, Shape: {query_vector.shape}")

    similarity_scores = compute_cosine_similarity_single(query_vector_flat, db_vectors)
    top_k_indices = fast_top_k_indices(similarity_scores, k)
    top_k_scores = similarity_scores[top_k_indices]
    
    return top_k_indices, top_k_scores

def optimized_batch_similarity_search(query_vectors: mx.array, db_vectors: mx.array, k: int = 10) -> Tuple[mx.array, mx.array]:
    """
    Optimierte Batch-Ähnlichkeitssuche.
    Gibt (all_top_k_indices, all_top_k_scores) zurück.
    """
    batch_similarity_scores = compute_cosine_similarity_batch(query_vectors, db_vectors)
    
    if batch_similarity_scores.shape[1] == 0:
        num_queries = batch_similarity_scores.shape[0]
        empty_indices = mx.zeros((num_queries, 0), dtype=mx.int32)
        empty_scores = mx.zeros((num_queries, 0), dtype=batch_similarity_scores.dtype)
        return empty_indices, empty_scores

    actual_k = min(k, batch_similarity_scores.shape[1])
    if actual_k == 0:
        return mx.array([[] for _ in range(query_vectors.shape[0])], dtype=mx.int32), \
               mx.array([[] for _ in range(query_vectors.shape[0])], dtype=query_vectors.dtype)

    sorted_indices_all_queries = mx.argsort(-batch_similarity_scores, axis=1)
    all_top_k_indices = sorted_indices_all_queries[:, :actual_k]
    
    # Scores für die Top-K Indizes sammeln
    all_top_k_scores_list = []
    for i in range(batch_similarity_scores.shape[0]):
        scores_for_query_i = batch_similarity_scores[i, all_top_k_indices[i]]
        all_top_k_scores_list.append(scores_for_query_i)
    
    all_top_k_scores = mx.stack(all_top_k_scores_list, axis=0) if all_top_k_scores_list else mx.array([])
    if all_top_k_scores.size == 0 and query_vectors.shape[0] > 0:
         all_top_k_scores = mx.zeros((query_vectors.shape[0], actual_k), dtype=batch_similarity_scores.dtype)

    return all_top_k_indices, all_top_k_scores

def optimized_vector_addition(existing_vectors: mx.array, new_vectors: mx.array, normalize: bool = False) -> mx.array:
    """Optimierte Vektor-Addition."""
    combined = fast_vector_concatenation(existing_vectors, new_vectors)
    if normalize:
        combined = normalize_vectors(combined)
    return combined

def warmup_compiled_functions(dimension: int = 384, n_vectors: int = 100):
    """Wärmt kompilierte MLX-Funktionen auf."""
    logger.info(f"Wärme kompilierte MLX-Funktionen auf (Dim: {dimension}, N_Vecs: {n_vectors})...")
    try:
        dummy_db_vectors = mx.random.normal((n_vectors, dimension), dtype=mx.float32)
        dummy_query_single = mx.random.normal((dimension,), dtype=mx.float32)
        num_batch_queries = min(10, n_vectors if n_vectors > 0 else 1)
        dummy_query_batch = mx.random.normal((num_batch_queries, dimension), dtype=mx.float32)

        k_warmup = min(5, n_vectors if n_vectors > 0 else 1)
        if k_warmup == 0 and n_vectors == 0:
            k_warmup = 1

        # Warmup-Aufrufe
        if n_vectors > 0:
            _ = compute_cosine_similarity_single(dummy_query_single, dummy_db_vectors)
            _ = compute_cosine_similarity_batch(dummy_query_batch, dummy_db_vectors)
            _ = fast_top_k_indices(mx.random.normal((n_vectors,), dtype=mx.float32), k_warmup)
        
        _ = normalize_vectors(dummy_db_vectors if n_vectors > 0 else mx.random.normal((1, dimension)))
        _ = fast_vector_concatenation(dummy_db_vectors[:n_vectors//2], dummy_db_vectors[n_vectors//2:])
        
        # Erzwinge Auswertung aller erstellten Arrays
        arrays_to_eval = [dummy_db_vectors, dummy_query_single, dummy_query_batch]
        arrays_to_eval = [arr for arr in arrays_to_eval if arr.size > 0]
        if arrays_to_eval:
            mx.eval(arrays_to_eval)
        
        logger.info("MLX-Funktions-Warmup erfolgreich abgeschlossen.")
    except Exception as e:
        logger.error(f"Fehler während des MLX-Funktions-Warmups: {e}", exc_info=True)