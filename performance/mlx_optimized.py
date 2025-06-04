# performance/mlx_optimized.py
"""
MLX compiled operations for maximum performance.
Optimized for Apple Silicon.
"""
import mlx.core as mx
# import numpy as np # NumPy wird hier nicht direkt benötigt
from typing import Tuple, List # List wird hier nicht direkt benötigt, aber Tuple schon
import logging
# import os # os wird hier nicht direkt benötigt

logger = logging.getLogger("mlx_vector_db.optimized") # Konsistenter Logger-Name

# performance/mlx_optimized.py
"""
MLX compiled operations for maximum performance.
Optimized for Apple Silicon.
"""
import mlx.core as mx
# import numpy as np # NumPy wird hier nicht direkt benötigt
from typing import Tuple, List # List wird hier nicht direkt benötigt, aber Tuple schon
import logging
# import os # os wird hier nicht direkt benötigt
import threading # <--- HIER DEN IMPORT HINZUFÜGEN

logger = logging.getLogger("mlx_vector_db.optimized") # Konsistenter Logger-Name

# ... (Rest Ihres Codes in mlx_optimized.py) ...

# PerformanceMonitor-Klasse bleibt unverändert, verwendet jetzt das importierte threading
class PerformanceMonitor:
    """Simple performance monitor"""
    def __init__(self):
        self.call_counts: Dict[str, int] = {}
        self.total_times: Dict[str, float] = {}
        self._lock = threading.Lock() # Für Thread-Sicherheit

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
                if self.call_counts[func_name] == 0: continue # Division durch Null vermeiden
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

performance_monitor = PerformanceMonitor() # Globale Instanz

# --- Kompilierte MLX-Kernoperationen ---

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
    # query_norm = mx.linalg.norm(query_vector_2d, axis=1, keepdims=True) # Alternative
    query_norm = mx.sqrt(mx.sum(mx.square(query_vector_2d), axis=1, keepdims=True))
    db_norms = mx.sqrt(mx.sum(mx.square(db_vectors), axis=1, keepdims=True))
    
    # Epsilon für numerische Stabilität (mit passendem dtype)
    epsilon = mx.array(1e-8, dtype=query_norm.dtype) 
    
    query_norm = mx.maximum(query_norm, epsilon)
    db_norms = mx.maximum(db_norms, epsilon)
    
    # Normalisierung
    query_normalized = query_vector_2d / query_norm
    db_normalized = db_vectors / db_norms
    
    # Cosine-Ähnlichkeit
    similarity_scores = mx.matmul(query_normalized, db_normalized.T)
    
    return similarity_scores.flatten() # Ergebnis als 1D-Array

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
        return mx.array([], dtype=mx.int32) # Leeres int32 Array für k<=0
        
    num_scores = scores.shape[0]
    actual_k = min(k, num_scores)
    
    if actual_k == 0:
        return mx.array([], dtype=mx.int32)

    # argsort sortiert aufsteigend. Für höchste Scores negieren wir die Scores.
    # Gibt die Indizes der k größten Elemente zurück.
    return mx.argsort(-scores)[:actual_k]

@mx.compile
def normalize_vectors(vectors: mx.array) -> mx.array: # Umbenannt von normalize_vectors_inplace
    """
    Kompilierte Vektor-Normalisierung (L2-Norm).
    Gibt ein neues Array mit normalisierten Vektoren zurück.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors muss 2D sein für Normalisierung, erhielt Shape {vectors.shape}")
    if vectors.shape[0] == 0: # Leeres Array direkt zurückgeben
        return vectors.astype(vectors.dtype) # Stellt sicher, dass Typ erhalten bleibt, auch wenn leer

    norms = mx.linalg.norm(vectors, axis=1, keepdims=True) # L2-Norm für jede Zeile
    epsilon = mx.array(1e-8, dtype=norms.dtype)
    norms = mx.maximum(norms, epsilon) # Vermeide Division durch Null
    
    return vectors / norms

@mx.compile
def fast_vector_concatenation(existing_vectors: mx.array, new_vectors: mx.array) -> mx.array:
    """Kompilierte Vektor-Konkatenation."""
    if existing_vectors.shape[0] == 0: # Wenn existierende Vektoren leer sind
        return new_vectors
    if new_vectors.shape[0] == 0: # Wenn neue Vektoren leer sind
        return existing_vectors
    if existing_vectors.shape[1] != new_vectors.shape[1]:
        raise ValueError("Dimensionen der zu konkatenierenden Vektoren stimmen nicht überein.")
        
    return mx.concatenate([existing_vectors, new_vectors], axis=0)


# --- Höherlevelige optimierte Wrapper-Funktionen ---

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
    top_k_scores = similarity_scores[top_k_indices] # Hole die Scores für die Top-Indizes
    
    return top_k_indices, top_k_scores

def optimized_batch_similarity_search(query_vectors: mx.array, db_vectors: mx.array, k: int = 10) -> Tuple[mx.array, mx.array]:
    """
    Optimierte Batch-Ähnlichkeitssuche.
    Gibt (all_top_k_indices, all_top_k_scores) zurück. Scores sind Ähnlichkeits-Scores.
    """
    batch_similarity_scores = compute_cosine_similarity_batch(query_vectors, db_vectors) # (num_queries, num_db_vectors)
    
    # Effizienteres Top-K für Batches:
    # argsort entlang der Achse der Datenbankvektoren (axis=1).
    # Negiere Scores, da argsort aufsteigend sortiert und wir höchste Ähnlichkeiten wollen.
    if batch_similarity_scores.shape[1] == 0: # Keine Datenbankvektoren
        num_queries = batch_similarity_scores.shape[0]
        empty_indices = mx.zeros((num_queries, 0), dtype=mx.int32)
        empty_scores = mx.zeros((num_queries, 0), dtype=batch_similarity_scores.dtype)
        return empty_indices, empty_scores

    actual_k = min(k, batch_similarity_scores.shape[1])
    if actual_k == 0 :
        return mx.array([[] for _ in range(query_vectors.shape[0])], dtype=mx.int32), \
               mx.array([[] for _ in range(query_vectors.shape[0])], dtype=query_vectors.dtype)


    sorted_indices_all_queries = mx.argsort(-batch_similarity_scores, axis=1)
    all_top_k_indices = sorted_indices_all_queries[:, :actual_k]
    
    # Scores für die Top-K Indizes sammeln
    # mx.take_along_axis ist hierfür ideal, aber wir implementieren es manuell für Klarheit oder Fallback.
    # Einfacher Weg mit einer Schleife (MLX kompiliert die inneren Operationen gut):
    all_top_k_scores_list = []
    for i in range(batch_similarity_scores.shape[0]): # Iteriere über jede Query
        scores_for_query_i = batch_similarity_scores[i, all_top_k_indices[i]]
        all_top_k_scores_list.append(scores_for_query_i)
    
    all_top_k_scores = mx.stack(all_top_k_scores_list, axis=0) if all_top_k_scores_list else mx.array([])
    # Wenn all_top_k_scores leer ist, sicherstellen, dass es die richtige Form hat
    if all_top_k_scores.size == 0 and query_vectors.shape[0] > 0:
         all_top_k_scores = mx.zeros((query_vectors.shape[0], actual_k), dtype=batch_similarity_scores.dtype)


    return all_top_k_indices, all_top_k_scores


def optimized_vector_addition(existing_vectors: mx.array, new_vectors: mx.array, normalize: bool = False) -> mx.array: # normalize standardmäßig False
    """Optimierte Vektor-Addition."""
    combined = fast_vector_concatenation(existing_vectors, new_vectors)
    if normalize:
        combined = normalize_vectors(combined) # Aufruf der umbenannten Funktion
    return combined

def warmup_compiled_functions(dimension: int = 384, n_vectors: int = 100): # Kleinere n_vectors für schnellen Warmup
    """Wärmt kompilierte MLX-Funktionen auf."""
    logger.info(f"Wärme kompilierte MLX-Funktionen auf (Dim: {dimension}, N_Vecs: {n_vectors})...")
    try:
        dummy_db_vectors = mx.random.normal((n_vectors, dimension), dtype=mx.float32)
        dummy_query_single = mx.random.normal((dimension,), dtype=mx.float32)
        # Stelle sicher, dass Batch-Query mindestens eine Zeile hat, auch wenn n_vectors klein ist
        num_batch_queries = min(10, n_vectors if n_vectors > 0 else 1)
        dummy_query_batch = mx.random.normal((num_batch_queries, dimension), dtype=mx.float32)

        # Stelle sicher, dass k nicht größer als n_vectors ist
        k_warmup = min(5, n_vectors if n_vectors > 0 else 1)
        if k_warmup == 0 and n_vectors == 0 : k_warmup = 1 # Falls n_vectors = 0, k=1 führt zu leeren Ergebnissen

        # Warmup-Aufrufe
        if n_vectors > 0:
            _ = compute_cosine_similarity_single(dummy_query_single, dummy_db_vectors)
            _ = compute_cosine_similarity_batch(dummy_query_batch, dummy_db_vectors)
            _ = fast_top_k_indices(mx.random.normal((n_vectors,), dtype=mx.float32), k_warmup)
        
        _ = normalize_vectors(dummy_db_vectors if n_vectors > 0 else mx.random.normal((1, dimension))) # Normiere mind. 1 Vektor
        _ = fast_vector_concatenation(dummy_db_vectors[:n_vectors//2], dummy_db_vectors[n_vectors//2:])
        
        # Erzwinge Auswertung aller erstellten Arrays
        arrays_to_eval = [dummy_db_vectors, dummy_query_single, dummy_query_batch]
        # Entferne leere Arrays vor mx.eval, falls n_vectors=0 war
        arrays_to_eval = [arr for arr in arrays_to_eval if arr.size > 0]
        if arrays_to_eval:
            mx.eval(arrays_to_eval)
        
        logger.info("MLX-Funktions-Warmup erfolgreich abgeschlossen.")
    except Exception as e:
        logger.error(f"Fehler während des MLX-Funktions-Warmups: {e}", exc_info=True)
        # Fehler hier nicht weiterwerfen, damit die Anwendung trotzdem starten kann
        # raise # Entfernt, um Start nicht zu blockieren


# PerformanceMonitor-Klasse bleibt unverändert (wie in Ihrer Datei)
# (Fügen Sie hier Ihre PerformanceMonitor-Klasse ein)
class PerformanceMonitor:
    """Simple performance monitor"""
    def __init__(self):
        self.call_counts: Dict[str, int] = {}
        self.total_times: Dict[str, float] = {}
        self._lock = threading.Lock() # Für Thread-Sicherheit

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
                if self.call_counts[func_name] == 0: continue # Division durch Null vermeiden
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

performance_monitor = PerformanceMonitor() # Globale Instanz