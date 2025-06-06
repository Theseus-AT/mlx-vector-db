# service/optimized_vector_store.py

import mlx.core as mx
import numpy as np
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import threading
import time
import logging
from dataclasses import dataclass

from performance.hnsw_index import ProductionHNSWIndex

logger = logging.getLogger("mlx_vector_db.optimized_store")


@mx.compile
def _compiled_cosine_similarity(query: mx.array, vectors: mx.array) -> mx.array:
    if query.ndim == 1: query = query[None, :]
    query_norm = mx.linalg.norm(query, axis=1, keepdims=True)
    vectors_norm = mx.linalg.norm(vectors, axis=1, keepdims=True)
    eps = mx.array(1e-8, dtype=query_norm.dtype)
    query_norm = mx.maximum(query_norm, eps)
    vectors_norm = mx.maximum(vectors_norm, eps)
    query_normalized = query / query_norm
    vectors_normalized = vectors / vectors_norm
    return mx.matmul(vectors_normalized, query_normalized.T).flatten()

@mx.compile
def _compiled_euclidean_distance(query: mx.array, vectors: mx.array) -> mx.array:
    if query.ndim == 1: query = query[None, :]
    diff = vectors - query
    distances_sq = mx.sum(diff * diff, axis=1)
    return mx.sqrt(distances_sq)


@dataclass
class MLXVectorStoreConfig:
    dimension: int = 384
    metric: str = "cosine"
    enable_hnsw: bool = False
    jit_compile: bool = True


class MLXVectorStore:
    def __init__(self, store_path: str, config: Optional[MLXVectorStoreConfig] = None):
        self.store_path = Path(store_path).expanduser()
        self.config = config or MLXVectorStoreConfig()
        self._lock = threading.RLock()
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._is_dirty = False
        self._compiled_similarity_fn = None
        
        self._vectors: Optional[mx.array] = None
        self._metadata: List[Dict] = []
        self._vector_count = 0
        
        self._hnsw_index: Optional[ProductionHNSWIndex] = None
        if self.config.enable_hnsw:
            self._hnsw_index = ProductionHNSWIndex(
                dimension=self.config.dimension,
                store_path=self.store_path,
                metric=self.config.metric
            )
        
        self._initialize_store()
        if self.config.jit_compile:
            self._compile_critical_functions()
        logger.info(f"MLX Store initialisiert: {self.store_path} | HNSW: {self.config.enable_hnsw}")

    # KORREKTUR 1: Fehlende Methode wieder hinzugefügt
    def _create_empty_store(self):
        """Initialisiert oder leert den Store-Zustand im Speicher."""
        self._vectors = None
        self._metadata = []
        self._vector_count = 0
        self._is_dirty = False

    def _initialize_store(self):
        self._load_store()

    def add_vectors(self, vectors: Union[np.ndarray, mx.array], metadata: List[Dict]):
        with self._lock:
            new_vectors_mx = self._convert_to_mlx_array(vectors)
            if self._vectors is None:
                self._vectors = new_vectors_mx
            else:
                self._vectors = mx.concatenate([self._vectors, new_vectors_mx], axis=0)
            
            self._metadata.extend(metadata)
            self._vector_count = self._vectors.shape[0]
            mx.eval(self._vectors)
            self._is_dirty = True
            self._save_store()

            if self.config.enable_hnsw and self._hnsw_index is not None:
                vectors_np = np.array(self._vectors.tolist(), dtype='float32')
                self._hnsw_index.build(vectors_np)
            
            return {'vectors_added': len(metadata), 'total_vectors': self._vector_count}

    def query(self, query_vector: Union[np.ndarray, mx.array], k: int = 10, filter_metadata: Optional[Dict] = None, use_hnsw: bool = True) -> Tuple:
        if self._vector_count == 0: return [], [], []
        query_mx = self._convert_to_mlx_array(query_vector)
        
        if use_hnsw and self.config.enable_hnsw and self._hnsw_index and self._hnsw_index.is_loaded:
            try:
                query_np = np.array(query_mx.tolist(), dtype='float32').reshape(1, -1)
                candidate_k = k * 10 if filter_metadata else k
                indices, distances = self._hnsw_index.search(query_np, k=candidate_k)
                indices, distances = indices[0], distances[0]
                
                if not filter_metadata:
                    metadata_list = [self._metadata[int(i)] for i in indices]
                    return indices.tolist(), distances.tolist(), metadata_list
                else:
                    filtered_results = []
                    for i, idx in enumerate(indices):
                        if idx < len(self._metadata):
                            meta = self._metadata[int(idx)]
                            if all(meta.get(key) == value for key, value in filter_metadata.items()):
                                filtered_results.append((idx, distances[i], meta))
                        if len(filtered_results) == k:
                            break
                    if not filtered_results: return [], [], []
                    final_indices, final_distances, final_metadata = zip(*filtered_results)
                    return list(final_indices), list(final_distances), list(final_metadata)
            except Exception as e:
                logger.warning(f"HNSW-Suche fehlgeschlagen, falle auf Brute-Force zurück: {e}")
        
        return self._brute_force_search(query_mx, k, filter_metadata)

# In service/optimized_vector_store.py

    def _brute_force_search(self, query_mx: mx.array, k: int, filter_metadata: Optional[Dict] = None):
        """
        Führt eine robuste Brute-Force-Suche durch und gibt rohe Distanzen oder Ähnlichkeiten zurück.
        """
        if not self._compiled_similarity_fn:
            raise RuntimeError("Keine kompilierte Ähnlichkeitsfunktion verfügbar.")

        target_vectors = self._vectors
        original_indices = None

        if filter_metadata:
            # Finde die Indizes, die dem Filter entsprechen
            original_indices = [
                i for i, meta in enumerate(self._metadata)
                if all(meta.get(key) == value for key, value in filter_metadata.items())
            ]
            if not original_indices:
                return [], [], []
            target_vectors = self._vectors[original_indices]

        if target_vectors.shape[0] == 0:
            return [], [], []

        # Führe die Suche auf den (gefilterten) Vektoren durch
        scores_or_distances = self._compiled_similarity_fn(query_mx, target_vectors)
        
        # Finde die Top-K Indizes innerhalb der gefilterten Menge
        if self.config.metric == "euclidean":
            # Kleinere Distanz ist besser
            sorted_local_indices = mx.argsort(scores_or_distances)[:k]
        else:
            # Höherer Score (Cosine/Dot) ist besser
            sorted_local_indices = mx.argsort(-scores_or_distances)[:k]
        
        top_results = scores_or_distances[sorted_local_indices]
        mx.eval(top_results)

        # Mappe die lokalen Indizes zurück auf die originalen Indizes des Stores
        final_indices = [original_indices[i] for i in sorted_local_indices.tolist()] if original_indices is not None else sorted_local_indices.tolist()
        
        final_metadata = [self._metadata[i] for i in final_indices]
        final_scores_or_distances = top_results.tolist()

        return final_indices, final_scores_or_distances, final_metadata

    def _warmup_kernels(self):
        # ... (Funktion bleibt unverändert)
        pass

    def clear(self):
        with self._lock:
            try:
                if self.store_path.exists():
                    shutil.rmtree(self.store_path)
                self.store_path.mkdir(parents=True, exist_ok=True)
                self._create_empty_store()
                if self.config.enable_hnsw:
                    self._hnsw_index = ProductionHNSWIndex(dimension=self.config.dimension, store_path=self.store_path, metric=self.config.metric)
                logger.info(f"Store {self.store_path} wurde vollständig bereinigt.")
            except Exception as e:
                logger.error(f"Fehler beim Bereinigen des Stores {self.store_path}: {e}")
    
    def _compile_critical_functions(self):
        if self.config.metric == "cosine": self._compiled_similarity_fn = _compiled_cosine_similarity
        elif self.config.metric == "euclidean": self._compiled_similarity_fn = _compiled_euclidean_distance
        
    def _convert_to_mlx_array(self, vectors: Union[np.ndarray, mx.array]) -> mx.array:
        return mx.array(vectors, dtype=mx.float32)

    def _save_store(self):
        if self._vectors is None or not self._is_dirty: return
        mx.savez(str(self.store_path / "vectors.npz"), vectors=self._vectors)
        with open(self.store_path / "metadata.jsonl", 'w') as f:
            for meta in self._metadata: f.write(json.dumps(meta) + '\n')
        self._is_dirty = False
    
    def _load_store(self):
        vectors_path = self.store_path / "vectors.npz"
        if not vectors_path.exists():
            self._create_empty_store()
            return
        
        try:
            self._vectors = mx.load(str(vectors_path))['vectors']
            self._vector_count = self._vectors.shape[0]
            metadata_path = self.store_path / "metadata.jsonl"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f: self._metadata = [json.loads(line) for line in f]
        except Exception as e:
            logger.error(f"Fehler beim Laden des Stores, erstelle leeren Store: {e}")
            self._create_empty_store()

    def get_stats(self):
        return {'vector_count': self._vector_count, 'dimension': self.config.dimension, 'metric': self.config.metric, 'index_type': 'hnsw' if self.config.enable_hnsw else 'flat'}

def create_optimized_vector_store(store_path: str, dimension: int = 384, jit_compile: bool = True, enable_hnsw: bool = False, **kwargs) -> MLXVectorStore:
    config = MLXVectorStoreConfig(dimension=dimension, jit_compile=jit_compile, enable_hnsw=enable_hnsw, **kwargs)
    return MLXVectorStore(store_path, config)