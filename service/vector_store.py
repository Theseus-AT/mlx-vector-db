# service/vector_store.py (oder wo Ihre Datei liegt, z.B. direkt im Root)
# OPTIMIERTE VERSION
# Fokus: MLX-nativ, Performance, Integration Ihres produktionsreifen HNSW-Index.

import mlx.core as mx
import numpy as np # Für Fallbacks und ggf. spezifische Operationen
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional, Any, Union # Union hinzugefügt
import threading
import time
from dataclasses import dataclass, field
import logging
import hashlib

# Import Ihrer produktionsreifen HNSW-Implementierung und Konfiguration
# Stellen Sie sicher, dass der Importpfad korrekt ist.
# z.B. from performance.hnsw_index import HNSWIndex, HNSWConfig
from performance.hnsw_index import HNSWIndex, HNSWConfig #

# Import der optimierten MLX-Funktionen
# z.B. from performance.mlx_optimized import optimized_similarity_search, optimized_batch_similarity_search
# Für dieses Beispiel nehmen wir an, mlx_optimized.py liegt so, dass es direkt importiert werden kann
# oder die Funktionen sind hier für _brute_force_search direkt implementiert/angepasst.
# Um Redundanz zu vermeiden, ist die Nutzung von mlx_optimized.py ideal.
try:
    from performance.mlx_optimized import compute_cosine_similarity_single, fast_top_k_indices, compute_cosine_similarity_batch #
    MLX_OPTIMIZED_AVAILABLE = True
except ImportError:
    logger.warning("mlx_optimized.py nicht gefunden. Brute-Force-Suche wird lokal implementiert (weniger optimal).")
    MLX_OPTIMIZED_AVAILABLE = False

try:
    from utils import ensure_directory, get_lock, validate_vector_shape
except ImportError:
    logging.critical("Failed import from utils.py.", exc_info=True)
    # ... (Definition von Dummy-Funktionen, die ImportError werfen) ...


logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Konfiguration für den VectorStore."""
    enable_hnsw: bool = True
    hnsw_config: Optional[HNSWConfig] = None # Wird durch HNSW-Parameter aus globaler Config befüllt
    
    # HNSW-spezifische Parameter, die an HNSWConfig übergeben werden
    # Diese sollten idealerweise aus Ihrer globalen PerformanceConfig (settings.py) stammen
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    hnsw_metric: str = 'l2' # oder 'cosine'
    hnsw_num_threads: Optional[int] = None # None für Auto-Detect in HNSWConfig

    auto_index_threshold: int = 100
    
    cache_enabled: bool = True
    query_cache_max_size: int = 1000
    query_cache_ttl_seconds: int = 3600

    def __post_init__(self):
        if self.hnsw_config is None:
            # Erstelle HNSWConfig basierend auf den Parametern in VectorStoreConfig
            self.hnsw_config = HNSWConfig(
                M=self.hnsw_m,
                ef_construction=self.hnsw_ef_construction,
                ef_search=self.hnsw_ef_search,
                metric=self.hnsw_metric,
                num_threads=self.hnsw_num_threads
                # Weitere HNSWConfig Parameter hier ggf. hinzufügen
            )
        # Überschreibe einzelne Werte in der HNSWConfig, falls sie in VectorStoreConfig spezifischer sind
        # (Dies ist optional, je nachdem wie Sie die Konfiguration gestalten wollen)
        self.hnsw_config.M = self.hnsw_m
        self.hnsw_config.ef_construction = self.hnsw_ef_construction
        self.hnsw_config.ef_search = self.hnsw_ef_search
        self.hnsw_config.metric = self.hnsw_metric
        if self.hnsw_num_threads is not None: # Nur wenn explizit gesetzt
            self.hnsw_config.num_threads = self.hnsw_num_threads


class VectorStore:
    def __init__(self, store_path: Union[str, Path], config: Optional[VectorStoreConfig] = None):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.config = config or VectorStoreConfig()
        # Sicherstellen, dass eine HNSWConfig-Instanz vorhanden ist
        if not isinstance(self.config.hnsw_config, HNSWConfig):
             self.config.hnsw_config = HNSWConfig( # Erzeuge basierend auf den Feldern in VectorStoreConfig
                M=self.config.hnsw_m,
                ef_construction=self.config.hnsw_ef_construction,
                ef_search=self.config.hnsw_ef_search,
                metric=self.config.hnsw_metric,
                num_threads=self.config.hnsw_num_threads
            )

        self.vectors: Optional[mx.array] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None # Wird beim Laden oder ersten Hinzufügen gesetzt
        
        self.hnsw_index: Optional[HNSWIndex] = None
        self._index_lock = threading.RLock()
        
        self._query_result_cache: Optional[Dict[str, Tuple[List[int], List[float], List[Dict[str, Any]]]]] = None
        if self.config.cache_enabled:
            self._query_result_cache = {}
        self._cache_lock = threading.RLock()
        self._cache_timestamps: Dict[str, float] = {}

        self.metrics = {
            'total_queries': 0, 'hnsw_queries': 0, 'brute_force_queries': 0,
            'cache_hits': 0, 'cache_misses': 0, 'avg_query_time_ms_ewma': 0.0,
            'last_query_time_ms': 0.0
        }
        self._ewma_alpha = 0.1

        self._load()
        logger.info(f"VectorStore initialisiert für Pfad: {self.store_path}. HNSW: {self.config.enable_hnsw}, Cache: {self.config.cache_enabled}")

    def _determine_dimension(self, vectors_array: Optional[mx.array]):
        if vectors_array is not None and vectors_array.ndim == 2 and vectors_array.shape[0] > 0:
            self.dimension = vectors_array.shape[1]
        elif self.dimension is None: # Fallback, falls noch keine Vektoren geladen wurden
            self.dimension = getattr(self.config.hnsw_config, 'dim', None) # Aus HNSWConfig, falls dort gesetzt
            if self.dimension is None:
                 # Letzter Fallback, sollte durch Konfig oder erste Daten gesetzt werden
                 logger.warning("Vektor-Dimension konnte nicht automatisch bestimmt werden und ist nicht in HNSWConfig gesetzt. Verwende Default 384.")
                 self.dimension = 384


    def _load(self):
        vectors_path = self.store_path / "vectors.npz"
        metadata_path = self.store_path / "metadata.jsonl"
        
        if vectors_path.exists():
            try:
                logger.debug(f"Versuche, Vektoren mit mx.load von {vectors_path} zu laden.")
                loaded_data = mx.load(str(vectors_path))
                # mx.load gibt ein dict-ähnliches Objekt zurück, wenn es eine .npz Datei ist
                if 'vectors' in loaded_data and isinstance(loaded_data['vectors'], mx.array):
                    self.vectors = loaded_data['vectors']
                    self._determine_dimension(self.vectors)
                    logger.info(f"{self.vectors.shape[0]} Vektoren (Dim: {self.dimension}) erfolgreich mit mx.load geladen.")
                else:
                    raise ValueError("Format von vectors.npz unerwartet oder 'vectors'-Schlüssel fehlt bei mx.load.")
            except Exception as e_mx_load:
                logger.warning(f"Fehler beim Laden von {vectors_path} mit mx.load: {e_mx_load}. Versuche NumPy-Fallback.")
                try:
                    np_data = np.load(vectors_path)
                    self.vectors = mx.array(np_data['vectors'])
                    self._determine_dimension(self.vectors)
                    logger.info(f"{self.vectors.shape[0]} Vektoren (Dim: {self.dimension}) mit NumPy-Fallback geladen.")
                except Exception as e_np_load:
                    logger.critical(f"Endgültiger Fehler beim Laden von Vektoren aus {vectors_path}: {e_np_load}", exc_info=True)
                    self.vectors = None
        else:
            logger.info(f"Keine Vektordatei ({vectors_path}) gefunden.")
            self.vectors = None
            self._determine_dimension(None) # Versucht Dimension aus Config zu lesen

        if metadata_path.exists():
            self.metadata = []
            try:
                with metadata_path.open('r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try: self.metadata.append(json.loads(line.strip()))
                        except json.JSONDecodeError: logger.warning(f"Ungültiges JSON in {metadata_path} Zeile {line_num}. Überspringe.")
                logger.info(f"{len(self.metadata)} Metadaten-Einträge geladen.")
            except Exception as e_meta: logger.error(f"Fehler beim Laden von Metadaten: {e_meta}", exc_info=True)
        
        if self.vectors is not None and len(self.metadata) != self.vectors.shape[0]:
            logger.error(f"Inkonsistenz nach Laden: {self.vectors.shape[0]} Vektoren, aber {len(self.metadata)} Metadaten! Store möglicherweise korrupt.")

        if self.config.enable_hnsw and self.vectors is not None and self.vectors.shape[0] > 0:
            self._load_or_build_index()
        elif self.config.enable_hnsw and (self.vectors is None or self.vectors.shape[0] == 0):
             logger.info("HNSW ist aktiviert, aber keine Vektoren zum Indizieren vorhanden.")


    def _load_or_build_index(self):
        if not self.config.enable_hnsw or self.vectors is None or self.vectors.shape[0] == 0: return
        if self.dimension is None: self._determine_dimension(self.vectors) # Sicherstellen, dass Dim bekannt ist

        index_base_path_str = str(self.store_path / "vector_store_hnsw_index")

        # HNSWConfig hier erstellen oder aus self.config.hnsw_config nehmen
        current_hnsw_config = self.config.hnsw_config
        if not isinstance(current_hnsw_config, HNSWConfig): # Fallback, falls nicht korrekt initialisiert
            current_hnsw_config = HNSWConfig()
            logger.warning("Verwende Default HNSWConfig für load_or_build_index.")

        # Ihre HNSW-Klasse erwartet 'dim' im Konstruktor
        # Die 'dim' in HNSWConfig ist eher für die Info, HNSWIndex.__init__ braucht es explizit.
        if self.dimension is None:
            logger.error("Dimension der Vektoren unbekannt. Kann HNSW-Index nicht laden/bauen.")
            return

        with self._index_lock:
            self.hnsw_index = HNSWIndex(dim=self.dimension, config=current_hnsw_config) #
            try:
                # Die load-Methode Ihrer HNSWIndex Klasse erwartet den Vektor-Array
                self.hnsw_index.load(index_base_path_str, vectors=self.vectors)
                logger.info(f"HNSW-Index mit {self.hnsw_index.n_points} Punkten von '{index_base_path_str}' geladen.")
                if self.hnsw_index.n_points != self.vectors.shape[0] or self.hnsw_index.dim != self.dimension:
                    logger.warning(f"Mismatch im geladenen HNSW-Index (Punkte: {self.hnsw_index.n_points} vs {self.vectors.shape[0]}, Dim: {self.hnsw_index.dim} vs {self.dimension}). Index wird neu gebaut.")
                    self._build_index_internal() # Ruft interne Build-Methode auf
            except FileNotFoundError:
                logger.info(f"Kein HNSW-Index unter '{index_base_path_str}' gefunden.")
                if self.vectors.shape[0] >= self.config.auto_index_threshold:
                    logger.info(f"Vektoranzahl ({self.vectors.shape[0]}) >= Schwelle ({self.config.auto_index_threshold}). Baue Index.")
                    self._build_index_internal()
            except Exception as e:
                logger.warning(f"Fehler beim Laden des HNSW-Index von '{index_base_path_str}': {e}. Index wird ggf. neu gebaut.", exc_info=True)
                if self.vectors.shape[0] >= self.config.auto_index_threshold:
                     self._build_index_internal()


    def _build_index_internal(self): # Umbenannt, um Konflikt mit öffentlicher `rebuild_index` zu vermeiden
        """Interne Methode zum Bauen des HNSW-Index. Lock wird extern erwartet."""
        if not self.config.enable_hnsw or self.vectors is None or self.vectors.shape[0] == 0:
            self.hnsw_index = None
            return
        if self.dimension is None: self._determine_dimension(self.vectors)
        if self.dimension is None:
             logger.error("Kann HNSW Index nicht bauen: Dimension unbekannt.")
             return

        logger.info(f"Erstelle HNSW-Index für {self.vectors.shape[0]} Vektoren (Dim: {self.dimension})...")
        # HNSWConfig aus self.config verwenden
        self.hnsw_index = HNSWIndex(dim=self.dimension, config=self.config.hnsw_config) #
        self.hnsw_index.build(self.vectors, show_progress=True) #
        
        index_base_path_str = str(self.store_path / "vector_store_hnsw_index")
        self.hnsw_index.save(index_base_path_str) #
        logger.info(f"HNSW-Index erfolgreich erstellt und unter '{index_base_path_str}' gespeichert.")

    def add_vectors(self, vectors_to_add: mx.array, metadata_to_add: List[Dict[str, Any]]):
        if not isinstance(vectors_to_add, mx.array): raise TypeError("Eingabe 'vectors_to_add' muss ein mx.array sein.")
        if vectors_to_add.ndim != 2: raise ValueError("Eingabe 'vectors_to_add' muss 2D sein.")
        if vectors_to_add.shape[0] != len(metadata_to_add): raise ValueError("Vektor- und Metadatenanzahl stimmen nicht überein.")
        if vectors_to_add.shape[0] == 0: return

        with self._index_lock:
            if self.vectors is None or self.vectors.shape[0] == 0:
                self.vectors = vectors_to_add
                self.metadata = list(metadata_to_add)
                self._determine_dimension(self.vectors)
                logger.info(f"{vectors_to_add.shape[0]} initiale Vektoren (Dim: {self.dimension}) hinzugefügt.")
                if self.config.enable_hnsw and self.vectors.shape[0] >= self.config.auto_index_threshold:
                    self._build_index_internal()
            else:
                if self.dimension is None: self._determine_dimension(self.vectors) # Sollte schon gesetzt sein
                if self.dimension != vectors_to_add.shape[1]:
                    raise ValueError(f"Dimensions-Mismatch: Store Dim {self.dimension}, neue Vektoren Dim {vectors_to_add.shape[1]}.")
                
                self.vectors = mx.concatenate([self.vectors, vectors_to_add], axis=0)
                self.metadata.extend(metadata_to_add)
                logger.info(f"{vectors_to_add.shape[0]} Vektoren angehängt. Gesamt: {self.vectors.shape[0]}.")

                if self.config.enable_hnsw:
                    if self.hnsw_index is not None:
                        logger.info("Erweitere bestehenden HNSW-Index...")
                        self.hnsw_index.extend_vectors(vectors_to_add) #
                        # Annahme: extend_vectors speichert den Index oder _save() tut es.
                        # Speichere explizit, falls extend_vectors es nicht tut:
                        # index_base_path_str = str(self.store_path / "vector_store_hnsw_index")
                        # self.hnsw_index.save(index_base_path_str)
                    elif self.vectors.shape[0] >= self.config.auto_index_threshold:
                        self._build_index_internal()
            
            mx.eval(self.vectors)
            self._save()
        self._clear_query_cache()

    def _save(self):
        if self.vectors is not None and self.vectors.shape[0] > 0:
            vectors_path = self.store_path / "vectors.npz"
            try:
                mx.savez(str(vectors_path), vectors=self.vectors)
                logger.debug(f"Vektoren mit mx.savez nach {vectors_path} gespeichert.")
            except Exception as e_mx_save:
                logger.error(f"Fehler bei mx.savez: {e_mx_save}. Fallback auf NumPy.")
                try: np.savez_compressed(vectors_path, vectors=np.array(self.vectors))
                except Exception as e_np_save: logger.critical(f"Speichern fehlgeschlagen: {e_np_save}", exc_info=True)
        elif self.vectors is not None and self.vectors.shape[0] == 0:
            (self.store_path / "vectors.npz").unlink(missing_ok=True)

        metadata_path = self.store_path / "metadata.jsonl"
        try:
            with metadata_path.open('w', encoding='utf-8') as f:
                for item in self.metadata: f.write(json.dumps(item) + '\n')
            logger.debug(f"Metadaten nach {metadata_path} gespeichert.")
        except Exception as e: logger.error(f"Fehler beim Speichern der Metadaten: {e}", exc_info=True)

        if self.config.enable_hnsw and self.hnsw_index is not None:
            with self._index_lock:
                try:
                    index_base_path_str = str(self.store_path / "vector_store_hnsw_index")
                    self.hnsw_index.save(index_base_path_str) #
                    logger.debug(f"HNSW-Index unter Präfix '{index_base_path_str}' gespeichert.")
                except Exception as e: logger.error(f"Fehler beim Speichern des HNSW-Index: {e}", exc_info=True)

    def _compute_cache_key(self, query_vector: mx.array, k: int, filter_str: Optional[str] = None) -> str:
        mx.eval(query_vector) # Sicherstellen, dass für Hashing konsistent
        query_bytes = query_vector.tobytes()
        filter_component = f"_filter_{hashlib.md5(filter_str.encode()).hexdigest()}" if filter_str else ""
        return f"{hashlib.md5(query_bytes).hexdigest()}_{k}{filter_component}"

    def _clear_query_cache(self):
        if self._query_result_cache is not None:
            with self._cache_lock:
                self._query_result_cache.clear()
                self._cache_timestamps.clear()
                logger.info("Query-Ergebnis-Cache geleert.")

    def _prune_query_cache(self):
        if self._query_result_cache is None or not self.config.cache_enabled : return
        with self._cache_lock:
            now = time.time()
            expired_keys = [k for k, ts in self._cache_timestamps.items() if now - ts > self.config.query_cache_ttl_seconds]
            for key in expired_keys:
                if key in self._query_result_cache: del self._query_result_cache[key]
                if key in self._cache_timestamps: del self._cache_timestamps[key]
            
            while len(self._query_result_cache) > self.config.query_cache_max_size:
                if not self._cache_timestamps: break
                oldest_key = min(self._cache_timestamps, key=self._cache_timestamps.get) # type: ignore
                if oldest_key in self._query_result_cache: del self._query_result_cache[oldest_key]
                if oldest_key in self._cache_timestamps: del self._cache_timestamps[oldest_key]

    def query(self, query_vector: mx.array, k: int = 10,
              use_hnsw: Optional[bool] = None,
              metadata_filter: Optional[Dict[str, Any]] = None
             ) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        if not isinstance(query_vector, mx.array): query_vector = mx.array(query_vector, dtype=mx.float32)
        query_1d_for_hnsw = query_vector.flatten() # HNSW erwartet 1D
        query_2d_for_brute = query_1d_for_hnsw.reshape(1,-1) if query_1d_for_hnsw.ndim == 1 else query_1d_for_hnsw

        if self.vectors is None or self.vectors.shape[0] == 0: return [], [], []

        query_start_time = time.perf_counter()
        filter_key_component = json.dumps(metadata_filter, sort_keys=True) if metadata_filter else ""
        cache_key = self._compute_cache_key(query_1d_for_hnsw, k, filter_key_component) if self._query_result_cache is not None else None

        if cache_key and self._query_result_cache is not None:
            self._prune_query_cache()
            with self._cache_lock:
                cached_result = self._query_result_cache.get(cache_key)
                if cached_result:
                    self._cache_timestamps[cache_key] = time.time()
                    self.metrics['cache_hits'] += 1
                    logger.debug(f"Query Cache HIT: {cache_key}")
                    self._update_query_time_metrics((time.perf_counter() - query_start_time) * 1000)
                    return cached_result
            self.metrics['cache_misses'] += 1
            logger.debug(f"Query Cache MISS: {cache_key}")

        use_hnsw_resolved = (use_hnsw is True or (use_hnsw is None and self.config.enable_hnsw)) and self.hnsw_index is not None
        if use_hnsw and (not self.config.enable_hnsw or not self.hnsw_index):
            logger.warning("HNSW-Nutzung angefordert, aber HNSW deaktiviert/Index nicht gebaut. Fallback.")

        raw_indices_mx: mx.array
        raw_distances_mx: mx.array

        if use_hnsw_resolved and self.hnsw_index:
            logger.debug(f"HNSW-Suche (k={k}, ef={self.hnsw_index.config.ef_search}).")
            raw_indices_mx, raw_distances_mx = self.hnsw_index.search(query_1d_for_hnsw, k=k * (3 if metadata_filter else 1)) # Mehr Kandidaten für Filterung holen
            self.metrics['hnsw_queries'] += 1
        else:
            logger.debug(f"Brute-Force-Suche (k={k}).")
            # _brute_force_search erwartet 2D Query
            raw_indices_mx, raw_distances_mx = self._brute_force_search(query_2d_for_brute, k=k * (3 if metadata_filter else 1))
            self.metrics['brute_force_queries'] += 1
        
        mx.eval(raw_indices_mx, raw_distances_mx)
        
        final_indices, final_distances, final_metadata = self._filter_and_prepare_results(
            raw_indices_mx.tolist(), raw_distances_mx.tolist(), metadata_filter, k
        )
        
        if cache_key and self._query_result_cache is not None:
            with self._cache_lock:
                if len(self._query_result_cache) < self.config.query_cache_max_size:
                    self._query_result_cache[cache_key] = (final_indices, final_distances, final_metadata)
                    self._cache_timestamps[cache_key] = time.time()

        self._update_query_time_metrics((time.perf_counter() - query_start_time) * 1000)
        self.metrics['total_queries'] += 1
        return final_indices, final_distances, final_metadata

    def _filter_and_prepare_results(self, indices_list: List[int], distances_list: List[float],
                                   metadata_filter: Optional[Dict[str, Any]], k_target: int
                                   ) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        final_indices, final_distances, final_metadata = [], [], []
        if not indices_list: return final_indices, final_distances, final_metadata

        for i in range(len(indices_list)):
            original_idx = indices_list[i]
            if not (0 <= original_idx < len(self.metadata)):
                logger.warning(f"Ungültiger Index {original_idx} von Suche erhalten. Übersprungen.")
                continue

            meta_item = self.metadata[original_idx]
            if metadata_filter:
                match = all(meta_item.get(key) == value for key, value in metadata_filter.items())
                if not match: continue
            
            final_indices.append(original_idx)
            final_distances.append(distances_list[i])
            final_metadata.append(meta_item)
            if len(final_indices) == k_target: break
        return final_indices, final_distances, final_metadata


    def _update_query_time_metrics(self, current_query_time_ms: float):
        self.metrics['last_query_time_ms'] = round(current_query_time_ms, 3)
        current_ewma = self.metrics['avg_query_time_ms_ewma']
        if current_ewma == 0.0 and self.metrics['total_queries'] <= 1 : # Erste Query
             self.metrics['avg_query_time_ms_ewma'] = round(current_query_time_ms, 3)
        else:
            self.metrics['avg_query_time_ms_ewma'] = round(
                (self._ewma_alpha * current_query_time_ms) + ((1 - self._ewma_alpha) * current_ewma), 3
            )

    def _brute_force_search(self, query_vector_2d: mx.array, k: int) -> Tuple[mx.array, mx.array]:
        if self.vectors is None or self.vectors.shape[0] == 0: return mx.array([]), mx.array([])
        if query_vector_2d.ndim != 2 or query_vector_2d.shape[0] != 1:
            raise ValueError(f"Erwartet 2D Query-Vektor (1, dim), erhielt {query_vector_2d.shape}")

        # Verwendung der kompilierten Funktionen aus mlx_optimized.py, falls verfügbar
        if MLX_OPTIMIZED_AVAILABLE:
            # compute_cosine_similarity_single erwartet 1D oder 2D Query und gibt 1D Scores zurück
            # query_vector_2d.flatten() wäre hier konsistenter, wenn compute_cosine_similarity_single 1D erwartet
            scores_mx = compute_cosine_similarity_single(query_vector_2d.flatten(), self.vectors) # Gibt Ähnlichkeiten zurück
            distances_mx = 1.0 - scores_mx # Konvertiere zu Distanz
            
            # Hole die Top K Indizes basierend auf *Distanz* (kleinste zuerst)
            # mx.argsort(distances_mx) sortiert aufsteigend (kleinste Distanz zuerst)
            indices_mx = mx.argsort(distances_mx)[:k]
            top_distances_mx = distances_mx[indices_mx]
            return indices_mx, top_distances_mx
        else: # Fallback auf lokale Implementierung (wie zuvor, aber mit korrektem Epsilon)
            query_flat = query_vector_2d.flatten()
            query_norm = mx.linalg.norm(query_flat)
            epsilon_q = mx.array(1e-8, dtype=query_norm.dtype)
            query_norm = mx.maximum(query_norm, epsilon_q)
            query_normalized = query_flat / query_norm

            db_vector_norms = mx.linalg.norm(self.vectors, axis=1)
            epsilon_db = mx.array(1e-8, dtype=db_vector_norms.dtype)
            db_vector_norms = mx.maximum(db_vector_norms, epsilon_db)
            normalized_db_vectors = self.vectors / db_vector_norms[:, None]
            
            similarities = mx.matmul(query_normalized.reshape(1, -1), normalized_db_vectors.T)
            distances = 1.0 - similarities.flatten()
            mx.eval(distances)
            
            num_items = distances.shape[0]; actual_k = min(k, num_items)
            if actual_k == 0: return mx.array([], dtype=mx.int32), mx.array([], dtype=mx.float32)
            
            indices = mx.argsort(distances)[:actual_k]
            return indices, distances[indices]


    def batch_query(self, query_vectors_batch: mx.array, k: int = 10,
                    metadata_filter: Optional[Dict[str, Any]] = None
                   ) -> Tuple[List[List[int]], List[List[float]], List[List[Dict[str, Any]]]]:
        if not isinstance(query_vectors_batch, mx.array): raise TypeError("Muss mx.array sein.")
        if query_vectors_batch.ndim != 2: raise ValueError("Muss 2D mx.array sein.")

        all_indices, all_distances, all_metadata = [], [], []
        
        # Erweiterte Logik: Nutze HNSW Batch Search wenn möglich und kein Filter
        # oder kompiliertes Batch Brute Force wenn HNSW nicht geht aber Filter auch nicht.
        # Wenn Filter, dann iterativ.
        
        can_use_hnsw_batch = (self.config.enable_hnsw and self.hnsw_index is not None and 
                              self.hnsw_index.n_points > 0 and metadata_filter is None)

        if can_use_hnsw_batch and self.hnsw_index:
            logger.debug(f"HNSW Batch-Suche für {query_vectors_batch.shape[0]} Queries.")
            # HNSW Batch Search erwartet k, holt k * (faktor falls filter) intern nicht
            # Die Filterung muss hier auch nachbearbeitet werden.
            # Wir holen mehr Ergebnisse, falls gefiltert werden soll.
            # Da HNSW Batch Search keinen Filter unterstützt, ist diese Logik nur für den Fall,
            # dass wir HNSW Batch Search machen und DANACH filtern.
            # Wenn metadata_filter hier gesetzt ist, ist can_use_hnsw_batch bereits False.
            # Also ist diese Bedingung hier immer metadata_filter is None.
            
            batch_indices_mx, batch_distances_mx = self.hnsw_index.batch_search(query_vectors_batch, k=k) #
            mx.eval(batch_indices_mx, batch_distances_mx)
            self.metrics['hnsw_queries'] += query_vectors_batch.shape[0]

            for i in range(query_vectors_batch.shape[0]):
                # Kein Post-Filter hier, da metadata_filter None sein muss für diesen Pfad
                indices_list = batch_indices_mx[i].tolist()
                distances_list = batch_distances_mx[i].tolist()
                metadata_list = [self.metadata[idx] for idx in indices_list if 0 <= idx < len(self.metadata)]
                
                valid_count = min(len(indices_list), len(distances_list), len(metadata_list))
                all_indices.append(indices_list[:valid_count])
                all_distances.append(distances_list[:valid_count])
                all_metadata.append(metadata_list[:valid_count])

        elif MLX_OPTIMIZED_AVAILABLE and metadata_filter is None and self.vectors is not None and self.vectors.shape[0] > 0:
            # Nutze kompiliertes Batch Brute Force, wenn kein Filter und HNSW nicht geht
            logger.debug(f"Kompilierte Brute-Force Batch-Suche für {query_vectors_batch.shape[0]} Queries.")
            batch_scores_mx = compute_cosine_similarity_batch(query_vectors_batch, self.vectors) #
            batch_distances_mx = 1.0 - batch_scores_mx
            mx.eval(batch_distances_mx)
            self.metrics['brute_force_queries'] += query_vectors_batch.shape[0]

            for i in range(query_vectors_batch.shape[0]):
                query_distances_mx = batch_distances_mx[i]
                num_items = query_distances_mx.shape[0]; actual_k = min(k, num_items)
                if actual_k == 0:
                    all_indices.append([]); all_distances.append([]); all_metadata.append([])
                    continue
                
                indices_mx = mx.argsort(query_distances_mx)[:actual_k]
                distances_mx = query_distances_mx[indices_mx]
                mx.eval(indices_mx, distances_mx)

                indices_list = indices_mx.tolist()
                distances_list = distances_mx.tolist()
                metadata_list = [self.metadata[idx] for idx in indices_list if 0 <= idx < len(self.metadata)]
                
                valid_count = min(len(indices_list), len(distances_list), len(metadata_list))
                all_indices.append(indices_list[:valid_count])
                all_distances.append(distances_list[:valid_count])
                all_metadata.append(metadata_list[:valid_count])
        else:
            logger.debug(f"Iterative Einzelabfragen für Batch ({query_vectors_batch.shape[0]} Queries). Filter: {metadata_filter is not None}")
            for i in range(query_vectors_batch.shape[0]):
                indices, distances, meta = self.query(query_vectors_batch[i], k, metadata_filter=metadata_filter)
                all_indices.append(indices)
                all_distances.append(distances)
                all_metadata.append(meta)
        
        self.metrics['total_queries'] += query_vectors_batch.shape[0] # Zähle jede Sub-Query
        return all_indices, all_distances, all_metadata


    def delete_vectors(self, indices_to_delete: List[int]):
        if self.vectors is None or self.vectors.shape[0] == 0: return 0
        num_to_delete = len(indices_to_delete)
        if num_to_delete == 0: return 0
        
        with self._index_lock:
            valid_indices_to_delete = sorted([idx for idx in set(indices_to_delete) if 0 <= idx < self.vectors.shape[0]], reverse=True)
            if not valid_indices_to_delete: return 0

            # Erstelle Maske oder lösche iterativ (Maske ist oft besser für Arrays)
            # Einfachere Methode für Python-Listen (Metadaten) und dann für Vektoren.
            # Da HNSW sowieso neu gebaut wird, ist die Performance des Löschens hier weniger kritisch.
            
            original_vector_count = self.vectors.shape[0]
            new_metadata = [m for i, m in enumerate(self.metadata) if i not in valid_indices_to_delete]
            
            # Für Vektoren: erstelle eine Maske der zu behaltenden Elemente
            keep_mask = np.ones(self.vectors.shape[0], dtype=bool)
            for idx in valid_indices_to_delete: keep_mask[idx] = False
            self.vectors = self.vectors[mx.array(keep_mask)] # Filter mit MLX boolean array

            self.metadata = new_metadata
            deleted_count_actual = original_vector_count - self.vectors.shape[0]
            logger.info(f"{deleted_count_actual} Vektoren gelöscht. Verbleibend: {self.vectors.shape[0]}.")

            if self.config.enable_hnsw:
                if self.vectors.shape[0] > 0 and self.vectors.shape[0] >= self.config.auto_index_threshold :
                    self._build_index_internal()
                else:
                    self.hnsw_index = None # Index entfernen oder nicht neu bauen, wenn unter Schwelle
                    (self.store_path / "vector_store_hnsw_index.config.json").unlink(missing_ok=True)
                    (self.store_path / "vector_store_hnsw_index.graph.pkl").unlink(missing_ok=True)
                    logger.info("HNSW-Index entfernt oder nicht neu gebaut (unter Schwelle).")
            self._save()
        self._clear_query_cache()
        return deleted_count_actual

    def update_vector(self, index_to_update: int, new_vector: mx.array, new_metadata: Dict[str, Any]):
        if not isinstance(new_vector, mx.array): raise TypeError("Muss mx.array sein.")
        if new_vector.ndim == 1: new_vector = new_vector.reshape(1,-1)
        if new_vector.shape[0] != 1: raise ValueError("Darf nur einen Vektor enthalten.")

        with self._index_lock:
            if self.vectors is None or not (0 <= index_to_update < self.vectors.shape[0]):
                raise IndexError(f"Index {index_to_update} ungültig (0 bis {self.vectors.shape[0]-1 if self.vectors else 0}).")
            if self.dimension is None: self._determine_dimension(self.vectors)
            if self.dimension != new_vector.shape[1]:
                 raise ValueError(f"Dim-Mismatch: Store {self.dimension}, neuer Vektor {new_vector.shape[1]}.")

            self.vectors[index_to_update] = new_vector[0]
            self.metadata[index_to_update] = new_metadata
            logger.info(f"Vektor an Index {index_to_update} aktualisiert.")

            if self.config.enable_hnsw:
                # Ein Update ist wie ein Delete + Add; Rebuild ist der konservative Weg
                logger.info("Baue HNSW-Index nach Vektor-Update neu auf.")
                self._build_index_internal()
            self._save()
        self._clear_query_cache()

    def get_stats(self) -> Dict[str, Any]:
        hnsw_info_stats = None
        if self.config.enable_hnsw and self.hnsw_index and hasattr(self.hnsw_index, 'get_stats'):
            # Ihre HNSWIndex Klasse hat keine get_stats() Methode,
            # aber wir können einige Attribute abfragen
            try:
                hnsw_info_stats = self.hnsw_index.get_stats() # Falls Sie es implementieren
            except AttributeError:
                 hnsw_info_stats = {
                     "nodes": self.hnsw_index.n_points, 
                     "entry_point": self.hnsw_index.entry_point,
                     "M": self.hnsw_index.config.M, # Zugriff auf config Attribut des Index
                     "ef_search": self.hnsw_index.config.ef_search,
                     "metric": self.hnsw_index.config.metric
                 }

        return {
            'path': str(self.store_path),
            'total_vectors': self.vectors.shape[0] if self.vectors is not None else 0,
            'vector_dimension': self.dimension,
            'metadata_count': len(self.metadata),
            'hnsw_enabled': self.config.enable_hnsw,
            'hnsw_index_active': self.hnsw_index is not None,
            'hnsw_stats': hnsw_info_stats,
            'cache_enabled': self.config.cache_enabled,
            'cache_current_size': len(self._query_result_cache) if self._query_result_cache is not None else 0,
            'cache_max_size': self.config.query_cache_max_size if self.config.cache_enabled else 0,
            'performance_metrics': self.metrics,
            'storage_size_mb': round(self._calculate_storage_size_mb(), 2)
        }

    def _calculate_storage_size_mb(self) -> float:
        total_size_bytes = 0
        if self.store_path.exists():
            for item_path_str in os.listdir(self.store_path): # Sicherer, falls store_path nicht existiert
                item = self.store_path / item_path_str
                if item.is_file(): total_size_bytes += item.stat().st_size
        return total_size_bytes / (1024 * 1024)

    def rebuild_index(self, force: bool = False):
        if not self.config.enable_hnsw: logger.info("HNSW deaktiviert."); return
        if self.vectors is None or self.vectors.shape[0] == 0: logger.info("Keine Vektoren zum Indizieren."); return
        
        with self._index_lock: # Lock für den Build-Prozess
            if force or self.vectors.shape[0] >= self.config.auto_index_threshold:
                logger.info(f"Fordere Neuerstellung des HNSW-Index an. Force: {force}")
                self._build_index_internal()
            else:
                logger.info(f"Bedingung für Index-Neuerstellung nicht erfüllt (Anzahl/Force).")

    def clear(self):
        with self._index_lock:
            self.vectors = None
            self.metadata = []
            self.hnsw_index = None
            self._clear_query_cache()
            
            if self.store_path.exists(): # Nur löschen, wenn das Verzeichnis existiert
                for item_path_str in os.listdir(self.store_path):
                    item = self.store_path / item_path_str
                    if item.is_file():
                        try: item.unlink()
                        except Exception as e: logger.error(f"Fehler beim Löschen von {item}: {e}")
            logger.info(f"VectorStore unter {self.store_path} geleert.")

# --- Veraltete Wrapper-Funktionen (zur Abwärtskompatibilität) ---
# Diese sollten in neuem Code nicht mehr verwendet werden.
_global_vector_stores: Dict[str, VectorStore] = {} # Key ist str(store_path)
_global_vector_stores_lock = threading.Lock()

def _get_store_path(user_id: str, model_name: str, base_path_str: Optional[str]=None) -> Path:
    """Hilfsfunktion, um den Store-Pfad zu generieren. Verwendet Config, falls base_path_str None."""
    if base_path_str is None:
        # Greife auf globale Config zu, um den Basispfad zu holen
        # Dies erfordert, dass die Config bereits initialisiert wurde.
        try:
            from config.settings import get_config
            config_manager = get_config()
            base_path_str = config_manager.storage.base_path
        except (ImportError, AttributeError):
            logger.warning("Globale Config für Basispfad nicht verfügbar. Verwende Default.")
            base_path_str = "~/.mlx_vector_db_data/vector_stores" # Fallback
            
    return Path(base_path_str).expanduser() / f"user_{user_id}" / model_name


def get_store(user_id: str, model_id: str, config: Optional[VectorStoreConfig] = None) -> VectorStore:
    """
    @deprecated: Ruft eine VectorStore-Instanz ab oder erstellt sie.
                 Verwenden Sie stattdessen direkte Instanziierung von VectorStore
                 und verwalten Sie Instanzen in Ihrer Anwendung (z.B. via FastAPI State/Dependency).
    """
    store_path = _get_store_path(user_id, model_id)
    store_path_key = str(store_path.resolve())

    with _global_vector_stores_lock:
        if store_path_key not in _global_vector_stores:
            logger.info(f"(Legacy Wrapper) Erstelle neue VectorStore-Instanz für: {store_path_key}")
            # Hier sollte die globale Konfiguration verwendet werden, wenn `config` None ist
            # Dies erfordert Zugriff auf die ConfigManager-Instanz
            effective_config = config
            if effective_config is None:
                try:
                    from config.settings import get_config
                    global_cfg = get_config()
                    # Erstelle eine VectorStoreConfig basierend auf globalen Einstellungen
                    effective_config = VectorStoreConfig(
                        enable_hnsw=global_cfg.performance.enable_hnsw,
                        hnsw_m=global_cfg.performance.hnsw_m,
                        hnsw_ef_construction=global_cfg.performance.hnsw_ef_construction,
                        hnsw_ef_search=global_cfg.performance.hnsw_ef_search,
                        hnsw_metric=global_cfg.performance.hnsw_metric,
                        hnsw_num_threads=getattr(global_cfg.performance, 'hnsw_num_threads', None), # Falls nicht in PerformanceConfig
                        auto_index_threshold=global_cfg.performance.auto_index_threshold,
                        cache_enabled=getattr(global_cfg.performance, 'cache_enabled', True), # query_cache_enabled
                        query_cache_max_size=global_cfg.performance.query_result_cache_max_size,
                        query_cache_ttl_seconds=global_cfg.performance.query_result_cache_ttl_seconds
                    )
                except (ImportError, AttributeError) as e_cfg:
                    logger.warning(f"Globale Config konnte nicht geladen werden für Legacy Wrapper: {e_cfg}. Verwende Default VectorStoreConfig.")
                    effective_config = VectorStoreConfig()

            _global_vector_stores[store_path_key] = VectorStore(store_path, config=effective_config)
        
        # Wenn Config übergeben wurde und sich von der existierenden unterscheidet, ggf. Instanz neu erstellen
        elif config is not None and _global_vector_stores[store_path_key].config != config:
            logger.warning(f"(Legacy Wrapper) Konfiguration für existierenden Store {store_path_key} hat sich geändert. Erstelle Instanz neu.")
            _global_vector_stores[store_path_key] = VectorStore(store_path, config=config)
            
        return _global_vector_stores[store_path_key]

# Wrapper für die alte API-Signatur aus service.vector_store.py
# Diese sind nun stark vereinfacht und gehen davon aus, dass die API-Routen
# die `user_id` und `model_id` korrekt übergeben und für die Pfad-Erstellung sorgen.
# Es ist besser, wenn die API-Routen direkt die `VectorStore`-Klasse und deren Methoden verwenden.

def create_store_legacy(user_id: str, model_name: str):
    """@deprecated"""
    logger.warning("DEPRECATED: create_store_legacy called. Use VectorStore instance.")
    get_store(user_id, model_name) # Stellt sicher, dass Verzeichnis erstellt wird

def add_vectors_legacy(user_id: str, model_id: str, vectors: mx.array, metadata: List[Dict]):
    """@deprecated"""
    logger.warning("DEPRECATED: add_vectors_legacy called. Use store.add_vectors().")
    store = get_store(user_id, model_id)
    store.add_vectors(vectors, metadata)

def query_vectors_legacy(user_id: str, model_id: str, query_vector: mx.array, k: int, filter_metadata: Optional[Dict] = None) -> List[Dict]:
    """@deprecated. Gibt nur Metadaten und Scores zurück."""
    logger.warning("DEPRECATED: query_vectors_legacy called. Use store.query().")
    store = get_store(user_id, model_id)
    _, distances, metadatas = store.query(query_vector, k, metadata_filter=filter_metadata)
    
    results = []
    for i in range(len(metadatas)):
        res_item = metadatas[i].copy()
        # Konvertiere Distanz zu Ähnlichkeit für die alte API (angenommen Cosine-Distanz)
        # Dies hängt von der Metrik ab, die im HNSW/BruteForce verwendet wird.
        # Wenn L2: höhere Werte sind schlechter. Wenn Cosine Sim: höhere Werte sind besser.
        # Wenn Cosine Distanz (1-sim): niedrigere Werte sind besser.
        # Die `query` Methode gibt Distanzen zurück.
        similarity_score = 1.0 - distances[i] if store.config.hnsw_config.metric == 'cosine' else -distances[i]
        res_item['similarity_score'] = similarity_score
        # Die alte API gab nicht die Indizes direkt zurück, nur die Metadaten mit Score.
        results.append(res_item)
    return results


# ... (weitere Legacy Wrapper für delete_vectors, count_vectors, store_exists, etc.) ...
# Diese sollten alle `get_store(user_id, model_id)` verwenden und dann die entsprechende
# Methode auf der `store`-Instanz aufrufen.