# performance/hnsw_index.py

import hnswlib
import numpy as np
import logging
import time
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger("mlx_hnsw_lib")

class ProductionHNSWIndex:
    """
    Eine produktionsreife HNSW-Implementierung, die die C++-Bibliothek
    'hnswlib' für maximale Performance nutzt. Diese Klasse managed den
    Lebenszyklus des Index, inklusive Persistenz auf der Festplatte.
    """

    def __init__(self, dimension: int, store_path: Path, metric: str = 'cosine', max_elements: int = 10000):
        self.dimension = dimension
        self.store_path = store_path
        self.index_file_path = self.store_path / "hnsw_index.bin"
        # hnswlib verwendet 'l2' für euklidische Distanz
        self.metric = 'l2' if metric == 'euclidean' else metric
        
        self.index: Optional[hnswlib.Index] = None
        self.is_loaded = False
        self.max_elements = max_elements

        # Versuche, einen existierenden Index beim Start zu laden
        self._load_index()

    def build(self, data: np.ndarray, M: int = 16, ef_construction: int = 200, num_threads: int = -1):
        """
        Baut oder aktualisiert den HNSW-Index mit neuen Vektoren.

        Args:
            data (np.ndarray): Ein 2D-NumPy-Array von Vektoren zum Indexieren.
            M (int): Maximale Anzahl von Verbindungen pro Knoten.
            ef_construction (int): Suchtiefe während des Index-Aufbaus.
            num_threads (int): Anzahl der Threads für den Aufbau (-1 für alle verfügbaren).
        """
        num_elements = data.shape[0]
        if num_elements == 0:
            logger.warning("Keine Vektoren zum Indexieren vorhanden.")
            return

        logger.info(f"Baue HNSW-Index für {num_elements} Vektoren (Metric: {self.metric})...")
        start_time = time.time()

        # Initialisiere den Index
        self.index = hnswlib.Index(space=self.metric, dim=self.dimension)
        # Setze die maximale Größe. Wenn mehr Daten kommen, muss der Index neu erstellt werden.
        self.max_elements = max(self.max_elements, num_elements)
        self.index.init_index(max_elements=self.max_elements, ef_construction=ef_construction, M=M)
        
        # Füge die Daten hinzu
        ids = np.arange(num_elements)
        self.index.add_items(data, ids, num_threads=num_threads)
        
        self.is_loaded = True
        build_time = time.time() - start_time
        logger.info(f"HNSW-Index-Aufbau abgeschlossen in {build_time:.2f} Sekunden.")

        # Speichere den neu gebauten Index
        self.save_index()

    def search(self, query_data: np.ndarray, k: int, ef_search: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sucht die k nächsten Nachbarn für einen oder mehrere Query-Vektoren.

        Args:
            query_data (np.ndarray): 1D- oder 2D-Array mit den Abfrage-Vektoren.
            k (int): Die Anzahl der zurückzugebenden Nachbarn.
            ef_search (int): Suchtiefe. Höher = genauer, aber langsamer.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Ein Tupel aus (Indizes, Distanzen).
        """
        if not self.is_loaded or self.index is None:
            raise RuntimeError("HNSW-Index ist nicht gebaut oder geladen.")
        
        if self.index.get_current_count() == 0:
             return np.array([]), np.array([])

        # Setze die Such-Effizienz
        self.index.set_ef(ef_search)
        
        # Führe die Suche aus
        labels, distances = self.index.knn_query(query_data, k=k)
        
        return labels, distances

    def save_index(self):
        """Speichert den HNSW-Index in einer Datei, um ihn wiederverwenden zu können."""
        if self.index is None:
            return
        
        logger.info(f"Speichere HNSW-Index nach: {self.index_file_path}")
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(self.index_file_path))

    def _load_index(self):
        """Lädt einen HNSW-Index aus einer Datei, falls vorhanden."""
        if not self.index_file_path.exists():
            logger.info(f"Keine Index-Datei unter {self.index_file_path} gefunden.")
            return
        
        try:
            logger.info(f"Lade existierenden HNSW-Index von: {self.index_file_path}")
            self.index = hnswlib.Index(space=self.metric, dim=self.dimension)
            self.index.load_index(str(self.index_file_path), max_elements=self.max_elements)
            self.is_loaded = True
            logger.info(f"Index mit {self.index.get_current_count()} Elementen erfolgreich geladen.")
        except Exception as e:
            logger.error(f"Fehler beim Laden des HNSW-Index: {e}. Index wird neu erstellt.")
            self.is_loaded = False
            self.index = None