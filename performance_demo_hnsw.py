# performance_demo_hnsw.py
# (Basiert auf Ihrer Datei performance_demo.py mit der PerformanceTester Klasse)
"""
Performance- und Genauigkeitstests für die HNSW-Implementierung und VectorStore-Klasse.
Testet direkt die Komponenten, nicht über die API.
"""
import mlx.core as mx
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt # matplotlib muss installiert sein: pip install matplotlib
# KORRIGIERT: Fehlender Import für List, Tuple, etc. hinzugefügt
from typing import List, Tuple, Dict, Any
import json
import logging 
import shutil # Für Verzeichnisbereinigung
from service.optimized_vector_store import MLXVectorStore as VectorStore, MLXVectorStoreConfig  
from performance.hnsw_index import AdaptiveHNSWConfig as HNSWConfig

logger = logging.getLogger("mlx_vector_db.perf_demo_hnsw")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class PerformanceTester:
    def __init__(self, base_test_dir: str = "./temp_perf_test_stores"):
        self.results: Dict[str, Dict[Any, Any]] = {
            'build_times': {},
            'query_times': {},
            'accuracy': {},
            'parameter_tuning': {}
        }
        self.base_test_dir = Path(base_test_dir)
        self.base_test_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PerformanceTester initialisiert. Testdaten in: {self.base_test_dir.resolve()}")

# ... (Rest der Datei bleibt unverändert) ...
# (Fügen Sie hier den Rest Ihrer performance_demo_hnsw.py Datei ein)
    # ... (Ihre bestehende Klasse)
    def test_build_performance(self, vector_counts: List[int], dim: int = 384):
        logger.info("\n=== Teste Index-Aufbau Performance ===")
        for n_vectors in vector_counts:
            # ... (Setup)
            vectors = mx.random.normal((n_vectors, dim), dtype=mx.float32)
            metadata = [{"id": f"vec_{i}"} for i in range(n_vectors)]
            store = ... # Store-Initialisierung
            
            start_time = time.perf_counter()
            store.add_vectors(vectors, metadata)
            store.optimize() # Index-Aufbau explizit anstoßen
            
            # KORRIGIERT: mx.block_until_ready() ersetzt durch mx.eval()
            mx.eval(store._vectors) 
            if store._hnsw_index and store._hnsw_index.vectors is not None:
                mx.eval(store._hnsw_index.vectors)

            build_time = time.perf_counter() - start_time
            # ... (Rest der Funktion)
            
    def test_query_performance(self, n_vectors: int = 10000, dim: int = 384, n_queries: int = 100, k_values: List[int] = [1, 10, 50]):
        # ... (Setup)
        query_vectors_mx = mx.random.normal((n_queries, dim), dtype=mx.float32)
        mx.eval(query_vectors_mx)

        for k in k_values:
            # ...
            for i in range(n_queries):
                start = time.perf_counter()
                indices, _, _ = store.query(query_vectors_mx[i], k=k, use_hnsw=True)
                # KORRIGIERT: Kein block_until_ready nötig, da query synchron ist 
                # und intern eval aufruft oder das Ergebnis direkt nutzt.
                # Wenn die Abfrage asynchron wäre, bräuchte man hier ein Await.
                # mx.eval(indices) # Normalerweise nicht nötig, da tolist() implizit evaluiert
                hnsw_times.append(time.perf_counter() - start)
            # ... (Rest der Funktion)

# ... (Rest der Datei performance_demo_hnsw.py)
# Der Hauptpunkt ist das Entfernen von `mx.block_until_ready()`
# und das Sicherstellen, dass Arrays evaluiert werden, bevor die Zeitmessung gestoppt wird.
def main():
    logger.info("Starte umfassende HNSW Performance- und Genauigkeitstests...")
    tester = PerformanceTester(base_test_dir="./temp_hnsw_perf_tests")

    tester.test_build_performance(vector_counts=[1000, 5000, 10000], dim=128)
    tester.test_query_performance(n_vectors=10000, dim=128, n_queries=100, k_values=[1, 10])
    tester.test_accuracy(n_vectors=1000, dim=64, n_queries=50, k=10)
    tester.test_parameter_tuning(n_vectors=2000, dim=64, n_queries_tune=20) 
    
    tester.generate_report_and_save()
    logger.info("Alle HNSW Performance- und Genauigkeitstests abgeschlossen.")
    return tester.results

if __name__ == "__main__":
    results = main()