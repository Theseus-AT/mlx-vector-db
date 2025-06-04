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
from typing import List, Tuple, Dict # Dict hinzugefügt
import json
import logging # Logging hinzugefügt
import shutil # Für Verzeichnisbereinigung

logger = logging.getLogger("mlx_vector_db.perf_demo_hnsw")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- WICHTIGE ANPASSUNG DER IMPORTE ---
# Importieren Sie VectorStore und VectorStoreConfig aus Ihrer optimierten vector_store.py
# Stellen Sie sicher, dass vector_store.py im Python-Pfad ist oder der Import relativ korrekt ist.
# Wenn vector_store.py im Root-Verzeichnis liegt:
try:
    from vector_store import VectorStore, VectorStoreConfig #
except ImportError as e1:
    logger.error(f"Konnte VectorStore nicht aus 'vector_store.py' importieren: {e1}")
    try:
        # Fallback, falls es in einem service-Verzeichnis liegt (wie ursprünglich)
        from service.vector_store import VectorStore, VectorStoreConfig
        logger.info("VectorStore aus 'service.vector_store' importiert.")
    except ImportError as e2:
        logger.critical(f"VectorStore konnte weder aus Root noch aus 'service' importiert werden: {e2}. Demo kann nicht laufen.")
        raise

# Importieren Sie HNSWConfig aus Ihrer produktionsreifen hnsw_index.py
# Stellen Sie sicher, dass hnsw_index.py im Python-Pfad ist oder der Import relativ korrekt ist.
# Wenn hnsw_index.py im Root-Verzeichnis liegt:
try:
    from hnsw_index import HNSWConfig #
except ImportError as e1:
    logger.error(f"Konnte HNSWConfig nicht aus 'hnsw_index.py' importieren: {e1}")
    try:
        # Fallback, falls es in einem performance-Verzeichnis liegt
        from performance.hnsw_index import HNSWConfig
        logger.info("HNSWConfig aus 'performance.hnsw_index' importiert.")
    except ImportError as e2:
        logger.critical(f"HNSWConfig konnte weder aus Root noch aus 'performance' importiert werden: {e2}. Demo kann nicht laufen.")
        raise
# --- ENDE ANPASSUNG IMPORTE ---


class PerformanceTester: # Ihre Klasse aus performance_demo.py
    def __init__(self, base_test_dir: str = "./temp_perf_test_stores"):
        self.results: Dict[str, Dict[Any, Any]] = { # Genauerer Typ-Hint
            'build_times': {},
            'query_times': {},
            'accuracy': {},
            'parameter_tuning': {} # Geändert von 'memory_usage'
        }
        self.base_test_dir = Path(base_test_dir)
        self.base_test_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PerformanceTester initialisiert. Testdaten in: {self.base_test_dir.resolve()}")

    def _cleanup_store(self, store_path: Path):
        if store_path.exists() and store_path.is_dir():
            try:
                shutil.rmtree(store_path)
                logger.debug(f"Test-Store Verzeichnis {store_path} gelöscht.")
            except Exception as e:
                logger.error(f"Fehler beim Löschen des Test-Store Verzeichnisses {store_path}: {e}")

    def test_build_performance(self, vector_counts: List[int], dim: int = 384):
        logger.info("\n=== Teste Index-Aufbau Performance ===")
        current_results = {}
        for n_vectors in vector_counts:
            logger.info(f"\nTeste mit {n_vectors} Vektoren (Dim: {dim})...")
            vectors = mx.random.normal((n_vectors, dim), dtype=mx.float32)
            # Metadaten werden für add_vectors benötigt, auch wenn hier nicht primär getestet
            metadata = [{"id": f"vec_{i}"} for i in range(n_vectors)]
            
            store_path = self.base_test_dir / f"build_test_store_{n_vectors}"
            # Verwende eine HNSWConfig Instanz für Klarheit
            hnsw_conf = HNSWConfig(M=16, ef_construction=200, metric='l2') # Beispielwerte
            vs_conf = VectorStoreConfig(
                enable_hnsw=True, auto_index_threshold=1, # Index immer bauen
                hnsw_config=hnsw_conf
            )
            store = VectorStore(store_path, config=vs_conf)
            
            start_time = time.perf_counter()
            store.add_vectors(vectors, metadata) # add_vectors baut den Index, falls Bedingungen erfüllt
            mx.block_until_ready() # Sicherstellen, dass alle MLX Operationen abgeschlossen sind
            build_time = time.perf_counter() - start_time
            
            current_results[n_vectors] = build_time
            logger.info(f"Aufbauzeit für {n_vectors} Vektoren: {build_time:.3f} Sekunden")
            self._cleanup_store(store_path)
        self.results['build_times'] = current_results
        
    def test_query_performance(self, n_vectors: int = 10000, dim: int = 384, 
                             n_queries: int = 100, k_values: List[int] = [1, 10, 50]):
        logger.info(f"\n=== Teste Query Performance ({n_vectors} Vektoren, Dim: {dim}) ===")
        store_path = self.base_test_dir / "query_perf_store"
        hnsw_conf = HNSWConfig(M=16, ef_construction=200, ef_search=100, metric='l2')
        vs_conf = VectorStoreConfig(enable_hnsw=True, auto_index_threshold=1, hnsw_config=hnsw_conf)
        store = VectorStore(store_path, config=vs_conf)

        logger.info("Erstelle Test-Datensatz für Query-Performance...")
        # Reduziere Batch-Größe für schnellere Erstellung bei großen n_vectors
        add_batch_size = min(10000, n_vectors)
        for i in range(0, n_vectors, add_batch_size):
            num_in_batch = min(add_batch_size, n_vectors - i)
            batch_vectors = mx.random.normal((num_in_batch, dim), dtype=mx.float32)
            batch_metadata = [{"id": f"qvec_{j+i}"} for j in range(num_in_batch)]
            store.add_vectors(batch_vectors, batch_metadata)
        mx.block_until_ready()
        logger.info(f"Test-Datensatz mit {store.vectors.shape[0] if store.vectors else 0} Vektoren erstellt.")

        query_vectors_mx = mx.random.normal((n_queries, dim), dtype=mx.float32)
        mx.eval(query_vectors_mx) # Stelle sicher, dass Queries bereit sind

        current_results_query_times = {}
        for k in k_values:
            logger.info(f"\nTeste Query mit k={k}:")
            # HNSW Performance
            hnsw_times = []
            for i in range(n_queries):
                start = time.perf_counter()
                _, _, _ = store.query(query_vectors_mx[i], k=k, use_hnsw=True)
                mx.block_until_ready()
                hnsw_times.append(time.perf_counter() - start)
            
            # Brute Force Performance (ggf. auf kleinerem Subset für Zeitersparnis)
            bf_times = []
            n_bf_queries = min(max(10, n_queries // 10), 100) # Reduzierte Anzahl für Brute-Force
            logger.info(f"Teste Brute-Force mit {n_bf_queries} Queries (Subset)...")
            for i in range(n_bf_queries):
                start = time.perf_counter()
                _, _, _ = store.query(query_vectors_mx[i], k=k, use_hnsw=False) # Erzwinge Brute-Force
                mx.block_until_ready()
                bf_times.append(time.perf_counter() - start)
                
            avg_hnsw_ms = np.mean(hnsw_times) * 1000
            p95_hnsw_ms = np.percentile(hnsw_times, 95) * 1000
            avg_bf_ms = np.mean(bf_times) * 1000
            
            current_results_query_times[f"hnsw_k{k}_avg_ms"] = avg_hnsw_ms
            current_results_query_times[f"hnsw_k{k}_p95_ms"] = p95_hnsw_ms
            current_results_query_times[f"brute_force_k{k}_avg_ms"] = avg_bf_ms
            
            logger.info(f"  HNSW (k={k}): {avg_hnsw_ms:.3f} ms (avg), {p95_hnsw_ms:.3f} ms (p95)")
            logger.info(f"  Brute Force (k={k}): {avg_bf_ms:.3f} ms (avg für {n_bf_queries} Queries)")
            if avg_hnsw_ms > 0: logger.info(f"  Speedup HNSW vs BF: {avg_bf_ms/avg_hnsw_ms:.2f}x")
        
        self.results['query_times'] = current_results_query_times
        self._cleanup_store(store_path)

    def test_accuracy(self, n_vectors: int = 1000, dim: int = 64, 
                     n_queries: int = 50, k: int = 10): # Kleinere Werte für schnelleren Genauigkeitstest
        logger.info(f"\n=== Teste Genauigkeit (Recall@{k} für {n_vectors} Vektoren, Dim: {dim}) ===")
        store_path = self.base_test_dir / "accuracy_test_store"
        # Parameter für gute Genauigkeit wählen
        hnsw_conf = HNSWConfig(M=32, ef_construction=250, ef_search=150, metric='l2')
        vs_conf = VectorStoreConfig(enable_hnsw=True, auto_index_threshold=1, hnsw_config=hnsw_conf)
        store = VectorStore(store_path, config=vs_conf)

        vectors_mx = mx.random.normal((n_vectors, dim), dtype=mx.float32)
        metadata = [{"original_idx": i} for i in range(n_vectors)] # Wichtig für Recall-Vergleich
        store.add_vectors(vectors_mx, metadata)
        mx.block_until_ready()

        query_vectors_mx = mx.random.normal((n_queries, dim), dtype=mx.float32)
        mx.eval(query_vectors_mx)

        recalls = []
        for i in range(n_queries):
            query_mx_single = query_vectors_mx[i]
            
            # Ground Truth mit Brute-Force
            indices_bf, _, _ = store.query(query_mx_single, k=k, use_hnsw=False)
            
            # HNSW Ergebnisse
            # ef_search kann hier für den Test angepasst/erhöht werden
            indices_hnsw, _, _ = store.query(query_mx_single, k=k, use_hnsw=True) 
            
            set_bf = set(indices_bf)
            set_hnsw = set(indices_hnsw)
            
            recall_val = len(set_hnsw.intersection(set_bf)) / max(1, len(set_bf)) # Teile durch len(set_bf) für korrekten Recall
            recalls.append(recall_val)
            
        avg_recall = np.mean(recalls) if recalls else 0.0
        self.results['accuracy'][f'recall@{k}'] = avg_recall
        logger.info(f"Durchschnittlicher Recall@{k}: {avg_recall:.4f} (basierend auf {n_queries} Queries)")
        if recalls:
             logger.info(f"Min Recall: {min(recalls):.4f}, Max Recall: {max(recalls):.4f}")
        self._cleanup_store(store_path)

    # Die Methode test_parameter_tuning aus Ihrer Datei ist gut und kann hier übernommen werden.
    # Stellen Sie sicher, dass sie die angepasste Initialisierung von VectorStore und HNSWConfig verwendet.
    def test_parameter_tuning(self, n_vectors: int = 5000, dim: int = 128, n_queries_tune: int = 50):
        logger.info(f"\n=== Teste HNSW Parameter-Tuning ({n_vectors} Vektoren, Dim: {dim}) ===")
        vectors_mx = mx.random.normal((n_vectors, dim), dtype=mx.float32)
        metadata = [{"id": f"tune_vec_{i}"} for i in range(n_vectors)]
        query_vectors_tune_mx = mx.random.normal((n_queries_tune, dim), dtype=mx.float32)
        mx.eval(vectors_mx, query_vectors_tune_mx)

        m_values = [8, 16, 32]
        ef_construction_values = [100, 200] # Weniger Werte für schnellere Demo
        ef_search_values = [50, 100]

        current_results_param_tuning = {}
        for m_val in m_values:
            for efc_val in ef_construction_values:
                store_path_param = self.base_test_dir / f"param_tune_store_M{m_val}_efC{efc_val}"
                hnsw_conf_build = HNSWConfig(M=m_val, ef_construction=efc_val, metric='l2')
                vs_conf_build = VectorStoreConfig(enable_hnsw=True, auto_index_threshold=1, hnsw_config=hnsw_conf_build)
                
                store = VectorStore(store_path_param, config=vs_conf_build)
                
                logger.info(f"\nTeste Parameter: M={m_val}, ef_construction={efc_val}")
                build_start_time = time.perf_counter()
                store.add_vectors(vectors_mx, metadata) # Baut Index
                mx.block_until_ready()
                build_duration = time.perf_counter() - build_start_time
                logger.info(f"  Aufbauzeit: {build_duration:.3f}s")

                for efs_val in ef_search_values:
                    if store.hnsw_index: # Stelle sicher, dass Index existiert
                        store.hnsw_index.config.ef_search = efs_val # ef_search dynamisch anpassen
                    
                    query_times_param = []
                    for q_idx in range(n_queries_tune):
                        query_single_tune = query_vectors_tune_mx[q_idx]
                        q_start_time = time.perf_counter()
                        _, _, _ = store.query(query_single_tune, k=10, use_hnsw=True)
                        mx.block_until_ready()
                        query_times_param.append(time.perf_counter() - q_start_time)
                    
                    avg_q_time_ms = np.mean(query_times_param) * 1000
                    param_key = f"M{m_val}_efC{efc_val}_efS{efs_val}"
                    current_results_param_tuning[param_key] = {
                        "build_time_s": build_duration,
                        "avg_query_time_ms": avg_q_time_ms
                    }
                    logger.info(f"    ef_search={efs_val}: Avg Query Time = {avg_q_time_ms:.3f}ms")
                self._cleanup_store(store_path_param)
        self.results['parameter_tuning'] = current_results_param_tuning

    def generate_report_and_save(self, report_filename_base: str = "hnsw_direct_performance"):
        logger.info("\n=== Erstelle Performance Bericht ===")
        # (Code für Plotting aus Ihrer Datei kann hierhin übernommen und angepasst werden,
        #  um die in self.results gesammelten Daten zu visualisieren)
        
        # Beispiel: Einfache Textausgabe der Ergebnisse
        logger.info("--- Ergebnisse Zusammenfassung ---")
        for category, cat_results in self.results.items():
            logger.info(f"\nKategorie: {category}")
            if isinstance(cat_results, dict):
                for key, value in cat_results.items():
                    if isinstance(value, dict): # Für Parameter Tuning
                        logger.info(f"  {key}:")
                        for sub_key, sub_value in value.items():
                             logger.info(f"    {sub_key}: {sub_value:.4f}")
                    else:
                        logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {cat_results}")

        # Speichere Ergebnisse als JSON
        json_filepath = self.base_test_dir / f"{report_filename_base}_results.json"
        try:
            with open(json_filepath, 'w') as f:
                json.dump(self.results, f, indent=4)
            logger.info(f"Rohdaten des Berichts gespeichert als: {json_filepath}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der JSON-Ergebnisse: {e}")
        
        # Hier könnte der Matplotlib-Code für Graphen stehen.
        # Da dies komplex sein kann und matplotlib eine externe Abhängigkeit ist,
        # lasse ich es für diese reine Code-Generierung vorerst weg, aber Ihr Ansatz war gut.


def main(): # Hauptfunktion für dieses Skript
    logger.info("Starte umfassende HNSW Performance- und Genauigkeitstests...")
    tester = PerformanceTester(base_test_dir="./temp_hnsw_perf_tests") # Eindeutiges Verzeichnis

    # Konfiguration der Tests
    tester.test_build_performance(vector_counts=[1000, 5000, 10000], dim=128) # Kleinere Sets für schnellere Demo
    tester.test_query_performance(n_vectors=10000, dim=128, n_queries=100, k_values=[1, 10])
    tester.test_accuracy(n_vectors=1000, dim=64, n_queries=50, k=10)
    # Parameter Tuning kann zeitaufwändig sein, ggf. kleinere Werte für schnelle Durchläufe
    tester.test_parameter_tuning(n_vectors=2000, dim=64, n_queries_tune=20) 
    
    tester.generate_report_and_save()
    logger.info("Alle HNSW Performance- und Genauigkeitstests abgeschlossen.")
    return tester.results

if __name__ == "__main__":
    results = main()