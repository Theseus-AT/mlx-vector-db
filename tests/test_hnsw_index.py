# tests/test_hnsw_index.py
# Überarbeitete Testdatei für Ihre produktionsreife HNSWIndex Implementierung.

import pytest
import mlx.core as mx
import numpy as np
from pathlib import Path
import tempfile
import os # Für os.cpu_count in HNSWConfig Fallback

# Passen Sie diesen Import an den tatsächlichen Speicherort Ihrer hnsw_index.py an
# z.B. wenn hnsw_index.py in performance/ liegt:
# from performance.hnsw_index import HNSWIndex, HNSWConfig
# Für dieses Beispiel nehmen wir an, sie liegt so, dass der direkte Import funktioniert:
from performance.hnsw_index import HNSWIndex, HNSWConfig # VERWENDET IHRE HOCHGELADENE hnsw_index.py

DIMENSION = 64  # Kleinere Dimension für schnellere Tests
NUM_VECTORS_SMALL = 100
NUM_VECTORS_MEDIUM = 500

# Helper zum Erstellen von sample_vectors, um Seed-Setzung zu zentralisieren
def create_sample_vectors(num_vectors, dim, seed=42) -> mx.array:
    np.random.seed(seed) # NumPy Seed für Reproduzierbarkeit der Daten
    random.seed(seed)    # Python Random Seed
    # MLX hat keine direkte globale Seed-Setzung für mx.random wie NumPy.
    # Die Zufälligkeit kommt von der NumPy-Konvertierung.
    return mx.array(np.random.rand(num_vectors, dim).astype(np.float32))

@pytest.fixture
def sample_vectors_small() -> mx.array:
    return create_sample_vectors(NUM_VECTORS_SMALL, DIMENSION, seed=42)

@pytest.fixture
def sample_vectors_medium() -> mx.array:
    return create_sample_vectors(NUM_VECTORS_MEDIUM, DIMENSION, seed=123)

@pytest.fixture
def default_config() -> HNSWConfig:
    # Eine Basiskonfiguration für die meisten Tests
    return HNSWConfig(M=8, ef_construction=100, ef_search=50, metric='l2', num_threads=2)

@pytest.fixture
def hnsw_index_empty(default_config: HNSWConfig) -> HNSWIndex:
    return HNSWIndex(dim=DIMENSION, config=default_config)

@pytest.fixture
def hnsw_index_built_small(hnsw_index_empty: HNSWIndex, sample_vectors_small: mx.array) -> HNSWIndex:
    hnsw_index_empty.build(sample_vectors_small, show_progress=False)
    return hnsw_index_empty

@pytest.fixture
def hnsw_index_built_medium(default_config: HNSWConfig, sample_vectors_medium: mx.array) -> HNSWIndex:
    # Eigene Instanz, um Konflikte mit hnsw_index_empty zu vermeiden, falls Tests parallel laufen
    index = HNSWIndex(dim=DIMENSION, config=default_config)
    index.build(sample_vectors_medium, show_progress=False)
    return index


class TestHNSWIndexProduction:
    def test_initialization(self, default_config: HNSWConfig):
        index = HNSWIndex(dim=DIMENSION, config=default_config)
        assert index.dim == DIMENSION
        assert index.n_points == 0
        assert index.entry_point is None
        assert index.config.M == default_config.M
        assert index.config.ef_search == default_config.ef_search

    def test_build_small_dataset(self, hnsw_index_empty: HNSWIndex, sample_vectors_small: mx.array):
        index = hnsw_index_empty
        index.build(sample_vectors_small, show_progress=False)
        
        assert index.n_points == NUM_VECTORS_SMALL
        assert index.entry_point is not None
        assert index.entry_point < NUM_VECTORS_SMALL # Entry Point muss ein gültiger Index sein
        assert len(index.nodes) == NUM_VECTORS_SMALL
        assert index.vectors is not None
        assert index.vectors.shape == (NUM_VECTORS_SMALL, DIMENSION)

    def test_add_single_vector_to_built_index(self, hnsw_index_built_small: HNSWIndex, sample_vectors_small: mx.array):
        index = hnsw_index_built_small
        original_n_points = index.n_points

        new_vector_idx = original_n_points # Nächster Index
        # Erstelle einen neuen Vektor, der sich leicht von anderen unterscheidet
        np.random.seed(777)
        new_vector_np = np.random.rand(1, DIMENSION).astype(np.float32)
        new_vector_mx = mx.array(new_vector_np)
        
        # extend_vectors erwartet ein 2D Array
        index.extend_vectors(new_vector_mx)

        assert index.n_points == original_n_points + 1
        assert new_vector_idx in index.nodes # Der neue Vektor sollte als Node existieren
        # Überprüfe, ob der Vektor korrekt am Ende von self.vectors angefügt wurde
        assert mx.array_equal(index.vectors[new_vector_idx], new_vector_mx[0])


    def test_search_basic(self, hnsw_index_built_small: HNSWIndex, sample_vectors_small: mx.array):
        index = hnsw_index_built_small
        query_vector = sample_vectors_small[0]
        k = 5
        
        indices, distances = index.search(query_vector, k=k)
        mx.eval(indices, distances) # Sicherstellen, dass die Berechnung abgeschlossen ist

        assert isinstance(indices, mx.array)
        assert isinstance(distances, mx.array)
        assert indices.shape[0] == k
        assert distances.shape[0] == k
        
        # Das erste Ergebnis sollte der Query-Vektor selbst sein (ID 0, Distanz nahe 0)
        indices_list = indices.tolist()
        distances_list = distances.tolist()

        assert indices_list[0] == 0 # Vektor-ID 0
        assert distances_list[0] < 1e-5 # Sehr kleine Distanz für L2 bei identischem Vektor

        for idx, dist in zip(indices_list, distances_list):
            assert 0 <= idx < NUM_VECTORS_SMALL
            assert dist >= -1e-5 # Distanz sollte nicht-negativ sein (kleine Toleranz für Float-Ungenauigkeiten)

    def test_search_cosine_metric(self, default_config: HNSWConfig, sample_vectors_small: mx.array):
        cosine_config = HNSWConfig(
            M=default_config.M,
            ef_construction=default_config.ef_construction,
            ef_search=default_config.ef_search,
            metric='cosine', # Teste Cosine-Distanz
            num_threads=2
        )
        index = HNSWIndex(dim=DIMENSION, config=cosine_config)
        index.build(sample_vectors_small, show_progress=False)

        query_vector = sample_vectors_small[10] # Ein anderer Vektor als Query
        k = 3
        indices, distances = index.search(query_vector, k=k)
        mx.eval(indices, distances)

        indices_list = indices.tolist()
        distances_list = distances.tolist()
        
        assert indices_list[0] == 10
        assert distances_list[0] < 1e-5 # Cosine Distanz zum selben Vektor ist 0

    def test_batch_search(self, hnsw_index_built_small: HNSWIndex, sample_vectors_small: mx.array):
        index = hnsw_index_built_small
        num_queries = 3
        query_vectors_mx = sample_vectors_small[:num_queries]
        k = 3
        
        all_indices, all_distances = index.batch_search(query_vectors_mx, k=k, num_threads=2)
        mx.eval(all_indices, all_distances)

        assert all_indices.shape == (num_queries, k)
        assert all_distances.shape == (num_queries, k)

        for i in range(num_queries):
            indices_list = all_indices[i].tolist()
            distances_list = all_distances[i].tolist()
            assert indices_list[0] == i # Erster Treffer sollte die Query selbst sein
            assert distances_list[0] < 1e-5


    def test_save_load_index(self, hnsw_index_built_medium: HNSWIndex, sample_vectors_medium: mx.array):
        index_original = hnsw_index_built_medium

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "prod_hnsw.pkl"
            index_original.save(str(index_path)) # Ihre save-Methode erwartet einen String-Pfad
            assert index_path.exists()

            # Erstelle eine neue Config für den geladenen Index (oder stelle sicher, dass sie kompatibel ist)
            loaded_config = HNSWConfig(
                M=index_original.config.M,
                ef_construction=index_original.config.ef_construction,
                ef_search=index_original.config.ef_search,
                metric=index_original.config.metric,
                num_threads=index_original.config.num_threads
            )
            index_loaded = HNSWIndex(dim=DIMENSION, config=loaded_config)
            # Wichtig: Die Vektoren müssen dem geladenen Index wieder bekannt gemacht werden.
            index_loaded.load(str(index_path), vectors=sample_vectors_medium)


            assert index_loaded.dim == index_original.dim
            assert index_loaded.n_points == index_original.n_points
            assert index_loaded.entry_point == index_original.entry_point
            assert len(index_loaded.nodes) == len(index_original.nodes)
            assert index_loaded.vectors is not None # Sicherstellen, dass Vektoren geladen wurden
            assert mx.array_equal(index_loaded.vectors, sample_vectors_medium)


            # Teste, ob der geladene Index funktioniert
            query_vector = sample_vectors_medium[0]
            k = 5
            
            indices_orig, dists_orig = index_original.search(query_vector, k=k)
            indices_load, dists_load = index_loaded.search(query_vector, k=k)
            mx.eval(indices_orig, dists_orig, indices_load, dists_load)

            assert mx.array_equal(indices_orig, indices_load)
            assert mx.allclose(dists_orig, dists_load, atol=1e-6)

    def test_extend_vectors(self, hnsw_index_empty: HNSWIndex, sample_vectors_small: mx.array):
        index = hnsw_index_empty
        
        # Baue Index mit der ersten Hälfte der Vektoren
        num_initial = NUM_VECTORS_SMALL // 2
        initial_vectors = sample_vectors_small[:num_initial]
        index.build(initial_vectors, show_progress=False)
        assert index.n_points == num_initial

        # Erweitere mit der zweiten Hälfte
        remaining_vectors = sample_vectors_small[num_initial:]
        index.extend_vectors(remaining_vectors)
        assert index.n_points == NUM_VECTORS_SMALL
        assert mx.array_equal(index.vectors[:num_initial], initial_vectors)
        assert mx.array_equal(index.vectors[num_initial:], remaining_vectors)

        # Teste Suche nach einem Vektor aus dem erweiterten Teil
        query_idx_in_sample = num_initial + 5 # Ein Index aus dem zweiten Batch
        query_vector = sample_vectors_small[query_idx_in_sample]
        
        # Der Index im HNSW ist query_idx_in_sample, da die Vektoren konkateniert wurden
        # und die Indizes in _insert_batch relativ zum *aktuellen* self.vectors sind.
        # In Ihrer extend_vectors Logik werden die neuen Indizes von old_size bis new_size vergeben.
        
        indices, distances = index.search(query_vector, k=1)
        mx.eval(indices, distances)
        
        assert indices.tolist()[0] == query_idx_in_sample
        assert distances.tolist()[0] < 1e-5

    def test_empty_index_search(self, hnsw_index_empty: HNSWIndex):
        index = hnsw_index_empty
        # Wichtig: Der leere Index hat self.vectors = None. `build` muss zuerst aufgerufen werden.
        # Oder die search-Methode muss diesen Fall (leerer, nicht gebauter Index) abfangen.
        # Ihre aktuelle search-Methode prüft `if self.entry_point is None`, was korrekt ist.
        
        query_vector = create_sample_vectors(1, DIMENSION)[0]
        indices, distances = index.search(query_vector, k=5)
        mx.eval(indices, distances)
        
        assert indices.shape[0] == 0
        assert distances.shape[0] == 0
        
    def test_search_k_greater_than_nodes(self, hnsw_index_empty: HNSWIndex, sample_vectors_small: mx.array):
        index = hnsw_index_empty
        num_nodes_to_add = 5
        index.build(sample_vectors_small[:num_nodes_to_add], show_progress=False)

        query_vector = sample_vectors_small[0]
        indices, distances = index.search(query_vector, k=10) # k > Anzahl der Knoten
        mx.eval(indices, distances)

        assert indices.shape[0] == num_nodes_to_add # Sollte alle Knoten zurückgeben
        assert indices.tolist()[0] == 0
        assert distances.tolist()[0] < 1e-5

    def test_accuracy_recall_small_set(self, default_config: HNSWConfig, sample_vectors_medium: mx.array):
        """Prüft den Recall für einen kleineren Datensatz, wo Brute-Force noch machbar ist."""
        index = HNSWIndex(dim=DIMENSION, config=default_config)
        index.build(sample_vectors_medium, show_progress=False)

        num_queries = 20
        k = 10
        total_recall = 0.0

        query_indices_to_test = np.random.choice(NUM_VECTORS_MEDIUM, num_queries, replace=False)

        for i in query_indices_to_test:
            query_vector = sample_vectors_medium[i]
            
            # HNSW Ergebnisse
            hnsw_indices_mx, _ = index.search(query_vector, k=k, ef=default_config.ef_search + 20) # Etwas höheres ef für Recall-Test
            mx.eval(hnsw_indices_mx)
            hnsw_indices_set = set(hnsw_indices_mx.tolist())

            # Brute-Force Ergebnisse (manuell für diesen Test)
            all_distances = []
            if default_config.metric == 'l2':
                diff = sample_vectors_medium - query_vector
                dists_mx = mx.sum(diff * diff, axis=1)
            else: # cosine
                # (vereinfachte Cosine-Distanz für Test, Ihre _batch_distances_mlx ist genauer)
                dot_prods = mx.sum(sample_vectors_medium * query_vector, axis=1)
                norms_db = mx.sqrt(mx.sum(sample_vectors_medium * sample_vectors_medium, axis=1))
                norm_q = mx.sqrt(mx.sum(query_vector * query_vector))
                similarities = dot_prods / (norms_db * norm_q + 1e-9)
                dists_mx = 1.0 - similarities
            
            mx.eval(dists_mx)
            
            # Sortiere alle Distanzen und nimm die Top K Indizes
            # argsort gibt Indizes, die das Array sortieren würden.
            brute_force_indices_sorted = mx.argsort(dists_mx).tolist()
            brute_force_top_k_set = set(brute_force_indices_sorted[:k])
            
            overlap = len(hnsw_indices_set.intersection(brute_force_top_k_set))
            recall = overlap / k
            total_recall += recall
            
        average_recall = total_recall / num_queries
        logger.info(f"HNSW Test - Average Recall@{k} (vs Brute Force): {average_recall:.4f} for {num_queries} queries on {NUM_VECTORS_MEDIUM} items.")
        assert average_recall >= 0.90 # Erwarte einen hohen Recall (z.B. >90% oder >95%)