# demo.py - Korrigierte Imports am Anfang der Datei
#!/usr/bin/env python3
"""
Demo script for MLX Vector Database
Demonstrates basic functionality and usage patterns using the VectorStore class.
"""
import numpy as np
import mlx.core as mx
import time
from pathlib import Path
import shutil
import logging

# Standardisierte Imports - KORRIGIERT
from service.vector_store import MLXVectorStore as VectorStore, VectorStoreConfig
from performance.hnsw_index import HNSWConfig

logger = logging.getLogger("mlx_vector_db.demo")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Globale Konfiguration für die Demo-Stores
DEMO_BASE_PATH = Path("~/.mlx_vector_db_demo_stores").expanduser()
DEMO_BASE_PATH.mkdir(parents=True, exist_ok=True)

demo_vs_config = VectorStoreConfig(
    dimension=384,
    metric="cosine",
    enable_hnsw=True,
    hnsw_config=HNSWConfig(
        M=16,
        ef_construction=100,
        ef_search=50,
        metric='l2'
    )
)

def get_demo_store(user_id: str, model_id: str) -> VectorStore:  # KORRIGIERT: Rückgabetyp
    """Hilfsfunktion zum Erstellen/Abrufen einer VectorStore-Instanz für Demos."""
    store_path = DEMO_BASE_PATH / f"user_{user_id}" / model_id
    return VectorStore(store_path, config=demo_vs_config)

def cleanup_store(store: VectorStore):  # KORRIGIERT: Parameter-Typ
    """Bereinigt einen Store und löscht sein Verzeichnis."""
    store_path_to_delete = store.store_path
    try:
        store.clear()
        if store_path_to_delete.exists() and store_path_to_delete.is_dir():
            shutil.rmtree(store_path_to_delete)
        print(f"🧹 Store-Verzeichnis {store_path_to_delete} bereinigt.")
    except Exception as e:
        print(f"Fehler beim Bereinigen von {store_path_to_delete}: {e}")

def run_basic_demo():
    """Führt grundlegende Operationen mit der VectorStore-Klasse vor."""
    print("🧠 MLX Vector Database Demo (mit VectorStore Klasse)")
    print("=" * 50)
    
    user_id = "demo_user_v2"
    model_id = "mistral_v2"
    store = get_demo_store(user_id, model_id)

    print(f"📁 Store wird initialisiert/geladen für {user_id}/{model_id} unter {store.store_path}")
    if store.store_path.exists():
        print(f"✅ Store-Verzeichnis existiert.")

    # Vektoren hinzufügen
    print(f"➕ Füge Beispiel-Vektoren hinzu...")
    vecs_np = np.random.rand(5, store.dimension or 384).astype(np.float32)
    meta = [{"id": f"chunk_{i}", "source": "demo", "content": f"Sample content {i}"} for i in range(vecs_np.shape[0])]
    
    start_time = time.time()
    store.add_vectors(vecs_np, meta)
    add_time = time.time() - start_time
    print(f"   {vecs_np.shape[0]} Vektoren hinzugefügt in {add_time:.3f}s")

    # Vektoren abfragen
    print(f"🔍 Frage Vektoren ab...")
    query_vec_np = vecs_np[0]
    k_val = 3
    
    start_time = time.time()
    indices, distances, result_metadata = store.query(query_vec_np, k=k_val)
    query_time = time.time() - start_time
    
    print(f"   Query abgeschlossen in {query_time:.3f}s")
    print(f"   {len(indices)} Ergebnisse gefunden für k={k_val}:")
    for i in range(len(indices)):
        print(f"     {i+1}. Index: {indices[i]}, Distanz: {distances[i]:.4f}, Meta-ID: {result_metadata[i].get('id')}")

    # Batch-Query-Demo
    print(f"🧠 Teste Batch-Query...")
    batch_query_vecs_np = vecs_np[:3]
    k_batch = 2
    
    start_time = time.time()
    all_indices, all_distances, all_meta = store.batch_query(batch_query_vecs_np, k=k_batch)
    batch_time = time.time() - start_time
    print(f"   Batch-Query ({batch_query_vecs_np.shape[0]} Queries) abgeschlossen in {batch_time:.3f}s")
    for i in range(len(all_indices)):
        print(f"     Ergebnisse für Query {i+1}: {len(all_indices[i])} Treffer")

    # Store-Statistiken
    print(f"📊 Store-Statistiken:")
    stats = store.get_stats()
    print(f"   Vektoren: {stats.get('vector_count', 0)}")
    print(f"   Dimension: {stats.get('dimension', 'N/A')}")

    # Vektor löschen
    print(f"🗑️ Lösche Vektor mit id 'chunk_1'...")
    idx_to_delete = -1
    for i, m in enumerate(store.metadata):
        if m.get("id") == "chunk_1":
            idx_to_delete = i
            break
    
    if idx_to_delete != -1:
        deleted_count = store.delete_vectors([idx_to_delete])
        print(f"   {deleted_count} Vektor(en) gelöscht.")
    else:
        print(f"   Vektor mit id 'chunk_1' nicht gefunden.")

    # Finale Statistiken
    final_stats = store.get_stats()
    print(f"📊 Finale Statistiken:")
    print(f"   Vektoren: {final_stats.get('vector_count', 0)}")

    # Aufräumen
    cleanup_store(store)
    print(f"✅ Basis-Demo (V2) erfolgreich abgeschlossen!")

def run_performance_demo_local():
    """Performance-Demo mit VectorStore"""
    print("\n🚀 Performance Demo (Lokal mit VectorStore Klasse)")
    print("=" * 50)
    
    user_id = "perf_user_v2"
    model_id = "performance_test_v2"
    store = get_demo_store(user_id, model_id)

    sizes = [100, 500, 1000]
    dim = store.dimension or 384

    for size in sizes:
        print(f"\n📈 Teste mit {size} Vektoren (Dim: {dim})...")
        
        vecs_np = np.random.normal(size=(size, dim)).astype(np.float32)
        meta = [{"id": f"vec_{i}", "batch": "perf_test_v2"} for i in range(size)]
        
        start_time = time.perf_counter()
        store.add_vectors(vecs_np, meta)
        add_time = time.perf_counter() - start_time
        
        query_vec_np = np.random.normal(size=(dim,)).astype(np.float32)

        query_times = []
        for _ in range(min(30, size // 10 + 1)):
            q_start_time = time.perf_counter()
            store.query(query_vec_np, k=10, use_hnsw=True)
            query_times.append(time.perf_counter() - q_start_time)
        
        avg_query_time_ms = (sum(query_times) / len(query_times)) * 1000 if query_times else 0
        
        print(f"   Hinzufügen-Zeit: {add_time:.3f}s ({size/add_time if add_time > 0 else float('inf'):.1f} Vektoren/s)")
        print(f"   Avg. Query-Zeit (HNSW, k=10): {avg_query_time_ms:.3f}ms")
    
    cleanup_store(store)
    print(f"✅ Lokale Performance-Demo abgeschlossen!")

def run_advanced_demo_local():
    """Erweiterte Features Demo"""
    print("\n🔧 Erweiterte Features Demo (Lokal mit VectorStore Klasse)")
    print("=" * 50)
    
    user_id = "advanced_user_v2"
    model_id = "advanced_test_v2"
    store = get_demo_store(user_id, model_id)
    dim = store.dimension or 128

    vecs_np = np.random.normal(size=(20, dim)).astype(np.float32)
    meta = [
        {"id": f"doc_{i}", "category": "A" if i < 10 else "B", "priority": i % 3, "lang": "de" if i % 2 == 0 else "en"}
        for i in range(vecs_np.shape[0])
    ]
    store.add_vectors(vecs_np, meta)

    print(f"🔍 Teste Metadaten-Filterung...")
    query_np_adv = vecs_np[0]

    # Filter nach Kategorie 'A'
    _, _, cat_a_results_meta = store.query(query_np_adv, k=10, filter_metadata={"category": "A"})
    print(f"   Ergebnisse für Kategorie 'A': {len(cat_a_results_meta)}")
    assert all(m["category"] == "A" for m in cat_a_results_meta)

    # Filter nach Priorität 1 UND Sprache 'en'
    _, _, prio_lang_results_meta = store.query(query_np_adv, k=10, filter_metadata={"priority": 1, "lang": "en"})
    print(f"   Ergebnisse für Priorität 1 & Sprache 'en': {len(prio_lang_results_meta)}")
    assert all(m["priority"] == 1 and m["lang"] == "en" for m in prio_lang_results_meta)

    # Testfall: Filter, der keine Ergebnisse liefert
    _, _, no_results_meta = store.query(query_np_adv, k=10, filter_metadata={"category": "C"})
    print(f"   Ergebnisse für nicht existente Kategorie 'C': {len(no_results_meta)}")
    assert len(no_results_meta) == 0

    cleanup_store(store)
    print(f"✅ Erweiterte Demo (V2) abgeschlossen!")

def main():
    """Führt alle Demo-Funktionen aus."""
    try:
        run_basic_demo()
        run_performance_demo_local()
        run_advanced_demo_local()
        
        print(f"\n🎉 Alle lokalen Demos (V2) erfolgreich abgeschlossen!")
        print(f"📖 Für API-Tests, starten Sie den Server und verwenden Sie die entsprechenden API-Demo-Skripte.")
        
    except Exception as e:
        logger.error(f"❌ Demo fehlgeschlagen: {e}", exc_info=True)

if __name__ == "__main__":
    main()