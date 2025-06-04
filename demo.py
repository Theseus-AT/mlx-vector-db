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

# Standardisierte Imports
from service.vector_store import MLXVectorStore, VectorStoreConfig
from performance.hnsw_index import HNSWConfig

logger = logging.getLogger("mlx_vector_db.demo")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Globale Konfiguration für die Demo-Stores (kann angepasst werden)
# Diese Werte würden idealerweise aus Ihrer settings.py stammen für Konsistenz.
DEMO_BASE_PATH = Path("~/.mlx_vector_db_demo_stores").expanduser()
DEMO_BASE_PATH.mkdir(parents=True, exist_ok=True)

# Standard HNSW-Konfiguration für die Demos (aus Ihrer hnsw_index.py)
# In VectorStoreConfig wird jetzt eine HNSWConfig erwartet oder über Parameter erstellt.
# Wir erstellen eine VectorStoreConfig, die intern eine HNSWConfig initialisiert.
demo_vs_config = VectorStoreConfig(
    enable_hnsw=True,
    auto_index_threshold=50, # Kleinere Schwelle für Demos, damit Index schneller gebaut wird
    hnsw_m=16, # Beispielwerte
    hnsw_ef_construction=100,
    hnsw_ef_search=50,
    hnsw_metric='l2',
    cache_enabled=True, # Query-Cache in VectorStore aktivieren
    query_cache_max_size=100
)

def get_demo_store(user_id: str, model_id: str) -> VectorStore:
    """Hilfsfunktion zum Erstellen/Abrufen einer VectorStore-Instanz für Demos."""
    store_path = DEMO_BASE_PATH / f"user_{user_id}" / model_id
    # `VectorStore` __init__ erstellt das Verzeichnis, falls nicht vorhanden.
    return VectorStore(store_path, config=demo_vs_config)

def cleanup_store(store: VectorStore):
    """Bereinigt einen Store und löscht sein Verzeichnis."""
    store_path_to_delete = store.store_path
    try:
        store.clear() # Leert die Inhalte des Stores (Dateien etc.)
        if store_path_to_delete.exists() and store_path_to_delete.is_dir():
            shutil.rmtree(store_path_to_delete) # Löscht das Verzeichnis selbst
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

    # 1. Store-Erstellung (implizit durch Instanziierung)
    print(f"📁 Store wird initialisiert/geladen für {user_id}/{model_id} unter {store.store_path}")
    if store.store_path.exists(): # Einfacher Check
        print(f"✅ Store-Verzeichnis existiert.")

    # 2. Vektoren hinzufügen
    print(f"➕ Füge Beispiel-Vektoren hinzu...")
    # Konvertiere NumPy-Arrays zu mx.array für die Übergabe an VectorStore
    vecs_np = np.random.rand(5, store.dimension or 384).astype(np.float32) # Dimension aus Store oder Default
    vecs_mx = mx.array(vecs_np)
    meta = [{"id": f"chunk_{i}", "source": "demo", "content": f"Sample content {i}"} for i in range(vecs_mx.shape[0])]
    
    start_time = time.time()
    store.add_vectors(vecs_mx, meta)
    add_time = time.time() - start_time
    print(f"   {vecs_mx.shape[0]} Vektoren hinzugefügt in {add_time:.3f}s")

    # 3. Vektoren abfragen
    print(f"🔍 Frage Vektoren ab...")
    query_vec_mx = vecs_mx[0] # Verwende den ersten hinzugefügten Vektor als Query
    k_val = 3
    
    start_time = time.time()
    # Die query-Methode der neuen VectorStore-Klasse gibt (indices, distances, metadata) zurück
    indices, distances, result_metadata = store.query(query_vec_mx, k=k_val)
    query_time = time.time() - start_time
    
    print(f"   Query abgeschlossen in {query_time:.3f}s")
    print(f"   {len(indices)} Ergebnisse gefunden für k={k_val}:")
    for i in range(len(indices)):
        print(f"     {i+1}. Index: {indices[i]}, Distanz: {distances[i]:.4f}, Meta-ID: {result_metadata[i].get('id')}")

    # 4. Batch-Query-Demo
    print(f"🧠 Teste Batch-Query...")
    batch_query_vecs_mx = vecs_mx[:3] # Nimm die ersten 3 Vektoren
    k_batch = 2
    
    start_time = time.time()
    # store.batch_query gibt Tuple(List[List[indices]], List[List[distances]], List[List[metadata]])
    all_indices, all_distances, all_meta = store.batch_query(batch_query_vecs_mx, k=k_batch)
    batch_time = time.time() - start_time
    print(f"   Batch-Query ({batch_query_vecs_mx.shape[0]} Queries) abgeschlossen in {batch_time:.3f}s")
    for i in range(len(all_indices)):
        print(f"     Ergebnisse für Query {i+1}: {len(all_indices[i])} Treffer")


    # 5. Store-Statistiken
    print(f"📊 Store-Statistiken:")
    stats = store.get_stats() # Verwendet die get_stats Methode von VectorStore
    print(f"   Vektoren: {stats.get('total_vectors', 0)}")
    print(f"   Metadaten: {stats.get('metadata_count', 0)}")
    print(f"   Dimension: {stats.get('vector_dimension', 'N/A')}")
    print(f"   HNSW aktiv: {stats.get('hnsw_index_active', False)}")

    # 6. Vektoren löschen (basierend auf Indizes)
    # Um dies analog zur alten Demo zu machen (Löschen nach Metadaten-ID),
    # müssten wir zuerst die Indizes der zu löschenden Vektoren finden.
    print(f"🗑️ Lösche Vektor mit id 'chunk_1'...")
    idx_to_delete = -1
    for i, m in enumerate(store.metadata):
        if m.get("id") == "chunk_1":
            idx_to_delete = i
            break
    
    if idx_to_delete != -1:
        deleted_count = store.delete_vectors([idx_to_delete]) # delete_vectors erwartet eine Liste von Indizes
        print(f"   {deleted_count} Vektor(en) gelöscht.")
    else:
        print(f"   Vektor mit id 'chunk_1' nicht gefunden.")

    # 7. Finale Statistiken
    final_stats = store.get_stats()
    print(f"📊 Finale Statistiken:")
    print(f"   Vektoren: {final_stats.get('total_vectors', 0)}")

    # 8. Aufräumen
    cleanup_store(store)
    print(f"✅ Basis-Demo (V2) erfolgreich abgeschlossen!")


def run_performance_demo_local(): # Umbenannt, um Konflikt mit API-Demo zu vermeiden
    """Führt eine Performance-Demonstration mit größeren Datensätzen direkt mit VectorStore durch."""
    print("\n🚀 Performance Demo (Lokal mit VectorStore Klasse)")
    print("=" * 50)
    
    user_id = "perf_user_v2"
    model_id = "performance_test_v2"
    store = get_demo_store(user_id, model_id)

    sizes = [100, 500, 1000] # Kleinere Größen für schnellere Demo hier
    dim = store.dimension or 384

    for size in sizes:
        print(f"\n📈 Teste mit {size} Vektoren (Dim: {dim})...")
        
        vecs_mx = mx.random.normal((size, dim), dtype=mx.float32)
        meta = [{"id": f"vec_{i}", "batch": "perf_test_v2"} for i in range(size)]
        
        start_time = time.perf_counter()
        store.add_vectors(vecs_mx, meta)
        mx.block_until_ready() # Wichtig für genaue Zeitmessung von MLX-Operationen
        add_time = time.perf_counter() - start_time
        
        query_vec_mx = mx.random.normal((dim,), dtype=mx.float32)
        mx.eval(query_vec_mx) # Stelle sicher, dass Query-Vektor bereit ist

        query_times = []
        for _ in range(min(30, size // 10 + 1)): # Mache einige Abfragen
            q_start_time = time.perf_counter()
            _, _, _ = store.query(query_vec_mx, k=10, use_hnsw=True) # Teste mit HNSW
            mx.block_until_ready()
            query_times.append(time.perf_counter() - q_start_time)
        
        avg_query_time_ms = (sum(query_times) / len(query_times)) * 1000 if query_times else 0
        
        print(f"   Hinzufügen-Zeit: {add_time:.3f}s ({size/add_time if add_time > 0 else float('inf'):.1f} Vektoren/s)")
        print(f"   Avg. Query-Zeit (HNSW, k=10): {avg_query_time_ms:.3f}ms")
        
        # Bereinige nur die Vektoren dieses Batches für den nächsten Durchlauf, wenn gewünscht
        # Einfacher für Demo: Store komplett neu erstellen oder spezifische Löschfunktion
        # Für diese Demo: Wir bauen auf dem bestehenden Store auf oder löschen ihn vorher.
        # Hier wäre store.clear() und erneutes Befüllen sauberer pro `size`.
        # Für Einfachheit lassen wir es kumulativ oder der User löscht manuell.
    
    cleanup_store(store)
    print(f"✅ Lokale Performance-Demo abgeschlossen!")


def run_advanced_demo_local(): # Umbenannt
    """Demonstriert erweiterte Funktionen wie Metadaten-Filterung direkt mit VectorStore."""
    print("\n🔧 Erweiterte Features Demo (Lokal mit VectorStore Klasse)")
    print("=" * 50)
    
    user_id = "advanced_user_v2"
    model_id = "advanced_test_v2"
    store = get_demo_store(user_id, model_id)
    dim = store.dimension or 128 # Kleinere Dimension für diese Demo

    vecs_mx = mx.random.normal((20, dim), dtype=mx.float32) # Mehr Vektoren für Filtertests
    meta = [
        {"id": f"doc_{i}", "category": "A" if i < 10 else "B", "priority": i % 3, "lang": "de" if i % 2 == 0 else "en"}
        for i in range(vecs_mx.shape[0])
    ]
    store.add_vectors(vecs_mx, meta)

    print(f"🔍 Teste Metadaten-Filterung...")
    query_mx_adv = vecs_mx[0]

    # Filter nach Kategorie 'A'
    _, _, cat_a_results_meta = store.query(query_mx_adv, k=10, metadata_filter={"category": "A"})
    print(f"   Ergebnisse für Kategorie 'A': {len(cat_a_results_meta)}")
    assert all(m["category"] == "A" for m in cat_a_results_meta)

    # Filter nach Priorität 1 UND Sprache 'en'
    _, _, prio_lang_results_meta = store.query(query_mx_adv, k=10, metadata_filter={"priority": 1, "lang": "en"})
    print(f"   Ergebnisse für Priorität 1 & Sprache 'en': {len(prio_lang_results_meta)}")
    assert all(m["priority"] == 1 and m["lang"] == "en" for m in prio_lang_results_meta)

    # Testfall: Filter, der keine Ergebnisse liefert
    _, _, no_results_meta = store.query(query_mx_adv, k=10, metadata_filter={"category": "C"})
    print(f"   Ergebnisse für nicht existente Kategorie 'C': {len(no_results_meta)}")
    assert len(no_results_meta) == 0
    
    # Die `stream_query`-Funktion existiert nicht in der neuen VectorStore-Klasse.
    # Man könnte sie als Generator implementieren, der `store.query` wiederholt aufruft,
    # oder `store.batch_query` nutzt und die Ergebnisse yieldet.
    # Für diese Demo wird sie vorerst entfernt oder als TODO markiert.
    # print(f"🌊 Streaming Query (nicht direkt in VectorStore, müsste extern implementiert werden)")

    cleanup_store(store)
    print(f"✅ Erweiterte Demo (V2) abgeschlossen!")


def main():
    """Führt alle Demo-Funktionen aus."""
    # Logging für die Demo konfigurieren (optional, falls nicht global schon geschehen)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        run_basic_demo()
        run_performance_demo_local()
        run_advanced_demo_local()
        
        print(f"\n🎉 Alle lokalen Demos (V2) erfolgreich abgeschlossen!")
        print(f"📖 Für API-Tests, starten Sie den Server und verwenden Sie die entsprechenden API-Demo-Skripte.")
        
    except Exception as e:
        logger.error(f"❌ Demo fehlgeschlagen: {e}", exc_info=True) # exc_info=True für Stacktrace

if __name__ == "__main__":
    main()