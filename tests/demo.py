#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

##!/usr/bin/env python3
"""
Demo script für MLX Vector Database
Korrigierte Version mit funktionierenden Imports und Operationen
FIXED: Directory creation issues
"""

import numpy as np
import mlx.core as mx
import time
from pathlib import Path
import shutil
import logging
import os

# Korrigierte Imports
from service.optimized_vector_store import MLXVectorStore, MLXVectorStoreConfig, ensure_directory_exists

logger = logging.getLogger("mlx_vector_db.demo")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Globale Konfiguration für die Demo-Stores
DEMO_BASE_PATH = Path("~/.mlx_vector_db_demo_stores").expanduser()

def ensure_demo_base_directory():
    """Stelle sicher, dass das Demo-Basisverzeichnis existiert - FIXED"""
    try:
        ensure_directory_exists(DEMO_BASE_PATH)
        
        # Test write permissions
        test_file = DEMO_BASE_PATH / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.info(f"Demo base directory ready: {DEMO_BASE_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to setup demo base directory: {e}")
        raise OSError(f"Cannot create or write to demo directory: {DEMO_BASE_PATH}")

def get_demo_store(user_id: str, model_id: str) -> MLXVectorStore:
    """Hilfsfunktion zum Erstellen/Abrufen einer VectorStore-Instanz für Demos - FIXED"""
    store_path = DEMO_BASE_PATH / f"user_{user_id}" / model_id
    
    # KRITISCHER FIX: Stelle sicher, dass das Verzeichnis existiert
    ensure_directory_exists(store_path)
    
    config = MLXVectorStoreConfig(
        dimension=384,
        metric="cosine",
        enable_hnsw=False,  # Deaktiviert für Demo-Stabilität
        jit_compile=True
    )
    
    # Zusätzliche Verifikation
    if not store_path.exists():
        raise OSError(f"Failed to create store directory: {store_path}")
    
    return MLXVectorStore(str(store_path), config=config)

def cleanup_store(store: MLXVectorStore):
    """Bereinigt einen Store und löscht sein Verzeichnis - IMPROVED"""
    store_path_to_delete = store.store_path
    try:
        # Clear store data first
        store.clear()
        
        # Remove directory if it exists
        if store_path_to_delete.exists() and store_path_to_delete.is_dir():
            shutil.rmtree(store_path_to_delete, ignore_errors=True)
            
        # Verify deletion
        if store_path_to_delete.exists():
            logger.warning(f"Directory still exists after cleanup: {store_path_to_delete}")
        else:
            print(f"🧹 Store-Verzeichnis {store_path_to_delete} bereinigt.")
            
    except Exception as e:
        logger.error(f"Fehler beim Bereinigen von {store_path_to_delete}: {e}")
        print(f"⚠️ Cleanup-Warnung: {e}")

def run_basic_demo():
    """Führt grundlegende Operationen mit der VectorStore-Klasse vor."""
    print("🧠 MLX Vector Database Demo")
    print("=" * 50)
    
    user_id = "demo_user"
    model_id = "test_model"
    store = get_demo_store(user_id, model_id)

    print(f"📁 Store wird initialisiert für {user_id}/{model_id} unter {store.store_path}")

    # Vektoren hinzufügen
    print(f"➕ Füge Beispiel-Vektoren hinzu...")
    vectors_np = np.random.rand(10, store.config.dimension).astype(np.float32)
    metadata = [{"id": f"chunk_{i}", "source": "demo", "content": f"Sample content {i}"} for i in range(vectors_np.shape[0])]
    
    start_time = time.time()
    result = store.add_vectors(vectors_np, metadata)
    add_time = time.time() - start_time
    print(f"   {result['vectors_added']} Vektoren hinzugefügt in {add_time:.3f}s")

    # Vektoren abfragen
    print(f"🔍 Frage Vektoren ab...")
    query_vector = vectors_np[0]
    k_val = 3
    
    start_time = time.time()
    indices, distances, result_metadata = store.query(query_vector, k=k_val)
    query_time = time.time() - start_time
    
    print(f"   Query abgeschlossen in {query_time:.3f}s")
    print(f"   {len(indices)} Ergebnisse gefunden für k={k_val}:")
    for i in range(len(indices)):
        print(f"     {i+1}. Index: {indices[i]}, Distanz: {distances[i]:.4f}, Meta-ID: {result_metadata[i].get('id')}")

    # Batch-Query-Demo
    print(f"🧠 Teste Batch-Query...")
    batch_query_vectors = vectors_np[:3]
    k_batch = 2
    
    start_time = time.time()
    batch_results = store.batch_query(batch_query_vectors, k=k_batch)
    batch_time = time.time() - start_time
    print(f"   Batch-Query ({batch_query_vectors.shape[0]} Queries) abgeschlossen in {batch_time:.3f}s")
    for i, (indices, distances, metadata) in enumerate(batch_results):
        print(f"     Ergebnisse für Query {i+1}: {len(indices)} Treffer")

    # Store-Statistiken
    print(f"📊 Store-Statistiken:")
    stats = store.get_stats()
    print(f"   Vektoren: {stats.get('vector_count', 0)}")
    print(f"   Dimension: {stats.get('dimension', 'N/A')}")
    print(f"   Metrik: {stats.get('metric', 'N/A')}")
    print(f"   Speicherverbrauch: {stats.get('memory_usage_mb', 0):.2f} MB")

    # Cleanup
    cleanup_store(store)
    print(f"✅ Basis-Demo erfolgreich abgeschlossen!")

def run_performance_demo():
    """Performance-Demo mit VectorStore"""
    print("\n🚀 Performance Demo")
    print("=" * 50)
    
    user_id = "perf_user"
    model_id = "performance_test"
    store = get_demo_store(user_id, model_id)

    sizes = [100, 500, 1000]
    dim = store.config.dimension

    for size in sizes:
        print(f"\n📈 Teste mit {size} Vektoren (Dim: {dim})...")
        
        vectors_np = np.random.normal(size=(size, dim)).astype(np.float32)
        metadata = [{"id": f"vec_{i}", "batch": "perf_test"} for i in range(size)]
        
        # Benchmark Addition
        start_time = time.perf_counter()
        add_result = store.add_vectors(vectors_np, metadata)
        add_time = time.perf_counter() - start_time
        
        # Benchmark Queries
        query_vector = np.random.normal(size=(dim,)).astype(np.float32)
        query_times = []
        
        # Führe mehrere Queries aus
        num_queries = min(20, size // 10) if size > 10 else 1
        for _ in range(num_queries):
            q_start_time = time.perf_counter()
            indices, distances, metadata_results = store.query(query_vector, k=5)
            query_times.append(time.perf_counter() - q_start_time)
        
        avg_query_time_ms = (sum(query_times) / len(query_times)) * 1000 if query_times else 0
        add_rate = size / add_time if add_time > 0 else 0
        
        print(f"   Hinzufügen: {add_time:.3f}s ({add_rate:.1f} Vektoren/s)")
        print(f"   Avg. Query-Zeit: {avg_query_time_ms:.3f}ms")
        print(f"   QPS: {1000/avg_query_time_ms:.1f}" if avg_query_time_ms > 0 else "   QPS: N/A")
    
    cleanup_store(store)
    print(f"✅ Performance-Demo abgeschlossen!")

def run_advanced_demo():
    """Erweiterte Features Demo"""
    print("\n🔧 Erweiterte Features Demo")
    print("=" * 50)
    
    user_id = "advanced_user"
    model_id = "advanced_test"
    
    # Verwende kleinere Dimension für Demo
    config = MLXVectorStoreConfig(
        dimension=128,
        metric="cosine",
        enable_hnsw=False,
        jit_compile=True
    )
    store_path = DEMO_BASE_PATH / f"user_{user_id}" / model_id
    ensure_directory_exists(store_path)
    store = MLXVectorStore(str(store_path), config=config)
    
    dim = store.config.dimension

    # Test-Daten mit verschiedenen Metadaten erstellen
    vectors_np = np.random.normal(size=(20, dim)).astype(np.float32)
    metadata = [
        {"id": f"doc_{i}", "category": "A" if i < 10 else "B", "priority": i % 3, "lang": "de" if i % 2 == 0 else "en"}
        for i in range(vectors_np.shape[0])
    ]
    store.add_vectors(vectors_np, metadata)

    print(f"🔍 Teste Metadaten-Filterung...")
    query_vector = vectors_np[0]

    # Filter nach Kategorie 'A'
    indices, distances, cat_a_results = store.query(query_vector, k=10, filter_metadata={"category": "A"})
    print(f"   Ergebnisse für Kategorie 'A': {len(cat_a_results)}")
    if cat_a_results:
        assert all(m["category"] == "A" for m in cat_a_results), "Filter funktioniert nicht korrekt"

    # Filter nach Priorität 1 UND Sprache 'en'
    indices, distances, prio_lang_results = store.query(query_vector, k=10, filter_metadata={"priority": 1, "lang": "en"})
    print(f"   Ergebnisse für Priorität 1 & Sprache 'en': {len(prio_lang_results)}")
    if prio_lang_results:
        assert all(m["priority"] == 1 and m["lang"] == "en" for m in prio_lang_results), "Filter funktioniert nicht korrekt"

    # Testfall: Filter, der keine Ergebnisse liefert
    indices, distances, no_results = store.query(query_vector, k=10, filter_metadata={"category": "C"})
    print(f"   Ergebnisse für nicht existente Kategorie 'C': {len(no_results)}")
    assert len(no_results) == 0, "Filter sollte keine Ergebnisse liefern"

    # Store-Optimierung testen
    print(f"⚙️ Teste Store-Optimierung...")
    start_time = time.time()
    store.optimize()
    optimize_time = time.time() - start_time
    print(f"   Optimierung abgeschlossen in {optimize_time:.3f}s")

    # Health Check
    print(f"🏥 Teste Health Check...")
    health = store.health_check()
    print(f"   Gesundheitsstatus: {'✅ Gesund' if health['healthy'] else '❌ Probleme erkannt'}")
    if not health['healthy']:
        print(f"   Probleme: {health.get('issues', [])}")

    cleanup_store(store)
    print(f"✅ Erweiterte Demo abgeschlossen!")

def run_mlx_integration_demo():
    """Demo der MLX-Integration"""
    print("\n🍎 MLX Integration Demo")
    print("=" * 50)
    
    # Test MLX-spezifische Funktionen
    print("🧪 Teste MLX-Arrays direkt...")
    
    user_id = "mlx_user"
    model_id = "mlx_test"
    store = get_demo_store(user_id, model_id)
    
    # Erstelle MLX-Arrays direkt
    mlx_vectors = mx.random.normal((50, 384), dtype=mx.float32)
    metadata = [{"id": f"mlx_vec_{i}", "type": "mlx_native"} for i in range(50)]
    
    print("   Teste Addition von MLX-Arrays...")
    start_time = time.time()
    result = store.add_vectors(mlx_vectors, metadata)
    add_time = time.time() - start_time
    print(f"   {result['vectors_added']} MLX-Vektoren hinzugefügt in {add_time:.3f}s")
    
    # Query mit MLX-Array
    print("   Teste Query mit MLX-Array...")
    mlx_query = mx.random.normal((384,), dtype=mx.float32)
    
    start_time = time.time()
    indices, distances, results = store.query(mlx_query, k=5)
    query_time = time.time() - start_time
    print(f"   MLX-Query abgeschlossen in {query_time:.3f}s")
    print(f"   Gefunden: {len(indices)} Ergebnisse")
    
    # Teste kompilierte Funktionen
    print("⚡ Teste kompilierte MLX-Funktionen...")
    store._warmup_kernels()
    print("   ✅ Kernel-Warmup abgeschlossen")
    
    # Performance-Vergleich: NumPy vs MLX
    print("📊 Performance-Vergleich: NumPy vs MLX...")
    
    numpy_vectors = np.random.rand(100, 384).astype(np.float32)
    mlx_vectors = mx.array(numpy_vectors)
    query_np = numpy_vectors[0]
    query_mx = mx.array(query_np)
    
    # NumPy Query
    start_time = time.time()
    store.query(query_np, k=10)
    numpy_time = time.time() - start_time
    
    # MLX Query
    start_time = time.time()
    store.query(query_mx, k=10)
    mlx_time = time.time() - start_time
    
    speedup = numpy_time / mlx_time if mlx_time > 0 else 1.0
    print(f"   NumPy Query: {numpy_time*1000:.3f}ms")
    print(f"   MLX Query: {mlx_time*1000:.3f}ms")
    print(f"   MLX Speedup: {speedup:.2f}x")
    
    cleanup_store(store)
    print(f"✅ MLX Integration Demo abgeschlossen!")

def main():
    """Führt alle Demo-Funktionen aus - FIXED VERSION"""
    try:
        print("🍎 MLX Vector Database - Vollständige Demo")
        print("⚡ Optimiert für Apple Silicon")
        print("=" * 60)
        
        # NEUER FIX: Stelle sicher, dass Basisverzeichnis existiert
        ensure_demo_base_directory()
        
        # MLX System Test
        print("🔧 Teste MLX System...")
        test_array = mx.random.normal((10, 10))
        mx.eval(test_array)
        print(f"   ✅ MLX funktioniert: Device = {mx.default_device()}")
        
        # Führe alle Demos aus
        run_basic_demo()
        run_performance_demo()
        run_advanced_demo()
        run_mlx_integration_demo()
        
        print(f"\n🎉 Alle Demos erfolgreich abgeschlossen!")
        print(f"📖 Für API-Tests, starten Sie den Server mit:")
        print(f"   python main.py")
        print(f"📝 Dann testen Sie mit:")
        print(f"   python working_test_fixed.py")
        
    except Exception as e:
        logger.error(f"❌ Demo fehlgeschlagen: {e}", exc_info=True)
        print(f"\n💡 Troubleshooting-Tipps:")
        print(f"   1. Stellen Sie sicher, dass MLX installiert ist: pip install mlx")
        print(f"   2. Verwenden Sie Apple Silicon (M1/M2/M3)")
        print(f"   3. Prüfen Sie die Python-Version (>=3.9)")
        print(f"   4. Prüfen Sie Schreibrechte für: {DEMO_BASE_PATH}")

if __name__ == "__main__":
    main()