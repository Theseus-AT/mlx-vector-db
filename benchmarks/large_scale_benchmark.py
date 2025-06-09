#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

# large_scale_benchmark.py

import numpy as np
import time
import argparse
from pathlib import Path
import sys

# Füge das Projekt-Root zum Pfad hinzu
try:
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from service.optimized_vector_store import MLXVectorStore, MLXVectorStoreConfig

def run_large_benchmark(num_vectors: int, dimension: int, num_queries: int):
    """
    Führt einen Benchmark mit einer großen Anzahl von Vektoren durch.
    """
    print("="*60)
    print(f"🚀 Starte Large-Scale Benchmark")
    print(f"   Vektoren: {num_vectors:,}")
    print(f"   Dimension: {dimension}")
    print(f"   Abfragen: {num_queries:,}")
    print("="*60)

    # Wichtiger Hinweis
    required_ram_gb = (num_vectors * dimension * 4 * 2) / (1024**3) # *2 für Index-Overhead
    print(f"⚠️ ACHTUNG: Dieser Test benötigt ca. {required_ram_gb:.2f} GB RAM und wird einige Zeit dauern.")
    time.sleep(5)

    # --- 1. Setup ---
    print("\n[Phase 1/3] Erstelle Store und generiere Testdaten...")
    start_setup = time.time()
    
    config = MLXVectorStoreConfig(
        dimension=dimension,
        enable_hnsw=True # Unbedingt HNSW für diese Größe aktivieren
    )
    store = MLXVectorStore(store_path=f"./temp_large_store_{num_vectors}", config=config)
    
    # Generiere Daten in Batches, um den Speicher zu schonen
    vectors = np.random.rand(num_vectors, dimension).astype('float32')
    metadata = [{"id": f"vec_{i}"} for i in range(num_vectors)]
    query_vectors = np.random.rand(num_queries, dimension).astype('float32')
    
    print(f"   Setup abgeschlossen in {time.time() - start_setup:.2f}s")
    
    # --- 2. Ingestion & Index-Aufbau ---
    print("\n[Phase 2/3] Füge Vektoren hinzu und baue HNSW-Index...")
    start_ingest = time.time()
    
    store.add_vectors(vectors, metadata) # Dies löst jetzt den HNSW-Build aus
    
    ingest_time = time.time() - start_ingest
    vps = num_vectors / ingest_time
    
    print(f"   Ingestion & Index-Aufbau abgeschlossen in {ingest_time:.2f}s")
    print(f"   ➡️  Rate: {vps:,.0f} Vektoren/Sekunde")

    # --- 3. Abfragen ---
    print("\n[Phase 3/3] Führe Abfragen aus...")
    latencies = []
    for i in range(num_queries):
        start_query = time.time()
        store.query(query_vectors[i], k=10)
        latencies.append(time.time() - start_query)
        if (i + 1) % 100 == 0:
            print(f"   ... {i+1}/{num_queries} Abfragen abgeschlossen")

    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    qps = 1 / (avg_latency_ms / 1000)

    # --- Ergebnisse ---
    print("\n" + "="*60)
    print("📊 BENCHMARK-ERGEBNISSE")
    print("="*60)
    print(f"   Vektor-Anzahl: {num_vectors:,}")
    print(f"   Ingestion & Indexing Time: {ingest_time:.2f} s")
    print(f"   Ingestion Rate: {vps:,.0f} vecs/s")
    print(f"   Durchschnittliche Latenz: {avg_latency_ms:.4f} ms")
    print(f"   Queries pro Sekunde (QPS): {qps:,.2f}")
    
    # --- Cleanup ---
    print("\n🧹 Bereinige temporären Store...")
    import shutil
    shutil.rmtree(store.store_path, ignore_errors=True)
    print("   Cleanup abgeschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large-Scale Benchmark für MLX Vector DB.")
    parser.add_argument("--vectors", type=int, default=500000, help="Anzahl der zu indexierenden Vektoren.")
    parser.add_argument("--dim", type=int, default=384, help="Dimension der Vektoren.")
    parser.add_argument("--queries", type=int, default=1000, help="Anzahl der Test-Abfragen.")
    args = parser.parse_args()
    
    run_large_benchmark(args.vectors, args.dim, args.queries)