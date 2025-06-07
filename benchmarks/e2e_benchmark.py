# benchmarks/e2e_benchmark.py

import time
import argparse
import asyncio
import sys
from pathlib import Path

# === ANFANG: ROBUSTER IMPORT-FIX ===
# Finde das Root-Verzeichnis des Projekts (eine Ebene über dem aktuellen Skript)
# und füge es zum Python-Pfad hinzu. Das löst das Import-Problem.
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    # Fallback für interaktive Umgebungen, wo __file__ nicht definiert ist
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
# === ENDE: ROBUSTER IMPORT-FIX ===


# Jetzt können die Projekt-internen Imports sicher erfolgen
from integrations.mlx_lm_pipeline import MLXPipelineFactory, SUPPORTED_EMBEDDING_MODELS
from service.optimized_vector_store import create_optimized_vector_store

# Der Rest des Skripts bleibt unverändert...
SAMPLE_DOCUMENTS = [
    "MLX is Apple's machine learning framework for Apple silicon.",
    "A vector database indexes high-dimensional vectors for fast retrieval and similarity search.",
    "The unified memory architecture of Apple silicon allows the CPU and GPU to share memory efficiently.",
    "HNSW is an algorithm for approximate nearest neighbor search, used for high-performance queries.",
    "Retrieval-Augmented Generation (RAG) combines a retriever with a generator model to produce informed responses.",
] * 200

async def run_e2e_benchmark(model_id: str):
    # ... (Die Funktion selbst bleibt unverändert)
    print("="*60)
    print(f"🚀 Starte End-to-End RAG Performance Benchmark für Modell:")
    print(f"   '{model_id}'")
    print("="*60)

    if model_id not in SUPPORTED_EMBEDDING_MODELS:
        print(f"❌ Fehler: Modell '{model_id}' nicht in der Konfiguration gefunden.")
        print(f"   Verfügbare Modelle: {list(SUPPORTED_EMBEDDING_MODELS.keys())}")
        return

    model_config = SUPPORTED_EMBEDDING_MODELS[model_id]
    dimension = model_config.dimension

    print(f"[Phase 1/3] Initialisiere Embedding-Modell ({model_config.model_type}) und Vektor-Store...")
    store = create_optimized_vector_store(
        store_path=f"./temp_e2e_benchmark_store_{model_id.replace('/', '_')}",
        dimension=dimension,
        enable_hnsw=True
    )
    
    pipeline = MLXPipelineFactory.create_embedding_pipeline(
        model_id, store, pipeline_type="rag"
    )
    await pipeline.initialize()
    print("   ...Initialisierung abgeschlossen.")

    print("\n[Phase 2/3] Messe End-to-End Ingestion-Durchsatz...")
    start_ingest = time.time()
    await pipeline.index_documents([{"content": doc} for doc in SAMPLE_DOCUMENTS])
    ingest_time = time.time() - start_ingest
    docs_per_sec = len(SAMPLE_DOCUMENTS) / ingest_time
    print(f"   ...Ingestion von {len(SAMPLE_DOCUMENTS)} Dokumenten abgeschlossen.")
    print(f"   ➡️  End-to-End Ingestion Rate: {docs_per_sec:.2f} Dokumente/Sekunde")

    print("\n[Phase 3/3] Messe End-to-End Abfrage-Latenz...")
    query_text = "What is a vector database?"
    
    start_embed = time.time()
    query_vector = await pipeline.embedding_model.encode_text(query_text)
    embed_time_ms = (time.time() - start_embed) * 1000

    start_search = time.time()
    results = store.query(query_vector, k=5)
    search_time_ms = (time.time() - start_search) * 1000
    
    total_e2e_time_ms = embed_time_ms + search_time_ms

    print("\n" + "="*60)
    print("📊 END-TO-END BENCHMARK-ERGEBNISSE")
    print("="*60)
    print(f"   Modell: '{model_id}' ({dimension}D)")
    print(f"   Abfragetext: '{query_text}'")
    print("-" * 60)
    print(f"   Latenz für Embedding: {embed_time_ms:.2f} ms")
    print(f"   Latenz für Vektorsuche: {search_time_ms:.2f} ms")
    print("-" * 60)
    print(f"   Gesamte End-to-End Latenz: {total_e2e_time_ms:.2f} ms")
    
    import shutil
    shutil.rmtree(store.store_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Benchmark für die MLX Vector DB RAG-Pipeline.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="multilingual-e5-small",
        help=f"ID des zu testenden Modells."
    )
    args = parser.parse_args()
    
    asyncio.run(run_e2e_benchmark(args.model))