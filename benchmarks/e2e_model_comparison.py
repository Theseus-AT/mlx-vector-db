# benchmarks/e2e_model_comparison_fixed.py
"""
KORRIGIERTE End-to-End Model Comparison
Funktioniert mit verfügbaren Modellen und besserer Fehlerbehandlung
"""

import time
import argparse
import asyncio
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fügt das Projekt-Root zum Python-Pfad hinzu
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from integrations.mlx_lm_pipeline import MLXPipelineFactory, SUPPORTED_EMBEDDING_MODELS
from service.optimized_vector_store import create_optimized_vector_store

# KORRIGIERTE und FUNKTIONIERENDE Modell-Liste
RELIABLE_MODELS_TO_TEST = [
    "multilingual-e5-small",     # 384D - Sentence Transformers (stabil)
    "all-MiniLM-L6-v2",         # 384D - Sentence Transformers (stabil)  
    "all-mpnet-base-v2",        # 768D - Sentence Transformers (stabil)
    "multilingual-e5-base",     # 768D - Sentence Transformers (stabil)
]

# Fallback auf Mock wenn Sentence Transformers nicht verfügbar
MOCK_MODELS_FALLBACK = [
    "mock-384",
    "mock-768",
]

SAMPLE_DOCUMENTS = [
    "MLX is Apple's machine learning framework designed for Apple silicon.",
    "A vector database indexes high-dimensional vectors for fast retrieval and similarity search.",
    "The unified memory architecture of Apple silicon allows the CPU and GPU to share memory efficiently.",
    "HNSW is an algorithm for approximate nearest neighbor search, used for high-performance queries.",
    "Retrieval-Augmented Generation (RAG) combines a retriever with a generator model to produce informed responses.",
] * 100  # Reduziert für Stabilität

async def run_single_model_benchmark(model_id: str):
    """Run benchmark for single model with better error handling"""
    print(f"\n--- Testing Model: {model_id} ---")

    try:
        # Check if model is in supported models
        if model_id not in SUPPORTED_EMBEDDING_MODELS:
            error_msg = f"Model '{model_id}' not in supported models configuration."
            print(f"❌ {error_msg}")
            return {"Model": model_id, "Status": "Error", "Details": error_msg}

        model_config = SUPPORTED_EMBEDDING_MODELS[model_id]
        dimension = model_config.dimension

        # Create vector store with unique path
        store_path = f"./temp_e2e_store_{model_id.replace('/', '_').replace('-', '_')}"
        store = create_optimized_vector_store(
            store_path=store_path,
            dimension=dimension,
            enable_hnsw=False  # Disable HNSW for stability
        )
        
        # Create pipeline
        pipeline = MLXPipelineFactory.create_embedding_pipeline(
            model_id, store, pipeline_type="rag"
        )

        # Initialize with timeout
        print(f"   Initializing {model_config.model_type} model...")
        init_start = time.time()
        
        try:
            await asyncio.wait_for(pipeline.initialize(), timeout=60.0)  # 60s timeout
            init_time = time.time() - init_start
            print(f"   ✅ Initialization: {init_time:.2f}s")
        except asyncio.TimeoutError:
            error_msg = "Initialization timeout (60s)"
            print(f"   ❌ {error_msg}")
            return {"Model": model_id, "Status": "Timeout", "Details": error_msg}

        # Test document indexing
        print(f"   Indexing {len(SAMPLE_DOCUMENTS)} documents...")
        ingest_start = time.time()
        
        try:
            index_result = await asyncio.wait_for(
                pipeline.index_documents([{"content": doc} for doc in SAMPLE_DOCUMENTS]), 
                timeout=120.0  # 2 minute timeout
            )
            ingest_time = time.time() - ingest_start
            docs_per_sec = len(SAMPLE_DOCUMENTS) / ingest_time
            print(f"   ✅ Ingestion Rate: {docs_per_sec:.2f} docs/s")
        except asyncio.TimeoutError:
            error_msg = "Indexing timeout (120s)"
            print(f"   ❌ {error_msg}")
            return {"Model": model_id, "Status": "Timeout", "Details": error_msg}

        # Test queries
        print(f"   Testing queries...")
        query_text = "What is a vector database?"
        
        embed_latencies = []
        search_latencies = []
        
        # Run multiple query tests
        for i in range(min(5, len(SAMPLE_DOCUMENTS))):  # Reduced for stability
            try:
                # Test embedding latency
                start_embed = time.time()
                query_vector = await pipeline.embedding_model.encode_text(query_text)
                embed_latencies.append((time.time() - start_embed) * 1000)

                # Test search latency
                start_search = time.time()
                query_np = np.array(query_vector.tolist())
                results = store.query(query_np, k=5)
                search_latencies.append((time.time() - start_search) * 1000)
                
            except Exception as e:
                print(f"   ⚠️ Query {i+1} failed: {e}")
                continue
        
        if not embed_latencies or not search_latencies:
            error_msg = "All queries failed"
            print(f"   ❌ {error_msg}")
            return {"Model": model_id, "Status": "Error", "Details": error_msg}
        
        avg_embed_ms = np.mean(embed_latencies)
        avg_search_ms = np.mean(search_latencies)
        total_latency = avg_embed_ms + avg_search_ms
        
        print(f"   ✅ Avg Embedding: {avg_embed_ms:.2f}ms")
        print(f"   ✅ Avg Search: {avg_search_ms:.2f}ms")
        print(f"   ✅ Total E2E: {total_latency:.2f}ms")
        
        # Cleanup
        try:
            store.clear()
            import shutil
            shutil.rmtree(store_path, ignore_errors=True)
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")
        
        return {
            "Model": model_id,
            "Type": model_config.model_type,
            "Dimension": dimension,
            "Ingestion (docs/s)": f"{docs_per_sec:.2f}",
            "Embedding Latency (ms)": f"{avg_embed_ms:.2f}",
            "Search Latency (ms)": f"{avg_search_ms:.2f}",
            "Total E2E Latency (ms)": f"{total_latency:.2f}",
            "Status": "OK",
            "Details": ""
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"❌ Test failed for {model_id}: {error_msg}")
        
        # Try cleanup anyway
        try:
            import shutil
            store_path = f"./temp_e2e_store_{model_id.replace('/', '_').replace('-', '_')}"
            shutil.rmtree(store_path, ignore_errors=True)
        except:
            pass
        
        return {"Model": model_id, "Status": "Error", "Details": error_msg}

def check_dependencies():
    """Check available dependencies and return working models"""
    working_models = []
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        print("✅ sentence-transformers available")
        working_models.extend(RELIABLE_MODELS_TO_TEST)
    except ImportError:
        print("❌ sentence-transformers not available")
    
    # Check MLX-LM
    try:
        import mlx_lm
        print("✅ mlx-lm available (but embedding models may not work)")
        # Don't add MLX-LM models for embeddings
    except ImportError:
        print("❌ mlx-lm not available")
    
    # Fallback to mock models if nothing works
    if not working_models:
        print("⚠️ No embedding libraries available, using mock models")
        working_models = MOCK_MODELS_FALLBACK
    
    return working_models

async def main(models_to_run=None):
    """Main benchmark function with dependency checking"""
    print("=" * 70)
    print("🚀 KORRIGIERTE End-to-End Multi-Model Performance Benchmark")
    print("=" * 70)
    
    # Check dependencies and get working models
    if models_to_run is None:
        working_models = check_dependencies()
    else:
        working_models = models_to_run
    
    print(f"🧪 Testing {len(working_models)} model(s): {', '.join(working_models)}")
    
    # Run benchmarks
    results = []
    for model_id in working_models:
        try:
            result = await run_single_model_benchmark(model_id)
            results.append(result)
        except Exception as e:
            print(f"💥 Critical error testing {model_id}: {e}")
            results.append({
                "Model": model_id, 
                "Status": "Critical Error", 
                "Details": str(e)
            })

    # Display results
    print("\n" + "="*70)
    print("📊 FINALE BENCHMARK-ERGEBNISSE")
    print("="*70)
    
    if results:
        df = pd.DataFrame(results)
        
        # Show successful results separately
        successful_results = df[df['Status'] == 'OK']
        failed_results = df[df['Status'] != 'OK']
        
        if not successful_results.empty:
            print("\n✅ ERFOLGREICHE MODELLE:")
            # Remove Details column for clean display
            display_df = successful_results.drop(columns=['Details'], errors='ignore')
            print(display_df.to_string(index=False))
            
            # Show performance summary
            if len(successful_results) > 1:
                print(f"\n🏆 PERFORMANCE RANKING:")
                # Sort by total latency (lower is better)
                if 'Total E2E Latency (ms)' in successful_results.columns:
                    ranked = successful_results.sort_values('Total E2E Latency (ms)')
                    for i, (_, row) in enumerate(ranked.iterrows(), 1):
                        latency = row['Total E2E Latency (ms)']
                        print(f"   {i}. {row['Model']}: {latency}ms")
        
        if not failed_results.empty:
            print(f"\n❌ FEHLGESCHLAGENE MODELLE ({len(failed_results)}):")
            for _, row in failed_results.iterrows():
                print(f"   • {row['Model']}: {row['Status']} - {row['Details'][:50]}...")
    
    # Summary
    successful_count = len([r for r in results if r['Status'] == 'OK'])
    print(f"\n📈 ZUSAMMENFASSUNG:")
    print(f"   Getestete Modelle: {len(results)}")
    print(f"   Erfolgreich: {successful_count}")
    print(f"   Fehlgeschlagen: {len(results) - successful_count}")
    
    if successful_count > 0:
        print(f"\n🎉 BENCHMARK ERFOLGREICH!")
        print(f"💡 Empfehlung: Nutzen Sie die Modelle mit der niedrigsten Latenz für Ihre Anwendung")
    else:
        print(f"\n⚠️ ALLE TESTS FEHLGESCHLAGEN")
        print(f"💡 Installieren Sie sentence-transformers: pip install sentence-transformers")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korrigierte End-to-End Benchmark für Embedding-Modelle.")
    parser.add_argument(
        "--models", 
        nargs='+',
        default=None,
        help=f"Liste der zu testenden Modell-IDs. Standard: automatische Erkennung"
    )
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args.models))
    except KeyboardInterrupt:
        print("\n🛑 Benchmark durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n💥 Benchmark fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()