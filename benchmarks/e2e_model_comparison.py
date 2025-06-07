# benchmarks/e2e_model_comparison.py (UPDATED FOR MLX NATIVE)
"""
MLX Native Multi-Model Performance Comparison
Vergleicht alle verfügbaren MLX native embedding models auf Apple Silicon
"""

import time
import argparse
import asyncio
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import mlx.core as mx

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from integrations.mlx_lm_pipeline import (
    MLXNativePipelineFactory, 
    MLX_NATIVE_MODELS,
    MLXNativeBenchmark
)
from service.optimized_vector_store import create_optimized_vector_store

# Test data optimiert für MLX native models
SAMPLE_DOCUMENTS = [
    {
        "title": "MLX Framework Deep Dive",
        "content": "MLX ist Apples machine learning framework optimiert für Apple Silicon. Es nutzt unified memory architektur für nahtlose CPU-GPU integration und bietet automatische differentiation mit lazy evaluation.",
        "source": "mlx_framework.md"
    },
    {
        "title": "Apple Silicon Performance",
        "content": "M-series chips revolutionieren ML performance durch unified memory design. Neural Engine, GPU und CPU teilen sich memory pool für optimale bandwidth und minimale latency bei ML workloads.",
        "source": "apple_silicon.md"
    },
    {
        "title": "Vector Database Optimization", 
        "content": "Moderne vector databases nutzen HNSW algorithmen für logarithmic search complexity. Metal Performance Shaders beschleunigen similarity calculations auf Apple Silicon GPUs erheblich.",
        "source": "vector_optimization.md"
    },
    {
        "title": "Quantization Techniques",
        "content": "4-bit quantization reduziert model size um 75% bei minimal accuracy loss. MLX native implementations nutzen specialized kernels für efficient quantized inference auf Apple hardware.",
        "source": "quantization.md"
    },
    {
        "title": "Embedding Models Comparison",
        "content": "BGE, E5, MiniLM und MPNet models erreichen state-of-the-art performance auf verschiedenen NLP tasks. MLX implementations bieten native Apple Silicon acceleration.",
        "source": "embedding_models.md"
    }
] * 20  # 100 documents total für consistent testing

async def run_single_model_benchmark(model_id: str) -> dict:
    """Run comprehensive benchmark for single MLX native model"""
    
    print(f"\n--- Testing MLX Native Model: {model_id} ---")
    
    try:
        # Check if model exists
        if model_id not in MLX_NATIVE_MODELS:
            error_msg = f"Model '{model_id}' not in MLX native models."
            print(f"❌ {error_msg}")
            return {"Model": model_id, "Status": "Error", "Details": error_msg}

        model_config = MLX_NATIVE_MODELS[model_id]
        dimension = model_config.dimension

        print(f"   📱 Model config: {dimension}D, {model_config.memory_pool_mb}MB, Quantized: {model_config.quantized}")

        # Create vector store with unique path
        store_path = f"./temp_comparison_store_{model_id.replace('/', '_').replace('-', '_')}"
        store = create_optimized_vector_store(
            store_path=store_path,
            dimension=dimension,
            enable_hnsw=False,  # Consistent testing
            jit_compile=True
        )
        
        # Create MLX native pipeline
        pipeline = await MLXNativePipelineFactory.create_pipeline(
            model_id, store, pipeline_type="rag"
        )

        # === INITIALIZATION BENCHMARK ===
        print(f"   🔧 Testing initialization...")
        init_start = time.time()
        
        # Pipeline already initialized, measure warmup
        await pipeline.embedding_model._warmup_model()
        
        init_time = time.time() - init_start
        print(f"   ✅ Initialization: {init_time:.2f}s")

        # === DOCUMENT INDEXING BENCHMARK ===
        print(f"   📚 Testing document indexing...")
        indexing_start = time.time()
        
        try:
            index_result = await pipeline.index_documents(
                SAMPLE_DOCUMENTS,
                chunk_size=300,  # Optimized for comparison
                chunk_overlap=30
            )
            indexing_time = time.time() - indexing_start
            docs_per_sec = len(SAMPLE_DOCUMENTS) / indexing_time
            embedding_throughput = index_result['throughput_texts_per_sec']
            
            print(f"   ✅ Indexing: {indexing_time:.2f}s ({docs_per_sec:.1f} docs/sec)")
            
        except asyncio.TimeoutError:
            error_msg = "Indexing timeout (120s)"
            print(f"   ❌ {error_msg}")
            return {"Model": model_id, "Status": "Timeout", "Details": error_msg}

        # === QUERY PERFORMANCE BENCHMARK ===
        print(f"   🔍 Testing query performance...")
        
        test_queries = [
            "Wie funktioniert MLX framework?",
            "Was sind Apple Silicon vorteile?", 
            "Wie optimiert man vector databases?",
            "Was bringt quantization für performance?",
            "Welche embedding models sind am besten?"
        ]
        
        query_latencies = []
        retrieval_latencies = []
        
        for i, query_text in enumerate(test_queries):
            try:
                # Measure full query pipeline
                query_start = time.time()
                results = await pipeline.search_similar_texts(query_text, k=5)
                query_latency = time.time() - query_start
                query_latencies.append(query_latency)
                
                # Measure retrieval specifically
                retrieval_start = time.time()
                context = await pipeline.retrieve_context(query_text, k=3)
                retrieval_latency = time.time() - retrieval_start
                retrieval_latencies.append(retrieval_latency)
                
                if i == 0:  # Show first query results
                    print(f"      Query results: {len(results)} found")
                
            except Exception as e:
                print(f"   ⚠️ Query {i+1} failed: {e}")
                continue
        
        if not query_latencies or not retrieval_latencies:
            error_msg = "All queries failed"
            print(f"   ❌ {error_msg}")
            return {"Model": model_id, "Status": "Error", "Details": error_msg}
        
        avg_query_ms = np.mean(query_latencies) * 1000
        avg_retrieval_ms = np.mean(retrieval_latencies) * 1000
        qps = 1 / np.mean(query_latencies)
        
        print(f"   ✅ Avg Query: {avg_query_ms:.2f}ms")
        print(f"   ✅ Avg Retrieval: {avg_retrieval_ms:.2f}ms")
        print(f"   ✅ QPS: {qps:.1f}")

        # === MEMORY USAGE ===
        memory_info = pipeline.embedding_model.get_memory_info()
        peak_memory_mb = memory_info.get('peak_memory_mb', 0)
        
        # === PERFORMANCE STATS ===
        pipeline_stats = pipeline.get_pipeline_stats()
        embedding_stats = pipeline_stats['embedding_model_stats']
        
        # Cleanup
        try:
            store.clear()
            import shutil
            shutil.rmtree(store_path, ignore_errors=True)
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {e}")
        
        return {
            "Model": model_id,
            "Type": "mlx-native",
            "Dimension": dimension,
            "Quantized": model_config.quantized,
            "Memory_Pool_MB": model_config.memory_pool_mb,
            "Init_Time_s": f"{init_time:.2f}",
            "Indexing_Docs_per_sec": f"{docs_per_sec:.1f}",
            "Embedding_Throughput": f"{embedding_throughput:.1f}",
            "Avg_Query_Latency_ms": f"{avg_query_ms:.2f}",
            "Avg_Retrieval_ms": f"{avg_retrieval_ms:.2f}",
            "QPS": f"{qps:.1f}",
            "Peak_Memory_MB": f"{peak_memory_mb:.0f}",
            "Total_Inferences": embedding_stats.get('total_inferences', 0),
            "Model_Throughput": f"{embedding_stats.get('throughput_texts_per_sec', 0):.1f}",
            "Status": "OK",
            "Details": ""
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        print(f"❌ Test failed for {model_id}: {error_msg}")
        
        # Try cleanup anyway
        try:
            import shutil
            store_path = f"./temp_comparison_store_{model_id.replace('/', '_').replace('-', '_')}"
            shutil.rmtree(store_path, ignore_errors=True)
        except:
            pass
        
        return {"Model": model_id, "Status": "Error", "Details": error_msg}

def check_mlx_availability():
    """Check MLX and dependencies availability"""
    working_models = []
    
    try:
        # Check MLX core
        import mlx.core as mx
        print(f"✅ MLX available on {mx.default_device()}")
        
        # Check MLX embeddings
        from integrations.mlx_lm_pipeline import MLX_NATIVE_MODELS
        print(f"✅ MLX embeddings available with {len(MLX_NATIVE_MODELS)} models")
        
        working_models = list(MLX_NATIVE_MODELS.keys())
        
    except ImportError as e:
        print(f"❌ MLX dependencies not available: {e}")
        print("Install with: pip install mlx mlx-embeddings")
        return []
    
    return working_models

async def main(models_to_run=None):
    """Main comparison function with MLX native models"""
    
    print("=" * 80)
    print("🍎 MLX NATIVE Multi-Model Performance Comparison")
    print("⚡ Apple Silicon Optimized Vector Database")
    print("=" * 80)
    
    # Check dependencies and get working models
    if models_to_run is None:
        working_models = check_mlx_availability()
    else:
        working_models = models_to_run
    
    if not working_models:
        print("❌ No MLX models available. Please install mlx-embeddings.")
        return
    
    print(f"🧪 Testing {len(working_models)} MLX native models")
    print(f"📱 Device: {mx.default_device()}")
    print(f"📊 Test documents: {len(SAMPLE_DOCUMENTS)}")
    
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

    # === RESULTS ANALYSIS ===
    print("\n" + "="*80)
    print("📊 MLX NATIVE MODEL COMPARISON RESULTS")
    print("="*80)
    
    if results:
        df = pd.DataFrame(results)
        
        # Separate successful and failed results
        successful_results = df[df['Status'] == 'OK']
        failed_results = df[df['Status'] != 'OK']
        
        if not successful_results.empty:
            print("\n✅ SUCCESSFUL MODELS:")
            
            # Clean display DataFrame
            display_columns = [
                'Model', 'Dimension', 'Quantized', 'Memory_Pool_MB',
                'Indexing_Docs_per_sec', 'Avg_Query_Latency_ms', 'QPS', 
                'Peak_Memory_MB', 'Model_Throughput'
            ]
            
            display_df = successful_results[display_columns].copy()
            print(display_df.to_string(index=False, max_colwidth=25))
            
            # === PERFORMANCE RANKINGS ===
            if len(successful_results) > 1:
                print(f"\n🏆 PERFORMANCE RANKINGS:")
                
                # Convert string columns to numeric for ranking
                numeric_cols = ['Avg_Query_Latency_ms', 'QPS', 'Indexing_Docs_per_sec', 'Model_Throughput', 'Peak_Memory_MB']
                ranking_df = successful_results.copy()
                for col in numeric_cols:
                    ranking_df[col] = pd.to_numeric(ranking_df[col], errors='coerce')
                
                # Best performers
                best_latency = ranking_df.loc[ranking_df['Avg_Query_Latency_ms'].idxmin()]
                best_qps = ranking_df.loc[ranking_df['QPS'].idxmax()]
                best_indexing = ranking_df.loc[ranking_df['Indexing_Docs_per_sec'].idxmax()]
                best_throughput = ranking_df.loc[ranking_df['Model_Throughput'].idxmax()]
                lowest_memory = ranking_df.loc[ranking_df['Peak_Memory_MB'].idxmin()]
                
                print(f"   🚀 Best Latency