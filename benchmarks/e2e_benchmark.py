#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

# benchmarks/e2e_benchmark.py (UPDATED FOR MLX NATIVE)

import time
import argparse
import asyncio
import sys
from pathlib import Path
import mlx.core as mx

# === MLX NATIVE IMPORT FIX ===
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    project_root = Path.cwd()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# MLX Native Pipeline Imports
from integrations.mlx_lm_pipeline import (
    MLXNativePipelineFactory, 
    MLX_NATIVE_MODELS,
    DEFAULT_MLX_MODEL
)
from service.optimized_vector_store import create_optimized_vector_store

# Test Documents für MLX optimiert
SAMPLE_DOCUMENTS = [
    {
        "title": "MLX Framework Performance",
        "content": "MLX ist Apples machine learning framework für Apple silicon. Es nutzt unified memory architektur für optimale performance bei ML workloads.",
        "source": "mlx_performance.md"
    },
    {
        "title": "Apple Silicon Architecture", 
        "content": "Apple Silicon M-series chips bieten unified memory design das CPU, GPU und Neural Engine nahtlos integriert für ML acceleration.",
        "source": "apple_silicon.md"
    },
    {
        "title": "Vector Database Optimization",
        "content": "Vector databases nutzen HNSW algorithmen für fast similarity search. Auf Apple Silicon können diese durch Metal optimiert werden.",
        "source": "vector_optimization.md"
    },
    {
        "title": "Quantized Models Performance",
        "content": "4-bit quantization reduziert memory usage um 75% bei minimaler accuracy loss. MLX native models sind für Apple Silicon optimiert.",
        "source": "quantization.md"
    },
    {
        "title": "Embedding Models Comparison",
        "content": "BGE, E5 und MiniLM models erreichen state-of-the-art performance. MLX implementations nutzen Metal Performance Shaders optimal.",
        "source": "embedding_comparison.md"
    }
] * 40  # 200 documents total

async def run_mlx_native_e2e_benchmark(model_id: str = None):
    """
    MLX Native End-to-End RAG Performance Benchmark
    Optimiert für Apple Silicon Performance Testing
    """
    print("=" * 70)
    print(f"🍎 MLX NATIVE End-to-End RAG Performance Benchmark")
    print(f"⚡ Apple Silicon Optimized Vector Database")
    print("=" * 70)

    # Model Selection
    if model_id is None or model_id not in MLX_NATIVE_MODELS:
        model_id = MLXNativePipelineFactory.get_recommended_model(
            use_case="general",
            performance_priority="balanced",
            memory_budget_mb=1024
        )
        print(f"🎯 Auto-selected model: {model_id}")
    else:
        print(f"📱 Using specified model: {model_id}")

    model_config = MLX_NATIVE_MODELS[model_id]
    dimension = model_config.dimension

    print(f"📊 Model Specifications:")
    print(f"   Dimension: {dimension}")
    print(f"   Quantized: {model_config.quantized}")
    print(f"   Memory Pool: {model_config.memory_pool_mb}MB")
    print(f"   Device: {mx.default_device()}")

    # === PHASE 1: SETUP & INITIALIZATION ===
    print(f"\n[Phase 1/4] MLX Native Setup & Initialization...")
    setup_start = time.time()
    
    # Create optimized vector store
    store_path = f"./temp_e2e_benchmark_mlx_{model_id.replace('/', '_').replace('-', '_')}"
    store = create_optimized_vector_store(
        store_path=store_path,
        dimension=dimension,
        enable_hnsw=False,  # For consistent benchmarking
        jit_compile=True
    )
    
    # Create MLX native RAG pipeline
    pipeline = await MLXNativePipelineFactory.create_pipeline(
        model_id, store, pipeline_type="rag"
    )
    
    setup_time = time.time() - setup_start
    print(f"   ✅ Setup completed in {setup_time:.2f}s")

    # === PHASE 2: DOCUMENT INDEXING PERFORMANCE ===
    print(f"\n[Phase 2/4] MLX Native Document Indexing Performance...")
    print(f"   Documents to index: {len(SAMPLE_DOCUMENTS)}")
    
    indexing_start = time.time()
    index_result = await pipeline.index_documents(
        SAMPLE_DOCUMENTS,
        chunk_size=400,  # Optimal für MLX models
        chunk_overlap=50
    )
    indexing_time = time.time() - indexing_start
    
    docs_per_sec = len(SAMPLE_DOCUMENTS) / indexing_time
    chunks_per_sec = index_result['chunks_created'] / indexing_time
    
    print(f"   ✅ Indexing completed in {indexing_time:.2f}s")
    print(f"   📄 Created {index_result['chunks_created']} chunks")
    print(f"   📈 Document throughput: {docs_per_sec:.2f} docs/sec")
    print(f"   🚀 Chunk throughput: {chunks_per_sec:.2f} chunks/sec")
    print(f"   ⚡ Embedding throughput: {index_result['throughput_texts_per_sec']:.2f} texts/sec")

    # === PHASE 3: QUERY PERFORMANCE TESTING ===
    print(f"\n[Phase 3/4] MLX Native Query Performance...")
    
    test_queries = [
        "Wie funktioniert MLX framework auf Apple Silicon?",
        "Was sind die Vorteile von unified memory architektur?",
        "Wie optimiert man vector databases für performance?",
        "Welche benefits bietet quantization für embedding models?",
        "Wie vergleichen sich verschiedene embedding models?"
    ]
    
    print(f"   Test queries: {len(test_queries)}")
    
    query_times = []
    retrieval_times = []
    
    for i, query_text in enumerate(test_queries):
        # Measure embedding time
        embed_start = time.time()
        query_embedding = await pipeline.embedding_model.encode_text(query_text)
        embed_time = time.time() - embed_start
        
        # Measure retrieval time  
        retrieval_start = time.time()
        context_chunks = await pipeline.retrieve_context(query_text, k=5)
        retrieval_time = time.time() - retrieval_start
        
        total_query_time = embed_time + retrieval_time
        
        query_times.append(total_query_time)
        retrieval_times.append(retrieval_time)
        
        print(f"   Query {i+1}: {total_query_time*1000:.2f}ms (embed: {embed_time*1000:.2f}ms, search: {retrieval_time*1000:.2f}ms)")
        print(f"      Found {len(context_chunks)} relevant chunks")

    # === PHASE 4: PERFORMANCE ANALYSIS ===
    print(f"\n[Phase 4/4] MLX Native Performance Analysis...")
    
    avg_query_time = sum(query_times) / len(query_times)
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    avg_embed_time = avg_query_time - avg_retrieval_time
    
    qps = 1 / avg_query_time if avg_query_time > 0 else 0
    
    # Get detailed pipeline stats
    pipeline_stats = pipeline.get_rag_stats()
    embedding_stats = pipeline_stats['embedding_model_stats']
    memory_info = pipeline_stats['memory_info']
    
    print(f"\n" + "=" * 70)
    print(f"📊 MLX NATIVE E2E BENCHMARK RESULTS")
    print(f"=" * 70)
    print(f"🧠 Model: {model_id}")
    print(f"📱 Device: {mx.default_device()}")
    print(f"📐 Dimension: {dimension}")
    print(f"⚡ Quantized: {model_config.quantized}")
    print("-" * 70)
    
    print(f"📚 INDEXING PERFORMANCE:")
    print(f"   Documents indexed: {len(SAMPLE_DOCUMENTS)}")
    print(f"   Chunks created: {index_result['chunks_created']}")
    print(f"   Indexing time: {indexing_time:.2f}s")
    print(f"   Document throughput: {docs_per_sec:.2f} docs/sec")
    print(f"   Embedding throughput: {index_result['throughput_texts_per_sec']:.2f} texts/sec")
    
    print(f"\n🔍 QUERY PERFORMANCE:")
    print(f"   Avg total query time: {avg_query_time*1000:.2f}ms")
    print(f"   Avg embedding time: {avg_embed_time*1000:.2f}ms")
    print(f"   Avg retrieval time: {avg_retrieval_time*1000:.2f}ms")
    print(f"   Queries per second: {qps:.2f} QPS")
    
    print(f"\n💾 MEMORY PERFORMANCE:")
    if 'error' not in memory_info:
        print(f"   Peak memory usage: {memory_info.get('peak_memory_mb', 0):.0f}MB")
        print(f"   Active memory: {memory_info.get('active_memory_mb', 0):.0f}MB")
        print(f"   Memory limit: {memory_info.get('metal_memory_limit_mb', 0):.0f}MB")
    
    if not embedding_stats.get('no_data', False):
        print(f"\n🧠 MODEL PERFORMANCE:")
        print(f"   Total inferences: {embedding_stats['total_inferences']}")
        print(f"   Avg inference time: {embedding_stats['avg_inference_time_ms']:.2f}ms")
        print(f"   Model throughput: {embedding_stats['throughput_texts_per_sec']:.2f} texts/sec")
    
    # === PERFORMANCE TARGETS ===
    print(f"\n🎯 PERFORMANCE TARGETS:")
    targets = {
        "Document Indexing": (docs_per_sec >= 50, f"{docs_per_sec:.1f} docs/sec (target: ≥50)"),
        "Query Latency": (avg_query_time <= 0.1, f"{avg_query_time*1000:.1f}ms (target: ≤100ms)"), 
        "QPS Performance": (qps >= 10, f"{qps:.1f} QPS (target: ≥10)"),
        "Memory Efficiency": (memory_info.get('peak_memory_mb', 9999) <= 2048, f"{memory_info.get('peak_memory_mb', 0):.0f}MB (target: ≤2GB)")
    }
    
    targets_met = 0
    for target_name, (met, description) in targets.items():
        status = "✅" if met else "❌"
        print(f"   {status} {target_name}: {description}")
        if met:
            targets_met += 1
    
    overall_score = (targets_met / len(targets)) * 100
    
    if overall_score >= 75:
        assessment = "🎉 EXCELLENT! Production ready for Apple Silicon"
    elif overall_score >= 50:
        assessment = "✅ GOOD performance, suitable for most use cases"
    else:
        assessment = "⚠️ Performance below targets, consider optimization"
    
    print(f"\n🏆 OVERALL ASSESSMENT: {assessment}")
    print(f"📈 Performance Score: {overall_score:.0f}/100")
    print(f"🍎 Apple Silicon Optimization: {'🟢 OPTIMAL' if model_config.quantized else '🟡 STANDARD'}")
    
    # === CLEANUP ===
    print(f"\n🧹 Cleanup...")
    store.clear()
    import shutil
    try:
        shutil.rmtree(store_path, ignore_errors=True)
        print(f"   ✅ Temporary files cleaned up")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")
    
    print(f"\n🎊 MLX Native E2E Benchmark completed!")
    
    # Return results for programmatic use
    return {
        "model_id": model_id,
        "model_config": {
            "dimension": dimension,
            "quantized": model_config.quantized,
            "memory_pool_mb": model_config.memory_pool_mb
        },
        "indexing_performance": {
            "documents_indexed": len(SAMPLE_DOCUMENTS),
            "chunks_created": index_result['chunks_created'],
            "indexing_time_s": indexing_time,
            "docs_per_sec": docs_per_sec,
            "chunks_per_sec": chunks_per_sec,
            "embedding_throughput": index_result['throughput_texts_per_sec']
        },
        "query_performance": {
            "avg_total_query_time_ms": avg_query_time * 1000,
            "avg_embedding_time_ms": avg_embed_time * 1000,
            "avg_retrieval_time_ms": avg_retrieval_time * 1000,
            "qps": qps
        },
        "memory_usage": memory_info,
        "performance_score": overall_score,
        "targets_met": targets_met,
        "total_targets": len(targets)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX Native End-to-End Benchmark für Apple Silicon Vector Database.")
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help=f"MLX Native Model ID zu testen. Verfügbar: {list(MLX_NATIVE_MODELS.keys())}"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Liste alle verfügbaren MLX native models"
    )
    args = parser.parse_args()
    
    if args.list_models:
        print("🍎 Verfügbare MLX Native Models:")
        print("=" * 50)
        for model_id, config in MLX_NATIVE_MODELS.items():
            print(f"📱 {model_id}")
            print(f"   Dimension: {config.dimension}")
            print(f"   Quantized: {config.quantized}")
            print(f"   Memory: {config.memory_pool_mb}MB")
            print()
        sys.exit(0)
    
    try:
        # Check MLX availability
        print(f"📱 MLX Device: {mx.default_device()}")
        
        # Run benchmark
        asyncio.run(run_mlx_native_e2e_benchmark(args.model))
        
    except ImportError as e:
        print(f"❌ MLX not available: {e}")
        print("Bitte installieren Sie MLX und mlx-embeddings:")
        print("pip install mlx mlx-embeddings")
    except KeyboardInterrupt:
        print("🛑 Benchmark durch Benutzer abgebrochen")
    except Exception as e:
        print(f"💥 Benchmark fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()