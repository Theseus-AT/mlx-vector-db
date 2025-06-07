## examples/mlx_native_example.py
"""
MLX Native Pipeline Usage Example
Zeigt wie man die Pipeline in der Vector Database verwendet
"""

import asyncio
import time
from service.optimized_vector_store import create_optimized_vector_store
from integrations.mlx_lm_pipeline import MLXNativePipelineFactory, MLXNativeRAGPipeline

async def basic_example():
    """Basic MLX Native Pipeline Example"""
    
    print("🍎 MLX Native Pipeline - Basic Example")
    print("=" * 50)
    
    # 1. Get recommended model for use case
    model_id = MLXNativePipelineFactory.get_recommended_model(
        use_case="general",
        performance_priority="speed",  # or "quality", "balanced"
        memory_budget_mb=1024
    )
    print(f"📱 Using model: {model_id}")
    
    # 2. Create optimized vector store
    vector_store = create_optimized_vector_store(
        store_path="./mlx_example_store",
        dimension=384,  # Match model dimension
        jit_compile=True,
        enable_hnsw=False
    )
    
    # 3. Create MLX native pipeline
    pipeline = await MLXNativePipelineFactory.create_pipeline(
        model_id, vector_store, pipeline_type="basic"
    )
    
    # 4. Process texts
    texts = [
        "MLX framework enables efficient ML on Apple Silicon",
        "Vector databases provide fast similarity search",
        "Quantized models reduce memory usage significantly"
    ]
    
    result = await pipeline.process_texts(texts)
    print(f"✅ Processed {result['texts_processed']} texts")
    print(f"🚀 Throughput: {result['throughput_texts_per_sec']:.1f} texts/sec")
    
    # 5. Search for similar texts
    query = "machine learning on Apple chips"
    results = await pipeline.search_similar_texts(query, k=2)
    
    print(f"\n🔍 Search Results for: '{query}'")
    for result in results:
        print(f"   {result['rank']}. Similarity: {result['similarity_score']:.3f}")
        print(f"      Text: {result['text_preview']}")
    
    # 6. Get performance stats
    stats = pipeline.get_pipeline_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Device: {stats['device_info']['mlx_device']}")
    print(f"   Memory: {stats['memory_info'].get('peak_memory_mb', 0):.0f}MB")
    
    # Cleanup
    vector_store.clear()

async def rag_example():
    """Advanced RAG Pipeline Example"""
    
    print("\n🧠 MLX Native RAG Pipeline - Advanced Example")
    print("=" * 50)
    
    # 1. Choose high-quality model for RAG
    model_id = MLXNativePipelineFactory.get_recommended_model(
        use_case="high_quality",
        performance_priority="quality",
        memory_budget_mb=2048
    )
    print(f"📱 Using RAG model: {model_id}")
    
    # 2. Create vector store with higher dimension
    from integrations.mlx_lm_pipeline import MLX_NATIVE_MODELS
    model_config = MLX_NATIVE_MODELS[model_id]
    
    vector_store = create_optimized_vector_store(
        store_path="./mlx_rag_store",
        dimension=model_config.dimension,
        jit_compile=True,
        enable_hnsw=False
    )
    
    # 3. Create RAG pipeline
    rag_pipeline = await MLXNativePipelineFactory.create_pipeline(
        model_id, vector_store, pipeline_type="rag"
    )
    
    # 4. Prepare knowledge base
    documents = [
        {
            "title": "Apple Silicon Architecture",
            "content": """
            Apple Silicon M-series chips revolutionieren die Computer-Architektur durch 
            unified memory design. CPU, GPU und Neural Engine teilen sich denselben 
            Speicherpool, was Zero-Copy Operationen ermöglicht und die Latenz drastisch 
            reduziert. Die Metal Performance Shaders bieten optimierte GPU-Kernels für 
            Machine Learning Workloads. Der On-Chip Cache und die hohe Memory Bandwidth 
            sorgen für außergewöhnliche Performance bei ML-Inferenz.
            """,
            "source": "apple_silicon_guide.md"
        },
        {
            "title": "MLX Framework Deep Dive",
            "content": """
            MLX ist Apples nationales Machine Learning Framework, speziell für Apple Silicon 
            entwickelt. Es nutzt die unified memory Architektur optimal aus und bietet 
            automatische Differentiation, JIT-Compilation und lazy evaluation. MLX 
            unterstützt sowohl Forschung als auch Produktion mit NumPy-ähnlicher API. 
            Die Integration mit Metal ermöglicht GPU-Beschleunigung ohne explizite 
            Speicherverwaltung.
            """,
            "source": "mlx_framework.md"
        },
        {
            "title": "Vector Database Optimization",
            "content": """
            Moderne Vector Databases verwenden spezialisierte Indexing-Algorithmen wie 
            HNSW (Hierarchical Navigable Small World) für logarithmische Suchzeiten. 
            Auf Apple Silicon können diese Operationen durch Metal Performance Shaders 
            beschleunigt werden. Quantization reduziert Speicherverbrauch um 75% bei 
            minimaler Genauigkeitseinbuße. Batch-Processing optimiert GPU-Auslastung.
            """,
            "source": "vector_optimization.md"
        },
        {
            "title": "Embedding Models Performance",
            "content": """
            Moderne Embedding Models wie BGE, E5 und MiniLM erreichen state-of-the-art 
            Qualität bei verschiedenen Textaufgaben. 4-bit Quantization ermöglicht es, 
            große Modelle auf Consumer Hardware zu betreiben. MLX-native Implementierungen 
            nutzen Apple Silicon optimal aus. Batch-Inferenz kann Durchsatz um 10x steigern.
            """,
            "source": "embedding_performance.md"
        }
    ]
    
    # 5. Index documents with chunking
    print(f"📚 Indexing {len(documents)} documents...")
    index_start = time.time()
    
    index_result = await rag_pipeline.index_documents(
        documents, 
        chunk_size=300,  # Smaller chunks for better retrieval
        chunk_overlap=50
    )
    
    index_time = time.time() - index_start
    print(f"✅ Indexed in {index_time:.2f}s")
    print(f"📄 Created {index_result['chunks_created']} chunks")
    print(f"🚀 Throughput: {index_result['throughput_texts_per_sec']:.1f} texts/sec")
    
    # 6. RAG Query Examples
    rag_queries = [
        "Wie funktioniert unified memory bei Apple Silicon?",
        "Was sind die Vorteile von MLX gegenüber anderen Frameworks?",
        "Welche Optimierungen gibt es für Vector Databases?",
        "Wie verbessert Quantization die Performance?"
    ]
    
    print(f"\n🔍 RAG Query Examples:")
    
    for i, query in enumerate(rag_queries, 1):
        print(f"\n   📝 Query {i}: {query}")
        
        # Retrieve relevant context
        retrieval_start = time.time()
        context_chunks = await rag_pipeline.retrieve_context(
            query, 
            k=3, 
            min_similarity=0.3,  # Lower threshold for demo
            max_context_length=1000
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        print(f"   ⏱️ Retrieval: {retrieval_time:.2f}ms")
        print(f"   📋 Found {len(context_chunks)} relevant chunks:")
        
        for j, chunk in enumerate(context_chunks):
            print(f"      {j+1}. [{chunk['source']}] Sim: {chunk['similarity']:.3f}")
            print(f"         {chunk['text'][:100]}...")
        
        # Generate RAG prompt
        rag_prompt = rag_pipeline.format_rag_prompt(
            query, 
            context_chunks,
            system_prompt="Du bist ein Experte für Apple Silicon und Machine Learning. Beantworte die Frage präzise basierend auf dem Kontext."
        )
        
        print(f"   📋 RAG Prompt Length: {len(rag_prompt)} chars")
        
        # Show first part of prompt for demo
        prompt_preview = rag_prompt[:200] + "..." if len(rag_prompt) > 200 else rag_prompt
        print(f"   📖 Prompt Preview: {prompt_preview}")
    
    # 7. Advanced RAG statistics
    print(f"\n📊 RAG Pipeline Statistics:")
    rag_stats = rag_pipeline.get_rag_stats()
    
    pipeline_stats = rag_stats['pipeline_stats']
    print(f"   📚 Documents indexed: {rag_stats['rag_stats']['documents_indexed']}")
    print(f"   📄 Total chunks: {rag_stats['rag_stats']['total_chunks']}")
    print(f"   📏 Avg chunk size: {rag_stats['rag_stats']['avg_chunk_size']:.0f} chars")
    print(f"   🔍 Queries processed: {rag_stats['rag_stats']['retrieval_queries']}")
    print(f"   ⏱️ Avg retrieval time: {rag_stats['rag_stats']['avg_retrieval_time_ms']:.2f}ms")
    
    model_stats = rag_stats['embedding_model_stats']
    if not model_stats.get('no_data', False):
        print(f"   🧠 Total inferences: {model_stats['total_inferences']}")
        print(f"   📈 Throughput: {model_stats['throughput_texts_per_sec']:.1f} texts/sec")
        print(f"   💾 Peak memory: {rag_stats['memory_info'].get('peak_memory_mb', 0):.0f}MB")
    
    # Cleanup
    vector_store.clear()

async def benchmark_example():
    """MLX Native Models Benchmark Example"""
    
    print("\n🏁 MLX Native Models Benchmark")
    print("=" * 50)
    
    from integrations.mlx_lm_pipeline import MLXNativeBenchmark
    
    # Initialize benchmark suite
    benchmark = MLXNativeBenchmark()
    
    # Models to benchmark (limit for demo)
    models_to_test = [
        "mlx-community/bge-small-en-v1.5-4bit",
        "mlx-community/all-MiniLM-L6-v2-4bit",
        "mlx-community/multilingual-e5-small-4bit"
    ]
    
    print(f"🧪 Testing {len(models_to_test)} MLX native models...")
    print(f"📱 Device: {mlx.core.mx.default_device()}")
    
    # Run comprehensive benchmark
    benchmark_results = await benchmark.run_comprehensive_benchmark(
        models_to_test=models_to_test,
        num_test_texts=50  # Smaller number for demo
    )
    
    # Display results
    summary = benchmark_results['summary']
    
    print(f"\n📊 Benchmark Results Summary:")
    print(f"   ✅ Successful models: {summary['test_configuration']['successful_models']}")
    print(f"   ❌ Failed models: {summary['test_configuration']['failed_models']}")
    
    if summary['test_configuration']['successful_models'] > 0:
        leaders = summary['performance_leaders']
        
        print(f"\n🏆 Performance Leaders:")
        print(f"   🚀 Best Throughput: {leaders['best_throughput']['model']}")
        print(f"      {leaders['best_throughput']['value']:.1f} {leaders['best_throughput']['unit']}")
        
        print(f"   ⚡ Best Latency: {leaders['best_latency']['model']}")
        print(f"      {leaders['best_latency']['value']:.2f} {leaders['best_latency']['unit']}")
        
        print(f"   🔍 Best Search: {leaders['best_search']['model']}")
        print(f"      {leaders['best_search']['value']:.2f} {leaders['best_search']['unit']}")
        
        print(f"\n💡 Recommendations:")
        recommendations = summary['recommendations']
        print(f"   For Speed: {recommendations['for_speed']}")
        print(f"   For Throughput: {recommendations['for_throughput']}")
        print(f"   For Search: {recommendations['for_search']}")
        print(f"   General Purpose: {recommendations['general_purpose']}")
        
        # Memory usage comparison
        memory_usage = summary['memory_usage']
        print(f"\n💾 Memory Usage:")
        for model, memory_mb in memory_usage.items():
            print(f"   {model}: {memory_mb:.0f}MB")
    
    # Show detailed results for first successful model
    for model_id, result in benchmark_results['benchmark_results'].items():
        if result.get('success', False):
            print(f"\n📈 Detailed Results for {model_id}:")
            perf = result['performance']
            print(f"   Initialization: {perf['initialization_time_s']:.2f}s")
            print(f"   Single encode: {perf['single_encode_time_ms']:.2f}ms")
            print(f"   Batch throughput: {perf['batch_throughput_texts_per_sec']:.1f} texts/sec")
            print(f"   Search latency: {perf['avg_search_time_ms']:.2f}ms")
            break  # Show only first for demo

async def integration_example():
    """Integration with existing Vector Database"""
    
    print("\n🔗 Integration with Vector Database")
    print("=" * 50)
    
    # This shows how to integrate with your existing vector database
    from api.routes.vectors import store_manager  # Your existing store manager
    
    # 1. Get recommended model
    model_id = MLXNativePipelineFactory.get_recommended_model("general")
    
    # 2. Create store via your existing API
    user_id = "mlx_demo_user"
    model_name = "mlx_demo_model"
    
    try:
        # Use your existing store creation
        await store_manager.create_store(user_id, model_name)
        store = await store_manager.get_store(user_id, model_name)
        
        # 3. Create MLX native pipeline with existing store
        pipeline = await MLXNativePipelineFactory.create_pipeline(
            model_id, store, pipeline_type="basic"
        )
        
        # 4. Process texts through MLX pipeline
        demo_texts = [
            "MLX native embeddings on Apple Silicon",
            "High performance vector database operations",
            "Optimized machine learning inference"
        ]
        
        process_result = await pipeline.process_texts(demo_texts)
        print(f"✅ Processed via MLX pipeline: {process_result['texts_processed']} texts")
        print(f"🚀 Performance: {process_result['throughput_texts_per_sec']:.1f} texts/sec")
        
        # 5. Search via MLX pipeline
        search_results = await pipeline.search_similar_texts(
            "machine learning performance", k=2
        )
        
        print(f"🔍 Search results:")
        for result in search_results:
            print(f"   Similarity: {result['similarity_score']:.3f}")
            print(f"   Text: {result['text_preview']}")
        
        # 6. Get comprehensive stats
        stats = pipeline.get_pipeline_stats()
        print(f"📊 MLX Pipeline integrated successfully!")
        print(f"   Model: {stats['model_configuration']['model_id']}")
        print(f"   Device: {stats['device_info']['mlx_device']}")
        
        # Cleanup
        await store_manager.delete_store(user_id, model_name)
        
    except Exception as e:
        print(f"❌ Integration example failed: {e}")
        print("Make sure your vector database server is running!")

async def main():
    """Run all examples"""
    
    print("🍎 MLX Native Pipeline - Complete Examples")
    print("⚡ Apple Silicon Optimized Vector Database")
    print("=" * 60)
    
    try:
        # Check MLX availability
        import mlx.core as mx
        print(f"📱 MLX Device: {mx.default_device()}")
        
        # Run examples
        await basic_example()
        await rag_example()
        await benchmark_example()
        
        # Integration example (may fail if server not running)
        print(f"\n🔗 Testing Integration (optional)...")
        try:
            await integration_example()
        except Exception as e:
            print(f"⚠️ Integration example skipped: {e}")
            print("   Start your vector database server to test integration")
        
        print(f"\n🎉 All MLX Native Pipeline examples completed!")
        print(f"✅ Ready for production on Apple Silicon!")
        
    except ImportError as e:
        print(f"❌ MLX not available: {e}")
        print("Please install MLX and mlx-embeddings:")
        print("pip install mlx mlx-embeddings")
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())