#!/usr/bin/env python3
"""
Performance Demo für MLX Vector Database
Zeigt die Performance-Verbesserungen durch HNSW, Caching und MLX Compilation
"""
import requests
import time
import numpy as np
import json

BASE_URL = "http://localhost:8000"

def wait_for_server():
    """Wait for server to be ready"""
    print("🔍 Checking server availability...")
    for i in range(30):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return
        except:
            pass
        time.sleep(1)
    raise Exception("❌ Server not available after 30 seconds")

def get_api_key():
    """Get API key from user or environment"""
    import os
    api_key = os.getenv("VECTOR_DB_API_KEY")
    if not api_key:
        api_key = input("Enter your API key (from .env file): ").strip()
    return api_key

def create_test_store(api_key: str):
    """Create a test store for performance testing"""
    user_id = "perf_test_user"
    model_id = "perf_test_model"
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Delete if exists
    try:
        delete_payload = {"user_id": user_id, "model_id": model_id}
        requests.delete(f"{BASE_URL}/admin/store", json=delete_payload, headers=headers)
    except:
        pass
    
    # Create store
    create_payload = {"user_id": user_id, "model_id": model_id}
    response = requests.post(f"{BASE_URL}/admin/create_store", json=create_payload, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to create store: {response.text}")
    
    print(f"✅ Created test store: {user_id}/{model_id}")
    return user_id, model_id

def run_performance_demo():
    """Run comprehensive performance demonstration"""
    print("🚀 MLX Vector Database Performance Demo")
    print("=" * 60)
    
    wait_for_server()
    api_key = get_api_key()
    
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    
    # Step 1: Performance Health Check
    print("\n1️⃣ Performance Health Check")
    response = requests.get(f"{BASE_URL}/performance/health", headers=headers)
    if response.status_code == 200:
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   MLX Operations: {health['mlx_operations']}")
        print(f"   Cache: {health['cache_status']}")
    else:
        print(f"   ❌ Health check failed: {response.text}")
        return
    
    # Step 2: Warmup Compiled Functions
    print("\n2️⃣ Warming up compiled functions...")
    response = requests.post(f"{BASE_URL}/performance/warmup?dimension=384", headers=headers)
    if response.status_code == 200:
        warmup = response.json()
        print(f"   ✅ Warmup completed in {warmup['warmup_time_seconds']:.3f}s")
    else:
        print(f"   ⚠️ Warmup failed: {response.text}")
    
    # Step 3: Create Test Store
    print("\n3️⃣ Setting up test environment...")
    user_id, model_id = create_test_store(api_key)
    
    # Step 4: Add Test Vectors
    print("\n4️⃣ Adding test vectors...")
    vector_count = 5000
    dimension = 384
    
    # Generate test data
    vectors = np.random.rand(vector_count, dimension).astype(np.float32)
    metadata = [{"id": f"test_vec_{i}", "category": f"cat_{i%10}"} for i in range(vector_count)]
    
    add_payload = {
        "user_id": user_id,
        "model_id": model_id,
        "vectors": vectors.tolist(),
        "metadata": metadata
    }
    
    start_time = time.time()
    response = requests.post(f"{BASE_URL}/admin/add_test_vectors", json=add_payload, headers=headers)
    add_time = time.time() - start_time
    
    if response.status_code == 200:
        print(f"   ✅ Added {vector_count} vectors in {add_time:.3f}s")
        print(f"   📊 Rate: {vector_count/add_time:.1f} vectors/second")
    else:
        print(f"   ❌ Failed to add vectors: {response.text}")
        return
    
    # Step 5: Run Performance Benchmark
    print("\n5️⃣ Running performance benchmark...")
    benchmark_payload = {
        "user_id": user_id,
        "model_id": model_id,
        "test_size": 1000,  # Additional test vectors
        "query_count": 100,
        "vector_dim": dimension
    }
    
    response = requests.post(f"{BASE_URL}/performance/benchmark", json=benchmark_payload, headers=headers)
    
    if response.status_code == 200:
        results = response.json()
        
        # Display results
        print("\n📊 BENCHMARK RESULTS:")
        print("-" * 40)
        
        # Single Query Performance
        single_query = results["single_query"]
        print(f"Single Query Performance:")
        print(f"   Optimized: {single_query['optimized_time']:.4f}s")
        print(f"   Basic: {single_query['basic_time']:.4f}s")
        print(f"   🚀 Speedup: {single_query['speedup_factor']:.1f}x")
        print(f"   📈 Capacity: {single_query['queries_per_second_optimized']:.1f} QPS")
        
        # Batch Query Performance
        batch_query = results["batch_query"]
        print(f"\nBatch Query Performance:")
        print(f"   Batch size: {batch_query['batch_size']} queries")
        print(f"   Total time: {batch_query['total_time']:.4f}s")
        print(f"   📈 Throughput: {batch_query['queries_per_second']:.1f} QPS")
        
        # Cache Performance
        cache_perf = results["cache_performance"]
        print(f"\nCache Performance:")
        print(f"   Hit rate: {cache_perf['hit_rate_percent']:.1f}%")
        print(f"   Memory usage: {cache_perf['memory_usage_gb']:.2f} GB")
        
        # Overall Summary
        summary = results["summary"]
        estimated_capacity = summary["performance_improvement"]["estimated_capacity"]
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        print(f"   Overall speedup: {summary['performance_improvement']['query_speedup']:.1f}x")
        print(f"   Estimated capacity: {estimated_capacity}")
        print(f"   Vector database size: {vector_count + benchmark_payload['test_size']} vectors")
        
        # Performance Rating
        speedup = summary["performance_improvement"]["query_speedup"]
        if speedup > 50:
            rating = "🔥 EXCELLENT"
        elif speedup > 10:
            rating = "⭐ VERY GOOD"
        elif speedup > 5:
            rating = "✅ GOOD"
        else:
            rating = "⚠️ NEEDS OPTIMIZATION"
        
        print(f"   Performance Rating: {rating}")
        
    else:
        print(f"   ❌ Benchmark failed: {response.text}")
        return
    
    # Step 6: Optimize Store
    print("\n6️⃣ Optimizing store...")
    optimize_params = {"user_id": user_id, "model_id": model_id, "force_rebuild_index": "true"}
    response = requests.post(f"{BASE_URL}/performance/optimize", params=optimize_params, headers=headers)
    
    if response.status_code == 200:
        opt_results = response.json()
        opt_data = opt_results["optimization_results"]
        
        if "index_optimization" in opt_data:
            index_info = opt_data["index_optimization"]
            print(f"   ✅ HNSW Index: {index_info['index_nodes']} nodes in {index_info['time_seconds']:.3f}s")
        
        print(f"   ✅ Store optimization completed")
    else:
        print(f"   ⚠️ Optimization failed: {response.text}")
    
    # Step 7: Final Performance Stats
    print("\n7️⃣ Final performance statistics...")
    response = requests.get(f"{BASE_URL}/performance/stats", headers=headers)
    
    if response.status_code == 200:
        stats = response.json()
        
        cache_stats = stats["cache"]
        print(f"Cache Statistics:")
        print(f"   Entries: {cache_stats['entries']}")
        print(f"   Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"   Memory usage: {cache_stats['memory_usage_gb']:.2f} GB")
        
        if "compiled_functions" in stats and stats["compiled_functions"]:
            print(f"\nCompiled Functions:")
            for func_name, func_stats in stats["compiled_functions"].items():
                avg_time = func_stats.get("avg_time", 0)
                calls = func_stats.get("calls", 0)
                print(f"   {func_name}: {calls} calls, {avg_time:.4f}s avg")
    
    print("\n🎉 Performance demo completed!")
    print("\n💡 Key Takeaways:")
    print("   • HNSW indexing provides logarithmic search time")
    print("   • Caching eliminates repeated disk I/O")
    print("   • MLX compilation optimizes Apple Silicon")
    print("   • Combined optimizations achieve 10-100x speedup")
    
    # Cleanup
    print(f"\n🧹 Cleaning up test store...")
    delete_payload = {"user_id": user_id, "model_id": model_id}
    requests.delete(f"{BASE_URL}/admin/store", json=delete_payload, headers=headers)
    print(f"   ✅ Test store deleted")

if __name__ == "__main__":
    try:
        run_performance_demo()
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()