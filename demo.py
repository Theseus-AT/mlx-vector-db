#!/usr/bin/env python3
"""
Demo script for MLX Vector Database
Demonstrates basic functionality and usage patterns.
"""
import numpy as np
import time
from service.vector_store import (
    create_store, add_vectors, query_vectors,
    delete_vectors, delete_store, store_exists,
    count_vectors, list_users, list_models,
    batch_query, stream_query, bulk_delete
)

def run_basic_demo():
    """Run basic vector database operations demo."""
    print("🧠 MLX Vector Database Demo")
    print("=" * 50)
    
    user_id = "demo_user"
    model = "mistral"

    # 1. Create the store
    print(f"📁 Creating store for {user_id}/{model}")
    create_store(user_id, model)
    if store_exists(user_id, model):
        print(f"✅ Store created successfully")

    # 2. Add vectors
    print(f"➕ Adding sample vectors...")
    vecs = np.random.rand(5, 384).astype(np.float32)
    meta = [{"id": f"chunk_{i}", "source": "demo", "content": f"Sample content {i}"} for i in range(5)]
    
    start_time = time.time()
    add_vectors(user_id, model, vecs, meta)
    add_time = time.time() - start_time
    print(f"   Added 5 vectors in {add_time:.3f}s")

    # 3. Query vectors
    print(f"🔍 Querying vectors...")
    qvec = vecs[0]
    start_time = time.time()
    results = query_vectors(user_id, model, qvec, k=3)
    query_time = time.time() - start_time
    
    print(f"   Query completed in {query_time:.3f}s")
    print(f"   Found {len(results)} results:")
    for i, res in enumerate(results):
        print(f"     {i+1}. ID: {res.get('id')}, Score: {res.get('similarity_score', 0):.4f}")

    # 4. Batch query demo
    print(f"🧠 Testing batch query...")
    start_time = time.time()
    batch_results = batch_query(user_id, model, vecs[:3], k=2)
    batch_time = time.time() - start_time
    print(f"   Batch query (3 queries) completed in {batch_time:.3f}s")
    
    # 5. Store statistics
    print(f"📊 Store statistics:")
    stats = count_vectors(user_id, model)
    print(f"   Vectors: {stats['vectors']}, Metadata: {stats['metadata']}")

    # 6. Delete specific vector
    print(f"🗑️ Deleting vector with id 'chunk_1'...")
    deleted = delete_vectors(user_id, model, {"id": "chunk_1"})
    print(f"   Deleted {deleted} vector(s)")

    # 7. Final stats
    final_stats = count_vectors(user_id, model)
    print(f"📊 Final statistics:")
    print(f"   Vectors: {final_stats['vectors']}, Metadata: {final_stats['metadata']}")

    # 8. Cleanup
    print(f"🧹 Cleaning up...")
    delete_store(user_id, model)
    print(f"✅ Demo completed successfully!")

def run_performance_demo():
    """Run performance demonstration with larger datasets."""
    print("\n🚀 Performance Demo")
    print("=" * 50)
    
    user_id = "perf_user"
    model = "performance_test"
    
    # Create store
    create_store(user_id, model)
    
    # Test with larger dataset
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n📈 Testing with {size} vectors...")
        
        # Generate random vectors
        vecs = np.random.rand(size, 384).astype(np.float32)
        meta = [{"id": f"vec_{i}", "batch": "perf_test"} for i in range(size)]
        
        # Measure add time
        start_time = time.time()
        add_vectors(user_id, model, vecs, meta)
        add_time = time.time() - start_time
        
        # Measure query time
        query_vec = np.random.rand(384).astype(np.float32)
        start_time = time.time()
        results = query_vectors(user_id, model, query_vec, k=10)
        query_time = time.time() - start_time
        
        print(f"   Add time: {add_time:.3f}s ({size/add_time:.1f} vectors/s)")
        print(f"   Query time: {query_time:.3f}s")
        print(f"   Found {len(results)} results")
        
        # Clean up for next iteration
        bulk_delete(user_id, model, "batch", "perf_test")
    
    # Cleanup
    delete_store(user_id, model)
    print(f"✅ Performance demo completed!")

def run_advanced_demo():
    """Demonstrate advanced features like filtering and streaming."""
    print("\n🔧 Advanced Features Demo")
    print("=" * 50)
    
    user_id = "advanced_user"
    model = "advanced_test"
    
    create_store(user_id, model)
    
    # Add vectors with diverse metadata
    vecs = np.random.rand(10, 384).astype(np.float32)
    meta = [
        {"id": f"doc_{i}", "category": "A" if i < 5 else "B", "priority": i % 3}
        for i in range(10)
    ]
    
    add_vectors(user_id, model, vecs, meta)
    
    # Test metadata filtering
    print(f"🔍 Testing metadata filtering...")
    
    # Filter by category
    cat_a_results = query_vectors(user_id, model, vecs[0], k=10, filter_metadata={"category": "A"})
    print(f"   Category A results: {len(cat_a_results)}")
    
    # Filter by priority
    prio_results = query_vectors(user_id, model, vecs[0], k=10, filter_metadata={"priority": 1})
    print(f"   Priority 1 results: {len(prio_results)}")
    
    # Test streaming query
    print(f"🌊 Testing streaming query...")
    query_vecs = vecs[:3]
    stream_count = 0
    for result in stream_query(user_id, model, query_vecs, k=2):
        stream_count += 1
        print(f"   Stream result {stream_count}: {len(result)} items")
    
    # Cleanup
    delete_store(user_id, model)
    print(f"✅ Advanced demo completed!")

def main():
    """Run all demo functions."""
    try:
        run_basic_demo()
        run_performance_demo()
        run_advanced_demo()
        
        print(f"\n🎉 All demos completed successfully!")
        print(f"📖 Check the API documentation at: http://localhost:8000/docs")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()