"""
Performance testing script for HNSW Index implementation
Compares HNSW vs brute force search performance
"""

import mlx.core as mx
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple
import json

# Import the implementations
from vector_store import VectorStore, VectorStoreConfig
from indexing.hnsw_index import HNSWConfig

class PerformanceTester:
    def __init__(self):
        self.results = {
            'build_times': {},
            'query_times': {},
            'accuracy': {},
            'memory_usage': {}
        }
        
    def test_build_performance(self, vector_counts: List[int], dim: int = 384):
        """Test index building performance for different dataset sizes"""
        print("\n=== Testing Build Performance ===")
        
        for n_vectors in vector_counts:
            print(f"\nTesting with {n_vectors} vectors...")
            
            # Generate random vectors
            vectors = mx.random.normal((n_vectors, dim))
            metadata = [{"id": f"vec_{i}"} for i in range(n_vectors)]
            
            # Test HNSW build time
            store_path = Path(f"./test_store_{n_vectors}")
            config = VectorStoreConfig(
                enable_hnsw=True,
                auto_index_threshold=100,  # Force HNSW build
                hnsw_config=HNSWConfig(M=16, ef_construction=200)
            )
            
            store = VectorStore(store_path, config)
            
            start_time = time.time()
            store.add_vectors(vectors, metadata)
            build_time = time.time() - start_time
            
            self.results['build_times'][n_vectors] = build_time
            print(f"Build time: {build_time:.2f} seconds")
            
            # Clean up
            store.clear()
            store_path.rmdir()
            
    def test_query_performance(self, n_vectors: int = 100000, dim: int = 384, 
                             n_queries: int = 1000, k_values: List[int] = [1, 10, 50, 100]):
        """Test query performance comparison"""
        print(f"\n=== Testing Query Performance ({n_vectors} vectors) ===")
        
        # Create test store
        store_path = Path("./test_query_store")
        config = VectorStoreConfig(
            enable_hnsw=True,
            auto_index_threshold=100,
            hnsw_config=HNSWConfig(M=16, ef_construction=200, ef_search=50)
        )
        
        store = VectorStore(store_path, config)
        
        # Add vectors
        print("Building test dataset...")
        batch_size = 10000
        for i in range(0, n_vectors, batch_size):
            batch_vectors = mx.random.normal((min(batch_size, n_vectors - i), dim))
            batch_metadata = [{"id": f"vec_{j}"} for j in range(i, i + len(batch_vectors))]
            store.add_vectors(batch_vectors, batch_metadata)
            
        # Generate query vectors
        query_vectors = mx.random.normal((n_queries, dim))
        
        for k in k_values:
            print(f"\nTesting k={k}:")
            
            # Test HNSW performance
            hnsw_times = []
            for i in range(n_queries):
                start = time.time()
                store.query(query_vectors[i], k=k, use_hnsw=True)
                hnsw_times.append(time.time() - start)
                
            # Test brute force performance (on subset for speed)
            bf_times = []
            n_bf_queries = min(100, n_queries)  # Limit brute force queries
            for i in range(n_bf_queries):
                start = time.time()
                store.query(query_vectors[i], k=k, use_hnsw=False)
                bf_times.append(time.time() - start)
                
            avg_hnsw = np.mean(hnsw_times) * 1000  # Convert to ms
            avg_bf = np.mean(bf_times) * 1000
            
            self.results['query_times'][f"hnsw_k{k}"] = avg_hnsw
            self.results['query_times'][f"brute_force_k{k}"] = avg_bf
            
            print(f"  HNSW: {avg_hnsw:.2f} ms (avg), {np.percentile(hnsw_times, 95)*1000:.2f} ms (p95)")
            print(f"  Brute Force: {avg_bf:.2f} ms (avg)")
            print(f"  Speedup: {avg_bf/avg_hnsw:.2f}x")
            
        # Clean up
        store.clear()
        store_path.rmdir()
        
    def test_accuracy(self, n_vectors: int = 10000, dim: int = 384, 
                     n_queries: int = 100, k: int = 10):
        """Test HNSW accuracy vs brute force"""
        print(f"\n=== Testing Accuracy ({n_vectors} vectors) ===")
        
        # Create test store
        store_path = Path("./test_accuracy_store")
        config = VectorStoreConfig(
            enable_hnsw=True,
            auto_index_threshold=100,
            hnsw_config=HNSWConfig(M=16, ef_construction=200, ef_search=50)
        )
        
        store = VectorStore(store_path, config)
        
        # Add vectors
        vectors = mx.random.normal((n_vectors, dim))
        metadata = [{"id": f"vec_{i}"} for i in range(n_vectors)]
        store.add_vectors(vectors, metadata)
        
        # Test queries
        recalls = []
        for i in range(n_queries):
            query = mx.random.normal((dim,))
            
            # Get ground truth (brute force)
            indices_bf, _, _ = store.query(query, k=k, use_hnsw=False)
            
            # Get HNSW results
            indices_hnsw, _, _ = store.query(query, k=k, use_hnsw=True)
            
            # Calculate recall
            recall = len(set(indices_hnsw) & set(indices_bf)) / k
            recalls.append(recall)
            
        avg_recall = np.mean(recalls)
        self.results['accuracy']['recall@10'] = avg_recall
        
        print(f"Average Recall@{k}: {avg_recall:.3f}")
        print(f"Min Recall: {min(recalls):.3f}")
        print(f"Max Recall: {max(recalls):.3f}")
        
        # Clean up
        store.clear()
        store_path.rmdir()
        
    def test_parameter_tuning(self, n_vectors: int = 50000, dim: int = 384):
        """Test different HNSW parameters"""
        print(f"\n=== Testing Parameter Tuning ({n_vectors} vectors) ===")
        
        # Generate test data once
        vectors = mx.random.normal((n_vectors, dim))
        metadata = [{"id": f"vec_{i}"} for i in range(n_vectors)]
        query_vectors = mx.random.normal((100, dim))
        
        # Test different M values
        m_values = [8, 16, 32]
        ef_values = [50, 100, 200]
        
        for m in m_values:
            for ef in ef_values:
                print(f"\nTesting M={m}, ef_search={ef}:")
                
                # Create store with specific config
                store_path = Path(f"./test_param_store")
                config = VectorStoreConfig(
                    enable_hnsw=True,
                    auto_index_threshold=100,
                    hnsw_config=HNSWConfig(M=m, ef_construction=200, ef_search=ef)
                )
                
                store = VectorStore(store_path, config)
                
                # Build index
                start = time.time()
                store.add_vectors(vectors, metadata)
                build_time = time.time() - start
                
                # Test queries
                query_times = []
                for q in query_vectors:
                    start = time.time()
                    store.query(q, k=10, use_hnsw=True)
                    query_times.append(time.time() - start)
                    
                avg_query_time = np.mean(query_times) * 1000
                
                print(f"  Build time: {build_time:.2f}s")
                print(f"  Avg query time: {avg_query_time:.2f}ms")
                
                # Store results
                param_key = f"M{m}_ef{ef}"
                self.results['accuracy'][f'{param_key}_build'] = build_time
                self.results['accuracy'][f'{param_key}_query'] = avg_query_time
                
                # Clean up
                store.clear()
                store_path.rmdir()
                
    def generate_report(self):
        """Generate performance report with visualizations"""
        print("\n=== Generating Performance Report ===")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('HNSW Performance Analysis', fontsize=16)
        
        # 1. Build time vs dataset size
        if self.results['build_times']:
            ax = axes[0, 0]
            sizes = sorted(self.results['build_times'].keys())
            times = [self.results['build_times'][s] for s in sizes]
            
            ax.plot(sizes, times, 'b-o')
            ax.set_xlabel('Number of Vectors')
            ax.set_ylabel('Build Time (seconds)')
            ax.set_title('Index Build Time Scaling')
            ax.grid(True)
            ax.set_xscale('log')
            
        # 2. Query performance comparison
        if self.results['query_times']:
            ax = axes[0, 1]
            k_values = []
            hnsw_times = []
            bf_times = []
            
            for key in sorted(self.results['query_times'].keys()):
                if 'hnsw' in key:
                    k = int(key.split('k')[1])
                    k_values.append(k)
                    hnsw_times.append(self.results['query_times'][key])
                    bf_key = key.replace('hnsw', 'brute_force')
                    if bf_key in self.results['query_times']:
                        bf_times.append(self.results['query_times'][bf_key])
                        
            x = np.arange(len(k_values))
            width = 0.35
            
            ax.bar(x - width/2, hnsw_times, width, label='HNSW', color='green')
            ax.bar(x + width/2, bf_times, width, label='Brute Force', color='red')
            
            ax.set_xlabel('k (number of neighbors)')
            ax.set_ylabel('Query Time (ms)')
            ax.set_title('Query Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(k_values)
            ax.legend()
            ax.grid(True, axis='y')
            
        # 3. Speedup analysis
        if self.results['query_times']:
            ax = axes[1, 0]
            speedups = []
            
            for i, (h, b) in enumerate(zip(hnsw_times, bf_times)):
                speedups.append(b / h)
                
            ax.bar(x, speedups, color='blue')
            ax.set_xlabel('k (number of neighbors)')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('HNSW Speedup vs Brute Force')
            ax.set_xticks(x)
            ax.set_xticklabels(k_values)
            ax.grid(True, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(speedups):
                ax.text(i, v + 0.5, f'{v:.1f}x', ha='center')
                
        # 4. Accuracy metrics
        if 'recall@10' in self.results['accuracy']:
            ax = axes[1, 1]
            ax.text(0.5, 0.5, f"Average Recall@10: {self.results['accuracy']['recall@10']:.3f}", 
                   ha='center', va='center', fontsize=20, transform=ax.transAxes)
            ax.set_title('Search Accuracy')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig('hnsw_performance_report.png', dpi=150)
        print("Report saved as 'hnsw_performance_report.png'")
        
        # Save raw results
        with open('hnsw_performance_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print("Raw results saved as 'hnsw_performance_results.json'")
        
        return self.results


def main():
    """Run comprehensive performance tests"""
    tester = PerformanceTester()
    
    # Test 1: Build performance scaling
    print("Starting performance tests...")
    tester.test_build_performance(
        vector_counts=[1000, 5000, 10000, 50000, 100000],
        dim=384
    )
    
    # Test 2: Query performance
    tester.test_query_performance(
        n_vectors=100000,
        n_queries=1000,
        k_values=[1, 10, 50, 100]
    )
    
    # Test 3: Accuracy
    tester.test_accuracy(
        n_vectors=10000,
        n_queries=100,
        k=10
    )
    
    # Test 4: Parameter tuning
    tester.test_parameter_tuning(
        n_vectors=50000
    )
    
    # Generate report
    results = tester.generate_report()
    
    print("\n=== Summary ===")
    print(f"Best query speedup: {max([v/results['query_times'][k.replace('hnsw', 'brute_force')] for k, v in results['query_times'].items() if 'hnsw' in k]):.2f}x")
    print(f"Average recall: {results['accuracy'].get('recall@10', 'N/A')}")
    
    return results


if __name__ == "__main__":
    # Run performance tests
    results = main()
