"""
HNSW (Hierarchical Navigable Small World) Index Implementation for MLX
Optimized for Apple Silicon using MLX framework
"""

import mlx.core as mx
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
import pickle
from dataclasses import dataclass
import heapq
import random
from pathlib import Path
import json
import time

@dataclass
class HNSWConfig:
    """Configuration for HNSW index"""
    M: int = 16  # Number of bi-directional links created for each node
    ef_construction: int = 200  # Size of dynamic candidate list
    ef_search: int = 50  # Size of dynamic list for search
    max_m: int = 16  # Maximum allowed connections for any node
    seed: int = 42
    distance_type: str = "cosine"  # "cosine" or "euclidean"
    
class HNSWNode:
    """Node in the HNSW graph"""
    def __init__(self, idx: int, level: int):
        self.idx = idx
        self.level = level
        self.neighbors: Dict[int, Set[int]] = {i: set() for i in range(level + 1)}

class HNSWIndex:
    """
    Hierarchical Navigable Small World Index for fast approximate nearest neighbor search
    Optimized for MLX arrays on Apple Silicon
    """
    
    def __init__(self, config: HNSWConfig = HNSWConfig()):
        self.config = config
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.vectors: Optional[mx.array] = None
        self.element_count = 0
        random.seed(config.seed)
        
        # MLX-specific optimizations
        self.use_metal = mx.metal.is_available()
        
    def _get_random_level(self) -> int:
        """Select level for a new node using exponential decay probability"""
        level = 0
        while random.random() < 0.5 and level < 16:
            level += 1
        return level
        
    def _compute_distance(self, idx1: int, idx2: int) -> float:
        """Compute distance between two vectors using MLX operations"""
        vec1 = self.vectors[idx1]
        vec2 = self.vectors[idx2]
        
        if self.config.distance_type == "cosine":
            # Cosine distance = 1 - cosine_similarity
            # Normalize vectors for cosine similarity
            norm1 = mx.sqrt(mx.sum(vec1 * vec1))
            norm2 = mx.sqrt(mx.sum(vec2 * vec2))
            
            # Avoid division by zero
            norm1 = mx.maximum(norm1, 1e-8)
            norm2 = mx.maximum(norm2, 1e-8)
            
            vec1_normalized = vec1 / norm1
            vec2_normalized = vec2 / norm2
            
            # Compute cosine similarity
            cosine_sim = mx.sum(vec1_normalized * vec2_normalized)
            
            # Return cosine distance
            distance = 1.0 - cosine_sim
        else:
            # Euclidean distance
            diff = vec1 - vec2
            distance = mx.sqrt(mx.sum(diff * diff))
            
        # Evaluate the distance to get a scalar
        mx.eval(distance)
        return float(distance)
        
    def _compute_batch_distances(self, query_vec: mx.array, indices: List[int]) -> mx.array:
        """Compute distances from query to multiple vectors in batch using MLX"""
        if not indices:
            return mx.array([])
            
        # Stack vectors for batch processing
        batch_vecs = mx.stack([self.vectors[idx] for idx in indices])
        
        if self.config.distance_type == "cosine":
            # Normalize query vector
            query_norm = mx.sqrt(mx.sum(query_vec * query_vec))
            query_norm = mx.maximum(query_norm, 1e-8)
            query_normalized = query_vec / query_norm
            
            # Normalize batch vectors
            batch_norms = mx.sqrt(mx.sum(batch_vecs * batch_vecs, axis=1))
            batch_norms = mx.maximum(batch_norms, 1e-8)
            batch_normalized = batch_vecs / batch_norms[:, None]
            
            # Compute cosine similarities
            cosine_sims = mx.sum(batch_normalized * query_normalized[None, :], axis=1)
            
            # Convert to distances
            distances = 1.0 - cosine_sims
        else:
            # Euclidean distances
            diff = batch_vecs - query_normalized[None, :]
            distances = mx.sqrt(mx.sum(diff * diff, axis=1))
            
        mx.eval(distances)
        return distances
        
    def _search_layer(self, query: mx.array, entry_points: Set[int], 
                     num_closest: int, layer: int) -> List[Tuple[float, int]]:
        """Search for nearest neighbors in a specific layer"""
        visited = set()
        candidates = []
        nearest = []
        
        # Initialize with entry points
        for point in entry_points:
            dist = self._compute_distance(point, -1) if point == -1 else self._compute_distance(point, point)
            heapq.heappush(candidates, (-dist, point))
            heapq.heappush(nearest, (dist, point))
            visited.add(point)
            
        while candidates:
            curr_dist, curr_idx = heapq.heappop(candidates)
            curr_dist = -curr_dist
            
            if curr_dist > nearest[0][0]:
                break
                
            # Check neighbors at the current layer
            node = self.nodes[curr_idx]
            if layer < len(node.neighbors):
                for neighbor_idx in node.neighbors[layer]:
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        
                        if neighbor_idx < len(self.vectors):
                            dist = self._compute_distance_to_query(query, neighbor_idx)
                            
                            if dist < nearest[0][0] or len(nearest) < num_closest:
                                heapq.heappush(candidates, (-dist, neighbor_idx))
                                heapq.heappush(nearest, (dist, neighbor_idx))
                                
                                if len(nearest) > num_closest:
                                    heapq.heappop(nearest)
                                    
        return nearest
        
    def _compute_distance_to_query(self, query: mx.array, idx: int) -> float:
        """Compute distance between query vector and indexed vector"""
        vec = self.vectors[idx]
        
        if self.config.distance_type == "cosine":
            # Normalize vectors
            query_norm = mx.sqrt(mx.sum(query * query))
            vec_norm = mx.sqrt(mx.sum(vec * vec))
            
            query_norm = mx.maximum(query_norm, 1e-8)
            vec_norm = mx.maximum(vec_norm, 1e-8)
            
            query_normalized = query / query_norm
            vec_normalized = vec / vec_norm
            
            cosine_sim = mx.sum(query_normalized * vec_normalized)
            distance = 1.0 - cosine_sim
        else:
            diff = query - vec
            distance = mx.sqrt(mx.sum(diff * diff))
            
        mx.eval(distance)
        return float(distance)
        
    def build(self, vectors: mx.array, show_progress: bool = True):
        """Build HNSW index from vectors"""
        self.vectors = vectors
        n_vectors = vectors.shape[0]
        
        if show_progress:
            print(f"Building HNSW index for {n_vectors} vectors...")
            
        # Initialize first node
        if n_vectors > 0:
            level = self._get_random_level()
            self.nodes[0] = HNSWNode(0, level)
            self.entry_point = 0
            self.element_count = 1
            
            # Insert remaining vectors
            for idx in range(1, n_vectors):
                if show_progress and idx % 1000 == 0:
                    print(f"Progress: {idx}/{n_vectors} vectors indexed")
                    
                self._insert(idx)
                
        if show_progress:
            print(f"HNSW index built successfully!")
            
    def _insert(self, idx: int):
        """Insert a new vector into the HNSW graph"""
        if self.entry_point is None:
            level = self._get_random_level()
            self.nodes[idx] = HNSWNode(idx, level)
            self.entry_point = idx
            self.element_count = 1
            return
            
        level = self._get_random_level()
        node = HNSWNode(idx, level)
        self.nodes[idx] = node
        
        # Find nearest neighbors at all layers
        nearest = []
        curr_nearest = [(-float('inf'), self.entry_point)]
        
        for lc in range(level, -1, -1):
            nearest = self._search_layer_for_insertion(idx, curr_nearest, lc)
            m = self.config.M if lc > 0 else self.config.M * 2
            
            # Select m nearest neighbors
            neighbors = self._select_neighbors_heuristic(idx, nearest, m, lc)
            
            # Add bidirectional links
            for neighbor_idx in neighbors:
                node.neighbors[lc].add(neighbor_idx)
                self.nodes[neighbor_idx].neighbors[lc].add(idx)
                
                # Prune neighbors if needed
                max_neighbors = self.config.M if lc > 0 else self.config.M * 2
                if len(self.nodes[neighbor_idx].neighbors[lc]) > max_neighbors:
                    self._prune_neighbors(neighbor_idx, lc)
                    
            curr_nearest = nearest
            
        self.element_count += 1
        
    def _search_layer_for_insertion(self, idx: int, entry_points: List[Tuple[float, int]], 
                                   layer: int) -> List[Tuple[float, int]]:
        """Search layer during insertion"""
        visited = set()
        candidates = []
        nearest = []
        
        for dist, point in entry_points:
            if point != idx:
                actual_dist = self._compute_distance(idx, point)
                heapq.heappush(candidates, (-actual_dist, point))
                heapq.heappush(nearest, (actual_dist, point))
                visited.add(point)
                
        while candidates:
            curr_dist, curr_idx = heapq.heappop(candidates)
            curr_dist = -curr_dist
            
            if curr_dist > nearest[0][0]:
                break
                
            node = self.nodes[curr_idx]
            if layer < len(node.neighbors):
                for neighbor_idx in node.neighbors[layer]:
                    if neighbor_idx not in visited and neighbor_idx != idx:
                        visited.add(neighbor_idx)
                        dist = self._compute_distance(idx, neighbor_idx)
                        
                        if dist < nearest[0][0] or len(nearest) < self.config.ef_construction:
                            heapq.heappush(candidates, (-dist, neighbor_idx))
                            heapq.heappush(nearest, (dist, neighbor_idx))
                            
                            if len(nearest) > self.config.ef_construction:
                                heapq.heappop(nearest)
                                
        return nearest
        
    def _select_neighbors_heuristic(self, idx: int, candidates: List[Tuple[float, int]], 
                                   m: int, layer: int) -> List[int]:
        """Select neighbors using a heuristic to maintain connectivity"""
        # Sort by distance
        candidates = sorted(candidates, key=lambda x: x[0])
        
        selected = []
        for dist, candidate_idx in candidates:
            if len(selected) >= m:
                break
                
            # Simple heuristic: always include closest neighbors
            selected.append(candidate_idx)
            
        return selected
        
    def _prune_neighbors(self, idx: int, layer: int):
        """Prune excess neighbors to maintain size constraints"""
        node = self.nodes[idx]
        neighbors = list(node.neighbors[layer])
        
        # Compute distances to all neighbors
        neighbor_dists = [(self._compute_distance(idx, n), n) for n in neighbors]
        neighbor_dists.sort()
        
        # Keep only the closest neighbors
        max_neighbors = self.config.M if layer > 0 else self.config.M * 2
        node.neighbors[layer] = set([n for _, n in neighbor_dists[:max_neighbors]])
        
        # Remove pruned connections from other nodes
        for _, neighbor_idx in neighbor_dists[max_neighbors:]:
            self.nodes[neighbor_idx].neighbors[layer].discard(idx)
            
    def search(self, query: mx.array, k: int, ef: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """
        Search for k nearest neighbors
        Returns: (indices, distances) as MLX arrays
        """
        if self.entry_point is None:
            return mx.array([]), mx.array([])
            
        ef = ef or self.config.ef_search
        
        # Search from top layer to layer 0
        nearest = [(-float('inf'), self.entry_point)]
        
        for layer in range(self.nodes[self.entry_point].level, -1, -1):
            nearest = self._search_layer(query, set([n[1] for n in nearest]), 
                                       ef if layer == 0 else 1, layer)
                                       
        # Extract top k results
        nearest.sort()
        indices = [idx for _, idx in nearest[:k]]
        distances = [dist for dist, _ in nearest[:k]]
        
        return mx.array(indices), mx.array(distances)
        
    def batch_search(self, queries: mx.array, k: int, 
                    ef: Optional[int] = None) -> Tuple[mx.array, mx.array]:
        """
        Batch search for multiple queries
        Returns: (indices, distances) with shape (n_queries, k)
        """
        n_queries = queries.shape[0]
        all_indices = []
        all_distances = []
        
        for i in range(n_queries):
            indices, distances = self.search(queries[i], k, ef)
            all_indices.append(indices)
            all_distances.append(distances)
            
        return mx.stack(all_indices), mx.stack(all_distances)
        
    def save(self, path: str):
        """Save HNSW index to disk"""
        path = Path(path)
        
        # Save configuration
        config_path = path.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)
            
        # Save graph structure
        graph_data = {
            'entry_point': self.entry_point,
            'element_count': self.element_count,
            'nodes': {}
        }
        
        for idx, node in self.nodes.items():
            graph_data['nodes'][idx] = {
                'level': node.level,
                'neighbors': {str(l): list(neighbors) 
                           for l, neighbors in node.neighbors.items()}
            }
            
        graph_path = path.with_suffix('.graph.pkl')
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_data, f)
            
        print(f"HNSW index saved to {path}")
        
    def load(self, path: str):
        """Load HNSW index from disk"""
        path = Path(path)
        
        # Load configuration
        config_path = path.with_suffix('.config.json')
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            self.config = HNSWConfig(**config_dict)
            
        # Load graph structure
        graph_path = path.with_suffix('.graph.pkl')
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
            
        self.entry_point = graph_data['entry_point']
        self.element_count = graph_data['element_count']
        self.nodes = {}
        
        for idx, node_data in graph_data['nodes'].items():
            idx = int(idx)
            node = HNSWNode(idx, node_data['level'])
            
            for level, neighbors in node_data['neighbors'].items():
                level = int(level)
                node.neighbors[level] = set(neighbors)
                
            self.nodes[idx] = node
            
        print(f"HNSW index loaded from {path}")
        
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the index"""
        if not self.nodes:
            return {'status': 'empty'}
            
        total_edges = 0
        layer_stats = {}
        
        for node in self.nodes.values():
            for layer, neighbors in node.neighbors.items():
                if layer not in layer_stats:
                    layer_stats[layer] = {'nodes': 0, 'edges': 0}
                    
                layer_stats[layer]['nodes'] += 1
                layer_stats[layer]['edges'] += len(neighbors)
                total_edges += len(neighbors)
                
        return {
            'total_nodes': len(self.nodes),
            'total_edges': total_edges // 2,  # Edges are bidirectional
            'entry_point': self.entry_point,
            'layers': layer_stats,
            'config': self.config.__dict__
        }
        
    def optimize(self):
        """Optimize the index for better search performance"""
        # Pre-compute and cache frequently accessed distances
        # This is especially useful for MLX's lazy evaluation
        print("Optimizing HNSW index...")
        
        # Force evaluation of all vectors to ensure they're in memory
        mx.eval(self.vectors)
        
        # Pre-normalize vectors if using cosine distance
        if self.config.distance_type == "cosine":
            norms = mx.sqrt(mx.sum(self.vectors * self.vectors, axis=1))
            norms = mx.maximum(norms, 1e-8)
            self.normalized_vectors = self.vectors / norms[:, None]
            mx.eval(self.normalized_vectors)
            
        print("Optimization complete!")


# Integration with VectorStore
class HNSWVectorStore:
    """Wrapper to integrate HNSW with existing VectorStore"""
    
    def __init__(self, vector_store, config: HNSWConfig = HNSWConfig()):
        self.vector_store = vector_store
        self.config = config
        self.index: Optional[HNSWIndex] = None
        
    def build_index(self, force_rebuild: bool = False):
        """Build or rebuild HNSW index"""
        if self.vector_store.vectors is None:
            return
            
        index_path = self.vector_store.store_path / "hnsw_index"
        
        if not force_rebuild and index_path.with_suffix('.graph.pkl').exists():
            # Load existing index
            self.index = HNSWIndex(self.config)
            self.index.load(str(index_path))
            self.index.vectors = self.vector_store.vectors
        else:
            # Build new index
            self.index = HNSWIndex(self.config)
            self.index.build(self.vector_store.vectors)
            self.index.save(str(index_path))
            
    def query(self, query_vector: mx.array, k: int = 10) -> Tuple[mx.array, mx.array]:
        """Query using HNSW index"""
        if self.index is None:
            # Fallback to brute force
            return self.vector_store._brute_force_search(query_vector, k)
            
        return self.index.search(query_vector, k)
        
    def add_vectors(self, vectors: mx.array, metadata: List[Dict]):
        """Add vectors and update index"""
        # Add to vector store
        start_idx = len(self.vector_store.vectors) if self.vector_store.vectors is not None else 0
        self.vector_store.add_vectors(vectors, metadata)
        
        # Update HNSW index
        if self.index is not None:
            # Add new vectors to index
            for i in range(vectors.shape[0]):
                self.index._insert(start_idx + i)
                
            # Save updated index
            index_path = self.vector_store.store_path / "hnsw_index"
            self.index.save(str(index_path))


# Demo usage
if __name__ == "__main__":
    # Create random vectors for testing
    n_vectors = 10000
    dim = 384
    
    print(f"Creating {n_vectors} random vectors of dimension {dim}...")
    vectors = mx.random.normal((n_vectors, dim))
    
    # Build HNSW index
    config = HNSWConfig(M=16, ef_construction=200, ef_search=50, distance_type="cosine")
    index = HNSWIndex(config)
    
    start_time = time.time()
    index.build(vectors)
    build_time = time.time() - start_time
    
    print(f"Index built in {build_time:.2f} seconds")
    print(f"Index stats: {index.get_stats()}")
    
    # Perform search
    query = mx.random.normal((dim,))
    k = 10
    
    start_time = time.time()
    indices, distances = index.search(query, k)
    search_time = time.time() - start_time
    
    print(f"\nSearch completed in {search_time*1000:.2f} ms")
    print(f"Found {len(indices)} nearest neighbors")
    print(f"Indices: {indices}")
    print(f"Distances: {distances}")
    
    # Save and load test
    index.save("test_index")
    
    new_index = HNSWIndex(config)
    new_index.load("test_index")
    new_index.vectors = vectors
    
    # Verify loaded index works
    indices2, distances2 = new_index.search(query, k)
    print(f"\nLoaded index produces same results: {mx.array_equal(indices, indices2)}")
