# performance/hnsw_index.py
"""
Hierarchical Navigable Small World (HNSW) Index for MLX Vector Database
Provides O(log n) approximate nearest neighbor search
"""
import math
import random
import logging
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import mlx.core as mx
# import mlx.core.linalg as mxl  # Falls nicht verfügbar, nutzen wir mx direkt
from dataclasses import dataclass
from pathlib import Path
import pickle
import time

logger = logging.getLogger("mlx_vector_db.hnsw")

@dataclass
class HNSWNode:
    """Node in the HNSW graph"""
    vector_id: int
    vector: mx.array
    level: int
    connections: Dict[int, Set[int]]  # level -> set of connected node IDs
    
    def __post_init__(self):
        if not self.connections:
            self.connections = {i: set() for i in range(self.level + 1)}

class HNSWIndex:
    """
    HNSW Index for fast approximate nearest neighbor search
    
    Based on the paper: "Efficient and robust approximate nearest neighbor 
    search using Hierarchical Navigable Small World graphs"
    """
    
    def __init__(
        self,
        dimension: int,
        max_connections: int = 16,
        max_connections_layer0: int = 32,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: float = 1.0 / math.log(2)
    ):
        self.dimension = dimension
        self.max_connections = max_connections  # M
        self.max_connections_layer0 = max_connections_layer0  # Mmax
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml  # level generation factor
        
        self.nodes: Dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.node_count = 0
        
        logger.info(f"HNSW Index initialized: dim={dimension}, M={max_connections}, ef_construction={ef_construction}")
    
    def _select_level(self) -> int:
        """Select level for new node using exponential decay"""
        level = int(-math.log(random.uniform(0, 1)) * self.ml)
        return level
    
    @mx.compile
    def _compute_distance(self, vec1: mx.array, vec2: mx.array) -> float:
        """Compiled cosine distance computation for performance"""
        # Normalize vectors
        norm1 = mx.sqrt(mx.sum(vec1 * vec1))
        norm2 = mx.sqrt(mx.sum(vec2 * vec2))
        
        # Avoid division by zero
        norm1 = mx.maximum(norm1, 1e-10)
        norm2 = mx.maximum(norm2, 1e-10)
        
        normalized1 = vec1 / norm1
        normalized2 = vec2 / norm2
        
        # Cosine similarity -> distance
        similarity = mx.sum(normalized1 * normalized2)
        distance = 1.0 - similarity
        
        return distance.item()
    
    def _search_layer(
        self, 
        query: mx.array, 
        entry_points: Set[int], 
        num_closest: int, 
        level: int
    ) -> List[Tuple[float, int]]:
        """Search for closest nodes in a specific layer"""
        visited = set()
        candidates = []
        dynamic_list = []
        
        # Initialize with entry points
        for ep_id in entry_points:
            if ep_id in self.nodes:
                distance = self._compute_distance(query, self.nodes[ep_id].vector)
                candidates.append((distance, ep_id))
                dynamic_list.append((distance, ep_id))
                visited.add(ep_id)
        
        candidates.sort()
        dynamic_list.sort(reverse=True)  # Keep worst candidates at front
        
        while candidates:
            current_dist, current_id = candidates.pop(0)
            
            # If current distance is worse than worst in dynamic list, stop
            if dynamic_list and current_dist > dynamic_list[0][0]:
                break
            
            # Explore neighbors
            current_node = self.nodes[current_id]
            neighbors = current_node.connections.get(level, set())
            
            for neighbor_id in neighbors:
                if neighbor_id not in visited and neighbor_id in self.nodes:
                    visited.add(neighbor_id)
                    distance = self._compute_distance(query, self.nodes[neighbor_id].vector)
                    
                    # Add to dynamic list if it's better or list is not full
                    if len(dynamic_list) < num_closest:
                        dynamic_list.append((distance, neighbor_id))
                        candidates.append((distance, neighbor_id))
                        dynamic_list.sort(reverse=True)
                        candidates.sort()
                    elif distance < dynamic_list[0][0]:
                        dynamic_list[0] = (distance, neighbor_id)
                        candidates.append((distance, neighbor_id))
                        dynamic_list.sort(reverse=True)
                        candidates.sort()
        
        # Return closest nodes (best first)
        dynamic_list.sort()
        return dynamic_list[:num_closest]
    
    def _select_neighbors_heuristic(
        self, 
        candidates: List[Tuple[float, int]], 
        max_connections: int
    ) -> Set[int]:
        """Select neighbors using heuristic to maintain connectivity"""
        if len(candidates) <= max_connections:
            return {node_id for _, node_id in candidates}
        
        # Sort by distance (closest first)
        candidates.sort()
        
        selected = set()
        
        # Always include closest
        if candidates:
            selected.add(candidates[0][1])
            candidates = candidates[1:]
        
        # Greedily select diverse neighbors
        while len(selected) < max_connections and candidates:
            best_candidate = candidates.pop(0)
            selected.add(best_candidate[1])
        
        return selected
    
    def add_vector(self, vector_id: int, vector: mx.array) -> None:
        """Add a new vector to the index"""
        if not isinstance(vector, mx.array):
            vector = mx.array(vector)
        
        if vector.shape[-1] != self.dimension:
            raise ValueError(f"Vector dimension {vector.shape[-1]} != index dimension {self.dimension}")
        
        # Flatten vector if needed
        if vector.ndim > 1:
            vector = vector.flatten()
        
        level = self._select_level()
        node = HNSWNode(vector_id, vector, level, {})
        
        # If this is the first node, make it the entry point
        if self.entry_point is None:
            self.entry_point = vector_id
            self.nodes[vector_id] = node
            self.node_count += 1
            logger.debug(f"Added first node {vector_id} as entry point at level {level}")
            return
        
        # Search for closest nodes at each level
        entry_points = {self.entry_point}
        
        # Search from top level down to level+1
        for lev in range(self.nodes[self.entry_point].level, level, -1):
            entry_points = {node_id for _, node_id in self._search_layer(vector, entry_points, 1, lev)}
        
        # Search and connect at each level from level down to 0
        for lev in range(min(level, self.nodes[self.entry_point].level), -1, -1):
            candidates = self._search_layer(vector, entry_points, self.ef_construction, lev)
            
            # Select neighbors
            max_conn = self.max_connections_layer0 if lev == 0 else self.max_connections
            neighbors = self._select_neighbors_heuristic(candidates, max_conn)
            
            # Add bidirectional connections
            node.connections[lev] = neighbors
            for neighbor_id in neighbors:
                if neighbor_id in self.nodes:
                    self.nodes[neighbor_id].connections[lev].add(vector_id)
                    
                    # Prune connections if necessary
                    if len(self.nodes[neighbor_id].connections[lev]) > max_conn:
                        # Recompute neighbors for this node
                        neighbor_candidates = []
                        for connected_id in self.nodes[neighbor_id].connections[lev]:
                            if connected_id in self.nodes:
                                dist = self._compute_distance(
                                    self.nodes[neighbor_id].vector,
                                    self.nodes[connected_id].vector
                                )
                                neighbor_candidates.append((dist, connected_id))
                        
                        new_neighbors = self._select_neighbors_heuristic(neighbor_candidates, max_conn)
                        
                        # Update connections
                        old_connections = self.nodes[neighbor_id].connections[lev]
                        self.nodes[neighbor_id].connections[lev] = new_neighbors
                        
                        # Remove bidirectional connections for pruned neighbors
                        for removed_id in old_connections - new_neighbors:
                            if removed_id in self.nodes and lev in self.nodes[removed_id].connections:
                                self.nodes[removed_id].connections[lev].discard(neighbor_id)
            
            entry_points = neighbors
        
        # Update entry point if necessary
        if level > self.nodes[self.entry_point].level:
            self.entry_point = vector_id
        
        self.nodes[vector_id] = node
        self.node_count += 1
        
        if self.node_count % 1000 == 0:
            logger.info(f"HNSW index now contains {self.node_count} nodes")
    
    def search(
        self, 
        query: mx.array, 
        k: int = 10, 
        ef: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Search for k nearest neighbors"""
        if not isinstance(query, mx.array):
            query = mx.array(query)
        
        if query.shape[-1] != self.dimension:
            raise ValueError(f"Query dimension {query.shape[-1]} != index dimension {self.dimension}")
        
        # Flatten query if needed
        if query.ndim > 1:
            query = query.flatten()
        
        if self.entry_point is None:
            return []
        
        ef = ef or max(self.ef_search, k)
        
        # Search from top level down to level 1
        entry_points = {self.entry_point}
        for lev in range(self.nodes[self.entry_point].level, 0, -1):
            entry_points = {node_id for _, node_id in self._search_layer(query, entry_points, 1, lev)}
        
        # Search at level 0 with ef
        candidates = self._search_layer(query, entry_points, ef, 0)
        
        # Return top k results as (vector_id, distance)
        return [(node_id, distance) for distance, node_id in candidates[:k]]
    
    def batch_search(
        self, 
        queries: mx.array, 
        k: int = 10, 
        ef: Optional[int] = None
    ) -> List[List[Tuple[int, float]]]:
        """Search for multiple queries"""
        if not isinstance(queries, mx.array):
            queries = mx.array(queries)
        
        if queries.ndim == 1:
            return [self.search(queries, k, ef)]
        
        results = []
        for i in range(queries.shape[0]):
            query = queries[i]
            results.append(self.search(query, k, ef))
        
        return results
    
    def save(self, file_path: Path) -> None:
        """Save HNSW index to disk"""
        # Convert MLX arrays to numpy for pickling
        save_data = {
            'dimension': self.dimension,
            'max_connections': self.max_connections,
            'max_connections_layer0': self.max_connections_layer0,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'ml': self.ml,
            'entry_point': self.entry_point,
            'node_count': self.node_count,
            'nodes': {}
        }
        
        # Convert nodes
        for node_id, node in self.nodes.items():
            save_data['nodes'][node_id] = {
                'vector_id': node.vector_id,
                'vector': np.array(node.vector),  # Convert to numpy
                'level': node.level,
                'connections': node.connections
            }
        
        with open(file_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"HNSW index saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: Path) -> 'HNSWIndex':
        """Load HNSW index from disk"""
        with open(file_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Create new index
        index = cls(
            dimension=save_data['dimension'],
            max_connections=save_data['max_connections'],
            max_connections_layer0=save_data['max_connections_layer0'],
            ef_construction=save_data['ef_construction'],
            ef_search=save_data['ef_search'],
            ml=save_data['ml']
        )
        
        index.entry_point = save_data['entry_point']
        index.node_count = save_data['node_count']
        
        # Convert nodes back
        for node_id, node_data in save_data['nodes'].items():
            vector = mx.array(node_data['vector'])  # Convert back to MLX
            node = HNSWNode(
                vector_id=node_data['vector_id'],
                vector=vector,
                level=node_data['level'],
                connections=node_data['connections']
            )
            index.nodes[node_id] = node
        
        logger.info(f"HNSW index loaded from {file_path} with {index.node_count} nodes")
        return index
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.nodes:
            return {"nodes": 0, "levels": 0, "avg_connections": 0}
        
        levels = [node.level for node in self.nodes.values()]
        connections_count = []
        
        for node in self.nodes.values():
            total_connections = sum(len(conns) for conns in node.connections.values())
            connections_count.append(total_connections)
        
        return {
            "nodes": len(self.nodes),
            "max_level": max(levels) if levels else 0,
            "avg_level": sum(levels) / len(levels) if levels else 0,
            "avg_connections": sum(connections_count) / len(connections_count) if connections_count else 0,
            "entry_point": self.entry_point,
            "dimension": self.dimension
        }