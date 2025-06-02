# performance/vector_cache.py
"""
High-performance caching system for MLX Vector Database
Provides LRU caching with memory management
"""
import os
import time
import threading
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field
import mlx.core as mx
import numpy as np
from collections import OrderedDict

logger = logging.getLogger("mlx_vector_db.cache")

@dataclass
class CacheEntry:
    """Entry in the vector cache"""
    vectors: mx.array
    metadata: list
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    
    def __post_init__(self):
        if self.size_bytes == 0:
            # Estimate memory usage
            vector_size = self.vectors.nbytes if hasattr(self.vectors, 'nbytes') else 0
            metadata_size = len(str(self.metadata).encode('utf-8'))
            self.size_bytes = vector_size + metadata_size
    
    def touch(self):
        """Update access timestamp and count"""
        self.timestamp = time.time()
        self.access_count += 1

class VectorStoreCache:
    """
    High-performance LRU cache for vector stores with memory management
    """
    
    def __init__(self, max_memory_gb: float = 4.0, cleanup_threshold: float = 0.9):
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.cleanup_threshold = cleanup_threshold
        self.current_memory_bytes = 0
        
        # Thread-safe cache storage
        self._cache: Dict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"VectorStoreCache initialized: max_memory={max_memory_gb}GB")
    
    def _get_cache_key(self, user_id: str, model_name: str) -> str:
        """Generate cache key from user_id and model_name"""
        return f"{user_id}::{model_name}"
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage"""
        return sum(entry.size_bytes for entry in self._cache.values())
    
    def _cleanup_if_needed(self):
        """Cleanup cache if memory usage is too high"""
        current_usage = self._estimate_memory_usage()
        
        if current_usage < self.max_memory_bytes * self.cleanup_threshold:
            return
        
        target_size = int(self.max_memory_bytes * 0.7)  # Clean to 70%
        
        logger.info(f"Cache cleanup triggered: {current_usage / 1024**3:.2f}GB -> target {target_size / 1024**3:.2f}GB")
        
        # Sort by LRU (least recently used first)
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: (x[1].timestamp, x[1].access_count)
        )
        
        bytes_freed = 0
        keys_to_remove = []
        
        for key, entry in sorted_items:
            if current_usage - bytes_freed <= target_size:
                break
            
            bytes_freed += entry.size_bytes
            keys_to_remove.append(key)
            self.evictions += 1
        
        # Remove entries
        for key in keys_to_remove:
            del self._cache[key]
        
        logger.info(f"Cache cleanup completed: removed {len(keys_to_remove)} entries, freed {bytes_freed / 1024**3:.2f}GB")
    
    def get(self, user_id: str, model_name: str) -> Optional[Tuple[mx.array, list]]:
        """Get vectors and metadata from cache"""
        cache_key = self._get_cache_key(user_id, model_name)
        
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.touch()
                
                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)
                
                self.hits += 1
                logger.debug(f"Cache HIT for {user_id}/{model_name}")
                return entry.vectors, entry.metadata
            
            self.misses += 1
            logger.debug(f"Cache MISS for {user_id}/{model_name}")
            return None
    
    def put(self, user_id: str, model_name: str, vectors: mx.array, metadata: list):
        """Store vectors and metadata in cache"""
        cache_key = self._get_cache_key(user_id, model_name)
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                vectors=vectors,
                metadata=metadata,
                timestamp=time.time()
            )
            
            # Check if adding this entry would exceed memory
            if entry.size_bytes > self.max_memory_bytes:
                logger.warning(f"Entry too large for cache: {entry.size_bytes / 1024**3:.2f}GB")
                return
            
            # Remove existing entry if it exists
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            # Add new entry
            self._cache[cache_key] = entry
            
            # Cleanup if needed
            self._cleanup_if_needed()
            
            logger.debug(f"Cached {user_id}/{model_name}: {vectors.shape} vectors, {entry.size_bytes / 1024**2:.1f}MB")
    
    def invalidate(self, user_id: str, model_name: str):
        """Remove entry from cache"""
        cache_key = self._get_cache_key(user_id, model_name)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"Invalidated cache for {user_id}/{model_name}")
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self.current_memory_bytes = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            current_memory = self._estimate_memory_usage()
            
            return {
                "entries": len(self._cache),
                "memory_usage_gb": current_memory / 1024**3,
                "memory_usage_percent": (current_memory / self.max_memory_bytes * 100),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate_percent": hit_rate,
                "evictions": self.evictions,
                "max_memory_gb": self.max_memory_bytes / 1024**3
            }
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics including per-store info"""
        with self._lock:
            stats = self.get_stats()
            
            # Add per-store information
            stores = []
            for key, entry in self._cache.items():
                user_id, model_name = key.split("::", 1)
                stores.append({
                    "user_id": user_id,
                    "model_name": model_name,
                    "vector_count": entry.vectors.shape[0] if hasattr(entry.vectors, 'shape') else 0,
                    "size_mb": entry.size_bytes / 1024**2,
                    "access_count": entry.access_count,
                    "last_accessed": entry.timestamp
                })
            
            # Sort by last accessed (most recent first)
            stores.sort(key=lambda x: x["last_accessed"], reverse=True)
            
            stats["stores"] = stores
            return stats

# Global cache instance
_global_cache: Optional[VectorStoreCache] = None

def get_global_cache() -> VectorStoreCache:
    """Get or create global cache instance"""
    global _global_cache
    
    if _global_cache is None:
        max_memory = float(os.getenv("MAX_VECTOR_CACHE_SIZE_GB", "4.0"))
        _global_cache = VectorStoreCache(max_memory_gb=max_memory)
    
    return _global_cache

def invalidate_cache(user_id: str, model_name: str):
    """Convenience function to invalidate cache"""
    cache = get_global_cache()
    cache.invalidate(user_id, model_name)