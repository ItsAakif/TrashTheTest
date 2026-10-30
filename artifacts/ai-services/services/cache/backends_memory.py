"""
In-Memory Cache Backend

Simple LRU cache implementation for development.
"""

import asyncio
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List
import logging

from .backends import CacheBackend, CacheBackendError
from .types import CacheType

logger = logging.getLogger(__name__)


class MemoryCacheBackend(CacheBackend):
    """
    In-memory LRU cache implementation.
    
    Features:
    - LRU eviction when max_size is reached
    - TTL support
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                logger.debug(f"Cache miss: {key}")
                return None
            
            # Check if expired
            if entry.get("expires_at") and entry["expires_at"] < datetime.utcnow():
                del self._cache[key]
                self._misses += 1
                logger.debug(f"Cache expired: {key}")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug(f"Cache hit: {key}")
            return entry.get("value")
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set a value in cache"""
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Evicted (LRU): {oldest_key}")
            
            # Calculate expiration
            expires_at = None
            if ttl:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            self._cache[key] = {
                "value": value,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at,
                "metadata": metadata or {},
            }
            
            # Move to end
            self._cache.move_to_end(key)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Cache deleted: {key}")
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            
            # Check expiration
            if entry.get("expires_at") and entry["expires_at"] < datetime.utcnow():
                del self._cache[key]
                return False
            
            return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern"""
        async with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            
            # Simple prefix matching
            prefix = pattern.rstrip("*")
            return [k for k in self._cache.keys() if k.startswith(prefix)]
    
    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all keys with a prefix"""
        async with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            
            for key in keys_to_delete:
                del self._cache[key]
            
            logger.info(f"Invalidated {len(keys_to_delete)} keys with prefix: {prefix}")
            return len(keys_to_delete)
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")
            return True
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "backend": "memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }
    
    async def health_check(self) -> bool:
        """Check if cache is healthy"""
        return True


# Global instance
_memory_cache: Optional[MemoryCacheBackend] = None


def get_memory_cache(max_size: int = 10000) -> MemoryCacheBackend:
    """Get or create global memory cache instance"""
    global _memory_cache
    if _memory_cache is None:
        _memory_cache = MemoryCacheBackend(max_size=max_size)
    return _memory_cache