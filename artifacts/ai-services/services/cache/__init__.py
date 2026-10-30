"""
Cache Service

Provides distributed caching with in-memory fallback.
"""

import time
import hashlib
import json
import logging
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStatus(str, Enum):
    """Cache health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class CacheHealth:
    """Cache health information"""
    status: str
    backend: str
    hits: int
    misses: int
    size: int


class CacheService:
    """
    Multi-backend caching service with in-memory and Redis support.
    
    Features:
    - In-memory LRU cache (always available)
    - Redis support for distributed caching
    - TTL support
    - Cache key generation
    - Hit/miss statistics
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self._memory_cache: dict[str, dict] = {}
        self._redis_url = redis_url
        self._redis_client = None
        self._hits = 0
        self._misses = 0
        
        # Initialize Redis if available
        if redis_url:
            try:
                import redis.asyncio as redis
                self._redis_client = redis.from_url(redis_url)
                logger.info(f"Redis cache initialized: {redis_url}")
            except ImportError:
                logger.warning("Redis not installed, using in-memory cache only")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using in-memory cache")
    
    async def get_health(self) -> CacheHealth:
        """Get cache health status"""
        backend = "redis" if self._redis_client else "memory"
        
        try:
            if self._redis_client:
                await self._redis_client.ping()
                status = CacheStatus.HEALTHY
            else:
                status = CacheStatus.HEALTHY
        except Exception:
            status = CacheStatus.DEGRADED if self._redis_client else CacheStatus.HEALTHY
        
        return CacheHealth(
            status=status.value,
            backend=backend,
            hits=self._hits,
            misses=self._misses,
            size=len(self._memory_cache)
        )
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key"""
        key_parts = [prefix]
        
        # Add positional args
        for arg in args:
            if isinstance(arg, str):
                key_parts.append(arg)
            elif isinstance(arg, (int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
        
        # Add kwargs (sorted for consistency)
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            key_parts.append(f"{k}={v}")
        
        # Create key with prefix
        key = ":".join(key_parts)
        
        # Hash if too long
        if len(key) > 200:
            key = f"{prefix}:{hashlib.md5(key.encode()).hexdigest()}"
        
        return key
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try Redis first
        if self._redis_client:
            try:
                value = await self._redis_client.get(key)
                if value:
                    self._hits += 1
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fallback to memory
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if entry["expires"] > time.time():
                self._hits += 1
                return entry["value"]
            else:
                # Expired
                del self._memory_cache[key]
        
        self._misses += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        if ttl is None:
            ttl = 3600  # Default 1 hour
        
        # Store in memory
        self._memory_cache[key] = {
            "value": value,
            "expires": time.time() + ttl,
            "created": time.time()
        }
        
        # Store in Redis if available
        if self._redis_client:
            try:
                serialized = json.dumps(value)
                await self._redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        # Remove from Redis
        if self._redis_client:
            try:
                await self._redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        # Check memory
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if entry["expires"] > time.time():
                return True
            del self._memory_cache[key]
        
        # Check Redis
        if self._redis_client:
            try:
                return await self._redis_client.exists(key) > 0
            except Exception:
                pass
        
        return False
    
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries"""
        count = 0
        
        if pattern:
            # Clear matching keys
            keys = [k for k in self._memory_cache.keys() if pattern in k]
            for key in keys:
                del self._memory_cache[key]
                count += 1
            
            if self._redis_client:
                try:
                    redis_keys = await self._redis_client.keys(pattern)
                    if redis_keys:
                        await self._redis_client.delete(*redis_keys)
                        count += len(redis_keys)
                except Exception as e:
                    logger.warning(f"Redis clear failed: {e}")
        else:
            # Clear all
            count = len(self._memory_cache)
            self._memory_cache.clear()
            
            if self._redis_client:
                try:
                    await self._redis_client.flushdb()
                except Exception as e:
                    logger.warning(f"Redis flush failed: {e}")
        
        return count
    
    async def get_stats(self) -> dict:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "size": len(self._memory_cache),
            "backend": "redis" if self._redis_client else "memory"
        }
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from memory cache"""
        current_time = time.time()
        expired_keys = [
            k for k, v in self._memory_cache.items()
            if v["expires"] <= current_time
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        return len(expired_keys)


# Global cache instance
_cache_service: Optional[CacheService] = None


async def initialize_cache(redis_url: Optional[str] = None) -> CacheService:
    """Initialize the cache service"""
    global _cache_service
    
    # Get Redis URL from settings if not provided
    if redis_url is None:
        try:
            from config.settings import get_settings
            settings = get_settings()
            redis_url = settings.cache.redis_url if hasattr(settings, 'cache') else None
        except Exception:
            pass
    
    _cache_service = CacheService(redis_url)
    return _cache_service


def get_cache_service() -> CacheService:
    """Get the global cache service instance"""
    if _cache_service is None:
        raise RuntimeError("Cache service not initialized. Call initialize_cache() first.")
    return _cache_service


__all__ = [
    "CacheService",
    "get_cache_service",
    "initialize_cache",
    "CacheType",
    "CacheTTL",
    "CacheConfig",
]


def __getattr__(name: str):
    # Re-export cache types without disturbing the legacy service above.
    if name in ("CacheType", "CacheTTL", "CacheConfig", "CacheEntry", "CacheStats"):
        from . import types as _types

        return getattr(_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")