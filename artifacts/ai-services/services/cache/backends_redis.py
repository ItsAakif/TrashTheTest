"""
Redis Cache Backend

Production-ready Redis cache implementation.
"""

import json
import asyncio
from datetime import datetime
from typing import Optional, Any, Dict, List
import logging

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .backends import CacheBackend, CacheBackendError

logger = logging.getLogger(__name__)


class RedisCacheBackend(CacheBackend):
    """
    Redis-based cache implementation.
    
    Features:
    - Persistent caching with Redis
    - TTL support
    - Pub/sub for cache invalidation
    - Cluster support
    """
    
    def __init__(
        self, 
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "ttt_cache",
        decode_responses: bool = True
    ):
        if not REDIS_AVAILABLE:
            raise ImportError(
                "redis package not installed. "
                "Install with: pip install redis"
            )
        
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._decode_responses = decode_responses
        self._client: Optional["redis.Redis"] = None
        self._hits = 0
        self._misses = 0
    
    async def _get_client(self) -> "redis.Redis":
        """Get or create Redis client"""
        if self._client is None:
            self._client = await redis.from_url(
                self._redis_url,
                decode_responses=self._decode_responses,
                encoding="utf-8"
            )
        return self._client
    
    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self._key_prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            
            value = await client.get(full_key)
            
            if value is None:
                self._misses += 1
                logger.debug(f"Cache miss: {key}")
                return None
            
            self._hits += 1
            logger.debug(f"Cache hit: {key}")
            
            # Try to parse JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._misses += 1
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set a value in cache"""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            
            # Serialize value to JSON
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                serialized = str(value)
            
            if ttl:
                await client.setex(full_key, ttl, serialized)
            else:
                await client.set(full_key, serialized)
            
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            
            result = await client.delete(full_key)
            logger.debug(f"Cache deleted: {key}")
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists"""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            
            return await client.exists(full_key) > 0
            
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern"""
        try:
            client = await self._get_client()
            
            # Convert pattern to Redis pattern
            full_pattern = self._make_key(pattern)
            
            keys = []
            async for key in client.scan_iter(match=full_pattern):
                # Remove prefix
                key = key.replace(f"{self._key_prefix}:", "")
                keys.append(key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []
    
    async def invalidate_by_prefix(self, prefix: str) -> int:
        """Invalidate all keys with a prefix"""
        try:
            client = await self._get_client()
            full_pattern = self._make_key(prefix)
            
            # Get all matching keys
            keys_to_delete = []
            async for key in client.scan_iter(match=full_pattern):
                keys_to_delete.append(key)
            
            if keys_to_delete:
                await client.delete(*keys_to_delete)
            
            logger.info(f"Invalidated {len(keys_to_delete)} keys with prefix: {prefix}")
            return len(keys_to_delete)
            
        except Exception as e:
            logger.error(f"Redis invalidate error: {e}")
            return 0
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            client = await self._get_client()
            
            # Get all keys with our prefix
            keys = []
            async for key in client.scan_iter(match=f"{self._key_prefix}:*"):
                keys.append(key)
            
            if keys:
                await client.delete(*keys)
            
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            client = await self._get_client()
            
            # Get keys with our prefix
            total_keys = 0
            async for _ in client.scan_iter(match=f"{self._key_prefix}:*"):
                total_keys += 1
            
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            # Try to get Redis info
            info = {}
            try:
                info = await client.info("memory")
            except Exception:
                pass
            
            return {
                "backend": "redis",
                "url": self._redis_url,
                "size": total_keys,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "used_memory": info.get("used_memory_human", "unknown"),
            }
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"backend": "redis", "error": str(e)}
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self._client:
            await self._client.close()
            self._client = None


# Global instance
_redis_cache: Optional[RedisCacheBackend] = None


def get_redis_cache(redis_url: str = "redis://localhost:6379/0") -> RedisCacheBackend:
    """Get or create global Redis cache instance"""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCacheBackend(redis_url=redis_url)
    return _redis_cache