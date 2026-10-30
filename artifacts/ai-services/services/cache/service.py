"""
Cache Service

Central caching layer for TrashTheTest.
Provides consistent interface for caching across all modules.
"""

import uuid
import logging
from typing import Optional, Any, Dict, List, Callable, Awaitable

from .backends import CacheBackend
from .backends_memory import get_memory_cache
from .backends_redis import get_redis_cache, RedisCacheBackend
from .types import (
    CacheType,
    CacheConfig,
    CacheStats,
    CacheHealthResponse,
    InvalidationRequest,
    InvalidationResponse,
    get_ttl_for_type,
    KEY_PREFIXES,
)
from . import keys as cache_keys

logger = logging.getLogger(__name__)


class CacheService:
    """
    Central cache service for the application.
    
    Features:
    - Unified interface for different cache backends
    - Automatic TTL management by cache type
    - Cache invalidation by document/project
    - Observability (hits/misses logging)
    - Fallback to computation on cache failure
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self._config = config or CacheConfig()
        self._backend: Optional[CacheBackend] = None
        self._stats: Dict[CacheType, Dict[str, int]] = {
            ct: {"hits": 0, "misses": 0} for ct in CacheType
        }
    
    async def initialize(self):
        """Initialize the cache backend"""
        if self._config.backend == "redis":
            try:
                self._backend = get_redis_cache(self._config.redis_url or "redis://localhost:6379/0")
                # Test connection
                if await self._backend.health_check():
                    logger.info("Cache initialized with Redis backend")
                else:
                    logger.warning("Redis health check failed, falling back to memory")
                    self._backend = get_memory_cache()
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}, falling back to memory")
                self._backend = get_memory_cache()
        else:
            self._backend = get_memory_cache(self._config.max_size or 10000)
            logger.info("Cache initialized with in-memory backend")
    
    @property
    def backend(self) -> CacheBackend:
        """Get the current cache backend"""
        if self._backend is None:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        return self._backend
    
    # ==================== Basic Operations ====================
    
    async def get(
        self, 
        key: str, 
        cache_type: Optional[CacheType] = None,
        fallback: Optional[Callable[[], Awaitable[Any]]] = None
    ) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            cache_type: Type of cached data (for stats)
            fallback: Optional async function to compute value if not in cache
            
        Returns:
            Cached value or fallback result
        """
        try:
            value = await self.backend.get(key)
            
            if value is not None and cache_type:
                self._stats[cache_type]["hits"] += 1
                logger.debug(f"Cache hit: {key} ({cache_type.value})")
                return value
            
            if value is None and cache_type:
                self._stats[cache_type]["misses"] += 1
                logger.debug(f"Cache miss: {key} ({cache_type.value})")
            
            # Execute fallback if provided
            if value is None and fallback:
                logger.debug(f"Computing fallback for: {key}")
                value = await fallback()
                # Cache the computed value
                if value is not None and cache_type:
                    await self.set(key, value, cache_type)
                return value
            
            return value
            
        except Exception as e:
            logger.error(f"Cache get error for {key}: {e}")
            # Fallback to computation on error
            if fallback:
                return await fallback()
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        cache_type: Optional[CacheType] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_type: Type of cached data (for TTL defaults)
            ttl: Optional TTL override in seconds
            
        Returns:
            True if successful
        """
        try:
            # Determine TTL
            if ttl is None and cache_type:
                ttl = get_ttl_for_type(cache_type)
            
            return await self.backend.set(key, value, ttl=ttl)
            
        except Exception as e:
            logger.error(f"Cache set error for {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        try:
            return await self.backend.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error for {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        try:
            return await self.backend.exists(key)
        except Exception as e:
            logger.error(f"Cache exists error for {key}: {e}")
            return False
    
    # ==================== Convenience Methods ====================
    
    async def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Awaitable[Any]],
        cache_type: CacheType,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or compute and cache.
        
        This is the primary interface for using the cache.
        
        Args:
            key: Cache key
            compute_fn: Async function to compute value if not cached
            cache_type: Type of data (determines TTL)
            ttl: Optional TTL override
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached = await self.get(key, cache_type=cache_type)
        if cached is not None:
            return cached
        
        # Compute value
        value = await compute_fn()
        
        # Cache the result
        if value is not None:
            await self.set(key, value, cache_type=cache_type, ttl=ttl)
        
        return value
    
    # ==================== Embedding Cache ====================
    
    async def get_embedding(self, content: str) -> Optional[List[float]]:
        """Get cached embedding for content"""
        key = cache_keys.embedding_cache_key(content)
        return await self.get(key, cache_type=CacheType.EMBEDDING)
    
    async def set_embedding(self, content: str, embedding: List[float]) -> bool:
        """Cache an embedding"""
        key = cache_keys.embedding_cache_key(content)
        return await self.set(key, embedding, cache_type=CacheType.EMBEDDING)
    
    async def get_embedding_by_hash(self, content_hash: str) -> Optional[List[float]]:
        """Get cached embedding by content hash"""
        key = cache_keys.embedding_cache_key_from_hash(content_hash)
        return await self.get(key, cache_type=CacheType.EMBEDDING)
    
    async def set_embedding_by_hash(self, content_hash: str, embedding: List[float]) -> bool:
        """Cache embedding by content hash"""
        key = cache_keys.embedding_cache_key_from_hash(content_hash)
        return await self.set(key, embedding, cache_type=CacheType.EMBEDDING)
    
    # ==================== Generated Output Cache ====================
    
    async def get_generated_output(
        self,
        chunk_hashes: List[str],
        mode: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached generated output"""
        key = cache_keys.generated_output_cache_key(chunk_hashes, mode, config)
        return await self.get(key, cache_type=CacheType.GENERATED_OUTPUT)
    
    async def set_generated_output(
        self,
        chunk_hashes: List[str],
        mode: str,
        output: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Cache generated output"""
        key = cache_keys.generated_output_cache_key(chunk_hashes, mode, config)
        return await self.set(key, output, cache_type=CacheType.GENERATED_OUTPUT)
    
    # ==================== Research Cache ====================
    
    async def get_search_result(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        key = cache_keys.research_cache_key(query, filters)
        return await self.get(key, cache_type=CacheType.RESEARCH_RESULT)
    
    async def set_search_result(
        self,
        query: str,
        results: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Cache search results"""
        key = cache_keys.research_cache_key(query, filters)
        return await self.set(key, results, cache_type=CacheType.RESEARCH_RESULT)
    
    async def get_url_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached URL extraction"""
        key = cache_keys.research_url_cache_key(url)
        return await self.get(key, cache_type=CacheType.RESEARCH_RESULT)
    
    async def set_url_content(self, url: str, content: Dict[str, Any]) -> bool:
        """Cache URL extraction"""
        key = cache_keys.research_url_cache_key(url)
        return await self.set(key, content, cache_type=CacheType.RESEARCH_RESULT)
    
    # ==================== Invalidation ====================
    
    async def invalidate_document(self, document_id: uuid.UUID) -> int:
        """
        Invalidate all cache entries for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        
        # Invalidate embeddings (by document prefix)
        doc_prefix = f"emb:{str(document_id)[:8]}"
        invalidated += await self.backend.invalidate_by_prefix(doc_prefix)
        
        # Invalidate generated outputs
        gen_prefix = f"gen:{str(document_id)[:8]}"
        invalidated += await self.backend.invalidate_by_prefix(gen_prefix)
        
        logger.info(f"Invalidated {invalidated} cache entries for document {document_id}")
        return invalidated
    
    async def invalidate_project(self, project_id: uuid.UUID) -> int:
        """
        Invalidate all cache entries for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        
        # Invalidate all generated outputs for project
        proj_prefix = f"gen:{str(project_id)}"
        invalidated += await self.backend.invalidate_by_prefix(proj_prefix)
        
        logger.info(f"Invalidated {invalidated} cache entries for project {project_id}")
        return invalidated
    
    async def invalidate_by_request(self, request: InvalidationRequest) -> InvalidationResponse:
        """
        Invalidate cache based on request.
        
        Args:
            request: Invalidation request
            
        Returns:
            Invalidation response
        """
        invalidated_count = 0
        
        if request.prefix:
            invalidated_count = await self.backend.invalidate_by_prefix(request.prefix)
        elif request.document_id:
            invalidated_count = await self.invalidate_document(request.document_id)
        elif request.project_id:
            invalidated_count = await self.invalidate_project(request.project_id)
        elif request.cache_type:
            prefix = KEY_PREFIXES.get(request.cache_type)
            if prefix:
                invalidated_count = await self.backend.invalidate_by_prefix(prefix)
        
        return InvalidationResponse(
            success=True,
            invalidated_count=invalidated_count,
            message=f"Invalidated {invalidated_count} entries"
        )
    
    async def clear_all(self) -> bool:
        """Clear all cache entries"""
        return await self.backend.clear()
    
    # ==================== Statistics ====================
    
    async def get_stats(self) -> Dict[str, CacheStats]:
        """Get cache statistics by type"""
        stats = {}
        
        for cache_type in CacheType:
            type_stats = self._stats[cache_type]
            hits = type_stats["hits"]
            misses = type_stats["misses"]
            total = hits + misses
            
            stats[cache_type.value] = CacheStats(
                cache_type=cache_type,
                hits=hits,
                misses=misses,
                hit_rate=hits / total if total > 0 else 0.0
            )
        
        return stats
    
    async def get_health(self) -> CacheHealthResponse:
        """Get cache health status"""
        backend_stats = await self.backend.stats()
        
        return CacheHealthResponse(
            status="healthy" if await self.backend.health_check() else "unhealthy",
            backend=backend_stats.get("backend", "unknown"),
            total_entries=backend_stats.get("size", 0),
            stats=await self.get_stats()
        )


# Global cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service(config: Optional[CacheConfig] = None) -> CacheService:
    """Get or create global cache service"""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService(config)
    return _cache_service


async def initialize_cache(config: Optional[CacheConfig] = None) -> CacheService:
    """Initialize and return cache service"""
    service = get_cache_service(config)
    await service.initialize()
    return service


__all__ = [
    "CacheService",
    "CacheConfig",
    "get_cache_service",
    "initialize_cache",
    "CacheType",
    "InvalidationRequest",
    "InvalidationResponse",
    "CacheHealthResponse",
]