"""
Cache Backend Interface

Abstract interface for cache implementations.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any, Dict, List

from .types import CacheType, CacheEntry, CacheStats


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., "emb:*")
            
        Returns:
            List of matching keys
        """
        pass
    
    @abstractmethod
    async def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Invalidate all keys with a prefix.
        
        Args:
            prefix: Key prefix
            
        Returns:
            Number of keys invalidated
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of statistics
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if cache backend is healthy.
        
        Returns:
            True if healthy
        """
        pass


class CacheBackendError(Exception):
    """Error during cache operation"""
    pass