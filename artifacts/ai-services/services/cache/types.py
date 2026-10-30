"""
Cache System Types

Type definitions for the caching layer.
"""

import uuid
from datetime import datetime
from typing import Optional, Any, Dict, List
from enum import Enum
from pydantic import BaseModel, Field


class CacheType(str, Enum):
    """Type of cached data"""
    EMBEDDING = "embedding"
    GENERATED_OUTPUT = "generated_output"
    RESEARCH_RESULT = "research_result"
    DOCUMENT_METADATA = "document_metadata"
    USER_SESSION = "user_session"
    RETRIEVAL_RESULT = "retrieval_result"


class CacheTTL(str, Enum):
    """TTL presets for different cache types"""
    # Long TTL (weeks)
    EMBEDDING = "embedding"  # 30 days
    
    # Medium TTL (days)
    GENERATED_OUTPUT = "generated_output"  # 7 days
    
    # Short TTL (hours)
    RESEARCH_RESULT = "research_result"  # 1 hour
    RETRIEVAL_RESULT = "retrieval_result"  # 15 minutes
    
    # Very short TTL (minutes)
    DOCUMENT_METADATA = "document_metadata"  # 5 minutes
    USER_SESSION = "user_session"  # 30 minutes


# TTL values in seconds
TTL_VALUES = {
    CacheTTL.EMBEDDING: 30 * 24 * 60 * 60,  # 30 days
    CacheTTL.GENERATED_OUTPUT: 7 * 24 * 60 * 60,  # 7 days
    CacheTTL.RESEARCH_RESULT: 60 * 60,  # 1 hour
    CacheTTL.RETRIEVAL_RESULT: 15 * 60,  # 15 minutes
    CacheTTL.DOCUMENT_METADATA: 5 * 60,  # 5 minutes
    CacheTTL.USER_SESSION: 30 * 60,  # 30 minutes
}


class CacheEntry(BaseModel):
    """Single cache entry"""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class CacheStats(BaseModel):
    """Cache statistics"""
    cache_type: CacheType
    hits: int = 0
    misses: int = 0
    size: int = 0
    hit_rate: float = 0.0


class CacheConfig(BaseModel):
    """Cache configuration"""
    backend: str = "memory"  # "memory" or "redis"
    redis_url: Optional[str] = None
    default_ttl: int = 3600  # 1 hour default
    max_size: Optional[int] = None  # Max entries (for memory backend)
    enable_logging: bool = True


class CacheKeyInfo(BaseModel):
    """Information about a cache key"""
    key: str
    cache_type: CacheType
    ttl_seconds: int
    created_at: Optional[datetime] = None


class InvalidationRequest(BaseModel):
    """Request to invalidate cache entries"""
    cache_type: Optional[CacheType] = None
    document_id: Optional[uuid.UUID] = None
    project_id: Optional[uuid.UUID] = None
    prefix: Optional[str] = None


class InvalidationResponse(BaseModel):
    """Response for cache invalidation"""
    success: bool
    invalidated_count: int = 0
    message: str


class CacheHealthResponse(BaseModel):
    """Cache health status"""
    status: str
    backend: str
    total_entries: int
    stats: Dict[str, CacheStats]
    timestamp: datetime


# Cache key prefix constants
KEY_PREFIXES = {
    CacheType.EMBEDDING: "emb",
    CacheType.GENERATED_OUTPUT: "gen",
    CacheType.RESEARCH_RESULT: "res",
    CacheType.DOCUMENT_METADATA: "doc",
    CacheType.USER_SESSION: "sess",
    CacheType.RETRIEVAL_RESULT: "ret",
}


def get_ttl_for_type(cache_type: CacheType) -> int:
    """Get TTL in seconds for a cache type"""
    ttl_map = {
        CacheType.EMBEDDING: TTL_VALUES[CacheTTL.EMBEDDING],
        CacheType.GENERATED_OUTPUT: TTL_VALUES[CacheTTL.GENERATED_OUTPUT],
        CacheType.RESEARCH_RESULT: TTL_VALUES[CacheTTL.RESEARCH_RESULT],
        CacheType.RETRIEVAL_RESULT: TTL_VALUES[CacheTTL.RETRIEVAL_RESULT],
        CacheType.DOCUMENT_METADATA: TTL_VALUES[CacheTTL.DOCUMENT_METADATA],
        CacheType.USER_SESSION: TTL_VALUES[CacheTTL.USER_SESSION],
    }
    return ttl_map.get(cache_type, 3600)