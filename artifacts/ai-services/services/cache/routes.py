"""
Cache Management API Routes

API endpoints for cache management:
- GET /api/cache/health - Cache health status
- GET /api/cache/stats - Cache statistics
- POST /api/cache/invalidate - Invalidate cache entries
- DELETE /api/cache/clear - Clear all cache
- GET /api/cache/keys - List cache keys
"""

import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from .service import get_cache_service, CacheService
from .types import CacheType, InvalidationRequest, InvalidationResponse, CacheHealthResponse


router = APIRouter(prefix="/api/cache", tags=["cache"])


# ==================== Request/Response Models ====================

class InvalidateRequest(BaseModel):
    """Request to invalidate cache"""
    cache_type: Optional[CacheType] = None
    document_id: Optional[uuid.UUID] = None
    project_id: Optional[uuid.UUID] = None
    prefix: Optional[str] = None


class ClearResponse(BaseModel):
    """Response for cache clear"""
    success: bool
    message: str


class KeysResponse(BaseModel):
    """Response for listing keys"""
    keys: List[str]
    total: int


# ==================== Endpoints ====================

@router.get("/health", response_model=CacheHealthResponse)
async def get_cache_health():
    """
    Get cache health status.
    
    Returns:
        Health status including backend type, entry count, and per-type statistics
    """
    cache = get_cache_service()
    return await cache.get_health()


@router.get("/stats")
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        Hit/miss counts and rates by cache type
    """
    cache = get_cache_service()
    stats = await cache.get_stats()
    
    return {
        "stats": {
            name: s.model_dump() for name, s in stats.items()
        }
    }


@router.post("/invalidate", response_model=InvalidationResponse)
async def invalidate_cache(request: InvalidateRequest):
    """
    Invalidate cache entries.
    
    Can invalidate by:
    - cache_type: All entries of a specific type
    - document_id: All entries related to a document
    - project_id: All entries related to a project
    - prefix: Custom prefix pattern
    """
    cache = get_cache_service()
    
    inv_request = InvalidationRequest(
        cache_type=request.cache_type,
        document_id=request.document_id,
        project_id=request.project_id,
        prefix=request.prefix
    )
    
    return await cache.invalidate_by_request(inv_request)


@router.delete("/clear", response_model=ClearResponse)
async def clear_cache():
    """
    Clear all cache entries.
    
    WARNING: This will remove all cached data including embeddings,
    generated outputs, and search results.
    """
    cache = get_cache_service()
    success = await cache.clear_all()
    
    if success:
        return ClearResponse(success=True, message="All cache entries cleared")
    else:
        return ClearResponse(success=False, message="Failed to clear cache")


@router.get("/keys", response_model=KeysResponse)
async def list_cache_keys(
    pattern: str = Query("*", description="Key pattern to match"),
    limit: int = Query(100, ge=1, le=1000, description="Max keys to return")
):
    """
    List cache keys matching a pattern.
    
    Useful for debugging and understanding cache contents.
    """
    cache = get_cache_service()
    
    try:
        keys = await cache.backend.keys(pattern)
        keys = keys[:limit]  # Apply limit
        
        return KeysResponse(
            keys=keys,
            total=len(keys)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Specific invalidation endpoints

@router.delete("/document/{document_id}")
async def invalidate_document_cache(document_id: uuid.UUID):
    """Invalidate all cache entries for a specific document."""
    cache = get_cache_service()
    count = await cache.invalidate_document(document_id)
    
    return {
        "success": True,
        "invalidated_count": count,
        "message": f"Invalidated {count} cache entries for document {document_id}"
    }


@router.delete("/project/{project_id}")
async def invalidate_project_cache(project_id: uuid.UUID):
    """Invalidate all cache entries for a specific project."""
    cache = get_cache_service()
    count = await cache.invalidate_project(project_id)
    
    return {
        "success": True,
        "invalidated_count": count,
        "message": f"Invalidated {count} cache entries for project {project_id}"
    }


__all__ = ["router"]