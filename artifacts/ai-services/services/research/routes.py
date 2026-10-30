"""
Research Augmentation API Routes

REST API endpoints for research augmentation operations.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List

from .service import ResearchAugmentationService, research_service
from .types import ResearchQuery, ResearchConfig, ResearchSource


router = APIRouter(prefix="/api/research", tags=["research"])


class ResearchRequest(BaseModel):
    """Request for research augmentation"""
    query: str
    max_sources: Optional[int] = 3


class ResearchResponse(BaseModel):
    """Response from research"""
    query: str
    sources: List[dict]
    summary: str
    key_findings: List[str]
    processing_time: float
    cached: bool


class ConfigUpdateRequest(BaseModel):
    """Request to update research config"""
    enabled: Optional[bool] = None
    max_sources: Optional[int] = None
    cache_enabled: Optional[bool] = None


@router.post("/search", response_model=ResearchResponse)
async def perform_research(
    request: ResearchRequest
):
    """
    Perform standalone web research.
    
    Returns research results without document context.
    """
    try:
        result = await research_service.research_only(
            query=request.query,
            max_sources=request.max_sources
        )
        
        if not result:
            return ResearchResponse(
                query=request.query,
                sources=[],
                summary="No research results found.",
                key_findings=[],
                processing_time=0,
                cached=False
            )
        
        return ResearchResponse(
            query=result.original_query,
            sources=[
                {
                    "url": s.url,
                    "title": s.title,
                    "content": s.content[:500],  # Limit content in response
                    "relevance_score": s.relevance_score
                }
                for s in result.sources
            ],
            summary=result.summary,
            key_findings=result.key_findings,
            processing_time=result.processing_time,
            cached=result.cached
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research failed: {str(e)}"
        )


@router.get("/config")
async def get_config():
    """Get current research configuration"""
    return {
        "enabled": research_service.config.enabled,
        "max_sources": research_service.config.max_sources,
        "max_content_length": research_service.config.max_content_length,
        "min_relevance_score": research_service.config.min_relevance_score,
        "cache_enabled": research_service.config.cache_enabled
    }


@router.post("/config")
async def update_config(
    request: ConfigUpdateRequest
):
    """Update research configuration"""
    if request.enabled is not None:
        research_service.config.enabled = request.enabled
    if request.max_sources is not None:
        research_service.config.max_sources = request.max_sources
    if request.cache_enabled is not None:
        research_service.config.cache_enabled = request.cache_enabled
    
    return {
        "status": "updated",
        "config": {
            "enabled": research_service.config.enabled,
            "max_sources": research_service.config.max_sources,
            "cache_enabled": research_service.config.cache_enabled
        }
    }


@router.get("/cache/stats")
async def get_cache_stats():
    """Get research cache statistics"""
    return research_service.get_cache_stats()


@router.post("/cache/clear")
async def clear_cache():
    """Clear research cache"""
    research_service.clear_cache()
    return {"status": "cache_cleared"}


@router.get("/health")
async def health_check():
    """Health check for research service"""
    return {
        "status": "healthy",
        "service": "research",
        "provider": "tavily",
        "enabled": research_service.config.enabled
    }


__all__ = ["router"]