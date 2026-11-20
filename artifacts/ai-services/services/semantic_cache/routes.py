"""Semantic cache developer controls + metrics (S3)."""

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from services.semantic_cache import semantic_cache


router = APIRouter(prefix="/api/semantic-cache", tags=["semantic-cache"])


class InvalidateRequest(BaseModel):
    project_id: Optional[str] = None
    document_id: Optional[str] = None
    user_id: Optional[str] = None


@router.get("/health")
async def health():
    return {"status": "ok", "enabled": semantic_cache.enabled,
            "threshold": semantic_cache.threshold,
            "ttl_seconds": semantic_cache.ttl_seconds}


@router.get("/stats")
async def stats(db: AsyncSession = Depends(get_db)):
    return await semantic_cache.stats(db)


@router.post("/invalidate")
async def invalidate(request: InvalidateRequest, db: AsyncSession = Depends(get_db)):
    count = await semantic_cache.invalidate(
        db,
        project_id=request.project_id,
        document_id=request.document_id,
        user_id=request.user_id,
    )
    return {"success": True, "invalidated": count}


@router.delete("/clear")
async def clear(db: AsyncSession = Depends(get_db)):
    count = await semantic_cache.clear(db)
    return {"success": True, "cleared": count}


__all__ = ["router"]
