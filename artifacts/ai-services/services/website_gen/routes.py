"""
Website Generation API Routes.

Pipeline: content -> assembly -> sanitise -> preview -> publish -> public.

- Manage routes (generate/list/detail/preview/raw/publish/visibility/update/
  delete) require auth + project ownership (404 for another user's project).
- The public route resolves PUBLISHED websites only; drafts/private get 404
  and never leak private docs (only the persisted sanitised HTML is served).
- Preview and published render use the same persisted sanitised HTML (parity).
"""

import asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import require_project_access
from database.engine import get_db
from database import repositories
from database.models.user import User
from services.auth.deps import get_current_user, get_optional_user

from .service import website_gen_service
from .sanitise import is_slug_safe
from .types import (
    WebsiteContentType,
    WebsiteGenerationRequest,
    WebsiteGenerationResult,
    WebsiteMetadata,
)


router = APIRouter(prefix="/api/website", tags=["website"])


class WebsiteCreateRequest(BaseModel):
    """Request to create a website (accepts frontend + backend field names)."""
    project_id: Optional[str] = Field(default=None, alias="projectId")
    content_type: Optional[WebsiteContentType] = Field(default=None, alias="contentType")
    title: Optional[str] = None
    description: Optional[str] = None
    # Structured content from the transformation engine (content refs).
    content: Optional[dict] = None
    structured_content: Optional[dict] = None

    model_config = {"populate_by_name": True}

    def resolved_content(self) -> dict:
        return self.content if self.content is not None else (self.structured_content or {})


class WebsitePublishRequest(BaseModel):
    """Request to publish a website"""
    slug: str


class WebsiteVisibilityRequest(BaseModel):
    """Request to toggle visibility (public/private)."""
    slug: str
    visibility: str  # public | private


class WebsiteUpdateRequest(BaseModel):
    """Manage update: title/description/visibility (ownership-checked)."""
    title: Optional[str] = None
    description: Optional[str] = None
    visibility: Optional[str] = None


def _require_slug(slug: str) -> str:
    if not is_slug_safe(slug):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid slug format",
        )
    return slug


async def _require_website_ownership(slug: str, current_user: User, db: AsyncSession):
    """Ownership on manage: the website's project must belong to the caller."""
    website = await repositories.get_website_by_slug(db, slug)
    if website is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Website with slug '{slug}' not found")
    await require_project_access(str(website.project_id), current_user, db)
    return website


# ==================== WEBSITE GENERATION ====================

# ADR-008 (T015): the render+persist path is template work only - no LLM
# call runs here - so it gets a hard ceiling instead of hanging forever.
# Module constant so the 504 contract is testable without patching literals.
WEBSITE_RENDER_TIMEOUT_S = 60.0


@router.post("/generate", response_model=WebsiteGenerationResult)
async def generate_website(
    request: WebsiteCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a study website from structured content.

    Takes content from the transformation engine and renders it
    as a standalone HTML website with appropriate templates.
    """
    try:
        if not request.project_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="project_id is required")
        if request.content_type is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="content_type is required")
        # Ownership (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)

        # Create generation request
        gen_request = WebsiteGenerationRequest(
            project_id=str(project.id),
            content_type=request.content_type,
            title=request.title,
            structured_content=request.resolved_content(),
        )

        # Generate website (bounded: assembly must never hang the request).
        try:
            result = await asyncio.wait_for(
                website_gen_service.generate_website(gen_request, db),
                timeout=WEBSITE_RENDER_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"website assembly timed out after "
                       f"{WEBSITE_RENDER_TIMEOUT_S:g}s",
            )

        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Website generation failed: {str(e)}"
        )


# ==================== WEBSITE METADATA ====================

@router.get("/list/{project_id}", response_model=List[WebsiteMetadata])
async def list_websites(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all websites for a project (ownership-checked)."""
    try:
        project = await require_project_access(project_id, current_user, db)
        websites = await website_gen_service.get_websites_by_project(str(project.id), db)
        return websites

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )


@router.get("/detail/{slug}", response_model=WebsiteMetadata)
async def get_website_detail(
    slug: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Manage detail: ownership-checked (private/draft visible to owner)."""
    _require_slug(slug)
    await _require_website_ownership(slug, current_user, db)
    metadata = await website_gen_service.get_website_by_slug(slug, db)

    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Website with slug '{slug}' not found"
        )

    return metadata


@router.get("/{slug}", response_model=WebsiteMetadata)
async def get_website(
    slug: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get website metadata by slug (ownership-checked manage read)."""
    _require_slug(slug)
    await _require_website_ownership(slug, current_user, db)
    metadata = await website_gen_service.get_website_by_slug(slug, db)

    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Website with slug '{slug}' not found"
        )

    return metadata


# ==================== WEBSITE PREVIEW ====================

@router.get("/preview/{slug}")
async def preview_website(
    slug: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Preview a generated website by slug (owner only).

    Returns the persisted sanitised HTML — the same bytes the public
    route serves after publish (preview/published parity).
    """
    _require_slug(slug)
    await _require_website_ownership(slug, current_user, db)
    html_content = await website_gen_service.get_website_html(slug, db)

    if not html_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Website with slug '{slug}' not found"
        )

    return HTMLResponse(content=html_content, media_type="text/html")


@router.get("/raw/{slug}")
async def get_raw_html(
    slug: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get raw HTML content for a website (owner only)."""
    _require_slug(slug)
    await _require_website_ownership(slug, current_user, db)
    html_content = await website_gen_service.get_website_html(slug, db)

    if not html_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Website with slug '{slug}' not found"
        )

    return {"slug": slug, "html_content": html_content}


# ==================== PUBLIC RESOLVE (published only) ====================

@router.get("/public/{slug}", response_model=WebsiteMetadata)
async def resolve_public_website(
    slug: str,
    db: AsyncSession = Depends(get_db),
):
    """Domain-agnostic public resolve: published websites only.

    No auth required; drafts/private/draft-hidden get 404 and never leak
    private docs (only sanitised HTML + metadata are exposed).
    """
    _require_slug(slug)
    if not await website_gen_service.is_published(slug, db):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Website not found")
    metadata = await website_gen_service.get_website_by_slug(slug, db)
    if not metadata:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Website not found")
    return metadata


@router.get("/public/{slug}/html")
async def resolve_public_html(
    slug: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve the published sanitised HTML (published only, safe render)."""
    _require_slug(slug)
    if not await website_gen_service.is_published(slug, db):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Website not found")
    html_content = await website_gen_service.get_website_html(slug, db)
    if not html_content:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Website not found")
    return HTMLResponse(content=html_content, media_type="text/html")


# ==================== PUBLISHING ====================

@router.post("/publish")
async def publish_website(
    request: WebsitePublishRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Mark a website as published (ownership-checked)."""
    _require_slug(request.slug)
    await _require_website_ownership(request.slug, current_user, db)
    success = await website_gen_service.publish_website(request.slug, db)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Website with slug '{request.slug}' not found"
        )

    return {
        "status": "published",
        "slug": request.slug,
        "preview_url": f"/api/website/preview/{request.slug}",
        "public_url": f"/api/website/public/{request.slug}/html",
    }


@router.post("/unpublish")
async def unpublish_website(
    request: WebsitePublishRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Unpublish: draft + private, no longer publicly resolvable."""
    _require_slug(request.slug)
    await _require_website_ownership(request.slug, current_user, db)
    success = await website_gen_service.unpublish_website(request.slug, db)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Website with slug '{request.slug}' not found")
    return {"status": "draft", "slug": request.slug}


@router.post("/visibility")
async def set_website_visibility(
    request: WebsiteVisibilityRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Visibility toggle (public/private), ownership-checked."""
    _require_slug(request.slug)
    if request.visibility not in ("public", "private"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="visibility must be 'public' or 'private'")
    await _require_website_ownership(request.slug, current_user, db)
    success = await website_gen_service.set_visibility(request.slug, request.visibility, db)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Website with slug '{request.slug}' not found")
    return {"status": "ok", "slug": request.slug, "visibility": request.visibility}


@router.patch("/manage/{slug}", response_model=WebsiteMetadata)
async def update_website(
    slug: str,
    request: WebsiteUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update title/description/visibility (ownership-checked)."""
    _require_slug(slug)
    if request.visibility is not None and request.visibility not in ("public", "private"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="visibility must be 'public' or 'private'")
    website = await _require_website_ownership(slug, current_user, db)
    if request.title is not None:
        website.title = request.title
    try:
        if request.description is not None:
            website.description = request.description
    except Exception:
        pass
    await db.commit()
    if request.visibility is not None:
        await website_gen_service.set_visibility(slug, request.visibility, db)
    metadata = await website_gen_service.get_website_by_slug(slug, db)
    if not metadata:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Website with slug '{slug}' not found")
    return metadata


@router.delete("/manage/{slug}")
async def delete_website(
    slug: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a website (ownership-checked)."""
    _require_slug(slug)
    website = await _require_website_ownership(slug, current_user, db)
    await repositories.delete_website(db, website)
    return {"success": True, "message": "Website deleted successfully"}


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check():
    """Health check for website generation service"""
    return {
        "status": "healthy",
        "service": "website_gen",
        "templates": ["notes", "flashcards", "quiz", "mixed"]
    }


__all__ = ["router"]
