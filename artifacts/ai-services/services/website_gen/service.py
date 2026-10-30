"""
Website Generation Service

Transforms structured content from the Transformation Engine into
standalone HTML study websites with Jinja2 templates.
"""

import hashlib
import uuid
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import Website, WebsiteType
from database import repositories
from .types import (
    WebsiteContentType,
    WebsiteGenerationRequest,
    WebsiteGenerationResult,
    WebsiteMetadata,
)


# Template directory - would be in a templates folder
TEMPLATE_DIR = "services/website_gen/templates"


class WebsiteGenerationService:
    """
    Service for generating study websites from structured content.
    
    Features:
    - Jinja2 template-based rendering
    - Multiple content type support (notes, flashcards, quiz, mixed)
    - Unique slug generation
    - Database storage
    - Preview URL generation
    - Self-contained, portable HTML
    """
    
    def __init__(self):
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Template mappings
        self.template_map = {
            WebsiteContentType.NOTES: 'notes.html',
            WebsiteContentType.FLASHCARDS: 'flashcards.html',
            WebsiteContentType.QUIZ: 'quiz.html',
            WebsiteContentType.MIXED: 'mixed.html',
        }
    
    def _generate_slug(self, project_id: str, title: str) -> str:
        """Generate a unique, traversal-safe slug for the website."""
        # Create hash from project_id + title + timestamp
        timestamp = str(int(time.time()))
        hash_input = f"{project_id}:{title}:{timestamp}"
        hash_suffix = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

        # Create readable slug: lowercase alnum + hyphens only (no traversal).
        title_slug = title.lower().replace(' ', '-') if title else "study"
        title_slug = ''.join(c for c in title_slug if c.isalnum() or c == '-')[:30]
        # The slug guard allows only single hyphens between alnum runs, so a
        # title containing ' - ' must not produce '---'.
        import re as _re
        title_slug = _re.sub(r"-{2,}", "-", title_slug).strip("-") or "study"

        return f"{title_slug}-{hash_suffix}"

    async def _unique_slug(self, db: AsyncSession, project_id: str, title: str) -> str:
        """Slug that is unique in the websites table (retry on collision)."""
        for _ in range(5):
            slug = self._generate_slug(project_id, title)
            if not await repositories.slug_exists(db, slug):
                return slug
        # Final fallback: full-entropy suffix.
        return f"study-{uuid.uuid4().hex[:12]}"
    
    def _render_template(
        self,
        content_type: WebsiteContentType,
        title: str,
        content: Dict[str, Any]
    ) -> str:
        """
        Render the appropriate template with content.
        
        Uses Jinja2 for clean separation of content and presentation.
        """
        template_name = self.template_map.get(content_type, 'mixed.html')
        
        try:
            template = self.jinja_env.get_template(template_name)
        except TemplateNotFound:
            # Fallback to mixed template
            template = self.jinja_env.get_template('mixed.html')
        
        # Render with content
        return template.render(
            title=title or "Study Website",
            content=content,
            generated_at=datetime.utcnow().isoformat()
        )
    
    async def generate_website(
        self,
        request: WebsiteGenerationRequest,
        db: AsyncSession
    ) -> WebsiteGenerationResult:
        """
        Generate a website from structured content.
        
        Args:
            request: Website generation request
            db: Database session
            
        Returns:
            WebsiteGenerationResult with HTML and metadata
        """
        start_time = time.time()
        
        # Get project to validate and get name
        project_uuid = uuid.UUID(request.project_id)
        project = await repositories.get_project_by_id(db, project_uuid)
        
        if not project:
            raise ValueError(f"Project {request.project_id} not found")
        
        # Generate unique slug
        slug = await self._unique_slug(db, request.project_id, request.title or project.name)

        # Render HTML from template, then sanitise before persist AND render
        # (same renderer/sanitiser for preview and published: parity).
        html_content = self._render_template(
            content_type=request.content_type,
            title=request.title or project.name,
            content=request.structured_content
        )
        from .sanitise import sanitise_html

        html_content = sanitise_html(html_content)

        # Convert content type to enum
        website_type = WebsiteType(request.content_type.value)

        # Create website record
        website = await repositories.create_website(
            db=db,
            project_id=project_uuid,
            slug=slug,
            website_type=website_type.value,
            html_content=html_content,
            title=request.title or project.name,
        )
        await db.commit()
        try:
            await db.refresh(website)
        except Exception as exc:
            # Pre-008 store: the refresh SELECT expands the additive
            # columns (title/...) and 500s. The row is committed; roll
            # the aborted statement back and serve the flushed instance
            # (title falls back via getattr below). No data loss.
            msg = str(exc).lower()
            if "does not exist" not in msg:
                raise
            try:
                await db.rollback()
            except Exception:
                pass

        processing_time = time.time() - start_time
        
        return WebsiteGenerationResult(
            website_id=website.id,
            slug=slug,
            title=getattr(website, "title", None) or request.title or project.name,
            content_type=request.content_type,
            html_content=html_content,
            preview_url=f"/api/website/preview/{slug}",
            created_at=website.created_at
        )
    
    def _to_metadata(self, website) -> "WebsiteMetadata":
        """Row -> metadata, tolerating pre-migration rows (getattr guards)."""
        from database.models.website import WebsiteStatus, WebsiteVisibility

        status = getattr(website, "status", None)
        visibility = getattr(website, "visibility", None)
        if isinstance(status, WebsiteStatus):
            status_value = status.value
        else:
            status_value = "published" if getattr(website, "is_published", False) else "draft"
        if isinstance(visibility, WebsiteVisibility):
            visibility_value = visibility.value
        else:
            visibility_value = "public" if getattr(website, "is_published", False) else "private"
        return WebsiteMetadata(
            id=website.id,
            project_id=website.project_id,
            slug=website.slug,
            content_type=WebsiteContentType(website.website_type.value),
            title=getattr(website, "title", None) or "Untitled",
            is_published=bool(getattr(website, "is_published", False)),
            status=status_value,
            visibility=visibility_value,
            created_at=website.created_at,
            updated_at=website.updated_at,
        )

    async def get_website_by_slug(
        self,
        slug: str,
        db: AsyncSession
    ) -> Optional[WebsiteMetadata]:
        """Get website metadata by slug"""
        website = await repositories.get_website_by_slug(db, slug)

        if not website:
            return None

        return self._to_metadata(website)

    async def get_website_html(
        self,
        slug: str,
        db: AsyncSession
    ) -> Optional[str]:
        """Get raw HTML content for a website (persisted sanitised HTML)."""
        website = await repositories.get_website_by_slug(db, slug)

        if not website:
            return None

        return website.html_content

    async def is_published(self, slug: str, db: AsyncSession) -> bool:
        """Public resolvability: published flag AND published status."""
        website = await repositories.get_website_by_slug(db, slug)
        if not website:
            return False
        if not getattr(website, "is_published", False):
            return False
        status = getattr(website, "status", None)
        status_value = status.value if hasattr(status, "value") else status
        return status_value in (None, "published")

    async def get_websites_by_project(
        self,
        project_id: str,
        db: AsyncSession
    ) -> List[WebsiteMetadata]:
        """Get all websites for a project"""
        project_uuid = uuid.UUID(project_id)
        websites = await repositories.get_websites_by_project(db, project_uuid)

        return [self._to_metadata(w) for w in websites]

    async def publish_website(
        self,
        slug: str,
        db: AsyncSession
    ) -> bool:
        """Mark a website as published (public visibility + published status)."""
        from database.models.website import WebsiteStatus, WebsiteVisibility

        website = await repositories.get_website_by_slug(db, slug)

        if not website:
            return False

        website.is_published = True
        try:
            website.status = WebsiteStatus.PUBLISHED
            website.visibility = WebsiteVisibility.PUBLIC
        except Exception:
            pass
        await db.commit()

        return True

    async def unpublish_website(self, slug: str, db: AsyncSession) -> bool:
        """Visibility toggle off: draft + private, no longer publicly resolvable."""
        from database.models.website import WebsiteStatus, WebsiteVisibility

        website = await repositories.get_website_by_slug(db, slug)
        if not website:
            return False
        website.is_published = False
        try:
            website.status = WebsiteStatus.DRAFT
            website.visibility = WebsiteVisibility.PRIVATE
        except Exception:
            pass
        await db.commit()
        return True

    async def set_visibility(self, slug: str, visibility: str, db: AsyncSession) -> bool:
        """Explicit visibility toggle (public/private); public implies published."""
        from database.models.website import WebsiteStatus, WebsiteVisibility

        website = await repositories.get_website_by_slug(db, slug)
        if not website:
            return False
        vis = WebsiteVisibility(visibility)
        try:
            website.visibility = vis
        except Exception:
            pass
        if vis == WebsiteVisibility.PUBLIC:
            website.is_published = True
            try:
                website.status = WebsiteStatus.PUBLISHED
            except Exception:
                pass
        else:
            website.is_published = False
            try:
                website.status = WebsiteStatus.DRAFT
            except Exception:
                pass
        await db.commit()
        return True


# Default service instance
website_gen_service = WebsiteGenerationService()