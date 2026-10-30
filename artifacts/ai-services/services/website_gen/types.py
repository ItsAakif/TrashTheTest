"""
Website Generation Types

Pydantic models for website generation input, output, and configuration.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
from enum import Enum


class WebsiteContentType(str, Enum):
    """Type of content for the website"""
    NOTES = "notes"
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    MIXED = "mixed"


class WebsiteGenerationRequest(BaseModel):
    """Request for generating a website"""
    project_id: str
    content_type: WebsiteContentType
    title: Optional[str] = None
    structured_content: Dict[str, Any]  # Content from transformation engine


class WebsiteMetadata(BaseModel):
    """Metadata about a generated website"""
    id: uuid.UUID
    project_id: uuid.UUID
    slug: str
    content_type: WebsiteContentType
    title: str
    description: Optional[str] = None
    is_published: bool
    # Lifecycle: draft (manage-only) vs published (public-resolvable).
    status: str = "draft"
    visibility: str = "private"
    created_at: datetime
    updated_at: datetime


class WebsiteContent(BaseModel):
    """Structured content for website rendering"""
    title: str
    notes: Optional[Dict[str, Any]] = None
    flashcards: Optional[List[Dict[str, Any]]] = None
    quiz: Optional[Dict[str, Any]] = None


class WebsiteGenerationResult(BaseModel):
    """Result from website generation"""
    website_id: uuid.UUID
    slug: str
    title: str
    content_type: WebsiteContentType
    html_content: str
    preview_url: str
    created_at: datetime


__all__ = [
    "WebsiteContentType",
    "WebsiteGenerationRequest",
    "WebsiteMetadata",
    "WebsiteContent",
    "WebsiteGenerationResult",
]