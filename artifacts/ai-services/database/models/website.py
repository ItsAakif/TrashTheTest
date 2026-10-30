"""
Website Model

Represents generated study websites from content.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, Enum as SQLEnum, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from ..engine import Base


class WebsiteType(enum.Enum):
    """Type of generated website"""
    NOTES = "notes"
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    MIXED = "mixed"


class WebsiteVisibility(str, enum.Enum):
    """Visibility of a generated website (shareable pipeline)."""
    PRIVATE = "private"
    PUBLIC = "public"


class WebsiteStatus(str, enum.Enum):
    """Lifecycle status: draft (manage-only) vs published (public-resolvable)."""
    DRAFT = "draft"
    PUBLISHED = "published"


class Website(Base):
    """
    Website entity - generated study websites from content.

    Content refs (project + structured content type) are stored; large
    rendered HTML lives in html_content. Visibility/status/version are
    additive (nullable/defaulted) so pre-migration rows keep working.
    """
    __tablename__ = "websites"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign key to project
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)

    # Website details (values_callable persists lowercase VALUES to match
    # the native PG enums created by migrations).
    website_type = Column(
        SQLEnum(WebsiteType, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False
    )

    # Human metadata (additive; nullable for pre-migration rows)
    title = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)

    # Visibility + lifecycle (additive; defaults preserve old behavior:
    # old rows are drafts/private until explicitly published).
    visibility = Column(
        SQLEnum(WebsiteVisibility, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=WebsiteVisibility.PRIVATE,
    )
    status = Column(
        SQLEnum(WebsiteStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=WebsiteStatus.DRAFT,
    )
    version = Column(String(20), nullable=True, default="1")
    
    # HTML content
    html_content = Column(Text, nullable=True)
    
    # URL slug (unique)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    
    # Publishing status
    is_published = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="websites")
    
    def __repr__(self):
        return f"<Website(id={self.id}, slug={self.slug}, type={self.website_type.value})>"