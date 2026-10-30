"""
GeneratedContentCache Model

Caches generated content to avoid recomputation.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum

from ..engine import Base


class OutputType(enum.Enum):
    """Type of generated content"""
    NOTES = "notes"
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    MCQ = "mcq"  # Phase 4 strict MCQ set
    ANSWER = "answer"  # Phase 4 marks-aware answer
    EXPLANATION = "explanation"
    SUMMARY = "summary"
    WEBSITE = "website"
    SYLLABUS = "syllabus"  # Academic: extracted syllabus
    PYQ = "pyq"  # Academic: extracted previous-year questions
    MAPPING = "mapping"  # Academic: PYQ -> unit mapping
    ANALYSIS = "analysis"  # Academic: frequency / marks alignment


class GeneratedContentCache(Base):
    """
    GeneratedContentCache entity - stores generated content for reuse.
    
    Content hash allows quick lookup to avoid regenerating identical content.
    """
    __tablename__ = "generated_content_cache"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to project
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    
    # Cache key - hash of input chunks + generation config
    content_hash = Column(String(64), nullable=False, index=True)  # SHA256 hash
    
    # Output type (values_callable persists lowercase VALUES to match the
    # native PG enums created by migrations).
    output_type = Column(
        SQLEnum(OutputType, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False
    )
    
    # Cached output content (JSONB for flexibility)
    output_content = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="generated_content_cache")
    
    def __repr__(self):
        return f"<GeneratedContentCache(id={self.id}, type={self.output_type.value}, hash={self.content_hash[:8]}...)>"