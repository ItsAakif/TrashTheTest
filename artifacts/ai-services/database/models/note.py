"""
Note Model

Represents structured notes generated from document content.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..engine import Base


class Note(Base):
    """
    Note entity - structured notes generated from document content.
    Content is stored as JSONB for structured rendering.
    """
    __tablename__ = "notes"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to project
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    
    # Note details
    title = Column(String(500), nullable=False)
    
    # Content stored as JSONB for structure (sections, formatting, etc.)
    content = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="notes")
    
    def __repr__(self):
        return f"<Note(id={self.id}, title={self.title})>"