"""
Quiz Model

Represents quizzes generated from document content.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..engine import Base


class Quiz(Base):
    """
    Quiz entity - quiz with questions generated from document content.
    Questions stored as JSONB for flexibility.
    """
    __tablename__ = "quizzes"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to project
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    
    # Quiz details
    title = Column(String(500), nullable=False)
    
    # Questions stored as JSONB
    # Structure: [{"question": "...", "options": [...], "correct_answer": "...", "explanation": "..."}]
    questions = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="quizzes")
    
    def __repr__(self):
        return f"<Quiz(id={self.id}, title={self.title})>"