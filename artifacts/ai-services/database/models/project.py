"""
Project Model

Represents a study project owned by a user.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum

from ..engine import Base


class ProjectStatus(enum.Enum):
    """Project processing status"""
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Project(Base):
    """
    Project entity - owned by a user, contains documents and learning materials.
    """
    __tablename__ = "projects"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to user
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Project details
    name = Column(String(255), nullable=False)
    description = Column(String(1000), nullable=True)
    subject = Column(String(255), nullable=True)
    
    # Status and progress
    # values_callable persists lowercase VALUES ("processing") to match the
    # native PG enums created by migrations; without it SQLAlchemy binds
    # enum NAMES ("PROCESSING") and every write fails on Postgres.
    status = Column(
        SQLEnum(ProjectStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=ProjectStatus.PROCESSING,
        nullable=False
    )
    progress = Column(Integer, default=0, nullable=False)  # 0-100

    # Optional per-project LLM override: {"provider": "groq"|"openrouter",
    # "model": "<model id>"}. The API key still comes from the user's own
    # saved credential for that provider (or the app default when the user
    # has none). NULL means "no override" (user -> app default precedence).
    llm_override = Column(JSONB, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="projects")
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    flashcard_decks = relationship("FlashcardDeck", back_populates="project", cascade="all, delete-orphan")
    notes = relationship("Note", back_populates="project", cascade="all, delete-orphan")
    quizzes = relationship("Quiz", back_populates="project", cascade="all, delete-orphan")
    study_sessions = relationship("StudySession", back_populates="project", cascade="all, delete-orphan")
    generated_content_cache = relationship("GeneratedContentCache", back_populates="project", cascade="all, delete-orphan")
    websites = relationship("Website", back_populates="project", cascade="all, delete-orphan")
    # Academic spine (STEP 1, additive): project-scoped knowledge nodes.
    knowledge_nodes = relationship("KnowledgeNode", backref="project", cascade="all, delete-orphan")
    knowledge_models = relationship("KnowledgeModel", backref="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name}, status={self.status.value})>"