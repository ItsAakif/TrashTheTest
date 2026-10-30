"""
StudySession Model

Represents study sessions for tracking user learning progress.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from ..engine import Base


class SessionType(enum.Enum):
    """Type of study session"""
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    NOTES = "notes"
    TUTOR = "tutor"


class StudySession(Base):
    """
    StudySession entity - tracks user study activities and performance.
    """
    __tablename__ = "study_sessions"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign keys
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    
    # Session details (values_callable persists lowercase VALUES to match
    # the native PG enums created by migrations).
    session_type = Column(
        SQLEnum(SessionType, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False
    )
    
    # Timing
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Performance metrics
    score = Column(Float, nullable=True)  # 0-100 percentage
    duration_ms = Column(Integer, nullable=True)  # Duration in milliseconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="study_sessions")
    project = relationship("Project", back_populates="study_sessions")
    
    @property
    def duration_seconds(self) -> int:
        """Get duration in seconds"""
        if self.duration_ms:
            return self.duration_ms // 1000
        return 0
    
    def __repr__(self):
        return f"<StudySession(id={self.id}, type={self.session_type.value}, score={self.score})>"