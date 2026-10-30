"""
Flashcard Models

FlashcardDeck and Flashcard entities for spaced repetition learning.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from ..engine import Base


class Difficulty(enum.Enum):
    """Flashcard difficulty level"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class FlashcardDeck(Base):
    """
    FlashcardDeck entity - collection of flashcards within a project.
    """
    __tablename__ = "flashcard_decks"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to project
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    
    # Deck details
    name = Column(String(255), nullable=False)
    card_count = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="flashcard_decks")
    flashcards = relationship("Flashcard", back_populates="deck", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<FlashcardDeck(id={self.id}, name={self.name}, cards={self.card_count})>"


class Flashcard(Base):
    """
    Flashcard entity - individual flashcard within a deck.
    """
    __tablename__ = "flashcards"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to deck
    deck_id = Column(UUID(as_uuid=True), ForeignKey("flashcard_decks.id"), nullable=False, index=True)
    
    # Flashcard content
    front = Column(String(2000), nullable=False)  # Question/prompt
    back = Column(String(5000), nullable=False)     # Answer/explanation
    
    # Learning metadata (values_callable persists lowercase VALUES to match
    # the native PG enums created by migrations).
    difficulty = Column(
        SQLEnum(Difficulty, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=Difficulty.MEDIUM,
        nullable=False
    )
    times_reviewed = Column(Integer, default=0, nullable=False)
    times_correct = Column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    deck = relationship("FlashcardDeck", back_populates="flashcards")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.times_reviewed == 0:
            return 0.0
        return (self.times_correct / self.times_reviewed) * 100
    
    def __repr__(self):
        return f"<Flashcard(id={self.id}, front={self.front[:50]}...)>"