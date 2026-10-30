"""
Study Session Types

Pydantic models for study sessions, responses, analytics, and recommendations.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
from enum import Enum


class SessionType(str, Enum):
    """Type of study session"""
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    NOTES = "notes"
    TUTOR = "tutor"


class SessionStatus(str, Enum):
    """Status of a study session"""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


# ==================== REQUEST/RESPONSE MODELS ====================

class StartSessionRequest(BaseModel):
    """Request to start a study session.

    user_id is server-authoritative (JWT); a client-supplied id is ignored.
    sessionId/session_id and projectId/project_id aliases are accepted.
    """
    user_id: Optional[str] = Field(default=None, alias="userId")
    project_id: str = Field(alias="projectId")
    session_type: SessionType = Field(alias="sessionType")
    # For flashcards/quiz: optionally specify content source
    content_id: Optional[str] = Field(default=None, alias="contentId")
    # For tutor: initial context
    initial_query: Optional[str] = Field(default=None, alias="initialQuery")
    # Optional page size hint from the client
    limit: Optional[int] = None

    model_config = {"populate_by_name": True}


class SubmitResponseRequest(BaseModel):
    """Request to submit a response during a session"""
    session_id: str = Field(alias="sessionId")
    item_id: str = Field(alias="itemId")  # ID of flashcard or question
    response: Any  # Can be: selected option index, "correct/incorrect", or text
    time_taken_ms: Optional[int] = Field(default=None, alias="timeTakenMs")

    model_config = {"populate_by_name": True}


class GetNextItemRequest(BaseModel):
    """Request to get the next item in a session"""
    session_id: str = Field(alias="sessionId")

    model_config = {"populate_by_name": True}


class CompleteSessionRequest(BaseModel):
    """Request to complete a session"""
    session_id: str = Field(alias="sessionId")

    model_config = {"populate_by_name": True}


# ==================== SESSION STATE MODELS ====================

class FlashcardItem(BaseModel):
    """A flashcard item in a session"""
    card_id: str
    front: str
    back: str
    difficulty: str = "medium"


class QuizQuestionItem(BaseModel):
    """A quiz question item in a session"""
    question_id: str
    question: str
    options: List[str]
    correct_answer: int


class SessionItem(BaseModel):
    """An item in an active session"""
    item_id: str
    item_type: str  # "flashcard" or "question"
    content: Dict[str, Any]
    # For adaptive difficulty tracking
    difficulty_weight: float = 1.0


class SessionState(BaseModel):
    """Current state of an active study session"""
    session_id: uuid.UUID
    user_id: uuid.UUID
    project_id: uuid.UUID
    session_type: SessionType
    status: SessionStatus
    
    # Items in the session
    items: List[SessionItem]
    current_index: int
    
    # Responses recorded
    responses: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Timing
    started_at: datetime
    last_activity_at: Optional[datetime] = None
    
    # Adaptive difficulty state
    current_difficulty: float = 1.0  # 0.5 to 2.0 multiplier


class SessionResponse(BaseModel):
    """A recorded response in a session"""
    item_id: str
    response: Any
    is_correct: bool
    time_taken_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== RESULT MODELS ====================

class ItemResult(BaseModel):
    """Result of a single item (flashcard/question)"""
    item_id: str
    is_correct: bool
    correct_answer: Optional[Any] = None
    feedback: Optional[str] = None


class SessionResult(BaseModel):
    """Result after completing a session"""
    session_id: uuid.UUID
    session_type: SessionType
    
    # Performance metrics
    total_items: int
    correct_count: int
    incorrect_count: int
    skipped_count: int
    score: float  # 0-100 percentage
    
    # Timing
    total_duration_ms: int
    avg_time_per_item: Optional[float] = None
    
    # Per-item results
    item_results: List[ItemResult] = Field(default_factory=list)


# ==================== ANALYTICS MODELS ====================

class SessionAnalytics(BaseModel):
    """Analytics for a study session"""
    session_id: uuid.UUID
    
    # Overview
    total_items: int
    correct_count: int
    incorrect_count: int
    score: float
    
    # Timing
    total_duration_ms: int
    avg_time_per_item_ms: float
    
    # Performance by difficulty (if available)
    performance_by_difficulty: Optional[Dict[str, float]] = None
    
    # Weak areas identified
    weak_areas: List[str] = Field(default_factory=list)


class UserStudyHistory(BaseModel):
    """User's overall study history"""
    user_id: uuid.UUID
    
    # Summary stats
    total_sessions: int
    total_study_time_ms: int
    average_score: float
    
    # Recent sessions
    recent_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Mode breakdown
    sessions_by_type: Dict[str, int] = Field(default_factory=dict)


class WeakArea(BaseModel):
    """A detected weak area for a user"""
    topic: str
    error_count: int
    recent_attempts: int
    suggested_action: str


# ==================== RECOMMENDATION MODELS ====================

class StudyRecommendation(BaseModel):
    """Recommendation for what to study next"""
    recommended_type: SessionType  # Which mode to use
    reason: str
    priority: int  # 1 = highest priority
    
    # Optional: specific content to focus on
    focus_areas: List[str] = Field(default_factory=list)
    
    # Estimated time
    suggested_duration_minutes: int = 15


class RecommendationResponse(BaseModel):
    """Response with study recommendations"""
    recommendations: List[StudyRecommendation]
    based_on_recent_score: Optional[float] = None
    weak_areas: List[WeakArea] = Field(default_factory=list)


__all__ = [
    # Enums
    "SessionType",
    "SessionStatus",
    # Request/Response
    "StartSessionRequest",
    "SubmitResponseRequest",
    "GetNextItemRequest",
    "CompleteSessionRequest",
    # Session State
    "FlashcardItem",
    "QuizQuestionItem",
    "SessionItem",
    "SessionState",
    "SessionResponse",
    # Results
    "ItemResult",
    "SessionResult",
    # Analytics
    "SessionAnalytics",
    "UserStudyHistory",
    "WeakArea",
    # Recommendations
    "StudyRecommendation",
    "RecommendationResponse",
]