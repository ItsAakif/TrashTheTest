"""
Study Session Module

Interactive study sessions with:
- Session lifecycle management
- Adaptive difficulty
- Weak area detection
- Recommendations
- RAG-powered tutor mode
"""

from .types import (
    SessionType,
    SessionStatus,
    StartSessionRequest,
    SubmitResponseRequest,
    GetNextItemRequest,
    CompleteSessionRequest,
    SessionState,
    SessionItem,
    SessionResponse,
    ItemResult,
    SessionResult,
    SessionAnalytics,
    UserStudyHistory,
    WeakArea,
    StudyRecommendation,
    RecommendationResponse,
)

from .intent import TutorIntent, TutorIntentResult, classify_tutor_intent, greeting_response
from .service import SessionManager, session_manager

from .routes import router


__all__ = [
    # Types
    "SessionType",
    "SessionStatus",
    "StartSessionRequest",
    "SubmitResponseRequest",
    "GetNextItemRequest",
    "CompleteSessionRequest",
    "SessionState",
    "SessionItem",
    "SessionResponse",
    "ItemResult",
    "SessionResult",
    "SessionAnalytics",
    "UserStudyHistory",
    "WeakArea",
    "StudyRecommendation",
    "RecommendationResponse",
    "TutorIntent",
    "TutorIntentResult",
    "classify_tutor_intent",
    "greeting_response",
    # Service
    "SessionManager",
    "session_manager",
    # Routes
    "router",
]