"""
Study Session API Routes

REST API endpoints for study sessions, interactions, and analytics.
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from database import repositories
from database.models.user import User
from services.auth.deps import get_current_user

from .intent import TutorIntent, classify_tutor_intent, greeting_response
from .service import NotGeneratedError, session_manager
from .types import (
    SessionType,
    StartSessionRequest,
    SubmitResponseRequest,
    GetNextItemRequest,
    CompleteSessionRequest,
    SessionState,
    SessionResult,
    SessionAnalytics,
    UserStudyHistory,
    RecommendationResponse,
)


router = APIRouter(prefix="/api/study", tags=["study"])


async def _select_request_llm(project_id, current_user, db) -> None:
    """Install the caller's AI Provider selection for this request.

    Precedence: project override -> user credential -> app default.
    Resolution never raises; without a selection the engine uses the
    app default. The install is task-local (ContextVar).
    """
    from services.provider_credentials.override import resolve_for_request

    await resolve_for_request(db, current_user, project_id)


# ==================== SESSION MANAGEMENT ====================

@router.post("/session/start")
async def start_session(
    request: StartSessionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Start a new study session.

    Identity comes from the JWT; a client-supplied user_id is ignored.
    Missing generated content yields a typed not_generated error naming
    the generation call (never an empty session, never a 500).
    """
    try:
        await _select_request_llm(getattr(request, "project_id", None), current_user, db)
        session = await session_manager.start_session(
            request, db, user_id=str(current_user.id))

        current_item = session_manager.get_current_item(str(session.session_id))
        return {
            # Snake_case contract + camelCase aliases (additive, T406 pattern).
            "session_id": str(session.session_id),
            "sessionId": str(session.session_id),
            "session_type": session.session_type.value,
            "sessionType": session.session_type.value,
            "status": session.status.value,
            "total_items": len(session.items),
            "totalItems": len(session.items),
            "current_item": current_item,
            "currentItem": current_item,
        }

    except NotGeneratedError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.to_dict()
        )
    except (LookupError, PermissionError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {str(e)}"
        )


@router.get("/session/{session_id}/current")
async def get_current_item(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get the current item in an active session"""
    item = session_manager.get_current_item(session_id)
    
    if not item:
        # Check if session exists
        session = session_manager._sessions.get(session_id)
        if session and session.current_index >= len(session.items):
            return {
                "status": "completed",
                "message": "No more items in session"
            }
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or no current item"
        )
    
    return item


@router.post("/session/respond")
async def submit_response(
    request: SubmitResponseRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a response during an active session.
    
    Evaluates the response and returns feedback.
    """
    try:
        result = await session_manager.submit_response(request, db)
        
        return {
            "item_id": result.item_id,
            "is_correct": result.is_correct,
            "correct_answer": result.correct_answer,
            "feedback": result.feedback,
            "next_item": session_manager.get_current_item(request.session_id)
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process response: {str(e)}"
        )


@router.post("/session/complete")
async def complete_session(
    request: CompleteSessionRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Complete a study session and get final results.
    """
    try:
        result = await session_manager.complete_session(request, db)
        
        return {
            "session_id": str(result.session_id),
            "session_type": result.session_type.value,
            "total_items": result.total_items,
            "correct_count": result.correct_count,
            "incorrect_count": result.incorrect_count,
            "skipped_count": result.skipped_count,
            "score": result.score,
            "total_duration_ms": result.total_duration_ms,
            "avg_time_per_item": result.avg_time_per_item
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete session: {str(e)}"
        )


# ==================== ANALYTICS ====================

@router.get("/session/{session_id}/analytics")
async def get_session_analytics(
    session_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get analytics for a specific session"""
    analytics = session_manager.get_session_analytics(session_id)
    
    if not analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return analytics


@router.get("/history/{user_id}")
async def get_study_history(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a user's study history"""
    try:
        history = await session_manager.get_user_history(user_id, db)
        
        return {
            "user_id": str(history.user_id),
            "total_sessions": history.total_sessions,
            "total_study_time_ms": history.total_study_time_ms,
            "average_score": history.average_score,
            "recent_sessions": history.recent_sessions,
            "sessions_by_type": history.sessions_by_type
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}"
        )


# ==================== RECOMMENDATIONS ====================

@router.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get study recommendations for a user"""
    try:
        recommendations = await session_manager.get_recommendations(user_id, db)
        
        return {
            "recommendations": [
                {
                    "recommended_type": r.recommended_type.value,
                    "reason": r.reason,
                    "priority": r.priority,
                    "focus_areas": r.focus_areas,
                    "suggested_duration_minutes": r.suggested_duration_minutes
                }
                for r in recommendations.recommendations
            ],
            "based_on_recent_score": recommendations.based_on_recent_score,
            "weak_areas": [
                {
                    "topic": w.topic,
                    "error_count": w.error_count,
                    "suggested_action": w.suggested_action
                }
                for w in recommendations.weak_areas
            ]
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


# ==================== TUTOR MODE ====================

class TutorMessageRequest(BaseModel):
    """Request for tutor mode message (snake_case + camelCase aliases)."""
    session_id: Optional[str] = Field(default=None, alias="sessionId")
    project_id: Optional[str] = Field(default=None, alias="projectId")
    message: str

    model_config = {"populate_by_name": True}


# Cap on the topic vocabulary handed to the intent classifier: the match is a
# substring scan, so a runaway spine must not turn every message into a
# project-scoped retrieval.
_MAX_TOPIC_TITLES = 200
_MAX_TOPIC_NODES = 2000


async def _project_topic_titles(db: Any, project_uuid: Any) -> List[str]:
    """Best-effort titles of this project's real topics (never raises).

    The classifier already knows how to turn a known topic title into a
    project-scoped question (``topic_titles=...`` -> PROJECT_SPECIFIC), but
    the route never supplied them, so a bare topic query ("photosynthesis")
    fell through to the ambiguous branch and got the canned clarifier
    instead of a retrieved, cited answer.  Source of truth is the knowledge
    spine; the newest notes artifact's headings are the fallback for a
    project whose spine has not been built yet.  Any failure returns what
    has been collected so far: a tutor answer must never depend on this
    lookup succeeding.
    """
    titles: List[str] = []
    seen: set = set()

    def _add(value: Any) -> None:
        text = str(value or "").strip()
        key = text.casefold()
        if len(text) > 2 and key not in seen:
            seen.add(key)
            titles.append(text)

    nodes: List[Any] = []
    try:
        from database import knowledge_repo

        nodes = list(await knowledge_repo.list_knowledge_nodes(
            db, project_id=project_uuid
        ) or [])
    except Exception:
        nodes = []
    for node in nodes[:_MAX_TOPIC_NODES]:
        if str(getattr(node, "type", "") or "") in {"unit", "topic", "subtopic", "concept"}:
            _add(getattr(node, "title", None))
        if len(titles) >= _MAX_TOPIC_TITLES:
            return titles[:_MAX_TOPIC_TITLES]

    if not titles:
        # No spine yet: the generated notes still name the topics the
        # student has been studying, and they are project-scoped rows.
        try:
            notes = list(await repositories.get_notes_by_project(db, project_uuid) or [])
        except Exception:
            notes = []
        for note in notes[:3]:
            content = getattr(note, "content", None)
            content = content if isinstance(content, dict) else {}
            sections = content.get("sections") or []
            for section in sections[:200]:
                if isinstance(section, dict):
                    _add(section.get("heading") or section.get("title"))
            if len(titles) >= _MAX_TOPIC_TITLES:
                break
    return titles[:_MAX_TOPIC_TITLES]


def _coerce_text(value: object) -> str:
    """Coerce an LLM result to str (T408); raise a typed error otherwise."""
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    elif isinstance(value, dict):
        for key in ("text", "response", "answer", "content"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    raise ValueError("tutor provider returned a non-text result")


@router.post("/tutor/message")
async def tutor_message(
    request: TutorMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message in tutor mode.

    Uses RAG to answer questions based on document context. Accepts a
    project-scoped call (JWT user + project_id) without requiring an
    in-memory session; an unknown session_id yields a typed 404.
    The response is always a string with sources[] (never a raw object).
    """
    from services.retrieval.service import retrieval_service

    if not (request.message or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message is required"
        )

    session = None
    project_id = request.project_id
    if request.session_id:
        session = await session_manager.resolve_session(request.session_id, db)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "session_not_found",
                        "message": f"Session {request.session_id} not found"}
            )
        if str(session.user_id) != str(current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "session_not_found",
                        "message": f"Session {request.session_id} not found"}
            )
        if session.session_type != SessionType.TUTOR and session.items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tutor mode only"
            )
        if not session.items:
            # Restart-rebuilt state has no items: typed, never fabricated.
            if not project_id:
                project_id = str(session.project_id)
            if not project_id:
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail={"code": "session_not_resumable",
                            "message": "Session items are not persisted; "
                                       "start a new tutor session with project_id."}
                )
        else:
            project_id = project_id or str(session.project_id)

    if not project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_id or a live tutor session_id is required"
        )

    # Ownership gate (404 for another user's project, never content).
    try:
        project_uuid = uuid.UUID(str(project_id))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project_id format"
        )
    project = await repositories.get_project_by_id(db, project_uuid)
    if project is None or project.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found"
        )

    # Gate intent before retrieval.  Social messages must never trigger a
    # vector query, and unrelated/ambiguous requests must not receive an
    # academic dump from a generic model prompt.
    intent_result = classify_tutor_intent(
        request.message,
        project_name=str(getattr(project, "name", "") or ""),
    )
    if intent_result.intent is TutorIntent.AMBIGUOUS:
        # A bare topic name from THIS project is a real question, not a vague
        # one. Only the ambiguous branch pays for the lookup, so greetings,
        # unsupported asks and already-academic questions keep their
        # behaviour (and their no-retrieval / retrieval verdict) exactly.
        topic_titles = await _project_topic_titles(db, project_uuid)
        if topic_titles:
            retry = classify_tutor_intent(
                request.message,
                project_name=str(getattr(project, "name", "") or ""),
                topic_titles=topic_titles,
            )
            if retry.intent is not TutorIntent.AMBIGUOUS:
                intent_result = retry
    if intent_result.intent is TutorIntent.CONVERSATIONAL:
        return {
            "response": greeting_response(request.message),
            "sources": [],
            "citations": [],
            "intent": intent_result.intent.value,
            "retrieval_used": False,
            "grounded": False,
        }
    if intent_result.intent in {TutorIntent.UNSUPPORTED, TutorIntent.AMBIGUOUS}:
        return {
            "response": intent_result.clarification or (
                "I can only answer from your uploaded study material."
            ),
            "sources": [],
            "citations": [],
            "intent": intent_result.intent.value,
            "retrieval_used": False,
            "grounded": False,
        }

    # Use retrieval to get relevant context for the question
    try:
        retrieval_result = await retrieval_service.retrieve_with_db(
            query=request.message,
            db=db,
            project_id=str(project.id),
            top_k=3
        )
    except Exception as e:
        return {
            "response": (
                "I could not find enough source material for that question. "
                "Try naming a topic from your uploaded documents."
            ),
            "sources": [],
            "citations": [],
            "intent": intent_result.intent.value,
            "retrieval_used": False,
            "grounded": False,
            "error": "tutor_source_unavailable",
        }

    chunks = list(getattr(retrieval_result, "chunks", []) or [])

    # No evidence is an honest source gap, not an invitation for the model
    # to fill the answer from general knowledge.
    if not chunks:
        return {
            "response": (
                "I could not find a supporting passage in your uploaded material "
                "for that question. Please name a topic or upload the relevant source."
            ),
            "sources": [],
            "citations": [],
            "intent": intent_result.intent.value,
            "retrieval_used": True,
            "grounded": False,
            "error": "no_source_evidence",
        }

    # Build context
    context = "\n\n".join([c.content for c in chunks if getattr(c, "content", None)])

    # Generate response using LLM
    prompt = f"""You are a helpful tutor helping a student learn from their study materials.
Answer only from the source passages below. If they do not support an answer, say
that the uploaded material does not contain enough information.

Source passages:
{context}

Student question: {request.message}

Give a clear, educational answer. Do not introduce unrelated facts."""

    # Generate response using the caller's provider selection.
    await _select_request_llm(project_id, current_user, db)
    try:
        from services.provider_credentials.override import EffectiveLLMProvider

        llm = EffectiveLLMProvider()
        result = await llm.generate(prompt)
        raw = getattr(result, "text", result)
        response_text = _coerce_text(raw)

        citations = []
        for chunk in chunks:
            cid = getattr(chunk, "chunk_id", None)
            if not cid:
                continue
            citation = getattr(chunk, "citation", None)
            page = getattr(chunk, "page", None)
            span = getattr(chunk, "content", "")[:160]
            citations.append({
                "chunk_id": str(cid),
                "document_id": (
                    str(getattr(chunk, "document_id", "") or "")
                    or str(getattr(citation, "document_id", "") or "")
                    or None
                ),
                "page": page,
                "span": span,
                "method": getattr(citation, "method", None) if citation else None,
            })
        return {
            "response": response_text,
            "sources": [c.chunk_id for c in chunks if getattr(c, "chunk_id", None)],
            "citations": citations,
            # The prompt context is returned by the SAME retrieval that built
            # it, so the displayed "retrieval context" is exactly what the
            # answer was grounded on (never a second, divergent client
            # search).
            "context": context,
            "intent": intent_result.intent.value,
            "retrieval_used": True,
            "grounded": True,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"code": "tutor_bad_result", "message": str(e)}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tutor response failed: {str(e)}"
        )


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check():
    """Health check for study service"""
    return {
        "status": "healthy",
        "service": "study",
        "active_sessions": len(session_manager._sessions)
    }


__all__ = ["router"]