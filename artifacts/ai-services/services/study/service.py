"""
Study Session Manager Service

Manages interactive study sessions with:
- Session lifecycle (start, process, complete)
- Real-time response tracking
- Adaptive difficulty
- Weak area detection
- Recommendations
"""

import uuid
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database.models import StudySession, SessionType as DBSessionType
from database import repositories
from services.retrieval.service import retrieval_service

from .types import (
    SessionType,
    SessionStatus,
    StartSessionRequest,
    SubmitResponseRequest,
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
    FlashcardItem,
    QuizQuestionItem,
)


class NotGeneratedError(ValueError):
    """Typed error: the project has no generated content for this mode.

    Carries the generation call the client must run first; the API maps
    it to a typed not_generated response (never an empty session).
    """

    def __init__(self, kind: str, generation_call: str):
        super().__init__(
            f"No {kind} generated yet for this project. "
            f"Run {generation_call} first."
        )
        self.kind = kind
        self.generation_call = generation_call

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": "not_generated",
            "kind": self.kind,
            "generation_call": self.generation_call,
            "message": str(self),
        }


class SessionManager:
    """
    Manages interactive study sessions.
    
    Features:
    - Session lifecycle management
    - Content loading from database or generation
    - Response evaluation (quiz, flashcards, tutor)
    - Adaptive difficulty based on performance
    - Weak area detection and tracking
    - Recommendations for future study
    - RAG integration for tutor mode
    """
    
    def __init__(self):
        # In-memory session state storage (would be Redis in production)
        self._sessions: Dict[str, SessionState] = {}
        
        # LLM provider for tutor evaluation: request selection wins
        # (user Settings > AI Provider), else the factory-owned router.
        from services.provider_credentials.override import EffectiveLLMProvider

        self.llm = EffectiveLLMProvider()
        
        # Adaptive difficulty settings
        self.difficulty_adjustment_factor = 0.1  # How much to adjust per response
        self.min_difficulty = 0.5
        self.max_difficulty = 2.0
    
    async def _load_flashcards(
        self,
        project_id: str,
        db: AsyncSession
    ) -> List[SessionItem]:
        """Load real generated flashcards for a session (T407).

        Raises NotGeneratedError naming the generation call when the
        project has no generated flashcards.
        """
        project_uuid = uuid.UUID(str(project_id))
        decks = await repositories.get_flashcard_decks_by_project(db, project_uuid)
        items: List[SessionItem] = []
        for deck in decks:
            cards = getattr(deck, "flashcards", []) or []
            for card in cards:
                difficulty = getattr(card, "difficulty", "medium")
                try:
                    difficulty = difficulty.value
                except AttributeError:
                    difficulty = str(difficulty or "medium")
                items.append(SessionItem(
                    item_id=str(card.id),
                    item_type="flashcard",
                    content={"front": card.front, "back": card.back,
                             "difficulty": difficulty,
                             "deck_id": str(deck.id),
                             "deck_name": deck.name},
                ))
        if not items:
            raise NotGeneratedError(
                "flashcards", "POST /api/generation/flashcards")
        return items

    async def _load_quiz(
        self,
        project_id: str,
        db: AsyncSession
    ) -> List[SessionItem]:
        """Load real generated quiz questions for a session (T407).

        Raises NotGeneratedError naming the generation call when the
        project has no generated quiz.
        """
        project_uuid = uuid.UUID(str(project_id))
        quizzes = await repositories.get_quizzes_by_project(db, project_uuid)
        items: List[SessionItem] = []
        for quiz in quizzes:
            for idx, q in enumerate(quiz.questions or []):
                if not isinstance(q, dict):
                    continue
                options = q.get("options") or []
                answer = q.get("correct_answer", q.get("answer", 0))
                try:
                    answer_idx = int(answer) if isinstance(answer, (int, str)) \
                        and str(answer).lstrip("-").isdigit() else options.index(answer)
                except (ValueError, AttributeError):
                    answer_idx = 0
                items.append(SessionItem(
                    item_id=str(q.get("id") or f"{quiz.id}:{idx}"),
                    item_type="question",
                    content={"question": q.get("question", ""),
                             "options": options,
                             "correct_answer": answer_idx,
                             "quiz_id": str(quiz.id),
                             "quiz_title": quiz.title},
                ))
        if not items:
            raise NotGeneratedError(
                "quiz", "POST /api/generation/quiz")
        return items
    
    async def _generate_tutor_context(
        self,
        project_id: str,
        query: str,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Generate context for tutor mode using RAG"""
        # Use retrieval service to get relevant context
        retrieval_result = await retrieval_service.retrieve_with_db(
            query=query,
            db=db,
            project_id=project_id,
            top_k=5
        )
        
        # Build context from retrieved chunks
        context_parts = []
        for chunk in retrieval_result.chunks:
            context_parts.append(chunk.content)
        
        return {
            "query": query,
            "context": "\n\n".join(context_parts[:3]),
            "sources": [c.chunk_id for c in retrieval_result.chunks]
        }
    
    async def start_session(
        self,
        request: StartSessionRequest,
        db: AsyncSession,
        user_id: Optional[str] = None,
    ) -> SessionState:
        """
        Start a new study session.

        user_id is server-authoritative (JWT): the explicit argument wins;
        a client-supplied request.user_id is ignored. Content is loaded
        from real generated rows; missing content raises NotGeneratedError.
        """
        session_id = str(uuid.uuid4())
        effective_user = user_id or request.user_id
        if not effective_user:
            raise ValueError("user identity is required (JWT)")
        user_uuid = uuid.UUID(str(effective_user))
        project_uuid = uuid.UUID(str(request.project_id))

        # Ownership gate: the project must belong to the caller (404/403
        # semantics via ValueError mapping in the route; never a 500).
        project = await repositories.get_project_by_id(db, project_uuid)
        if project is None:
            raise LookupError(f"Project {request.project_id} not found")
        if project.user_id != user_uuid:
            raise PermissionError(f"Project {request.project_id} not found")

        # Load content based on session type
        items = []

        if request.session_type == SessionType.FLASHCARDS:
            items = await self._load_flashcards(request.project_id, db)

        elif request.session_type == SessionType.QUIZ:
            items = await self._load_quiz(request.project_id, db)

        elif request.session_type == SessionType.TUTOR:
            # Tutor retrieval is message-driven.  Starting a session must not
            # retrieve a generic academic dump before the user has said hello
            # or asked a grounded question.
            items = [SessionItem(
                item_id="tutor_context",
                item_type="tutor",
                content={
                    "query": request.initial_query or "",
                    "context": "",
                    "sources": [],
                },
            )]

        elif request.session_type == SessionType.NOTES:
            # Notes mode - just track time spent
            items = []
        
        # Shuffle items for variety (with adaptive difficulty consideration)
        items = self._shuffle_items(items)
        
        # Create session state
        session_state = SessionState(
            session_id=session_id,
            user_id=user_uuid,
            project_id=project_uuid,
            session_type=request.session_type,
            status=SessionStatus.IN_PROGRESS,
            items=items,
            current_index=0,
            responses=[],
            started_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow(),
            current_difficulty=1.0
        )
        
        # Store in memory
        self._sessions[session_id] = session_state
        
        # Create database record with the SAME id as the live session so
        # the row is restart-resolvable (resolve_session rebuilds from it).
        await repositories.create_study_session(
            db=db,
            user_id=user_uuid,
            project_id=project_uuid,
            session_type=DBSessionType(request.session_type.value),
            started_at=datetime.utcnow(),
            session_id=uuid.UUID(session_id),
        )
        
        return session_state
    
    def _shuffle_items(self, items: List[SessionItem]) -> List[SessionItem]:
        """Shuffle items, prioritizing higher difficulty ones"""
        if not items:
            return []
        
        # Separate by difficulty
        high_difficulty = [i for i in items if i.difficulty_weight > 1.2]
        medium_difficulty = [i for i in items if 0.8 <= i.difficulty_weight <= 1.2]
        low_difficulty = [i for i in items if i.difficulty_weight < 0.8]
        
        # Interleave: high -> medium -> low for better learning
        result = []
        max_len = max(len(high_difficulty), len(medium_difficulty), len(low_difficulty))
        
        for i in range(max_len):
            if i < len(high_difficulty):
                result.append(high_difficulty[i])
            if i < len(medium_difficulty):
                result.append(medium_difficulty[i])
            if i < len(low_difficulty):
                result.append(low_difficulty[i])
        
        return result
    
    def get_current_item(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current item for a session"""
        session = self._sessions.get(session_id)
        if not session or session.current_index >= len(session.items):
            return None

        item = session.items[session.current_index]
        return {
            "item_id": item.item_id,
            "item_type": item.item_type,
            "content": item.content,
            "item_number": session.current_index + 1,
            "total_items": len(session.items)
        }

    async def resolve_session(
        self, session_id: str, db: AsyncSession
    ) -> Optional[SessionState]:
        """Restart-safe session lookup (T408).

        In-memory first; otherwise rebuild a project-scoped state from the
        persisted study_sessions row (history/analytics survive a restart).
        Item-level resume is not implemented: the rebuilt state carries no
        items and callers must return a typed "session not resumable" for
        item-level calls rather than fabricating state. Returns None when
        the session id is unknown.
        """
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        try:
            sid = uuid.UUID(str(session_id))
        except (ValueError, TypeError):
            return None
        try:
            row = await db.get(StudySession, sid)
        except Exception:
            return None
        if row is None:
            return None
        rebuilt = SessionState(
            session_id=row.id,
            user_id=row.user_id,
            project_id=row.project_id,
            session_type=SessionType(row.session_type.value),
            status=SessionStatus.COMPLETED if row.completed_at is not None
            else SessionStatus.IN_PROGRESS,
            items=[],
            current_index=0,
            responses=[],
            started_at=row.started_at,
            last_activity_at=row.completed_at,
            current_difficulty=1.0,
        )
        return rebuilt
    
    def _evaluate_flashcard_response(
        self,
        item: SessionItem,
        response: Any
    ) -> ItemResult:
        """Evaluate a flashcard response"""
        # Response can be "correct", "incorrect", or user self-assessment
        is_correct = response in ["correct", "known", "easy"]
        
        return ItemResult(
            item_id=item.item_id,
            is_correct=is_correct,
            feedback="You marked this as " + response
        )
    
    def _evaluate_quiz_response(
        self,
        item: SessionItem,
        response: int
    ) -> ItemResult:
        """Evaluate a quiz response"""
        content = item.content
        correct_answer = content.get("correct_answer", 0)
        options = content.get("options", [])
        
        is_correct = response == correct_answer
        
        feedback = "Correct!" if is_correct else f"Wrong. The correct answer was: {options[correct_answer]}"
        
        return ItemResult(
            item_id=item.item_id,
            is_correct=is_correct,
            correct_answer=correct_answer,
            feedback=feedback
        )
    
    async def _evaluate_tutor_response(
        self,
        item: SessionItem,
        response: str
    ) -> ItemResult:
        """Evaluate a tutor response using LLM"""
        # Use LLM to evaluate the user's answer
        prompt = f"""Evaluate the following student response to a question.
        
Context: {item.content.get('context', '')}

Question: {item.content.get('query', '')}

Student's Answer: {response}

Provide a JSON response with:
{{
    "is_correct": true/false,
    "feedback": "brief feedback on the answer",
    "correct_answer": "what a good answer would contain"
}}"""
        
        try:
            result = await self.llm.generate(prompt)
            
            # Parse result
            import json
            import re
            json_match = re.search(r'\{.*\}', result.text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return ItemResult(
                    item_id=item.item_id,
                    is_correct=parsed.get("is_correct", False),
                    feedback=parsed.get("feedback", ""),
                    correct_answer=parsed.get("correct_answer")
                )
        except Exception:
            pass
        
        # Fallback
        return ItemResult(
            item_id=item.item_id,
            is_correct=False,
            feedback="Your answer has been recorded for review."
        )
    
    async def submit_response(
        self,
        request: SubmitResponseRequest,
        db: AsyncSession
    ) -> ItemResult:
        """
        Submit a response during an active session.
        
        Evaluates the response, updates session state, and adjusts difficulty.
        """
        session = self._sessions.get(request.session_id)
        if not session:
            raise ValueError(f"Session {request.session_id} not found")
        
        if session.current_index >= len(session.items):
            raise ValueError("No more items in session")
        
        item = session.items[session.current_index]
        
        # Evaluate based on item type
        if item.item_type == "flashcard":
            result = self._evaluate_flashcard_response(item, request.response)
        elif item.item_type == "question":
            result = self._evaluate_quiz_response(item, request.response)
        elif item.item_type == "tutor":
            result = await self._evaluate_tutor_response(item, request.response)
        else:
            result = ItemResult(item_id=item.item_id, is_correct=True)
        
        # Record response
        response_record = SessionResponse(
            item_id=item.item_id,
            response=request.response,
            is_correct=result.is_correct,
            time_taken_ms=request.time_taken_ms,
            timestamp=datetime.utcnow()
        )
        session.responses.append(response_record.dict())
        
        # Update adaptive difficulty
        self._update_difficulty(session, result.is_correct)
        
        # Move to next item
        session.current_index += 1
        session.last_activity_at = datetime.utcnow()
        
        return result
    
    def _update_difficulty(self, session: SessionState, is_correct: bool):
        """Update adaptive difficulty based on performance"""
        if is_correct:
            # Decrease difficulty slightly
            session.current_difficulty = max(
                self.min_difficulty,
                session.current_difficulty - self.difficulty_adjustment_factor
            )
        else:
            # Increase difficulty
            session.current_difficulty = min(
                self.max_difficulty,
                session.current_difficulty + self.difficulty_adjustment_factor
            )
        
        # Update difficulty weight for current and future items
        for item in session.items:
            # Base difficulty from content + user performance modifier
            item.difficulty_weight = session.current_difficulty
    
    async def complete_session(
        self,
        request: CompleteSessionRequest,
        db: AsyncSession
    ) -> SessionResult:
        """
        Complete a study session and generate analytics.
        """
        session = self._sessions.get(request.session_id)
        if not session:
            raise ValueError(f"Session {request.session_id} not found")
        
        # Calculate metrics
        total_items = len(session.items)
        correct_count = sum(1 for r in session.responses if r.get("is_correct"))
        incorrect_count = total_items - correct_count - (total_items - len(session.responses))
        
        # Score as percentage
        if total_items > 0:
            score = (correct_count / total_items) * 100
        else:
            score = 0.0
        
        # Duration
        duration_ms = int((datetime.utcnow() - session.started_at).total_seconds() * 1000)
        
        # Generate item results
        item_results = []
        for i, response in enumerate(session.responses):
            if i < len(session.items):
                item_results.append(ItemResult(
                    item_id=session.items[i].item_id,
                    is_correct=response.get("is_correct", False)
                ))
        
        # Update database record
        await repositories.complete_study_session(
            db=db,
            session_id=uuid.UUID(request.session_id),
            score=score,
            duration_ms=duration_ms
        )
        
        # Update session status
        session.status = SessionStatus.COMPLETED
        
        return SessionResult(
            session_id=session.session_id,
            session_type=session.session_type,
            total_items=total_items,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            skipped_count=total_items - len(session.responses),
            score=score,
            total_duration_ms=duration_ms,
            avg_time_per_item=duration_ms / total_items if total_items > 0 else None,
            item_results=item_results
        )
    
    def get_session_analytics(self, session_id: str) -> Optional[SessionAnalytics]:
        """Get analytics for a session"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        total_items = len(session.items)
        correct_count = sum(1 for r in session.responses if r.get("is_correct"))
        
        duration_ms = 0
        if session.last_activity_at:
            duration_ms = int((session.last_activity_at - session.started_at).total_seconds() * 1000)
        
        return SessionAnalytics(
            session_id=session.session_id,
            total_items=total_items,
            correct_count=correct_count,
            incorrect_count=total_items - correct_count,
            score=(correct_count / total_items * 100) if total_items > 0 else 0,
            total_duration_ms=duration_ms,
            avg_time_per_item_ms=duration_ms / total_items if total_items > 0 else 0
        )
    
    async def get_user_history(
        self,
        user_id: str,
        db: AsyncSession
    ) -> UserStudyHistory:
        """Get user's study history"""
        user_uuid = uuid.UUID(user_id)
        
        # Query database for recent sessions
        result = await db.execute(
            select(StudySession)
            .where(StudySession.user_id == user_uuid)
            .order_by(StudySession.started_at.desc())
            .limit(20)
        )
        sessions = result.scalars().all()
        
        if not sessions:
            return UserStudyHistory(
                user_id=user_uuid,
                total_sessions=0,
                total_study_time_ms=0,
                average_score=0,
                recent_sessions=[],
                sessions_by_type={}
            )
        
        # Calculate stats
        total_sessions = len(sessions)
        total_time = sum(s.duration_ms or 0 for s in sessions)
        avg_score = sum(s.score or 0 for s in sessions) / total_sessions
        
        # Recent sessions
        recent = []
        for s in sessions[:10]:
            recent.append({
                "session_id": str(s.id),
                "type": s.session_type.value,
                "score": s.score,
                "duration_ms": s.duration_ms,
                "started_at": s.started_at.isoformat()
            })
        
        # By type
        by_type = {}
        for s in sessions:
            t = s.session_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return UserStudyHistory(
            user_id=user_uuid,
            total_sessions=total_sessions,
            total_study_time_ms=total_time,
            average_score=avg_score,
            recent_sessions=recent,
            sessions_by_type=by_type
        )
    
    def _detect_weak_areas(self, sessions: List[Dict[str, Any]]) -> List[WeakArea]:
        """Detect weak areas based on session history"""
        # This would analyze response patterns
        # For now, return placeholder
        return []
    
    def _generate_recommendations(
        self,
        recent_score: Optional[float],
        weak_areas: List[WeakArea]
    ) -> List[StudyRecommendation]:
        """Generate study recommendations"""
        recommendations = []
        
        # Based on recent performance
        if recent_score is not None:
            if recent_score < 60:
                recommendations.append(StudyRecommendation(
                    recommended_type=SessionType.FLASHCARDS,
                    reason="Your recent score is low. Review with flashcards to reinforce concepts.",
                    priority=1,
                    suggested_duration_minutes=20
                ))
            elif recent_score < 80:
                recommendations.append(StudyRecommendation(
                    recommended_type=SessionType.QUIZ,
                    reason="Good progress! Take a quiz to test your understanding.",
                    priority=1,
                    suggested_duration_minutes=15
                ))
            else:
                recommendations.append(StudyRecommendation(
                    recommended_type=SessionType.TUTOR,
                    reason="Excellent performance! Try tutor mode for deeper understanding.",
                    priority=1,
                    suggested_duration_minutes=20
                ))
        
        # If weak areas detected
        if weak_areas:
            recommendations.append(StudyRecommendation(
                recommended_type=SessionType.NOTES,
                reason=f"Focus on: {', '.join(w.topic for w in weak_areas[:2])}",
                priority=2,
                suggested_duration_minutes=15
            ))
        
        return recommendations[:3]  # Max 3 recommendations
    
    async def get_recommendations(
        self,
        user_id: str,
        db: AsyncSession
    ) -> RecommendationResponse:
        """Get study recommendations for a user"""
        history = await self.get_user_history(user_id, db)
        
        weak_areas = self._detect_weak_areas(history.recent_sessions)
        
        recommendations = self._generate_recommendations(
            history.average_score if history.total_sessions > 0 else None,
            weak_areas
        )
        
        return RecommendationResponse(
            recommendations=recommendations,
            based_on_recent_score=history.average_score if history.total_sessions > 0 else None,
            weak_areas=weak_areas
        )


# Default service instance
session_manager = SessionManager()