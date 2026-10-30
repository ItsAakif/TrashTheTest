"""
Transformation Engine API Routes

REST API endpoints for content generation using the unified transformation engine.
"""

import asyncio
import os
import uuid
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import require_project_access
from database.engine import get_db
from database import repositories
from database.models.project import Project
from database.models.user import User
from services.auth.deps import get_current_user

from .engine import TransformationEngine, transformation_engine
from .types import GenerationMode, GenerationRequest, GenerationResult


router = APIRouter(prefix="/api/generation", tags=["generation"])

# Practice generation includes a cold model load on the first call after a
# restart (Qwen embedding + bge-reranker). Bound the request so a slow
# download returns a typed 504 instead of holding the connection until the
# browser/client deadline fires. Overridable for slow boxes.
try:
    PRACTICE_TIMEOUT_S = float(os.getenv("PRACTICE_TIMEOUT_S", "180"))
except ValueError:  # pragma: no cover - misconfigured env
    PRACTICE_TIMEOUT_S = 180.0

PRACTICE_TIMEOUT_DETAIL = (
    "Practice generation timed out while loading models; retry in a moment"
)


async def _select_request_llm(request, current_user, db) -> None:
    """Install the caller's AI Provider selection for this request.

    Precedence: project override -> user credential -> app default
    (services.provider_credentials). Resolution never raises; without a
    selection the engine uses the app default. The install is
    task-local (ContextVar), so concurrent requests never share keys.
    """
    from services.provider_credentials.override import resolve_for_request

    await resolve_for_request(db, current_user, getattr(request, "project_id", None))


class _AliasedRequest(BaseModel):
    """Snake_case contract with camelCase aliases (T406/ADR-006 A6).

    The frontend sends snake_case; generated clients may send camelCase.
    A 422 on these routes is a regression.
    """

    model_config = {"populate_by_name": True}


class NotesRequest(_AliasedRequest):
    """Request for generating notes"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class FlashcardsRequest(_AliasedRequest):
    """Request for generating flashcards"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class QuizRequest(_AliasedRequest):
    """Request for generating quiz"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class ExplanationRequest(_AliasedRequest):
    """Request for generating explanation"""
    project_id: str = Field(alias="projectId")
    query: str
    enable_research: bool = Field(default=False, alias="enableResearch")


class SummaryRequest(_AliasedRequest):
    """Request for generating summary"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class WebsiteRequest(_AliasedRequest):
    """Request for generating website content"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class AnswerRequest(_AliasedRequest):
    """Request for generating a marks-aware grounded answer"""
    project_id: str = Field(alias="projectId")
    query: str
    marks: int = 5
    style: str = "detailed"
    enable_research: bool = Field(default=False, alias="enableResearch")


class MCQRequest(_AliasedRequest):
    """Request for generating a strict MCQ set"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class SyllabusRequest(_AliasedRequest):
    """Request for extracting a syllabus (units + topics)"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class PYQRequest(_AliasedRequest):
    """Request for extracting previous-year questions"""
    project_id: str = Field(alias="projectId")
    query: Optional[str] = None
    enable_research: bool = Field(default=False, alias="enableResearch")


class MappingRequest(_AliasedRequest):
    """Request for mapping PYQs to syllabus units (deterministic).

    Inline syllabus/questions override the project's latest cached
    extraction when supplied; otherwise cached rows are used.
    """
    project_id: str = Field(alias="projectId")
    syllabus: Optional[dict] = None
    questions: Optional[list] = None
    min_score: float = Field(default=0.15, alias="minScore")


class AnalysisRequest(_AliasedRequest):
    """Request for frequency / marks alignment analysis (deterministic)."""
    project_id: str = Field(alias="projectId")
    syllabus: Optional[dict] = None
    questions: Optional[list] = None
    mappings: Optional[list] = None
    min_score: float = Field(default=0.15, alias="minScore")


def _completion_state(kind: str, content: Any) -> dict:
    """Honest completion state for a single-call generation payload.

    A headings-only notes payload or a zero-card deck completed the LLM call
    but produced no artifact. Reporting a bare 200 that reads as a finished
    result is how an empty deck/note became "ready" downstream, so the
    envelope carries an explicit state and a retry hint instead. Nothing is
    dropped from `content`: existing callers keep their shape.
    """
    from services.academic.validation import is_placeholder

    payload = content if isinstance(content, dict) else {}
    if kind == "notes":
        bodies = [
            str(section.get("content") or "")
            for section in (payload.get("sections") or [])
            if isinstance(section, dict)
        ]
    else:
        bodies = [
            f"{card.get('front')} {card.get('back')}"
            for card in (payload.get("flashcards") or [])
            if isinstance(card, dict)
        ]
    if any(not is_placeholder(body) for body in bodies):
        return {"state": "ready"}
    return {
        "state": "empty",
        "message": (
            f"Generation returned no usable {kind}; nothing was persisted. "
            "Add source material or try again."
        ),
    }


# ==================== NOTES ====================

@router.post("/notes")
async def generate_notes(
    request: NotesRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate structured study notes"""
    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)
        
        # Generate notes
        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_notes(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research
        )
        
        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time,
            "strategy": "single_shot",
            **_completion_state("notes", result.content),
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )


# ==================== FLASHCARDS ====================

@router.post("/flashcards")
async def generate_flashcards(
    request: FlashcardsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate flashcards"""
    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)
        
        # Generate flashcards
        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_flashcards(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research
        )
        
        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time,
            **_completion_state("flashcards", result.content),
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )


# ==================== QUIZ ====================

@router.post("/quiz")
async def generate_quiz(
    request: QuizRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate quiz questions"""
    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)
        
        # Generate quiz
        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_quiz(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research
        )
        
        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )


# ==================== EXPLANATION ====================

@router.post("/explanation")
async def generate_explanation(
    request: ExplanationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate explanation for a concept"""
    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)
        
        # Validate query is provided
        if not request.query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required for explanation generation"
            )
        
        # Generate explanation
        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_explanation(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research
        )
        
        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )
    except Exception as exc:
        from services.strategies.llm.structured import StructuredGenerationError
        from services.strategies.llm.router import ProviderUnavailable

        if isinstance(exc, (StructuredGenerationError, ProviderUnavailable)):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "code": "generation_provider_failed",
                    "message": str(exc),
                },
            ) from exc
        raise


# ==================== SUMMARY ====================

@router.post("/summary")
async def generate_summary(
    request: SummaryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate summary of document content"""
    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)
        
        # Generate summary
        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_summary(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research
        )
        
        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )


# ==================== WEBSITE ====================

@router.post("/website")
async def generate_website(
    request: WebsiteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate website content for study page"""
    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)
        
        # Generate website content
        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_website(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research
        )
        
        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format"
        )


# ==================== ANSWER (marks-aware) ====================

@router.post("/answer")
async def generate_answer(
    request: AnswerRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a marks-aware grounded answer.

    Word-count constraints adapt by marks via constraints_for_marks
    (1->15-60 ... 10->300-700, interpolated). Query is required;
    marks must be a positive integer (validated to 1..100).
    """
    from .types import constraints_for_marks

    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)

        if not (request.query or "").strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required for answer generation",
            )
        try:
            marks = int(request.marks)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="marks must be a positive integer",
            )
        if marks < 1 or marks > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="marks must be between 1 and 100",
            )
        band = constraints_for_marks(marks)

        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_answer(
            project_id=request.project_id,
            db=db,
            query=request.query,
            marks=marks,
            style=request.style or "detailed",
            enable_research=request.enable_research,
        )

        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time,
            "marks": marks,
            "word_band": {"min_words": band["min_words"], "max_words": band["max_words"]},
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )


# ==================== MCQ (strict) ====================

@router.post("/mcq")
async def generate_mcq(
    request: MCQRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a strict MCQ set (exactly 4 distinct options per question)."""
    try:
        # Ownership via the shared dependency (404 for another user's project).
        project = await require_project_access(request.project_id, current_user, db)

        await _select_request_llm(request, current_user, db)
        try:
            result = await asyncio.wait_for(
                transformation_engine.generate_mcq(
                    project_id=request.project_id,
                    db=db,
                    query=request.query,
                    enable_research=request.enable_research,
                ),
                timeout=PRACTICE_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, TimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=PRACTICE_TIMEOUT_DETAIL,
            )

        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": result.content,
            "cached": result.cached,
            "processing_time": result.processing_time,
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )
    except Exception as exc:
        from services.strategies.llm.structured import StructuredGenerationError
        from services.strategies.llm.router import ProviderUnavailable

        if isinstance(exc, (StructuredGenerationError, ProviderUnavailable)):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "code": "generation_provider_failed",
                    "message": str(exc),
                },
            ) from exc
        raise


# ==================== ACADEMIC INTELLIGENCE ====================
# Syllabus extraction + PYQ extraction (LLM, grounded via the shared
# retrieval -> structured pipeline) + deterministic mapping + analysis.
# Display strings are sanitised at response time (storage keeps raw).

def _sanitised_academic(content: dict) -> dict:
    from .engine import sanitise_display_text

    out = dict(content)
    if isinstance(out.get("units"), list):
        units = []
        for u in out["units"]:
            if not isinstance(u, dict):
                units.append(u)
                continue
            u = dict(u)
            u["title"] = sanitise_display_text(str(u.get("title", "")))
            u["topics"] = [sanitise_display_text(str(t)) for t in (u.get("topics") or [])]
            u["outcomes"] = [sanitise_display_text(str(o)) for o in (u.get("outcomes") or [])]
            units.append(u)
        out["units"] = units
    if isinstance(out.get("questions"), list):
        qs = []
        for q in out["questions"]:
            if not isinstance(q, dict):
                qs.append(q)
                continue
            q = dict(q)
            q["question"] = sanitise_display_text(str(q.get("question", "")))
            if q.get("topic"):
                q["topic"] = sanitise_display_text(str(q["topic"]))
            qs.append(q)
        out["questions"] = qs
    if isinstance(out.get("mappings"), list):
        ms = []
        for m in out["mappings"]:
            if not isinstance(m, dict):
                ms.append(m)
                continue
            m = dict(m)
            m["question"] = sanitise_display_text(str(m.get("question", "")))
            ms.append(m)
        out["mappings"] = ms
    if isinstance(out.get("frequency"), list):
        fs = []
        for f in out["frequency"]:
            if not isinstance(f, dict):
                fs.append(f)
                continue
            f = dict(f)
            f["unit_title"] = sanitise_display_text(str(f.get("unit_title", "")))
            fs.append(f)
        out["frequency"] = fs
    if out.get("title"):
        out["title"] = sanitise_display_text(str(out["title"]))
    if out.get("syllabus_title"):
        out["syllabus_title"] = sanitise_display_text(str(out["syllabus_title"]))
    return out


@router.post("/syllabus")
async def generate_syllabus(
    request: SyllabusRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Extract syllabus units + topics (grounded in project documents)."""
    try:
        project = await require_project_access(request.project_id, current_user, db)

        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_syllabus(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research,
        )

        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": _sanitised_academic(result.content),
            "cached": result.cached,
            "processing_time": result.processing_time,
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )


@router.post("/pyq")
async def generate_pyq(
    request: PYQRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Extract previous-year questions with year/marks (grounded)."""
    try:
        project = await require_project_access(request.project_id, current_user, db)

        await _select_request_llm(request, current_user, db)
        result = await transformation_engine.generate_pyq(
            project_id=request.project_id,
            db=db,
            query=request.query,
            enable_research=request.enable_research,
        )

        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": _sanitised_academic(result.content),
            "cached": result.cached,
            "processing_time": result.processing_time,
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )


@router.post("/mapping")
async def generate_mapping(
    request: MappingRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Map PYQs to syllabus units (deterministic token-overlap).

    Uses the project's latest cached syllabus/PYQ unless inline
    syllabus/questions are supplied. Below-threshold questions are
    reported unmapped, never force-assigned.
    """
    try:
        project = await require_project_access(request.project_id, current_user, db)

        try:
            await _select_request_llm(request, current_user, db)
            result = await transformation_engine.generate_mapping(
                project_id=request.project_id,
                db=db,
                syllabus=request.syllabus,
                questions=request.questions,
                min_score=request.min_score,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": _sanitised_academic(result.content),
            "cached": result.cached,
            "processing_time": result.processing_time,
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )


@router.post("/analysis")
async def generate_analysis(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Frequency + marks alignment over mapped PYQs (deterministic)."""
    try:
        project = await require_project_access(request.project_id, current_user, db)

        try:
            await _select_request_llm(request, current_user, db)
            result = await transformation_engine.generate_analysis(
                project_id=request.project_id,
                db=db,
                syllabus=request.syllabus,
                questions=request.questions,
                mappings=request.mappings,
                min_score=request.min_score,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )

        return {
            "project_id": result.project_id,
            "mode": result.mode.value,
            "content": _sanitised_academic(result.content),
            "cached": result.cached,
            "processing_time": result.processing_time,
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )


# ==================== HEALTH CHECK ====================

@router.get("/health")
async def health_check():
    """Health check for transformation engine"""
    return {
        "status": "healthy",
        "service": "transformation",
        "modes": [m.value for m in GenerationMode]
    }


# ==================== READS (T107) ====================
# Real rows scoped by require_project_access. Generation itself is Phase 3;
# until content exists these return an explicit empty "not_generated" state -
# never fabricated content, never a 500.

def _empty_state(kind: str) -> dict:
    return {"state": "not_generated", "message": f"No {kind} generated yet for this project."}


@router.get("/flashcards/{project_id}")
async def read_flashcards(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    decks = await repositories.get_flashcard_decks_by_project(db, project.id)
    if not decks:
        return {"projectId": str(project.id), **_empty_state("flashcards"), "decks": [], "total": 0}
    out = []
    for deck in decks:
        cards = getattr(deck, "flashcards", []) or []
        out.append(
            {
                "id": str(deck.id),
                "projectId": str(project.id),
                "name": deck.name,
                "cards": [
                    {"id": str(c.id), "front": c.front, "back": c.back,
                     "difficulty": c.difficulty.value if hasattr(c.difficulty, "value") else c.difficulty}
                    for c in cards
                ],
                "cardCount": deck.card_count,
            }
        )
    first = out[0]
    total_cards = sum(len(d["cards"]) for d in out)
    if total_cards == 0:
        # A deck row with zero cards is not "ready": say so, so the UI
        # offers a retry instead of rendering an empty study surface.
        return {
            "projectId": str(project.id),
            "state": "empty",
            "message": "This deck has no cards yet. Generate flashcards to fill it.",
            **first,
            "decks": out,
            "total": len(out),
        }
    return {"projectId": str(project.id), "state": "ready", **first, "decks": out, "total": len(out)}


def _note_summary(note, project_id) -> dict:
    """One notes list row: id, title, created, mode, coverage, pages.

    The stored payload is the hierarchical notes artifact, so its config
    carries the user's mode/detail and its coverage report carries the topic
    coverage. Keys are emitted in BOTH camelCase and snake_case so existing
    camel consumers and snake consumers stay compatible.
    """
    content = getattr(note, "content", None)
    content = content if isinstance(content, dict) else {}
    config = content.get("config") if isinstance(content.get("config"), dict) else {}
    coverage = content.get("coverage") if isinstance(content.get("coverage"), dict) else {}
    observability = (
        content.get("observability")
        if isinstance(content.get("observability"), dict)
        else {}
    )
    created = getattr(note, "created_at", None)
    created_iso = created.isoformat() if created is not None else None
    pages = content.get("pdf_pages", content.get("pages"))
    version = content.get("pdf_version", content.get("version"))
    artifact_id = content.get("artifact_id")
    mode = config.get("mode") or observability.get("mode")
    detail = config.get("detail") or observability.get("detail")
    return {
        "id": str(note.id),
        "projectId": str(project_id),
        "project_id": str(project_id),
        "title": getattr(note, "title", None),
        "content": getattr(note, "content", None),
        "createdAt": created_iso,
        "created_at": created_iso,
        "mode": mode,
        "detail": detail,
        "version": version,
        "pdfVersion": version,
        "pages": pages,
        "coverage": coverage,
        "topics": observability.get("topic_count") or config.get("topic_count"),
        "sections": observability.get("section_count") or config.get("section_count"),
        "artifactId": artifact_id,
        "artifact_id": artifact_id,
        "status": content.get("artifact_status") or "ready",
        "model": content.get("model"),
        # The stored storage key is deliberately NOT projected: the viewer
        # streams the bytes through the ownership-scoped endpoint instead.
    }


@router.get("/notes/{project_id}")
async def read_notes(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Every notes artifact for this project, newest first.

    Not just the latest: each row carries id, title, created, mode, coverage
    and pages so the UI can LIST the notes and open or delete a specific one.
    """
    notes = await repositories.get_notes_by_project(db, project.id)
    if not notes:
        return {"projectId": str(project.id), **_empty_state("notes"), "notes": [], "total": 0}
    return {
        "projectId": str(project.id),
        "state": "ready",
        "notes": [_note_summary(n, project.id) for n in notes],
        "total": len(notes),
    }


@router.delete("/notes/{project_id}/{note_id}")
async def delete_note(
    project_id: uuid.UUID,
    note_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Delete one note of this project, with its stored PDF.

    ``require_project_access`` proves the caller owns the PROJECT, and the
    note must belong to that same project, so another user's note id is a
    404 here (never a cross-project delete). Deleting the compatibility row
    also removes the notes-PDF version it points at and its stored bytes, so
    the two read models cannot disagree about what exists.
    """
    note = await repositories.get_note_by_id(db, note_id)
    if note is None or str(getattr(note, "project_id", "")) != str(project.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note not found for this project",
        )
    content = getattr(note, "content", None)
    content = content if isinstance(content, dict) else {}
    artifact_id = content.get("artifact_id")
    result = {"deleted": True, "noteId": str(note_id), "projectId": str(project.id),
              "artifactId": artifact_id, "bytes_deleted": False, "artifact_deleted": False}
    if artifact_id:
        try:
            from services.academic import artifacts as academic_artifacts

            artifact_result = await academic_artifacts.delete_notes_pdf_artifact(
                db, project.id, str(artifact_id)
            )
            result["artifact_deleted"] = bool(artifact_result.get("deleted"))
            result["bytes_deleted"] = bool(artifact_result.get("bytes_deleted"))
            result["currentVersion"] = artifact_result.get("current_version")
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            # Never leave a half-deleted note: the row delete below still runs
            # and the artifact is reported as retained, not silently dropped.
            result["artifact_error"] = str(exc)
    await repositories.delete_note(db, note_id)
    await db.commit()
    return result


@router.get("/quiz/{project_id}")
async def read_quiz(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    quizzes = await repositories.get_quizzes_by_project(db, project.id)
    if not quizzes:
        return {"projectId": str(project.id), **_empty_state("quizzes"), "quizzes": [], "total": 0}
    return {
        "projectId": str(project.id),
        "state": "ready",
        "quizzes": [
            {"id": str(q.id), "projectId": str(project.id), "title": q.title,
             "questions": q.questions or [],
             "createdAt": q.created_at.isoformat() if q.created_at else None}
            for q in quizzes
        ],
        "total": len(quizzes),
    }


@router.get("/summary/{project_id}")
async def read_summary(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select

    from database.models.generated_content import GeneratedContentCache, OutputType

    rows = (
        await db.execute(
            select(GeneratedContentCache)
            .where(
                GeneratedContentCache.project_id == project.id,
                GeneratedContentCache.output_type == OutputType("summary"),
            )
            .order_by(GeneratedContentCache.created_at.desc())
        )
    ).scalars().all()
    if not rows:
        return {"projectId": str(project.id), **_empty_state("summaries"), "summaries": [], "total": 0}
    return {
        "projectId": str(project.id),
        "state": "ready",
        "summaries": [
            {"id": str(r.id), "content": r.output_content,
             "createdAt": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ],
        "total": len(rows),
    }


async def _read_cached_academic(
    db: AsyncSession,
    project: Project,
    output_type: str,
    collection_key: str,
) -> dict:
    from sqlalchemy import select

    from database.models.generated_content import GeneratedContentCache, OutputType

    rows = (
        await db.execute(
            select(GeneratedContentCache)
            .where(
                GeneratedContentCache.project_id == project.id,
                GeneratedContentCache.output_type == OutputType(output_type),
            )
            .order_by(GeneratedContentCache.created_at.desc())
        )
    ).scalars().all()
    if not rows:
        return {"projectId": str(project.id),
                **_empty_state(collection_key),
                collection_key: [], "total": 0}
    return {
        "projectId": str(project.id),
        "state": "ready",
        collection_key: [
            {"id": str(r.id), "content": _sanitised_academic(r.output_content or {}),
             "createdAt": r.created_at.isoformat() if r.created_at else None}
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/syllabus/{project_id}")
async def read_syllabus(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return await _read_cached_academic(db, project, "syllabus", "syllabi")


@router.get("/pyq/{project_id}")
async def read_pyq(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return await _read_cached_academic(db, project, "pyq", "pyq_sets")


@router.get("/mapping/{project_id}")
async def read_mapping(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return await _read_cached_academic(db, project, "mapping", "mappings")


@router.get("/analysis/{project_id}")
async def read_analysis(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return await _read_cached_academic(db, project, "analysis", "analyses")


__all__ = ["router"]