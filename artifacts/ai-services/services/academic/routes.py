"""STEP 10 read paths: study-notes PDF metadata + bytes, insights tables/cards.

Additive routes under /api/academic (existing generation/syllabus/PYQ
routes are untouched). Every route is ownership-scoped via
require_project_access (another user's project is 404, never content).
PDF bytes stream from the storage backend with the same inline
Content-Disposition pattern as GET /api/documents/{id}/file; a missing
storage key yields a typed 404 (artifact_unavailable), never a 500.

Generation itself stays offline in this slice (structured notes ->
renderer -> stored PDF): empty projects get an honest not_generated
state naming the generation call, never fabricated metadata.
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import require_project_access
from database.engine import get_db
from database.models.project import Project
from database.models.user import User
from database.models.job import JobType
from services.academic import artifacts as academic_artifacts
from services.academic import orchestrator as academic_orchestrator
from services.auth.deps import get_current_user
from services.academic.types import NotesGenerationConfig, recommend_notes_config
from services.jobs import queue as job_queue
from services.storage import get_default_backend

router = APIRouter(prefix="/api/academic", tags=["academic"])
generation_router = APIRouter(prefix="/api/generation", tags=["academic-generation"])
knowledge_router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/notes/{project_id}")
async def read_study_notes(
    project_id: uuid.UUID,
    version: Optional[int] = Query(default=None, ge=1),
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Study-notes PDF metadata, or an honest not_generated state.

    Every stored version is returned in ``artifacts`` (id, title, created,
    mode, coverage, pages, version, status) so one read powers the notes
    LIST and the header. ``?version=N`` pins one version, which is how the
    viewer opens the note the user clicked instead of only the newest one.
    """
    state = await academic_artifacts.get_notes_pdf_status(
        db, project.id, version=version
    )
    state.setdefault("artifacts", [])
    state.setdefault("total", len(state["artifacts"]))
    if state.get("state") != "ready":
        state["projectId"] = str(project.id)
    return state


@router.get("/notes/{project_id}/versions")
async def read_study_notes_versions(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """All study-notes PDF versions for this project (list read model)."""
    return await academic_artifacts.list_notes_pdf_versions(db, project.id)


@router.delete("/notes/{project_id}/artifacts/{artifact_id}")
async def delete_study_notes_artifact(
    project_id: uuid.UUID,
    artifact_id: str,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Delete one study-notes version (record + stored bytes).

    Ownership is enforced by ``require_project_access`` on the project, and
    the scan is limited to this project's document roots, so another user's
    artifact id is a 404 here. The caller owns the commit.
    """
    result = await academic_artifacts.delete_notes_pdf_artifact(
        db, project.id, artifact_id
    )
    if not result.get("deleted"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study notes artifact not found for this project",
        )
    await db.commit()
    return _sanitize(result)


@router.get("/notes/{project_id}/pdf")
async def download_study_notes_pdf(
    project_id: uuid.UUID,
    version: Optional[int] = Query(default=None, ge=1),
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Stream the stored study-notes PDF inline for the EXISTING viewer.

    The frontend DocumentViewer fetches these bytes (JWT, never in a URL)
    and renders them with the same react-pdf surface as source documents.
    ``?version=N`` streams that stored version; without it the latest READY
    version is streamed (unchanged behaviour).
    """
    state = await academic_artifacts.get_notes_pdf_status(
        db, project.id, version=version
    )
    if state.get("state") != "ready":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study notes PDF unavailable for this project",
        )
    roots = await academic_artifacts._document_roots(db, project.id)
    key: str | None = None
    wanted = state.get("artifact_id")
    for root in roots:
        versions = await academic_artifacts.knowledge_repo.list_generated_artifacts(
            db, academic_artifacts._node_id(root), academic_artifacts.NOTES_PDF_TYPE
        )
        for artifact in versions or []:
            if str(artifact.get("id")) == str(wanted):
                key = str(artifact.get("path") or "")
                break
        if key:
            break
    backend = get_default_backend()
    if not key or not await backend.exists(key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Study notes PDF unavailable for this project",
        )
    title = str(state.get("title") or "study-notes")
    safe_name = "".join(
        c for c in title if c.isalnum() or c in ("-", "_", ".", " ")
    ).strip() or "study-notes"
    return StreamingResponse(
        backend.open_stream(key),
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{safe_name}.pdf"'},
    )


@router.get("/insights/{project_id}")
async def read_insights(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Backend-first insights payload (summary cards + topic table), or not_generated."""
    state = await academic_artifacts.get_insights_status(db, project.id)
    if state.get("state") != "ready":
        state["projectId"] = str(project.id)
    return state


# ---------------------------------------------------------------------------
# Production generation routes
# ---------------------------------------------------------------------------


class _AcademicRequest(BaseModel):
    """Snake+camel compatible base for academic generation requests."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    project_id: str = Field(alias="projectId", min_length=1)
    document_ids: List[str] = Field(default_factory=list, alias="documentIds")
    force: bool = False


class SpineBuildRequest(_AcademicRequest):
    background: bool = False
    project_title: Optional[str] = Field(default=None, alias="projectTitle")


class HierarchicalNotesRequest(_AcademicRequest):
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_live: bool = Field(default=True, alias="useLive")
    use_retrieval: bool = Field(default=True, alias="useRetrieval")
    project_title: Optional[str] = Field(default=None, alias="projectTitle")
    config: Optional[NotesGenerationConfig] = Field(
        default=None,
        validation_alias=AliasChoices("config", "notes_config", "notesConfig"),
    )


class TypedFlashcardsRequest(_AcademicRequest):
    max_cards: Optional[int] = Field(default=None, ge=1, le=1000, alias="maxCards")


class MCQMatrixRequest(_AcademicRequest):
    total: Optional[int] = Field(default=None, ge=1, le=1000, alias="totalQuestions")


class MetricsGenerateRequest(_AcademicRequest):
    current_year: Optional[int] = Field(default=None, ge=1950, le=2100, alias="currentYear")
    half_life: float = Field(default=5.0, gt=0.0, le=100.0, alias="halfLife")
    include_narrative: bool = Field(default=False, alias="includeNarrative")


class PdfBuildRequest(_AcademicRequest):
    pass


def _uuid(value: Any) -> Optional[uuid.UUID]:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


async def _owned_project(request: _AcademicRequest, user: User, db: AsyncSession) -> Project:
    # Resolve identity through the existing dependency; never accept a
    # client-supplied user_id in an academic generation body.
    return await require_project_access(request.project_id, user, db)


async def _project_notes_recommendation(
    db: AsyncSession,
    project: Project,
    document_ids: Optional[List[str]] = None,
) -> NotesGenerationConfig:
    """Build an advisory recommendation from real project sources."""

    topic_keys: set[str] = set()
    source_types: set[str] = set()
    try:
        _, structures = await academic_orchestrator.build_source_structures(
            db, project.id, document_ids=document_ids or None,
        )
        for structure in structures:
            source_types.add(str(getattr(structure, "source_type", "study_material")))
            for unit in getattr(structure, "units", []) or []:
                for topic in getattr(unit, "topics", []) or []:
                    key = str(getattr(topic, "title", "")).strip().lower()
                    if key:
                        topic_keys.add(key)
    except Exception:
        # Advisory metadata must never make the primary Generate action
        # unavailable when source parsing is temporarily incomplete.
        try:
            from database import repositories

            documents = await repositories.get_documents_by_project(db, project.id)
            source_types = {
                str(getattr(getattr(doc, "source_type", None), "value", getattr(doc, "source_type", "study_material")))
                for doc in (documents or [])
            }
        except Exception:
            pass
    return recommend_notes_config(
        topic_count=len(topic_keys),
        has_question_papers=bool(source_types & {"question_paper", "question_bank"}),
        has_syllabus="syllabus" in source_types,
    )


async def _safe_provider_selection(
    db: AsyncSession, user: User, project: Project,
) -> Dict[str, Any]:
    """Resolve BYOK for this boundary, persisting only a safe reference."""

    token = None
    try:
        from services.provider_credentials.override import (
            current_request_llm,
            reset_request_llm,
            resolve_for_request,
        )

        # Use the same request-scoped resolver as transformation/study.  The
        # decrypted selection exists only in this ContextVar/request; the
        # returned metadata below deliberately drops ``api_key``.
        token = await resolve_for_request(db, user, project.id)
        selection = current_request_llm() or {}
    except Exception:
        return {}
    finally:
        if token is not None:
            try:
                from services.provider_credentials.override import reset_request_llm

                reset_request_llm(token)
            except Exception:
                pass
    if not isinstance(selection, dict):
        return {}
    return {
        key: selection.get(key)
        for key in ("provider", "model", "source", "using_user_key", "key_hint")
        if selection.get(key) is not None
    }


def _sanitize(value: Any) -> Any:
    """Sanitize display strings at the response boundary, storage stays raw."""
    from services.transformation.engine import sanitise_display_text

    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return sanitise_display_text(value)
    return value


def _job_response(job: Any, project_id: Any, artifact_id: Optional[str] = None) -> Dict[str, Any]:
    job_id = str(getattr(job, "id", "") or "")
    return {
        "project_id": str(project_id),
        "job_id": job_id,
        "id": job_id,
        "state": "queued",
        "status": "pending",
        "artifact_id": artifact_id,
        "poll_url": f"/api/jobs/{job_id}",
    }


@generation_router.post("/notes-hierarchical", status_code=status.HTTP_202_ACCEPTED, include_in_schema=False)
@generation_router.post("/notes/hierarchical", status_code=status.HTTP_202_ACCEPTED)
async def start_hierarchical_notes(
    request: HierarchicalNotesRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue the resumable extraction -> notes -> PDF pipeline."""
    project = await _owned_project(request, current_user, db)
    user_config = request.config or NotesGenerationConfig()
    # Honest guard: a project with no processed chunks has nothing to
    # generate from.  The recommendation path loads the source chunks first,
    # so its typed NoExtractableTextError IS the chunk-count-0 signal.
    # Refuse BEFORE any job is queued (no LLM call, no job created); main.py
    # maps the typed error to the 422 not-generated shape already used for
    # empty academic source scopes.
    try:
        recommended = await _project_notes_recommendation(
            db, project, document_ids=list(request.document_ids),
        )
    except academic_orchestrator.NoExtractableTextError:
        raise academic_orchestrator.NoExtractableTextError(
            "Upload study material before generating notes."
        ) from None
    # Attach advisory metadata only.  The user's mode/detail/web choices are
    # never replaced by the recommendation.
    config = user_config.model_copy(update={
        "recommendation": dict(recommended.recommendation),
    })
    provider_selection = await _safe_provider_selection(db, current_user, project)
    data = {
        "project_id": str(project.id),
        "user_id": str(current_user.id),
        "document_ids": list(request.document_ids),
        "threshold": request.threshold,
        "use_live": request.use_live,
        "use_retrieval": request.use_retrieval,
        "project_title": request.project_title or project.name,
        "force": request.force,
        "config": config.persisted_dict(),
        # Safe selection metadata only; decrypted keys stay in request memory.
        "provider_selection": provider_selection,
    }
    job = await job_queue.enqueue(
        db,
        JobType.NOTES_HIERARCHICAL,
        data,
        project_id=project.id,
        max_attempts=1,
        idempotency_key=f"academic-notes:{project.id}",
        # Academic jobs yield to uploads (worker fairness rule keeps at most
        # one of these running while document_processing work is pending).
        priority=job_queue.PRIORITY_ACADEMIC,
    )
    await db.commit()
    return _job_response(job, project.id)


@router.get("/notes/recommendation/{project_id}")
async def read_notes_recommendation(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Return advisory notes settings derived from current project sources."""

    recommendation = await _project_notes_recommendation(db, project)
    return {
        "projectId": str(project.id),
        "state": "ready",
        "recommendation": recommendation.recommendation,
        "config": {
            "mode": recommendation.mode.value,
            "detail": recommendation.detail.value,
            "exam_focus": recommendation.exam_focus,
            "memory_aids": recommendation.memory_aids,
            "web": recommendation.web,
        },
    }


@generation_router.get("/notes/hierarchical/{project_id}")
async def read_hierarchical_notes(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Read the latest hierarchical notes job state honestly."""
    from services.jobs.service import job_manager

    jobs = await job_manager.get_jobs_for_project(db, project.id, limit=100)
    academic_jobs = [
        job for job in jobs
        if str(getattr(getattr(job, "job_type", None), "value", getattr(job, "job_type", "")))
        in {"notes_hierarchical", "academic_generation"}
    ]
    if not academic_jobs:
        state = academic_artifacts.not_generated_state(
            "hierarchical-notes", "POST /api/generation/notes/hierarchical"
        )
        state["projectId"] = str(project.id)
        return state
    job = academic_jobs[0]
    output = dict(getattr(job, "output_data", None) or {})
    state = output.get("academic") if isinstance(output.get("academic"), dict) else {}
    job_status = getattr(getattr(job, "status", None), "value", getattr(job, "status", ""))
    if job_status == "completed":
        read_state = "ready"
    elif job_status in {"failed", "cancelled"}:
        read_state = job_status
    else:
        read_state = "processing"
    result = {
        "projectId": str(project.id),
        "job_id": str(job.id),
        "state": read_state,
        "status": job_status,
        "stage": getattr(job, "stage", None) or output.get("_stage"),
        "progress": int(getattr(job, "progress", 0) or 0),
        "lifecycle": list(output.get("_lifecycle") or state.get("lifecycle") or []),
        "artifact_id": output.get("artifact_id") or state.get("pdf_artifact_id"),
        # Partial-success signal: the notes can be ready while some source
        # topics were honest gaps.  Surface the run-level truth at the top
        # level so the UI does not have to dig into the nested job output.
        "gaps": list(state.get("gaps") or []),
        "failed_topic_count": int(state.get("failed_topic_count") or 0),
        "provider_slow": bool(state.get("provider_slow") or False),
        "output": _sanitize(output),
    }
    if read_state != "ready":
        result["generation_call"] = "POST /api/generation/notes/hierarchical"
    if read_state == "failed":
        result["error"] = getattr(job, "error_message", None)
    return result


@generation_router.post("/knowledge-spine", include_in_schema=False)
@generation_router.post("/spine")
@knowledge_router.post("/model")
async def generate_spine(
    request: SpineBuildRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Build the canonical spine synchronously or queue it explicitly."""
    project = await _owned_project(request, current_user, db)
    if request.background:
        job = await job_queue.enqueue(
            db,
            JobType.KNOWLEDGE_MODEL,
            {
                "project_id": str(project.id),
                "document_ids": list(request.document_ids),
                "project_title": request.project_title or project.name,
                "force": request.force,
            },
            project_id=project.id,
            max_attempts=1,
            idempotency_key=f"knowledge-model:{project.id}",
            priority=job_queue.PRIORITY_ACADEMIC,
        )
        await db.commit()
        return _job_response(job, project.id)
    result = await academic_orchestrator.build_project_spine(
        db,
        project.id,
        document_ids=request.document_ids,
        project_title=request.project_title or project.name,
        force=request.force,
    )
    await db.commit()
    return {
        "project_id": str(project.id),
        "projectId": str(project.id),
        "state": "ready",
        **_sanitize(result.as_dict()),
    }


@knowledge_router.get("/model/{project_id}")
async def read_model(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return _sanitize(await academic_orchestrator.read_spine(db, project.id))


@knowledge_router.get("/model/{project_id}/node/{node_id}")
async def read_model_node(
    project_id: uuid.UUID,
    node_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    from database import knowledge_repo

    node = await knowledge_repo.get_knowledge_node(db, node_id)
    if node is None or str(getattr(node, "project_id", "")) != str(project.id):
        raise HTTPException(status_code=404, detail="Knowledge node not found")
    nodes = await knowledge_repo.list_knowledge_nodes(db, project_id=project.id)
    by_id = {str(getattr(item, "id", "")): item for item in nodes or []}
    ancestors: List[Dict[str, Any]] = []
    parent_id = getattr(node, "parent_id", None)
    while parent_id and str(parent_id) in by_id:
        parent = by_id[str(parent_id)]
        ancestors.insert(0, {"id": str(parent.id), "type": parent.type, "title": parent.title})
        parent_id = getattr(parent, "parent_id", None)
    children = [
        {"id": str(item.id), "type": item.type, "title": item.title,
         "order_index": item.order_index, "chunks": list(item.chunks or [])}
        for item in nodes or []
        if getattr(item, "parent_id", None) is not None
        and str(getattr(item, "parent_id", "")) == str(node_id)
    ]
    return _sanitize({
        "project_id": str(project.id),
        "node": {
            "id": str(node.id), "type": node.type, "title": node.title,
            "parent_id": str(node.parent_id) if node.parent_id else None,
            "order_index": node.order_index, "chunks": list(node.chunks or []),
            "aliases": list(node.aliases or []), "confidence": node.confidence,
            "metadata": dict(node.node_metadata or {}),
        },
        "ancestors": ancestors,
        "children": children,
    })


@generation_router.post("/flashcards-typed", include_in_schema=False)
@generation_router.post("/flashcards/typed")
@router.post("/flashcards/typed")
async def generate_typed_flashcards_route(
    request: TypedFlashcardsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(request, current_user, db)
    result = await academic_orchestrator.generate_typed_flashcards(
        db,
        project.id,
        document_ids=request.document_ids,
        max_cards=request.max_cards,
        force=request.force,
    )
    await db.commit()
    return _sanitize(result)


@generation_router.get("/flashcards/typed/{project_id}")
@router.get("/flashcards/typed/{project_id}")
async def read_typed_flashcards_route(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return _sanitize(await academic_orchestrator.read_typed_artifact(
        db,
        project.id,
        "flashcards_artifact",
        "typed-flashcards",
        "POST /api/generation/flashcards/typed",
    ))


@generation_router.post("/mcq-matrix", include_in_schema=False)
@generation_router.post("/mcq/matrix")
@router.post("/mcq/matrix")
async def generate_mcq_matrix_route(
    request: MCQMatrixRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(request, current_user, db)
    result = await academic_orchestrator.generate_mcq_matrix(
        db,
        project.id,
        document_ids=request.document_ids,
        total=request.total,
        force=request.force,
    )
    await db.commit()
    return _sanitize(result)


@generation_router.get("/mcq/matrix/{project_id}")
@router.get("/mcq/matrix/{project_id}")
async def read_mcq_matrix_route(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return _sanitize(await academic_orchestrator.read_typed_artifact(
        db,
        project.id,
        "mcq_artifact",
        "mcq-matrix",
        "POST /api/generation/mcq/matrix",
    ))


@generation_router.post("/metrics")
@generation_router.post("/metrics-generate")
@router.post("/metrics")
async def generate_metrics_route(
    request: MetricsGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(request, current_user, db)
    result = await academic_orchestrator.generate_metrics_report(
        db,
        project.id,
        document_ids=request.document_ids,
        current_year=request.current_year,
        half_life=request.half_life,
        include_narrative=request.include_narrative,
        force=request.force,
    )
    await db.commit()
    return _sanitize(result)


@generation_router.get("/metrics/{project_id}")
@router.get("/metrics/{project_id}")
async def read_metrics_route(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    return _sanitize(await academic_orchestrator.read_typed_artifact(
        db,
        project.id,
        "metrics_artifact",
        "metrics",
        "POST /api/generation/metrics",
    ))


@generation_router.post("/notes/pdf-build", include_in_schema=False)
@generation_router.post("/notes/pdf")
async def build_notes_pdf_route(
    request: PdfBuildRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(request, current_user, db)
    result = await academic_orchestrator.build_pdf_for_project(
        db,
        project.id,
        document_ids=request.document_ids,
        force=request.force,
    )
    await db.commit()
    return _sanitize(result)


@router.post("/notes/{project_id}/pdf")
async def build_academic_notes_pdf_short_route(
    project_id: uuid.UUID,
    request: Optional[PdfBuildRequest] = None,
    project: Project = Depends(require_project_access),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if request is not None and str(request.project_id) != str(project.id):
        raise HTTPException(status_code=400, detail="project_id does not match path")
    result = await academic_orchestrator.build_pdf_for_project(
        db,
        project.id,
        document_ids=request.document_ids if request is not None else [],
        force=request.force if request is not None else False,
    )
    await db.commit()
    return _sanitize(result)


@router.post("/notes/{project_id}/pdf/build")
async def build_academic_notes_pdf_route(
    project_id: uuid.UUID,
    request: PdfBuildRequest,
    project: Project = Depends(require_project_access),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if str(request.project_id) != str(project.id):
        raise HTTPException(status_code=400, detail="project_id does not match path")
    result = await academic_orchestrator.build_pdf_for_project(
        db,
        project.id,
        document_ids=request.document_ids,
        force=request.force,
    )
    await db.commit()
    return _sanitize(result)


__all__ = [
    "router",
    "generation_router",
    "knowledge_router",
]
