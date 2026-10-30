"""STEP 9 orchestration: structured notes -> PDF -> versioned artifact.

build_notes_pdf_artifact() is the single pipeline entry point: it takes
the ordered synthesis output (NotesArtifact + spine), renders the
university notes PDF (STEP 8), fingerprints sources/model/config/coverage,
and assigns the next version (short-circuiting on a fingerprint match).
Storage persistence (bytes -> backend, key on the artifact `path`) and the
node_metadata write stay caller-side so the pure build is unit-testable
with no DB and no storage.

get_notes_pdf_status() / get_insights_status() aggregate the per-document
artifact stores into project-level read models for the /api/academic
routes: latest READY version wins, old versions stay addressable, and an
empty project gets an honest not_generated state naming the generation
call. Helpers flush; the caller owns the commit (worker/route commits).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from database import knowledge_repo
from services.academic.notes_pdf import (
    NotesPdfReport,
    notes_pdf_storage_hint,
    render_notes_pdf,
)
from services.academic.types import artifact_fingerprint, not_generated_state

NOTES_PDF_TYPE = knowledge_repo.NOTES_PDF_ARTIFACT_TYPE
INSIGHTS_TYPE = knowledge_repo.INSIGHTS_ARTIFACT_TYPE

logger = logging.getLogger(__name__)

# Honest empty reads name the production call that produces the content.
NOTES_GENERATION_CALL = "POST /api/generation/notes/hierarchical"
INSIGHTS_GENERATION_CALL = "POST /api/generation/metrics"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coverage_summary(coverage: Any) -> Dict[str, Any]:
    if coverage is None:
        return {}
    if isinstance(coverage, dict):
        return {
            "topic_coverage": coverage.get("topic_coverage"),
            "concept_coverage": coverage.get("concept_coverage"),
            "passes": coverage.get("passes"),
            "threshold": coverage.get("threshold"),
            "uncovered": list(coverage.get("uncovered") or []),
        }
    try:
        payload = coverage.model_dump()
    except Exception:
        return {}
    return {
        "topic_coverage": payload.get("topic_coverage"),
        "concept_coverage": payload.get("concept_coverage"),
        "passes": payload.get("passes"),
        "threshold": payload.get("threshold"),
        "uncovered": list(payload.get("uncovered") or []),
    }


def _source_ids(structure: Any, extra: Optional[List[str]] = None) -> List[str]:
    ids: List[str] = []
    if structure is not None:
        doc_id = structure.get("doc_id") if isinstance(structure, dict) else getattr(structure, "doc_id", "")
        if doc_id:
            ids.append(str(doc_id))
        else:
            title = structure.get("title") if isinstance(structure, dict) else getattr(structure, "title", "")
            if title:
                ids.append(str(title))
    for item in extra or []:
        if item and item not in ids:
            ids.append(str(item))
    return ids


def build_notes_pdf_artifact(
    notes_artifact: Any,
    structure: Any = None,
    model: str = "academic-notes",
    config: Optional[Dict[str, Any]] = None,
    existing_versions: Optional[List[Dict[str, Any]]] = None,
    created_at: Optional[str] = None,
    topic_extras: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
    chunk_pages: Optional[Dict[str, int]] = None,
    chunk_texts: Optional[Dict[str, str]] = None,
    allowed_chunk_ids: Optional[set[str]] = None,
) -> Tuple[Dict[str, Any], NotesPdfReport, bool]:
    """Build (or short-circuit) one notes-PDF artifact version.

    Returns (artifact_dict, pdf_report, reused): reused=True when the
    fingerprint matches an existing READY version (no new version, no PDF
    render, and no bytes to store; a small report envelope is returned for
    shape parity). The artifact dict carries a
    `fingerprint` alongside the GeneratedArtifact fields so the next
    rebuild can short-circuit.
    """
    from services.academic.types import GeneratedArtifact

    config = dict(config or {})
    coverage = _coverage_summary(
        notes_artifact.get("coverage") if isinstance(notes_artifact, dict)
        else getattr(notes_artifact, "coverage", None)
    )
    sources = _source_ids(structure, config.get("source_ids"))
    fingerprint = artifact_fingerprint(sources, model, config, coverage)
    version, match = knowledge_repo.assign_artifact_version(existing_versions or [], fingerprint)
    # Fingerprint short-circuit is deliberately BEFORE rendering.  An
    # unchanged rebuild must not call ReportLab or produce bytes that will
    # never be stored.  Reconstruct the small report envelope from metadata
    # so callers keep the same return shape without a second render.
    if match is not None:
        old = dict(match)
        old_report = NotesPdfReport(
            pdf_bytes=b"",
            page_count=int(old.get("pages", 0) or 0),
            toc_entries=list(old.get("toc_entries") or []),
            bookmark_count=len(old.get("toc_entries") or []),
            sha256=str(old.get("sha256") or old.get("hash") or ""),
            topic_count=int((old.get("config") or {}).get("topic_count", 0) or 0),
            section_count=int((old.get("config") or {}).get("section_count", 0) or 0),
        )
        return old, old_report, True

    stamp = created_at or _utcnow_iso()

    title = (
        notes_artifact.get("title") if isinstance(notes_artifact, dict)
        else getattr(notes_artifact, "title", "Study Notes")
    ) or "Study Notes"
    project_id = str(config.get("project_id") or "")
    meta = {
        "title": str(title),
        "version": version,
        "model": model,
        "created_at": stamp,
        "coverage": coverage,
        "chunk_pages": dict(chunk_pages or {}),
        "source_count": len(sources) or "course documents",
    }
    report = render_notes_pdf(
        notes_artifact,
        structure,
        meta,
        topic_extras,
        chunk_texts=chunk_texts,
        allowed_chunk_ids=allowed_chunk_ids,
    )

    artifact = GeneratedArtifact(
        id=str(uuid.uuid4()),
        project_id=project_id,
        sources=sources,
        type=NOTES_PDF_TYPE,
        version=version,
        status="ready",
        path=notes_pdf_storage_hint(project_id or "project", version),
        pages=report.page_count,
        hash=report.sha256,
        model=model,
        model_version=(config.get("model_version") if config.get("model_version") is not None else None),
        payload=(notes_artifact.model_dump(mode="json")
                 if hasattr(notes_artifact, "model_dump") else dict(notes_artifact or {})),
        provenance={
            "stage": "pdf",
            "source_fingerprint": fingerprint,
            "renderer": "reportlab",
        },
        config={**config, "topic_count": report.topic_count,
                "section_count": report.section_count},
        coverage=coverage,
        created_at=stamp,
        updated_at=stamp,
    )
    record = artifact.model_dump(by_alias=True, mode="json")
    record["artifact_type"] = NOTES_PDF_TYPE
    record["title"] = str(title)
    record["fingerprint"] = fingerprint
    record["toc_entries"] = list(report.toc_entries)
    return record, report, False


async def _document_roots(db: Any, project_id: Any) -> List[Any]:
    nodes = await knowledge_repo.list_knowledge_nodes(db, project_id=project_id)
    roots = [n for n in (nodes or []) if getattr(n, "parent_id", None) is None]
    roots.sort(key=lambda n: str(getattr(n, "id", "")))
    return roots


def _node_id(node: Any) -> Any:
    return getattr(node, "id", None)


def _node_title(node: Any) -> str:
    return str(getattr(node, "title", "") or "")


async def _collect_notes_pdf_versions(
    db: Any, project_id: Any
) -> List[Tuple[Dict[str, Any], str, Any]]:
    """Every stored notes-PDF version for the project: (artifact, doc, node).

    Ordered newest version first, then newest created_at, so the list the
    user sees is stable and the head of it is the current READY version.
    """
    rows: List[Tuple[Dict[str, Any], str, Any]] = []
    for root in await _document_roots(db, project_id):
        versions = await knowledge_repo.list_generated_artifacts(
            db, _node_id(root), NOTES_PDF_TYPE
        )
        for artifact in versions or []:
            rows.append((dict(artifact), _node_title(root), _node_id(root)))
    # Stable two-pass sort: newest version first, newest created_at first.
    rows.sort(key=lambda item: str(item[0].get("created_at") or ""), reverse=True)
    rows.sort(key=lambda item: int(item[0].get("version", 0) or 0), reverse=True)
    return rows


def _notes_artifact_summary(
    artifact: Dict[str, Any], document: str = ""
) -> Dict[str, Any]:
    """List row for one notes-PDF version (never internal JSON)."""
    config = dict(artifact.get("config") or {})
    notes_config = dict(config.get("notes_config") or {})
    return {
        "id": artifact.get("id"),
        "artifact_id": artifact.get("id"),
        "artifactId": artifact.get("id"),
        "artifact_type": NOTES_PDF_TYPE,
        "title": artifact.get("title") or document or "Study Notes",
        "version": artifact.get("version"),
        "status": artifact.get("status"),
        "pages": artifact.get("pages"),
        "topics": config.get("topic_count"),
        "sections": config.get("section_count"),
        "mode": notes_config.get("mode"),
        "detail": notes_config.get("detail"),
        "coverage": dict(artifact.get("coverage") or {}),
        "model": artifact.get("model"),
        "sha256": artifact.get("sha256") or artifact.get("hash"),
        "created_at": artifact.get("created_at"),
        "document": document,
    }


async def list_notes_pdf_versions(db: Any, project_id: Any) -> Dict[str, Any]:
    """All notes-PDF versions for a project (read model for the list UI)."""
    rows = await _collect_notes_pdf_versions(db, project_id)
    ready = [row for row in rows if row[0].get("status") == "ready"]
    if not rows:
        state = not_generated_state("study-notes-pdf", NOTES_GENERATION_CALL)
        state["projectId"] = str(project_id)
        state["artifacts"] = []
        state["total"] = 0
        return state
    return {
        "projectId": str(project_id),
        "state": "ready",
        "kind": "study-notes-pdf-list",
        "artifacts": [_notes_artifact_summary(a, doc) for a, doc, _ in rows],
        "total": len(rows),
        "ready_total": len(ready),
    }


async def get_notes_pdf_status(
    db: Any, project_id: Any, version: Optional[int] = None
) -> Dict[str, Any]:
    """Project-level study-notes PDF state for GET /api/academic/notes.

    ``version`` pins one stored version (so the viewer can open the note the
    user clicked, not only the newest); without it the latest READY version
    wins, as before. The response also carries the full ``artifacts`` list so
    one read powers both the list and the header.
    """
    pid = str(project_id)
    rows = await _collect_notes_pdf_versions(db, project_id)
    selected: Optional[Dict[str, Any]] = None
    best_doc: Optional[str] = None
    if version is not None:
        for artifact, doc, _ in rows:
            try:
                if int(artifact.get("version", 0) or 0) == int(version):
                    selected, best_doc = artifact, doc
                    break
            except (TypeError, ValueError):
                continue
    else:
        ready = [row for row in rows if row[0].get("status") == "ready"]
        pool = ready or rows
        if pool:
            selected, best_doc = pool[0][0], pool[0][1]
    state = {
        "projectId": pid,
        "artifacts": [_notes_artifact_summary(a, doc) for a, doc, _ in rows],
        "total": len(rows),
    }
    if selected is None or (version is None and selected.get("status") != "ready"):
        empty = not_generated_state("study-notes-pdf", NOTES_GENERATION_CALL)
        empty["projectId"] = pid
        empty["artifacts"] = state["artifacts"]
        empty["total"] = state["total"]
        return empty
    coverage = dict(selected.get("coverage") or {})
    config = dict(selected.get("config") or {})
    notes_config = dict(config.get("notes_config") or {})
    return {
        **state,
        "state": "ready",
        "kind": "study-notes-pdf",
        "title": selected.get("title") or best_doc or "Study Notes",
        "artifact_id": selected.get("id"),
        "artifactId": selected.get("id"),
        "version": selected.get("version"),
        "status": selected.get("status"),
        "model_version": selected.get("model_version") or config.get("model_version"),
        "topics": config.get("topic_count"),
        "sections": config.get("section_count"),
        "pages": selected.get("pages"),
        "mode": notes_config.get("mode"),
        "detail": notes_config.get("detail"),
        "coverage": coverage,
        "model": selected.get("model"),
        "sha256": selected.get("sha256") or selected.get("hash"),
        "created_at": selected.get("created_at"),
        "document": best_doc,
    }


async def delete_notes_pdf_artifact(
    db: Any, project_id: Any, artifact_id: str
) -> Dict[str, Any]:
    """Delete one notes-PDF version (record + stored bytes) for a project.

    Scans this project's document roots only, so an id from another project
    is never touched. Returns an honest report: what was removed, whether the
    stored bytes existed, and which version is now current.
    """
    wanted = str(artifact_id or "")
    removed: Optional[Dict[str, Any]] = None
    for root in await _document_roots(db, project_id):
        versions = await knowledge_repo.list_generated_artifacts(
            db, _node_id(root), NOTES_PDF_TYPE
        )
        if not any(str(v.get("id")) == wanted for v in versions or []):
            continue
        removed = await knowledge_repo.delete_generated_artifact_version(
            db, _node_id(root), NOTES_PDF_TYPE, artifact_id=wanted
        )
        break
    if removed is None:
        return {"deleted": False, "artifact_id": wanted, "bytes_deleted": False}
    bytes_deleted = False
    key = str(removed.get("path") or "")
    if key:
        try:
            from services.storage import get_default_backend

            backend = get_default_backend()
            if await backend.exists(key):
                bytes_deleted = bool(await backend.delete(key))
        except Exception as exc:
            logger.warning("notes artifact bytes not deleted for %s: %s", key, exc)
    current = await get_notes_pdf_status(db, project_id)
    return {
        "deleted": True,
        "artifact_id": wanted,
        "version": removed.get("version"),
        "path": key,
        "bytes_deleted": bytes_deleted,
        "current_version": current.get("version"),
        "total": current.get("total"),
    }


async def get_insights_status(db: Any, project_id: Any) -> Dict[str, Any]:
    """Project-level insights state for GET /api/academic/insights.

    Serves the backend-first tables/cards from the stored metrics
    artifact (Step 7 metrics_artifact payload: tables + cards), or an
    honest not_generated state. No numbers are recomputed here.
    """
    from database.knowledge_repo import METRICS_ARTIFACT_KEY

    pid = str(project_id)
    best: Optional[Dict[str, Any]] = None
    best_stamp = ""
    for root in await _document_roots(db, project_id):
        artifact = await knowledge_repo.get_academic_artifact(
            db, _node_id(root), METRICS_ARTIFACT_KEY)
        if artifact is None:
            continue
        stamp = str(
            artifact.get("created_at")
            or getattr(root, "created_at", "")
            or ""
        )
        if best is None or stamp >= best_stamp:
            best = artifact
            best_stamp = stamp
    if best is None:
        state = not_generated_state("insights", INSIGHTS_GENERATION_CALL)
        state["projectId"] = pid
        return state
    return {
        "projectId": pid,
        "state": "ready",
        "kind": "insights",
        "cards": list(best.get("cards") or []),
        "table": list(best.get("tables") or []),
        "never_asked": list((best.get("metrics") or {}).get("never_asked") or []),
        "missing": list((best.get("metrics") or {}).get("missing") or []),
        "totals": dict((best.get("metrics") or {}).get("totals") or {}),
        "narrative": best.get("narrative"),
        "narrative_status": best.get("narrative_status", "not_requested"),
        "provenance_note": best.get("provenance_note", ""),
    }


__all__ = [
    "NOTES_PDF_TYPE",
    "INSIGHTS_TYPE",
    "NOTES_GENERATION_CALL",
    "INSIGHTS_GENERATION_CALL",
    "build_notes_pdf_artifact",
    "get_notes_pdf_status",
    "list_notes_pdf_versions",
    "delete_notes_pdf_artifact",
    "get_insights_status",
]
