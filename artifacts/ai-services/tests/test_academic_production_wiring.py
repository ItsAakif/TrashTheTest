"""Focused production-wiring tests for the academic vertical.

These tests stay offline: no live database, provider, or storage service is
required.  They protect the seams that were missing from the Step 0-12
foundation (mode registration, job stages, fingerprint order, persistence
shape, and route registration).
"""

import uuid
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from services.academic import artifacts as academic_artifacts
from services.academic import orchestrator
from services.academic.extraction import extract_structure
from services.academic.types import ArtifactStatus
from tests.eval.corpus import CORPUS

pytestmark = pytest.mark.asyncio


def _rows(doc_id="doc-biology"):
    return [dict(row) for row in CORPUS if row["doc"] == doc_id]


def _structure():
    return extract_structure("doc-biology", "Photosynthesis primer", _rows())


async def test_knowledge_spine_is_registered_in_all_generation_places():
    from services.transformation.engine import TransformationEngine
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import GenerationMode, KnowledgeSpineOutput

    assert TransformationEngine.MODE_MODELS[GenerationMode.KNOWLEDGE_SPINE] is KnowledgeSpineOutput
    assert TransformationEngine.STRICT_MODELS[GenerationMode.KNOWLEDGE_SPINE] is KnowledgeSpineOutput
    prompt = PromptTemplates.get_prompt(
        GenerationMode.KNOWLEDGE_SPINE, "[source chunk_id=c1] text", "title"
    )
    assert PromptTemplates.INJECTION_GUARD in prompt
    assert PromptTemplates.CITATION_RULES in prompt


async def test_academic_stage_budget_does_not_replace_document_budget():
    from services.jobs.worker import STAGE_PROGRESS_RANGES as document_ranges

    assert set(document_ranges) == {
        "validate", "extract", "structure", "validate2", "chunk", "embed", "complete"
    }
    assert orchestrator.NOTES_HIERARCHICAL_STAGES == (
        "model", "outline", "group", "per_group", "coverage", "repair",
        "synthesis", "validate", "persist", "pdf",
    )
    assert orchestrator.STAGE_PROGRESS_RANGES is not document_ranges
    assert orchestrator.STAGE_PROGRESS_RANGES["per_group"] == (20, 65)


async def test_notes_pdf_storage_write_is_scoped_to_user_and_project(monkeypatch):
    """The local backend rejects bare filenames; scope before writing bytes."""
    project_id = uuid.uuid4()
    user_id = uuid.uuid4()
    saved = {}

    async def _project(_db, requested_id):
        assert requested_id == project_id
        return SimpleNamespace(id=project_id, user_id=user_id)

    async def _existing(_db, _root_id, _artifact_type):
        return []

    class _Backend:
        async def save(self, key_hint, data):
            saved["key"] = key_hint
            saved["data"] = data
            return f"{user_id}/{project_id}/server-generated.pdf"

    async def _save_artifact(*args, **kwargs):
        return None

    record = {"path": "study-notes-project-v1.pdf", "id": "artifact-1"}
    report = SimpleNamespace(pdf_bytes=b"%PDF-test", sha256="a" * 64)
    monkeypatch.setattr(orchestrator.repositories, "get_project_by_id", _project)
    monkeypatch.setattr(
        orchestrator.academic_artifacts.knowledge_repo,
        "list_generated_artifacts",
        _existing,
    )
    monkeypatch.setattr(
        orchestrator.academic_artifacts,
        "build_notes_pdf_artifact",
        lambda *args, **kwargs: (record, report, False),
    )
    monkeypatch.setattr(
        "services.storage.get_default_backend",
        lambda: _Backend(),
    )
    monkeypatch.setattr(
        orchestrator.academic_artifacts.knowledge_repo,
        "save_generated_artifact",
        _save_artifact,
    )

    result = await orchestrator.build_and_store_pdf(
        object(),
        project_id,
        SimpleNamespace(),
        SimpleNamespace(),
        uuid.uuid4(),
        [],
    )
    assert saved["data"] == b"%PDF-test"
    assert saved["key"] == (
        f"{user_id}/{project_id}/study-notes-project-v1.pdf"
    )
    assert result["path"].endswith(f"{user_id}/{project_id}/server-generated.pdf")


async def test_terminal_notes_failure_uses_clean_session():
    job_id = uuid.uuid4()
    job = SimpleNamespace(
        id=job_id,
        status="processing",
        attempts=0,
        max_attempts=1,
        progress=94,
        locked_by="worker-old",
        locked_at=object(),
        error_message=None,
        completed_at=None,
        updated_at=None,
    )

    class _DB:
        def __init__(self, target):
            self.target = target
            self.commits = 0

        async def get(self, _model, requested_id):
            return self.target

        async def flush(self):
            return None

        async def commit(self):
            self.commits += 1

    fresh = _DB(job)

    class _Factory:
        def __call__(self):
            class _CM:
                async def __aenter__(self):
                    return fresh

                async def __aexit__(self, *exc):
                    return False

            return _CM()

    class _Original:
        async def rollback(self):
            return None

    ctx = SimpleNamespace(db=_Original(), session_factory=_Factory())
    await orchestrator._record_hierarchical_failure(ctx, job_id, "pdf failed")

    assert str(job.status) == "JobStatus.FAILED" or job.status == "failed"
    assert job.error_message == "pdf failed"
    assert job.locked_by is None
    assert job.locked_at is None
    assert fresh.commits == 1


async def test_failed_model_header_is_not_reused_as_a_ready_spine():
    async def _models(*args, **kwargs):
        return [SimpleNamespace(
            version=1,
            status="failed",
            source_fingerprint="same",
            id="header-1",
        )]

    with patch.object(orchestrator.knowledge_repo, "list_knowledge_models", _models):
        version, header, fingerprint = await orchestrator._existing_model_version(
            object(), uuid.uuid4()
        )
    assert version == 1
    assert header.id == "header-1"
    assert fingerprint == ""


async def test_academic_job_stage_and_resume_ledger_are_preserved():
    from datetime import datetime, timezone

    from services.jobs.types import JobProgress, JobStatus

    progress = JobProgress(
        job_id=uuid.uuid4(),
        status=JobStatus.PROCESSING,
        progress=42,
        stage="per_group",
        created_at=datetime.now(timezone.utc),
    )
    assert progress.stage == "per_group"

    job = SimpleNamespace(
        stage="coverage",
        output_data={"_stages_done": ["model", "outline", "group"]},
    )
    assert orchestrator._completed_stages(job) == {
        "model", "outline", "group", "per_group"
    }


async def test_build_spine_alias_accepts_legacy_argument_order(monkeypatch):
    calls = {}

    async def _build(db, project_id, **kwargs):
        calls["db"] = db
        calls["project_id"] = project_id
        calls.update(kwargs)
        return "built"

    db = object()
    monkeypatch.setattr(orchestrator, "build_project_spine", _build)
    result = await orchestrator.build_spine(
        "project-1", db, source_document_ids=["doc-1"], force=True
    )
    assert result == "built"
    assert calls["db"] is db
    assert calls["project_id"] == "project-1"
    assert calls["document_ids"] == ["doc-1"]
    assert calls["force"] is True


async def test_merge_is_source_ordered_and_deduplicates_topic_surfaces():
    first = _structure()
    second = extract_structure("doc-biology-copy", "Photosynthesis primer", _rows())
    merged = orchestrator.merge_document_structures([first, second])
    titles = [topic.title for unit in merged.units for topic in unit.topics]
    assert titles == [topic.title for topic in first.units[0].topics]
    assert len(titles) == len(set(titles))
    assert merged.units[0].topics[0].chunks == first.units[0].topics[0].chunks


async def test_fingerprint_short_circuit_precedes_pdf_render(monkeypatch):
    from services.academic.coverage import attach_report, evaluate_coverage
    from services.academic.notes_hier import render_topic_notes, synthesize_notes

    structure = _structure()
    notes = [
        render_topic_notes(unit.title, topic, _rows())
        for unit in structure.units for topic in unit.topics
    ]
    artifact = attach_report(
        synthesize_notes(
            "Notes", structure, notes,
            chunk_texts={row["id"]: row["text"] for row in _rows()},
        ),
        evaluate_coverage(structure, notes),
    )
    first, _, reused = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m", config={"project_id": "p"}
    )
    assert reused is False
    called = {"n": 0}

    def _unexpected_render(*args, **kwargs):
        called["n"] += 1
        raise AssertionError("PDF renderer called for an unchanged fingerprint")

    monkeypatch.setattr(academic_artifacts, "render_notes_pdf", _unexpected_render)
    second, report, reused_again = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m", config={"project_id": "p"},
        existing_versions=[first],
    )
    assert reused_again is True
    assert second["id"] == first["id"]
    assert report.pdf_bytes == b""
    assert called["n"] == 0


async def test_generated_artifact_accepts_superseded_status():
    from services.academic.types import GeneratedArtifact

    row = {
        "id": str(uuid.uuid4()), "project_id": "p", "sources": ["d"],
        "type": "notes_pdf", "version": 1, "status": "superseded",
        "path": "x", "pages": 1, "hash": "a" * 64, "model": "m",
        "created_at": "now", "updated_at": "now",
    }
    assert GeneratedArtifact.model_validate(row).status is ArtifactStatus.SUPERSEDED


async def test_no_extractable_text_is_an_explicit_pipeline_error():
    class _DB:
        pass

    async def _docs(*args, **kwargs):
        return []

    async def _chunks(*args, **kwargs):
        return []

    with patch.object(orchestrator.repositories, "get_documents_by_project", _docs), \
         patch.object(orchestrator.repositories, "get_chunks_by_project", _chunks):
        with pytest.raises(orchestrator.NoExtractableTextError, match="no extractable text"):
            await orchestrator.load_processed_chunks(_DB(), uuid.uuid4())


async def test_direct_spine_empty_scope_returns_typed_4xx(monkeypatch):
    """An empty source scope must not become an opaque 500 response."""
    from database.engine import get_db
    from main import create_app
    from services.academic import routes
    from services.auth.deps import get_current_user

    project_id = uuid.uuid4()
    user = SimpleNamespace(id=uuid.uuid4())
    project = SimpleNamespace(id=project_id, name="Empty project", user_id=user.id)

    class _DB:
        async def commit(self):
            return None

    async def _db():
        return _DB()

    async def _current_user():
        return user

    async def _owned(*args, **kwargs):
        return project

    async def _empty_spine(*args, **kwargs):
        raise orchestrator.NoExtractableTextError(
            "no extractable text: processed chunks are empty for this project"
        )

    app = create_app()
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_current_user] = _current_user
    monkeypatch.setattr(routes, "require_project_access", _owned)
    monkeypatch.setattr(orchestrator, "build_project_spine", _empty_spine)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/generation/spine", json={"projectId": str(project_id)}
            )
        assert response.status_code == 422
        assert response.json() == {
            "success": False,
            "error": "no extractable text: processed chunks are empty for this project",
            "error_code": "NO_EXTRACTABLE_TEXT",
        }
    finally:
        app.dependency_overrides.clear()


async def test_hierarchical_read_reports_failed_job_as_failed():
    from services.academic import routes

    project = SimpleNamespace(id=uuid.uuid4())
    job = SimpleNamespace(
        id=uuid.uuid4(),
        job_type="notes_hierarchical",
        status="failed",
        error_message="no extractable text",
        stage="model",
        progress=10,
        output_data={"academic": {}},
    )

    async def _jobs(*args, **kwargs):
        return [job]

    with patch("services.jobs.service.job_manager.get_jobs_for_project", _jobs):
        result = await routes.read_hierarchical_notes(
            project.id, project=project, db=object()
        )

    assert result["state"] == "failed"
    assert result["status"] == "failed"
    assert result["error"] == "no extractable text"
    assert result["generation_call"] == "POST /api/generation/notes/hierarchical"


async def test_persisted_typed_and_metrics_reads_return_real_shapes():
    old_root = SimpleNamespace(id="old", created_at="2026-09-23T10:00:00")
    new_root = SimpleNamespace(id="new", created_at="2026-09-24T10:00:00")

    async def _nodes(*args, **kwargs):
        return [old_root, new_root]

    async def _artifact(_db, node_id, key):
        if node_id == "old":
            return {
                "state": "ready",
                "cards": [{"card": "old"}],
                "tables": [{"topic": "old"}],
                "metrics": {"never_asked": ["old"], "missing": [], "totals": {}},
            }
        return {
            "state": "ready",
            "artifact_id": "metric-new",
            "cards": [{"card": "new"}],
            "tables": [{"topic": "new"}],
            "metrics": {"never_asked": ["new"], "missing": [], "totals": {}},
        }

    with patch.object(orchestrator.knowledge_repo, "list_knowledge_nodes", _nodes), \
         patch.object(academic_artifacts.knowledge_repo, "get_academic_artifact", _artifact):
        typed = await orchestrator.read_typed_artifact(
            object(), "project-1", "metrics_artifact", "metrics", "POST /metrics"
        )
        insights = await academic_artifacts.get_insights_status(object(), "project-1")

    assert typed["state"] == "ready"
    assert typed["project_id"] == "project-1"
    assert typed["artifact_id"] == "metric-new"
    assert typed["artifact"]["artifact_id"] == "metric-new"
    assert insights["state"] == "ready"
    assert insights["table"] == [{"topic": "new"}]
    assert insights["cards"] == [{"card": "new"}]


async def test_generation_routers_expose_production_paths():
    from services.academic.routes import generation_router, knowledge_router, router

    paths = {route.path for route in generation_router.routes}
    paths.update(route.path for route in knowledge_router.routes)
    paths.update(route.path for route in router.routes)
    for expected in (
        "/api/generation/notes/hierarchical",
        "/api/generation/spine",
        "/api/generation/flashcards/typed",
        "/api/generation/mcq/matrix",
        "/api/generation/metrics",
        "/api/generation/notes/pdf",
        "/api/knowledge/model",
    ):
        assert expected in paths
