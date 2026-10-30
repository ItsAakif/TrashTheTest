"""Puter heavy-document E2E (optional path, default OFF). Offline.

Covers the complete default-OFF heavy flow with a mocked Puter provider:
heavy detection (~100 pages, configurable) -> authorised endpoint honoring
X-Puter-Token per call -> Puter vision -> NORMAL pipeline reuse
(extract -> chunk -> embed -> pgvector) with provenance preserved
(method "puter-ocr", citations valid after resolve), failure explicit
degraded never silent, no creds stored, Puter NOT required for normal flow.

Live Puter + 300-page document = NOT RUN (no keys here).
"""

import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

from database.models.job import JobStatus

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# API-layer fakes (mirror tests/test_frontend_website.py)
# --------------------------------------------------------------------------

def _install_api(monkeypatch, dummy_db, user, doc=None, project=None):
    from database import repositories
    from database.engine import get_db
    from main import app
    from services.auth.deps import get_current_user

    async def fake_get_db():
        yield dummy_db

    async def fake_user():
        return user

    async def fake_doc(db, did):
        return doc

    async def fake_project(db, pid):
        return project

    monkeypatch.setattr(repositories, "get_document_by_id", fake_doc)
    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    return app


def _doc_ns(doc_id, project, file_path="heavy.png", file_type="png"):
    return SimpleNamespace(
        id=doc_id, project_id=project.id, file_name="heavy.png",
        file_path=file_path, file_type=SimpleNamespace(value=file_type),
        processing_status=SimpleNamespace(value="pending"),
        page_count=150, indexing_state=None, extraction_status=None,
        project=SimpleNamespace(user_id=project.user_id))


async def test_e2e_heavy_check_threshold_gate(app_client, dummy_db, user_a, monkeypatch):
    """Heavy detection: >= ~100 pages heavy, below is not; threshold echoed."""
    from main import app

    _install_api(monkeypatch, dummy_db, user_a)
    try:
        heavy = await app_client.get("/api/ocr/puter/heavy-check",
                                     params={"page_count": 150})
        assert heavy.status_code == 200, heavy.text
        body = heavy.json()
        assert body["heavy"] is True
        assert body["pageCount"] == 150
        assert body["threshold"] == 100  # configurable default, not a hard invariant
        assert body["warning"] is not None
        light = await app_client.get("/api/ocr/puter/heavy-check",
                                     params={"page_count": 10})
        assert light.status_code == 200, light.text
        assert light.json()["heavy"] is False
        assert light.json()["warning"] is None
    finally:
        app.dependency_overrides.clear()


async def test_e2e_authorised_endpoint_requires_token(app_client, dummy_db, user_a, monkeypatch):
    """Authorised endpoint honors X-Puter-Token per call: missing -> 401."""
    from main import app
    from services.strategies.puter import PuterOCRProvider

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=user_a.id)
    doc = _doc_ns(doc_id, project)
    _install_api(monkeypatch, dummy_db, user_a, doc=doc, project=project)

    async def fake_health(self):
        return {"available": True, "provider": "puter", "method": "puter-ocr"}

    monkeypatch.setattr(PuterOCRProvider, "health", fake_health)
    try:
        resp = await app_client.post(f"/api/ocr/puter/document/{doc_id}")
        assert resp.status_code == 401
        assert "token" in resp.text.lower()
    finally:
        app.dependency_overrides.clear()


async def test_e2e_authorised_endpoint_default_off(app_client, dummy_db, user_a, monkeypatch):
    """Puter NOT required: default OFF -> explicit 503 even with a token."""
    from main import app

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=user_a.id)
    doc = _doc_ns(doc_id, project)
    _install_api(monkeypatch, dummy_db, user_a, doc=doc, project=project)
    try:
        resp = await app_client.post(
            f"/api/ocr/puter/document/{doc_id}",
            headers={"X-Puter-Token": "user-tok-123"})
        assert resp.status_code == 503
        assert "disabled" in resp.text.lower()
    finally:
        app.dependency_overrides.clear()


async def test_e2e_authorised_endpoint_foreign_doc_404(app_client, dummy_db, user_a,
                                                      user_b, monkeypatch):
    """Another user's document -> 404 (ownership holds on the heavy path)."""
    from main import app
    from services.strategies.puter import PuterOCRProvider

    doc_id = uuid.uuid4()
    project = SimpleNamespace(id=uuid.uuid4(), user_id=user_b.id)
    doc = _doc_ns(doc_id, project)
    _install_api(monkeypatch, dummy_db, user_a, doc=doc, project=project)

    async def fake_health(self):
        return {"available": True, "provider": "puter", "method": "puter-ocr"}

    monkeypatch.setattr(PuterOCRProvider, "health", fake_health)
    try:
        resp = await app_client.post(
            f"/api/ocr/puter/document/{doc_id}",
            headers={"X-Puter-Token": "user-tok-123"})
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# Pipeline-layer fakes (mirror tests/test_pipeline.py)
# --------------------------------------------------------------------------

class _FakeDB:
    def __init__(self, job=None, doc=None):
        self.job = job
        self.doc = doc
        self.pages = []
        self.chunks = []

    async def get(self, model, job_id):
        if self.job is not None and str(self.job.id) == str(job_id):
            return self.job
        return None

    async def commit(self):
        return None

    async def rollback(self):
        return None

    async def flush(self):
        return None


def _job_ns(doc_id, project_id):
    now = datetime.utcnow()
    return SimpleNamespace(
        id=uuid.uuid4(), job_type=SimpleNamespace(value="document_processing"),
        status=JobStatus.PROCESSING, progress=0, stage=None, stage_message=None,
        attempts=0, max_attempts=3,
        input_data={"document_id": str(doc_id), "project_id": str(project_id),
                    "file_path": "heavy.png", "file_type": "png"},
        output_data={}, error_message=None, document_id=doc_id,
        project_id=project_id, locked_by="w", locked_at=now, created_at=now,
        updated_at=now, started_at=now, completed_at=None)


def _install_pipeline(monkeypatch, db, local_provider=None):
    from database import repositories
    from services.ingestion import storage as storage_mod
    from services.jobs import queue as q
    from pathlib import Path as _P

    stages = []

    async def get_doc(d, i):
        return db.doc

    async def update_status(d, i, status, extracted_text=None, error=None):
        db.doc.processing_status = SimpleNamespace(value=status.value)
        return db.doc

    async def get_pages(d, i):
        return list(db.pages)

    async def breakdown(d, i):
        out = {}
        for p in db.pages:
            out[p.method] = out.get(p.method, 0) + 1
        return out

    async def chunks_by_doc(d, i):
        return list(db.chunks)

    async def upsert_page(d, document_id, page_number, text="", method="native",
                          confidence=0.0, parser_version=None, image_stored=False,
                          bbox_summary=None, failure_reason=None):
        page = SimpleNamespace(document_id=document_id, page_number=page_number,
                               text=text, method=method, confidence=confidence,
                               failure_reason=failure_reason)
        db.pages = [p for p in db.pages if p.page_number != page_number] + [page]
        return page

    async def report_progress(d, job_id, progress, stage=None, stage_message=None):
        stages.append(stage)
        db.job.progress = progress
        db.job.stage = stage
        db.job.stage_message = stage_message
        out = dict(db.job.output_data or {})
        if stage:
            out["_stage"] = stage
        if stage_message:
            out["_stage_message"] = stage_message
        db.job.output_data = out
        return db.job

    async def complete(d, job_id, output_data=None):
        stages.append("complete")
        db.job.status = JobStatus.COMPLETED
        db.job.progress = 100
        db.job.output_data = output_data
        return db.job

    async def fail(d, job_id, error, retry=True):
        db.job.attempts += 1
        if retry and db.job.attempts < db.job.max_attempts:
            db.job.status = JobStatus.PENDING
        else:
            db.job.status = JobStatus.FAILED
            db.job.error_message = error
        return db.job

    async def store_chunks(self, d, document_id, project_id, chunks):
        for c in chunks:
            db.chunks.append(c)

    monkeypatch.setattr(repositories, "get_document_by_id", get_doc)
    monkeypatch.setattr(repositories, "update_document_status", update_status)
    monkeypatch.setattr(repositories, "get_pages", get_pages)
    monkeypatch.setattr(repositories, "page_method_breakdown", breakdown)
    monkeypatch.setattr(repositories, "get_chunks_by_document", chunks_by_doc)
    monkeypatch.setattr(repositories, "upsert_page", upsert_page)
    monkeypatch.setattr(q, "report_progress", report_progress)
    monkeypatch.setattr(q, "complete", complete)
    monkeypatch.setattr(q, "fail", fail)
    monkeypatch.setattr(storage_mod.storage_manager, "get_absolute_path",
                        lambda p: _P(p), raising=False)
    import services.chunking.service as chunk_svc

    monkeypatch.setattr(chunk_svc.ChunkingService, "_store_chunks", store_chunks)
    import services.embedding_service.service as emb_svc

    monkeypatch.setattr(emb_svc.LocalEmbeddingService, "__init__", lambda self, *a, **k: None)

    async def fake_generate(self, document_id, project_id, db, job_id=None, **kw):
        return SimpleNamespace(embedded_chunks=len(db.chunks))

    monkeypatch.setattr(emb_svc.LocalEmbeddingService, "generate_embeddings", fake_generate)
    if local_provider is not None:
        import factory as factory_mod

        monkeypatch.setattr(factory_mod, "get_strategy_factory",
                            lambda: SimpleNamespace(get_ocr=lambda: local_provider))
    return stages


def _local_provider(method="ppocr-test", text=None, raises=None):
    from services.strategies.base import ExtractionResult, OCRResult

    async def _extract(file_path, pages=None):
        if raises is not None:
            raise raises
        body = text or ("local fallback transcription " * 30)
        return ExtractionResult(
            pages=[OCRResult(text=body, confidence=0.5, page_number=1,
                             method=method)],
            total_pages=1, method=method, processing_time=0.01)

    async def _page(file_path, page_number):
        result = await _extract(file_path)
        return result.pages[0]

    return SimpleNamespace(extract_text=_extract, extract_page=_page)


async def test_e2e_puter_success_reuses_normal_pipeline(monkeypatch, tmp_path, user_a):
    """Mocked Puter -> extract->chunk->embed reuse; provenance + citations valid."""
    from services.ocr.puter_pipeline import run_puter_heavy_pipeline
    from services.strategies.puter import PuterConfig, PuterOCRProvider

    img = tmp_path / "heavy.png"
    img.write_bytes(b"png-bytes")
    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=user_a.id)
    doc = _doc_ns(doc_id, project, file_path=str(img))
    job = _job_ns(doc_id, project_id)
    db = _FakeDB(job=job, doc=doc)
    stages = _install_pipeline(monkeypatch, db)

    provider = PuterOCRProvider(PuterConfig(enabled=True))

    async def _fake_call(image_path, token, timeout):
        assert token == "tok-abc"  # explicit per-call auth reaches the provider
        return "heavy transcription " * 30, 0.91

    monkeypatch.setattr(provider, "_call_puter", _fake_call)
    summary = await run_puter_heavy_pipeline(db, doc, job, user_token="tok-abc",
                                             provider=provider)

    # Normal pipeline reused: real extract + real chunker + staged completion.
    assert summary["stage"] == "complete"
    assert summary["puter_pages"] >= 1
    assert summary["page_methods"].get("puter-ocr", 0) >= 1
    assert summary["chunks"] >= 1 and len(db.chunks) >= 1
    assert "puter-offered" in stages and "puter-processing" in stages
    assert db.job.status == JobStatus.COMPLETED
    assert db.job.output_data["_stage"] == "complete"
    # Provenance preserved into chunks; citations resolve valid after the run.
    from services.retrieval.citations import build_citation_from_chunk

    first = db.chunks[0]
    meta = first.metadata.model_dump() if hasattr(first.metadata, "model_dump") else dict(first.metadata or {})
    assert meta.get("method") == "puter-ocr"
    # Stored chunks gain ids at persist time; resolve the persisted shape.
    stored = SimpleNamespace(id=uuid.uuid4(), document_id=doc_id,
                             content=first.content, chunk_metadata=meta,
                             page_start=meta.get("page_start"),
                             page_end=meta.get("page_end"),
                             section_path=meta.get("section_path") or [],
                             heading=meta.get("heading"))
    citation = build_citation_from_chunk(stored, doc)
    assert citation.valid is True
    assert (citation.method or "") == "puter-ocr"
    # No creds stored: the token lives in the call only.
    blob = str(db.job.output_data) + str(db.job.input_data) + str(vars(provider))
    assert "tok-abc" not in blob
    assert getattr(provider, "_token", None) is None


async def test_e2e_puter_failure_falls_back_to_local(monkeypatch, tmp_path, user_a):
    """Puter failure -> explicit local fallback per page, never silent."""
    from services.ocr.puter_pipeline import run_puter_heavy_pipeline
    from services.strategies.puter import PuterConfig, PuterOCRProvider

    img = tmp_path / "heavy.png"
    img.write_bytes(b"png-bytes")
    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=user_a.id)
    doc = _doc_ns(doc_id, project, file_path=str(img))
    job = _job_ns(doc_id, project_id)
    db = _FakeDB(job=job, doc=doc)
    _install_pipeline(monkeypatch, db, local_provider=_local_provider())

    provider = PuterOCRProvider(PuterConfig(enabled=True))

    async def _boom(image_path, token, timeout):
        raise RuntimeError("puter transport down")

    monkeypatch.setattr(provider, "_call_puter", _boom)
    summary = await run_puter_heavy_pipeline(db, doc, job, user_token="tok-abc",
                                             provider=provider)
    assert summary["stage"] == "complete"
    assert summary["fallback_pages"] >= 1
    assert summary["page_methods"].get("ppocr-test", 0) >= 1
    assert all((p.text or "").strip() for p in db.pages)  # never silent empty text
    assert db.job.status == JobStatus.COMPLETED


async def test_e2e_puter_total_failure_is_explicit(monkeypatch, tmp_path, user_a):
    """Both Puter + local down -> explicit failure with reason, never silent."""
    from services.ocr.puter_pipeline import run_puter_heavy_pipeline
    from services.strategies.puter import PuterConfig, PuterOCRProvider

    img = tmp_path / "heavy.png"
    img.write_bytes(b"png-bytes")
    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=user_a.id)
    doc = _doc_ns(doc_id, project, file_path=str(img))
    job = _job_ns(doc_id, project_id)
    db = _FakeDB(job=job, doc=doc)
    _install_pipeline(monkeypatch, db,
                      local_provider=_local_provider(raises=RuntimeError("local engine down")))

    provider = PuterOCRProvider(PuterConfig(enabled=True))

    async def _boom(image_path, token, timeout):
        raise RuntimeError("puter transport down")

    monkeypatch.setattr(provider, "_call_puter", _boom)
    with pytest.raises(RuntimeError, match="explicitly"):
        await run_puter_heavy_pipeline(db, doc, job, user_token="tok-abc",
                                       provider=provider)
    assert db.job.status == JobStatus.FAILED
    assert "degraded explicitly" in (db.job.error_message or "")
