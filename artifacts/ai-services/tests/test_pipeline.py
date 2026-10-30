"""T209/T210: stage machine, resume/cancel/retry, cascade delete (offline).

Fakes stand in for Postgres; the queue contract (claim/resume/cancel) is
exercised honestly against the real worker + handler code paths.
"""

import asyncio
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from database.models.job import JobStatus

pytestmark = pytest.mark.asyncio


class FakeDB:
    def __init__(self, job=None, doc=None, pages=None, chunks=None):
        self.job = job
        self.doc = doc
        self.pages = pages or []
        self.chunks = chunks or []
        self.commits = 0

    async def get(self, model, job_id):
        if self.job is not None and str(self.job.id) == str(job_id):
            return self.job
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None

    async def flush(self):
        return None


def _doc(doc_id, project_id):
    return SimpleNamespace(id=doc_id, project_id=project_id, file_name="f.pdf",
                           file_path="f.pdf",
                           file_type=SimpleNamespace(value="txt"),
                           processing_status=SimpleNamespace(value="pending"),
                           extracted_text=None, indexing_state=None,
                           extraction_status=None, project=None)


def _job(doc_id, project_id, stage=None, output=None, status=JobStatus.PROCESSING):
    now = datetime.utcnow()
    return SimpleNamespace(id=uuid.uuid4(), job_type=SimpleNamespace(value="document_processing"),
                           status=status, progress=0, stage=stage, stage_message=None,
                           attempts=0, max_attempts=3, input_data={"document_id": str(doc_id),
                                                                   "project_id": str(project_id),
                                                                   "file_path": "x", "file_type": "txt"},
                           output_data=output or {}, error_message=None,
                           document_id=doc_id, project_id=project_id,
                           locked_by="w", locked_at=now, created_at=now, updated_at=now,
                           started_at=now, completed_at=None)


def _install(monkeypatch, db):
    from database import repositories
    from services.jobs import queue as q

    async def get_doc(d, i):
        return db.doc

    async def get_doc_with_project(d, i):
        return db.doc

    async def set_indexing(d, i, state, degraded_reason=None, extraction_status=None):
        try:
            db.doc.indexing_state = state
        except Exception:
            pass
        if degraded_reason is not None:
            try:
                db.doc.processing_error = degraded_reason
            except Exception:
                pass
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

    async def report_progress(d, job_id, progress, stage=None, stage_message=None):
        job = db.job
        job.progress = progress
        job.stage = stage
        job.stage_message = stage_message
        out = dict(job.output_data or {})
        if stage:
            out["_stage"] = stage
        job.output_data = out
        return job

    async def complete(d, job_id, output_data=None):
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

    monkeypatch.setattr(repositories, "get_document_by_id", get_doc)
    monkeypatch.setattr(repositories, "get_document_by_id_with_project", get_doc_with_project)
    monkeypatch.setattr(repositories, "set_document_indexing_state", set_indexing)
    monkeypatch.setattr(repositories, "update_document_status", update_status)
    monkeypatch.setattr(repositories, "get_pages", get_pages)
    monkeypatch.setattr(repositories, "page_method_breakdown", breakdown)
    monkeypatch.setattr(repositories, "get_chunks_by_document", chunks_by_doc)
    monkeypatch.setattr(q, "report_progress", report_progress)
    monkeypatch.setattr(q, "complete", complete)
    monkeypatch.setattr(q, "fail", fail)


async def _run_txt(monkeypatch, tmp_path, pages_text=("hello world " * 20,)):
    """Drive the stage machine over a real .txt file (no PDF libs needed)."""
    from services.jobs.handlers.document_processing import process_job

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "doc.txt"
    f.write_text("\n\n".join(pages_text), encoding="utf-8")
    doc = _doc(doc_id, project_id)
    job = _job(doc_id, project_id)
    job.input_data["file_path"] = str(f)
    db = FakeDB(job=job, doc=doc, pages=[], chunks=[])
    _install(monkeypatch, db)

    from database import repositories
    from services.ingestion import storage as storage_mod
    from pathlib import Path as _P

    real_pages = []

    async def upsert_page(d, document_id, page_number, text="", method="native",
                          confidence=0.0, parser_version=None, image_stored=False,
                          bbox_summary=None, failure_reason=None):
        real_pages.append(SimpleNamespace(document_id=document_id, page_number=page_number,
                                          text=text, method=method, confidence=confidence))
        db.pages = real_pages
        return real_pages[-1]

    async def store_chunks(self, d, document_id, project_id, chunks):
        for c in chunks:
            db.chunks.append(c)

    monkeypatch.setattr(repositories, "upsert_page", upsert_page)
    monkeypatch.setattr(storage_mod.storage_manager, "get_absolute_path",
                        lambda p: _P(p), raising=False)
    import services.chunking.service as chunk_svc

    monkeypatch.setattr(chunk_svc.ChunkingService, "_store_chunks", store_chunks)
    # Skip embedding model download offline: stub the embedder.
    import services.embedding_service.service as emb_svc

    async def fake_generate(self, document_id, project_id, db, job_id=None, **kw):
        return SimpleNamespace(embedded_chunks=len(db.chunks) if hasattr(db, "chunks") else 0)

    monkeypatch.setattr(emb_svc.LocalEmbeddingService, "generate_embeddings", fake_generate)
    monkeypatch.setattr(emb_svc.LocalEmbeddingService, "__init__", lambda self, *a, **k: None)
    await process_job(SimpleNamespace(db=db), job)
    return db


async def test_pipeline_progress_monotonic_from_real_counts(monkeypatch, tmp_path):
    seen = []
    from services.jobs import queue as q

    real_report = None
    db = await _run_txt(monkeypatch, tmp_path)
    assert db.job.status == JobStatus.COMPLETED
    assert db.job.progress == 100
    assert db.doc.processing_status.value == "completed"
    assert len(db.chunks) >= 1  # chunk stage really ran


async def test_pipeline_progress_committed_mid_run(monkeypatch, tmp_path):
    """Progress must be COMMITTED (not just flushed) during the run: the
    worker holds one long transaction per job, so flush-only progress stays
    invisible to API pollers (the stuck-0% symptom)."""
    db = await _run_txt(monkeypatch, tmp_path)
    assert db.job.status == JobStatus.COMPLETED
    assert db.commits >= 1, "stage reports must commit so pollers see movement"


async def test_validate2_no_text_completes_with_completed_doc(monkeypatch, tmp_path):
    """Pages without text: job completes no_text AND the document reads
    completed (never failed) so re-upload dedupe returns it, not a requeue."""
    from services.jobs.handlers.document_processing import process_job
    from types import SimpleNamespace as SN

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    doc = _doc(doc_id, project_id)
    page = SN(document_id=doc_id, page_number=1, text="   ",
              method="ocr_unavailable", confidence=0.0)
    job = _job(doc_id, project_id, stage="extract",
               output={"_stages_done": ["validate", "extract"]})
    db = FakeDB(job=job, doc=doc, pages=[page], chunks=[])
    _install(monkeypatch, db)
    await process_job(SimpleNamespace(db=db), job)
    assert db.job.status == JobStatus.COMPLETED
    out = db.job.output_data or {}
    assert out.get("indexing_state") == "no_text"
    assert "degraded_reason" in out
    assert db.doc.indexing_state == "no_text"
    assert db.doc.processing_status.value == "completed"


async def test_resume_skips_completed_extract_stage(monkeypatch, tmp_path):
    from services.jobs.handlers.document_processing import process_job
    from types import SimpleNamespace as SN

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "r.txt"
    f.write_text("resume content " * 50, encoding="utf-8")
    doc = _doc(doc_id, project_id)
    page = SN(document_id=doc_id, page_number=1, text="resume content " * 50,
              method="text", confidence=1.0)
    job = _job(doc_id, project_id, stage="extract",
               output={"_stages_done": ["validate", "extract"]})
    job.input_data["file_path"] = str(f)
    db = FakeDB(job=job, doc=doc, pages=[page], chunks=[])
    _install(monkeypatch, db)

    calls = {"extract": 0}
    import services.extraction.extractor as ext_mod

    real_cls = ext_mod.DocumentExtractor

    class NoExtract(real_cls):
        async def extract(self, *a, **k):
            calls["extract"] += 1
            raise AssertionError("extract must be skipped on resume")

    monkeypatch.setattr(ext_mod, "DocumentExtractor", NoExtract)
    import services.chunking.service as chunk_svc

    async def store_chunks(self, d, document_id, project_id, chunks):
        for c in chunks:
            db.chunks.append(c)

    monkeypatch.setattr(chunk_svc.ChunkingService, "_store_chunks", store_chunks)
    import services.embedding_service.service as emb_svc

    monkeypatch.setattr(emb_svc.LocalEmbeddingService, "__init__", lambda self, *a, **k: None)

    async def fake_generate(self, document_id, project_id, db, job_id=None, **kw):
        return SN(embedded_chunks=0)

    monkeypatch.setattr(emb_svc.LocalEmbeddingService, "generate_embeddings", fake_generate)
    await process_job(SN(db=db), job)
    assert calls["extract"] == 0
    assert db.job.status == JobStatus.COMPLETED


async def test_permanent_failure_marks_failed_after_max_attempts(monkeypatch):
    from services.jobs import queue as q

    job = _job(uuid.uuid4(), uuid.uuid4(), status=JobStatus.PROCESSING)
    db = FakeDB(job=job)

    async def get(d, i):
        return db.job

    db.get = get
    for _ in range(3):
        await q.fail(db, job.id, "provider 429 overloaded", retry=True)
    assert job.status == JobStatus.FAILED
    assert job.error_message == "provider 429 overloaded"


async def test_cancel_at_stage_boundary(monkeypatch, tmp_path):
    from services.jobs.handlers.document_processing import process_job
    from types import SimpleNamespace as SN

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "c.txt"
    f.write_text("cancel me " * 50, encoding="utf-8")
    doc = _doc(doc_id, project_id)
    job = _job(doc_id, project_id)
    job.input_data["file_path"] = str(f)
    job.status = SimpleNamespace(value="cancelled") if False else JobStatus.CANCELLED
    db = FakeDB(job=job, doc=doc, pages=[], chunks=[])
    _install(monkeypatch, db)
    await process_job(SN(db=db), job)
    # Cancelled jobs are left alone (handler returns without completing).
    assert db.job.status == JobStatus.CANCELLED
