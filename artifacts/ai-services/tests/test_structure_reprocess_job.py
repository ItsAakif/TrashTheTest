"""ADR-008 / T009: STRUCTURE_REPROCESS job handler contract.

Everything runs offline against in-memory fakes: no database, no file, no
network.  Proves the three acceptance points from the task contract:

  1. no ``document_id``  -> terminal failure with the exact message,
  2. deleted document    -> terminal failure with the matching message,
  3. valid document      -> chunk -> delete-stale -> store -> stamp order.
"""

import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

from database.models.job import JobStatus, JobType
from services.jobs import queue as job_queue
from services.jobs.handlers import structure_reprocess as srh

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------- fakes ----

class FakeDb:
    """Minimal AsyncSession double: the queue helpers only need get/flush."""

    def __init__(self, job=None):
        self.jobs = {}
        if job is not None:
            self.jobs[job.id] = job
        self.commits = 0
        self.executed = []

    def get_bind(self):
        return SimpleNamespace(url="postgresql+asyncpg://localhost/db")

    async def get(self, model, pk):
        return self.jobs.get(pk)

    async def flush(self):
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None


def _job(**over):
    now = datetime.utcnow()
    base = dict(
        id=uuid.uuid4(),
        job_type=JobType.STRUCTURE_REPROCESS,
        status=JobStatus.PENDING,
        progress=0,
        attempts=0,
        max_attempts=3,
        locked_by=None,
        locked_at=None,
        run_after=None,
        input_data={},
        output_data=None,
        error_message=None,
        started_at=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
        document_id=None,
        project_id=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _FakeChunk:
    def __init__(self, content_hash, method=None, confidence=None):
        self.content_hash = content_hash
        self.content = f"content-{content_hash}"
        self.token_count = 4
        self.metadata = SimpleNamespace(
            structure_method=method,
            structure_confidence=confidence,
        )


class _FakeChunker:
    def __init__(self, calls):
        self.calls = calls

    def chunk_document(self, input_data):
        self.calls.append("chunk_document")
        return SimpleNamespace(
            chunks=[_FakeChunk("h1", method="heading", confidence=0.9)],
            total_chunks=1,
        )


class _FakeChunkingService:
    def __init__(self, calls):
        self.calls = calls
        self.chunker = _FakeChunker(calls)
        self.inputs = []

    async def _store_chunks(self, db, document_id, project_id, chunks):
        self.calls.append("_store_chunks")


def _doc(**over):
    base = dict(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        file_name="lecture.pdf",
        extracted_text="Heading\n\nBody paragraph.",
    )
    base.update(over)
    return SimpleNamespace(**base)


# -------------------------------------------------------------- helpers ----

def _patch_document(monkeypatch, document):
    async def _with_project(db, document_id):
        return document

    async def _plain(db, document_id):
        return document

    monkeypatch.setattr(
        "database.repositories.get_document_by_id_with_project", _with_project)
    monkeypatch.setattr(
        "database.repositories.get_document_by_id", _plain)


def _patch_success_path(monkeypatch, calls):
    async def _get_pages(db, document_id):
        return []

    async def _delete(db, document_id, hashes):
        calls.append("delete_chunks_not_in_hashes")
        return 0

    async def _stamp(db, document_id, *, method, confidence, degraded):
        calls.append("set_document_structure_verdict")

    async def _complete(db, job_id, output_data=None):
        job = await db.get(object, job_id)
        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.output_data = output_data
        job.completed_at = datetime.utcnow()
        return job

    monkeypatch.setattr("database.repositories.get_pages", _get_pages)
    monkeypatch.setattr(
        "database.repositories.delete_chunks_not_in_hashes", _delete)
    monkeypatch.setattr(
        "database.repositories.set_document_structure_verdict", _stamp)
    monkeypatch.setattr(job_queue, "complete", _complete)


# --------------------------------------------------------------- tests -----

async def test_job_type_is_registered_with_the_worker():
    from services.jobs.worker import _HANDLERS

    assert JobType.STRUCTURE_REPROCESS.value == "structure_reprocess"
    assert "structure_reprocess" in _HANDLERS


async def test_missing_document_id_fails_without_retry():
    job = _job(input_data={})
    db = FakeDb(job)

    await srh.process_job(SimpleNamespace(db=db), job)

    assert job.status == JobStatus.FAILED
    assert job.error_message == "STRUCTURE_REPROCESS: no document_id"


async def test_malformed_document_id_fails_without_retry():
    job = _job(input_data={"document_id": "not-a-uuid"})
    db = FakeDb(job)

    await srh.process_job(SimpleNamespace(db=db), job)

    assert job.status == JobStatus.FAILED
    assert "STRUCTURE_REPROCESS: invalid document_id" in job.error_message


async def test_deleted_document_fails_with_matching_message(monkeypatch):
    _patch_document(monkeypatch, None)
    job = _job(input_data={"document_id": str(uuid.uuid4())})
    db = FakeDb(job)

    await srh.process_job(SimpleNamespace(db=db), job)

    assert job.status == JobStatus.FAILED
    assert job.error_message == "STRUCTURE_REPROCESS: document was deleted"


async def test_empty_extracted_text_fails_without_retry(monkeypatch):
    _patch_document(monkeypatch, _doc(extracted_text="   \n  "))
    job = _job(input_data={"document_id": str(uuid.uuid4())})
    db = FakeDb(job)

    await srh.process_job(SimpleNamespace(db=db), job)

    assert job.status == JobStatus.FAILED
    assert "no extracted_text to reprocess" in job.error_message


async def test_valid_document_runs_the_pipeline_in_order(monkeypatch):
    calls = []
    document = _doc()
    _patch_document(monkeypatch, document)
    _patch_success_path(monkeypatch, calls)
    monkeypatch.setattr(srh, "chunking_service", _FakeChunkingService(calls))

    job = _job(input_data={"document_id": str(document.id)})
    db = FakeDb(job)

    await srh.process_job(SimpleNamespace(db=db), job)

    assert calls == [
        "chunk_document",
        "delete_chunks_not_in_hashes",
        "_store_chunks",
        "set_document_structure_verdict",
    ]
    assert job.status == JobStatus.COMPLETED
    assert job.output_data["document_id"] == str(document.id)
    assert job.output_data["chunks_total"] == 1
    assert job.output_data["method"] == "heading"
    assert job.output_data["confidence"] == pytest.approx(0.9)
    assert job.output_data["degraded"] is False
    assert db.commits == 2  # store commit + verdict commit


async def test_missing_chunk_metadata_reports_degraded_defaults(monkeypatch):
    """No structure metadata at all -> conservative page_bounded verdict."""
    calls = []
    document = _doc()
    _patch_document(monkeypatch, document)
    _patch_success_path(monkeypatch, calls)

    class _MetadatalessChunker:
        def chunk_document(self, input_data):
            calls.append("chunk_document")
            return SimpleNamespace(
                chunks=[SimpleNamespace(content_hash="h2", metadata=None)],
                total_chunks=1,
            )

    async def _store(db, document_id, project_id, chunks):
        calls.append("_store_chunks")

    monkeypatch.setattr(
        srh, "chunking_service",
        SimpleNamespace(chunker=_MetadatalessChunker(),
                        _store_chunks=_store))

    job = _job(input_data={"document_id": str(document.id)})
    db = FakeDb(job)

    await srh.process_job(SimpleNamespace(db=db), job)

    assert job.status == JobStatus.COMPLETED
    assert job.output_data["method"] == "page_bounded"
    assert job.output_data["confidence"] == 0.0
    assert job.output_data["degraded"] is True


async def test_verdict_stamp_failure_still_completes_the_job(monkeypatch):
    calls = []
    document = _doc()
    _patch_document(monkeypatch, document)
    _patch_success_path(monkeypatch, calls)

    async def _boom(db, document_id, *, method, confidence, degraded):
        calls.append("set_document_structure_verdict")
        raise RuntimeError("no structure columns (pre-014 schema)")

    monkeypatch.setattr(
        "database.repositories.set_document_structure_verdict", _boom)
    monkeypatch.setattr(srh, "chunking_service", _FakeChunkingService(calls))

    job = _job(input_data={"document_id": str(document.id)})
    db = FakeDb(job)

    await srh.process_job(SimpleNamespace(db=db), job)

    # The stamp is best-effort: chunks are stored, the job still completes,
    # and q.complete() still gets a bound (non NameError) payload.
    assert calls == [
        "chunk_document",
        "delete_chunks_not_in_hashes",
        "_store_chunks",
        "set_document_structure_verdict",
    ]
    assert job.status == JobStatus.COMPLETED
    # The verdict is computed before the stamp call, so a stamp failure
    # still reports the real computed values (never an unbound local).
    assert job.output_data["method"] == "heading"
    assert job.output_data["confidence"] == pytest.approx(0.9)
    assert job.output_data["degraded"] is False
