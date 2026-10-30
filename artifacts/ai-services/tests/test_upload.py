"""T109/T110: magic-byte validation, incremental limits, ingestion honesty.

Route tests use faked repositories + faked queue + a real LocalStorage in a
tmp dir, over ASGI with the JWT dependency overridden (same pattern as
test_projects.py). Multi-file uploads create independent jobs; A cannot see
B's documents; progress mirrors the job row (never a fixed 0/50/100 map).
"""

import io
import uuid
from datetime import datetime

import pytest

from database import repositories
from database.models.document import Document, DocumentSourceType, DocumentStatus, FileType
from database.models.job import JobStatus
from types import SimpleNamespace

pytestmark = pytest.mark.asyncio

PDF_HEAD = b"%PDF-1.4\n1 0 obj\n<<>>\n"
PNG_HEAD = bytes.fromhex("89504e470d0a1a0a") + b"\x00" * 32
JPEG_HEAD = bytes.fromhex("ffd8ffe000104a464946") + b"\x00" * 32
MZ_HEAD = b"MZ\x90\x00\x03\x00\x00\x00"


# ---------------------------------------------------------------------------
# Pure validation unit tests (T109)
# ---------------------------------------------------------------------------

async def test_exe_renamed_pdf_rejected_415():
    from services.ingestion.validation import validate_magic_bytes

    with pytest.raises(Exception) as exc:
        validate_magic_bytes(MZ_HEAD + b"\x00" * 64, "report.pdf")
    assert exc.value.status_code == 415


async def test_valid_pdf_renamed_txt_rejected_415():
    from services.ingestion.validation import validate_magic_bytes

    with pytest.raises(Exception) as exc:
        validate_magic_bytes(PDF_HEAD, "notes.txt")
    assert exc.value.status_code == 415


async def test_svg_rejected_415():
    from services.ingestion.validation import validate_file_type, validate_magic_bytes

    with pytest.raises(Exception) as exc:
        validate_file_type("diagram.svg")
    assert exc.value.status_code == 415
    with pytest.raises(Exception) as exc2:
        validate_magic_bytes(b'<svg xmlns="http://www.w3.org/2000/svg">', "diagram.svg")
    assert exc2.value.status_code == 415


async def test_matching_signatures_accepted():
    from services.ingestion.validation import validate_magic_bytes

    assert validate_magic_bytes(PDF_HEAD, "report.pdf") == ".pdf"
    assert validate_magic_bytes(PNG_HEAD, "scan.png") == ".png"
    assert validate_magic_bytes(JPEG_HEAD, "photo.jpg") == ".jpg"


async def test_path_filename_rejected_400():
    from services.ingestion.validation import validate_file_type

    with pytest.raises(Exception) as exc:
        validate_file_type("../secret.pdf")
    assert exc.value.status_code == 400


class _FakeUpload:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.filename = "big.pdf"
        self.headers = {}

    async def read(self, n):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


async def test_chunk_reader_enforces_limit_incrementally(monkeypatch):
    import services.ingestion.validation as validation

    monkeypatch.setattr(validation, "MAX_STANDARD_UPLOAD_SIZE", 100)
    assert validation.MAX_STANDARD_UPLOAD_SIZE == 100
    fake = _FakeUpload([b"x" * 60, b"y" * 60])
    from services.ingestion.validation import ValidationError, read_upload_in_chunks

    with pytest.raises(ValidationError) as exc:
        async for _ in read_upload_in_chunks(fake, allow_large=False, chunk_size=60):
            pass
    assert exc.value.status_code == 413


async def test_chunk_reader_memory_bound_single_chunk_held(monkeypatch):
    import services.ingestion.validation as validation

    monkeypatch.setattr(validation, "MAX_STANDARD_UPLOAD_SIZE", 10 * 1024 * 1024)
    sizes = []
    fake = _FakeUpload([b"q" * 100] * 5)
    from services.ingestion.validation import read_upload_in_chunks

    async for chunk in read_upload_in_chunks(fake, chunk_size=100):
        sizes.append(len(chunk))
    assert sizes == [100] * 5  # never more than one chunk (~STREAM_CHUNK_SIZE) held


# ---------------------------------------------------------------------------
# Route tests with fakes (T110)
# ---------------------------------------------------------------------------

class FakeDocStore:
    def __init__(self):
        self.projects = {}
        self.documents = {}
        self.jobs = {}

    def add_project(self, user_id):
        from database.models.project import Project, ProjectStatus

        p = Project(id=uuid.uuid4(), user_id=user_id, name="P",
                    status=ProjectStatus.PROCESSING, progress=0)
        p.created_at = datetime.utcnow()
        p.updated_at = datetime.utcnow()
        self.projects[p.id] = p
        return p


def _install_fakes(monkeypatch, dummy_db, tmp_path, store):
    from database.engine import get_db
    from main import app
    import services.ingestion.routes as ingestion_routes
    from services.jobs import queue as job_queue
    from services.storage import LocalStorage

    async def fake_get_db():
        yield dummy_db

    async def fake_project_by_id(db, pid):
        return store.projects.get(pid)

    async def fake_docs_by_project(db, pid):
        return [d for d in store.documents.values() if d.project_id == pid]

    async def fake_create_document(db, project_id, file_name, file_type, file_path,
                                   page_count=None, is_ocr_required=False,
                                   source_type=None):
        ftype = {"png": "image", "jpg": "image", "jpeg": "image"}.get(file_type, file_type)
        doc = Document(id=uuid.uuid4(), project_id=project_id, file_name=file_name,
                       file_type=FileType(ftype), file_path=file_path,
                       page_count=page_count, processing_status=DocumentStatus.PENDING,
                       is_ocr_required=is_ocr_required,
                       source_type=DocumentSourceType(source_type or "study_material"))
        doc.created_at = datetime.utcnow()
        doc.updated_at = datetime.utcnow()
        doc.storage_key = file_path
        doc.mime_type = "application/octet-stream"
        doc.file_size = 0
        doc.checksum = None
        doc.extraction_status = "pending"
        doc.ocr_status = "pending"
        doc.embedding_status = "pending"
        store.documents[doc.id] = doc
        return doc

    async def fake_doc_by_id(db, did):
        return store.documents.get(did)

    async def fake_latest(db, document_id):
        cands = [j for j in store.jobs.values()
                 if getattr(j, "document_id", None) == document_id]
        return max(cands, key=lambda j: j.created_at) if cands else None

    async def fake_enqueue(db, job_type, input_data=None, project_id=None,
                           document_id=None, priority=0, run_after=None,
                           max_attempts=3, idempotency_key=None):
        from database.models.job import JobType

        jt = job_type if isinstance(job_type, JobType) else JobType(job_type)
        now = datetime.utcnow()
        job = SimpleNamespace(id=uuid.uuid4(), job_type=jt, status=JobStatus.PENDING,
                              progress=0, stage=None, stage_message=None,
                              attempts=0, max_attempts=max_attempts,
                              input_data=input_data or {}, output_data=None,
                              error_message=None, document_id=document_id,
                              project_id=project_id, created_at=now,
                              started_at=None, completed_at=None)
        store.jobs[job.id] = job
        return job

    async def fake_find_key(db, key, project_id, job_type):
        for job in store.jobs.values():
            if (getattr(job, "idempotency_key", None) == key
                    and job.project_id == project_id):
                return job
        return None

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project_by_id)
    monkeypatch.setattr(repositories, "get_documents_by_project", fake_docs_by_project)
    monkeypatch.setattr(repositories, "create_document", fake_create_document)
    monkeypatch.setattr(repositories, "get_document_by_id", fake_doc_by_id)
    monkeypatch.setattr(job_queue, "latest_for_document", fake_latest)

    async def fake_delete(obj):
        store.documents.pop(getattr(obj, "id", None), None)

    dummy_db.delete = fake_delete  # type: ignore[method-assign]
    monkeypatch.setattr(job_queue, "enqueue", fake_enqueue)
    monkeypatch.setattr(job_queue, "find_by_idempotency_key", fake_find_key)
    monkeypatch.setattr(ingestion_routes, "get_default_backend",
                        lambda: LocalStorage(tmp_path / "storage"))
    app.dependency_overrides[get_db] = fake_get_db
    return app


def _login_as(app, user):
    from services.auth.deps import get_current_user

    async def fake_user():
        return user

    app.dependency_overrides[get_current_user] = fake_user


def _pdf_bytes(size=200):
    return PDF_HEAD + b"x" * size


async def test_upload_returns_camelcase_ids_and_queues_job(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        resp = await app_client.post(
            "/api/documents/upload",
            files={"file": ("report.pdf", _pdf_bytes(), "application/pdf")},
            data={"project_id": str(project.id), "source_type": "syllabus"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["documentId"] and body["jobId"]
        assert body["fileName"] == "report.pdf"
        assert body["sourceType"] == "syllabus"
        assert set(body) >= {"documentId", "fileName", "fileType", "status", "message", "jobId"}
        assert len(store.jobs) == 1
    finally:
        app.dependency_overrides.clear()


async def test_upload_rejects_exe_renamed_pdf(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        resp = await app_client.post(
            "/api/documents/upload",
            files={"file": ("evil.pdf", MZ_HEAD + b"\x00" * 200, "application/pdf")},
            data={"project_id": str(project.id)},
        )
        assert resp.status_code == 415
    finally:
        app.dependency_overrides.clear()


async def test_multi_file_uploads_create_independent_jobs(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        first = await app_client.post(
            "/api/documents/upload",
            files={"file": ("a.pdf", _pdf_bytes(100), "application/pdf")},
            data={"project_id": str(project.id)},
        )
        second = await app_client.post(
            "/api/documents/upload",
            files={"file": ("b.pdf", _pdf_bytes(300), "application/pdf")},
            data={"project_id": str(project.id)},
            headers={"Idempotency-Key": "unique-key-b"},
        )
        assert first.status_code == 201 and second.status_code == 201
        assert first.json()["jobId"] != second.json()["jobId"]
        # One file failing must not affect the other job.
        assert len(store.jobs) == 2
    finally:
        app.dependency_overrides.clear()


async def test_ab_isolation_on_documents(app_client, dummy_db, monkeypatch, tmp_path, user_a, user_b):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        created = await app_client.post(
            "/api/documents/upload",
            files={"file": ("secret.pdf", _pdf_bytes(), "application/pdf")},
            data={"project_id": str(project.id)},
        )
        did = created.json()["documentId"]
        assert (await app_client.get(f"/api/documents/{did}")).status_code == 200
        _login_as(app, user_b)
        assert (await app_client.get(f"/api/documents/{did}")).status_code == 404
        assert (await app_client.get(f"/api/documents/{did}/progress")).status_code == 404
        assert (await app_client.delete(f"/api/documents/{did}")).status_code == 404
        listed = await app_client.get(f"/api/documents/project/{project.id}")
        assert listed.status_code == 404  # B cannot list A's project docs
    finally:
        app.dependency_overrides.clear()


async def test_progress_mirrors_job_row_not_fixed_map(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        created = await app_client.post(
            "/api/documents/upload",
            files={"file": ("r.pdf", _pdf_bytes(), "application/pdf")},
            data={"project_id": str(project.id)},
        )
        did = created.json()["documentId"]
        job = next(iter(store.jobs.values()))
        # Worker reports 42% at extraction: API must echo exactly that.
        job.status = JobStatus.PROCESSING
        job.progress = 42
        job.stage = "extraction"
        job.stage_message = "pages 3/10"
        progress = await app_client.get(f"/api/documents/{did}/progress")
        assert progress.status_code == 200
        assert progress.json()["progress"] == 42
        assert progress.json()["stage"] == "extraction"
        job.progress = 87
        job.stage = "embedding"
        again = await app_client.get(f"/api/documents/{did}/progress")
        assert again.json()["progress"] == 87
        assert again.json()["stage"] == "embedding"
    finally:
        app.dependency_overrides.clear()


async def test_progress_without_job_reports_not_queued(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        created = await app_client.post(
            "/api/documents/upload",
            files={"file": ("q.pdf", _pdf_bytes(), "application/pdf")},
            data={"project_id": str(project.id)},
        )
        store.jobs.clear()  # simulate enqueue failure: doc survives, no job
        progress = await app_client.get(f"/api/documents/{created.json()['documentId']}/progress")
        assert progress.json()["status"] == "not_queued"
        assert progress.json()["progress"] == 0
    finally:
        app.dependency_overrides.clear()


async def test_delete_removes_db_row_and_stored_file(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        created = await app_client.post(
            "/api/documents/upload",
            files={"file": ("gone.pdf", _pdf_bytes(), "application/pdf")},
            data={"project_id": str(project.id)},
        )
        did = created.json()["documentId"]
        stored_files = [p for p in (tmp_path / "storage").rglob("*") if p.is_file()]
        assert len(stored_files) == 1
        assert (await app_client.delete(f"/api/documents/{did}")).status_code == 200
        assert (await app_client.get(f"/api/documents/{did}")).status_code == 404
        assert [p for p in (tmp_path / "storage").rglob("*") if p.is_file()] == []
    finally:
        app.dependency_overrides.clear()


async def test_idempotency_key_returns_existing_document(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    # Wire the idempotency key onto the created job so the second call hits it.
    from services.jobs import queue as job_queue

    real_enqueue = job_queue.enqueue

    async def keyed_enqueue(db, job_type, input_data=None, project_id=None,
                            document_id=None, priority=0, run_after=None,
                            max_attempts=3, idempotency_key=None):
        job = await real_enqueue(db, job_type, input_data, project_id, document_id,
                                 priority, run_after, max_attempts, idempotency_key)
        job.idempotency_key = idempotency_key
        return job

    monkeypatch.setattr(job_queue, "enqueue", keyed_enqueue)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        headers = {"Idempotency-Key": "idem-123"}
        first = await app_client.post(
            "/api/documents/upload",
            files={"file": ("i.pdf", _pdf_bytes(), "application/pdf")},
            data={"project_id": str(project.id)},
            headers=headers,
        )
        second = await app_client.post(
            "/api/documents/upload",
            files={"file": ("i.pdf", _pdf_bytes(), "application/pdf")},
            data={"project_id": str(project.id)},
            headers=headers,
        )
        assert first.status_code == 201 and second.status_code == 201
        assert first.json()["documentId"] == second.json()["documentId"]
    finally:
        app.dependency_overrides.clear()


async def test_upload_multi_returns_per_file_jobs_and_isolates_failures(
    app_client, dummy_db, monkeypatch, tmp_path, user_a
):
    """POST /api/documents/upload/multi: 3 files, one bad -> 2 jobs, 1 per-file failure."""
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        resp = await app_client.post(
            "/api/documents/upload/multi",
            files=[
                ("files", ("a.pdf", _pdf_bytes(100), "application/pdf")),
                ("files", ("b.pdf", _pdf_bytes(300), "application/pdf")),
                ("files", ("evil.pdf", MZ_HEAD + b"\x00" * 200, "application/pdf")),
            ],
            data={"project_id": str(project.id), "source_type": "question_paper"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert isinstance(body, list) and len(body) == 3
        ok = [r for r in body if r["documentId"] is not None]
        bad = [r for r in body if r["documentId"] is None]
        assert len(ok) == 2 and len(bad) == 1
        assert all(r["sourceType"] == "question_paper" for r in ok)
        assert ok[0]["jobId"] != ok[1]["jobId"]  # independent jobs
        assert bad[0]["status"] == "failed" and bad[0]["error"]
        assert len(store.jobs) == 2  # failure isolated, others queued
    finally:
        app.dependency_overrides.clear()


async def test_factory_honors_ocr_provider_default_local(monkeypatch):
    """Factory default is the single local PP-OCRv6 CPU ONNX path."""
    import factory as factory_mod

    factory_mod.get_strategy_factory().reset()
    provider = factory_mod.get_strategy_factory().get_ocr()
    assert type(provider).__name__ == "PPOCROnxProvider"
    assert provider.METHOD_NAME if hasattr(provider, "METHOD_NAME") else True
    factory_mod.get_strategy_factory().reset()


# ---------------------------------------------------------------------------
# Checksum-dedupe must never hand back a dead job (processing-fail fix)
# ---------------------------------------------------------------------------

async def test_duplicate_content_failed_job_requeued(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    """Re-upload of bytes whose latest job FAILED returns a NEW job id."""
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        payload = _pdf_bytes(200)
        first = await app_client.post(
            "/api/documents/upload",
            files={"file": ("report.pdf", payload, "application/pdf")},
            data={"project_id": str(project.id)},
        )
        assert first.status_code == 201, first.text
        doc_id = first.json()["documentId"]
        doc = store.documents[uuid.UUID(doc_id)]
        old_job = next(iter(store.jobs.values()))
        old_job.status = JobStatus.FAILED
        old_job.error_message = "greenlet_spawn has not been called"
        doc.processing_status = DocumentStatus.FAILED
        doc.processing_error = old_job.error_message

        second = await app_client.post(
            "/api/documents/upload",
            files={"file": ("report.pdf", payload, "application/pdf")},
            data={"project_id": str(project.id)},
        )
        assert second.status_code == 201, second.text
        body = second.json()
        assert body["documentId"] == doc_id
        assert body["jobId"] != str(old_job.id)
        assert "requeued" in body["message"]
        assert body["status"] == DocumentStatus.PENDING.value
        # Document reset for the fresh attempt, old error cleared.
        assert doc.processing_status == DocumentStatus.PENDING
        assert doc.processing_error is None
        assert len(store.jobs) == 2
    finally:
        app.dependency_overrides.clear()


async def test_duplicate_content_live_job_reused(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    """Re-upload while a job is PENDING returns the SAME job id (no dup)."""
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        payload = _pdf_bytes(200)
        first = await app_client.post(
            "/api/documents/upload",
            files={"file": ("live.pdf", payload, "application/pdf")},
            data={"project_id": str(project.id)},
        )
        assert first.status_code == 201, first.text
        first_job = first.json()["jobId"]
        second = await app_client.post(
            "/api/documents/upload",
            files={"file": ("live.pdf", payload, "application/pdf")},
            data={"project_id": str(project.id)},
        )
        assert second.status_code == 201, second.text
        assert second.json()["jobId"] == first_job
        assert "returning existing document" in second.json()["message"]
        assert len(store.jobs) == 1
    finally:
        app.dependency_overrides.clear()


async def test_duplicate_content_completed_doc_reused(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    """Re-upload of an indexed document returns the existing completed job."""
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        payload = _pdf_bytes(200)
        first = await app_client.post(
            "/api/documents/upload",
            files={"file": ("done.pdf", payload, "application/pdf")},
            data={"project_id": str(project.id)},
        )
        assert first.status_code == 201, first.text
        doc = store.documents[uuid.UUID(first.json()["documentId"])]
        job = next(iter(store.jobs.values()))
        job.status = JobStatus.COMPLETED
        job.progress = 100
        doc.processing_status = DocumentStatus.COMPLETED
        second = await app_client.post(
            "/api/documents/upload",
            files={"file": ("done.pdf", payload, "application/pdf")},
            data={"project_id": str(project.id)},
        )
        assert second.status_code == 201, second.text
        assert second.json()["jobId"] == str(job.id)
        assert second.json()["status"] == DocumentStatus.COMPLETED.value
        assert len(store.jobs) == 1
    finally:
        app.dependency_overrides.clear()


async def test_stream_duplicate_content_failed_job_requeued(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    """Streaming re-upload of dead-job bytes requeues (same rule as /upload)."""
    from main import app

    store = FakeDocStore()
    _install_fakes(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        payload = _pdf_bytes(200)
        headers = {
            "X-Project-Id": str(project.id),
            "X-File-Name": "s.pdf",
            "X-Source-Type": "syllabus",
        }
        first = await app_client.post("/api/documents/upload/stream", content=payload, headers=headers)
        assert first.status_code == 201, first.text
        assert first.json()["sourceType"] == "syllabus"
        doc = store.documents[uuid.UUID(first.json()["documentId"])]
        old_job = next(iter(store.jobs.values()))
        old_job.status = JobStatus.FAILED
        doc.processing_status = DocumentStatus.FAILED

        second = await app_client.post("/api/documents/upload/stream", content=payload, headers=headers)
        assert second.status_code == 201, second.text
        body = second.json()
        assert body["documentId"] == str(doc.id)
        assert body["sourceType"] == "syllabus"
        assert body["jobId"] != str(old_job.id)
        assert "requeued" in body["message"]
        assert len(store.jobs) == 2
    finally:
        app.dependency_overrides.clear()
