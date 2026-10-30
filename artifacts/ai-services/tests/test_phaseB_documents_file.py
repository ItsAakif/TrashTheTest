"""Phase B (B1): GET /api/documents/{id}/file streams stored source bytes.

Offline fakes (same harness as test_upload.py): faked repositories + faked
queue + real LocalStorage in tmp dir, JWT dependency overridden, ASGI only.

Pins: owner gets the exact bytes inline as application/pdf; another user's
document is 404 (IDOR, never content); unauthenticated is 401; a document
whose storage bytes are gone is a typed 404 (never 500); a malformed id
is 400.
"""

import uuid

import pytest

from database import repositories
from database.models.document import DocumentSourceType, DocumentStatus, FileType

pytestmark = pytest.mark.asyncio

PDF_HEAD = b"%PDF-1.4\n1 0 obj\n<<>>\n"


def _pdf_bytes(size=200):
    return PDF_HEAD + b"x" * size


class FakeDocStore:
    def __init__(self):
        self.projects = {}
        self.documents = {}
        self.jobs = {}

    def add_project(self, user_id):
        from datetime import datetime

        from database.models.project import Project, ProjectStatus

        p = Project(id=uuid.uuid4(), user_id=user_id, name="P",
                    status=ProjectStatus.PROCESSING, progress=0)
        p.created_at = datetime.utcnow()
        p.updated_at = datetime.utcnow()
        self.projects[p.id] = p
        return p


def _install(monkeypatch, dummy_db, tmp_path, store):
    from datetime import datetime

    from database.engine import get_db
    from database.models.document import Document
    from database.models.job import JobStatus
    from main import app
    import services.ingestion.routes as ingestion_routes
    from services.jobs import queue as job_queue
    from services.storage import LocalStorage
    from types import SimpleNamespace

    async def fake_get_db():
        yield dummy_db

    async def fake_project_by_id(db, pid):
        return store.projects.get(pid)

    async def fake_doc_by_id(db, did):
        return store.documents.get(did)

    async def fake_create_document(db, project_id, file_name, file_type, file_path,
                                   page_count=None, is_ocr_required=False,
                                   source_type=None):
        doc = Document(id=uuid.uuid4(), project_id=project_id, file_name=file_name,
                       file_type=FileType(file_type), file_path=file_path,
                       page_count=page_count, processing_status=DocumentStatus.PENDING,
                       is_ocr_required=is_ocr_required,
                       source_type=DocumentSourceType(source_type or "study_material"))
        doc.created_at = datetime.utcnow()
        doc.updated_at = datetime.utcnow()
        doc.storage_key = file_path
        doc.mime_type = "application/pdf"
        doc.file_size = 0
        doc.checksum = None
        store.documents[doc.id] = doc
        return doc

    async def fake_latest(db, document_id):
        return None

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
        return None

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project_by_id)
    monkeypatch.setattr(repositories, "get_document_by_id", fake_doc_by_id)
    monkeypatch.setattr(repositories, "create_document", fake_create_document)
    monkeypatch.setattr(job_queue, "latest_for_document", fake_latest)
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


async def _upload(app_client, project, body):
    resp = await app_client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", body, "application/pdf")},
        data={"project_id": str(project.id)},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["documentId"]


async def test_file_streams_bytes_inline_as_pdf(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        body = _pdf_bytes()
        did = await _upload(app_client, project, body)

        resp = await app_client.get(f"/api/documents/{did}/file")
        assert resp.status_code == 200, resp.text
        assert resp.content == body
        assert resp.headers["content-type"] == "application/pdf"
        assert resp.headers["content-disposition"].startswith("inline;")
    finally:
        app.dependency_overrides.clear()


async def test_file_other_user_gets_404_never_bytes(app_client, dummy_db, monkeypatch, tmp_path, user_a, user_b):
    from main import app

    store = FakeDocStore()
    _install(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        did = await _upload(app_client, project, _pdf_bytes())

        _login_as(app, user_b)
        resp = await app_client.get(f"/api/documents/{did}/file")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


async def test_file_unauthenticated_is_401(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        did = await _upload(app_client, project, _pdf_bytes())

        from services.auth.deps import get_current_user

        app.dependency_overrides.pop(get_current_user, None)
        resp = await app_client.get(f"/api/documents/{did}/file")
        assert resp.status_code in (401, 403), resp.status_code
    finally:
        app.dependency_overrides.clear()


async def test_file_missing_storage_bytes_is_typed_404(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install(monkeypatch, dummy_db, tmp_path, store)
    try:
        project = store.add_project(user_a.id)
        _login_as(app, user_a)
        did = await _upload(app_client, project, _pdf_bytes())

        doc = store.documents[uuid.UUID(did)]
        doc.file_path = "nope/missing.pdf"
        resp = await app_client.get(f"/api/documents/{did}/file")
        assert resp.status_code == 404
        assert "unavailable" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


async def test_file_malformed_id_is_400(app_client, dummy_db, monkeypatch, tmp_path, user_a):
    from main import app

    store = FakeDocStore()
    _install(monkeypatch, dummy_db, tmp_path, store)
    try:
        _login_as(app, user_a)
        resp = await app_client.get("/api/documents/not-a-uuid/file")
        assert resp.status_code == 400
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Phase B live find: FUSED_SQL reached live Postgres for the first time and
# asyncpg rejected `:embedding_model::text` (SQLAlchemy text() misparses a
# bound param followed by ::) plus untyped LIMIT params; the swallowed
# failure then aborted the session transaction and 500'd generation.
# ---------------------------------------------------------------------------

import re as _re


async def test_fused_sql_params_use_cast_not_bare_colons():
    from services.retrieval.hybrid import FUSED_SQL

    code = "\n".join(
        line for line in FUSED_SQL.splitlines()
        if not line.strip().startswith("--")
    )
    # No bound param may be directly followed by :: (misparsed live).
    assert not _re.search(r":[a-z_]+::", code), \
        "bare :param::type breaks SQLAlchemy text() on asyncpg"
    # LIMIT inputs must be explicitly integer so LIMIT (SELECT ...) is bigint.
    assert "CAST(:per_arm AS integer)" in FUSED_SQL
    assert "CAST(:limit AS integer)" in FUSED_SQL
    assert "CAST(:embedding_model AS text)" in FUSED_SQL


class _AbortOnceDB:
    """First execute raises (poisoned txn), then behaves after rollback."""

    def __init__(self):
        from types import SimpleNamespace

        self.rollbacks = 0
        self.calls = 0
        # Look like a Postgres bind so the hybrid path executes SQL.
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

    async def execute(self, *args, **kwargs):
        self.calls += 1
        # Call 1 (SET LOCAL) succeeds; call 2 (the fused statement) fails,
        # poisoning the txn exactly like the live Phase B find.
        if self.calls == 2:
            raise Exception("synthetic SQL failure")

        class _Scalars:
            def all(self):
                return []

        class _Result:
            def scalars(self):
                return _Scalars()

        return _Result()

    async def rollback(self):
        self.rollbacks += 1


async def test_hybrid_search_rolls_back_then_falls_back(monkeypatch):
    from services.retrieval import service as retrieval_module

    svc = retrieval_module.retrieval_service
    db = _AbortOnceDB()
    rows, backend = await svc._hybrid_search(
        db, "photosynthesis", [0.01] * 8, None, None, None,
        per_arm=5, limit=3,
    )
    assert backend == "offline-fallback"
    assert rows == []
    assert db.rollbacks == 1, "aborted txn must be rolled back before fallback"


async def test_build_citation_never_lazy_loads_document():
    """Phase B live find: getattr(chunk, "document") lazy-loads and
    MissingGreenlets in async context. The builder must only read an
    already-loaded relationship (never emit IO)."""
    from services.retrieval.citations import build_citation_from_chunk

    class ExplodingChunk:
        id = "c1"
        document_id = "d1"
        content = "Photosynthesis uses chlorophyll."
        chunk_metadata = {"page": 2}
        page_start = 2
        page_end = 2
        section_path = []
        heading = None

        @property
        def document(self):  # pragma: no cover - must never be touched
            raise AssertionError("lazy relationship IO in async path")

    citation = build_citation_from_chunk(ExplodingChunk())
    assert citation.chunk_id == "c1"
    assert citation.document_id == "d1"
    assert citation.page_number == 2
    assert citation.span.startswith("Photosynthesis")
    assert citation.valid is True
