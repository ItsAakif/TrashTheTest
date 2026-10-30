"""Phase A recovery tests (T401-T409, ADR-006 D3).

Offline fakes; no network, no model weights, no Postgres. Each test pins
one acceptance: eager project load, honest validate2, embedding write
correctness, failure honesty + merge writer, flush-only chunks, alias
acceptance, JWT study start with real content, tutor coercion, monotonic
progress.
"""

import pathlib
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

from database.models.job import JobStatus

# ---------------- fakes ----------------

class FakeDB:
    def __init__(self, job=None, doc=None, pages=None, chunks=None):
        self.job = job
        self.doc = doc
        self.pages = pages or []
        self.chunks = chunks or []
        self.commits = 0
        self.flushes = 0
        self.progress_seen = []

    async def get(self, model, job_id):
        if self.job is not None and str(self.job.id) == str(job_id):
            return self.job
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None

    async def flush(self):
        self.flushes += 1


def _doc(doc_id, project_id, user_id=None):
    return SimpleNamespace(
        id=doc_id, project_id=project_id, file_name="f.pdf",
        file_path="f.pdf", file_type=SimpleNamespace(value="pdf"),
        processing_status=SimpleNamespace(value="pending"),
        extracted_text=None, indexing_state=None, extraction_status=None,
        project=SimpleNamespace(user_id=user_id or uuid.uuid4()))


def _job(doc_id, project_id, output=None, max_attempts=3):
    now = datetime.utcnow()
    return SimpleNamespace(
        id=uuid.uuid4(), job_type=SimpleNamespace(value="document_processing"),
        status=JobStatus.PROCESSING, progress=0, stage=None, stage_message=None,
        attempts=0, max_attempts=max_attempts,
        input_data={"document_id": str(doc_id), "project_id": str(project_id),
                    "file_path": "x", "file_type": "txt"},
        output_data=output or {}, error_message=None,
        document_id=doc_id, project_id=project_id,
        locked_by="w", locked_at=now, created_at=now, updated_at=now,
        started_at=now, completed_at=None)


def _install(monkeypatch, db, fail_captured=None):
    from database import repositories
    from services.jobs import queue as q

    async def get_doc_eager(d, i):
        return db.doc

    async def get_doc(d, i):
        return db.doc

    async def update_status(d, i, status, extracted_text=None, error=None):
        db.doc.processing_status = SimpleNamespace(value=status.value)
        return db.doc

    async def set_indexing(d, i, state, degraded_reason=None, extraction_status=None):
        db.doc.indexing_state = state
        if degraded_reason is not None:
            db.doc.processing_error = degraded_reason
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
        # Mirror the real single-writer merge semantics.
        if progress < job.progress:
            progress = job.progress
        job.progress = progress
        job.stage = stage
        job.stage_message = stage_message
        out = dict(job.output_data or {})
        if stage:
            out["_stage"] = stage
        job.output_data = out
        db.progress_seen.append(progress)
        return job

    async def complete(d, job_id, output_data=None):
        db.job.status = JobStatus.COMPLETED
        db.job.progress = 100
        db.job.output_data = output_data
        return db.job

    async def fail(d, job_id, error, retry=True):
        db.job.attempts += 1
        if fail_captured is not None:
            fail_captured.append({"error": error, "retry": retry,
                                  "attempts": db.job.attempts})
        if retry and db.job.attempts < db.job.max_attempts:
            db.job.status = JobStatus.PENDING
        else:
            db.job.status = JobStatus.FAILED
            db.job.error_message = error
        return db.job

    monkeypatch.setattr(repositories, "get_document_by_id_with_project", get_doc_eager)
    monkeypatch.setattr(repositories, "get_document_by_id", get_doc)
    monkeypatch.setattr(repositories, "update_document_status", update_status)
    monkeypatch.setattr(repositories, "set_document_indexing_state", set_indexing)
    monkeypatch.setattr(repositories, "get_pages", get_pages)
    monkeypatch.setattr(repositories, "page_method_breakdown", breakdown)
    monkeypatch.setattr(repositories, "get_chunks_by_document", chunks_by_doc)
    monkeypatch.setattr(q, "report_progress", report_progress)
    monkeypatch.setattr(q, "complete", complete)
    monkeypatch.setattr(q, "fail", fail)


def _stub_pipeline(monkeypatch, db, embed_side_effect="ok"):
    """Stub chunk-store + embedder; drive the real handler end to end."""
    import services.chunking.service as chunk_svc
    import services.embedding_service.service as emb_svc

    async def store_chunks(self, d, document_id, project_id, chunks):
        for c in chunks:
            db.chunks.append(c)

    monkeypatch.setattr(chunk_svc.ChunkingService, "_store_chunks", store_chunks)
    monkeypatch.setattr(emb_svc.LocalEmbeddingService, "__init__",
                        lambda self, *a, **k: None)

    if embed_side_effect == "fail":
        async def boom(self, document_id, project_id, db, job_id=None, **kw):
            raise RuntimeError("weights absent offline")
        monkeypatch.setattr(emb_svc.LocalEmbeddingService,
                            "generate_embeddings", boom)
    else:
        async def fake_generate(self, document_id, project_id, db, job_id=None,
                                progress_callback=None, **kw):
            from services.embedding_service.types import EmbeddingProgress
            n = len(db.chunks)
            if progress_callback and n:
                await progress_callback(EmbeddingProgress(
                    document_id=str(document_id), current_batch=1,
                    total_batches=1, chunks_processed=n, total_chunks=n,
                    status="embedding", progress_percent=100.0,
                    message=f"Processed {n}/{n} chunks"))
            return SimpleNamespace(embedded_chunks=n)
        monkeypatch.setattr(emb_svc.LocalEmbeddingService,
                            "generate_embeddings", fake_generate)


async def _run_txt(monkeypatch, tmp_path, text, pages=None):
    from services.jobs.handlers.document_processing import process_job

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "doc.txt"
    f.write_text(text, encoding="utf-8")
    doc = _doc(doc_id, project_id)
    job = _job(doc_id, project_id)
    job.input_data["file_path"] = str(f)
    db = FakeDB(job=job, doc=doc, pages=list(pages or []), chunks=[])
    _install(monkeypatch, db)

    from database import repositories
    from services.ingestion import storage as storage_mod
    from pathlib import Path as _P

    real_pages = []

    async def upsert_page(d, document_id, page_number, text="", method="native",
                          confidence=0.0, parser_version=None, image_stored=False,
                          bbox_summary=None, failure_reason=None):
        real_pages.append(SimpleNamespace(document_id=document_id,
                                          page_number=page_number, text=text,
                                          method=method, confidence=confidence))
        db.pages = real_pages
        return real_pages[-1]

    monkeypatch.setattr(repositories, "upsert_page", upsert_page)
    monkeypatch.setattr(storage_mod.storage_manager, "get_absolute_path",
                        lambda p: _P(p), raising=False)
    _stub_pipeline(monkeypatch, db)
    await process_job(SimpleNamespace(db=db), job)
    return db


# ---------------- T401 ----------------

def test_t401_no_lazy_relationship_access_in_worker():
    src = pathlib.Path("services/jobs/handlers/document_processing.py").read_text()
    assert "document.project." not in src
    assert "get_document_by_id_with_project" in src


@pytest.mark.asyncio
async def test_t401_worker_prefers_input_data_ids(monkeypatch, tmp_path):
    """Chunker receives project_id/user_id from job.input_data, not ORM."""
    import services.chunking.service as chunk_svc

    seen = {}
    real_chunk = chunk_svc.semantic_chunker.chunk_document

    def spy_chunk(doc_input):
        seen["project_id"] = doc_input.project_id
        seen["user_id"] = doc_input.user_id
        return real_chunk(doc_input)

    monkeypatch.setattr(chunk_svc.semantic_chunker, "chunk_document", spy_chunk)
    user_id = uuid.uuid4()
    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "d.txt"
    f.write_text("honest content " * 40, encoding="utf-8")
    # Document WITHOUT a project relationship at all.
    doc = SimpleNamespace(id=doc_id, project_id=project_id, file_name="d.txt",
                          file_path=str(f), file_type=SimpleNamespace(value="txt"),
                          processing_status=SimpleNamespace(value="pending"),
                          extracted_text=None, indexing_state=None,
                          extraction_status=None, project=None)
    job = _job(doc_id, project_id)
    job.input_data.update({"file_path": str(f), "project_id": str(project_id),
                           "user_id": str(user_id)})
    db = FakeDB(job=job, doc=doc)
    _install(monkeypatch, db)
    from database import repositories
    from services.ingestion import storage as storage_mod
    from pathlib import Path as _P
    real_pages = []

    async def upsert_page(d, document_id, page_number, text="", method="native",
                          confidence=0.0, **kw):
        real_pages.append(SimpleNamespace(document_id=document_id,
                                          page_number=page_number, text=text,
                                          method=method, confidence=confidence))
        db.pages = real_pages
        return real_pages[-1]

    monkeypatch.setattr(repositories, "upsert_page", upsert_page)
    monkeypatch.setattr(storage_mod.storage_manager, "get_absolute_path",
                        lambda p: _P(p), raising=False)
    _stub_pipeline(monkeypatch, db)
    from services.jobs.handlers.document_processing import process_job
    await process_job(SimpleNamespace(db=db), job)
    assert db.job.status == JobStatus.COMPLETED
    assert seen["project_id"] == str(project_id)
    assert seen["user_id"] == str(user_id)


# ---------------- T402 ----------------

@pytest.mark.asyncio
async def test_t402_zero_pages_fails_terminal_no_retry(monkeypatch, tmp_path):
    """Extractor produced nothing: fail once with retry=False."""
    from services.jobs.handlers.document_processing import process_job
    import services.extraction.extractor as ext_mod

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    doc = _doc(doc_id, project_id)
    job = _job(doc_id, project_id, max_attempts=3)
    job.input_data["file_path"] = str(f)
    db = FakeDB(job=job, doc=doc, pages=[])
    captured = []
    _install(monkeypatch, db, fail_captured=captured)

    class NoPages(ext_mod.DocumentExtractor):
        async def extract(self, *a, **k):
            return SimpleNamespace(pages=[], full_text="")

    monkeypatch.setattr(ext_mod, "DocumentExtractor", NoPages)
    _stub_pipeline(monkeypatch, db)
    await process_job(SimpleNamespace(db=db), job)
    assert db.job.status == JobStatus.FAILED
    assert captured and captured[0]["retry"] is False
    assert "zero pages" in captured[0]["error"]


@pytest.mark.asyncio
async def test_t402_pages_without_text_degraded_once(monkeypatch, tmp_path):
    """Pages exist, OCR unavailable: completed + no_text, never retried."""
    from services.jobs.handlers.document_processing import process_job

    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "scan.txt"
    f.write_text("x" * 10, encoding="utf-8")
    doc = _doc(doc_id, project_id)
    job = _job(doc_id, project_id)
    job.input_data["file_path"] = str(f)
    pages = [SimpleNamespace(document_id=doc_id, page_number=i, text="",
                             method="ocr_unavailable", confidence=0.0)
             for i in (1, 2)]
    db = FakeDB(job=job, doc=doc, pages=list(pages))
    captured = []
    _install(monkeypatch, db, fail_captured=captured)
    # Skip extract (already have pages): mark it done.
    job.output_data["_stages_done"] = ["validate", "extract", "structure"]
    job.stage = "structure"
    _stub_pipeline(monkeypatch, db)
    await process_job(SimpleNamespace(db=db), job)
    assert captured == []  # never failed/retried
    assert db.job.status == JobStatus.COMPLETED
    assert db.doc.indexing_state == "no_text"
    assert db.job.output_data.get("indexing_state") == "no_text"
    assert db.job.output_data.get("degraded_reason")


# ---------------- T403 ----------------

def test_t403_vector_round_trip_dimension_and_components():
    from services.embedding_service.service import LocalEmbeddingService

    svc = LocalEmbeddingService.__new__(LocalEmbeddingService)
    vec = [1.5, -2.25, 3.125, 0.0]
    blob = svc._pack_vector(vec)
    assert isinstance(blob, bytes) and len(blob) == 4 * len(vec)
    back = LocalEmbeddingService.unpack_vector(blob)
    assert len(back) == len(vec)
    for a, b in zip(vec, back):
        assert abs(a - b) < 1e-6


def test_t403_cache_key_identical_everywhere():
    from services.embedding_service.service import LocalEmbeddingService

    svc = LocalEmbeddingService.__new__(LocalEmbeddingService)
    svc._embedding_cache = {}
    svc._cache_backend = None
    model, chash = "Qwen/Qwen3-Embedding-0.6B", "abc123"
    svc._cache_set(model, chash, b"vec")
    assert svc._cache_key(model, chash) == "emb:Qwen/Qwen3-Embedding-0.6B:abc123"
    assert svc._cache_get(model, chash) == b"vec"


def test_t403_no_print_import_top_chunk_metadata():
    src = pathlib.Path("services/embedding_service/service.py").read_text()
    assert "print(" not in src
    lines = src.splitlines()
    top_imports = [i for i, l in enumerate(lines) if l.startswith("import asyncio")]
    assert top_imports and top_imports[0] < 20
    # ORM Chunk reads must use chunk_metadata (EmbeddingChunkInput.metadata
    # in the batch loops is the pydantic field and is correct).
    assert "metadata = chunk.metadata" not in src


# ---------------- T404/T409 writer ----------------

@pytest.mark.asyncio
async def test_t404_update_job_status_merges_output_data():
    from database import repositories

    job = SimpleNamespace(output_data={"_stages_done": ["validate"], "_stage": "validate"},
                          status=None, progress=None, stage=None, stage_message=None,
                          input_data=None, error_message=None, started_at=None,
                          completed_at=None, updated_at=None)

    class DB:
        async def get(self, model, job_id):
            return job

    out = await repositories.update_job_status(
        DB(), uuid.uuid4(), JobStatus.PROCESSING, progress=50,
        output_data={"message": "Embedding 1/2 chunks"})
    assert out.output_data["_stages_done"] == ["validate"]
    assert out.output_data["_stage"] == "validate"
    assert out.output_data["message"] == "Embedding 1/2 chunks"


@pytest.mark.asyncio
async def test_t404_embedding_failure_is_unembedded_never_indexed(monkeypatch, tmp_path):
    doc_id, project_id = uuid.uuid4(), uuid.uuid4()
    f = tmp_path / "e.txt"
    f.write_text("real text " * 40, encoding="utf-8")
    doc = _doc(doc_id, project_id)
    job = _job(doc_id, project_id)
    job.input_data["file_path"] = str(f)
    db = FakeDB(job=job, doc=doc)
    _install(monkeypatch, db)
    from database import repositories
    from services.ingestion import storage as storage_mod
    from pathlib import Path as _P
    real_pages = []

    async def upsert_page(d, document_id, page_number, text="", method="native",
                          confidence=0.0, **kw):
        real_pages.append(SimpleNamespace(document_id=document_id,
                                          page_number=page_number, text=text,
                                          method=method, confidence=confidence))
        db.pages = real_pages
        return real_pages[-1]

    monkeypatch.setattr(repositories, "upsert_page", upsert_page)
    monkeypatch.setattr(storage_mod.storage_manager, "get_absolute_path",
                        lambda p: _P(p), raising=False)
    _stub_pipeline(monkeypatch, db, embed_side_effect="fail")
    from services.jobs.handlers.document_processing import process_job
    await process_job(SimpleNamespace(db=db), job)
    assert db.job.status == JobStatus.COMPLETED
    assert db.job.output_data.get("indexing_state") == "unembedded"
    assert db.job.output_data.get("indexing_state") != "indexed"
    assert db.doc.indexing_state == "unembedded"


# ---------------- T405 ----------------

@pytest.mark.asyncio
async def test_t405_store_chunks_flushes_never_commits(monkeypatch):
    import services.chunking.service as chunk_svc
    from services.chunking.types import ChunkResult, ChunkMetadata

    stored = []

    async def fake_upsert(db, chunks_data):
        stored.extend(chunks_data)
        return []

    monkeypatch.setattr("database.repositories.upsert_chunks_batch", fake_upsert)
    db = FakeDB()
    svc = chunk_svc.ChunkingService()
    meta = ChunkMetadata(page_number=1, position=0, document_id=str(uuid.uuid4()))
    chunks = [ChunkResult(content="hello world", metadata=meta,
                          token_count=3, content_hash="h1")]
    await svc._store_chunks(db, str(uuid.uuid4()), str(uuid.uuid4()), chunks)
    assert db.commits == 0
    assert db.flushes >= 1
    assert len(stored) == 1


# ---------------- T406 ----------------

def test_t406_generation_models_accept_both_casings():
    from services.transformation import routes as r

    models = [r.NotesRequest, r.FlashcardsRequest, r.QuizRequest,
              r.ExplanationRequest, r.SummaryRequest, r.WebsiteRequest,
              r.AnswerRequest, r.MCQRequest, r.SyllabusRequest, r.PYQRequest,
              r.MappingRequest, r.AnalysisRequest]
    for model in models:
        a = model.model_validate({"projectId": "p1", "query": "q"})
        b = model.model_validate({"project_id": "p1", "query": "q"})
        assert a.project_id == "p1" and b.project_id == "p1"


def test_t406_research_alias_accepted():
    from services.transformation.routes import NotesRequest

    assert NotesRequest.model_validate(
        {"project_id": "p", "enableResearch": True}).enable_research is True


# ---------------- T407 ----------------

def test_t407_start_request_jwt_shape_and_aliases():
    from services.study.types import StartSessionRequest

    # No user_id required: identity comes from the JWT.
    a = StartSessionRequest.model_validate(
        {"project_id": "p", "session_type": "quiz"})
    assert a.user_id is None
    b = StartSessionRequest.model_validate(
        {"projectId": "p", "sessionType": "flashcards",
         "userId": "forged", "contentId": "c", "initialQuery": "q"})
    assert b.project_id == "p"
    assert b.session_type.value == "flashcards"


@pytest.mark.asyncio
async def test_t407_start_ignores_forged_user_id_and_loads_real_cards(monkeypatch):
    from database import repositories
    from services.study.service import session_manager, NotGeneratedError
    from services.study.types import StartSessionRequest

    owner, stranger, project_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    card = SimpleNamespace(id=uuid.uuid4(), front="Q?", back="A",
                           difficulty="easy")
    deck = SimpleNamespace(id=uuid.uuid4(), name="Deck", flashcards=[card])
    project = SimpleNamespace(id=project_id, user_id=owner)

    async def get_project(db, pid):
        return project

    async def get_decks(db, pid):
        return [deck]

    async def get_quizzes(db, pid):
        return []

    async def create_session(db, user_id, project_id, session_type, started_at=None,
                           session_id=None):
        return SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(repositories, "get_project_by_id", get_project)
    monkeypatch.setattr(repositories, "get_flashcard_decks_by_project", get_decks)
    monkeypatch.setattr(repositories, "get_quizzes_by_project", get_quizzes)
    monkeypatch.setattr(repositories, "create_study_session", create_session)

    req = StartSessionRequest.model_validate(
        {"project_id": str(project_id), "session_type": "flashcards",
         "user_id": str(stranger)})
    state = await session_manager.start_session(
        req, FakeDB(), user_id=str(owner))
    assert state.user_id == owner  # forged id ignored
    assert len(state.items) > 0


@pytest.mark.asyncio
async def test_t407_no_content_returns_typed_not_generated(monkeypatch):
    from database import repositories
    from services.study.service import session_manager, NotGeneratedError
    from services.study.types import StartSessionRequest

    owner, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=owner)

    async def get_project(db, pid):
        return project

    async def get_decks(db, pid):
        return []

    async def get_quizzes(db, pid):
        return []

    monkeypatch.setattr(repositories, "get_project_by_id", get_project)
    monkeypatch.setattr(repositories, "get_flashcard_decks_by_project", get_decks)
    monkeypatch.setattr(repositories, "get_quizzes_by_project", get_quizzes)

    req = StartSessionRequest.model_validate(
        {"project_id": str(project_id), "session_type": "quiz"})
    with pytest.raises(NotGeneratedError) as exc:
        await session_manager.start_session(req, FakeDB(), user_id=str(owner))
    assert exc.value.to_dict()["code"] == "not_generated"
    assert "POST /api/generation/quiz" in str(exc.value)


# ---------------- T408 ----------------

def test_t408_coerce_text_and_aliases():
    from services.study.routes import TutorMessageRequest, _coerce_text

    assert _coerce_text("  hello  ") == "hello"
    assert _coerce_text({"text": "ans"}) == "ans"
    with pytest.raises(ValueError):
        _coerce_text({"nope": 1})
    with pytest.raises(ValueError):
        _coerce_text(42)
    with pytest.raises(ValueError):
        _coerce_text("   ")

    a = TutorMessageRequest.model_validate(
        {"sessionId": "s", "message": "hi"})
    assert a.session_id == "s"
    b = TutorMessageRequest.model_validate(
        {"projectId": "p", "message": "hi"})
    assert b.project_id == "p" and b.session_id is None


# ---------------- T409 ----------------

@pytest.mark.asyncio
async def test_t409_progress_monotonic_and_stages_preserved(monkeypatch, tmp_path):
    db = await _run_txt(monkeypatch, tmp_path, "honest content " * 40)
    assert db.job.status == JobStatus.COMPLETED
    assert db.progress_seen == sorted(db.progress_seen)
    assert "_stages_done" in (db.job.output_data or {})
    assert set(db.job.output_data["_stages_done"]) >= {"validate", "extract"}


def test_t409_no_fabricated_percent_in_worker():
    src = pathlib.Path(
        "services/jobs/handlers/document_processing.py").read_text()
    assert "frac" in src  # real-count fractions still drive progress


# ---------------- T406/T407/T408 over HTTP (ASGI, no network) ----------------

def _owned_project(user_id, project_id=None):
    return SimpleNamespace(id=project_id or uuid.uuid4(), user_id=user_id,
                           name="P")


@pytest.mark.asyncio
async def test_t406_all_generation_routes_accept_both_casings(monkeypatch):
    """AC-A6: POST /api/generation/* non-422 for {project_id} and {projectId}."""
    import httpx
    from main import app
    from database.engine import get_db
    from database import repositories
    from services.auth.deps import get_current_user
    from services.transformation import engine as engine_mod

    user_id = uuid.uuid4()
    project = _owned_project(user_id)

    async def fake_get_db():
        yield SimpleNamespace()

    async def fake_user():
        return SimpleNamespace(id=user_id)

    async def fake_project(db, pid):
        return project

    def _result(mode):
        return SimpleNamespace(project_id=str(project.id),
                               mode=SimpleNamespace(value=mode),
                               content={"ok": True}, cached=False,
                               processing_time=0.01)

    async def _notes(**kw):
        return _result("notes")

    async def _cards(**kw):
        return _result("flashcards")

    async def _quiz(**kw):
        return _result("quiz")

    async def _expl(**kw):
        return _result("explanation")

    async def _summ(**kw):
        return _result("summary")

    async def _web(**kw):
        return _result("website")

    async def _ans(**kw):
        return _result("answer")

    async def _mcq(**kw):
        return _result("mcq")

    async def _syll(**kw):
        return _result("syllabus")

    async def _pyq(**kw):
        return _result("pyq")

    async def _mapping(**kw):
        return _result("mapping")

    async def _analysis(**kw):
        return _result("analysis")

    eng = engine_mod.transformation_engine
    monkeypatch.setattr(eng, "generate_notes", _notes, raising=False)
    monkeypatch.setattr(eng, "generate_flashcards", _cards, raising=False)
    monkeypatch.setattr(eng, "generate_quiz", _quiz, raising=False)
    monkeypatch.setattr(eng, "generate_explanation", _expl, raising=False)
    monkeypatch.setattr(eng, "generate_summary", _summ, raising=False)
    monkeypatch.setattr(eng, "generate_website", _web, raising=False)
    monkeypatch.setattr(eng, "generate_answer", _ans, raising=False)
    monkeypatch.setattr(eng, "generate_mcq", _mcq, raising=False)
    monkeypatch.setattr(eng, "generate_syllabus", _syll, raising=False)
    monkeypatch.setattr(eng, "generate_pyq", _pyq, raising=False)
    monkeypatch.setattr(eng, "generate_mapping", _mapping, raising=False)
    monkeypatch.setattr(eng, "generate_analysis", _analysis, raising=False)
    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            routes = ["notes", "flashcards", "quiz", "summary", "website",
                      "mcq", "answer", "explanation",
                      "syllabus", "pyq", "mapping", "analysis"]
            for route in routes:
                for body in ({"project_id": str(project.id), "query": "q"},
                             {"projectId": str(project.id), "query": "q"}):
                    resp = await client.post(f"/api/generation/{route}", json=body)
                    assert resp.status_code != 422, (route, body, resp.text)
                    assert resp.status_code == 200, (route, body, resp.text)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_t407_t408_study_start_and_tutor_over_http(monkeypatch):
    """AC-A7/A8: JWT start, real items, typed not_generated, tutor 200+404."""
    import httpx
    from main import app
    from database.engine import get_db
    from database import repositories
    from services.auth.deps import get_current_user

    owner, stranger = uuid.uuid4(), uuid.uuid4()
    project = _owned_project(owner)
    card = SimpleNamespace(id=uuid.uuid4(), front="Q?", back="A", difficulty="easy")
    deck = SimpleNamespace(id=uuid.uuid4(), name="Deck", flashcards=[card])

    async def fake_get_db():
        yield SimpleNamespace()

    async def fake_user():
        crud = SimpleNamespace(id=owner)
        return crud

    async def fake_project(db, pid):
        return project

    async def fake_decks(db, pid):
        return [deck]

    async def fake_quizzes(db, pid):
        return []

    async def fake_create_session(db, user_id, project_id, session_type,
                                  started_at=None, session_id=None):
        return SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    monkeypatch.setattr(repositories, "get_flashcard_decks_by_project", fake_decks)
    monkeypatch.setattr(repositories, "get_quizzes_by_project", fake_quizzes)
    monkeypatch.setattr(repositories, "create_study_session", fake_create_session)
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            # JWT identity; forged user_id in body is ignored.
            resp = await client.post(
                "/api/study/session/start",
                json={"projectId": str(project.id), "sessionType": "flashcards",
                      "userId": str(stranger)})
            assert resp.status_code == 200, resp.text
            assert resp.json()["totalItems"] > 0

            # No generated quiz -> typed not_generated, never empty/500.
            resp = await client.post(
                "/api/study/session/start",
                json={"project_id": str(project.id),
                      "session_type": "quiz"})
            assert resp.status_code == 404, resp.text
            assert resp.json()["detail"]["code"] == "not_generated"

            # Unknown project -> 404, not 500.
            async def no_project(db, pid):
                return None

            monkeypatch.setattr(repositories, "get_project_by_id", no_project)
            resp = await client.post(
                "/api/study/session/start",
                json={"project_id": str(uuid.uuid4()),
                      "session_type": "tutor"})
            assert resp.status_code == 404, resp.text
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_t408_tutor_project_scoped_200_and_unknown_session_404(monkeypatch):
    import httpx
    from main import app
    from database.engine import get_db
    from database import repositories
    from services.auth.deps import get_current_user
    from services.retrieval import service as retrieval_mod
    import factory as factory_mod

    owner = uuid.uuid4()
    project = _owned_project(owner)

    async def fake_get_db():
        yield SimpleNamespace()

    async def fake_user():
        return SimpleNamespace(id=owner)

    async def fake_project(db, pid):
        return project

    chunk = SimpleNamespace(chunk_id="c1", content="Mitosis stages...",
                            document_id=str(uuid.uuid4()))

    async def fake_retrieve(**kw):
        return SimpleNamespace(chunks=[chunk])

    class FakeLLM:
        async def generate(self, prompt):
            return SimpleNamespace(text="Mitosis has four main stages.")

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    monkeypatch.setattr(retrieval_mod.retrieval_service, "retrieve_with_db",
                        fake_retrieve)
    monkeypatch.setattr(factory_mod, "get_llm", lambda: FakeLLM())
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            # Project-scoped call, camelCase alias, no session needed.
            resp = await client.post(
                "/api/study/tutor/message",
                json={"projectId": str(project.id), "message": "Explain?"})
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert isinstance(body["response"], str) and body["response"]
            assert body["sources"] == ["c1"]

            # Unknown sessionId -> typed 404, never 500.
            resp = await client.post(
                "/api/study/tutor/message",
                json={"sessionId": str(uuid.uuid4()), "message": "hi"})
            assert resp.status_code == 404, resp.text
    finally:
        app.dependency_overrides.clear()
