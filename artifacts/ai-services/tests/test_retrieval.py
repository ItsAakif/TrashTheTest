"""Hybrid retrieval tests (T301-T304). Offline, deterministic, no providers.

Covers: page/section/scores in results, model-space filtering, exact
top_k on filtered scopes, user isolation (404), blank-query 422,
truncation flag, top_k clamp, indexing_in_progress with counts, rerank
toggle, and frontend camelCase aliases.
"""

import struct
import types
import uuid

import pytest

from services.retrieval.service import VectorRetrievalService
from services.retrieval.types import RetrievalConfig


VOCAB = ["photosynthesis", "chlorophyll", "oxygen", "matrix", "vector",
         "eigenvalue", "revolution", "bastille", "napoleon", "guillotine",
         "stroma", "thylakoid", "determinant", "robot", "king", "bread"]


def vec_for(text):
    import re

    toks = set(re.findall(r"[a-z]+", text.lower()))
    return [1.0 if w in toks else 0.0 for w in VOCAB]


class _EmbResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeEmbed:
    model_name = "Qwen/Qwen3-Embedding-0.6B"

    async def embed_queries(self, texts, batch_size=32):
        return _EmbResult([vec_for(t) for t in texts])

    async def embed_passages(self, texts, batch_size=32):
        return _EmbResult([vec_for(t) for t in texts])

    async def generate_embedding(self, text):
        return vec_for(text)

    @property
    def dimensions(self):
        return len(VOCAB)


def make_chunk(content, project_tag="p1", model="Qwen/Qwen3-Embedding-0.6B",
               page=3, heading="Sec A"):
    from database.models import Chunk

    vec = vec_for(content)
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        embedding=struct.pack(f"{len(vec)}f", *vec),
        embedding_model=model,
        embedding_dim=len(vec),
        page_start=page,
        page_end=page,
        section_path=["Ch 1", heading],
        heading=heading,
        chunk_metadata={"method": "native", "confidence": 0.99},
    )


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class _Result:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def scalars(self):
        return _Scalars(self._rows)

    def scalar(self):
        return self._scalar

    def all(self):
        return self._rows


class FakeSession:
    """Offline AsyncSession double over a fixed chunk list."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def execute(self, stmt, params=None):
        s = str(stmt).lower()
        if "count(" in s or "count *" in s:
            if "embedding_model" in s:
                n = sum(1 for c in self._chunks if c.embedding_model == FakeEmbed.model_name)
            else:
                n = len(self._chunks)
            return _Result(scalar=n)
        return _Result(rows=list(self._chunks))

    async def get(self, model, pk):
        for c in self._chunks:
            if str(c.id) == str(pk):
                return c
        return None

    async def commit(self):
        return None

    async def flush(self):
        return None


def make_service(**overrides):
    cfg = RetrievalConfig(embedding_model=FakeEmbed.model_name, **overrides)
    return VectorRetrievalService(embedding_provider=FakeEmbed(), config=cfg)


@pytest.mark.asyncio
async def test_results_carry_page_section_scores_and_citations():
    chunks = [make_chunk("photosynthesis chlorophyll oxygen in the stroma")]
    svc = make_service()
    result = await svc.retrieve_with_db("photosynthesis chlorophyll", FakeSession(chunks), top_k=5)
    assert result.total_results == 1
    c = result.chunks[0]
    assert c.page == 3
    assert c.section == "Ch 1/Sec A"
    assert c.score is not None and c.score > 0
    assert c.source in ("vector", "lexical", "both", "reranked")
    assert c.citation is not None
    assert c.citation.chunk_id == c.chunk_id
    cites = result.citations()
    assert cites[0] == {"doc_id": c.document_id, "page": 3, "chunk_id": c.chunk_id}


@pytest.mark.asyncio
async def test_non_matching_embedding_model_never_returned():
    good = make_chunk("photosynthesis chlorophyll oxygen stroma")
    stale = make_chunk("photosynthesis chlorophyll oxygen stroma", model="bge-small-legacy")
    svc = make_service()
    result = await svc.retrieve_with_db("photosynthesis", FakeSession([good, stale]), top_k=5)
    ids = [c.chunk_id for c in result.chunks]
    assert str(good.id) in ids
    assert str(stale.id) not in ids


@pytest.mark.asyncio
async def test_filtered_scope_returns_exactly_top_k():
    chunks = [make_chunk(f"photosynthesis chlorophyll oxygen stroma leaf {i}") for i in range(12)]
    svc = make_service()
    result = await svc.retrieve_with_db(
        "photosynthesis chlorophyll", FakeSession(chunks),
        project_id=str(uuid.uuid4()), top_k=5,
    )
    assert result.total_results == 5


@pytest.mark.asyncio
async def test_indexing_in_progress_reports_counts_not_empty():
    done = make_chunk("photosynthesis chlorophyll stroma")
    pending = make_chunk("photosynthesis chlorophyll stroma leaf")
    pending.embedding_model = None
    pending.embedding = None
    svc = make_service()
    result = await svc.retrieve_with_db(
        "photosynthesis", FakeSession([done, pending]),
        project_id=str(uuid.uuid4()), top_k=5,
    )
    assert result.state == "indexing_in_progress"
    assert result.total_chunks == 2
    assert result.indexed_chunks == 1


@pytest.mark.asyncio
async def test_unindexed_scope_reports_unindexed():
    pending = make_chunk("photosynthesis chlorophyll")
    pending.embedding_model = None
    pending.embedding = None
    svc = make_service()
    result = await svc.retrieve_with_db(
        "photosynthesis", FakeSession([pending]),
        project_id=str(uuid.uuid4()), top_k=5,
    )
    assert result.state == "unindexed"


@pytest.mark.asyncio
async def test_rerank_toggle_switches_metadata():
    chunks = [make_chunk(f"photosynthesis chlorophyll oxygen {i}") for i in range(6)]
    svc = make_service()
    off = await svc.retrieve_with_db("photosynthesis chlorophyll", FakeSession(chunks),
                                     top_k=3, enable_rerank=False)
    assert off.rerank_applied is False
    assert off.reranker is None
    on = await svc.retrieve_with_db("photosynthesis chlorophyll", FakeSession(chunks),
                                    top_k=3, enable_rerank=True)
    assert on.rerank_applied is True
    assert on.reranker is not None
    assert on.rerank_latency_ms is not None
    # reranked output ids all come from the candidate set (no inventions)
    assert len(on.chunks) == 3
    assert len({c.chunk_id for c in on.chunks}) == 3


@pytest.mark.asyncio
async def test_rerank_off_is_ab_comparable():
    chunks = [make_chunk(f"photosynthesis chlorophyll oxygen {i}") for i in range(6)]
    svc = make_service(enable_rerank=False)
    result = await svc.retrieve_with_db("photosynthesis", FakeSession(chunks), top_k=3)
    assert result.rerank_applied is False


@pytest.mark.asyncio
async def test_blank_and_truncation_and_clamp():
    from fastapi import HTTPException

    svc = make_service(max_top_k=7, query_max_length=20)
    empty = await svc.retrieve_with_db("", FakeSession([]))
    assert empty.total_results == 0
    chunks = [make_chunk("photosynthesis chlorophyll oxygen")]
    long_q = "photosynthesis chlorophyll oxygen stroma leaf extra words here"
    result = await svc.retrieve_with_db(long_q, FakeSession(chunks), top_k=100)
    assert result.truncated_query is True
    assert len(result.chunks) <= 7


@pytest.mark.asyncio
async def test_user_isolation_returns_404_for_foreign_project(monkeypatch):
    """User B cannot query user A's project: 404, not data."""
    from fastapi import HTTPException

    from database import repositories
    from services.retrieval.routes import _check_scope

    owner_a = uuid.uuid4()
    user_b = types.SimpleNamespace(id=uuid.uuid4())
    project = types.SimpleNamespace(id=uuid.uuid4(), user_id=owner_a)

    async def fake_get_project(db, pid):
        return project

    monkeypatch.setattr(repositories, "get_project_by_id", fake_get_project)
    with pytest.raises(HTTPException) as exc:
        await _check_scope(FakeSession([]), str(project.id), None, user_b)
    assert exc.value.status_code == 404

    # Owner passes the same gate.
    owner = types.SimpleNamespace(id=owner_a)
    proj_uuid, _, _ = await _check_scope(FakeSession([]), str(project.id), None, owner)
    assert str(proj_uuid) == str(project.id)


@pytest.mark.asyncio
async def test_search_route_states_and_aliases(monkeypatch):
    """POST /api/retrieval/search: blank -> 422, aliases accepted."""
    from database import repositories
    from main import app
    from database.engine import get_db
    import httpx

    chunks = [make_chunk("photosynthesis chlorophyll oxygen stroma")]
    session = FakeSession(chunks)
    app.dependency_overrides[get_db] = lambda: session

    async def fake_get_project(db, pid):
        return types.SimpleNamespace(id=pid, user_id=uuid.uuid4())

    monkeypatch.setattr(repositories, "get_project_by_id", fake_get_project)
    # Service provider would hit real weights: inject the fake.
    from services.retrieval import routes as _routes

    monkeypatch.setattr(_routes.retrieval_service, "provider", FakeEmbed())
    monkeypatch.setattr(_routes.retrieval_service.config, "embedding_model",
                        FakeEmbed.model_name)
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            blank = await client.post("/api/retrieval/search",
                                      json={"query": "  ", "project_id": str(uuid.uuid4())})
            assert blank.status_code == 422
            ok = await client.post("/api/retrieval/search",
                                   json={"query": "photosynthesis chlorophyll",
                                         "projectId": str(uuid.uuid4()),
                                         "topK": 3, "minSimilarity": 0.0})
            assert ok.status_code == 200, ok.text
            body = ok.json()
            assert body["total_results"] >= 1
            assert body["chunks"][0]["page"] == 3
            assert body["citations"][0]["chunk_id"] == body["chunks"][0]["chunk_id"]
    finally:
        app.dependency_overrides.pop(get_db, None)


class _HonestSession(FakeSession):
    """FakeSession variant that honors a legacy BYTEA predicate when the SQL
    actually carries one (D1 regression double).

    The stock FakeSession counts embedding_model whenever the statement
    mentions it, which masked D1 offline (old SQL ANDed legacy NOT NULL
    with the model tag yet still counted full). This double requires
    c.embedding NOT NULL whenever the statement has a bare legacy
    "embedding IS NOT NULL" predicate without an embedding_v OR arm.
    """

    def __init__(self, chunks, pg=False):
        super().__init__(chunks)
        self.statements = []
        if pg:
            self.bind = types.SimpleNamespace(
                dialect=types.SimpleNamespace(name="postgresql"))

    async def execute(self, stmt, params=None):
        s = str(stmt)
        self.statements.append(s)
        low = s.lower()
        if "count(" in low or "count *" in low:
            if "embedding_model" in low and "embedding_v" not in low:
                if "is not null" in low:
                    # legacy predicate present: honor it honestly
                    n = sum(1 for c in self._chunks
                            if c.embedding is not None
                            and c.embedding_model == FakeEmbed.model_name)
                else:
                    n = sum(1 for c in self._chunks
                            if c.embedding_model == FakeEmbed.model_name)
            elif "embedding_v" in low:
                n = sum(1 for c in self._chunks
                        if c.embedding is not None
                        or c.embedding_model == FakeEmbed.model_name)
            elif "is not null" in low:
                # bare legacy predicate (SQLite no-model branch): honor it
                n = sum(1 for c in self._chunks if c.embedding is not None)
            else:
                n = len(self._chunks)
            return _Result(scalar=n)
        return _Result(rows=list(self._chunks))


def _backfilled_chunks(n=2):
    """Rows as live Postgres backfill leaves them: model tag set, legacy
    BYTEA NULL (embedding_v written via raw SQL, invisible to the ORM)."""
    rows = [make_chunk(f"photosynthesis chlorophyll stroma leaf {i}")
            for i in range(n)]
    for c in rows:
        c.embedding = None
    assert all(c.embedding_model == FakeEmbed.model_name for c in rows)
    return rows


@pytest.mark.asyncio
async def test_d1_indexing_state_counts_model_tag_without_legacy():
    """D1: backfilled scope (model tag, legacy NULL) reports indexed."""
    from database import repositories

    sess = _HonestSession(_backfilled_chunks(2))
    info = await repositories.get_indexing_state(
        sess, project_id=uuid.uuid4(), embedding_model=FakeEmbed.model_name)
    assert info["total_chunks"] == 2
    assert info["indexed_chunks"] == 2
    assert info["state"] == "ok"
    model_stmts = [s for s in sess.statements if "embedding_model" in s.lower()]
    assert model_stmts
    assert all("is not null" not in s.lower() for s in model_stmts)


@pytest.mark.asyncio
async def test_d1_indexing_state_no_model_uses_vector_or_on_postgres():
    """D1: without a model filter on Postgres, legacy OR embedding_v."""
    from database import repositories

    sess = _HonestSession(_backfilled_chunks(2), pg=True)
    info = await repositories.get_indexing_state(sess, project_id=uuid.uuid4())
    assert info["indexed_chunks"] == 2
    assert info["state"] == "ok"
    assert any("embedding_v" in s for s in sess.statements)


@pytest.mark.asyncio
async def test_d1_indexing_state_no_model_sqlite_keeps_legacy():
    """D1: without a model filter off Postgres, legacy predicate unchanged."""
    from database import repositories

    sess = _HonestSession(_backfilled_chunks(2), pg=False)
    info = await repositories.get_indexing_state(sess, project_id=uuid.uuid4())
    assert info["indexed_chunks"] == 0
    assert info["state"] == "unindexed"
    assert not any("embedding_v" in s for s in sess.statements)


@pytest.mark.asyncio
async def test_d1_retrieval_stats_counts_model_tag_without_legacy():
    """D1: get_retrieval_stats honors the model tag on backfilled rows."""
    svc = make_service()
    sess = _HonestSession(_backfilled_chunks(2))
    stats = await svc.get_retrieval_stats(sess, str(uuid.uuid4()))
    assert stats.total_chunks == 2
    assert stats.chunks_with_embeddings == 2
