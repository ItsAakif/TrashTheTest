"""S1: embedding write-path, regeneration, failure truthfulness, lifecycle."""

import uuid
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.asyncio


class FakeResult:
    def __init__(self, rows=None, rowcount=0, first=None):
        self._rows = rows or []
        self.rowcount = rowcount
        self._first = first

    def scalars(self):
        rows = self._rows

        class _S:
            def all(self):
                return rows

        return _S()

    def first(self):
        return self._first


class FakeDB:
    """Records SQL; no real database. Dialect selectable."""

    def __init__(self, dialect="postgresql"):
        self.calls = []
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect))
        self.deleted = []
        self.flushed = 0
        self.committed = 0

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        return FakeResult()

    async def flush(self):
        self.flushed += 1

    async def commit(self):
        self.committed += 1

    async def delete(self, obj):
        self.deleted.append(obj)


class FakeProvider:
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    dimensions = 1024

    def __init__(self, fail_times=0, always_fail=False):
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.calls = 0

    async def generate_embeddings(self, texts, batch_size=32):
        self.calls += 1
        if self.always_fail or self.calls <= self.fail_times:
            raise RuntimeError("provider unavailable")
        return SimpleNamespace(embeddings=[[0.01 * (i + 1)] * 1024 for i in range(len(texts))])


def _chunk(i=0):
    return SimpleNamespace(
        id=uuid.uuid4(),
        content=f"chunk text {i}",
        chunk_metadata={"content_hash": f"hash-{i}"},
    )


def _service(provider):
    from services.embedding_service.service import LocalEmbeddingService
    from services.embedding_service.types import EmbeddingConfig

    return LocalEmbeddingService(
        embedding_provider=provider,
        config=EmbeddingConfig(batch_size=32, max_retries=2, retry_delay_seconds=0, cache_enabled=False),
    )


async def test_postgres_write_updates_both_columns():
    svc = _service(FakeProvider())
    db = FakeDB("postgresql")
    cid = str(uuid.uuid4())
    blob = svc._pack_vector([0.5] * 1024)

    await svc._update_chunks_with_embeddings(db, {cid: blob})

    sql = " | ".join(s for s, _ in db.calls)
    assert "UPDATE chunks SET embedding" in sql          # legacy BYTEA
    assert "embedding_v = CAST" in sql                   # vector arm column
    text_params = [p for s, p in db.calls if "embedding_v" in s]
    assert text_params and text_params[0]["cid"] == cid
    assert text_params[0]["vec"].startswith("[0.5,0.5")


async def test_non_postgres_write_skips_vector_column():
    svc = _service(FakeProvider())
    db = FakeDB("sqlite")
    await svc._update_chunks_with_embeddings(db, {str(uuid.uuid4()): svc._pack_vector([0.1] * 1024)})
    sql = " | ".join(s for s, _ in db.calls)
    assert "embedding_v" not in sql


async def test_regeneration_actually_embeds(monkeypatch):
    from database import repositories

    provider = FakeProvider()
    svc = _service(provider)
    chunks = [_chunk(0), _chunk(1)]

    async def fake_chunks(db, doc_id):
        return chunks

    monkeypatch.setattr(repositories, "get_chunks_by_document", fake_chunks)
    db = FakeDB("postgresql")

    result = await svc.generate_embeddings(
        uuid.uuid4(), uuid.uuid4(), db, force_regenerate=True
    )

    assert provider.calls >= 1
    assert result.embedded_chunks == 2
    assert result.is_complete is True
    assert any("embedding_v" in s for s, _ in db.calls)


async def test_transient_failure_retries_and_recovers(monkeypatch):
    from database import repositories

    provider = FakeProvider(fail_times=1)
    svc = _service(provider)

    async def fake_chunks(db, doc_id):
        return [_chunk(0)]

    monkeypatch.setattr(repositories, "get_chunks_by_document", fake_chunks)
    result = await svc.generate_embeddings(uuid.uuid4(), uuid.uuid4(), FakeDB())

    assert provider.calls == 2
    assert result.is_complete is True
    assert result.failed_batches == 0


async def test_exhausted_batch_is_not_reported_complete(monkeypatch):
    from database import repositories

    svc = _service(FakeProvider(always_fail=True))

    async def fake_chunks(db, doc_id):
        return [_chunk(0)]

    monkeypatch.setattr(repositories, "get_chunks_by_document", fake_chunks)
    result = await svc.generate_embeddings(uuid.uuid4(), uuid.uuid4(), FakeDB())

    assert result.is_complete is False
    assert result.failed_batches == 1
    assert result.embedded_chunks == 0
    assert result.errors


async def test_prune_pages_beyond_count(monkeypatch):
    from database import repositories

    pages = [SimpleNamespace(page_number=n) for n in range(1, 6)]

    async def fake_pages(db, doc_id):
        return pages

    monkeypatch.setattr(repositories, "get_pages", fake_pages)
    db = FakeDB()
    removed = await repositories.prune_pages_beyond_count(db, uuid.uuid4(), 3)

    assert removed == 2
    assert [p.page_number for p in db.deleted] == [4, 5]


async def test_bump_document_version(monkeypatch):
    from database import repositories

    async def fake_columns(db, table):
        return {"version", "embedding_model_version", "chunk_config"}

    monkeypatch.setattr(repositories, "_existing_columns", fake_columns)

    class VersionDB(FakeDB):
        async def execute(self, stmt, params=None):
            self.calls.append((str(stmt), params))
            if "SELECT version" in str(stmt):
                return FakeResult(first=(2,))
            return FakeResult()

    db = VersionDB()
    version = await repositories.bump_document_version(
        db, uuid.uuid4(), embedding_model_version="Qwen/Qwen3-Embedding-0.6B",
        chunk_config={"max_chunk_size": 1000},
    )

    assert version == 3
    assert any("UPDATE documents SET version" in s for s, _ in db.calls)


async def test_migration_015_manages_hnsw_and_cascade():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "migrations" / "versions" / "015_s1_vector_and_versioning.py"
    text = path.read_text(encoding="utf-8")

    assert "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_v_hnsw" in text
    assert "USING hnsw (embedding_v vector_cosine_ops)" in text
    assert "DROP INDEX IF EXISTS ix_chunks_embedding_v_hnsw" in text
    assert 'ondelete="CASCADE"' in text
    assert 'revision = "015_s1_vector_and_versioning"' in text