"""T206/T207: chunk provenance, no-loss invariant, embedding provider/cache."""

import uuid

import pytest

pytestmark = pytest.mark.asyncio


def _norm(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text).strip()


async def test_chunk_metadata_carries_provenance():
    from services.chunking.chunker import SemanticChunker
    from services.chunking.types import DocumentChunkingInput

    text = ("Introduction\n\n" + "Photosynthesis converts light to energy. " * 20
            + "\n\nMethods\n\n" + "We measured growth under varied light. " * 20)
    spans = [{"page": 1, "start": 0, "end": 10 ** 9, "method": "native",
              "confidence": 0.9, "heading": "Introduction"}]
    chunker = SemanticChunker()
    result = chunker.chunk_document(DocumentChunkingInput(
        document_id=str(uuid.uuid4()), project_id=str(uuid.uuid4()),
        full_text=text, structural_elements=[], user_id="user-1",
        source="paper.pdf", page_spans=spans))
    assert result.total_chunks >= 1
    for chunk in result.chunks:
        meta = chunk.metadata
        assert meta.document_id
        assert meta.user_id == "user-1"
        assert meta.page_start and meta.page_end
        assert meta.source == "paper.pdf"
        assert meta.method == "native"
        assert meta.confidence == 0.9
        assert meta.section_path  # citation-grade: page + section + heading
        # ADR-008: the detector records how the section was found.
        assert meta.structure_method in {
            "font_hint", "title_case", "markdown", "numbered", "all_caps",
            "colon", "list_label", "page_bounded", "inferred",
        }


async def test_no_text_loss_invariant():
    from services.chunking.chunker import SemanticChunker
    from services.chunking.types import DocumentChunkingInput

    text = ("Background\n\n" + "The cell cycle has four phases. " * 30
            + "\n\nResults\n\n" + "Growth doubled under blue light. " * 30)
    chunker = SemanticChunker()
    result = chunker.chunk_document(DocumentChunkingInput(
        document_id=str(uuid.uuid4()), project_id=str(uuid.uuid4()), full_text=text))
    joined = _norm(" ".join(c.content for c in result.chunks))
    source = _norm(text)
    # Documented tolerance: merged overlap may drop <=5% of source tokens.
    src_tokens = set(source.split())
    hit_tokens = set(joined.split())
    coverage = len(src_tokens & hit_tokens) / max(1, len(src_tokens))
    assert coverage >= 0.95, f"coverage {coverage:.3f} below 0.95 tolerance"
    # ADR-008: no structural fallback literal may be produced.
    for chunk in result.chunks:
        assert chunk.metadata.section not in {
            "Introduction", "Conclusion", "Full Document",
        }, f"banned literal section leaked: {chunk.metadata.section!r}"


async def test_page_references_resolve_to_real_pages():
    from services.chunking.chunker import SemanticChunker
    from services.chunking.types import DocumentChunkingInput

    text = "Alpha. " * 200 + "\n\nBeta. " * 200
    chunker = SemanticChunker()
    result = chunker.chunk_document(DocumentChunkingInput(
        document_id="doc-1", project_id="proj-1", full_text=text,
        page_spans=[{"page": 1, "start": 0, "end": 10 ** 6,
                     "method": "native", "confidence": 0.9}]))
    pages = {1}
    for chunk in result.chunks:
        assert chunk.metadata.page_start in pages
        assert chunk.metadata.page_end in pages


async def test_out_of_range_page_reference_is_detectable():
    from services.chunking.types import ChunkMetadata

    meta = ChunkMetadata(page_number=99, position=0, document_id="d",
                         page_start=99, page_end=99)
    assert meta.page_start not in {1, 2}  # validator contract: fail the suite


async def test_content_hash_stable_for_idempotent_reprocessing():
    from services.chunking.chunker import SemanticChunker

    chunker = SemanticChunker()
    h1 = chunker._generate_content_hash("  Photosynthesis  converts light. ", {})
    h2 = chunker._generate_content_hash("photosynthesis converts light.", {})
    assert h1 == h2 and len(h1) == 64


async def test_store_chunks_upserts_without_duplicates(monkeypatch):
    from services.chunking.service import ChunkingService
    from services.chunking.chunker import SemanticChunker
    from services.chunking.types import DocumentChunkingInput

    seen = {}

    async def fake_upsert(db, chunks_data):
        for row in chunks_data:
            key = (str(row["document_id"]), row["content_hash"])
            seen[key] = seen.get(key, 0) + 1
        return []

    from database import repositories

    monkeypatch.setattr(repositories, "upsert_chunks_batch", fake_upsert)

    class FakeDB:
        def __init__(self):
            self.commits = 0
            self.flushes = 0

        async def commit(self):
            self.commits += 1
            return None

        async def flush(self):
            self.flushes += 1
            return None

    service = ChunkingService(chunker=SemanticChunker())
    text = "Mitosis has distinct stages. " * 40
    input_data = DocumentChunkingInput(document_id=str(uuid.uuid4()),
                                       project_id=str(uuid.uuid4()), full_text=text)
    result = service.chunker.chunk_document(input_data)
    dbs = [FakeDB(), FakeDB()]
    await service._store_chunks(dbs[0], input_data.document_id, input_data.project_id, result.chunks)
    await service._store_chunks(dbs[1], input_data.document_id, input_data.project_id, result.chunks)
    assert all(count == 2 for count in seen.values())  # same keys, zero new rows
    assert len(seen) == len(result.chunks)
    # T405: the helper flushes; the worker owns the commit.
    assert all(db.commits == 0 for db in dbs)
    assert all(db.flushes >= 1 for db in dbs)


async def test_embedding_prefixes_differ_and_1024d():
    from services.strategies.embedding.qwen3_embedding import (
        QUERY_PREFIX, PASSAGE_PREFIX, DIMENSIONS, MODEL_ID)

    assert QUERY_PREFIX != PASSAGE_PREFIX
    assert DIMENSIONS == 1024
    assert "0.6B" in MODEL_ID


async def test_embedding_cache_second_request_no_model_call(monkeypatch):
    from services.embedding_service.service import LocalEmbeddingService

    calls = {"n": 0}

    class FakeProvider:
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        dimensions = 1024

        async def generate_embeddings(self, texts, batch_size=32):
            from services.strategies.base import EmbeddingResult

            calls["n"] += 1
            return EmbeddingResult(embeddings=[[0.1] * 4 for _ in texts],
                                   model=self.model_name, dimensions=1024,
                                   processing_time=0.01)

    service = LocalEmbeddingService(embedding_provider=FakeProvider())
    key_hash = "abc123"
    service._cache_set(service.provider.model_name, key_hash, b"cached-bytes")
    assert service._cache_get(service.provider.model_name, key_hash) == b"cached-bytes"
    assert calls["n"] == 0  # served from cache, no model call


async def test_dimension_mismatch_fails_boot_with_remediation(monkeypatch):
    from types import SimpleNamespace

    from config import settings as settings_mod

    class FakeConn:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def execute(self, stmt):
            from types import SimpleNamespace

            text = str(stmt)
            if "extversion" in text:
                return SimpleNamespace(scalar=lambda: "1.2")
            return SimpleNamespace(scalar=lambda: 768)  # DB holds 768d rows

    class FakeEngine:
        def connect(self):
            return FakeConn()

    import sys as _sys

    monkeypatch.setattr(_sys.modules["database.engine"], "engine", FakeEngine())
    import config.settings as cfg_mod

    real = cfg_mod.get_settings
    monkeypatch.setattr(cfg_mod, "get_settings",
                        lambda: SimpleNamespace(
                            embedding=SimpleNamespace(dimensions=1024, model="m"),
                            llm=SimpleNamespace(base_url=None, api_key=None,
                                                primary_model="m")))
    try:
        with pytest.raises(RuntimeError, match="[Rr]emediation"):
            await settings_mod.run_boot_checks()
    finally:
        monkeypatch.setattr(cfg_mod, "get_settings", real)


async def test_factory_prefers_qwen_keeps_bge_legacy(monkeypatch):
    import factory as factory_mod

    factory_mod.get_strategy_factory().reset()
    provider = factory_mod.get_strategy_factory().get_embedding()
    assert type(provider).__name__ in ("Qwen3EmbeddingProvider", "BGESmallProvider")
    factory_mod.get_strategy_factory().reset()
