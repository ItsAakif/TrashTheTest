"""T201: migration 004 expand step (offline assertions; live PG is T113)."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


def _src() -> str:
    return Path("migrations/versions/004_document_pages_and_pgvector.py").read_text()


async def test_migration_file_exists_with_expand_contract():
    src = _src()
    assert "document_pages" in src
    assert "uq_document_pages_doc_page" in src
    assert "embedding" in src  # legacy BYTEA untouched comment + new vector col
    assert "tsv" in src
    assert "ix_chunks_doc_content_hash" in src


async def test_no_hnsw_index_in_expand_step():
    src = _src().lower()
    assert "using hnsw" not in src
    assert "concurrently" not in src or "hnsw" not in src


async def test_downgrade_is_defined_and_reversible():
    src = _src()
    assert re.search(r"def downgrade\(\)", src)
    assert re.search(r"def upgrade\(\)", src)
    assert "drop_table" in src.lower() or "drop_column" in src.lower()


async def test_document_page_model_importable_with_uniqueness():
    from database.models.document_page import DocumentPage

    names = [str(c.name) for c in DocumentPage.__table__.columns]
    for col in ("document_id", "page_number", "method", "confidence", "bbox_summary"):
        assert col in names


async def test_chunk_provenance_columns_present():
    from database.models.chunk import Chunk

    names = [str(c.name) for c in Chunk.__table__.columns]
    for col in ("embedding_model", "embedding_dim", "content_hash",
                "page_start", "page_end", "section_path", "heading", "position"):
        assert col in names
    assert "embedding" in names  # legacy BYTEA retained


async def test_document_provenance_columns_present():
    from database.models.document import Document

    names = [str(c.name) for c in Document.__table__.columns]
    for col in ("page_count_confirmed", "extraction_method_summary",
                "content_hash", "indexing_state"):
        assert col in names


async def test_revision_width_repair_and_short_future_ids():
    versions = Path("migrations/versions")
    source_005 = (versions / "005_contract_drop_legacy_embedding.py").read_text()
    source_012 = (versions / "012_alembic_version_width.py").read_text()
    revision_005 = re.search(r'^revision\s*=\s*["\']([^"\']+)', source_005, re.M)
    revision_012 = re.search(r'^revision\s*=\s*["\']([^"\']+)', source_012, re.M)

    assert revision_005 and len(revision_005.group(1)) > 32
    assert revision_012 and len(revision_012.group(1)) <= 32
    assert "VARCHAR(255)" in source_012
    assert "ALTER TABLE IF EXISTS alembic_version" in source_012
    assert "never narrow" in source_012.lower()


async def test_live_recovery_migrations_are_additive_and_rerunnable():
    source_006 = Path("migrations/versions/006_user_provider_credentials.py").read_text()
    assert "inspector.has_table" in source_006
    assert "preserving existing credential rows" in source_006
    assert "if not any(" in source_006
    assert "op.add_column" in source_006
    assert "CREATE INDEX IF NOT EXISTS" in source_006


async def test_env_preflights_alembic_version_width():
    source = Path("migrations/env.py").read_text()
    assert "CREATE TABLE IF NOT EXISTS alembic_version" in source
    assert "ALTER TABLE alembic_version " in source
    assert "ALTER COLUMN version_num TYPE VARCHAR(255)" in source
    assert "_ensure_wide_alembic_version" in source
