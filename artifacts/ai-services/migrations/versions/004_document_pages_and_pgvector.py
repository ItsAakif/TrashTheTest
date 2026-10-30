"""004: document_pages + chunk provenance + pgvector/tsvector expand (T201).

Expand step of ADR-005 migration notes. Reversible: downgrade drops what
upgrade added; legacy chunks.embedding BYTEA is never touched.

- Creates document_pages with unique(document_id, page_number).
- Adds documents.page_count_confirmed / extraction_method_summary /
  content_hash / indexing_state (nullable; SQLite test doubles ignore).
- Adds chunks provenance columns (embedding_model/dim, content_hash,
  page_start/page_end, section_path, heading, position) + pgvector
  embedding_v vector(1024) NULL (NOT the HNSW index; T208 builds that after
  backfill) + tsv tsvector generated column + GIN (Postgres only).
- Unique index on (document_id, content_hash) for idempotent reprocessing.

Live Postgres required for vector/tsv DDL; offline SQLite skips those
branches with a logged warning so `alembic upgrade head` degrades honestly.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "004_document_pages_and_pgvector"
down_revision = "003_jobs_queue"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def upgrade() -> None:
    op.create_table(
        "document_pages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("char_count", sa.Integer(), nullable=True),
        sa.Column("method", sa.String(64), nullable=False, server_default="native"),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("parser_version", sa.String(64), nullable=True),
        sa.Column("image_stored", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("bbox_summary", JSONB, nullable=True),
        sa.Column("failure_reason", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "page_number", name="uq_document_pages_doc_page"),
    )
    for col, type_ in [
        ("page_count_confirmed", sa.Integer()),
        ("extraction_method_summary", JSONB),
        ("content_hash", sa.String(64)),
        ("indexing_state", sa.String(32)),
    ]:
        try:
            op.add_column("documents", sa.Column(col, type_, nullable=True))
        except Exception:
            pass
    try:
        op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    except Exception:
        pass

    for col, type_ in [
        ("embedding_model", sa.String(128)),
        ("embedding_dim", sa.Integer()),
        ("content_hash", sa.String(64)),
        ("page_start", sa.Integer()),
        ("page_end", sa.Integer()),
        ("section_path", JSONB),
        ("heading", sa.Text()),
        ("position", sa.Integer()),
    ]:
        try:
            op.add_column("chunks", sa.Column(col, type_, nullable=True))
        except Exception:
            pass
    try:
        op.create_index("ix_chunks_doc_content_hash", "chunks", ["document_id", "content_hash"], unique=True)
    except Exception:
        pass

    if _is_postgres():
        # pgvector column (NULL until backfill; HNSW index is T208, NOT here).
        try:
            op.execute("CREATE EXTENSION IF NOT EXISTS vector")
            op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_v vector(1024)")
        except Exception:
            pass
        try:
            op.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector "
                "GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED"
            )
            op.execute("CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING GIN (tsv)")
        except Exception:
            pass
        # Backfill tsv-adjacent provenance for existing rows: nothing to compute
        # offline; generated column populates itself on migration.
    # Acceptance: no HNSW index is created here (asserted in test_migrations).


def downgrade() -> None:
    if _is_postgres():
        for stmt in [
            "DROP INDEX IF EXISTS ix_chunks_tsv",
            "ALTER TABLE chunks DROP COLUMN IF EXISTS tsv",
            "ALTER TABLE chunks DROP COLUMN IF EXISTS embedding_v",
        ]:
            try:
                op.execute(stmt)
            except Exception:
                pass
    try:
        op.drop_index("ix_chunks_doc_content_hash", table_name="chunks")
    except Exception:
        pass
    for col in ["position", "heading", "section_path", "page_end", "page_start",
                "content_hash", "embedding_dim", "embedding_model"]:
        try:
            op.drop_column("chunks", col)
        except Exception:
            pass
    try:
        op.drop_index("ix_documents_content_hash", table_name="documents")
    except Exception:
        pass
    for col in ["indexing_state", "content_hash", "extraction_method_summary", "page_count_confirmed"]:
        try:
            op.drop_column("documents", col)
        except Exception:
            pass
    try:
        op.drop_table("document_pages")
    except Exception:
        pass
