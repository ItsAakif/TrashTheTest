"""015: S1 embedding/indexing reliability.

- HNSW index on chunks.embedding_v (migration-managed; previously a manual
  script). Measured chunks heap is ~2.6 MB locally, so a non-concurrent index
  inside the migration transaction is appropriate. For large tables use the
  CONCURRENTLY form in scripts/build_vector_index.py --prod after pausing
  writes; do not run this migration concurrently.
- Document versioning columns: version, embedding_model_version, chunk_config.
- Chunk.document_id FK gains ON DELETE CASCADE (was plain FK).

Additive only. Legacy embedding BYTEA is untouched.
"""

from alembic import op
import sqlalchemy as sa

revision = "015_s1_vector_and_versioning"
down_revision = "014_structure_metadata"
branch_labels = None
depends_on = None


def _fk_columns(fk: dict) -> list[str]:
    """Alembic returns constrained_columns as names, not dicts."""
    return [str(c) for c in (fk.get("constrained_columns") or [])]


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    dialect_name = bind.dialect.name if bind is not None else "sqlite"

    # --- documents: versioning columns (additive, nullable-safe) ---
    if "documents" in inspector.get_table_names():
        cols = {str(r["name"]) for r in inspector.get_columns("documents")}
        if "version" not in cols:
            op.add_column("documents", sa.Column("version", sa.Integer(), nullable=True))
        if "embedding_model_version" not in cols:
            op.add_column("documents", sa.Column("embedding_model_version", sa.String(128), nullable=True))
        if "chunk_config" not in cols:
            op.add_column("documents", sa.Column("chunk_config", sa.JSON(), nullable=True))
        op.execute("UPDATE documents SET version = 1 WHERE version IS NULL")

    # --- chunks FK cascade (Postgres only; SQLite tests use create_all) ---
    if dialect_name == "postgresql" and "chunks" in inspector.get_table_names():
        fks = inspector.get_foreign_keys("chunks")
        existing = next((f for f in fks if "document_id" in _fk_columns(f)), None)
        if existing is not None and existing.get("ondelete") != "CASCADE":
            name = existing["name"]
            op.drop_constraint(name, "chunks", type_="foreignkey")
            op.create_foreign_key("fk_chunks_document_id_cascade", "chunks", "documents", ["document_id"], ["id"], ondelete="CASCADE")

        # --- HNSW index (Postgres + pgvector only) ---
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_v_hnsw "
            "ON chunks USING hnsw (embedding_v vector_cosine_ops)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    dialect_name = bind.dialect.name if bind is not None else "sqlite"

    if dialect_name == "postgresql" and "chunks" in inspector.get_table_names():
        op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_v_hnsw")
        fks = inspector.get_foreign_keys("chunks")
        cascade = next((f for f in fks if f.get("ondelete") == "CASCADE" and "document_id" in _fk_columns(f)), None)
        if cascade is not None:
            op.drop_constraint(cascade["name"], "chunks", type_="foreignkey")
            op.create_foreign_key("fk_chunks_document_id", "chunks", "documents", ["document_id"], ["id"])

    if "documents" in inspector.get_table_names():
        cols = {str(r["name"]) for r in inspector.get_columns("documents")}
        for name in ("chunk_config", "embedding_model_version", "version"):
            if name in cols:
                op.drop_column("documents", name)