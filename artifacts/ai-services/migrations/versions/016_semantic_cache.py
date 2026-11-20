"""016: S3 semantic cache storage.

Postgres holds the semantic index (pgvector HNSW) and is the source of
truth; Redis is an optional exact-key hot path. Additive only.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "016_semantic_cache"
down_revision = "015_s1_vector_and_versioning"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name if bind is not None else "sqlite"
    inspector = sa.inspect(bind)
    if "semantic_cache_entries" not in inspector.get_table_names():
        op.create_table(
            "semantic_cache_entries",
            sa.Column("id", postgresql.UUID(as_uuid=True) if dialect == "postgresql" else sa.String(36), primary_key=True),
            sa.Column("identity_key", sa.String(64), nullable=False),
            sa.Column("layer", sa.String(16), nullable=False),
            sa.Column("query_hash", sa.String(64), nullable=False),
            sa.Column("query_text", sa.Text(), nullable=True),
            sa.Column("response", sa.JSON(), nullable=False),
            sa.Column("citations", sa.JSON(), nullable=True),
            sa.Column("document_ids", sa.JSON(), nullable=True),
            sa.Column("doc_versions", sa.JSON(), nullable=True),
            sa.Column("project_id", postgresql.UUID(as_uuid=True) if dialect == "postgresql" else sa.String(36), nullable=True),
            sa.Column("user_id", postgresql.UUID(as_uuid=True) if dialect == "postgresql" else sa.String(36), nullable=True),
            sa.Column("embedding_model", sa.String(128), nullable=True),
            sa.Column("llm_model", sa.String(128), nullable=True),
            sa.Column("prompt_version", sa.String(64), nullable=True),
            sa.Column("retrieval_config_hash", sa.String(64), nullable=True),
            sa.Column("response_format", sa.String(32), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("hits", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_hit_at", sa.DateTime(), nullable=True),
            sa.UniqueConstraint("identity_key", "layer", "query_hash",
                                name="uq_semantic_cache_identity_layer_query"),
        )
        op.create_index("ix_semantic_cache_identity_layer",
                        "semantic_cache_entries", ["identity_key", "layer", "expires_at"])
        op.create_index("ix_semantic_cache_project",
                        "semantic_cache_entries", ["project_id"])
        op.create_index("ix_semantic_cache_user",
                        "semantic_cache_entries", ["user_id"])
        op.create_index("ix_semantic_cache_expires",
                        "semantic_cache_entries", ["expires_at"])

    if dialect == "postgresql":
        op.execute(
            "ALTER TABLE semantic_cache_entries "
            "ADD COLUMN IF NOT EXISTS query_embedding vector(1024)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_semantic_cache_embedding_hnsw "
            "ON semantic_cache_entries USING hnsw (query_embedding vector_cosine_ops)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "semantic_cache_entries" in inspector.get_table_names():
        op.drop_table("semantic_cache_entries")
