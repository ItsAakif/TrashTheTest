"""017: retrieval traces (S4 observability). Additive, append-only."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "017_retrieval_traces"
down_revision = "016_semantic_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name if bind is not None else "sqlite"
    inspector = sa.inspect(bind)
    if "retrieval_traces" not in inspector.get_table_names():
        op.create_table(
            "retrieval_traces",
            sa.Column("id", postgresql.UUID(as_uuid=True) if dialect == "postgresql" else sa.String(36), primary_key=True),
            sa.Column("request_id", sa.String(64), nullable=True),
            sa.Column("user_id", postgresql.UUID(as_uuid=True) if dialect == "postgresql" else sa.String(36), nullable=True),
            sa.Column("project_id", postgresql.UUID(as_uuid=True) if dialect == "postgresql" else sa.String(36), nullable=True),
            sa.Column("document_id", postgresql.UUID(as_uuid=True) if dialect == "postgresql" else sa.String(36), nullable=True),
            sa.Column("query_text", sa.Text(), nullable=True),
            sa.Column("retrieval_mode", sa.String(16), nullable=True),
            sa.Column("rerank", sa.Integer(), nullable=True),
            sa.Column("chunk_ids", sa.JSON(), nullable=True),
            sa.Column("scores", sa.JSON(), nullable=True),
            sa.Column("cache_layer", sa.String(16), nullable=True),
            sa.Column("cache_hit", sa.Integer(), nullable=True),
            sa.Column("latency_ms", sa.Float(), nullable=True),
            sa.Column("rerank_latency_ms", sa.Float(), nullable=True),
            sa.Column("context_tokens_estimate", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        for name in ("request_id", "user_id", "project_id", "document_id", "created_at"):
            op.create_index(f"ix_retrieval_traces_{name}", "retrieval_traces", [name])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "retrieval_traces" in inspector.get_table_names():
        op.drop_table("retrieval_traces")
