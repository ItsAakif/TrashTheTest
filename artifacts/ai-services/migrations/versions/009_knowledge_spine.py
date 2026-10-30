"""009: academic knowledge spine tables (STEP 1, additive).

Creates knowledge_nodes (Document -> Unit -> Topic -> Subtopic -> Concept
via parent_id self-reference + dense order_index) and canonical_topics
(shared registry for alias resolution). No existing table is altered, so
downgrade is a pure drop. The GIN index on aliases needs live Postgres;
all other DDL is portable.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB

revision = "009_knowledge_spine"
down_revision = "008_website_metadata_columns"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def upgrade() -> None:
    op.create_table(
        "canonical_topics",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("canonical_key", sa.String(300), nullable=False, unique=True),
        sa.Column("name", sa.String(300), nullable=False),
        sa.Column("aliases", JSONB, nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
    )
    op.create_table(
        "knowledge_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=True),
                  sa.ForeignKey("projects.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("document_id", UUID(as_uuid=True),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("parent_id", UUID(as_uuid=True),
                  sa.ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
                  nullable=True, index=True),
        sa.Column("type", sa.String(16), nullable=False, index=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("pages", JSONB, nullable=False, server_default="[]"),
        sa.Column("chunks", JSONB, nullable=False, server_default="[]"),
        sa.Column("aliases", JSONB, nullable=False, server_default="[]"),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("canonical_topic_id", UUID(as_uuid=True),
                  sa.ForeignKey("canonical_topics.id", ondelete="SET NULL"),
                  nullable=True, index=True),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.func.now()),
        sa.UniqueConstraint(
            "document_id", "parent_id", "order_index",
            name="uq_knowledge_nodes_doc_parent_ord",
        ),
    )
    op.create_index(
        "ix_knowledge_nodes_doc_type", "knowledge_nodes", ["document_id", "type"],
    )
    op.create_index(
        "ix_knowledge_nodes_parent_ord", "knowledge_nodes",
        ["parent_id", "order_index"],
    )
    if _is_postgres():
        # Alias-containment lookups for registry resolution (GIN, PG only).
        try:
            op.execute(
                "CREATE INDEX IF NOT EXISTS ix_knowledge_nodes_aliases "
                "ON knowledge_nodes USING GIN (aliases)"
            )
            op.execute(
                "CREATE INDEX IF NOT EXISTS ix_canonical_topics_aliases "
                "ON canonical_topics USING GIN (aliases)"
            )
        except Exception:
            pass


def downgrade() -> None:
    if _is_postgres():
        for stmt in [
            "DROP INDEX IF EXISTS ix_canonical_topics_aliases",
            "DROP INDEX IF EXISTS ix_knowledge_nodes_aliases",
        ]:
            try:
                op.execute(stmt)
            except Exception:
                pass
    try:
        op.drop_index("ix_knowledge_nodes_parent_ord", table_name="knowledge_nodes")
    except Exception:
        pass
    try:
        op.drop_index("ix_knowledge_nodes_doc_type", table_name="knowledge_nodes")
    except Exception:
        pass
    try:
        op.drop_table("knowledge_nodes")
    except Exception:
        pass
    try:
        op.drop_table("canonical_topics")
    except Exception:
        pass
