"""010: minimal knowledge-model version headers.

This is deliberately small: the canonical node tree remains in
``knowledge_nodes`` and JSONB metadata remains the artifact store.  The
header adds an atomic project/version/fingerprint identity so production
rebuilds can mark an older version superseded without duplicating nodes.
The new node column is nullable for rows created before this migration.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "010_knowledge_model_version"
down_revision = "009_knowledge_spine"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "knowledge_models",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False, server_default="building"),
        sa.Column("source_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("node_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("project_id", "version", name="uq_knowledge_models_project_version"),
    )
    op.create_index("ix_knowledge_models_project_id", "knowledge_models", ["project_id"])
    op.create_index(
        "ix_knowledge_models_source_fingerprint",
        "knowledge_models",
        ["source_fingerprint"],
    )
    op.add_column(
        "knowledge_nodes",
        sa.Column("model_version", sa.Integer(), nullable=True),
    )
    # The original uniqueness rule did not include a model version.  Replace
    # it with a version-scoped partial unique index so old versions remain
    # readable while each version still has dense sibling ordinals.
    try:
        op.drop_constraint(
            "uq_knowledge_nodes_doc_parent_ord",
            "knowledge_nodes",
            type_="unique",
        )
    except Exception:
        pass
    op.create_index(
        "uq_knowledge_nodes_version_doc_parent_ord",
        "knowledge_nodes",
        ["document_id", "model_version", "parent_id", "order_index"],
        unique=True,
        postgresql_where=sa.text("parent_id IS NOT NULL"),
    )
    op.create_index(
        "ix_knowledge_nodes_model_version",
        "knowledge_nodes",
        ["model_version"],
    )


def downgrade() -> None:
    try:
        op.drop_index(
            "uq_knowledge_nodes_version_doc_parent_ord",
            table_name="knowledge_nodes",
        )
    except Exception:
        pass
    try:
        op.drop_index("ix_knowledge_nodes_model_version", table_name="knowledge_nodes")
    except Exception:
        pass
    try:
        op.drop_column("knowledge_nodes", "model_version")
    except Exception:
        pass
    try:
        op.drop_index("ix_knowledge_models_source_fingerprint", table_name="knowledge_models")
    except Exception:
        pass
    try:
        op.drop_index("ix_knowledge_models_project_id", table_name="knowledge_models")
    except Exception:
        pass
    try:
        op.drop_table("knowledge_models")
    except Exception:
        pass
