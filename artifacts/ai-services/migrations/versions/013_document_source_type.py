"""013: classify uploaded academic source documents.

Additive only.  ``documents.source_type`` is nullable during rolling
upgrades; the application maps NULL to study_material.  No file, page,
chunk, or generated-content data is touched.
"""

from alembic import op
import sqlalchemy as sa

revision = "013_document_source_type"
down_revision = "012_alembic_version_width"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "documents" not in inspector.get_table_names():
        return
    columns = {str(row["name"]) for row in inspector.get_columns("documents")}
    if "source_type" in columns:
        return
    # Keep the application enum non-native so old databases do not need a
    # destructive enum conversion.  The column is nullable for old rows.
    op.add_column(
        "documents",
        sa.Column("source_type", sa.String(32), nullable=True),
    )
    op.create_index(
        "ix_documents_source_type", "documents", ["source_type"], unique=False
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "documents" not in inspector.get_table_names():
        return
    columns = {str(row["name"]) for row in inspector.get_columns("documents")}
    if "source_type" not in columns:
        return
    try:
        op.drop_index("ix_documents_source_type", table_name="documents")
    except Exception:
        pass
    op.drop_column("documents", "source_type")
