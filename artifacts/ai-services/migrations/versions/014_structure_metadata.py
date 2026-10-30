"""014: per-document structure verdict columns.

Additive only.  documents.structure_method, structure_confidence,
structure_degraded are nullable; old rows continue to work and the
backfill writes them as it goes.  chunks.metadata is already JSONB;
no chunks table change is required.

Also registers the ``structure_reprocess`` label on the existing Postgres
``jobtype`` enum (same pattern as 011) so STRUCTURE_REPROCESS jobs can be
enqueued.
"""

from alembic import op
import sqlalchemy as sa

revision = "014_structure_metadata"
down_revision = "013_document_source_type"
branch_labels = None
depends_on = None


_VERDICT_COLUMNS = (
    sa.Column("structure_method", sa.String(32), nullable=True),
    sa.Column("structure_confidence", sa.Float(), nullable=True),
    sa.Column("structure_degraded", sa.Boolean(), nullable=True),
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "documents" in inspector.get_table_names():
        columns = {str(row["name"]) for row in inspector.get_columns("documents")}
        for column in _VERDICT_COLUMNS:
            if column.name not in columns:
                op.add_column("documents", column)

    # PostgreSQL-only: the jobtype enum predates the non-native ORM mapping,
    # so a new job label must be added to the live type (011 pattern).
    if bind.dialect.name == "postgresql":
        op.execute(
            "ALTER TYPE jobtype ADD VALUE IF NOT EXISTS 'structure_reprocess'"
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "documents" not in inspector.get_table_names():
        return
    columns = {str(row["name"]) for row in inspector.get_columns("documents")}
    for name in ("structure_degraded", "structure_confidence",
                 "structure_method"):
        if name in columns:
            op.drop_column("documents", name)
    # PostgreSQL cannot remove an enum label transaction-safely; the
    # structure_reprocess value is retained, matching the 011 policy.
