"""011: add academic job type labels to the existing Postgres jobtype enum."""

from alembic import op

revision = "011_academic_job_types"
down_revision = "010_knowledge_model_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # PostgreSQL 12+ supports IF NOT EXISTS.  Keep each statement separate so
    # already-migrated databases are harmless on re-apply; enum values are
    # intentionally retained on downgrade because removing one is unsafe.
    for value in ("knowledge_model", "notes_hierarchical", "academic_generation"):
        op.execute(f"ALTER TYPE jobtype ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    # PostgreSQL cannot remove an enum label transaction-safely.  The values
    # are harmless and are retained, matching the existing jobstatus policy.
    return None
