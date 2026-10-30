"""012: widen Alembic's revision bookkeeping column.

Alembic creates ``alembic_version.version_num`` as ``VARCHAR(32)`` by
 default. The historical ``005_contract_drop_legacy_embedding`` revision is
34 characters, so a database stamped at 004 cannot record that revision.
This migration widens the bookkeeping column to VARCHAR(255) without
changing application data. It is intentionally idempotent; the live runbook
also performs the same ALTER before ``alembic upgrade head`` because a
later migration cannot repair the first failed version stamp.
"""

from alembic import op

revision = "012_alembic_version_width"
down_revision = "011_academic_job_types"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "012 skipped: live Postgres required for the Alembic bookkeeping repair."
        )
        return

    op.execute(
        "ALTER TABLE IF EXISTS alembic_version "
        "ALTER COLUMN version_num TYPE VARCHAR(255)"
    )


def downgrade() -> None:
    # Never narrow the column: revision 005 is already longer than 32
    # characters, and narrowing would make a future stamp fail again.
    return None
