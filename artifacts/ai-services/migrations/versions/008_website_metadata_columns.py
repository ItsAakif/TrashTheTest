"""008: additive study-website columns (live-bug batch, bug 7).

The 001 websites table predates the Website ORM metadata: live rows
have no title/description/visibility/status/version, so website
generate/detail/publish 500 with `column websites.title does not
exist` on any database stamped before this revision.

Additive only, no data loss:
- title VARCHAR(500) NULL, description TEXT NULL, version VARCHAR(20)
  NULL (nullable so pre-migration rows keep working; the ORM already
  treats them as optional).
- visibility/status stored as plain VARCHAR with server defaults
  ('private'/'draft') matching the ORM enum values, so old rows read
  back as private drafts until explicitly published. Plain VARCHAR
  (not PG ENUM) keeps the migration re-runnable and avoids enum-lock
  issues on a live table.

Idempotent (IF NOT EXISTS / DO guards): safe to run on databases
where 006/007 were applied directly while stamped at 004 (005
refuses by design until the embedding backfill + ORM cleanup land,
so those databases traverse 008's DDL directly -- same precedent as
the 007 docstring).

Offline SQLite degrades honestly (logs a warning, changes nothing).
"""

from alembic import op

revision = "008_website_metadata_columns"
down_revision = "007_outputtype_phase4_values"
branch_labels = None
depends_on = None

_COLUMNS = (
    ("title", "VARCHAR(500)", None),
    ("description", "TEXT", None),
    ("visibility", "VARCHAR(16)", "private"),
    ("status", "VARCHAR(16)", "draft"),
    ("version", "VARCHAR(20)", "1"),
)


def _is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def upgrade() -> None:
    if not _is_postgres():
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "008 skipped: live Postgres required (offline SQLite changes nothing)."
        )
        return
    for name, type_, default in _COLUMNS:
        if default is None:
            op.execute(f"ALTER TABLE websites ADD COLUMN IF NOT EXISTS {name} {type_}")
        else:
            op.execute(
                f"ALTER TABLE websites ADD COLUMN IF NOT EXISTS {name} {type_} "
                f"DEFAULT '{default}'"
            )
    # Backfill pre-migration rows, then enforce the ORM's non-null
    # visibility/status defaults (nullable add first so the statement
    # never fails on a populated live table).
    op.execute("UPDATE websites SET visibility = 'private' WHERE visibility IS NULL")
    op.execute("UPDATE websites SET status = 'draft' WHERE status IS NULL")
    op.execute("ALTER TABLE websites ALTER COLUMN visibility SET DEFAULT 'private'")
    op.execute("ALTER TABLE websites ALTER COLUMN status SET DEFAULT 'draft'")
    op.execute("ALTER TABLE websites ALTER COLUMN visibility SET NOT NULL")
    op.execute("ALTER TABLE websites ALTER COLUMN status SET NOT NULL")


def downgrade() -> None:
    if not _is_postgres():
        return
    for name, _, _ in _COLUMNS:
        op.execute(f"ALTER TABLE websites DROP COLUMN IF EXISTS {name}")
