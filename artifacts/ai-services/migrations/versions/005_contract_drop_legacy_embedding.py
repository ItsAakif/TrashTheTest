"""005: contract step - drop the legacy BYTEA embedding column (T309).

Contract step of ADR-005 migration notes (step 6). Runs ONLY when the
backfill is complete: zero chunks with NULL embedding_model. Otherwise
upgrade DEFERS (warn + no-op) instead of refusing: live-bug batch find
is that a hard refuse at 005 wedges `alembic upgrade head` on
databases stamped at 004, so additive fixes in 006 (provider keys),
007 (outputtype labels) and 008 (websites.title et al) can never
apply and the website surface 500s with `column websites.title does
not exist`. Deferring is data-safe: the legacy BYTEA column is simply
left in place (the ORM still maps it) until the backfill + ORM cleanup
land. A later `alembic upgrade head` does not re-run an already stamped
005; do not downgrade through 006+ (that can remove live rows). After the
ORM cleanup is deployed, use the guarded DROP in the runbook instead.
No data loss either way: chunks.content / content_hash are untouched.

Pre-conditions (checked in upgrade):
  SELECT count(*) FROM chunks
  WHERE embedding_v IS NULL OR embedding_model IS NULL  -> must be 0
  else: log a warning naming the remediation and return WITHOUT
  marking failure, so the chain proceeds to 006+.

Rollback procedure (exercised once against a staging copy before prod):
  1. Re-add the column:
       ALTER TABLE chunks ADD COLUMN embedding BYTEA;
  2. Re-embed from the (embedding_model, content_hash) cache - no source
     data is lost because chunks.content and chunks.content_hash are
     untouched by this migration:
       python -m scripts.reindex_embeddings --all
     (rows whose content_hash already has a cached vector skip the model
     call, so rollback re-embed is cache-fast).
  3. Downgrade this revision (alembic downgrade -1), which re-adds the
     column if it was dropped by upgrade.

For a live database where 005 was already stamped deferred, do not use
``alembic downgrade 005`` as a shortcut: 006+ may contain application data.
After ``scripts.reindex_embeddings --all`` reports zero missing vectors and
the legacy ORM mapping is removed, run the guarded DROP from
``.planning/progress/alembic-fix.md``.

Live Postgres required for the vector guard and the DROP; offline
SQLite degrades honestly (logs a warning, changes nothing).

NOTE: the ORM keeps the legacy `embedding` mapping until the follow-up
cleanup removes the Python-cosine fallback paths. Until then, running
this contract migration on a database that application code still reads
requires that follow-up deployed first - upgrade refuses when the
running code declares the legacy column (see _legacy_in_use()).
"""

from alembic import op
import sqlalchemy as sa

revision = "005_contract_drop_legacy_embedding"
down_revision = "004_document_pages_and_pgvector"
branch_labels = None
depends_on = None


def _is_postgres() -> bool:
    try:
        return op.get_bind().dialect.name == "postgresql"
    except Exception:
        return False


def _legacy_in_use() -> bool:
    """True while ORM/service code still maps the legacy BYTEA column."""
    try:
        from database.models import Chunk as _Chunk

        return hasattr(_Chunk, "embedding")
    except Exception:
        return True


def upgrade() -> None:
    if not _is_postgres():
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "005 contract step skipped: live Postgres required (offline SQLite changes nothing)."
        )
        return
    remaining = op.get_bind().execute(sa.text(
        "SELECT count(*) FROM chunks WHERE embedding_v IS NULL OR embedding_model IS NULL"
    )).scalar() or 0
    if int(remaining) != 0:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            f"005 contract step DEFERRED: {remaining} chunks still lack "
            "embedding_v/embedding_model (head proceeds to 006+ so additive "
            "fixes apply). Remediation then re-run DROP: "
            "python -m scripts.reindex_embeddings --all"
        )
        return
    if _legacy_in_use():
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "005 contract step DEFERRED: application code still maps the legacy "
            "chunks.embedding BYTEA column (head proceeds to 006+). Remediation: "
            "deploy the ORM cleanup that removes the legacy mapping first, then "
            "re-run the DROP."
        )
        return
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS embedding")


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding BYTEA")
