"""Jobs SKIP LOCKED queue + document file-metadata columns (T110/T111).

Revision ID: 003_jobs_queue
Revises: 002_auth_hardening
Create Date: 2026-09-17

Jobs:
- document_id / project_id FK links (nullable for old rows), stage /
  stage_message written by the worker, attempts / max_attempts, priority,
  run_after, locked_by / locked_at for SKIP LOCKED claiming, idempotency_key.
- jobstatus enum gains 'cancelled'. Postgres native enums cannot be altered
  portably inside a transaction block on all versions, so the migration adds
  the value with a standalone ALTER TYPE (IF NOT EXISTS guarded via a
  DO block for idempotent re-runs).
- Indexes: (status, run_after) for claiming, locked_at for reclaim,
  document_id for per-document lookup.
- Partial unique index preventing two ACTIVE jobs for the same
  document+type: only rows with status in (pending, processing) participate.

Documents (file metadata, T110):
- storage_key, mime_type, file_size, checksum (+index), extraction_status,
  extraction_method, ocr_status, embedding_status. file_path is kept as the
  legacy storage reference; new rows write both.

Rollback: drops the added indexes/columns and leaves the 'cancelled' enum
value in place (Postgres cannot drop an enum value; documented - a full
rollback requires recreating the type, which would break live rows, so the
value is retained harmlessly).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003_jobs_queue"
down_revision: Union[str, None] = "002_auth_hardening"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enum value first so new rows can use it.
    op.execute(
        "DO $$ BEGIN "
        "IF NOT EXISTS (SELECT 1 FROM pg_enum e JOIN pg_type t ON t.oid = e.enumtypid "
        "WHERE t.typname = 'jobstatus' AND e.enumlabel = 'cancelled') THEN "
        "ALTER TYPE jobstatus ADD VALUE 'cancelled'; "
        "END IF; END $$"
    )

    op.add_column("jobs", sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("jobs", sa.Column("project_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("jobs", sa.Column("stage", sa.String(length=64), nullable=True))
    op.add_column("jobs", sa.Column("stage_message", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("jobs", sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"))
    op.add_column("jobs", sa.Column("priority", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("jobs", sa.Column("run_after", sa.DateTime(), nullable=True))
    op.add_column("jobs", sa.Column("locked_by", sa.String(length=255), nullable=True))
    op.add_column("jobs", sa.Column("locked_at", sa.DateTime(), nullable=True))
    op.add_column("jobs", sa.Column("idempotency_key", sa.String(length=255), nullable=True))

    op.create_foreign_key("fk_jobs_document_id", "jobs", "documents", ["document_id"], ["id"], ondelete="SET NULL")
    op.create_foreign_key("fk_jobs_project_id", "jobs", "projects", ["project_id"], ["id"], ondelete="CASCADE")

    op.create_index("ix_jobs_status_run_after", "jobs", ["status", "run_after"])
    op.create_index("ix_jobs_locked_at", "jobs", ["locked_at"])
    op.create_index("ix_jobs_document_id", "jobs", ["document_id"])
    op.create_index("ix_jobs_project_id", "jobs", ["project_id"])
    op.create_index("ix_jobs_idempotency_key", "jobs", ["idempotency_key"])
    # One active job per document+type; completed/failed/cancelled rows exempt.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_active_document_type "
        "ON jobs (document_id, job_type) "
        "WHERE document_id IS NOT NULL AND status IN ('pending', 'processing')"
    )

    op.add_column("documents", sa.Column("storage_key", sa.String(length=1000), nullable=True))
    op.add_column("documents", sa.Column("mime_type", sa.String(length=255), nullable=True))
    op.add_column("documents", sa.Column("file_size", sa.BigInteger(), nullable=True))
    op.add_column("documents", sa.Column("checksum", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("extraction_status", sa.String(length=32), nullable=True))
    op.add_column("documents", sa.Column("extraction_method", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("ocr_status", sa.String(length=32), nullable=True))
    op.add_column("documents", sa.Column("embedding_status", sa.String(length=32), nullable=True))
    op.create_index("ix_documents_checksum", "documents", ["checksum"])
    # Backfill: storage_key mirrors file_path for existing rows.
    op.execute("UPDATE documents SET storage_key = file_path WHERE storage_key IS NULL")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_jobs_active_document_type")
    op.drop_index("ix_jobs_idempotency_key", table_name="jobs")
    op.drop_index("ix_jobs_project_id", table_name="jobs")
    op.drop_index("ix_jobs_document_id", table_name="jobs")
    op.drop_index("ix_jobs_locked_at", table_name="jobs")
    op.drop_index("ix_jobs_status_run_after", table_name="jobs")
    op.drop_constraint("fk_jobs_project_id", "jobs", type_="foreignkey")
    op.drop_constraint("fk_jobs_document_id", "jobs", type_="foreignkey")
    for col in ("idempotency_key", "locked_at", "locked_by", "run_after", "priority",
                "max_attempts", "attempts", "stage_message", "stage",
                "project_id", "document_id"):
        op.drop_column("jobs", col)
    # NOTE: 'cancelled' enum value intentionally retained (see docstring).
    op.drop_index("ix_documents_checksum", table_name="documents")
    for col in ("embedding_status", "ocr_status", "extraction_method",
                "extraction_status", "checksum", "file_size", "mime_type", "storage_key"):
        op.drop_column("documents", col)
