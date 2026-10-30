"""STRUCTURE_REPROCESS enqueue CLI (idempotent, no data loss).

Usage (from artifacts/ai-services):

    D:\\Python-3.13\\python.exe scripts/reprocess_structure.py [--dry-run]
        [--limit N] [--project PROJECT_ID] [--all]

Default behaviour: enqueue one STRUCTURE_REPROCESS job for every
document whose structure_method IS NULL OR structure_method IN
('legacy_conclusion','legacy_introduction') OR indexing_state IN
('indexing','partial','failed').  The worker is idempotent so this
is safe to re-run.

--dry-run never opens a write transaction: it prints the candidate
document ids and exits 0.

ADR-008 (T011).
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, List, Tuple
from uuid import UUID

# Bootstrap: run either as `python -m scripts.reprocess_structure` or as
# `python scripts/reprocess_structure.py` from artifacts/ai-services.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect as sa_inspect, text as sa_text

from database.engine import DatabaseSession
from database import repositories
from services.jobs import queue as job_queue
from services.jobs.types import JobType


def _documents_columns(sync_session):
    """Column names of the documents table, using a sync view of the
    AsyncSession (safe: runs inside db.run_sync)."""
    inspector = sa_inspect(sync_session.get_bind())
    if "documents" not in inspector.get_table_names():
        return set()
    return {str(column["name"]) for column in inspector.get_columns("documents")}


logger = logging.getLogger("reprocess_structure")

# Re-running the CLI must not stack duplicate pending jobs for the same
# document (enqueue() dedupes on (idempotency_key, project_id)).
_IDEMPOTENCY_PREFIX = "structure_reprocess"


async def _candidates(db: Any, project_id: str = None,
                      limit: int | None = None,
                      all_documents: bool = False) -> List[Tuple[UUID, Any]]:
    """(document_id, project_id) pairs to enqueue, oldest first."""
    if all_documents:
        ids = await repositories.list_all_document_ids(
            db, project_id=project_id, limit=limit)
        return [(uid, None) for uid in ids]
    rows = await repositories.list_documents_for_reprocess(
        db, project_id=project_id, limit=limit)
    return [(row.id, row.project_id) for row in rows]


async def main() -> int:
    p = argparse.ArgumentParser(
        description="Enqueue STRUCTURE_REPROCESS jobs (ADR-008 backfill).")
    p.add_argument("--dry-run", action="store_true",
                   help="print candidate document ids and exit; no writes")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--all", action="store_true",
                   help="reprocess every document, not just degraded")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    async with DatabaseSession() as db:
        # Fail loudly when the database is unreachable: otherwise the
        # pre-014 tolerance below would report "0 candidates" for a service
        # that is simply down, which reads like a successful empty run.
        try:
            await db.execute(sa_text("SELECT 1"))
        except Exception as exc:
            logger.error("database unreachable: %s", exc)
            await db.rollback()
            return 1

        # The repository helpers degrade to [] on a pre-014 schema; tell the
        # operator explicitly instead of printing an empty "success".
        try:
            names = await db.run_sync(_documents_columns)
        except Exception as exc:
            logger.error("schema introspection failed: %s", exc)
            await db.rollback()
            return 1
        if "structure_method" not in names:
            logger.error(
                "documents.structure_method is missing - apply migration 014: "
                "python -m alembic upgrade head")
            await db.rollback()
            return 1

        try:
            candidates = await _candidates(
                db,
                project_id=args.project,
                limit=args.limit,
                all_documents=args.all,
            )
        except Exception as exc:
            logger.error("candidate query failed (is migration 014 applied?): %s",
                         exc)
            await db.rollback()
            return 1

        logger.info("candidate documents: %d (dry-run=%s)",
                    len(candidates), args.dry_run)
        if args.dry_run:
            for did, _pid in candidates:
                print(did)
            return 0

        enq = 0
        for did, pid in candidates:
            try:
                project_id = pid
                if project_id is None:
                    doc = await repositories.get_document_by_id(db, did)
                    project_id = getattr(doc, "project_id", None) if doc else None
                await job_queue.enqueue(
                    db,
                    JobType.STRUCTURE_REPROCESS,
                    input_data={"document_id": str(did),
                                "user_id": "system:reprocess"},
                    document_id=did,
                    project_id=project_id,
                    idempotency_key=f"{_IDEMPOTENCY_PREFIX}:{did}",
                    priority=job_queue.PRIORITY_STRUCTURE_REPROCESS,
                )
                await db.commit()
                enq += 1
            except Exception as exc:
                logger.warning("enqueue failed for %s: %s", did, exc)
                await db.rollback()
        logger.info("enqueued %d jobs", enq)
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
