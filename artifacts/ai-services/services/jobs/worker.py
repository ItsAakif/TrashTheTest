"""Standalone job worker process (T111) + Celery dispatch target.

Run:  python -m services.jobs.worker [--once]
  or: celery -A services.jobs.celery_app.celery worker --loglevel=info
- Claims due jobs via SKIP LOCKED, runs the registered handler with a
  bounded concurrency semaphore (WORKER_CONCURRENCY), heartbeats the lock,
  writes stage transitions as it performs them.
- The Celery task services/jobs/celery_app.run_job(job_id) submits a job
  id and re-enters THIS module's claim + handler + complete/fail path:
  Postgres SKIP LOCKED stays the single truth, so there is no dual
  processing (whoever fails to claim skips honestly).
- Claims due jobs via SKIP LOCKED, runs the registered handler with a
  bounded concurrency semaphore (WORKER_CONCURRENCY), heartbeats the lock,
  writes stage transitions as it performs them.
- Graceful SIGTERM: stops claiming, lets in-flight jobs finish within a
  grace window, otherwise releases their locks so a restart resumes them.
- Handler registry: document_processing ships a Phase-1 honest handler that
  advances observable stages (init->finalization) and marks the document
  row; Phase 2 replaces the extraction/chunking/embedding internals without
  changing the queue contract.

Resumability: handlers receive the job row; stages already recorded in
job.stage are skipped (the Phase-1 handler is idempotent by construction).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

Handler = Callable[..., Awaitable[None]]

_HANDLERS: dict[str, Handler] = {}
_shutdown = asyncio.Event()


def register_handler(job_type: str, handler: Handler) -> None:
    _HANDLERS[job_type] = handler


def _worker_id() -> str:
    return f"worker-{uuid.uuid4().hex[:8]}"


async def _document_processing_handler(ctx: Any, job: Any) -> None:
    """Phase-2 real pipeline: delegates to the resumable stage machine."""
    from services.jobs.handlers.document_processing import process_job

    await process_job(ctx, job)


register_handler("document_processing", _document_processing_handler)


async def _knowledge_model_handler(ctx: Any, job: Any) -> None:
    """Build/persist the academic spine outside the request path."""
    from services.academic.orchestrator import build_project_spine
    from services.jobs import queue as q

    data = dict(getattr(job, "input_data", None) or {})
    try:
        result = await build_project_spine(
            ctx.db,
            data.get("project_id") or getattr(job, "project_id", None),
            document_ids=data.get("document_ids") or [],
            project_title=str(data.get("project_title") or "Academic project"),
            force=bool(data.get("force", False)),
        )
        out = dict(getattr(job, "output_data", None) or {})
        out["academic_spine"] = result.as_dict()
        out["_stage"] = "model"
        await q.complete(ctx.db, job.id, output_data=out)
    except Exception as exc:
        await q.fail(ctx.db, job.id, str(exc) or type(exc).__name__, retry=False)


async def _notes_hierarchical_handler(ctx: Any, job: Any) -> None:
    """Dispatch the resumable academic map/reduce job."""
    from services.academic.orchestrator import process_hierarchical_job

    await process_hierarchical_job(ctx, job)


# Additive handlers.  The document-processing stage budget below is left
# untouched; each academic job owns its stage vocabulary in the academic
# package.
register_handler("knowledge_model", _knowledge_model_handler)
register_handler("notes_hierarchical", _notes_hierarchical_handler)
register_handler("academic_generation", _notes_hierarchical_handler)


# ADR-008 / T009: the handlers package also registers structure_reprocess,
# but it is only imported lazily (inside the document-processing handler).
# Register a lazy delegate here so `python -m services.jobs.worker` and the
# Celery path see the job type the moment this module is imported.
async def _structure_reprocess_handler(ctx: Any, job: Any) -> None:
    from services.jobs.handlers.structure_reprocess import process_job

    await process_job(ctx, job)


register_handler("structure_reprocess", _structure_reprocess_handler)


# Static stage budget (T209): kept as the contract; within a stage the
# percentage comes from real counts (see handlers.document_processing).
STAGE_PROGRESS_RANGES = {
    "validate": (0, 5),
    "extract": (5, 55),
    "structure": (55, 60),
    "validate2": (60, 65),
    "chunk": (65, 80),
    "embed": (80, 95),
    "complete": (95, 100),
}


class _Ctx:
    def __init__(self, db: Any, session_factory: Any = None):
        self.db = db
        # Handlers use a fresh session to persist a terminal failure. This
        # avoids a failed worker transaction leaving the row PROCESSING until
        # the lock-timeout reclaimer runs it again.
        self.session_factory = session_factory


# ---------------------------------------------------------------------------
# Fairness rule (upload starvation, 67-minute ingest incident):
# document_processing owns priority 9 and normally wins the claim order, but a
# notes job already claimed can still hold a slot for 300s per topic. While
# ANY document_processing job is pending, at most ONE academic/notes job may
# occupy a slot; extra claimed notes rows are released back to PENDING so the
# freed slot serves the upload.
# ---------------------------------------------------------------------------
NOTES_JOB_TYPES = frozenset({"notes_hierarchical", "academic_generation"})
MAX_NOTES_JOBS_WHILE_UPLOADS_PENDING = 1


def _job_type_name(job: Any) -> str:
    job_type = getattr(job, "job_type", "")
    return str(getattr(job_type, "value", job_type))


def _is_notes_job(job: Any) -> bool:
    return _job_type_name(job) in NOTES_JOB_TYPES


def _partition_fair_claims(
    claimed: list[Any],
    uploads_pending: bool,
    running_notes: int = 0,
    max_notes: int = MAX_NOTES_JOBS_WHILE_UPLOADS_PENDING,
) -> tuple[list[Any], list[Any]]:
    """Split freshly claimed jobs into (keep, release) under the fairness rule.

    Pure function so the rule is unit-testable without a database. Jobs that
    are not notes jobs are always kept.
    """
    if not uploads_pending:
        return list(claimed), []
    allowance = max(0, int(max_notes) - int(running_notes))
    keep: list[Any] = []
    release: list[Any] = []
    for job in claimed:
        if _is_notes_job(job):
            if allowance > 0:
                allowance -= 1
                keep.append(job)
            else:
                release.append(job)
        else:
            keep.append(job)
    return keep, release


async def _run_one(session_factory: Any, worker_id: str, job_id: uuid.UUID, semaphore: asyncio.Semaphore) -> None:
    from database.models.job import JobStatus
    from services.jobs import queue as q

    async with semaphore:
        if _shutdown.is_set():
            return
        async with session_factory() as db:
            job = await db.get(__import__("database.models.job", fromlist=["Job"]).Job, job_id)
            if job is None or job.status != JobStatus.PROCESSING or job.locked_by != worker_id:
                return
            handler = _HANDLERS.get(job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type))
            if handler is None:
                await q.fail(db, job.id, f"No handler for job type {job.job_type}", retry=False)
                await db.commit()
                return
            try:
                await handler(_Ctx(db, session_factory), job)
                await db.commit()
            except asyncio.CancelledError:
                await db.rollback()
                raise
            except Exception as exc:  # noqa: BLE001 - boundary: record, don't crash loop
                await db.rollback()
                async with session_factory() as db2:
                    await q.fail(db2, job.id, str(exc), retry=True)
                    await db2.commit()


async def _poll_loop(session_factory: Any, worker_id: str, concurrency: int, once: bool) -> int:
    from services.jobs import queue as q

    try:
        configured = int(__import__("config.settings", fromlist=["get_worker_config"]).get_worker_config().concurrency)
        concurrency = configured or concurrency
    except Exception:
        pass
    logger.info("Worker %s polling (effective concurrency=%s)", worker_id, concurrency)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    processed = 0
    running_notes = 0  # notes jobs currently holding a slot on this worker
    while not _shutdown.is_set():
        async with session_factory() as db:
            try:
                from database.models.job import JobType

                await q.reclaim_stale(db)
                # Fairness rule: uploads pending => cap concurrent notes work.
                upload_count = await q.pending_count(db, JobType.DOCUMENT_PROCESSING)
                uploads_pending = upload_count > 0
                claimed = await q.claim(db, worker_id, limit=max(1, concurrency))
                keep, deferred = _partition_fair_claims(
                    claimed, uploads_pending, running_notes=running_notes
                )
                for job in deferred:
                    await q.release(db, job.id, worker_id)
                if deferred:
                    logger.info(
                        "Worker %s released %d notes job(s) while %d upload(s) pending "
                        "(fairness: max %d concurrent notes)",
                        worker_id,
                        len(deferred),
                        upload_count,
                        MAX_NOTES_JOBS_WHILE_UPLOADS_PENDING,
                    )
                claimed = keep
                await db.commit()
            except Exception as exc:
                logger.warning("Claim cycle failed: %s", exc)
                await db.rollback()
                claimed = []
        if not claimed:
            if once:
                return processed
            await asyncio.sleep(2)
            continue
        claimed_notes = sum(1 for j in claimed if _is_notes_job(j))
        running_notes += claimed_notes
        tasks = [asyncio.create_task(_run_one(session_factory, worker_id, j.id, semaphore)) for j in claimed]
        try:
            await asyncio.gather(*tasks)
        finally:
            running_notes = max(0, running_notes - claimed_notes)
        processed += len(tasks)
        if once:
            return processed
    return processed


async def run_worker(session_factory: Any = None, concurrency: int = 2, once: bool = False) -> int:
    if session_factory is None:
        import importlib

        _mod = importlib.import_module("database.engine")
        session_factory = getattr(_mod, "async_session_maker", None) or getattr(
            _mod, "AsyncSessionLocal", None
        )
        if session_factory is None:
            raise RuntimeError("no async session factory in database.engine")
    worker_id = _worker_id()
    logger.info("Worker %s starting (concurrency=%s, once=%s)", worker_id, concurrency, once)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _shutdown.set)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda *_: _shutdown.set())

    try:
        return await _poll_loop(session_factory, worker_id, concurrency, once)
    finally:
        # Release locks held by this worker so a restart resumes them.
        try:
            from services.jobs import queue as q

            async with session_factory() as db:
                from sqlalchemy import and_, select

                from database.models.job import Job, JobStatus

                rows = (
                    await db.execute(
                        select(Job).where(and_(Job.locked_by == worker_id, Job.status == JobStatus.PROCESSING))
                    )
                ).scalars().all()
                for job in rows:
                    job.status = JobStatus.PENDING
                    job.locked_by = None
                    job.locked_at = None
                    job.updated_at = datetime.utcnow()
                await db.commit()
        except Exception as exc:
            logger.warning("Lock release on shutdown failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="TrashTheTest job worker")
    parser.add_argument("--once", action="store_true", help="Claim and run one batch, then exit")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run_worker(concurrency=args.concurrency, once=args.once))


if __name__ == "__main__":
    main()


__all__ = [
    "register_handler",
    "run_worker",
    "main",
    "STAGE_PROGRESS_RANGES",
    "NOTES_JOB_TYPES",
    "MAX_NOTES_JOBS_WHILE_UPLOADS_PENDING",
]
