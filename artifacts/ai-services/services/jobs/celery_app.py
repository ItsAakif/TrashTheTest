"""Celery dispatch for jobs (optional). Postgres SKIP LOCKED is the truth.

- Broker: REDIS_URL (default redis://localhost:6379/0).
- Task run_job(job_id): opens an async session, re-reads the job row with
  ownership checks (locked_by/terminal-state guards), then calls the same
  claim + process_job + complete/fail path as `python -m services.jobs.worker`.
- NO dual processing: Celery never executes pipeline logic directly; it
  only submits the job id, and queue.claim (SKIP LOCKED) decides who owns it.
- Progress/cancel/idempotency/resume/SIGTERM semantics all live in
  services/jobs/queue.py + handlers/document_processing.py and are reused
  unchanged here.

Run: celery -A services.jobs.celery_app.celery worker --loglevel=info
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


def _broker_url() -> str:
    try:
        from config.settings import get_worker_config

        return getattr(get_worker_config(), "redis_url", None) or os.getenv(
            "REDIS_URL", "redis://localhost:6379/0"
        )
    except Exception:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _make_celery():
    try:
        from celery import Celery
    except ImportError:
        return None
    app = Celery("trashthetest", broker=_broker_url())
    app.conf.update(
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_reject_on_worker_lost=True,
    )
    return app


celery = _make_celery()


def _session_factory():
    """Session factory (async_session_maker is the canonical name)."""
    import importlib

    _mod = importlib.import_module("database.engine")
    for attr in ("async_session_maker", "AsyncSessionLocal"):
        factory = getattr(_mod, attr, None)
        if factory is not None:
            return factory
    raise RuntimeError("no async session factory in database.engine")


async def _run_job_async(job_id: str) -> dict:
    """Claim-gated single-job run: the queue decides ownership, not Celery."""
    SessionFactory = _session_factory()
    from database.models.job import JobStatus
    from services.jobs import queue as q
    from services.jobs.worker import _Ctx, _HANDLERS

    try:
        jid = uuid.UUID(str(job_id))
    except Exception:
        return {"ok": False, "error": f"malformed job id: {job_id}"}

    async with SessionFactory() as db:
        job = await db.get(
            __import__("database.models.job", fromlist=["Job"]).Job, jid
        )
        if job is None:
            return {"ok": False, "error": "job not found"}
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return {"ok": True, "skipped": True,
                    "status": job.status.value}  # idempotent: terminal stays
        if job.status == JobStatus.PROCESSING and job.locked_by:
            return {"ok": True, "skipped": True,
                    "status": "processing-owned"}  # owned elsewhere: no steal
        # Not claimed yet: claim via SKIP LOCKED (single truth decides).
        worker_id = f"celery-{os.getpid()}"
        claimed = await q.claim(db, worker_id, limit=100)
        await db.commit()
        owned = [j for j in claimed if str(j.id) == str(jid)]
        if not owned:
            return {"ok": True, "skipped": True,
                    "status": "not-claimed"}  # someone else owns it: no dual run

    # Run the registered handler with the same boundary semantics as worker.
    async with SessionFactory() as db:
        job = await db.get(
            __import__("database.models.job", fromlist=["Job"]).Job, jid
        )
        handler = _HANDLERS.get(
            job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)
        )
        if handler is None:
            await q.fail(db, job.id, f"No handler for job type {job.job_type}",
                         retry=False)
            await db.commit()
            return {"ok": False, "error": "no handler"}
        try:
            await handler(_Ctx(db), job)
            await db.commit()
        except asyncio.CancelledError:
            await db.rollback()
            raise
        except Exception as exc:  # noqa: BLE001 - boundary: record, don't crash
            await db.rollback()
            async with SessionFactory() as db2:
                await q.fail(db2, jid, str(exc), retry=True)
                await db2.commit()
            return {"ok": False, "error": str(exc)}
    return {"ok": True, "job_id": str(jid)}


def run_job_sync(job_id: str) -> dict:
    """Sync wrapper (Celery task body + direct-call helper for tests)."""
    return asyncio.run(_run_job_async(job_id))


if celery is not None:  # pragma: no cover - requires broker at runtime

    @celery.task(name="jobs.run_job", bind=True, max_retries=3)
    def run_job(self, job_id: str) -> dict:  # type: ignore[no-redef]
        try:
            return run_job_sync(job_id)
        except Exception as exc:
            logger.warning("run_job(%s) failed: %s", job_id, exc)
            raise self.retry(exc=exc, countdown=30)

else:  # pragma: no cover - celery absent (offline/test env)

    def run_job(job_id: str) -> dict:  # type: ignore[no-redef]
        """Fallback: direct sync run when celery is not installed."""
        return run_job_sync(job_id)


def enqueue_celery(job_id: str | uuid.UUID) -> Optional[str]:
    """Submit a job id to the Celery broker (None when celery unavailable)."""
    if celery is None:
        return None
    try:
        result = celery.send_task("jobs.run_job", args=[str(job_id)])
        return str(getattr(result, "id", ""))
    except Exception as exc:
        logger.warning("Celery submit failed for %s: %s", job_id, exc)
        return None


__all__ = ["celery", "run_job", "run_job_sync", "enqueue_celery"]
