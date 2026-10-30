"""Postgres SKIP LOCKED job queue (T111).

enqueue() runs inside the caller's session/transaction (rolls back with it).
claim() uses SELECT ... FOR UPDATE SKIP LOCKED so concurrent workers never
take the same row. Reclaim releases rows whose locked_at exceeds
JOB_LOCK_TIMEOUT_SECONDS. Cancel marks `cancelled` (not failed).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import and_, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.job import Job, JobStatus, JobType

# ---------------------------------------------------------------------------
# Job priorities (higher number == claimed first).
#
# Uploads/ingest must never starve behind multi-minute academic LLM work: a
# 75-page PDF once sat at 0% for 40+ minutes because notes jobs (12 topics x
# up to 300s each) held both WORKER_CONCURRENCY slots. Document processing is
# therefore the highest priority, structure repair sits in the middle, and
# the long academic jobs run only on slots uploads are not using.
# ---------------------------------------------------------------------------
PRIORITY_DOCUMENT_PROCESSING = 9
PRIORITY_STRUCTURE_REPROCESS = 5
PRIORITY_ACADEMIC = 2
PRIORITY_DEFAULT = 0


def _lock_timeout() -> int:
    try:
        from config.settings import get_worker_config

        return int(get_worker_config().lock_timeout_seconds)
    except Exception:
        return 600


def _is_postgres(session: AsyncSession) -> bool:
    try:
        bind = session.get_bind()
        return bind is not None and "postgresql" in str(bind.url)
    except Exception:
        return False


async def enqueue(
    db: AsyncSession,
    job_type: JobType | str,
    input_data: Optional[dict[str, Any]] = None,
    project_id: Optional[uuid.UUID] = None,
    document_id: Optional[uuid.UUID] = None,
    priority: int = 0,
    run_after: Optional[datetime] = None,
    max_attempts: int = 3,
    idempotency_key: Optional[str] = None,
) -> Job:
    """Create a job row WITHOUT committing (caller's transaction owns it)."""
    jt = job_type if isinstance(job_type, JobType) else JobType(job_type)
    if idempotency_key and project_id is not None:
        existing = await find_by_idempotency_key(db, idempotency_key, project_id, jt)
        if existing is not None and existing.status in (JobStatus.PENDING, JobStatus.PROCESSING):
            return existing
    job = Job(
        id=uuid.uuid4(),
        job_type=jt,
        status=JobStatus.PENDING,
        progress=0,
        document_id=document_id,
        project_id=project_id,
        attempts=0,
        max_attempts=max_attempts,
        priority=priority,
        run_after=run_after,
        idempotency_key=idempotency_key,
        input_data=input_data,
    )
    db.add(job)
    await db.flush()
    return job


async def find_by_idempotency_key(
    db: AsyncSession, key: str, project_id: uuid.UUID, job_type: JobType
) -> Optional[Job]:
    result = await db.execute(
        select(Job).where(
            and_(
                Job.idempotency_key == key,
                Job.project_id == project_id,
                Job.job_type == job_type,
            )
        ).order_by(Job.created_at.desc())
    )
    return result.scalars().first()


async def claim(db: AsyncSession, worker_id: str, limit: int = 1) -> list[Job]:
    """Claim up to `limit` due jobs for `worker_id`. Never double-claims.

    Ordering: priority DESC (9 = upload/ingest) then created_at ASC (oldest
    first inside a priority band). Reversing either direction would let a
    multi-minute notes job delay a freshly queued upload again.
    """
    now = datetime.utcnow()
    base = (
        select(Job)
        .where(
            and_(
                Job.status == JobStatus.PENDING,
                or_(Job.run_after.is_(None), Job.run_after <= now),
            )
        )
        .order_by(Job.priority.desc(), Job.created_at.asc())
        .limit(limit)
    )
    if _is_postgres(db):
        base = base.with_for_update(skip_locked=True)
    rows = (await db.execute(base)).scalars().all()
    claimed: list[Job] = []
    for job in rows:
        if job.status != JobStatus.PENDING:
            continue  # raced in non-Postgres (test fakes); skip honestly
        job.status = JobStatus.PROCESSING
        job.locked_by = worker_id
        job.locked_at = now
        job.started_at = job.started_at or now
        job.updated_at = now
        claimed.append(job)
    if claimed:
        await db.flush()
    return claimed


async def heartbeat(db: AsyncSession, job_id: uuid.UUID, worker_id: str) -> Optional[Job]:
    job = await db.get(Job, job_id)
    if job is None or job.locked_by != worker_id or job.status != JobStatus.PROCESSING:
        return None
    job.locked_at = datetime.utcnow()
    await db.flush()
    return job


async def report_progress(
    db: AsyncSession,
    job_id: uuid.UUID,
    progress: int,
    stage: Optional[str] = None,
    stage_message: Optional[str] = None,
) -> Optional[Job]:
    job = await db.get(Job, job_id)
    if job is None:
        return None
    job.progress = max(0, min(100, int(progress)))
    if stage is not None:
        job.stage = stage
    if stage_message is not None:
        job.stage_message = stage_message
        out = dict(job.output_data or {})
        out["_stage"] = stage or out.get("_stage")
        out["_stage_message"] = stage_message
        job.output_data = out
    elif stage is not None:
        out = dict(job.output_data or {})
        out["_stage"] = stage
        job.output_data = out
    now = datetime.utcnow()
    # A real progress write is also a heartbeat. Without refreshing
    # locked_at, a healthy long academic/provider stage is reclaimed by
    # another worker after JOB_LOCK_TIMEOUT_SECONDS and restarted forever.
    job.locked_at = now
    job.updated_at = now
    await db.flush()
    return job


async def complete(db: AsyncSession, job_id: uuid.UUID, output_data: Optional[dict] = None) -> Optional[Job]:
    job = await db.get(Job, job_id)
    if job is None:
        return None
    now = datetime.utcnow()
    job.status = JobStatus.COMPLETED
    job.progress = 100
    if output_data is not None:
        job.output_data = output_data
    job.completed_at = now
    job.updated_at = now
    job.locked_by = None
    job.locked_at = None
    await db.flush()
    return job


async def fail(
    db: AsyncSession, job_id: uuid.UUID, error: str, retry: bool = True
) -> Optional[Job]:
    """Fail a job; retry re-queues until max_attempts, then terminal failed."""
    job = await db.get(Job, job_id)
    if job is None:
        return None
    now = datetime.utcnow()
    job.attempts = int(job.attempts or 0) + 1
    if retry and job.attempts < int(job.max_attempts or 3):
        job.status = JobStatus.PENDING
        job.progress = 0
        job.locked_by = None
        job.locked_at = None
        job.error_message = f"Retry {job.attempts}/{job.max_attempts}: {error}"
        job.run_after = now  # immediate requeue; backoff is a Phase 2 tweak
    else:
        job.status = JobStatus.FAILED
        job.error_message = error
        job.completed_at = now
        job.locked_by = None
        job.locked_at = None
    job.updated_at = now
    await db.flush()
    return job


async def cancel(db: AsyncSession, job_id: uuid.UUID) -> Job:
    job = await db.get(Job, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        raise ValueError(f"Cannot cancel job in {job.status.value} state")
    job.status = JobStatus.CANCELLED
    job.error_message = "Cancelled by user"
    job.completed_at = datetime.utcnow()
    job.updated_at = datetime.utcnow()
    job.locked_by = None
    job.locked_at = None
    await db.flush()
    return job


async def reclaim_stale(db: AsyncSession, timeout_seconds: Optional[int] = None) -> int:
    """Release PROCESSING rows whose lock is older than the timeout."""
    cutoff = datetime.utcnow() - timedelta(seconds=timeout_seconds or _lock_timeout())
    result = await db.execute(
        select(Job).where(
            and_(Job.status == JobStatus.PROCESSING, Job.locked_at.is_not(None), Job.locked_at < cutoff)
        )
    )
    count = 0
    for job in result.scalars().all():
        job.status = JobStatus.PENDING
        job.locked_by = None
        job.locked_at = None
        job.updated_at = datetime.utcnow()
        count += 1
    if count:
        await db.flush()
    return count


async def latest_for_document(db: AsyncSession, document_id: uuid.UUID) -> Optional[Job]:
    result = await db.execute(
        select(Job)
        .where(Job.document_id == document_id)
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def queue_depth(db: AsyncSession) -> dict[str, int]:
    rows = (
        await db.execute(
            select(Job.status, func.count()).group_by(Job.status)
        )
    ).all()
    depth = {"pending": 0, "processing": 0, "completed": 0, "failed": 0, "cancelled": 0}
    for status_val, count in rows:
        key = getattr(status_val, "value", str(status_val))
        depth[key] = int(count)
    return depth


async def pending_count(db: AsyncSession, job_type: JobType | str | None = None) -> int:
    """Count PENDING jobs, optionally narrowed to one job type.

    Used by the worker's fairness rule: while any `document_processing` job is
    pending, at most one notes/academic job may hold a slot.
    """
    conditions = [Job.status == JobStatus.PENDING]
    if job_type is not None:
        raw = getattr(job_type, "value", job_type)
        conditions.append(Job.job_type == JobType(str(raw)))
    value = (
        await db.execute(select(func.count()).select_from(Job).where(and_(*conditions)))
    ).scalar()
    return int(value or 0)


async def release(db: AsyncSession, job_id: uuid.UUID, worker_id: str) -> Optional[Job]:
    """Return a claimed-but-not-started job to PENDING (worker fairness rule).

    Only the owning worker may release; a job already running (or terminal)
    is left untouched.
    """
    job = await db.get(Job, job_id)
    if job is None or job.locked_by != worker_id or job.status != JobStatus.PROCESSING:
        return None
    job.status = JobStatus.PENDING
    job.locked_by = None
    job.locked_at = None
    job.updated_at = datetime.utcnow()
    await db.flush()
    return job


__all__ = [
    "PRIORITY_DOCUMENT_PROCESSING",
    "PRIORITY_STRUCTURE_REPROCESS",
    "PRIORITY_ACADEMIC",
    "PRIORITY_DEFAULT",
    "enqueue",
    "find_by_idempotency_key",
    "claim",
    "heartbeat",
    "report_progress",
    "complete",
    "fail",
    "cancel",
    "reclaim_stale",
    "latest_for_document",
    "queue_depth",
    "pending_count",
    "release",
]
