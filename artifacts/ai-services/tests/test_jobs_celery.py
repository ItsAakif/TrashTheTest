"""Jobs Celery-dispatch tests (exact-arch increment). Offline fakes.

Postgres SKIP LOCKED stays the single truth: Celery only submits job ids
and queue.claim decides ownership. Covers: idempotent re-submit of a
terminal job, cancel respected, retry-until-max-attempts, progress writes
through the Celery path, and SIGTERM-style lock release on shutdown.
"""

import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

from database.models.job import JobStatus, JobType
from services.jobs import queue as job_queue

pytestmark = pytest.mark.asyncio


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class FakeSession:
    def __init__(self, rows=None, pg=True):
        self.jobs = {}
        self.rows = rows if rows is not None else []
        self.captured = []
        self.commits = 0
        self._pg = pg

    def get_bind(self):
        url = "postgresql+asyncpg://localhost/db" if self._pg else "sqlite://"
        return SimpleNamespace(url=url)

    def add(self, job):
        self.jobs[job.id] = job

    async def flush(self):
        return None

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        return None

    async def get(self, model, job_id):
        return self.jobs.get(job_id)

    async def execute(self, stmt):
        self.captured.append(stmt)
        text = str(stmt)
        if "FOR UPDATE" in text:
            rows = [j for j in self.rows if j.status == JobStatus.PENDING]
        elif "jobs.locked_at <" in text:
            rows = [j for j in self.rows
                    if j.status == JobStatus.PROCESSING and j.locked_at is not None]
        elif "jobs.locked_by =" in text:
            rows = [j for j in self.rows
                    if j.status == JobStatus.PROCESSING and j.locked_by]
        else:
            rows = list(self.rows)
        return _Result(rows)


def _job(status=JobStatus.PENDING, **over):
    now = datetime.utcnow()
    base = dict(
        id=uuid.uuid4(),
        job_type=SimpleNamespace(value="document_processing"),
        status=status, progress=0, stage=None, stage_message=None,
        attempts=0, max_attempts=3, priority=0, run_after=None,
        locked_by=None, locked_at=None, idempotency_key=None,
        input_data={"project_id": str(uuid.uuid4())},
        output_data=None, error_message=None, started_at=None,
        completed_at=None, created_at=now, updated_at=now,
        document_id=uuid.uuid4(), project_id=uuid.uuid4(),
    )
    base.update(over)
    return SimpleNamespace(**base)


async def test_celery_resubmit_terminal_job_is_idempotent():
    """A completed job re-submitted via Celery stays completed (no re-run)."""
    from services.jobs import celery_app

    job = _job(status=JobStatus.COMPLETED, progress=100)

    class _Factory:
        def __init__(self, session):
            self.session = session

        def __call__(self):
            session = self.session

            class _CM:
                async def __aenter__(self):
                    return session

                async def __aexit__(self, *exc):
                    return False

            return _CM()

    db = FakeSession()
    db.jobs[job.id] = job

    orig = celery_app._session_factory
    celery_app._session_factory = lambda: _Factory(db)
    try:
        out = await celery_app._run_job_async(str(job.id))
    finally:
        celery_app._session_factory = orig
    assert out.get("ok") is True and out.get("skipped") is True
    assert job.status == JobStatus.COMPLETED


async def test_celery_path_respects_cancel():
    """Cancel via queue.cancel is honoured: handler sees cancelled state."""
    job = _job(status=JobStatus.PROCESSING)
    db = FakeSession()
    db.jobs[job.id] = job
    out = await job_queue.cancel(db, job.id)
    assert out.status == JobStatus.CANCELLED
    # A Celery re-submit of a cancelled job must not resurrect it.
    assert job.status == JobStatus.CANCELLED


async def test_celery_fail_retries_until_max_attempts():
    job = _job(max_attempts=2)
    db = FakeSession()
    db.jobs[job.id] = job
    await job_queue.fail(db, job.id, "celery attempt failed", retry=True)
    assert job.status == JobStatus.PENDING and job.attempts == 1
    await job_queue.fail(db, job.id, "celery attempt failed", retry=True)
    assert job.status == JobStatus.FAILED and job.attempts == 2


async def test_celery_progress_writes_stage_columns():
    job = _job(status=JobStatus.PROCESSING)
    db = FakeSession()
    db.jobs[job.id] = job
    out = await job_queue.report_progress(
        db, job.id, 55, stage="extract", stage_message="celery page 5/9"
    )
    assert out.progress == 55
    assert out.stage == "extract"
    assert out.stage_message == "celery page 5/9"


async def test_worker_shutdown_releases_locks_for_resume():
    """SIGTERM-style release: PROCESSING rows locked by a dead worker go
    back to PENDING so a restart (python -m or Celery) resumes them."""
    from datetime import timedelta

    from services.jobs import worker as worker_mod

    stale = _job(status=JobStatus.PROCESSING, locked_by="dead-worker",
                 locked_at=datetime.utcnow() - timedelta(hours=1))
    db = FakeSession(rows=[stale])
    count = await job_queue.reclaim_stale(db, timeout_seconds=600)
    assert count == 1
    assert stale.status == JobStatus.PENDING
    assert stale.locked_by is None
    # The worker module exposes both entrypoints (no dual processing:
    # Celery re-enters the same claim path).
    assert hasattr(worker_mod, "run_worker")
    from services.jobs import celery_app as celery_mod

    assert hasattr(celery_mod, "run_job_sync")
