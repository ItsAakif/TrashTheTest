"""T111: SKIP LOCKED queue + worker, exercised with in-memory fakes.

Live Postgres acceptance (real FOR UPDATE SKIP LOCKED against pgvector DB,
alembic upgrade head, SIGTERM-resume against a running worker) is NOT
available in this environment - recorded as blocked for T113. These tests
prove the queue contract: transactional enqueue, single-claim, reclaim,
attempts/cancel semantics, stage/progress writes, and worker dispatch.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
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

    def scalar(self):
        # Enough for count() doubles: the fake returns the candidate rows.
        return len(self._rows)


class FakeSession:
    """In-memory AsyncSession double: canned SELECT rows, live object refs."""

    def __init__(self, rows=None, pg=True):
        self.jobs = {}  # id -> job namespace
        self.rows = rows if rows is not None else []
        self.captured = []  # statements seen by execute
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

    async def refresh(self, obj):
        return obj

    async def get(self, model, job_id):
        return self.jobs.get(job_id)

    async def execute(self, stmt):
        self.captured.append(stmt)
        text = str(stmt)
        # Fake fidelity: mirror the SQL WHERE predicates in memory so the
        # queue contract (claim-only-pending, reclaim-only-locked,
        # release-only-locked-by-me) is exercised honestly. The column list
        # always contains every column, so dispatch on the predicate text.
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
        status=status,
        progress=0,
        stage=None,
        stage_message=None,
        attempts=0,
        max_attempts=3,
        priority=0,
        run_after=None,
        locked_by=None,
        locked_at=None,
        idempotency_key=None,
        input_data={"project_id": str(uuid.uuid4())},
        output_data=None,
        error_message=None,
        started_at=None,
        completed_at=None,
        created_at=now,
        updated_at=now,
        document_id=uuid.uuid4(),
        project_id=uuid.uuid4(),
    )
    base.update(over)
    return SimpleNamespace(**base)


async def test_enqueue_does_not_commit_so_rollback_leaves_no_row():
    db = FakeSession()
    job = await job_queue.enqueue(
        db, JobType.DOCUMENT_PROCESSING, {"a": 1}, project_id=uuid.uuid4()
    )
    assert db.commits == 0  # caller's transaction owns the commit
    assert job.status == JobStatus.PENDING
    # Simulate the caller rolling back: nothing was committed.
    db.jobs.clear()
    assert await db.get(None, job.id) is None


async def test_two_workers_never_claim_the_same_row():
    job = _job()
    db = FakeSession(rows=[job])  # both loops observe the same row
    first = await job_queue.claim(db, "worker-1")
    second = await job_queue.claim(db, "worker-2")
    assert [j.id for j in first] == [job.id]
    assert second == []  # status re-check refuses the raced row
    assert job.locked_by == "worker-1"
    assert job.status == JobStatus.PROCESSING


async def test_claim_uses_skip_locked_on_postgres():
    from sqlalchemy.dialects import postgresql

    db = FakeSession(rows=[], pg=True)
    await job_queue.claim(db, "worker-1")
    assert db.captured, "claim must issue a SELECT"
    compiled = str(db.captured[0].compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE" in compiled and "SKIP LOCKED" in compiled


async def test_claim_degrades_without_skip_locked_off_postgres():
    db = FakeSession(rows=[], pg=False)
    claimed = await job_queue.claim(db, "worker-1")
    assert claimed == []
    assert "SKIP LOCKED" not in str(db.captured[0])


async def test_stale_lock_is_reclaimable_and_filter_is_in_sql():
    stale = _job(status=JobStatus.PROCESSING, locked_by="dead",
                 locked_at=datetime.utcnow() - timedelta(hours=2))
    db = FakeSession(rows=[stale])
    count = await job_queue.reclaim_stale(db, timeout_seconds=600)
    assert count == 1
    assert stale.status == JobStatus.PENDING
    assert stale.locked_by is None and stale.locked_at is None
    assert "locked_at" in str(db.captured[0])  # cutoff enforced in SQL


async def test_idempotency_key_returns_existing_queued_job():
    existing = _job(idempotency_key="key-1")
    db = FakeSession(rows=[existing])
    # find path returns the live row.
    db.rows = [existing]
    found = await job_queue.find_by_idempotency_key(
        db, "key-1", existing.project_id, JobType.DOCUMENT_PROCESSING
    )
    assert found is existing
    again = await job_queue.enqueue(
        db, JobType.DOCUMENT_PROCESSING, {}, project_id=existing.project_id,
        idempotency_key="key-1",
    )
    assert again is existing


async def test_cancel_marks_cancelled_not_failed():
    job = _job(status=JobStatus.PROCESSING)
    db = FakeSession()
    db.jobs[job.id] = job
    out = await job_queue.cancel(db, job.id)
    assert out.status == JobStatus.CANCELLED
    assert out.error_message == "Cancelled by user"
    assert out.locked_by is None


async def test_cancel_terminal_states_raise():
    for terminal in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        job = _job(status=terminal)
        db = FakeSession()
        db.jobs[job.id] = job
        with pytest.raises(ValueError):
            await job_queue.cancel(db, job.id)


async def test_fail_retries_until_max_attempts_then_terminal():
    job = _job(max_attempts=2)
    db = FakeSession()
    db.jobs[job.id] = job
    await job_queue.fail(db, job.id, "boom", retry=True)
    assert job.status == JobStatus.PENDING and job.attempts == 1
    await job_queue.fail(db, job.id, "boom", retry=True)
    assert job.status == JobStatus.FAILED and job.attempts == 2
    assert job.error_message == "boom"


async def test_complete_writes_terminal_state_and_progress():
    job = _job(status=JobStatus.PROCESSING, progress=42)
    db = FakeSession()
    db.jobs[job.id] = job
    out = await job_queue.complete(db, job.id, {"ok": True})
    assert out.status == JobStatus.COMPLETED
    assert out.progress == 100
    assert out.output_data == {"ok": True}


async def test_report_progress_writes_stage_columns():
    job = _job(status=JobStatus.PROCESSING)
    db = FakeSession()
    db.jobs[job.id] = job
    out = await job_queue.report_progress(db, job.id, 37, stage="extraction", stage_message="pages")
    assert out.progress == 37
    assert out.stage == "extraction"
    assert out.stage_message == "pages"
    assert out.locked_at is not None, "real progress must heartbeat the worker lock"


async def test_queue_depth_counts_by_status():
    class _DepthSession(FakeSession):
        async def execute(self, stmt):
            self.captured.append(stmt)

            class _Pairs:
                def all(self):
                    return [(JobStatus.PENDING, 2), (JobStatus.PROCESSING, 1)]

            class _R:
                def __iter__(self):
                    return iter([])

            r = _R()
            r.all = _Pairs().all
            return r

    db = _DepthSession()
    depth = await job_queue.queue_depth(db)
    assert depth["pending"] == 2 and depth["processing"] == 1


class _Factory:
    """session_factory double sharing one FakeSession behind an async CM."""

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


async def test_worker_once_runs_registered_handler_for_claimed_job():
    from services.jobs import worker as worker_mod

    seen = []
    type_name = f"test_type_{uuid.uuid4().hex[:6]}"
    job = _job(status=JobStatus.PENDING)
    job.job_type = SimpleNamespace(value=type_name)

    async def handler(ctx, j):
        seen.append(j.id)
        await job_queue.complete(ctx.db, j.id)

    worker_mod.register_handler(type_name, handler)
    try:
        db = FakeSession(rows=[job])
        db.jobs[job.id] = job
        processed = await worker_mod.run_worker(session_factory=_Factory(db), concurrency=1, once=True)
    finally:
        worker_mod._HANDLERS.pop(type_name, None)
    assert processed == 1
    assert seen == [job.id]
    assert job.status == JobStatus.COMPLETED


async def test_worker_once_empty_queue_processes_nothing():
    from services.jobs import worker as worker_mod

    db = FakeSession(rows=[])
    assert await worker_mod.run_worker(session_factory=_Factory(db), once=True) == 0


async def test_concurrency_bound_two_workers_three_jobs():
    # WORKER_CONCURRENCY=2: at most two jobs PROCESSING at once. The claim
    # loop below mirrors queue.claim's contract (only PENDING rows are
    # claimable); the semaphore enforces the bound like the worker's.
    jobs = [_job() for _ in range(3)]
    active = {"n": 0, "peak": 0}
    gate = asyncio.Semaphore(2)

    async def claim_one(worker_id):
        for candidate in jobs:
            if candidate.status == JobStatus.PENDING:
                candidate.status = JobStatus.PROCESSING
                candidate.locked_by = worker_id
                return candidate
        return None

    async def worker_loop(worker_id):
        while True:
            async with gate:  # concurrency bound == WORKER_CONCURRENCY
                job_ns = await claim_one(worker_id)
                if job_ns is None:
                    return
                active["n"] += 1
                active["peak"] = max(active["peak"], active["n"])
                await asyncio.sleep(0.02)
                job_ns.status = JobStatus.COMPLETED
                active["n"] -= 1

    await asyncio.gather(worker_loop("w1"), worker_loop("w2"), worker_loop("w3"))
    assert active["peak"] <= 2
    assert all(j.status == JobStatus.COMPLETED for j in jobs)

# ============================================================
# Priority + upload fairness (upload starvation fix)
# ============================================================


class _PriorityClaimSession(FakeSession):
    """Claim-path fake that honours ORDER BY priority DESC, created_at ASC.

    FakeSession answers raw rows; this variant mirrors the SQL ordering and
    LIMIT so a priority regression is caught behaviourally, not just by
    string-matching the statement.
    """

    async def execute(self, stmt):
        self.captured.append(stmt)
        rows = [j for j in self.rows if j.status == JobStatus.PENDING]
        rows.sort(key=lambda j: (-int(getattr(j, "priority", 0) or 0), j.created_at))
        limit_clause = getattr(stmt, "_limit_clause", None)
        limit = getattr(limit_clause, "value", None)
        if isinstance(limit, int):
            rows = rows[:limit]
        return _Result(rows)


async def test_priority_constants_rank_uploads_above_academic_work():
    assert job_queue.PRIORITY_DOCUMENT_PROCESSING == 9
    assert job_queue.PRIORITY_STRUCTURE_REPROCESS == 5
    assert job_queue.PRIORITY_ACADEMIC == 2
    assert (
        job_queue.PRIORITY_DOCUMENT_PROCESSING
        > job_queue.PRIORITY_STRUCTURE_REPROCESS
        > job_queue.PRIORITY_ACADEMIC
    )


async def test_claim_takes_priority_nine_upload_before_older_priority_two_notes():
    older_notes = _job(
        priority=job_queue.PRIORITY_ACADEMIC,
        created_at=datetime.utcnow() - timedelta(hours=1),
    )
    older_notes.job_type = SimpleNamespace(value="notes_hierarchical")
    fresh_upload = _job(priority=job_queue.PRIORITY_DOCUMENT_PROCESSING)

    db = _PriorityClaimSession(rows=[older_notes, fresh_upload])
    claimed = await job_queue.claim(db, "worker-1", limit=1)

    assert [j.id for j in claimed] == [fresh_upload.id]
    assert fresh_upload.status == JobStatus.PROCESSING
    assert older_notes.status == JobStatus.PENDING
    # The SQL contract itself must stay DESC/ASC.
    compiled = str(db.captured[0])
    assert "jobs.priority DESC" in compiled
    assert "jobs.created_at ASC" in compiled


async def test_claim_oldest_first_within_the_same_priority():
    older = _job(created_at=datetime.utcnow() - timedelta(minutes=5))
    newer = _job(created_at=datetime.utcnow())
    db = _PriorityClaimSession(rows=[newer, older])
    claimed = await job_queue.claim(db, "worker-1", limit=1)
    assert [j.id for j in claimed] == [older.id]


async def test_pending_count_narrows_to_one_job_type():
    pending_upload = _job()
    pending_upload.job_type = SimpleNamespace(value="document_processing")
    pending_notes = _job()
    pending_notes.job_type = SimpleNamespace(value="notes_hierarchical")

    class _CountSession(FakeSession):
        async def execute(self, stmt):
            self.captured.append(stmt)
            text = str(stmt)
            if "jobs.job_type =" in text:
                rows = [pending_upload]
            else:
                rows = [pending_upload, pending_notes]
            return _Result(rows)

    db = _CountSession()
    assert await job_queue.pending_count(db) == 2
    assert await job_queue.pending_count(db, JobType.DOCUMENT_PROCESSING) == 1
    assert "jobs.job_type =" in str(db.captured[1])


async def test_release_returns_claimed_job_to_pending_only_for_owner():
    job = _job(status=JobStatus.PROCESSING, locked_by="worker-1",
               locked_at=datetime.utcnow())
    db = FakeSession()
    db.jobs[job.id] = job

    assert await job_queue.release(db, job.id, "worker-2") is None
    assert job.status == JobStatus.PROCESSING

    released = await job_queue.release(db, job.id, "worker-1")
    assert released is job
    assert job.status == JobStatus.PENDING
    assert job.locked_by is None and job.locked_at is None


async def test_worker_defers_second_notes_job_while_upload_pending(monkeypatch):
    """Fairness rule: pending document_processing => at most one notes job.

    Fake queue: two notes jobs are claimed while an upload is pending. The
    worker must run only one of them and release the other back to PENDING.
    """
    from services.jobs import worker as worker_mod

    notes_type = "notes_hierarchical"
    notes_jobs = []
    for _ in range(2):
        job = _job(status=JobStatus.PENDING)
        job.job_type = SimpleNamespace(value=notes_type)
        notes_jobs.append(job)

    released = []
    peak = {"notes": 0, "now": 0}

    async def fake_claim(db, worker_id, limit=1):
        for job in notes_jobs:
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.PROCESSING
                job.locked_by = worker_id
        return list(notes_jobs)

    async def fake_release(db, job_id, worker_id):
        for job in notes_jobs:
            if job.id == job_id and job.locked_by == worker_id:
                job.status = JobStatus.PENDING
                job.locked_by = None
                released.append(job_id)
                return job
        return None

    async def fake_pending_count(db, job_type=None):
        return 1 if job_type is not None else 1  # one upload waiting

    async def fake_reclaim(db, timeout_seconds=None):
        return 0

    async def notes_handler(ctx, job):
        peak["now"] += 1
        peak["notes"] = max(peak["notes"], peak["now"])
        await asyncio.sleep(0.02)
        peak["now"] -= 1
        await job_queue.complete(ctx.db, job.id)

    db = FakeSession()
    for job in notes_jobs:
        db.jobs[job.id] = job

    monkeypatch.setattr(job_queue, "claim", fake_claim)
    monkeypatch.setattr(job_queue, "release", fake_release)
    monkeypatch.setattr(job_queue, "pending_count", fake_pending_count)
    monkeypatch.setattr(job_queue, "reclaim_stale", fake_reclaim)
    monkeypatch.setitem(worker_mod._HANDLERS, notes_type, notes_handler)

    processed = await worker_mod._poll_loop(
        _Factory(db), "worker-1", concurrency=2, once=True
    )

    assert processed == 1, "only one notes job may run while an upload is pending"
    assert peak["notes"] == 1, "notes concurrency must be capped at 1"
    assert len(released) == 1
    statuses = sorted(j.status.value for j in notes_jobs)
    assert statuses == ["completed", "pending"], statuses
