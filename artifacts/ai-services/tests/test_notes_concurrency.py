"""Per-topic notes concurrency: one AsyncSession, one DB lock.

Reproduces the live-path blocker: with ``NOTES_TOPIC_CONCURRENCY=2`` and
retrieval ENABLED, the per-topic evidence fetches and the collecting loop's
progress commits all touch ONE AsyncSession.  SQLAlchemy rejects that with
``InvalidRequestError: concurrent operations are not permitted on this
AsyncSession``.

The production ``_db_lock`` must serialise every DB-touching await (retrieval
in ``generate_topic_live_or_deterministic`` and the progress commit in
``_report``) while the LLM calls stay overlapped.

Offline and deterministic: the retrieval service and the LLM helper are stubs,
so no network and no Postgres are required.  A probe counts how many
coroutines are inside a DB-touching region at once; it must never exceed 1.
"""

import asyncio
import uuid
from types import SimpleNamespace

import pytest

from services.academic import orchestrator


# ---------------------------------------------------------------------------
# Stub data and probes
# ---------------------------------------------------------------------------

_CHUNK_TEXT = {
    "c1": (
        "Alpha evidence paragraph one contains a long enough explanation of the "
        "alpha concept for the notes validation gate."
    ),
    "c2": (
        "Beta evidence paragraph one contains a long enough explanation of the "
        "beta concept for the notes validation gate."
    ),
    "c3": (
        "Gamma evidence paragraph one contains a long enough explanation of the "
        "gamma concept for the notes validation gate."
    ),
}
_CHUNK_ROWS = [
    {"id": cid, "chunk_id": cid, "text": text, "page": 1, "position": index}
    for index, (cid, text) in enumerate(_CHUNK_TEXT.items())
]
_TOPIC_CHUNK = {"Alpha": "c1", "Beta": "c2", "Gamma": "c3"}
_FAIL_TITLE = "Quarantine ZZZ"


class _Probe:
    """Max-in-flight counter for one class of region (DB or LLM)."""

    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0

    def enter(self):
        self.in_flight += 1
        if self.in_flight > self.max_in_flight:
            self.max_in_flight = self.in_flight

    def exit(self):
        self.in_flight -= 1


class _FakeDB:
    """Only the session surface the notes job actually awaits."""

    def __init__(self, probe):
        self.probe = probe
        self.commits = 0

    async def commit(self):
        self.commits += 1
        self.probe.enter()
        try:
            await asyncio.sleep(0.01)
        finally:
            self.probe.exit()

    async def rollback(self):
        self.probe.enter()
        try:
            await asyncio.sleep(0)
        finally:
            self.probe.exit()

    async def flush(self):
        return None


def _structure():
    from services.academic.types import DocumentStructure, ExtractedTopic, StructuredUnit

    return DocumentStructure(
        doc_id="doc-notes-concurrency",
        title="Concurrency",
        units=[
            StructuredUnit(title="U1", topics=[
                ExtractedTopic(title="Alpha", chunks=["c1"]),
                ExtractedTopic(title="Beta", chunks=["c2"]),
                ExtractedTopic(title="Gamma", chunks=["c3"]),
            ]),
            # Own unit with no claimed chunks, so a failing evidence fetch
            # cannot be masked by unit-level recovery.
            StructuredUnit(title="U2", topics=[
                ExtractedTopic(title=_FAIL_TITLE, chunks=[]),
            ]),
        ],
        fingerprint="f" * 64,
    )


def _groups(structure):
    return [
        {"unit_title": unit.title, "topic_title": topic.title}
        for unit in structure.units
        for topic in unit.topics
    ]


def _install_retrieval(monkeypatch, probe, fail_titles=(_FAIL_TITLE,)):
    """Stub evidence fetcher: records the DB region and sleeps to expose races."""
    from services.retrieval import service as retrieval_module

    async def fake_retrieve(topic_title, db, **kwargs):
        probe.enter()
        try:
            await asyncio.sleep(0.05)
            if topic_title in fail_titles:
                raise RuntimeError("evidence index unavailable")
            cid = _TOPIC_CHUNK.get(topic_title)
            rows = [row for row in _CHUNK_ROWS if row["id"] == cid]
            return SimpleNamespace(chunks=rows)
        finally:
            probe.exit()

    monkeypatch.setattr(
        retrieval_module.retrieval_service,
        "retrieve_topic_evidence",
        fake_retrieve,
    )


def _install_llm(monkeypatch, probe):
    """Stub LLM helper: records the (unlocked) LLM region and sleeps."""
    from services.academic import notes_hier

    async def fake_generate(
        engine, unit_title, topic, context, allowed, query=None, max_tokens=None
    ):
        probe.enter()
        try:
            await asyncio.sleep(0.12)
            cid = sorted(allowed)[0]
            body = _CHUNK_TEXT[cid]
            return {
                "body_paragraphs": [body],
                "citations": [{"chunk_id": cid, "span": body}],
                "key_terms": [],
                "shape": "concept",
            }
        finally:
            probe.exit()

    monkeypatch.setattr(notes_hier, "generate_topic_notes", fake_generate)


def _install_progress(monkeypatch, probe):
    """Stub the queue's progress write so the commit path stays offline."""
    from services.jobs import queue as job_queue

    async def fake_report_progress(
        db, job_id, progress, stage=None, stage_message=None
    ):
        probe.enter()
        try:
            await asyncio.sleep(0.01)
        finally:
            probe.exit()

    monkeypatch.setattr(job_queue, "report_progress", fake_report_progress)


def _fake_job(project_id=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        input_data={
            "project_id": str(project_id or uuid.uuid4()),
            "use_live": True,
            "use_retrieval": True,
        },
        output_data={},
        progress=0,
        stage="",
        stage_message="",
    )


# ---------------------------------------------------------------------------
# 1. DB regions never overlap; LLM calls do.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_topics_serialize_db_and_overlap_llm(monkeypatch):
    monkeypatch.setenv("NOTES_TOPIC_CONCURRENCY", "2")
    structure = _structure()
    groups = _groups(structure)
    db_probe = _Probe()
    llm_probe = _Probe()
    _install_retrieval(monkeypatch, db_probe)
    _install_llm(monkeypatch, llm_probe)
    _install_progress(monkeypatch, db_probe)

    db = _FakeDB(db_probe)
    job = _fake_job()
    state = {}
    done = set()

    async def _generate(unit_title, topic):
        return await orchestrator._generate_topic_with_honest_error(
            db,
            "project-1",
            object(),  # engine is ignored by the stubbed LLM helper
            structure,
            unit_title,
            topic,
            _CHUNK_ROWS,
            use_live=True,
            use_retrieval=True,
        )

    async def _on_complete(index, note):
        done.add("per_group")
        await orchestrator._report(
            db, job, "per_group", 0.5, note.topic_title, state, done
        )

    ordered, gaps = await orchestrator._generate_topic_notes_bounded(
        groups,
        structure,
        _generate,
        timeout_s=30.0,
        concurrency=2,
        on_complete=_on_complete,
    )

    # The map completes in spine order with no timeouts.
    assert [note.topic_title for note in ordered] == [
        "Alpha", "Beta", "Gamma", _FAIL_TITLE,
    ]
    assert gaps == []
    # Retrieval + the collecting-loop commit share ONE locked DB region.
    assert db_probe.max_in_flight == 1
    # ...but the LLM calls were genuinely overlapped (the lock is not global).
    assert llm_probe.max_in_flight >= 2
    # Progress was reported for every completed topic.
    assert job.progress > 0


# ---------------------------------------------------------------------------
# 2. A failing evidence fetch becomes an honest source gap; the map completes.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evidence_fetch_failure_becomes_honest_gap(monkeypatch):
    monkeypatch.setenv("NOTES_TOPIC_CONCURRENCY", "2")
    structure = _structure()
    groups = _groups(structure)
    db_probe = _Probe()
    llm_probe = _Probe()
    _install_retrieval(monkeypatch, db_probe)  # _FAIL_TITLE raises
    _install_llm(monkeypatch, llm_probe)
    _install_progress(monkeypatch, db_probe)

    db = _FakeDB(db_probe)
    job = _fake_job()
    state = {}
    done = set()

    async def _generate(unit_title, topic):
        return await orchestrator._generate_topic_with_honest_error(
            db,
            "project-1",
            object(),
            structure,
            unit_title,
            topic,
            _CHUNK_ROWS,
            use_live=True,
            use_retrieval=True,
        )

    async def _on_complete(index, note):
        await orchestrator._report(
            db, job, "per_group", 0.5, note.topic_title, state, done
        )

    ordered, gaps = await orchestrator._generate_topic_notes_bounded(
        groups,
        structure,
        _generate,
        timeout_s=30.0,
        concurrency=2,
        on_complete=_on_complete,
    )

    # No exception escaped: the map completed every topic.
    assert {note.topic_title for note in ordered} == {
        "Alpha", "Beta", "Gamma", _FAIL_TITLE,
    }
    # The topic whose evidence fetch raised is an honest gap, never invented.
    failed = next(note for note in ordered if note.topic_title == _FAIL_TITLE)
    assert failed.no_source is True
    assert failed.citations == []
    assert failed.body_paragraphs == []
    assert failed.source_gap_reason
    # The healthy topics still carry cited prose (the failure was isolated).
    healthy = next(note for note in ordered if note.topic_title == "Alpha")
    assert healthy.no_source is False
    assert healthy.citations
    assert db_probe.max_in_flight == 1


# ---------------------------------------------------------------------------
# 3. The real job handler runs retrieval-enabled without a concurrent session.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_hierarchical_job_retrieval_enabled_serializes_session(
    monkeypatch,
):
    from services.jobs import queue as job_queue
    from services.academic import notes_hier

    monkeypatch.setenv("NOTES_TOPIC_CONCURRENCY", "2")
    structure = _structure()
    project_id = uuid.uuid4()
    db_probe = _Probe()
    llm_probe = _Probe()
    _install_retrieval(monkeypatch, db_probe)
    _install_llm(monkeypatch, llm_probe)
    _install_progress(monkeypatch, db_probe)

    # Keep the stage machine offline but faithful: only the persistence and PDF
    # edges are stubbed, the per-topic concurrency path is production code.
    async def fake_report_progress(
        db, job_id, progress, stage=None, stage_message=None
    ):
        db_probe.enter()
        try:
            await asyncio.sleep(0.01)
        finally:
            db_probe.exit()

    completed = {}

    async def fake_complete(db, job_id, output_data=None):
        completed["output"] = output_data
        return None

    monkeypatch.setattr(job_queue, "report_progress", fake_report_progress)
    monkeypatch.setattr(job_queue, "complete", fake_complete)

    async def fake_spine(*args, **kwargs):
        return orchestrator.SpineBuildResult(
            structure=structure,
            source_structures=[structure],
            source_ids=["doc-1"],
            source_fingerprint="f" * 64,
            model_version=1,
            root_ids=["root-1"],
            chunks=list(_CHUNK_ROWS),
            reused=False,
            model_header_id="header-1",
        )

    monkeypatch.setattr(orchestrator, "build_project_spine", fake_spine)

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(orchestrator, "_save_notes_payload", _noop)
    monkeypatch.setattr(orchestrator, "_persist_compat_notes", _noop)

    async def fake_pdf(*args, **kwargs):
        return {
            "id": "pdf-1",
            "path": "user/project/study-notes-v1.pdf",
            "version": 1,
            "pages": 3,
            "sha256": "a" * 64,
            "hash": "a" * 64,
            "model": "academic-notes-hierarchical",
            "status": "ready",
            "created_at": "2026-10-06T00:00:00Z",
            "reused": False,
        }

    monkeypatch.setattr(orchestrator, "build_and_store_pdf", fake_pdf)

    failures = []

    async def _capture_failure(ctx, job_id, error):
        failures.append(error)

    monkeypatch.setattr(orchestrator, "_record_hierarchical_failure", _capture_failure)

    db = _FakeDB(db_probe)
    ctx = SimpleNamespace(db=db)
    job = _fake_job(project_id=project_id)

    await orchestrator.process_hierarchical_job(ctx, job)

    state = dict((completed.get("output") or {}).get("academic") or {})
    assert failures == [], failures
    assert state.get("state") == "ready"
    assert state.get("pdf_ready") is True
    assert db_probe.max_in_flight == 1
    assert llm_probe.max_in_flight >= 2
    assert len(state.get("topic_notes") or []) == 4
