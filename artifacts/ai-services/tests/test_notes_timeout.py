"""Per-topic notes generation bounds: timeout, bounded concurrency, order.

Offline and deterministic: no network, no Postgres, no providers. The
per-topic map seam (``_generate_topic_notes_bounded``) is exercised with stub
generators so a slow provider (free router) cannot hang a notes job.
"""

import asyncio

import pytest

from services.academic import orchestrator
from services.academic.types import (
    DocumentStructure,
    ExtractedTopic,
    NotesGenerationConfig,
    StructuredUnit,
    TopicCitation,
    TopicNotes,
    TopicShape,
)

# Async tests are marked individually (asyncio_mode = strict).

CID = {
    "T1": "11111111-2222-4333-8444-555555555501",
    "T2": "11111111-2222-4333-8444-555555555502",
    "T3": "11111111-2222-4333-8444-555555555503",
    "T4": "11111111-2222-4333-8444-555555555504",
}


def _structure():
    return DocumentStructure(
        doc_id="doc-notes-timeout",
        title="Bounds",
        units=[
            StructuredUnit(title="U1", topics=[
                ExtractedTopic(title="T1", chunks=[CID["T1"]]),
                ExtractedTopic(title="T2", chunks=[CID["T2"]]),
            ]),
            StructuredUnit(title="U2", topics=[
                ExtractedTopic(title="T3", chunks=[CID["T3"]]),
                ExtractedTopic(title="T4", chunks=[CID["T4"]]),
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


def _good_note(unit_title, topic):
    return TopicNotes(
        unit_title=unit_title,
        topic_title=topic.title,
        heading=topic.title,
        body_paragraphs=[f"Body for {topic.title}."],
        key_terms=[],
        shape=TopicShape.CONCEPT,
        citations=[TopicCitation(chunk_id=CID[topic.title], span=f"Body for {topic.title}")],
        provenance_note="stub per-topic notes",
    )


# ---------------------------------------------------------------------------
# 1. A slow per-topic call becomes an honest gap; every other topic survives.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_slow_topic_times_out_into_gap_and_job_still_completes():
    structure = _structure()
    groups = _groups(structure)

    async def slow_generate(unit_title, topic):
        await asyncio.sleep(5)
        return _good_note(unit_title, topic)

    ordered, gaps = await orchestrator._generate_topic_notes_bounded(
        groups,
        structure,
        slow_generate,
        timeout_s=1.0,
        concurrency=2,
    )

    # The stage completes and every topic is still represented.
    assert len(ordered) == 4
    assert [note.topic_title for note in ordered] == ["T1", "T2", "T3", "T4"]
    # A timeout is an honest gap, never a fabricated note and never a hang.
    assert len(gaps) == 4
    assert all("timed out" in gap["reason"] for gap in gaps)
    for note in ordered:
        assert note.no_source is True
        assert note.sufficiency == "INSUFFICIENT"
        assert "timed out" in (note.source_gap_reason or "")
        assert note.provenance_note


# ---------------------------------------------------------------------------
# 2. Bounded concurrency preserves spine order, not completion order.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrency_preserves_spine_order():
    structure = _structure()
    groups = _groups(structure)
    # T1 finishes LAST: if the map returned completion order the titles would
    # come back T2, T3, T4, T1 instead of the spine order T1..T4.
    delays = {"T1": 0.15, "T2": 0.05, "T3": 0.02, "T4": 0.01}
    peak = {"in_flight": 0, "max": 0}

    async def ordered_generate(unit_title, topic):
        peak["in_flight"] += 1
        peak["max"] = max(peak["max"], peak["in_flight"])
        try:
            await asyncio.sleep(delays[topic.title])
            return _good_note(unit_title, topic)
        finally:
            peak["in_flight"] -= 1

    ordered, gaps = await orchestrator._generate_topic_notes_bounded(
        groups,
        structure,
        ordered_generate,
        timeout_s=30.0,
        concurrency=2,
    )

    assert gaps == []
    assert [note.topic_title for note in ordered] == ["T1", "T2", "T3", "T4"]
    # The semaphore really bounded the pool at two.
    assert peak["max"] == 2

    # And the final synthesized artifact is spine-ordered too.
    chunk_texts = {CID[title]: f"Body for {title}." for title in CID}
    artifact = orchestrator.synthesize_notes(
        "Bounds", structure, ordered, chunk_texts=chunk_texts
    )
    assert [section.topic_title for section in artifact.sections] == [
        "T1", "T2", "T3", "T4",
    ]


# ---------------------------------------------------------------------------
# 3. Env clamps and the optional config fields.
# ---------------------------------------------------------------------------

def test_topic_timeout_env_clamps(monkeypatch):
    monkeypatch.setenv("NOTES_PER_TOPIC_TIMEOUT_S", "5")
    assert orchestrator._topic_timeout_s() == 30.0
    monkeypatch.setenv("NOTES_PER_TOPIC_TIMEOUT_S", "9999")
    assert orchestrator._topic_timeout_s() == 1800.0
    monkeypatch.setenv("NOTES_PER_TOPIC_TIMEOUT_S", "not-a-number")
    assert orchestrator._topic_timeout_s() == orchestrator.DEFAULT_PER_TOPIC_TIMEOUT_S
    monkeypatch.delenv("NOTES_PER_TOPIC_TIMEOUT_S", raising=False)
    assert orchestrator._topic_timeout_s() == orchestrator.DEFAULT_PER_TOPIC_TIMEOUT_S


def test_topic_concurrency_env_clamps(monkeypatch):
    monkeypatch.setenv("NOTES_TOPIC_CONCURRENCY", "0")
    assert orchestrator._topic_concurrency() == 1
    monkeypatch.setenv("NOTES_TOPIC_CONCURRENCY", "99")
    assert orchestrator._topic_concurrency() == 4
    monkeypatch.setenv("NOTES_TOPIC_CONCURRENCY", "nope")
    assert orchestrator._topic_concurrency() == orchestrator.DEFAULT_TOPIC_CONCURRENCY
    monkeypatch.delenv("NOTES_TOPIC_CONCURRENCY", raising=False)
    assert orchestrator._topic_concurrency() == orchestrator.DEFAULT_TOPIC_CONCURRENCY


def test_config_fields_are_optional_and_additive():
    default = NotesGenerationConfig()
    assert default.topic_timeout_s is None
    assert default.topic_concurrency is None
    configured = NotesGenerationConfig(topic_timeout_s=42.0, topic_concurrency=3)
    assert configured.topic_timeout_s == 42.0
    assert configured.topic_concurrency == 3
