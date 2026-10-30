"""Notes validation policy: advisory overlap, spine labels, bounds, merging.

These tests pin the five local fixes for the live ``notes_hierarchical`` job
that hard-failed synthesis on ``topic_mismatch`` even though bodies, citations
and coverage all passed.  All deterministic: no network, no Postgres.
"""

import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from services.academic import orchestrator
from services.academic.extraction import merge_adjacent_topics
from services.academic.orchestrator import _payload_to_topic_notes
from services.academic.types import (
    DocumentStructure,
    ExtractedTopic,
    StructuredUnit,
)
from services.academic.validation import (
    validate_notes_artifact,
    validate_topic_notes,
)


def _structure_with(topic: str) -> dict:
    return {"units": [{"title": "U", "topics": [{"title": topic}]}]}


# ---------------------------------------------------------------------------
# FIX 1: label/content overlap is advisory, never a hard failure.
# ---------------------------------------------------------------------------


def test_artifact_low_overlap_is_warning_and_ok():
    artifact = {
        "sections": [{
            "unit_title": "U",
            # A real extractor label the model body will never repeat.
            "topic_title": "Answer True and Fals",
            "heading": "Answer True and Fals",
            "content": (
                "A binary decision records one of two possible states, "
                "carrying one unit of information."
            ),
            "citations": [{"chunk_id": "c1", "span": "binary decision"}],
        }],
        "source_gaps": [],
    }
    report = validate_notes_artifact(
        artifact,
        structure=_structure_with("Answer True and Fals"),
        allowed_chunk_ids={"c1"},
        chunk_texts={"c1": "binary decision records one of two possible states"},
    )
    assert report.ok is True
    assert any(w["code"] == "topic_overlap_low" for w in report.warnings)
    assert not any(e["code"] == "topic_mismatch" for e in report.errors)


def test_topic_validator_low_overlap_is_warning_and_ok():
    note = {
        "unit_title": "U",
        "topic_title": "Mode",
        "heading": "Mode",
        "body_paragraphs": [
            "This paragraph explains a completely different idea in enough detail."
        ],
        "citations": [{"chunk_id": "c1", "span": "different"}],
        "no_source": False,
        "provenance_note": "test",
    }
    report = validate_topic_notes(note, topic={"title": "Mode"})
    assert report.ok is True
    assert any(w["code"] == "topic_overlap_low" for w in report.warnings)


def test_short_section_still_hard_fails():
    artifact = {"sections": [{
        "unit_title": "U", "topic_title": "Cells", "heading": "Cells",
        "content": "Cells.",
        "citations": [{"chunk_id": "c1", "span": "Cells"}],
    }]}
    report = validate_notes_artifact(
        artifact, structure=_structure_with("Cells"), allowed_chunk_ids={"c1"}
    )
    assert report.ok is False
    assert any(e["code"] == "section_too_short" for e in report.errors)


def test_placeholder_section_still_hard_fails():
    artifact = {"sections": [{
        "unit_title": "U", "topic_title": "Cells", "heading": "Cells",
        "content": "No available source material for this topic right now.",
        "citations": [{"chunk_id": "c1", "span": "Cells"}],
    }]}
    report = validate_notes_artifact(
        artifact, structure=_structure_with("Cells"), allowed_chunk_ids={"c1"}
    )
    assert report.ok is False
    assert any(e["code"] == "placeholder_section" for e in report.errors)


def test_missing_citations_still_hard_fails():
    artifact = {"sections": [{
        "unit_title": "U", "topic_title": "Cells", "heading": "Cells",
        "content": (
            "Cells are the smallest structural unit of every living organism."
        ),
        "citations": [],
    }]}
    report = validate_notes_artifact(
        artifact, structure=_structure_with("Cells"), allowed_chunk_ids={"c1"}
    )
    assert report.ok is False
    assert any(e["code"] == "section_missing_citations" for e in report.errors)


def test_invalid_citation_still_hard_fails():
    artifact = {"sections": [{
        "unit_title": "U", "topic_title": "Cells", "heading": "Cells",
        "content": (
            "Cells are the smallest structural unit of every living organism."
        ),
        "citations": [{"chunk_id": "foreign", "span": "Cells"}],
    }]}
    report = validate_notes_artifact(
        artifact, structure=_structure_with("Cells"), allowed_chunk_ids={"c1"}
    )
    assert report.ok is False
    assert any(e["code"] == "invalid_citation" for e in report.errors)


def test_long_body_with_unavailable_phrase_is_not_placeholder():
    from services.academic.validation import is_placeholder

    body = (
        "Comments are lines the interpreter ignores when running a program. " * 4
        + "Multiline comments are not available in python but we can achieve "
        "them by using a doc string instead. "
        + "This is a normal explanatory sentence about the topic. " * 4
    )
    assert is_placeholder(body) is False


def test_short_not_available_is_still_a_placeholder():
    from services.academic.validation import is_placeholder

    assert is_placeholder("This content is not available yet.") is True


def test_folded_topic_is_named_as_gap_not_silently_dropped():
    from services.academic.notes_hier import synthesize_notes
    from services.academic.types import (
        ExtractedTopic,
        StructuredUnit,
        TopicCitation,
        TopicNotes,
        TopicShape,
    )

    structure = DocumentStructure(
        doc_id="d", title="T",
        units=[StructuredUnit(title="U", topics=[
            ExtractedTopic(title="A", chunks=["c1"]),
            ExtractedTopic(title="B", chunks=["c2"]),
        ])],
        fingerprint="f" * 64,
    )
    shared = "This is an identical body paragraph about the same topic for both sections."
    notes = [
        TopicNotes(
            unit_title="U", topic_title=title, heading=title,
            body_paragraphs=[shared],
            citations=[TopicCitation(chunk_id=cid, span="identical body")],
            shape=TopicShape.CONCEPT, provenance_note="test",
        )
        for title, cid in (("A", "c1"), ("B", "c2"))
    ]
    artifact = synthesize_notes(
        "T", structure, notes, allowed_chunk_ids={"c1", "c2"}
    )
    assert [s.topic_title for s in artifact.sections] == ["A"]
    assert any("U::B" in gap for gap in artifact.source_gaps)
    report = validate_notes_artifact(
        artifact, structure=structure, allowed_chunk_ids={"c1", "c2"}
    )
    assert report.ok is True



# ---------------------------------------------------------------------------
# FIX 2: the spine owns unit/topic/heading labels.
# ---------------------------------------------------------------------------


def test_payload_wrong_topic_title_is_corrected_to_spine():
    topic = ExtractedTopic(title="Spine Topic", chunks=["c1"])
    payload = {
        "unit_title": "Model Unit",
        "topic_title": "Model Title",
        "heading": "Model Heading",
        "body_paragraphs": ["A grounded body paragraph."],
        "citations": [{"chunk_id": "c1", "span": "grounded"}],
        "shape": "concept",
        "provenance_note": "test",
    }
    notes = _payload_to_topic_notes(payload, "Spine Unit", topic, "prov")
    assert notes.unit_title == "Spine Unit"
    assert notes.topic_title == "Spine Topic"
    assert notes.heading == "Spine Topic"
    assert notes.shape_extras["llm_topic_title"] == "Model Title"
    assert notes.shape_extras["llm_unit_title"] == "Model Unit"
    assert notes.shape_extras["llm_heading"] == "Model Heading"


def test_payload_topic_notes_instance_is_also_relabelled():
    from services.academic.types import TopicNotes, TopicShape

    topic = ExtractedTopic(title="Spine Topic", chunks=["c1"])
    payload = TopicNotes(
        unit_title="Model Unit",
        topic_title="Model Title",
        heading="Model Heading",
        body_paragraphs=["A grounded body paragraph."],
        citations=[{"chunk_id": "c1", "span": "grounded"}],
        shape=TopicShape.CONCEPT,
        provenance_note="test",
    )
    notes = _payload_to_topic_notes(payload, "Spine Unit", topic, "prov")
    assert notes.unit_title == "Spine Unit"
    assert notes.topic_title == "Spine Topic"
    assert notes.heading == "Spine Topic"


# ---------------------------------------------------------------------------
# FIX 3: the per-topic bound allows a slow provider to finish.
# ---------------------------------------------------------------------------


def test_topic_timeout_default_is_300(monkeypatch):
    monkeypatch.delenv("NOTES_PER_TOPIC_TIMEOUT_S", raising=False)
    assert orchestrator.DEFAULT_PER_TOPIC_TIMEOUT_S == 120.0
    assert orchestrator._topic_timeout_s() == 120.0


def test_topic_timeout_clamp_allows_1800(monkeypatch):
    monkeypatch.setenv("NOTES_PER_TOPIC_TIMEOUT_S", "9999")
    assert orchestrator._topic_timeout_s() == 1800.0
    monkeypatch.setenv("NOTES_PER_TOPIC_TIMEOUT_S", "600")
    assert orchestrator._topic_timeout_s() == 600.0


def test_max_topics_env_clamps(monkeypatch):
    monkeypatch.delenv("NOTES_MAX_TOPICS", raising=False)
    assert orchestrator.MAX_TOPICS_PER_JOB == 8
    assert orchestrator._max_topics_per_job() == 8
    monkeypatch.setenv("NOTES_MAX_TOPICS", "1")
    assert orchestrator._max_topics_per_job() == 4
    monkeypatch.setenv("NOTES_MAX_TOPICS", "99")
    assert orchestrator._max_topics_per_job() == 30
    monkeypatch.setenv("NOTES_MAX_TOPICS", "nope")
    assert orchestrator._max_topics_per_job() == 8


def test_gaps_provider_slow_detection():
    assert orchestrator._gaps_provider_slow(
        [{"topic": "X", "reason": "generation timed out after 300s"}]
    ) is True
    assert orchestrator._gaps_provider_slow(
        [{"topic": "X", "reason": "no source evidence"}]
    ) is False
    assert orchestrator._gaps_provider_slow([]) is False


# ---------------------------------------------------------------------------
# FIX 4: many tiny topics merge into bounded page-window topics.
# ---------------------------------------------------------------------------


def _many_topic_structure(count: int = 20) -> DocumentStructure:
    topics = [
        ExtractedTopic(title=f"Topic {i + 1}", pages=[i + 1], chunks=[f"c{i + 1}"])
        for i in range(count)
    ]
    return DocumentStructure(
        doc_id="doc-many",
        title="Many",
        units=[StructuredUnit(title="Unit 1", topics=topics)],
        fingerprint="f" * 64,
    )


def test_merge_reduces_to_limit_preserving_order_and_pages():
    structure = _many_topic_structure(20)
    merged = merge_adjacent_topics(structure, 12)
    topics = [t for unit in merged.units for t in unit.topics]
    assert len(topics) == 12
    # Nothing dropped: every original chunk survives.
    assert sum(len(t.chunks) for t in topics) == 20
    # Source order preserved (page sequence stays monotonic).
    pages = [p for t in topics for p in t.pages]
    assert pages == sorted(pages)
    # The first (smallest, earliest) pair merged carries its page span.
    first = merged.units[0].topics[0]
    assert "-" in first.title
    assert "(p. 1-2)" in first.title
    assert list(first.chunks) == ["c1", "c2"]


def test_merge_is_idempotent():
    structure = _many_topic_structure(20)
    merged = merge_adjacent_topics(structure, 12)
    again = merge_adjacent_topics(merged, 12)
    assert [t.title for u in again.units for t in u.topics] == [
        t.title for u in merged.units for t in u.topics
    ]


def test_merge_noop_when_within_limit():
    structure = _many_topic_structure(5)
    assert merge_adjacent_topics(structure, 12) is structure


def test_merge_keeps_union_of_pages_chunks_concepts():
    left = ExtractedTopic(title="A", pages=[1, 2], chunks=["c1", "c2"])
    right = ExtractedTopic(title="B", pages=[3], chunks=["c3"])
    merged = merge_adjacent_topics(
        DocumentStructure(
            doc_id="d", title="T",
            units=[StructuredUnit(title="U", topics=[left, right])],
            fingerprint="f" * 64,
        ),
        1,
    )
    topic = merged.units[0].topics[0]
    assert topic.pages == [1, 2, 3]
    assert topic.chunks == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# FIX 5: the read payload surfaces partial success honestly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hierarchical_read_surfaces_gaps_and_provider_slow():
    from services.academic import routes

    project = SimpleNamespace(id=uuid.uuid4())
    job = SimpleNamespace(
        id=uuid.uuid4(),
        job_type="notes_hierarchical",
        status="completed",
        error_message=None,
        stage="pdf",
        progress=100,
        output_data={
            "academic": {
                "gaps": [
                    {"topic": "Data Structures",
                     "reason": "generation timed out after 300s"}
                ],
                "failed_topic_count": 1,
                "provider_slow": True,
            }
        },
    )

    async def _jobs(*args, **kwargs):
        return [job]

    with patch("services.jobs.service.job_manager.get_jobs_for_project", _jobs):
        result = await routes.read_hierarchical_notes(
            project.id, project=project, db=object()
        )

    assert result["state"] == "ready"
    assert result["failed_topic_count"] == 1
    assert result["provider_slow"] is True
    assert result["gaps"][0]["topic"] == "Data Structures"
