"""Structure integrity regression suite (ADR-008)."""

import uuid
import pytest

pytestmark = pytest.mark.asyncio


async def test_chunker_never_emits_banned_fallback_headings():
    from services.chunking.chunker import SemanticChunker
    from services.chunking.types import DocumentChunkingInput

    # A single-line 5000-char document with no headings at all.
    text = ("This is a single very long paragraph with no newlines "
            "and certainly no headings. " * 80)
    chunker = SemanticChunker()
    result = chunker.chunk_document(DocumentChunkingInput(
        document_id=str(uuid.uuid4()),
        project_id=str(uuid.uuid4()),
        full_text=text,
    ))
    assert result.total_chunks >= 1
    for chunk in result.chunks:
        banned = {"introduction", "conclusion", "full document"}
        assert chunk.metadata.section.lower() not in banned, (
            f"banned literal section leaked: "
            f"{chunk.metadata.section!r}")
        assert chunk.metadata.structure_method in {
            "page_bounded", "inferred", None,
        }


async def test_chunker_preserves_real_headings():
    from services.chunking.chunker import SemanticChunker
    from services.chunking.types import DocumentChunkingInput

    text = (
        "Introduction\n\n"
        + ("Paragraph one body sentence. " * 30) + "\n\n"
        + "Methods\n\n"
        + ("Paragraph two body sentence. " * 30) + "\n\n"
        + "Results\n\n"
        + ("Paragraph three body sentence. " * 30)
    )
    spans = [
        {"page": 1, "start": 0, "end": 200, "method": "native",
         "confidence": 0.9, "heading": "Introduction"},
        {"page": 1, "start": 200, "end": 500, "method": "native",
         "confidence": 0.9, "heading": "Methods"},
        {"page": 2, "start": 500, "end": 900, "method": "native",
         "confidence": 0.9, "heading": "Results"},
    ]
    chunker = SemanticChunker()
    result = chunker.chunk_document(DocumentChunkingInput(
        document_id=str(uuid.uuid4()),
        project_id=str(uuid.uuid4()),
        full_text=text,
        page_spans=spans,
    ))
    seen = {chunk.metadata.section for chunk in result.chunks}
    # A detected heading that is a banned literal is never surfaced as a
    # section name (it is renamed to a page-bounded label); real headings
    # are kept.
    assert "Introduction" not in seen
    assert "Methods" in seen
    assert "Results" in seen
    assert not any(
        (name or "").strip().rstrip(":").strip().lower()
        in {"introduction", "conclusion", "full document"}
        for name in seen
    )
    for chunk in result.chunks:
        assert chunk.metadata.structure_method is not None


async def test_extraction_collapses_letter_spacing():
    from services.extraction.extractor import DocumentExtractor
    extractor = DocumentExtractor()
    # T005 acceptance input: every isolated letter collapses.
    assert extractor._clean_text(
        "s h e r y i a n s c is a term") == "sheryiansc is a term"
    # T005 acceptance output: a doubled letter survives the collapse.
    assert extractor._clean_text(
        "s h e r r y i a n s c is a term") == "sherryiansc is a term"


async def test_extraction_dehyphenates():
    from services.extraction.extractor import DocumentExtractor
    cleaned = DocumentExtractor()._clean_text("exam-\nple")
    assert cleaned == "example"


async def test_validator_topic_mismatch_degrades_on_page_bounded():
    from services.academic.validation import validate_notes_artifact
    artifact = {
        "sections": [{
            "unit_title": "Unit 1",
            "topic_title": "Pages 1-2",
            "heading": "Pages 1-2",
            "content": ("This is a real body paragraph about the "
                        "document with plenty of content. " * 3),
            "citations": [{"chunk_id": "c1", "span": "real body"}],
            "metadata": {
                "structure_method": "page_bounded",
                "structure_confidence": 0.3,
            },
        }],
        "source_gaps": [],
    }
    structure = {"units": [
        {"title": "Unit 1", "topics": [{"title": "Pages 1-2"}]},
    ]}
    report = validate_notes_artifact(
        artifact, structure=structure,
        allowed_chunk_ids={"c1"},
        chunk_texts={"c1": "real body about the document with content"},
    )
    assert report.ok is True
    codes = [w["code"] for w in report.warnings]
    assert "topic_mismatch_degraded" in codes


async def test_validator_topic_overlap_low_is_advisory_for_real_topics():
    from services.academic.validation import validate_notes_artifact
    artifact = {
        "sections": [{
            "unit_title": "Unit 1",
            "topic_title": "Cell Division",
            "heading": "Cell Division",
            "content": "Photosynthesis converts light to energy. " * 4,
            "citations": [{"chunk_id": "c1",
                           "span": "photosynthesis converts"}],
        }],
        "source_gaps": [],
    }
    structure = {"units": [
        {"title": "Unit 1", "topics": [{"title": "Cell Division"}]},
    ]}
    report = validate_notes_artifact(
        artifact, structure=structure,
        allowed_chunk_ids={"c1"},
        chunk_texts={"c1": "photosynthesis converts light to energy"},
    )
    # The extractor owns the label, so label/body overlap is advisory only.
    assert report.ok is True
    assert any(w["code"] == "topic_overlap_low" for w in report.warnings)


async def test_flashcards_skip_garbage_topics():
    from services.academic.flashcards import build_flashcards
    from services.academic.types import (
        DocumentStructure, ExtractedTopic, StructuredUnit,
        ExtractedConcept, ExtractedSubtopic, ConceptProvenance,
        StructureMethod,
    )

    bad_topic = ExtractedTopic(
        title="Conclusion",
        pages=[1],
        chunks=["c1"],
        subtopics=[ExtractedSubtopic(
            title="Conclusion",
            pages=[1],
            chunks=["c1"],
            concepts=[ExtractedConcept(
                title="Conclusion",
                pages=[1],
                chunks=["c1"],
                confidence=0.5,
                provenance=ConceptProvenance(
                    method=StructureMethod.HEADING_GROUP,
                    source_heading="Conclusion",
                    chunk_ids=["c1"],
                    note="garbage",
                ),
            )],
            method=StructureMethod.HEADING_GROUP,
        )],
        concepts=[],
        method=StructureMethod.HEADING_GROUP,
    )
    good_topic = ExtractedTopic(
        title="Cell Structure",
        pages=[1, 2],
        chunks=["c2"],
        subtopics=[ExtractedSubtopic(
            title="Cell Structure",
            pages=[1, 2],
            chunks=["c2"],
            concepts=[ExtractedConcept(
                title="Mitochondria",
                pages=[1],
                chunks=["c2"],
                confidence=0.9,
                provenance=ConceptProvenance(
                    method=StructureMethod.HEADING_GROUP,
                    source_heading="Cell Structure",
                    chunk_ids=["c2"],
                    note="ok",
                ),
            )],
            method=StructureMethod.HEADING_GROUP,
        )],
        concepts=[],
        method=StructureMethod.HEADING_GROUP,
    )
    structure = DocumentStructure(
        doc_id="doc-x",
        title="Biology",
        units=[StructuredUnit(
            title="Unit 1 -- Biology",
            pages=[1, 2],
            topics=[bad_topic, good_topic],
            method=StructureMethod.HEADING_GROUP,
        )],
        fingerprint="x" * 64,
        source_type="study_material",
    )
    deck = build_flashcards(
        structure,
        {"c2": "Mitochondria are the organelles that produce ATP in the cell. "
               "They power nearly every chemical reaction inside a living cell."})
    topic_titles = [c.topic for c in deck.flashcards]
    assert "Conclusion" not in topic_titles
    assert "Cell Structure" in topic_titles
    assert any("invalid_topic" in note
               for note in deck.provenance_note.splitlines()) \
        or "invalid_topic" in deck.provenance_note


async def test_banned_heading_never_becomes_a_topic_title():
    """A genuine 'Conclusion' heading must never surface as a section name."""
    from services.chunking.chunker import SemanticChunker
    from services.chunking.types import DocumentChunkingInput
    import uuid as _uuid

    text = "Methods\nWe measured things carefully here.\n\nConclusion\nWe conclude the thing works.\n"
    result = SemanticChunker().chunk_document(DocumentChunkingInput(
        document_id=str(_uuid.uuid4()),
        project_id=str(_uuid.uuid4()),
        full_text=text,
        page_spans=None,
    ))
    names = {(chunk.metadata.section or "").strip().lower()
             for chunk in result.chunks}
    assert "conclusion" not in names
    for chunk in result.chunks:
        if (chunk.metadata.section or "").strip().lower() == "conclusion":
            raise AssertionError("banned literal leaked as a section name")
