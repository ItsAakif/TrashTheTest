"""Regression tests: concept names are short noun phrases, never sentences.

The old ``_concept_title`` returned ``"<heading> -- <first 10 words>"`` so
every flashcard/MCQ frame interpolated a sentence ("Define <sentence> --
<more sentence> (<topic>)").  These tests pin the fixed behaviour:

- letter-spaced heading lines collapse and stay short (no " -- ", <= 8 words)
- a defined term ("<Term> is ...") becomes the bare term
- a usable leading sub-heading line becomes the concept name
- an unusable concept name is SKIPPED by the deck, never emitted
- a useless back (too short or equal to the front) is skipped, never emitted
"""

import pytest

from services.academic.extraction import (
    _concept_title,
    _is_usable_concept_name,
    extract_structure,
)
from services.academic.flashcards import build_flashcards
from services.academic.types import (
    ConceptProvenance,
    DocumentStructure,
    ExtractedConcept,
    ExtractedSubtopic,
    ExtractedTopic,
    StructureMethod,
    StructuredUnit,
)

pytestmark = pytest.mark.asyncio


def _chunk(cid, text, heading, page=1, position=0):
    return {
        "id": cid,
        "chunk_id": cid,
        "text": text,
        "content": text,
        "heading": heading,
        "section": heading,
        "page": page,
        "position": position,
        "source_type": "study_material",
        "document_id": "doc-1",
        "document_title": "primer.pdf",
    }


def _structure_with_concepts(concepts):
    """One unit/topic/subtopic holding the given (title, chunk_id) concepts."""
    typed = [
        ExtractedConcept(
            title=title,
            pages=[1],
            chunks=[cid],
            confidence=0.85,
            provenance=ConceptProvenance(
                method=StructureMethod.HEADING_GROUP,
                source_heading="Functions",
                chunk_ids=[cid],
                note="regression test concept",
            ),
        )
        for title, cid in concepts
    ]
    topic = ExtractedTopic(
        title="Functions",
        pages=[1],
        chunks=[cid for _, cid in concepts],
        subtopics=[
            ExtractedSubtopic(
                title="Functions",
                pages=[1],
                chunks=[cid for _, cid in concepts],
                concepts=typed,
                method=StructureMethod.HEADING_GROUP,
            )
        ],
        concepts=[],
        method=StructureMethod.HEADING_GROUP,
    )
    unit = StructuredUnit(
        title="Unit 1", topics=[topic], method=StructureMethod.HEADING_GROUP,
    )
    return DocumentStructure(
        doc_id="doc-1", title="primer", units=[unit], fingerprint="f" * 64,
    )


# -- _concept_title directly -------------------------------------------------


async def test_letter_spaced_chunk_has_no_sentence_title():
    # The letter-spaced running header ("sheryianscodingschool") is a
    # credits/brand line: R3 rejects it, so no concept name is produced.
    chunk = _chunk(
        "chunk-0",
        "s h e r y i a n s c o d i n g s c h o o l\n"
        "To truly understand this book by heart and make the most out of every concept",
        "Answer True and Fals",
    )
    assert _concept_title(chunk, "Answer True and Fals") == ""


async def test_defined_term_becomes_bare_term():
    chunk = _chunk(
        "chunk-1",
        "Polymorphism is the ability of an object to take many forms.",
        "Inheritance",
    )
    assert _concept_title(chunk, "Inheritance") == "Polymorphism"


async def test_leading_subheading_line_wins():
    chunk = _chunk(
        "chunk-2",
        "File Handling\nFiles are opened with open() and closed after use.",
        "Functions",
    )
    assert _concept_title(chunk, "Functions") == "File Handling"


async def test_unusable_source_returns_empty():
    # Punctuation-only text has no usable short name: the caller must skip,
    # not invent a sentence.
    chunk = _chunk("chunk-3", "!!! ??? ...", "Functions")
    assert _concept_title(chunk, "Functions") == ""


async def test_usable_name_rules():
    assert _is_usable_concept_name("Polymorphism", "Inheritance")
    assert not _is_usable_concept_name("", "Inheritance")
    assert not _is_usable_concept_name("x", "Inheritance")  # too short
    assert not _is_usable_concept_name("Inheritance", "Inheritance")  # == topic
    assert not _is_usable_concept_name("A -- B", "Inheritance")
    assert not _is_usable_concept_name("one two three four five six seven eight nine", "Inheritance")
    assert not _is_usable_concept_name("Trailing sentence.", "Inheritance")


# -- extract_structure end to end ---------------------------------------------


async def test_extract_structure_letter_spaced_chunk_title_is_short():
    structure = extract_structure(
        "doc-1", "primer",
        [_chunk(
            "chunk-0",
            "s h e r y i a n s c o d i n g s c h o o l\n"
            "To truly understand this book by heart and make the most out of every concept",
            "Answer True and Fals",
        )],
        source_type="study_material",
        source_metadata={"document_id": "x"},
    )
    names = [
        c.title
        for u in structure.units
        for t in u.topics
        for s in t.subtopics
        for c in s.concepts
    ]
    # The chunk is not sentence-backed (R1) and its only line is a credits
    # header (R3): no concept is invented for it.
    assert names == []


async def test_extract_structure_defined_term_title():
    structure = extract_structure(
        "doc-1", "primer",
        [_chunk(
            "chunk-1",
            "Polymorphism is the ability of an object to take many forms.",
            "Inheritance",
        )],
        source_type="study_material",
        source_metadata={"document_id": "x"},
    )
    names = [
        c.title
        for u in structure.units
        for t in u.topics
        for s in t.subtopics
        for c in s.concepts
    ]
    assert names == ["Polymorphism"]


# -- build_flashcards guards ---------------------------------------------------


async def test_card_from_unusable_concept_is_skipped_not_emitted():
    structure = _structure_with_concepts([
        ("Polymorphism", "c-good"),
        ("Functions -- For downloading python go on any browser go to", "c-bad"),
    ])
    chunk_texts = {
        "c-good": "Polymorphism is the ability of an object to take many forms.",
        "c-bad": "For downloading python go on any browser go to the official site.",
    }
    deck = build_flashcards(structure, chunk_texts)
    assert len(deck.flashcards) == 1
    assert deck.flashcards[0].concept_id == "concept-0001"
    assert deck.flashcards[0].metadata["concept"] == "Polymorphism"
    assert ("junk_concept" in deck.provenance_note
            or "invalid_concept" in deck.provenance_note)
    # No emitted front may carry the sentence-like title.
    for card in deck.flashcards:
        assert " -- " not in card.question
        assert len(card.question.split()) <= 24


async def test_card_with_useless_answer_is_skipped_not_emitted():
    # One good concept plus one whose only evidence sentence is too short for
    # a back: the short-answer card is skipped, the good card is emitted.
    # "Adenosine" passes the concept-name floor (>= 6 chars) so the skip is
    # the useless answer, not the name.
    structure = _structure_with_concepts([
        ("Polymorphism", "c-good"),
        ("Adenosine", "c-short"),
    ])
    chunk_texts = {
        "c-good": "Polymorphism is the ability of an object to take many forms.",
        "c-short": "ATP forms.",  # back would be < 12 chars
    }
    deck = build_flashcards(structure, chunk_texts)
    assert len(deck.flashcards) == 1
    assert deck.flashcards[0].concept_id == "concept-0001"
    assert len(deck.flashcards[0].answer) >= 12
    # "ATP forms." is not sentence-backed (R1), so it is skipped before an
    # answer is ever built.
    assert "not_sentence_backed" in deck.provenance_note


async def test_mcqs_leave_sentence_concept_cell_unfilled():
    from services.academic.mcq import build_matrix, build_mcqs

    structure = _structure_with_concepts([
        ("Functions -- For downloading python go on any browser go to", "c-bad"),
    ])
    chunk_texts = {"c-bad": "For downloading python go on any browser go to the official site."}
    plan = build_matrix(structure, 6)  # one cell per Bloom level
    mcq_set = build_mcqs(structure, chunk_texts, plan)
    assert mcq_set.questions == []
    assert len(mcq_set.plan.unfilled) == 6
    assert all("concept name rejected" in u.reason for u in mcq_set.plan.unfilled)


async def test_realistic_deck_fronts_are_short_questions():
    structure = extract_structure(
        "doc-1", "primer",
        [
            _chunk(
                "chunk-1",
                "Polymorphism is the ability of an object to take many forms.",
                "Inheritance", position=0,
            ),
            _chunk(
                "chunk-2",
                "Inheritance lets a child class reuse the parent class behaviour.",
                "Inheritance", position=1,
            ),
            _chunk(
                "chunk-3",
                "s h e r y i a n s c o d i n g s c h o o l\n"
                "To truly understand this book by heart and make the most out of every concept",
                "Answer True and Fals", position=2,
            ),
        ],
        source_type="study_material",
        source_metadata={"document_id": "x"},
    )
    chunk_texts = {
        "chunk-1": "Polymorphism is the ability of an object to take many forms.",
        "chunk-2": "Inheritance lets a child class reuse the parent class behaviour.",
        "chunk-3": "s h e r y i a n s c o d i n g s c h o o l\n"
                   "To truly understand this book by heart and make the most out of every concept",
    }
    deck = build_flashcards(structure, chunk_texts, max_cards=25, title="eval")
    assert deck.flashcards, "usable concepts must produce cards"
    for card in deck.flashcards:
        assert " -- " not in card.question
        assert len(card.question.split()) <= 24
        assert len(card.answer) >= 12
