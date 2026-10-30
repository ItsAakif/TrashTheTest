"""Notes-quality rules (R1-R4) + flashcard quality gates + PDF page numbering.

Pins the confirmed-defect fixes:

- R1 ``is_sentence_backed``: a chunk without a >= 8-word sentence ending in
  [.!?] may not seed a concept or a card.
- R2 front/back-matter window: an edge chunk is admissible only with
  >= 2 qualifying sentences.
- R3 ``is_junk_title``: credits/nav/brand lines and document/unit title
  restatements are never concept names.
- R4 anchoring: a concept name must appear within the first 40% of its chunk.
- flashcard atomicity / yes-no / proper-noun / duplicate rejections and
  CLOZE emission for a definitional chunk.
- the notes PDF no longer emits every page twice and the footer counts up.
"""

import io
import logging
import re

import pytest

from services.academic.extraction import extract_structure
from services.academic.flashcards import (
    _answer_is_acceptable,
    build_flashcards,
)
from services.academic.types import (
    ConceptProvenance,
    DocumentStructure,
    ExtractedConcept,
    ExtractedSubtopic,
    ExtractedTopic,
    StructureMethod,
    StructuredUnit,
)
from services.text_utils import (
    appears_early,
    count_qualifying_sentences,
    is_front_back_admissible,
    is_front_back_matter,
    is_junk_title,
    is_sentence_backed,
)

pytestmark = pytest.mark.asyncio

PINNED_EPOCH = "1704067200"

# Two known 8-word sentences (used to build qualifying multi-sentence text).
S8_A = "The first sentence contains exactly eight words here."
S8_B = "The second sentence also contains eight words now."


# -- R1: sentence backing ----------------------------------------------------


def test_r1_requires_eight_word_terminal_sentence():
    assert is_sentence_backed(S8_A)
    assert count_qualifying_sentences(S8_A + " " + S8_B) == 2
    assert not is_sentence_backed("Too short here.")  # < 8 words
    assert not is_sentence_backed("No terminal punctuation at all here")  # no [.!?]


# -- R2: front/back-matter window --------------------------------------------


def test_r2_front_back_window_and_two_sentence_rule():
    # 20 chunks -> 10% = 2 chunks at each end.
    assert is_front_back_matter(0, 20)
    assert is_front_back_matter(1, 20)
    assert is_front_back_matter(19, 20)
    assert not is_front_back_matter(5, 20)
    # Edge chunk with one sentence is inadmissible; two sentences pass.
    assert not is_front_back_admissible(0, 20, S8_A)
    assert is_front_back_admissible(0, 20, S8_A + " " + S8_B)
    # Interior chunks are never gated.
    assert is_front_back_admissible(5, 20, "Short.")
    # 10% of a 3-chunk document floors to 0 -> no window.
    assert not is_front_back_matter(0, 3)


# -- R3: junk titles ---------------------------------------------------------


def test_r3_blocklist_and_title_subset():
    for junk in (
        "Sheryians AI", "Downloading Python", "Much love",
        "Answer True and Fals", "Page 12", "Slide 4", "Credits",
        "References", "Agenda", "Thank you", "http://example.com",
        "www.example.com", "follow us", "Subscribe now", "click here",
    ):
        assert is_junk_title(junk), junk
    assert is_junk_title("", "Doc")
    # subset / equality of document or unit title tokens
    assert is_junk_title(
        "Photosynthesis", "Photosynthesis primer",
        "Unit 1 -- Photosynthesis primer")
    assert is_junk_title(
        "Unit 1", "Photosynthesis primer",
        "Unit 1 -- Photosynthesis primer")
    # exact running-header/boilerplate token set
    assert is_junk_title(
        "sheryianscodingschool", boilerplate={"sheryianscodingschool"})
    # a real concept survives
    assert not is_junk_title(
        "Polymorphism", "Biology primer", "Unit 1 -- Biology primer")


# -- R4: anchoring -----------------------------------------------------------


def test_r4_concept_must_appear_in_first_40_percent():
    late = ("x" * 100) + " Concept Name"
    assert not appears_early(late, "Concept Name")
    assert appears_early("Concept Name is defined right here.", "Concept Name")
    assert not appears_early("Concept Name is defined right here.", "Missing")


# -- extraction wiring -------------------------------------------------------


def _chunk(cid, text, heading, page=1, position=0):
    return {
        "id": cid, "chunk_id": cid, "text": text, "content": text,
        "heading": heading, "section": heading, "page": page,
        "position": position, "source_type": "study_material",
        "document_id": "doc-1", "document_title": "primer.pdf",
    }


def _concept_names(structure):
    return [
        c.title
        for u in structure.units
        for t in u.topics
        for s in t.subtopics
        for c in s.concepts
    ]


async def test_extraction_skips_unsentencebacked_and_junk_chunks(caplog):
    chunks = [
        _chunk("c1", "s h e r y i a n s c o d i n g s c h o o l",
               "Sheryians AI", position=0),
        _chunk("c2", "Polymorphism is the ability of an object to take many forms.",
               "Inheritance", position=1),
    ]
    with caplog.at_level(logging.INFO, logger="services.academic.extraction"):
        structure = extract_structure("doc-1", "primer", chunks)
    names = _concept_names(structure)
    assert names == ["Polymorphism"]
    assert any(
        "academic.junk_filtered" in r.getMessage()
        for r in caplog.records
    ), "one audit line per document is required"


# -- flashcard gates ---------------------------------------------------------


def _structure_with(concepts, topic="Topic"):
    typed = [
        ExtractedConcept(
            title=title, pages=[1], chunks=[cid], confidence=0.85,
            provenance=ConceptProvenance(
                method=StructureMethod.HEADING_GROUP,
                source_heading=topic, chunk_ids=[cid], note="test",
            ),
        )
        for title, cid, _ in concepts
    ]
    topic_obj = ExtractedTopic(
        title=topic, pages=[1], chunks=[cid for _, cid, _ in concepts],
        subtopics=[ExtractedSubtopic(
            title=topic, pages=[1],
            chunks=[cid for _, cid, _ in concepts],
            concepts=typed, method=StructureMethod.HEADING_GROUP)],
        concepts=[], method=StructureMethod.HEADING_GROUP,
    )
    return DocumentStructure(
        doc_id="doc-1", title="primer",
        units=[StructuredUnit(title="Unit 1", topics=[topic_obj],
                              method=StructureMethod.HEADING_GROUP)],
        fingerprint="f" * 64,
    )


def test_flashcard_atomicity_yes_no_length_substring():
    # one sentence, no list -> acceptable
    assert _answer_is_acceptable("A single atomic fact lives right here.", "Q")
    # more than one sentence -> rejected
    assert not _answer_is_acceptable(
        "First fact is here. Second fact is here.", "Q")
    # >= 3 comma items -> rejected
    assert not _answer_is_acceptable("It has alpha, beta, gamma inside it.", "Q")
    # more than one "and" joining facts -> rejected
    assert not _answer_is_acceptable(
        "It joins this and that and the other thing.", "Q")
    # starts with Yes / No -> rejected
    assert not _answer_is_acceptable("Yes, this is the correct statement.", "Q")
    assert not _answer_is_acceptable("No, that is not the right answer.", "Q")
    # substring of the question -> rejected
    assert not _answer_is_acceptable("Polymorphism", "What is Polymorphism?")
    # shorter than 12 characters -> rejected
    assert not _answer_is_acceptable("Short.", "Q")


async def test_flashcard_skips_proper_noun_without_definitional_span():
    concepts = [
        ("Kubernetes", "c1",
         "Kubernetes orchestrates containers across many machines at scale. "
         "It schedules workloads across a cluster."),
        ("Polymorphism", "c2",
         "Polymorphism is the ability of an object to take many forms. "
         "It lets one interface serve many different types."),
    ]
    deck = build_flashcards(_structure_with(concepts), {
        cid: text for _, cid, text in concepts})
    assert len(deck.flashcards) == 1
    assert deck.flashcards[0].metadata["concept"] == "Polymorphism"
    assert "no_definitional_span" in deck.provenance_note


async def test_flashcard_skips_junk_and_unsentencebacked_concepts():
    concepts = [
        ("Sheryians AI", "c1",
         "Sheryians AI is a coding school for many students. "
         "It teaches web development to everyone."),
        ("Polymorphism", "c2", "Short."),
        ("Inheritance", "c3",
         "Inheritance is the ability to reuse behaviour from a parent. "
         "It lets a child class extend another class."),
    ]
    deck = build_flashcards(_structure_with(concepts), {
        cid: text for _, cid, text in concepts})
    assert [c.metadata["concept"] for c in deck.flashcards] == ["Inheritance"]
    assert "junk_concept" in deck.provenance_note
    assert "not_sentence_backed" in deck.provenance_note


async def test_flashcard_duplicate_front_is_deduplicated():
    # Positions 0 and 8 both use the DEFINITION frame and the same chunk
    # text, so their CLOZE fronts collide: the second must be dropped.
    terms = ["Polymorphism", "Inheritance", "Encapsulation", "Abstraction",
             "Recursion", "Iteration", "Compilation", "Interpretation",
             "Polymorphism"]
    concepts = []
    for i, term in enumerate(terms):
        if i == 8:
            # Identical text to index 0 -> identical CLOZE front.
            text = ("Polymorphism is a distinct concept numbered 0 here. "
                    "It appears throughout the material.")
        else:
            text = (f"{term} is a distinct concept numbered {i} here. "
                    f"It appears throughout the material.")
        concepts.append((term, f"c{i}", text))
    deck = build_flashcards(_structure_with(concepts), {
        cid: text for _, cid, text in concepts})
    assert len(deck.flashcards) == 8
    assert "duplicate_front" in deck.provenance_note


async def test_flashcard_cloze_emitted_for_definitional_chunk():
    concepts = [
        ("Polymorphism", "c1",
         "Polymorphism is the ability of an object to take many forms. "
         "It lets one interface serve many different types."),
        ("Inheritance", "c2",
         "Inheritance is the ability to reuse behaviour from a parent. "
         "It lets a child class extend another class."),
    ]
    deck = build_flashcards(_structure_with(concepts), {
        cid: text for _, cid, text in concepts})
    cloze = [c for c in deck.flashcards if c.metadata.get("cloze")]
    assert cloze, "a definitional chunk must emit a CLOZE card"
    card = cloze[0]
    assert "_____" in card.question
    assert card.answer.startswith("Polymorphism")
    assert card.card_type.value == "definition"
    # front/back still differ after normalisation (DB/UI contract unchanged)
    assert card.question != card.answer


# -- notes PDF: no doubled pages, incrementing footer ------------------------


def _pdf_fixture():
    from services.academic.coverage import attach_report, evaluate_coverage
    from services.academic.notes_hier import render_topic_notes, synthesize_notes
    from tests.eval.corpus import CORPUS

    chunks = [dict(r) for r in CORPUS if r["doc"] == "doc-biology"]
    structure = extract_structure(
        "doc-biology", "Photosynthesis primer", chunks)
    notes = [
        render_topic_notes(unit.title, topic, chunks)
        for unit in structure.units
        for topic in unit.topics
    ]
    artifact = synthesize_notes(
        "Photosynthesis primer", structure, notes,
        chunk_texts={r["id"]: r["text"] for r in chunks})
    return structure, attach_report(artifact, evaluate_coverage(structure, notes))


def _meta():
    return {
        "title": "Photosynthesis primer", "version": 1,
        "model": "test-model", "created_at": "2026-09-23",
        "epoch": int(PINNED_EPOCH),
    }


async def test_pdf_pages_are_not_doubled_and_footer_counts_up(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    from services.academic.notes_pdf import render_notes_pdf

    structure, artifact = _pdf_fixture()
    report = render_notes_pdf(artifact, structure, _meta())
    reader = PdfReader(io.BytesIO(report.pdf_bytes))
    pages_text = [(page.extract_text() or "") for page in reader.pages]
    assert len(pages_text) == report.page_count

    # The title-page brand banner appears exactly once (the previous
    # NumberedCanvas bug emitted every page twice).
    assert sum(text.count("TRASHTHETEST") for text in pages_text) == 1

    # No two consecutive pages are byte-identical content.
    assert all(pages_text[i] != pages_text[i + 1]
               for i in range(len(pages_text) - 1))

    # The footer increments 1..N (the old footer read "Page 1 of N").
    numbers = []
    for text in pages_text:
        match = re.search(r"Page (\d+) of (\d+)", text)
        assert match, f"footer missing on a page: {text[:80]!r}"
        numbers.append(int(match.group(1)))
    assert numbers == list(range(1, len(pages_text) + 1))


def test_split_long_bullet_never_truncates_mid_word():
    from services.academic.notes_pdf import _split_long_bullet

    long_bullet = " ".join(f"word{i}" for i in range(60))
    pieces = _split_long_bullet(long_bullet)
    assert all(len(piece.split()) <= 25 for piece in pieces)
    # No word is dropped or cut: re-joining restores the original words.
    assert " ".join(pieces).split() == long_bullet.split()
    assert _split_long_bullet("short bullet here") == ["short bullet here"]
