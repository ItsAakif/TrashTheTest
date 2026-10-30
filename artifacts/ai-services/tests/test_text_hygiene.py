"""Regression tests: shared text hygiene + concept naming v2 + topic caps.

Pins the PDF text-layer cleanup contract:

- strip_control_chars turns bullet-glyph control codes into "• " and
  drops the rest of the Cc/Cf range (control chars never reach cards).
- collapse_letter_spacing repairs letter-spaced running headers.
- find_boilerplate_lines flags a line on 69/70 pages but not one on 3/70.
- is_usable_title_candidate rejects page numbers, number-glued titles and
  sentences; accepts real title lines.
- _concept_title picks the best usable line (never the running header,
  never a sentence) and returns "" when nothing qualifies.
- a 22-chunk topic splits into page-window subtopics of <= 8 chunks.
- generated card/MCQ text is free of control characters.
"""

import unicodedata

import pytest

from services.academic.extraction import (
    MAX_TOPIC_CHUNKS,
    _concept_title,
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
from services.text_utils import (
    clean_display_text,
    collapse_letter_spacing,
    collapse_spaces,
    find_boilerplate_lines,
    is_page_number_line,
    is_usable_title_candidate,
    normalise_text,
    remove_boilerplate,
    strip_control_chars,
)

pytestmark = pytest.mark.asyncio

HEADER = "s h e r y i a n s c o d i n g s c h o o l"
HEADER_COLLAPSED = "sheryianscodingschool"


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


def _has_control_chars(text: str) -> bool:
    return any(
        unicodedata.category(ch) in ("Cc", "Cf") and ch not in ("\n", "\t")
        for ch in (text or "")
    )


# -- strip_control_chars ------------------------------------------------------


async def test_strip_control_chars_marks_bullets_and_drops_cc():
    out = strip_control_chars("\x0f a\x16 b")
    assert out == "• a• b"
    assert not _has_control_chars(out)


async def test_strip_control_chars_keeps_newlines_and_tabs():
    assert strip_control_chars("a\nb\tc") == "a\nb\tc"


async def test_strip_control_chars_collapses_space_runs_per_line():
    # Runs of spaces collapse and each line is right-stripped (newlines
    # are kept; only trailing spaces are removed per the contract).
    assert strip_control_chars("a\x0f   b\n  c  ") == "a• b\n c"


# -- collapse_letter_spacing --------------------------------------------------


async def test_collapse_letter_spacing_repairs_running_header():
    assert collapse_letter_spacing(HEADER) == HEADER_COLLAPSED


async def test_collapse_letter_spacing_leaves_prose_alone():
    line = "What are Data Types"
    assert collapse_letter_spacing(line) == line


# -- collapse_spaces / normalise_text ------------------------------------------


async def test_collapse_spaces_keeps_paragraph_breaks():
    assert collapse_spaces("a\n\n\n\nb") == "a\n\nb"
    assert collapse_spaces("  a  b  ") == "a b"


async def test_normalise_text_end_to_end():
    text = HEADER + "\n\x0f Bullet item\nSome prose line."
    assert normalise_text(text) == (
        "sheryianscodingschool\n• Bullet item\nSome prose line."
    )


# -- is_page_number_line -------------------------------------------------------


async def test_is_page_number_line():
    assert is_page_number_line("01")
    assert is_page_number_line("11")
    assert is_page_number_line("7")
    assert is_page_number_line("iv")
    assert not is_page_number_line("What are Data Types")
    assert not is_page_number_line("")


# -- find_boilerplate_lines -----------------------------------------------------


async def test_find_boilerplate_lines_flags_69_of_70_pages():
    pages = []
    for i in range(70):
        if i == 0:
            pages.append("Intro text without the header\nContent A")
        else:
            pages.append(f"{HEADER}\nContent {i}")
    boilerplate = find_boilerplate_lines(pages)
    assert HEADER_COLLAPSED in boilerplate


async def test_find_boilerplate_lines_ignores_3_of_70_pages():
    pages = [f"Content {i}" for i in range(70)]
    for i in (1, 2, 3):
        pages[i] = "Duplicates\n" + pages[i]
    boilerplate = find_boilerplate_lines(pages)
    assert "Duplicates" not in boilerplate


async def test_remove_boilerplate_drops_flagged_and_page_number_lines():
    text = f"{HEADER}\n01\nReal content line.\n12"
    out = remove_boilerplate(text, {HEADER_COLLAPSED})
    assert out == "Real content line."


# -- is_usable_title_candidate ---------------------------------------------------


async def test_is_usable_title_candidate_rejects_bad_shapes():
    assert not is_usable_title_candidate("01")
    assert not is_usable_title_candidate(
        "Data Structures 11 sheryianscodingschool In-build")
    assert not is_usable_title_candidate("Some sentence.")
    assert not is_usable_title_candidate("x")
    assert not is_usable_title_candidate("")


async def test_is_usable_title_candidate_accepts_real_titles():
    assert is_usable_title_candidate("What are Data Types")
    assert is_usable_title_candidate("While loop")
    assert not is_usable_title_candidate(
        "What are Data Types", forbidden={"what are data types"})


# -- _concept_title (naming v2) --------------------------------------------------


async def test_concept_title_prefers_real_title_over_header():
    chunk = _chunk(
        "chunk-h",
        f"{HEADER}\nWhat are Data Types\n"
        "Data types classify the kind of value a variable can hold in python.",
        "Data Types in python",
    )
    assert _concept_title(chunk, "Data Types in python") == "What are Data Types"


async def test_concept_title_never_returns_a_sentence():
    chunk = _chunk(
        "chunk-s",
        "To truly understand this book by heart and make the most out of every concept",
        "Answer True and Fals",
    )
    assert _concept_title(chunk, "Answer True and Fals") == ""


async def test_concept_title_drops_header_when_boilerplate_supplied():
    chunk = _chunk(
        "chunk-b",
        f"{HEADER}\n"
        "To truly understand this book by heart and make the most out of every concept",
        "Answer True and Fals",
    )
    assert _concept_title(
        chunk, "Answer True and Fals", boilerplate={HEADER_COLLAPSED}) == ""


async def test_concept_title_defined_term_still_works():
    chunk = _chunk(
        "chunk-d",
        "Polymorphism is the ability of an object to take many forms.",
        "Inheritance",
    )
    assert _concept_title(chunk, "Inheritance") == "Polymorphism"


async def test_concept_title_rejects_number_glued_line():
    chunk = _chunk(
        "chunk-n",
        "Data Structures 11 sheryianscodingschool In-build\n"
        "A data structure organises data so it can be used efficiently.",
        "Data Structures",
    )
    # The number-glued line is not a usable candidate; the prose line is a
    # sentence, so nothing qualifies and the caller must skip the concept.
    assert _concept_title(chunk, "Data Structures") == ""


# -- topic cap (FIX E) ------------------------------------------------------------


async def test_topic_with_22_chunks_splits_into_page_windows():
    terms = ["Polymorphism", "Inheritance", "Encapsulation", "Abstraction"]
    chunks = [
        _chunk(
            f"chunk-{i}",
            f"{terms[i % len(terms)]} is the ability of an object to take many forms.",
            "Answer True and Fals",
            page=5 + i,
            position=i,
        )
        for i in range(22)
    ]
    structure = extract_structure(
        "doc-1", "primer", chunks,
        source_type="study_material",
        source_metadata={"document_id": "x"},
    )
    topics = [t for u in structure.units for t in u.topics]
    assert len(topics) == 1
    topic = topics[0]
    assert len(topic.chunks) == 22
    subs = topic.subtopics
    assert [len(s.chunks) for s in subs] == [8, 8, 6]
    assert all(len(s.chunks) <= MAX_TOPIC_CHUNKS for s in subs)
    assert [s.title for s in subs] == [
        "Answer True and Fals (p. 5-12)",
        "Answer True and Fals (p. 13-20)",
        "Answer True and Fals (p. 21-26)",
    ]
    # Every concept survives the split, in source order.  R2 gates the
    # front/back-matter pages (5, 6, 25, 26): their single-sentence chunks
    # are dropped, leaving 18 concepts from pages 7-24.
    concept_titles = [c.title for s in subs for c in s.concepts]
    assert len(concept_titles) == 18
    assert concept_titles[0] == "Encapsulation"  # terms cycle with period 4
    assert concept_titles[-1] == "Abstraction"


async def test_small_topic_is_not_split():
    chunks = [
        _chunk(
            "chunk-0",
            "Polymorphism is the ability of an object to take many forms.",
            "Inheritance", page=1, position=0,
        )
    ]
    structure = extract_structure(
        "doc-1", "primer", chunks,
        source_type="study_material",
        source_metadata={"document_id": "x"},
    )
    topic = [t for u in structure.units for t in u.topics][0]
    assert len(topic.chunks) == 1
    assert len(topic.subtopics) == 1
    assert topic.subtopics[0].title == "Inheritance"


# -- chunker structure detector -----------------------------------------------------


async def test_chunker_drops_boilerplate_and_page_numbers_from_headings():
    from services.chunking.chunker import SemanticChunker

    # Only the running header repeats across pages; the content lines are
    # unique to the page under test.
    page_text = "\n\n".join([
        HEADER,
        "01",
        "What are Data Types",
        "Data types classify the kind of value a variable can hold.",
        HEADER,
        "02",
        "While Loops",
        "A while loop repeats while the condition holds.",
    ])
    other_pages = [
        f"{HEADER}\nUnique content {i}." for i in range(6)
    ]
    boilerplate = find_boilerplate_lines(other_pages + [page_text])
    assert HEADER_COLLAPSED in boilerplate
    sections = SemanticChunker()._detect_structure(
        page_text, boilerplate=boilerplate)
    headings = [s["heading"] for s in sections]
    assert HEADER_COLLAPSED not in headings
    assert "01" not in headings
    assert "02" not in headings
    assert "What are Data Types" in headings
    assert "While Loops" in headings


# -- generated card/MCQ text is sanitised (FIX F) -------------------------------------


async def test_flashcard_text_has_no_control_chars():
    structure = _structure_with_concepts([("Polymorphism", "c-good")])
    chunk_texts = {"c-good": "Polymorphism\x0f is the ability\x16 of an object to take many forms. It lets one interface serve many different types."}
    deck = build_flashcards(structure, chunk_texts)
    assert deck.flashcards, "the good concept must produce a card"
    for card in deck.flashcards:
        assert not _has_control_chars(card.question)
        assert not _has_control_chars(card.answer)


async def test_mcq_text_has_no_control_chars():
    from services.academic.mcq import build_matrix, build_mcqs

    # Four concepts so the real-fact distractor pool has 3 candidates.
    structure = _structure_with_concepts([
        ("Polymorphism", "c1"), ("Inheritance", "c2"),
        ("Encapsulation", "c3"), ("Abstraction", "c4"),
    ])
    chunk_texts = {
        "c1": "Polymorphism\x0f is the ability\x16 of an object to take many forms.",
        "c2": "Inheritance lets a class acquire properties of another class.",
        "c3": "Encapsulation bundles data with the methods that operate on it.",
        "c4": "Abstraction hides complex reality while exposing only the essential parts.",
    }
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, chunk_texts, plan)
    assert mcq_set.questions, "the cell must be filled from the evidence"
    for question in mcq_set.questions:
        texts = [question.question, question.explanation]
        texts += [option.text for option in question.options]
        for text in texts:
            assert not _has_control_chars(text)


async def test_clean_display_text_replaces_bullets():
    assert clean_display_text("a\x0fb") == "a• b"
