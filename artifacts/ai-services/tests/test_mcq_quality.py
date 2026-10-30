"""Regression tests: MCQ real-fact distractors + flashcard concept-name floor.

Pins the distractor-quality contract:

- distractors are REAL cleaned source facts from OTHER concepts, never
  templated filler ("It is the reverse that holds", "always behaves this
  way", "This describes ...");
- a cell with fewer than 3 valid distractors is UNFILLED (never padded);
- a cell whose concept has no usable key fact (8..35 words) is UNFILLED;
- no emitted option contains a newline, a bullet glyph, or a template
  string;
- every emitted option has 8..35 words and >= 3 alphabetic words;
- the shingle-similarity ceiling rejects a near-duplicate distractor;
- the flashcard concept-name floor skips garbage names (bullet glyph,
  leading punctuation/digit, single short word, single lowercase word).
"""

import inspect

import pytest

from services.academic import flashcards as fc
from services.academic import mcq as mcqm
from services.academic.flashcards import build_flashcards
from services.academic.mcq import build_matrix, build_mcqs
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

# The three template distractor strings that must never be generated.
_TEMPLATE_STRINGS = (
    "It is the reverse that holds",
    "always behaves this way",
    "This describes",
)


def _concept(title, cid):
    return ExtractedConcept(
        title=title, pages=[1], chunks=[cid], confidence=0.85,
        provenance=ConceptProvenance(
            method=StructureMethod.HEADING_GROUP,
            source_heading="Topic", chunk_ids=[cid], note="test concept",
        ),
    )


def _structure_with_texts(concepts):
    """One unit/topic/subtopic holding the given (title, chunk_id, text)."""
    typed = [_concept(title, cid) for title, cid, _ in concepts]
    topic = ExtractedTopic(
        title="Topic", pages=[1], chunks=[cid for _, cid, _ in concepts],
        subtopics=[ExtractedSubtopic(
            title="Topic", pages=[1], chunks=[cid for _, cid, _ in concepts],
            concepts=typed, method=StructureMethod.HEADING_GROUP,
        )],
        concepts=[], method=StructureMethod.HEADING_GROUP,
    )
    unit = StructuredUnit(
        title="Unit 1", topics=[topic], method=StructureMethod.HEADING_GROUP,
    )
    return DocumentStructure(
        doc_id="doc-1", title="primer", units=[unit], fingerprint="f" * 64,
    )


def _texts(concepts):
    return {cid: text for _, cid, text in concepts}


def _distinct_concepts():
    """Five concepts with distinct 8..35-word source facts."""
    return [
        ("Alpha concept", "c1",
         "Alpha concept is the first idea in this topic."),
        ("Beta concept", "c2",
         "Beta concept is the second idea in this topic."),
        ("Gamma concept", "c3",
         "Gamma concept is the third idea in this topic."),
        ("Delta concept", "c4",
         "Delta concept is the fourth idea in this topic."),
        ("Epsilon concept", "c5",
         "Epsilon concept is the fifth idea in this topic."),
    ]


# -- FIX 1/3: unfilled cells ---------------------------------------------------


async def test_cell_with_two_usable_distractors_is_unfilled():
    # Three concepts: the key concept plus only two others. Two valid
    # distractors exist, which is fewer than 3 -> UNFILLED, never padded.
    concepts = _distinct_concepts()[:3]
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, _texts(concepts), plan)
    assert mcq_set.questions == []
    assert len(mcq_set.plan.unfilled) == 6
    assert all("insufficient distinct source facts" in u.reason
               for u in mcq_set.plan.unfilled)


async def test_cell_with_no_usable_key_fact_is_unfilled():
    # c1's only sentence is too short (< 8 words): its cells are unfilled
    # with the no-key-fact reason. The other concepts then have only two
    # valid distractors left, so their cells are unfilled too.
    concepts = [
        ("Alpha concept", "c1", "Too short."),
        ("Beta concept", "c2", "Beta concept is the second idea in this topic."),
        ("Gamma concept", "c3", "Gamma concept is the third idea in this topic."),
        ("Delta concept", "c4", "Delta concept is the fourth idea in this topic."),
    ]
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, _texts(concepts), plan)
    assert mcq_set.questions == []
    reasons = [u.reason for u in mcq_set.plan.unfilled]
    assert any("no usable key fact" in r for r in reasons)
    assert any("insufficient distinct source facts" in r for r in reasons)


async def test_unfilled_cells_are_never_padded():
    # A cell with no source material at all stays unfilled (0 questions).
    concepts = _distinct_concepts()[:1]
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, {}, plan)  # no chunk texts at all
    assert mcq_set.questions == []
    assert len(mcq_set.plan.unfilled) == 6
    assert all("no source material" in u.reason for u in mcq_set.plan.unfilled)


# -- FIX 1/2: real facts, never templates --------------------------------------


async def test_no_option_contains_newline_bullet_or_template():
    concepts = _distinct_concepts()
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, _texts(concepts), plan)
    assert mcq_set.questions, "five concepts must fill the cells"
    for question in mcq_set.questions:
        for option in question.options:
            assert "\n" not in option.text, option.text
            assert "•" not in option.text, option.text
            for template in _TEMPLATE_STRINGS:
                assert template not in option.text, option.text


async def test_template_strings_appear_nowhere_in_module_source():
    source = inspect.getsource(mcqm)
    for template in _TEMPLATE_STRINGS:
        assert template not in source, (
            f"template distractor {template!r} still in mcq.py source")


async def test_every_option_has_valid_word_count_and_alpha_words():
    concepts = _distinct_concepts()
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, _texts(concepts), plan)
    assert mcq_set.questions
    for question in mcq_set.questions:
        for option in question.options:
            words = option.text.split()
            assert 8 <= len(words) <= 35, option.text
            alpha = [w for w in words if w.isalpha()]
            assert len(alpha) >= 3, option.text


async def test_shingle_ceiling_rejects_near_duplicate_distractor():
    # c2 is a near-duplicate of c1 (one final word differs): shingle
    # similarity >= 0.8, so c2 must never be emitted as a distractor.
    near = ("Photosynthesis converts light energy into chemical energy "
            "stored in glucose daily.")
    concepts = [
        ("Photosynthesis", "c1",
         "Photosynthesis converts light energy into chemical energy stored in glucose."),
        ("Near duplicate", "c2", near),
        ("Chlorophyll", "c3",
         "Chlorophyll a and chlorophyll b absorb red and blue light in the thylakoid membranes."),
        ("Light reactions", "c4",
         "The light dependent reactions split water, release oxygen, and produce ATP."),
        ("Calvin cycle", "c5",
         "The Calvin cycle fixes carbon dioxide into sugar in the stroma using ATP."),
    ]
    assert mcqm.shingle_similarity(concepts[0][2], near) >= 0.8
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, _texts(concepts), plan)
    assert mcq_set.questions, "three distinct facts remain, so cells fill"
    # For the cell whose key is c1's fact, the near-duplicate c2 must be
    # rejected as a distractor (it is a perfectly valid distractor for the
    # unrelated concepts c3/c4/c5, which is correct behaviour).
    c1_fact = concepts[0][2]
    key_cells = [q for q in mcq_set.questions
                 if q.options[q.correct_answer].text == c1_fact]
    assert key_cells, "c1's cell must be filled"
    for question in key_cells:
        for option in question.options:
            if not option.is_correct:
                assert option.text != near, "near-duplicate leaked as distractor"


async def test_key_fact_drops_repeated_header_line():
    # A line repeated across every chunk is boilerplate and must never
    # reach an option, even though no page-derived set is supplied.
    header = "sheryianscodingschool"
    concepts = [
        ("Alpha concept", "c1",
         f"{header}\nAlpha concept is the first idea in this topic."),
        ("Beta concept", "c2",
         f"{header}\nBeta concept is the second idea in this topic."),
        ("Gamma concept", "c3",
         f"{header}\nGamma concept is the third idea in this topic."),
        ("Delta concept", "c4",
         f"{header}\nDelta concept is the fourth idea in this topic."),
    ]
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, _texts(concepts), plan)
    assert mcq_set.questions
    for question in mcq_set.questions:
        for option in question.options:
            assert header not in option.text, option.text


async def test_key_is_the_concepts_own_source_fact():
    # The correct option is a real sentence from the cited chunk.
    concepts = _distinct_concepts()
    structure = _structure_with_texts(concepts)
    plan = build_matrix(structure, 6)
    mcq_set = build_mcqs(structure, _texts(concepts), plan)
    assert mcq_set.questions
    for question in mcq_set.questions:
        key = question.options[question.correct_answer].text
        cid = question.citations[0].chunk_id
        assert key in _texts(concepts)[cid], (key, cid)


# -- FIX 4: flashcard concept-name floor ---------------------------------------


async def test_is_usable_concept_name_floor():
    # Passes: multi-word names and capitalised single words >= 6 chars.
    assert fc._is_usable_concept_name("Classes")
    assert fc._is_usable_concept_name("Objects")
    assert fc._is_usable_concept_name("What are Data Types")
    assert fc._is_usable_concept_name("Args and Kwargs")
    # Rejected: bullet glyph, leading digit, single short word, lowercase.
    assert not fc._is_usable_concept_name("is True and False•")
    assert not fc._is_usable_concept_name("5 Comparison operators")
    assert not fc._is_usable_concept_name("False")
    assert not fc._is_usable_concept_name("File")
    assert not fc._is_usable_concept_name("methods")
    assert not fc._is_usable_concept_name("")


async def test_flashcard_floor_skips_garbage_concept_names():
    concepts = [
        ("Classes", "c1", "Classes are the blueprint for creating objects in code."),
        ("False", "c2", "False is a boolean value."),
        ("File", "c3", "File is a named location on disk."),
        ("is True and False•", "c4", "This line has a bullet glyph."),
        ("5 Comparison operators", "c5", "These are comparison operators."),
        ("methods", "c6", "Methods are functions defined inside a class."),
    ]
    structure = _structure_with_texts(concepts)
    deck = build_flashcards(structure, _texts(concepts))
    # Only "Classes" passes the floor.
    assert len(deck.flashcards) == 1
    assert deck.flashcards[0].metadata["concept"] == "Classes"
    provenance = deck.provenance_note
    for garbage in ("False", "File", "is True and False•",
                    "5 Comparison operators", "methods"):
        assert garbage in provenance
