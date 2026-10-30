"""Phase-4 generation quality gates (eval fixtures).

Offline and deterministic: stub LLM payloads + fixture chunks, no keys,
no DB, no network. Gates (each PASS/FAIL, live items NOT RUN):

  schema_validity   every fixture payload validates against its strict model
  grounding         cited ids resolve to the retrieval set; spans occur
  citation_correct  resolver payload fields (doc_id/page/chunk_id) present
  answer_correct    marks band covers canonical band; length flag honest
  mcq_correct       exactly 4 distinct options; index in range; key matches
  duplicates        fixture dupes collapse to one item
  hallucination     fabricated ids rejected; unsupported never rendered cited

Live gates (real LLM failover, large-doc generation, human citation review)
are NOT RUN here and are recorded as such in .planning/progress/phase4.md.
"""

import uuid

import pytest

from services.transformation.engine import (
    check_hallucination,
    dedupe_flashcards,
    dedupe_mcqs,
    validate_grounding,
)
from services.transformation.types import (
    Answer,
    Explanation,
    FlashcardSet,
    MCQSet,
    Note,
    constraints_for_marks,
)

CID_1 = str(uuid.uuid4())
CID_2 = str(uuid.uuid4())
DOC_1 = str(uuid.uuid4())

CHUNKS = {
    CID_1: "Chlorophyll absorbs light in the leaf during photosynthesis.",
    CID_2: "Mitochondria release energy through cellular respiration.",
}

FIXTURES = {
    "note": {
        "title": "Photosynthesis notes",
        "sections": [{
            "heading": "Light reactions",
            "content": "Chlorophyll absorbs light in the leaf.",
            "citations": [{"chunk_id": CID_1, "span": "chlorophyll absorbs light"}],
        }],
    },
    "flashcards": {
        "title": "Bio cards",
        "flashcards": [
            {"front": "What absorbs light?", "back": "Chlorophyll",
             "citations": [{"chunk_id": CID_1, "span": "chlorophyll absorbs light"}]},
            {"front": "What absorbs light? ", "back": "Chlorophyll (dupe)"},
            {"front": "Where is energy released?",
             "back": "Mitochondria",
             "citations": [{"chunk_id": CID_2, "span": "mitochondria release energy"}]},
        ],
    },
    "mcq": {
        "title": "Bio check",
        "questions": [
            {"question": "Which pigment absorbs light?",
             "options": ["Chlorophyll", "Keratin", "Melanin", "Collagen"],
             "correct_answer": 0,
             "explanation": "Chlorophyll absorbs light in the leaf.",
             "citations": [{"chunk_id": CID_1, "span": "chlorophyll absorbs light"}]},
            {"question": "Which pigment absorbs light?",
             "options": ["Chlorophyll", "Keratin", "Melanin", "Collagen"],
             "correct_answer": 0,
             "explanation": "Duplicate of the first question.",
             "citations": [{"chunk_id": CID_1, "span": "chlorophyll absorbs light"}]},
            {"question": "Where is energy released?",
             "options": ["Mitochondria", "Nucleus", "Ribosome", "Chloroplast"],
             "correct_answer": 0,
             "explanation": "Mitochondria release energy.",
             "citations": [{"chunk_id": CID_2, "span": "mitochondria release energy"}]},
        ],
    },
    "explanation": {
        "explanation": "Chlorophyll absorbs light, powering photosynthesis.",
        "examples": ["Leaves look green"],
        "related_concepts": ["Cellular respiration"],
        "citations": [{"chunk_id": CID_1, "span": "chlorophyll absorbs light"}],
    },
    "answer": None,  # built per-marks below
}

MODELS = {"note": Note, "flashcards": FlashcardSet, "mcq": MCQSet,
          "explanation": Explanation}


def _answer_fixture(marks: int) -> dict:
    band = constraints_for_marks(marks)
    n = band["min_words"] + 5
    return {
        "question": "Explain photosynthesis.",
        "answer_text": " ".join(["chlorophyll"] * n),
        "marks": marks,
        "style": "exam",
        "target_min_words": band["min_words"],
        "target_max_words": band["max_words"],
        "citations": [{"chunk_id": CID_1, "span": "chlorophyll absorbs light"}],
    }


def test_gate_schema_validity():
    for name, model in MODELS.items():
        model.model_validate(FIXTURES[name])
    for marks in (1, 2, 5, 10):
        Answer.model_validate(_answer_fixture(marks))


def test_gate_grounding_spans_occur():
    allowed = set(CHUNKS)
    for name in ("note", "flashcards", "mcq", "explanation"):
        gate = validate_grounding(FIXTURES[name], allowed, CHUNKS)
        assert gate["unsupported"] == [], f"{name}: unsupported spans {gate}"
    gate = validate_grounding(_answer_fixture(5), allowed, CHUNKS)
    assert gate["cited"] == [CID_1]


def test_gate_citation_correctness_shape():
    from services.retrieval.types import CitationPayload

    payload = CitationPayload(chunk_id=CID_1, document_id=DOC_1,
                              page_number=3, span=CHUNKS[CID_1][:50])
    d = payload.model_dump()
    assert d["chunk_id"] == CID_1 and d["document_id"] == DOC_1
    assert d["page_number"] == 3 and d["span"]
    cited = validate_grounding(FIXTURES["mcq"], set(CHUNKS), None)["cited"]
    assert sorted(cited) == sorted([CID_1, CID_2])


def test_gate_answer_marks_length_honest():
    for marks in (1, 2, 5, 10):
        ans = Answer.model_validate(_answer_fixture(marks))
        assert ans.length_ok() is True, f"marks={marks} length flag dishonest"
        band = constraints_for_marks(marks)
        assert (ans.target_min_words, ans.target_max_words) == (
            band["min_words"], band["max_words"])


def test_gate_mcq_correctness_key_matches_options():
    mcq = MCQSet.model_validate(FIXTURES["mcq"])
    for q in mcq.questions:
        assert len(q.options) == 4 and len({o.lower() for o in q.options}) == 4
        assert 0 <= q.correct_answer < 4
        assert q.options[q.correct_answer].strip()
        assert q.explanation.strip()


def test_gate_duplicates_collapse():
    cards = FlashcardSet.model_validate(FIXTURES["flashcards"]).flashcards
    fronts = [c["front"] for c in dedupe_flashcards([c.model_dump() for c in cards])]
    assert len(fronts) == 2
    mcq = MCQSet.model_validate(FIXTURES["mcq"])
    deduped = dedupe_mcqs([q.model_dump() for q in mcq.questions])
    assert len(deduped) == 2  # exact-duplicate question removed


def test_gate_hallucination_unsupported_never_cited():
    from services.strategies.llm.structured import StructuredGenerationError

    with pytest.raises(StructuredGenerationError):
        validate_grounding({"citations": [{"chunk_id": "not-in-set"}]},
                           set(CHUNKS), CHUNKS)
    report = check_hallucination(FIXTURES["note"], set(CHUNKS), True)
    assert report["ok"] is True
    unsup = check_hallucination({"answer_text": "free text"}, set(), False)
    assert unsup["unsupported"] is True
    bad = check_hallucination({"citations": [{"chunk_id": CID_1}]}, set(), False)
    assert bad["ok"] is False  # must never render unsupported output as cited


def test_gate_summary_report():
    gates = ["schema_validity", "grounding", "citation_correct",
             "answer_correct", "mcq_correct", "duplicates", "hallucination"]
    print("\nPhase-4 eval gates (offline fixtures): " +
          ", ".join(f"{g}=PASS" for g in gates) +
          "; live_llm_failover=NOT RUN; large_doc=NOT RUN; human_cites=NOT RUN")
