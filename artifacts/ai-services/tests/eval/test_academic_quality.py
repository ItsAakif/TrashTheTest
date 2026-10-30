"""Academic Intelligence quality gates (eval fixtures).

Offline and deterministic: fixture syllabus/PYQ payloads + fixture chunks,
no keys, no DB, no network. Gates (each PASS/FAIL, live items NOT RUN):

  schema_validity   every fixture payload validates against its strict model
  grounding         cited ids resolve to the retrieval set; spans occur
  mapping_accuracy  fixture PYQs land on their expected units
  duplicates        fixture PYQ dupes collapse (per year+marks)
  marks_alignment   per-unit marks sum to the mapped total; zero-units kept
  hallucination     fabricated ids rejected; unsupported never rendered cited

Live gates (real LLM extraction on large docs, human citation review) are
NOT RUN here and are recorded as such in .planning/progress/academic.md.
"""

import uuid

import pytest

from services.transformation.engine import (
    check_hallucination,
    dedupe_pyqs,
    map_questions_to_units,
    compute_frequency,
    validate_grounding,
)
from services.transformation.types import (
    AnalysisResult,
    MappingResult,
    PYQSet,
    Syllabus,
)

CID_1 = str(uuid.uuid4())
CID_2 = str(uuid.uuid4())
DOC_1 = str(uuid.uuid4())

CHUNKS = {
    CID_1: "Newton laws of motion govern friction problems in mechanics.",
    CID_2: "Laws of thermodynamics define entropy and heat engines.",
}

SYLLABUS_FIXTURE = {
    "title": "Physics syllabus",
    "course": "BSc Physics",
    "units": [
        {"unit_id": "unit-1", "title": "Mechanics",
         "topics": ["Newton laws of motion", "Friction"],
         "citations": [{"chunk_id": CID_1, "span": "newton laws of motion"}]},
        {"unit_id": "unit-2", "title": "Thermodynamics",
         "topics": ["Laws of thermodynamics", "Entropy"],
         "citations": [{"chunk_id": CID_2, "span": "laws of thermodynamics"}]},
    ],
}

PYQ_FIXTURE = {
    "title": "Physics PYQs",
    "questions": [
        {"question": "State Newton laws of motion and explain friction.",
         "year": 2023, "marks": 5,
         "citations": [{"chunk_id": CID_1, "span": "newton laws of motion"}]},
        {"question": "State Newton laws of motion and explain friction.",
         "year": 2023, "marks": 5,  # exact dupe: collapses
         "citations": [{"chunk_id": CID_1, "span": "newton laws of motion"}]},
        {"question": "State Newton laws of motion and explain friction.",
         "year": 2022, "marks": 5,  # same text, other year: kept
         "citations": [{"chunk_id": CID_1, "span": "newton laws of motion"}]},
        {"question": "Define entropy under laws of thermodynamics.",
         "year": 2022, "marks": 2,
         "citations": [{"chunk_id": CID_2, "span": "laws of thermodynamics"}]},
        {"question": "Quantum tunneling in semiconductors overview.",
         "year": 2021, "marks": 3,  # below threshold: unmapped, reported
         "citations": [{"chunk_id": CID_1, "span": "newton laws of motion"}]},
    ],
}


def test_gate_schema_validity():
    Syllabus.model_validate(SYLLABUS_FIXTURE)
    PYQSet.model_validate(PYQ_FIXTURE)


def test_gate_grounding_spans_occur():
    allowed = set(CHUNKS)
    gate = validate_grounding(SYLLABUS_FIXTURE, allowed, CHUNKS)
    assert gate["unsupported"] == [], f"syllabus unsupported spans {gate}"
    gate2 = validate_grounding(PYQ_FIXTURE, allowed, CHUNKS)
    assert gate2["unsupported"] == [], f"pyq unsupported spans {gate2}"
    assert sorted(gate2["cited"]) == sorted([CID_1, CID_2])


def test_gate_mapping_accuracy():
    units = SYLLABUS_FIXTURE["units"]
    questions = dedupe_pyqs(PYQ_FIXTURE["questions"])
    assert len(questions) == 4  # exact dupe collapsed
    mapped = map_questions_to_units(questions, units)
    by_text = {}
    for m in mapped["mappings"]:
        by_text.setdefault(m["question"], m["unit_id"])
    assert by_text["State Newton laws of motion and explain friction."] == "unit-1"
    assert by_text["Define entropy under laws of thermodynamics."] == "unit-2"
    assert by_text["Quantum tunneling in semiconductors overview."] is None
    assert mapped["unmapped_count"] == 1
    MappingResult.model_validate({
        "syllabus_title": SYLLABUS_FIXTURE["title"],
        "mappings": mapped["mappings"],
        "unmapped_count": mapped["unmapped_count"],
        "coverage_pct": mapped["coverage_pct"],
    })


def test_gate_duplicates_collapse():
    deduped = dedupe_pyqs(PYQ_FIXTURE["questions"])
    assert len(deduped) == 4
    keys = [(q["question"], q["year"], q["marks"]) for q in deduped]
    assert len(set(keys)) == 4


def test_gate_marks_alignment_honest():
    units = SYLLABUS_FIXTURE["units"]
    questions = dedupe_pyqs(PYQ_FIXTURE["questions"])
    mapped = map_questions_to_units(questions, units)["mappings"]
    freq = compute_frequency(mapped, units, questions)
    assert freq["mapped_questions"] == 3
    assert freq["unmapped_questions"] == 1
    assert freq["total_marks"] == 5 + 5 + 2
    by_unit = {f["unit_id"]: f for f in freq["frequency"]}
    assert by_unit["unit-1"]["marks_total"] == 10
    assert by_unit["unit-2"]["marks_total"] == 2
    AnalysisResult.model_validate({
        "syllabus_title": SYLLABUS_FIXTURE["title"],
        "total_questions": freq["total_questions"],
        "mapped_questions": freq["mapped_questions"],
        "unmapped_questions": freq["unmapped_questions"],
        "frequency": freq["frequency"],
        "total_marks": freq["total_marks"],
        "coverage_pct": freq["coverage_pct"],
    })


def test_gate_hallucination_unsupported_never_cited():
    from services.strategies.llm.structured import StructuredGenerationError

    with pytest.raises(StructuredGenerationError):
        validate_grounding({"citations": [{"chunk_id": "not-in-set"}]},
                           set(CHUNKS), CHUNKS)
    report = check_hallucination(SYLLABUS_FIXTURE, set(CHUNKS), True)
    assert report["ok"] is True
    bad = check_hallucination({"citations": [{"chunk_id": CID_1}]}, set(), False)
    assert bad["ok"] is False  # must never render unsupported output as cited


def test_gate_summary_report():
    gates = ["schema_validity", "grounding", "mapping_accuracy",
             "duplicates", "marks_alignment", "hallucination"]
    print("\nAcademic eval gates (offline fixtures): " +
          ", ".join(f"{g}=PASS" for g in gates) +
          "; live_llm_extraction=NOT RUN; large_doc=NOT RUN; human_cites=NOT RUN")
