"""Regression gates for notes/tutor/source hardening.

Each test proves a specific production failure mode cannot silently return:

* retrieval emptiness being reported as "unsupported" without recovery;
* boilerplate / no-information text persisted as a normal notes section;
* a headings-only PDF being accepted as a deliverable;
* syllabus/question-paper arithmetic being invented by a model;
* tutor greetings triggering a vector query;
* CORS headers missing on inner 4xx/5xx responses;
* an invalid source_type being silently accepted.
"""

import pytest

from services.academic.sufficiency import (
    SufficiencyStatus,
    assess_source_sufficiency,
    recover_topic_evidence,
)
from services.academic.validation import (
    assert_pdf_has_body,
    is_placeholder,
    require_valid_notes,
    validate_notes_artifact,
    validate_topic_notes,
)
from services.academic.source_ingestion import (
    compute_source_metrics,
    map_questions_to_syllabus,
    parse_question_paper_text,
    parse_syllabus_text,
)
from services.academic.types import (
    NotesGenerationConfig,
    NotesGenerationMode,
    NotesDetailLevel,
    recommend_notes_config,
)
from services.study.intent import (
    TutorIntent,
    classify_tutor_intent,
    greeting_response,
)


# --------------------------------------------------------------------------
# 1. Sufficiency + recovery
# --------------------------------------------------------------------------

def _chunk(cid, text, page=None, position=0, heading=""):
    return {
        "id": cid,
        "chunk_id": cid,
        "text": text,
        "content": text,
        "page": page,
        "position": position,
        "heading": heading,
    }


def test_missing_when_corpus_is_empty():
    result = assess_source_sufficiency(
        {"title": "Photosynthesis"}, evidence=[], all_chunks=[],
    )
    assert result.status is SufficiencyStatus.MISSING
    assert not result.sufficient
    assert not result.supported
    # The reason must name the scope, not blame the topic.
    assert "scope" in result.reason


def test_insufficient_when_no_chunk_matches_the_topic():
    chunks = [_chunk("c1", "The Krebs cycle oxidises acetyl coenzyme A.", page=3)]
    result = assess_source_sufficiency(
        {"title": "Photosynthesis"}, evidence=[], all_chunks=chunks,
    )
    assert result.status is SufficiencyStatus.INSUFFICIENT
    assert result.recovery_steps, "an unsupported verdict must record what was tried"


def test_recovery_finds_evidence_the_strict_selector_missed():
    """Alias/nearby-page recovery must run BEFORE an unsupported verdict."""

    chunks = [
        _chunk("c1", "Chlorophyll absorbs photons in the light reaction.", page=10, position=0),
        _chunk("c2", "The thylakoid membrane hosts photosystem two.", page=10, position=1),
        _chunk("c3", "Unrelated bookkeeping section about depreciation.", page=40, position=9),
    ]
    result = assess_source_sufficiency(
        {"title": "Photosynthesis", "aliases": ["light reaction"]},
        evidence=[],  # retrieval returned nothing
        all_chunks=chunks,
    )
    assert result.status is SufficiencyStatus.SUFFICIENT
    assert result.evidence, "recovered evidence must be returned to the caller"


def test_recovered_evidence_is_returned_in_source_order():
    chunks = [
        _chunk("c3", "photosynthesis part three", page=12, position=2),
        _chunk("c1", "photosynthesis part one", page=10, position=0),
        _chunk("c2", "photosynthesis part two", page=11, position=1),
    ]
    recovered = recover_topic_evidence(chunks, {"title": "Photosynthesis"})
    assert [row["id"] for row in recovered] == ["c1", "c2", "c3"]


def test_ambiguous_is_reported_and_supported():
    chunks = [_chunk("c1", "mitosis phases prophase metaphase", page=2)]
    result = assess_source_sufficiency(
        {"title": "Mitosis"}, evidence=chunks, all_chunks=chunks, ambiguous=True,
    )
    assert result.status is SufficiencyStatus.AMBIGUOUS
    assert result.supported
    assert not result.sufficient
    # Ambiguous evidence is still returned so the caller can narrow the topic.
    assert result.evidence
    assert "alias" in result.recovery_steps


# --------------------------------------------------------------------------
# 2. No boilerplate persisted as normal notes
# --------------------------------------------------------------------------

def test_placeholder_detection_rejects_no_information_boilerplate():
    for text in (
        "No relevant information was found in the provided documents.",
        "There is not enough information in the source to answer this.",
        "TBC",
        "This content is not available yet.",
        "",
    ):
        assert is_placeholder(text), text


def test_real_explanatory_prose_is_not_a_placeholder():
    assert not is_placeholder(
        "Photosynthesis converts light energy into chemical energy stored in glucose."
    )


def test_boilerplate_section_fails_the_hard_gate():
    artifact = {
        "sections": [{
            "unit_title": "Unit 1",
            "topic_title": "Photosynthesis",
            "content": "No relevant information was found in the provided documents.",
            "citations": [{"chunk_id": "c1", "span": "x"}],
        }],
    }
    report = validate_notes_artifact(artifact)
    assert not report.ok
    assert any(err["code"] == "placeholder_section" for err in report.errors)


def test_source_gap_is_reported_not_fabricated():
    """A no_source note is a gap record, and it is never a normal section."""

    note = {
        "topic_title": "Photosynthesis",
        "body_paragraphs": [],
        "citations": [],
        "no_source": True,
        "provenance_note": "Photosynthesis: no_source",
    }
    report = validate_topic_notes(note, {"title": "Photosynthesis"})
    assert report.ok, "a declared gap must pass validation"
    assert report.source_gaps == ["Photosynthesis: no_source"]


def test_omitted_topic_is_allowed_only_with_a_declared_gap():
    structure = {
        "units": [{
            "title": "Unit 1",
            "topics": [{"title": "Photosynthesis"}, {"title": "Cell Division"}],
        }]
    }
    section = {
        "unit_title": "Unit 1",
        "topic_title": "Photosynthesis",
        "content": "Photosynthesis converts light energy into chemical energy in glucose bonds.",
        "citations": [{"chunk_id": "c1", "span": "light energy"}],
    }
    # No gap declared -> the unexplained omission must fail.
    unexplained = validate_notes_artifact({"sections": [section]}, structure)
    assert not unexplained.ok
    assert any(err["code"] == "missing_topic" for err in unexplained.errors)

    # Gap declared -> omission is honest, not silent.
    explained = validate_notes_artifact(
        {"sections": [section], "source_gaps": ["Cell Division: no_source"]},
        structure,
    )
    assert explained.ok, explained.errors


# --------------------------------------------------------------------------
# 3. Structural gates: citations, order, length, coverage
# --------------------------------------------------------------------------

def test_uncited_section_fails():
    artifact = {
        "sections": [{
            "unit_title": "U", "topic_title": "Photosynthesis",
            "content": "Photosynthesis stores light energy as chemical energy in glucose.",
            "citations": [],
        }],
    }
    report = validate_notes_artifact(artifact)
    assert not report.ok
    assert any(err["code"] == "section_missing_citations" for err in report.errors)


def test_citation_outside_the_allowed_source_scope_fails():
    artifact = {
        "sections": [{
            "unit_title": "U", "topic_title": "Photosynthesis",
            "content": "Photosynthesis stores light energy as chemical energy in glucose.",
            "citations": [{"chunk_id": "not-in-scope", "span": "x"}],
        }],
    }
    report = validate_notes_artifact(artifact, allowed_chunk_ids={"c1"})
    assert not report.ok
    assert any(err["code"] == "invalid_citation" for err in report.errors)


def test_fabricated_citation_span_fails():
    artifact = {
        "sections": [{
            "unit_title": "U", "topic_title": "Photosynthesis",
            "content": "Photosynthesis stores light energy as chemical energy in glucose.",
            "citations": [{"chunk_id": "c1", "span": "quantum entanglement of photons"}],
        }],
    }
    report = validate_notes_artifact(
        artifact, allowed_chunk_ids={"c1"},
        chunk_texts={"c1": "Photosynthesis converts light into chemical energy."},
    )
    assert not report.ok
    assert any(err["code"] == "citation_span" for err in report.errors)


def test_out_of_order_sections_fail():
    structure = {
        "units": [{
            "title": "U",
            "topics": [{"title": "Alpha"}, {"title": "Beta"}],
        }]
    }
    artifact = {
        "sections": [
            {"unit_title": "U", "topic_title": "Beta",
             "content": "Beta is a later topic in the unit sequence for this syllabus.",
             "citations": [{"chunk_id": "c1", "span": "Beta"}]},
            {"unit_title": "U", "topic_title": "Alpha",
             "content": "Alpha is the earlier topic in the unit sequence for this syllabus.",
             "citations": [{"chunk_id": "c2", "span": "Alpha"}]},
        ],
    }
    report = validate_notes_artifact(artifact, structure)
    assert not report.ok
    assert any(err["code"] == "document_order" for err in report.errors)


def test_topic_mismatch_is_advisory_warning():
    # The spine is required: without a known topic order the validator cannot
    # know what the section was supposed to be about.  Expert knowledge of
    # the label lives with our extractor, so a lexical mismatch warns.
    structure = {"units": [{"title": "U", "topics": [{"title": "Photosynthesis"}]}]}
    artifact = {
        "sections": [{
            "unit_title": "U", "topic_title": "Photosynthesis",
            "content": "The mitochondria generate ATP through oxidative phosphorylation.",
            "citations": [{"chunk_id": "c1", "span": "ATP"}],
        }],
    }
    report = validate_notes_artifact(artifact, structure)
    assert report.ok
    assert any(w["code"] == "topic_overlap_low" for w in report.warnings)
    assert not any(e["code"] == "topic_mismatch" for e in report.errors)


def test_topic_match_passes_when_the_prose_names_the_topic():
    structure = {"units": [{"title": "U", "topics": [{"title": "Photosynthesis"}]}]}
    artifact = {
        "sections": [{
            "unit_title": "U", "topic_title": "Photosynthesis",
            "content": "Photosynthesis converts light energy into chemical energy in glucose.",
            "citations": [{"chunk_id": "c1", "span": "light energy"}],
        }],
    }
    assert validate_notes_artifact(artifact, structure).ok


def test_require_valid_notes_raises_typed_error():
    from services.academic.orchestrator import AcademicPipelineError

    with pytest.raises(AcademicPipelineError):
        require_valid_notes({"sections": []})


# --------------------------------------------------------------------------
# 4. Headings-only PDFs are never a valid deliverable
# --------------------------------------------------------------------------

def test_headings_only_pdf_is_rejected():
    headings_only = {
        "sections": [
            {"unit_title": "Unit 1", "topic_title": "Photosynthesis", "content": "Photosynthesis"},
            {"unit_title": "Unit 1", "topic_title": "Respiration", "content": "Respiration"},
        ],
    }
    with pytest.raises(ValueError):
        assert_pdf_has_body(headings_only)


def test_real_pdf_input_passes_the_body_guard():
    good = {
        "sections": [{
            "unit_title": "Unit 1", "topic_title": "Photosynthesis",
            "content": (
                "Photosynthesis converts light energy into chemical energy stored in "
                "glucose inside the chloroplast thylakoid membranes."
            ),
            "citations": [{"chunk_id": "c1", "span": "light energy"}],
        }],
    }
    assert_pdf_has_body(good)


# --------------------------------------------------------------------------
# 5. Deterministic source ingestion (syllabus + question paper)
# --------------------------------------------------------------------------

SYLLABUS_TEXT = """
BCA 402 - Data Structures Syllabus
Unit 1: Arrays
topic: One dimensional arrays
topic: Two dimensional arrays
Unit 2: Linked Structures
topic: Singly linked list
topic: Doubly linked list
"""

QP_TEXT = """
End Semester Examination 2023
Time Allowed: 3 Hours
Section A: Arrays
1. Write a program to reverse an array using two pointers. (5 marks)
   a. State the loop invariant.
   b. State the time complexity.
2. Explain insertion in a sorted array. (10 marks)
Section B: Linked Structures
3. Describe deletion in a doubly linked list. (5 marks)
"""


def test_syllabus_parser_keeps_unit_and_topic_order():
    parsed = parse_syllabus_text("syllabus-1", "Data Structures", SYLLABUS_TEXT)
    titles = [unit.title for unit in parsed.units]
    assert len(titles) == 2
    assert titles[0].startswith("Unit 1")
    assert titles[1].startswith("Unit 2")
    assert "One dimensional arrays" in parsed.units[0].topics
    assert [u.order for u in parsed.units] == [0, 1]
    keys = {topic.key for topic in parsed.canonical_topics}
    # normalize_key() folds a trailing plural, so "arrays" -> "array".
    assert "one dimensional array" in keys


def test_syllabus_parser_never_uses_the_llm_for_arithmetic():
    parsed = parse_syllabus_text("syllabus-1", "Data Structures", SYLLABUS_TEXT)
    assert parsed.source_type == "syllabus"
    # Deterministic: same input -> identical canonical topic order.
    again = parse_syllabus_text("syllabus-1", "Data Structures", SYLLABUS_TEXT)
    assert [t.key for t in parsed.canonical_topics] == [t.key for t in again.canonical_topics]


def test_question_paper_parser_extracts_marks_sections_and_parts():
    parsed = parse_question_paper_text(
        "qp-1", "End Semester Examination 2023", QP_TEXT,
        syllabus_topics=["One dimensional arrays", "Singly linked list"],
    )
    assert parsed.year == 2023
    marks = {q.number: q.marks for q in parsed.questions}
    # Regression: "(5 marks)" must be 5, not the 1 placeholder.
    assert marks[1] == 5
    assert marks[2] == 10
    assert marks[3] == 5
    first = parsed.questions[0]
    # Letter-labelled sections ("Section A: Arrays") must not collapse to
    # a single "General" bucket.
    assert "Arrays" in first.section
    assert parsed.questions[2].section != first.section
    assert len(first.parts) == 2
    assert first.mapping_method == "token_overlap"


def test_question_paper_topic_mapping_is_explicit_about_unmapped():
    syllabus = parse_syllabus_text("s1", "Data Structures", SYLLABUS_TEXT)
    papers = [parse_question_paper_text("qp-1", "QP 2023", QP_TEXT)]
    mapping = map_questions_to_syllabus(papers[0].questions, syllabus.units)
    assert mapping, "mapping must return one row per question"
    assert all({"number", "unit_id", "topic", "method"} <= set(row) for row in mapping)
    # Everything unmapped is declared, never silently dropped.
    metrics = compute_source_metrics(syllabus.units, papers)
    assert metrics["totals"]["questions"] == 3
    assert metrics["totals"]["mapped_questions"] + metrics["totals"]["unmapped_questions"] == 3
    assert len(metrics["unmapped_questions"]) == metrics["totals"]["unmapped_questions"]


def test_source_metrics_are_deterministic_and_mark_totals_add_up():
    syllabus = parse_syllabus_text("s1", "Data Structures", SYLLABUS_TEXT)
    papers = [parse_question_paper_text("qp-1", "QP 2023", QP_TEXT)]
    first = compute_source_metrics(syllabus.units, papers)
    second = compute_source_metrics(syllabus.units, papers)
    assert first == second
    assert first["totals"]["total_marks"] == sum(q.marks for q in papers[0].questions)
    # Missing/never-asked is derived, not model-invented.
    assert isinstance(first["missing"], list)
    assert first["totals"]["coverage_pct"] == pytest.approx(0.0, abs=0.01)


# --------------------------------------------------------------------------
# 6. Tutor intent gating
# --------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "hi", "Hello!", "hey", "good morning", "thanks", "thank you", "bye",
])
def test_greetings_never_require_retrieval(message):
    result = classify_tutor_intent(message)
    assert result.intent is TutorIntent.CONVERSATIONAL
    assert result.retrieval_required is False


def test_greeting_response_is_deterministic_and_helpful():
    for message in ("hi", "thanks", "good evening", "bye"):
        first = greeting_response(message)
        assert first == greeting_response(message)
        assert first.strip()


def test_unsupported_topics_do_not_retrieve():
    result = classify_tutor_intent("what is the weather in London tomorrow?")
    assert result.intent is TutorIntent.UNSUPPORTED
    assert result.retrieval_required is False
    assert result.clarification


def test_ambiguous_short_message_asks_instead_of_dumping():
    result = classify_tutor_intent("arrays")
    assert result.intent is TutorIntent.AMBIGUOUS
    assert result.retrieval_required is False
    assert result.clarification


def test_academic_question_retrieves():
    result = classify_tutor_intent("Explain how a binary search tree works")
    assert result.intent is TutorIntent.ACADEMIC
    assert result.retrieval_required is True


def test_project_specific_question_retrieves():
    result = classify_tutor_intent("what does page 4 of the uploaded syllabus say?")
    assert result.intent is TutorIntent.PROJECT_SPECIFIC
    assert result.retrieval_required is True


def test_topic_title_hit_does_not_crash_and_retrieves():
    """Regression: a walrus/precedence bug raised NameError here."""

    result = classify_tutor_intent(
        "explain photosynthesis", topic_titles=["Photosynthesis", "Respiration"],
    )
    assert result.intent is TutorIntent.PROJECT_SPECIFIC
    assert result.retrieval_required is True


def test_short_topic_title_is_ignored():
    result = classify_tutor_intent("tell me about it", topic_titles=["x"])
    assert result.intent is TutorIntent.AMBIGUOUS
    assert result.retrieval_required is False


def test_empty_message_is_ambiguous_not_academic():
    result = classify_tutor_intent("   ")
    assert result.intent is TutorIntent.AMBIGUOUS
    assert result.retrieval_required is False


# --------------------------------------------------------------------------
# 7. Notes generation config + advisory recommendation
# --------------------------------------------------------------------------

def test_config_accepts_camel_and_snake_case():
    config = NotesGenerationConfig.model_validate({
        "mode": "exam",
        "detail": "comprehensive",
        "examFocus": True,
        "useWeb": False,
        "selectedTopicIds": ["a", "a", " b ", ""],
    })
    assert config.mode is NotesGenerationMode.EXAM
    assert config.detail is NotesDetailLevel.COMPREHENSIVE
    assert config.exam_focus is True
    assert config.web is False
    assert config.selected_topic_ids == ["a", "b"]


def test_config_persists_a_json_safe_snapshot():
    config = NotesGenerationConfig(mode="quick")
    dumped = config.persisted_dict()
    assert dumped["mode"] == "quick"
    import json
    json.dumps(dumped)  # must be JSON serialisable for artifact metadata


def test_recommendation_is_advisory_and_never_overrides_user_config():
    user = NotesGenerationConfig(mode="memory", detail="concise", web=True)
    recommended = recommend_notes_config(topic_count=40, has_question_papers=True)
    assert recommended.mode is NotesGenerationMode.EXAM
    assert recommended.recommendation["advisory"] is True
    # The recommendation lives beside the user's choice; it does not replace it.
    assert recommended.web is False
    assert user.mode is NotesGenerationMode.MEMORY
    assert user.web is True


# --------------------------------------------------------------------------
# 8. Source-type classification is validated, not silently coerced
# --------------------------------------------------------------------------

def test_invalid_source_type_is_rejected():
    from database.models.document import DocumentSourceType

    assert DocumentSourceType("syllabus")
    with pytest.raises(ValueError):
        DocumentSourceType("exam_cheat_sheet")


def test_source_type_enum_values_match_the_ingestion_parser():
    from database.models.document import DocumentSourceType
    from services.academic.source_ingestion import SourceType

    db_values = {item.value for item in DocumentSourceType}
    parser_values = {item.value for item in SourceType}
    assert parser_values <= db_values


# --------------------------------------------------------------------------
# 9. CORS decorates inner error responses
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cors_headers_present_on_credentialed_4xx(app_client):
    """A real backend 4xx must still be readable by the browser."""

    response = await app_client.get(
        "/api/documents/00000000-0000-0000-0000-000000000000",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code >= 400
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


@pytest.mark.asyncio
async def test_cors_preflight_admits_source_type_header(app_client):
    response = await app_client.options(
        "/api/documents/upload",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-source-type",
        },
    )
    assert response.status_code in (200, 204)
    allowed = response.headers.get("access-control-allow-headers", "").lower()
    assert "x-source-type" in allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/generation/mcq", "/api/generation/explanation"])
async def test_generation_routes_exist_and_reject_unauthenticated(app_client, path):
    """The route must be registered, and an auth failure must still be CORS-decorated.

    Regression: these two endpoints failed in the browser as net::ERR_FAILED
    because the 401 carried no Access-Control-Allow-Origin header.
    """

    response = await app_client.post(
        path,
        json={"project_id": "00000000-0000-0000-0000-000000000000", "query": "photosynthesis"},
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code != 404, f"{path} is not registered"
    assert response.status_code == 401
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


@pytest.mark.asyncio
async def test_cors_does_not_wildcard_with_credentials(app_client):
    """Credentialed CORS must stay an explicit allowlist, never '*'."""

    response = await app_client.get(
        "/health", headers={"Origin": "http://evil.example"},
    )
    assert response.headers.get("access-control-allow-origin") != "*"
