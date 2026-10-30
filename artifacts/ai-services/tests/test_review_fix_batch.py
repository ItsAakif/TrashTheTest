"""Focused regressions for the reviewer defect-fix batch."""

import uuid
from types import SimpleNamespace

import pytest

from services.academic import orchestrator
from services.academic.sufficiency import SufficiencyStatus, assess_source_sufficiency
from services.academic.validation import (
    assert_pdf_has_body,
    validate_notes_artifact,
    validate_topic_notes,
)
from services.academic.routes import HierarchicalNotesRequest


def _chunk(cid, text, page, position=0):
    return {
        "id": cid,
        "chunk_id": cid,
        "text": text,
        "content": text,
        "page": page,
        "position": position,
    }


def test_nonempty_irrelevant_evidence_recovers_and_source_sorts():
    chunks = [
        _chunk("irrelevant", "Unrelated bookkeeping notes", 9),
        _chunk("relevant", "Photosynthesis converts light into chemical energy.", 2),
    ]
    rerank_calls = []

    def rerank(query, rows):
        rerank_calls.append((query, rows))
        return rows

    result = assess_source_sufficiency(
        {"title": "Photosynthesis"},
        evidence=[chunks[0]],
        all_chunks=chunks,
        rerank=rerank,
    )

    assert result.status is SufficiencyStatus.SUFFICIENT
    assert [row["id"] for row in result.evidence] == ["relevant", "irrelevant"]
    assert rerank_calls


def test_generic_gap_does_not_waive_other_topics_and_gap_payloads_are_empty():
    structure = {
        "units": [{
            "title": "Unit 1",
            "topics": [{"title": "Alpha"}, {"title": "Beta"}],
        }]
    }
    section = {
        "unit_title": "Unit 1",
        "topic_title": "Alpha",
        "content": "Alpha explains the source-backed concept in enough detail.",
        "citations": [{"chunk_id": "c1", "span": "source-backed"}],
    }
    report = validate_notes_artifact(
        {"sections": [section], "source_gaps": ["no_source"]}, structure,
    )
    assert not report.ok
    assert any(error["code"] == "missing_topic" for error in report.errors)

    gap = validate_topic_notes(
        {
            "topic_title": "Beta",
            "body_paragraphs": ["Fabricated body text that should not be accepted."],
            "citations": [{"chunk_id": "c1", "span": "text"}],
            "no_source": True,
            "provenance_note": "Beta: no_source",
        },
        {"title": "Beta"},
    )
    assert not gap.ok
    assert {error["code"] for error in gap.errors} == {
        "no_source_has_body", "no_source_has_citations",
    }


def test_long_title_only_section_is_rejected_with_render_scope():
    title = "A comprehensive title about photosynthesis and cellular energy conversion"
    structure = {"units": [{"title": "Unit 1", "topics": [{"title": title}]}]}
    artifact = {
        "sections": [{
            "unit_title": "Unit 1",
            "topic_title": title,
            "heading": title,
            "content": title,
            "citations": [{"chunk_id": "c1", "span": title}],
        }],
    }
    with pytest.raises(ValueError, match="title_only_section"):
        assert_pdf_has_body(
            artifact,
            structure=structure,
            allowed_chunk_ids={"c1"},
            chunk_texts={"c1": title},
        )


def test_source_dispatch_persists_qp_parts_marks_and_mapping():
    syllabus_rows = [{
        **_chunk(
            "s1",
            "Unit 1: Arrays\ntopic: One dimensional arrays",
            1,
        ),
        "document_id": "syllabus-doc",
        "source_type": "syllabus",
    }]
    syllabus, parsed_syllabus, _ = orchestrator._build_dispatch_structure(
        "syllabus-doc", "Syllabus", syllabus_rows, [],
    )
    assert syllabus.source_type == "syllabus"
    assert syllabus.source_metadata["syllabus"]["units"]

    question_text = (
        "Section A: Arrays\n"
        "1. Explain a one dimensional array. (5 marks)\n"
        "a. State the invariant. (2 marks)\n"
        "b. State the complexity. (3 marks)"
    )
    question_rows = [{
        **_chunk("q1", question_text, 2),
        "document_id": "paper-doc",
        "source_type": "question_paper",
    }]
    paper, _, parsed_paper = orchestrator._build_dispatch_structure(
        "paper-doc", "Question paper", question_rows, parsed_syllabus.units,
    )
    question = parsed_paper.questions[0]
    assert question.topic == "One dimensional arrays"
    assert question.unit_id
    assert [part.marks for part in question.parts] == [2, 3]
    assert paper.source_metadata["mapping"][0]["method"] == "token_overlap"
    assert paper.source_metadata["questions"][0]["parts"][0]["marks"] == 2

    merged = orchestrator.merge_document_structures([syllabus, paper])
    assert "syllabus" in merged.source_metadata
    assert "question_paper" in merged.source_metadata


def test_source_metric_projection_uses_written_back_topic_mapping():
    syllabus_rows = [{
        **_chunk("s1", "Unit 1: Arrays\ntopic: One dimensional array", 1),
        "document_id": "syllabus-doc",
        "source_type": "syllabus",
    }]
    syllabus, parsed_syllabus, _ = orchestrator._build_dispatch_structure(
        "syllabus-doc", "Syllabus", syllabus_rows, [],
    )
    paper_rows = [{
        **_chunk(
            "q1",
            "Section A: Arrays\n1. Explain a one dimensional array. (5 marks)",
            2,
        ),
        "document_id": "paper-doc",
        "source_type": "question_paper",
    }]
    paper, _, _ = orchestrator._build_dispatch_structure(
        "paper-doc", "Question paper", paper_rows, parsed_syllabus.units,
    )
    merged = orchestrator.merge_document_structures([syllabus, paper])
    units, pyqs, mapping, projection = orchestrator._source_metric_inputs([merged])
    assert units and pyqs and mapping
    assert pyqs[0]["topic"] == "One dimensional array"
    assert projection["metrics"]["totals"]["mapped_questions"] == 1


def test_hierarchical_request_accepts_explicit_null_config():
    request = HierarchicalNotesRequest(project_id="p1", config=None)
    assert request.config is None


@pytest.mark.asyncio
async def test_hierarchical_route_normalizes_missing_config_and_attaches_advice(monkeypatch):
    from services.academic import routes
    from services.academic.types import NotesGenerationConfig, recommend_notes_config

    project = SimpleNamespace(id=uuid.uuid4(), name="Study")
    captured = {}

    async def owned(*args, **kwargs):
        return project

    async def advice(*args, **kwargs):
        return recommend_notes_config(topic_count=20, has_question_papers=True)

    async def selection(*args, **kwargs):
        return {}

    async def enqueue(db, job_type, input_data=None, **kwargs):
        captured.update(input_data or {})
        return SimpleNamespace(id=uuid.uuid4())

    class _DB:
        async def commit(self):
            return None

    monkeypatch.setattr(routes, "_owned_project", owned)
    monkeypatch.setattr(routes, "_project_notes_recommendation", advice)
    monkeypatch.setattr(routes, "_safe_provider_selection", selection)
    monkeypatch.setattr(routes.job_queue, "enqueue", enqueue)

    result = await routes.start_hierarchical_notes(
        routes.HierarchicalNotesRequest(project_id=str(project.id)),
        current_user=SimpleNamespace(id=uuid.uuid4()),
        db=_DB(),
    )
    assert result["job_id"]
    assert captured["config"]["mode"] == "detailed"
    assert captured["config"]["recommendation"]["advisory"] is True
    assert "api_key" not in repr(captured["provider_selection"])


@pytest.mark.asyncio
async def test_academic_provider_selection_persists_no_raw_key(monkeypatch):
    from services.academic import routes
    from services.provider_credentials import override

    async def fake_resolve(*args, **kwargs):
        return override.set_request_llm({
            "provider": "groq",
            "model": "model-x",
            "source": "user",
            "using_user_key": True,
            "key_hint": "****abcd",
            "api_key": "must-not-be-persisted",
        })

    monkeypatch.setattr(override, "resolve_for_request", fake_resolve)
    selection = await routes._safe_provider_selection(
        object(), SimpleNamespace(id=uuid.uuid4()), SimpleNamespace(id=uuid.uuid4()),
    )
    assert selection["provider"] == "groq"
    assert "api_key" not in selection
    assert "must-not-be-persisted" not in repr(selection)


@pytest.mark.asyncio
async def test_tutor_citations_preserve_document_id(monkeypatch):
    from services.study import routes
    from services.study.routes import TutorMessageRequest
    from services.provider_credentials import override
    from services.retrieval.service import retrieval_service
    from database import repositories

    user_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=user_id, name="Biology")
    chunk = SimpleNamespace(
        chunk_id="c1",
        document_id="document-9",
        content="Photosynthesis converts light into chemical energy.",
        page=4,
        citation=SimpleNamespace(method="dense"),
    )

    async def project_for_request(*args, **kwargs):
        return project

    async def retrieve(*args, **kwargs):
        return SimpleNamespace(chunks=[chunk])

    async def generate(self, prompt, config=None):
        return SimpleNamespace(text="Photosynthesis stores energy in glucose.")

    monkeypatch.setattr(repositories, "get_project_by_id", project_for_request)
    monkeypatch.setattr(retrieval_service, "retrieve_with_db", retrieve)
    monkeypatch.setattr(override.EffectiveLLMProvider, "generate", generate)
    result = await routes.tutor_message(
        TutorMessageRequest(project_id=str(project_id), message="Explain photosynthesis"),
        current_user=SimpleNamespace(id=user_id),
        db=object(),
    )
    assert result["citations"][0]["document_id"] == "document-9"


@pytest.mark.asyncio
async def test_idempotent_upload_returns_stored_source_type(monkeypatch):
    from services.ingestion import routes
    from database.models.document import DocumentSourceType, DocumentStatus, FileType

    project_id = uuid.uuid4()
    document_id = uuid.uuid4()
    project = SimpleNamespace(id=project_id)
    document = SimpleNamespace(
        id=document_id,
        file_name="syllabus.pdf",
        file_type=FileType.PDF,
        source_type=DocumentSourceType.SYLLABUS,
        processing_status=DocumentStatus.PENDING,
    )
    existing_job = SimpleNamespace(id=uuid.uuid4(), document_id=document_id)

    async def owned(*args, **kwargs):
        return project

    async def find(*args, **kwargs):
        return existing_job

    async def get_document(*args, **kwargs):
        return document

    monkeypatch.setattr(routes, "require_project_access", owned)
    monkeypatch.setattr(routes.job_queue, "find_by_idempotency_key", find)
    monkeypatch.setattr(routes.repositories, "get_document_by_id", get_document)

    class _Upload:
        filename = "syllabus.pdf"

    result = await routes.upload_document(
        file=_Upload(),
        project_id=str(project_id),
        source_type="study_material",
        current_user=SimpleNamespace(id=uuid.uuid4()),
        db=object(),
        idempotency_key="same-request",
    )
    assert result.source_type == "syllabus"
    assert result.model_dump(by_alias=True)["sourceType"] == "syllabus"


@pytest.mark.asyncio
async def test_mcq_provider_fault_is_typed_502(monkeypatch):
    from fastapi import HTTPException
    from services.transformation import routes
    from services.strategies.llm.router import ProviderUnavailable
    from services.transformation.routes import MCQRequest

    async def owned(*args, **kwargs):
        return SimpleNamespace(id=uuid.uuid4())

    async def select(*args, **kwargs):
        return None

    async def fail(*args, **kwargs):
        raise ProviderUnavailable("provider unavailable")

    monkeypatch.setattr(routes, "require_project_access", owned)
    monkeypatch.setattr(routes, "_select_request_llm", select)
    monkeypatch.setattr(routes.transformation_engine, "generate_mcq", fail)

    with pytest.raises(HTTPException) as exc:
        await routes.generate_mcq(
            MCQRequest(project_id=str(uuid.uuid4())),
            current_user=SimpleNamespace(id=uuid.uuid4()),
            db=object(),
        )
    assert exc.value.status_code == 502
    assert exc.value.detail["code"] == "generation_provider_failed"
