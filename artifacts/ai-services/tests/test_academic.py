"""Academic Intelligence vertical tests. Offline (stub LLM, fake retrieval).

Covers: strict schema validity (Syllabus/SyllabusUnit/PYQQuestion/PYQSet/
MappingResult/AnalysisResult, no silent malformed accept), grounding /
citation validation reuse, PYQ duplicate detection, deterministic
unit mapping accuracy, frequency computation, marks alignment, display
sanitising, and the syllabus/PYQ generation pipeline (retrieval ->
structured max-2-repairs -> typed failure) plus mapping/analysis
routes with ownership scoping.
"""

import json
import uuid

import pytest
from pydantic import ValidationError

from services.strategies.base import GenerationResult
from services.strategies.llm.structured import StructuredGenerationError
from services.transformation import engine as enginemod
from services.transformation.engine import (
    TransformationEngine,
    compute_frequency,
    dedupe_pyqs,
    map_questions_to_units,
    sanitise_display_text,
    score_unit_for_question,
    validate_grounding,
)
from services.transformation.types import (
    AnalysisResult,
    GenerationMode,
    MappingResult,
    PYQQuestion,
    PYQSet,
    Syllabus,
    SyllabusUnit,
)

pytestmark = pytest.mark.asyncio

CID_A = str(uuid.uuid4())
CID_B = str(uuid.uuid4())
FABRICATED = str(uuid.uuid4())


def _result(text, model="stub"):
    return GenerationResult(text=text, model=model, usage={},
                            finish_reason="stop", processing_time=0.0)


class StubLLM:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    async def generate(self, prompt, config=None):
        self.calls += 1
        idx = min(self.calls - 1, len(self.texts) - 1)
        return _result(self.texts[idx])

    async def chat(self, messages, config=None):
        return await self.generate(messages[0].content if messages else "")


def _syllabus_payload():
    return {
        "title": "Physics syllabus",
        "course": "BSc Physics",
        "units": [
            {"unit_id": "unit-1", "title": "Mechanics",
             "topics": ["Newton laws of motion", "Friction"],
             "outcomes": ["Solve motion problems"],
             "citations": [{"chunk_id": CID_A, "span": "newton laws of motion"}]},
            {"unit_id": "unit-2", "title": "Thermodynamics",
             "topics": ["Laws of thermodynamics", "Entropy"],
             "citations": [{"chunk_id": CID_B, "span": "laws of thermodynamics"}]},
        ],
    }


def _pyq_payload():
    return {
        "title": "Physics PYQs",
        "questions": [
            {"question": "State Newton laws of motion and explain friction.",
             "year": 2023, "marks": 5, "topic": "Motion",
             "citations": [{"chunk_id": CID_A, "span": "newton laws of motion"}]},
            {"question": "Define entropy under laws of thermodynamics.",
             "year": 2022, "marks": 2,
             "citations": [{"chunk_id": CID_B, "span": "laws of thermodynamics"}]},
        ],
    }


# -- schema validity --------------------------------------------------------

async def test_syllabus_schema_valid_and_rejects_malformed():
    syl = Syllabus(**_syllabus_payload())
    assert [u.unit_id for u in syl.units] == ["unit-1", "unit-2"]
    with pytest.raises(ValidationError):
        Syllabus(title="", units=_syllabus_payload()["units"])
    with pytest.raises(ValidationError):
        Syllabus(title="T", units=[])
    dupes = _syllabus_payload()
    dupes["units"] = [dict(dupes["units"][0]), dict(dupes["units"][0])]
    with pytest.raises(ValidationError):
        Syllabus(**dupes)
    with pytest.raises(ValidationError):
        Syllabus(**{**_syllabus_payload(), "bogus_key": "no silent extras"})
    with pytest.raises(ValidationError):
        SyllabusUnit(unit_id="u", title="T", topics=[""],
                     citations=[])
    with pytest.raises(ValidationError):
        SyllabusUnit(unit_id="u", title="T", weight_percent=150)


async def test_pyq_schema_valid_and_rejects_malformed():
    pyq = PYQSet(**_pyq_payload())
    assert pyq.questions[0].year == 2023 and pyq.questions[0].marks == 5
    with pytest.raises(ValidationError):
        PYQQuestion(question="Q?", year=1800, marks=5)
    with pytest.raises(ValidationError):
        PYQQuestion(question="Q?", year=2023, marks=0)
    with pytest.raises(ValidationError):
        PYQQuestion(question="", year=2023, marks=5)
    with pytest.raises(ValidationError):
        PYQSet(title="T", questions=[])
    with pytest.raises(ValidationError):
        PYQSet(**{**_pyq_payload(), "bogus_key": "nope"})


async def test_mapping_analysis_schema_consistency():
    good_map = {
        "syllabus_title": "Physics syllabus",
        "mappings": [
            {"question_index": 0, "question": "Newton laws?",
             "unit_id": "unit-1", "unit_title": "Mechanics",
             "confidence": 0.8, "method": "token_overlap"},
            {"question_index": 1, "question": "Unrelated?",
             "unit_id": None, "unit_title": None,
             "confidence": 0.0, "method": "unmapped"},
        ],
        "unmapped_count": 1,
        "coverage_pct": 50.0,
    }
    assert MappingResult(**good_map).coverage_pct == 50.0
    with pytest.raises(ValidationError):
        MappingResult(**{**good_map, "unmapped_count": 0})
    with pytest.raises(ValidationError):
        MappingResult(**{**good_map, "coverage_pct": 99.0})
    with pytest.raises(ValidationError):
        MappingResult(**{**good_map, "bogus": 1})
    good_analysis = {
        "syllabus_title": "Physics syllabus",
        "total_questions": 2,
        "mapped_questions": 1,
        "unmapped_questions": 1,
        "frequency": [{"unit_id": "unit-1", "unit_title": "Mechanics",
                       "count": 1, "marks_total": 5, "share_pct": 100.0},
                      {"unit_id": "unit-2", "unit_title": "Thermodynamics",
                       "count": 0, "marks_total": 0, "share_pct": 0.0}],
        "total_marks": 5,
        "coverage_pct": 50.0,
    }
    assert AnalysisResult(**good_analysis).total_marks == 5
    with pytest.raises(ValidationError):
        AnalysisResult(**{**good_analysis, "mapped_questions": 2})
    with pytest.raises(ValidationError):
        AnalysisResult(**{**good_analysis, "total_marks": 99})
    with pytest.raises(ValidationError):
        AnalysisResult(**{**good_analysis, "extra": "forbidden"})


# -- grounding (reused gates) -------------------------------------------------

async def test_academic_grounding_reuses_engine_gates():
    chunk_texts = {CID_A: "Newton laws of motion govern friction problems.",
                   CID_B: "Laws of thermodynamics define entropy."}
    gate = validate_grounding(_syllabus_payload(), {CID_A, CID_B}, chunk_texts)
    assert gate["unsupported"] == []
    assert sorted(gate["cited"]) == sorted([CID_A, CID_B])
    gate2 = validate_grounding(_pyq_payload(), {CID_A, CID_B}, chunk_texts)
    assert gate2["unsupported"] == []
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        validate_grounding({"citations": [{"chunk_id": FABRICATED}]},
                           {CID_A, CID_B})


# -- duplicates ---------------------------------------------------------------

async def test_dedupe_pyqs_collapses_exact_repeats_keeps_years():
    qs = [
        {"question": "State Newton laws?", "year": 2023, "marks": 5},
        {"question": "  state NEWTON laws! ", "year": 2023, "marks": 5},
        {"question": "State Newton laws?", "year": 2022, "marks": 5},
        {"question": "Define entropy?", "year": 2022, "marks": 2},
    ]
    out = dedupe_pyqs(qs)
    assert [(q["question"], q["year"]) for q in out] == [
        ("State Newton laws?", 2023),
        ("State Newton laws?", 2022),
        ("Define entropy?", 2022),
    ]


# -- mapping accuracy -----------------------------------------------------------

async def test_mapping_accuracy_on_fixture():
    units = _syllabus_payload()["units"]
    questions = _pyq_payload()["questions"]
    mapped = map_questions_to_units(questions, units)
    assert mapped["mappings"][0]["unit_id"] == "unit-1"
    assert mapped["mappings"][0]["method"] == "token_overlap"
    assert mapped["mappings"][1]["unit_id"] == "unit-2"
    assert mapped["unmapped_count"] == 0
    assert mapped["coverage_pct"] == 100.0


async def test_mapping_reports_unmapped_honestly():
    units = _syllabus_payload()["units"]
    questions = [{"question": "Quantum tunneling in semiconductors overview",
                  "year": 2021, "marks": 3}]
    mapped = map_questions_to_units(questions, units)
    assert mapped["mappings"][0]["unit_id"] is None
    assert mapped["mappings"][0]["method"] == "unmapped"
    assert mapped["unmapped_count"] == 1
    assert mapped["coverage_pct"] == 0.0


async def test_mapping_explicit_unit_wins():
    units = _syllabus_payload()["units"]
    questions = [{"question": "Something entirely different here",
                  "year": 2020, "marks": 1, "unit_id": "unit-2"}]
    mapped = map_questions_to_units(questions, units)
    assert mapped["mappings"][0]["unit_id"] == "unit-2"
    assert mapped["mappings"][0]["method"] == "explicit_unit"
    assert mapped["mappings"][0]["confidence"] == 1.0


async def test_score_unit_for_question_bounds():
    units = _syllabus_payload()["units"]
    assert score_unit_for_question("", units[0]) == 0.0
    score = score_unit_for_question("Newton laws of motion friction", units[0])
    assert 0.0 < score <= 1.0
    other = score_unit_for_question("Newton laws of motion friction", units[1])
    assert score > other


# -- frequency / marks alignment --------------------------------------------------

async def test_frequency_and_marks_alignment():
    units = _syllabus_payload()["units"]
    questions = [
        {"question": "State Newton laws of motion and explain friction.",
         "year": 2023, "marks": 5},
        {"question": "Apply Newton laws of motion to friction problems.",
         "year": 2022, "marks": 3},
        {"question": "Define entropy under laws of thermodynamics.",
         "year": 2022, "marks": 2},
    ]
    mapped = map_questions_to_units(questions, units)["mappings"]
    freq = compute_frequency(mapped, units, questions)
    assert freq["total_questions"] == 3
    assert freq["mapped_questions"] == 3
    assert freq["unmapped_questions"] == 0
    by_unit = {f["unit_id"]: f for f in freq["frequency"]}
    assert by_unit["unit-1"]["count"] == 2
    assert by_unit["unit-1"]["marks_total"] == 8
    assert by_unit["unit-2"]["count"] == 1
    assert by_unit["unit-2"]["marks_total"] == 2
    assert freq["total_marks"] == 10
    assert freq["coverage_pct"] == 100.0
    assert sum(f["count"] for f in freq["frequency"]) == 3


async def test_frequency_includes_zero_units_and_unmapped():
    units = _syllabus_payload()["units"]
    questions = [
        {"question": "State Newton laws of motion and explain friction.",
         "year": 2023, "marks": 5},
        {"question": "Quantum tunneling in semiconductors overview",
         "year": 2021, "marks": 3},
    ]
    mapped = map_questions_to_units(questions, units)["mappings"]
    freq = compute_frequency(mapped, units, questions)
    assert freq["mapped_questions"] == 1
    assert freq["unmapped_questions"] == 1
    by_unit = {f["unit_id"]: f for f in freq["frequency"]}
    assert by_unit["unit-2"]["count"] == 0  # honest gap row, no fake data
    assert by_unit["unit-2"]["marks_total"] == 0
    assert freq["total_marks"] == 5


# -- sanitise ---------------------------------------------------------------------

async def test_sanitise_display_strips_markup():
    dirty = 'Newton <script>alert("x")</script> laws <b>motion</b> onclick="evil"'
    clean = sanitise_display_text(dirty)
    assert "<script>" not in clean and "alert" not in clean
    assert "<b>" not in clean and "Newton" in clean and "motion" in clean


# -- generation pipeline (syllabus / PYQ via shared structured path) ----------------

async def test_generate_syllabus_pipeline_grounded(monkeypatch):
    from services.retrieval.types import RetrievalResult, RetrievedChunk

    stub = StubLLM([json.dumps(_syllabus_payload())])
    engine = TransformationEngine(llm_provider=stub)
    chunk = RetrievedChunk(chunk_id=CID_A,
                           content="Newton laws of motion govern friction. " * 20,
                           similarity=0.9, document_id=str(uuid.uuid4()))
    chunk2 = RetrievedChunk(chunk_id=CID_B,
                            content="Laws of thermodynamics define entropy. " * 20,
                            similarity=0.8, document_id=str(uuid.uuid4()))
    fake_rr = RetrievalResult(query="q", total_results=2, chunks=[chunk, chunk2],
                              processing_time=0.0)

    async def _fake_retrieve(**kwargs):
        return fake_rr

    async def _noop_cache(*a, **k):
        return None

    monkeypatch.setattr(enginemod.retrieval_service, "retrieve_with_db", _fake_retrieve)
    monkeypatch.setattr(engine, "_check_cache", _noop_cache)
    monkeypatch.setattr(engine, "_save_to_cache", _noop_cache)

    result = await engine.generate_syllabus(
        project_id=str(uuid.uuid4()), db=None, query="syllabus")
    assert result.mode == GenerationMode.SYLLABUS
    assert result.content["grounding"]["support"] == "retrieved"
    assert len(result.content["units"]) == 2


async def test_generate_pyq_pipeline_dedupes(monkeypatch):
    from services.retrieval.types import RetrievalResult, RetrievedChunk

    dupes = _pyq_payload()
    dupes["questions"] = [dupes["questions"][0], dict(dupes["questions"][0]),
                          dupes["questions"][1]]
    stub = StubLLM([json.dumps(dupes)])
    engine = TransformationEngine(llm_provider=stub)
    chunk = RetrievedChunk(chunk_id=CID_A,
                           content="Newton laws of motion question paper. " * 20,
                           similarity=0.9, document_id=str(uuid.uuid4()))
    chunk2 = RetrievedChunk(chunk_id=CID_B,
                            content="Laws of thermodynamics entropy paper. " * 20,
                            similarity=0.8, document_id=str(uuid.uuid4()))
    fake_rr = RetrievalResult(query="q", total_results=2, chunks=[chunk, chunk2],
                              processing_time=0.0)

    async def _fake_retrieve(**kwargs):
        return fake_rr

    async def _noop_cache(*a, **k):
        return None

    monkeypatch.setattr(enginemod.retrieval_service, "retrieve_with_db", _fake_retrieve)
    monkeypatch.setattr(engine, "_check_cache", _noop_cache)
    monkeypatch.setattr(engine, "_save_to_cache", _noop_cache)

    result = await engine.generate_pyq(
        project_id=str(uuid.uuid4()), db=None, query="pyq")
    assert result.mode == GenerationMode.PYQ
    assert len(result.content["questions"]) == 2  # exact dupe collapsed


async def test_structured_academic_fails_typed_after_two_repairs():
    from services.strategies.llm.structured import generate_structured

    bad = StubLLM(['{"definitely": "wrong shape"}'] * 3)
    with pytest.raises(StructuredGenerationError):
        await generate_structured(bad, Syllabus, "prompt", max_repairs=2)
    assert bad.calls == 3


# -- prompts carry grounding + security rules ---------------------------------------

async def test_academic_prompts_carry_grounding_and_security():
    from services.transformation.prompts import PromptTemplates

    for mode in (GenerationMode.SYLLABUS, GenerationMode.PYQ):
        prompt = PromptTemplates.get_prompt(mode, "ctx", "Q?")
        assert "GROUNDING RULES" in prompt and "SECURITY RULES" in prompt
    assert "unit_id" in PromptTemplates.get_prompt(GenerationMode.SYLLABUS, "ctx")
    pyq_prompt = PromptTemplates.get_prompt(GenerationMode.PYQ, "ctx")
    assert "year" in pyq_prompt and "marks" in pyq_prompt
    assert "OMIT" in pyq_prompt  # never invent years/marks


# -- routes: academic POSTs + reads, ownership-scoped ---------------------------------

def _install_academic_auth(monkeypatch, app_client_unused, dummy_db, user):
    """Ownership-scoped harness: fake db + fake user (repo pattern)."""
    from database.engine import get_db
    from main import app
    from services.auth.deps import get_current_user

    async def fake_get_db():
        yield dummy_db

    async def fake_user():
        return user

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    return app


async def test_academic_routes_require_ownership(monkeypatch, dummy_db, user_a,
                                                 user_b, app_client):
    import uuid as _uuid

    import services.transformation.routes as troutes
    from fastapi import HTTPException
    from main import app

    _install_academic_auth(monkeypatch, None, dummy_db, user_a)

    async def _deny(project_id, user, db):
        raise HTTPException(status_code=404, detail="Project not found")

    monkeypatch.setattr(troutes, "require_project_access", _deny)
    try:
        async with app_client as client:
            pid = str(_uuid.uuid4())
            for path, body in (
                ("/api/generation/syllabus", {"project_id": pid}),
                ("/api/generation/pyq", {"project_id": pid}),
                ("/api/generation/mapping", {"project_id": pid}),
                ("/api/generation/analysis", {"project_id": pid}),
            ):
                resp = await client.post(path, json=body)
                assert resp.status_code == 404, (path, resp.text)
    finally:
        app.dependency_overrides.clear()


async def test_academic_reads_empty_state_not_500(monkeypatch, dummy_db, user_a,
                                                  app_client):
    from api_gateway.deps import require_project_access
    from database.models.project import Project
    from main import app

    _install_academic_auth(monkeypatch, None, dummy_db, user_a)
    project = Project(id=uuid.uuid4(), user_id=user_a.id, name="P")

    async def _allow(project_id=None, user=None, db=None):
        return project

    app.dependency_overrides[require_project_access] = _allow
    try:
        async with app_client as client:
            for path in (
                f"/api/generation/syllabus/{project.id}",
                f"/api/generation/pyq/{project.id}",
                f"/api/generation/mapping/{project.id}",
                f"/api/generation/analysis/{project.id}",
            ):
                resp = await client.get(path)
                assert resp.status_code == 200, (path, resp.text)
                assert resp.json()["state"] == "not_generated"
    finally:
        app.dependency_overrides.clear()


async def test_mapping_route_needs_syllabus_and_pyqs(monkeypatch, dummy_db, user_a,
                                                     app_client):
    import services.transformation.routes as troutes
    from database.models.project import Project
    from main import app

    _install_academic_auth(monkeypatch, None, dummy_db, user_a)
    project = Project(id=uuid.uuid4(), user_id=user_a.id, name="P")

    async def _allow(project_id, user, db):
        return project

    monkeypatch.setattr(troutes, "require_project_access", _allow)
    try:
        async with app_client as client:
            resp = await client.post(
                "/api/generation/mapping",
                json={"project_id": str(project.id)},
            )
            assert resp.status_code == 400, resp.text
            assert "syllabus" in resp.json()["detail"].lower()
            resp2 = await client.post(
                "/api/generation/analysis",
                json={"project_id": str(project.id)},
            )
            assert resp2.status_code == 400, resp2.text
    finally:
        app.dependency_overrides.clear()


async def test_health_lists_academic_modes(app_client):
    async with app_client as client:
        resp = await client.get("/api/generation/health")
        assert resp.status_code == 200
        for mode in ("syllabus", "pyq", "mapping", "analysis"):
            assert mode in resp.json()["modes"]
