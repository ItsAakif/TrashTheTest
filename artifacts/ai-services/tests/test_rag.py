"""RAG generation tests (T306/T307 via the transformation engine).

Covers: schema validation of successful responses, bounded repair then
typed failure on forced-invalid output, no unvalidated json.loads in the
engine, RAG context trim (relevant-first), citations + grounding flags
(retrieved vs generated vs unsupported), and fabricated-citation
rejection.
"""

import json
import uuid

import pytest

from services.transformation.engine import TransformationEngine
from services.transformation.types import GenerationMode
from services.strategies.llm.structured import StructuredGenerationError


class FakeLLM:
    """Scripted provider: pop canned response texts in order."""

    model_name = "fake-test-llm"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    async def generate(self, prompt, config=None):
        self.calls += 1
        idx = min(self.calls - 1, len(self.texts) - 1)
        from services.strategies.base import GenerationResult as _GR

        return _GR(text=self.texts[idx], model=self.model_name, usage={},
                   finish_reason="stop", processing_time=0.0)

    async def chat(self, messages, config=None):
        prompt = messages[0].content if messages else ""
        return await self.generate(prompt, config)


def valid_flashcards(n=2):
    cards = [{"front": f"Q{i}?", "back": f"A{i}", "difficulty": "easy"} for i in range(n)]
    return json.dumps({"flashcards": cards})


def fake_retrieval(n=3):
    from services.retrieval.types import RetrievedChunk, RetrievalResult

    chunks = [
        RetrievedChunk(chunk_id=str(uuid.uuid4()), content=f"content piece {i} " * 20,
                       similarity=0.9 - i * 0.1, metadata={}, document_id=str(uuid.uuid4()),
                       page=i + 1, section=f"Ch {i}", score=0.9 - i * 0.1,
                       source="both", citation=None)
        for i in range(n)
    ]
    return RetrievalResult(query="q", total_results=n, chunks=chunks, processing_time=0.1)


@pytest.mark.asyncio
async def test_successful_response_validates_against_model():
    llm = FakeLLM([valid_flashcards()])
    engine = TransformationEngine(llm_provider=llm)
    out = await engine._generate_structured(GenerationMode.FLASHCARDS, "prompt", {"x"})
    assert len(out["flashcards"]) == 2
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_forced_invalid_triggers_bounded_repairs_then_typed_failure():
    llm = FakeLLM(["not json at all {{{", "still bad", "bad again", "bad forever"])
    engine = TransformationEngine(llm_provider=llm)
    with pytest.raises(StructuredGenerationError):
        await engine._generate_structured(GenerationMode.FLASHCARDS, "prompt", set())
    assert llm.calls == 3  # initial + at most two repairs


@pytest.mark.asyncio
async def test_repair_recovers_after_validation_error():
    llm = FakeLLM([json.dumps({"wrong": "shape"}), valid_flashcards(1)])
    engine = TransformationEngine(llm_provider=llm)
    out = await engine._generate_structured(GenerationMode.FLASHCARDS, "prompt", set())
    assert len(out["flashcards"]) == 1
    assert llm.calls == 2


def test_no_unvalidated_json_loads_in_engine():
    import pathlib

    src = pathlib.Path("services/transformation/engine.py").read_text()
    assert "json.loads" not in src, "engine must route JSON through structured validation"


@pytest.mark.asyncio
async def test_fabricated_citation_id_rejected():
    from services.strategies.llm.structured import StructuredGenerationError

    fabricated = str(uuid.uuid4())
    allowed = str(uuid.uuid4())
    engine = TransformationEngine(llm_provider=FakeLLM(["{}"]))
    payload = {"flashcards": [{"front": "Q?", "back": f"see chunk {fabricated}",
                               "difficulty": "easy"}]}
    with pytest.raises(StructuredGenerationError):
        engine._check_citations(payload, {allowed})


def test_rag_trim_keeps_relevant_first():
    engine = TransformationEngine(llm_provider=FakeLLM(["{}"]))
    parts = [f"[source chunk_id={i}] " + ("x" * 5000) for i in range(5)]
    context = "\n\n---\n\n".join(parts)
    trimmed = engine._trim_context(context, max_tokens=1000)
    assert len(trimmed) <= 4000
    assert "[source chunk_id=0]" in trimmed  # earliest (most relevant) kept
    assert "[source chunk_id=4]" not in trimmed  # tail dropped first


@pytest.mark.asyncio
async def test_generation_carries_citations_and_grounding(monkeypatch):
    import services.transformation.engine as _eng

    llm = FakeLLM([valid_flashcards()])
    engine = TransformationEngine(llm_provider=llm)
    retrieval = fake_retrieval(2)

    async def fake_retrieve(**kwargs):
        return retrieval

    async def no_cache(*args, **kwargs):
        return None

    async def no_persist(*args, **kwargs):
        # Domain-row persist is covered by test_phaseB_persist.py with a
        # fake DB; these grounding tests run db=None by design.
        return None

    monkeypatch.setattr(_eng.retrieval_service, "retrieve_with_db", fake_retrieve)
    monkeypatch.setattr(TransformationEngine, "_check_cache", no_cache)
    monkeypatch.setattr(TransformationEngine, "_save_to_cache", no_cache)
    monkeypatch.setattr(TransformationEngine, "_persist_domain_rows", no_persist)

    from services.transformation.types import GenerationRequest

    result = await engine.generate(
        GenerationRequest(project_id=str(uuid.uuid4()), mode=GenerationMode.FLASHCARDS,
                          query="q"),
        db=None,
    )
    assert result.content["citations"] and len(result.content["citations"]) == 2
    assert result.content["grounding"]["support"] == "retrieved"
    assert result.tokens_used > 0
    assert result.content["attribution"]["provider"]


@pytest.mark.asyncio
async def test_unsupported_answer_flagged_not_cited(monkeypatch):
    import services.transformation.engine as _eng

    llm = FakeLLM([valid_flashcards()])
    engine = TransformationEngine(llm_provider=llm)
    empty = fake_retrieval(0)

    async def fake_retrieve(**kwargs):
        return empty

    async def no_cache(*args, **kwargs):
        return None

    async def no_persist(*args, **kwargs):
        # Domain-row persist is covered by test_phaseB_persist.py with a
        # fake DB; these grounding tests run db=None by design.
        return None

    monkeypatch.setattr(_eng.retrieval_service, "retrieve_with_db", fake_retrieve)
    monkeypatch.setattr(TransformationEngine, "_check_cache", no_cache)
    monkeypatch.setattr(TransformationEngine, "_save_to_cache", no_cache)
    monkeypatch.setattr(TransformationEngine, "_persist_domain_rows", no_persist)
    from services.transformation.types import GenerationRequest

    result = await engine.generate(
        GenerationRequest(project_id=str(uuid.uuid4()), mode=GenerationMode.FLASHCARDS,
                          query="q"),
        db=None,
    )
    assert result.content["citations"] == []
    assert result.content["grounding"]["support"] == "unsupported"
    assert result.content["grounding"]["generated_without_citation"] is True
