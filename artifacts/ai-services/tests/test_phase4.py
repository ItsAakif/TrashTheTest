"""Phase-4 structured generation tests. Offline (stub LLM, fake retrieval).

Covers: strict schema validity (Note/FlashcardSet/MCQ/MCQSet/Explanation/
Answer, no silent malformed accept), grounding/citation validation,
MCQ correctness (4 distinct options, index in range), duplicates,
hallucination/unsupported guard, marks-aware length/grounding, and the
LangGraph generation-node wiring (answer/notes/flashcards/mcq nodes reuse
the engine helpers; legacy graph paths unchanged).
"""

import json
import uuid

import pytest
from pydantic import ValidationError

from services.strategies.base import GenerationResult
from services.strategies.llm.structured import (
    StructuredGenerationError,
    generate_structured,
)
from services.transformation import engine as enginemod
from services.transformation.engine import (
    TransformationEngine,
    check_hallucination,
    collect_cited_ids,
    dedupe_flashcards,
    dedupe_mcqs,
    normalize_text,
    validate_grounding,
)
from services.transformation.types import (
    Answer,
    Explanation,
    FlashcardSet,
    GenerationMode,
    MCQ,
    MCQSet,
    Note,
    constraints_for_marks,
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


def _answer_payload(marks=5, words=160, cid=CID_A):
    band = constraints_for_marks(marks)
    n = max(words, band["min_words"] + 1)
    return {
        "question": "What is photosynthesis?",
        "answer_text": " ".join(["chlorophyll"] * n),
        "marks": marks,
        "style": "exam",
        "target_min_words": band["min_words"],
        "target_max_words": band["max_words"],
        "citations": [{"chunk_id": cid, "span": "chlorophyll absorbs light"}],
    }


def _mcq_payload(cid=CID_A):
    return {
        "title": "Bio check",
        "questions": [{
            "question": "Which pigment absorbs light?",
            "options": ["Chlorophyll", "Keratin", "Melanin", "Collagen"],
            "correct_answer": 0,
            "explanation": "Chlorophyll absorbs light in leaves.",
            "citations": [{"chunk_id": cid, "span": "chlorophyll absorbs light"}],
        }],
    }


# -- schema validity --------------------------------------------------------

async def test_note_schema_valid_and_rejects_malformed():
    note = Note(title="T", sections=[{"heading": "H", "content": "Body text here."}])
    assert note.sections[0].heading == "H"
    with pytest.raises(ValidationError):
        Note(title="", sections=[{"heading": "H", "content": "x"}])
    with pytest.raises(ValidationError):
        Note(title="T", sections=[])
    with pytest.raises(ValidationError):
        Note(title="T", sections=[{"heading": "H", "content": "x"}],
             bogus_key="no silent extras")


async def test_flashcard_schema_rejects_bad_difficulty():
    good = FlashcardSet(title="T", flashcards=[{"front": "Q?", "back": "A"}])
    assert good.flashcards[0].difficulty == "medium"
    with pytest.raises(ValidationError):
        FlashcardSet(title="T", flashcards=[{"front": "Q?", "back": "A",
                                             "difficulty": "extreme"}])
    with pytest.raises(ValidationError):
        FlashcardSet(title="T", flashcards=[{"front": "", "back": "A"}])


async def test_mcq_schema_enforces_four_distinct_options_and_index():
    mcq = MCQ(**_mcq_payload()["questions"][0])
    assert mcq.correct_answer == 0
    bad_options = dict(_mcq_payload()["questions"][0], options=["A", "B", "C"])
    with pytest.raises(ValidationError):
        MCQ(**bad_options)
    dupes = dict(_mcq_payload()["questions"][0],
                 options=["Same", "same ", "SAME", "same"])
    with pytest.raises(ValidationError):
        MCQ(**dupes)
    bad_idx = dict(_mcq_payload()["questions"][0], correct_answer=7)
    with pytest.raises(ValidationError):
        MCQ(**bad_idx)


async def test_explanation_schema_rejects_empty_and_extras():
    exp = Explanation(explanation="Because.", examples=["e1"],
                      related_concepts=["r1"])
    assert exp.examples == ["e1"]
    with pytest.raises(ValidationError):
        Explanation(explanation="", examples=[], related_concepts=[])
    with pytest.raises(ValidationError):
        Explanation(explanation="x", unknown_field="nope")


async def test_answer_schema_enforces_marks_band_coverage():
    ans = Answer(**_answer_payload(marks=5))
    assert ans.length_ok() is True
    assert ans.word_count() == 160
    narrow = _answer_payload(marks=5)
    narrow["target_min_words"] = 300  # narrower than the marks-5 band
    narrow["target_max_words"] = 320
    with pytest.raises(ValidationError):
        Answer(**narrow)
    with pytest.raises(ValidationError):
        Answer(**{**_answer_payload(marks=5), "marks": 0})


async def test_structured_generation_validates_answer_and_fails_typed():
    stub = StubLLM([json.dumps(_answer_payload(marks=2))])
    out = await generate_structured(stub.llm if hasattr(stub, "llm") else stub,
                                    Answer, "prompt")
    assert isinstance(out, Answer) and out.marks == 2
    bad = StubLLM(['{"definitely": "wrong shape"}'] * 3)
    with pytest.raises(StructuredGenerationError):
        await generate_structured(bad, MCQSet, "prompt", max_repairs=2)
    assert bad.calls == 3  # initial + max 2 repairs


# -- grounding / citations ---------------------------------------------------

async def test_check_citations_rejects_fabricated_ids():
    engine = TransformationEngine(llm_provider=StubLLM(["{}"]))
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        engine._check_citations({"citations": [{"chunk_id": FABRICATED}]}, {CID_A})


async def test_validate_grounding_ids_and_spans():
    chunk_texts = {CID_A: "Chlorophyll absorbs light in the leaf."}
    payload = {"citations": [{"chunk_id": CID_A, "span": "chlorophyll absorbs light"}]}
    gate = validate_grounding(payload, {CID_A}, chunk_texts)
    assert gate == {"cited": [CID_A], "unsupported": []}
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        validate_grounding({"citations": [{"chunk_id": FABRICATED}]}, {CID_A})
    gate2 = validate_grounding(
        {"citations": [{"chunk_id": CID_A, "span": "quantum tunneling pork"}]},
        {CID_A}, chunk_texts)
    assert len(gate2["unsupported"]) == 1  # span not in chunk: flagged, not dropped


async def test_collect_cited_ids_walks_nested_payload():
    payload = {"sections": [{"citations": [{"chunk_id": CID_A}]},
                            {"sub": {"citations": [{"chunk_id": CID_B}]}}]}
    assert collect_cited_ids(payload) == {CID_A, CID_B}


# -- duplicates / hallucination ----------------------------------------------

async def test_dedupe_flashcards_and_mcqs_keep_first():
    cards = [{"front": "What is X?", "back": "1"},
             {"front": "  what is X? ", "back": "2"},
             {"front": "What is Y?", "back": "3"}]
    assert [c["back"] for c in dedupe_flashcards(cards)] == ["1", "3"]
    qs = [{"question": "Which pigment?"},
          {"question": "WHICH pigment!"},
          {"question": "Other?"}]
    assert [q["question"] for q in dedupe_mcqs(qs)] == ["Which pigment?", "Other?"]
    assert normalize_text("  Hello,  WORLD! ") == "hello world"


async def test_hallucination_guard_cases():
    ok_payload = {"citations": [{"chunk_id": CID_A}]}
    assert check_hallucination(ok_payload, {CID_A}, True)["ok"] is True
    fab = check_hallucination({"citations": [{"chunk_id": FABRICATED}]},
                              {CID_A}, True)
    assert fab["fabricated"] == [FABRICATED] and fab["ok"] is False
    missing = check_hallucination({"text": "claims without cites"}, {CID_A}, True)
    assert missing["missing_citations"] is True
    unsup = check_hallucination({"text": "free text"}, set(), False)
    assert unsup["unsupported"] is True
    # unsupported content must never be rendered as cited
    bad = check_hallucination({"citations": [{"chunk_id": CID_A}]}, set(), False)
    assert bad["ok"] is False


# -- marks-aware --------------------------------------------------------------

async def test_marks_bands_grow_and_reject_nonpositive():
    bands = [constraints_for_marks(m) for m in (1, 2, 5, 10)]
    widths = [(b["min_words"], b["max_words"]) for b in bands]
    assert widths[0][0] < widths[-1][0] and widths[0][1] < widths[-1][1]
    assert constraints_for_marks(4)["min_words"] == 115  # interpolated
    with pytest.raises(ValueError):
        constraints_for_marks(0)


async def test_answer_prompt_carries_marks_band_style_and_grounding_rules():
    from services.transformation.prompts import PromptTemplates

    prompt = PromptTemplates.get_prompt(GenerationMode.ANSWER, "ctx", "Q?",
                                        marks=2, style="exam")
    assert "40-120 words" in prompt and "exam" in prompt.lower()
    assert "GROUNDING RULES" in prompt and "SECURITY RULES" in prompt
    mcq_prompt = PromptTemplates.get_prompt(GenerationMode.MCQ, "ctx", "Q?")
    assert "EXACTLY 4 distinct options" in mcq_prompt


async def test_generate_answer_pipeline_marks_grounding_and_length(monkeypatch):
    from services.retrieval.types import RetrievalResult, RetrievedChunk

    band = constraints_for_marks(2)
    n = band["min_words"] + 5
    stub = StubLLM([json.dumps(_answer_payload(marks=2, words=n))])
    engine = TransformationEngine(llm_provider=stub)

    chunk = RetrievedChunk(chunk_id=CID_A, content="Chlorophyll absorbs light here. " * 20,
                           similarity=0.9, document_id=str(uuid.uuid4()))
    fake_rr = RetrievalResult(query="q", total_results=1, chunks=[chunk],
                              processing_time=0.0)

    async def _fake_retrieve(**kwargs):
        return fake_rr

    async def _noop_cache(*a, **k):
        return None

    monkeypatch.setattr(enginemod.retrieval_service, "retrieve_with_db", _fake_retrieve)
    monkeypatch.setattr(engine, "_check_cache", _noop_cache)
    monkeypatch.setattr(engine, "_save_to_cache", _noop_cache)

    result = await engine.generate_answer(
        project_id=str(uuid.uuid4()), db=None, query="What absorbs light?",
        marks=2, style="exam")
    assert result.mode == GenerationMode.ANSWER
    assert result.content["grounding"]["support"] == "retrieved"
    assert result.content["_length_check"]["ok"] is True
    assert result.content["_grounding_gate"]["unsupported"] == []
    assert result.content["citations"][0]["chunk_id"] == CID_A


# -- LangGraph generation nodes ------------------------------------------------

async def test_answer_node_unsupported_without_retrieval():
    from services.graph import nodes as N

    out = N.answer_node({"query": "Q?", "marks": 5, "citations": []})
    assert out["support"] == "unsupported" and "error" in out


async def test_answer_node_rejects_fabricated_citations():
    from services.graph import nodes as N

    out = N.answer_node({"query": "Q?", "marks": 2,
                         "candidates": [{"id": CID_A, "text": "chlorophyll"}],
                         "citations": [{"chunk_id": FABRICATED}]})
    assert "fabricated" in out["error"]


async def test_study_nodes_grounded_and_explicit_on_empty():
    from services.graph import nodes as N

    empty = N.notes_node({"study_mode": "flashcards", "candidates": []})
    assert empty["generated"] is None and "error" in empty
    ok = N.mcq_node({"query": "Q?",
                     "candidates": [{"id": CID_A, "text": "chlorophyll absorbs light"}],
                     "citations": [{"chunk_id": CID_A}]})
    assert ok["generated"]["support"] == "retrieved"
    assert ok["generated"]["mode"] == "mcq"
    cards = N.flashcards_node({"candidates": [{"id": CID_A, "text": "t"}]})
    assert cards["generated"]["mode"] == "flashcards"


async def test_legacy_graph_paths_unchanged():
    from services.graph import nodes as N

    assert N.study_content_node({"content": "abc"}) == {"content": "abc"}
    gen = N.generate_node({"query": "hi", "transformed_query": "hi!"})
    assert "answer" in gen  # legacy LLM path (adapter handles offline)
