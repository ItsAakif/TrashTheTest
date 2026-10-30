"""STEPS 5-7 tests: typed flashcards -> MCQ matrix -> metrics (offline).

No DB, no LLM, no network. Provider doubles with call counters cover the
live _generate_structured paths. Live-Postgres artifact persistence
(save_academic_artifact) needs a real database and is NOT RUN here; the
artifact payload shapes it stores are asserted offline instead.
"""

import json

import pytest
from pydantic import ValidationError

from services.academic.extraction import extract_structure
from services.academic.flashcards import (
    bloom_targets,
    build_flashcards,
    concept_index,
    flashcards_artifact,
    repair_bloom_once,
    to_deck_cards,
    validate_deck,
)
from services.academic.mcq import (
    build_matrix,
    build_mcqs,
    mcq_artifact,
    project_quiz_questions,
    shingle_similarity,
    validate_mcq,
    validate_set,
)
from services.academic.metrics import (
    compute_metrics,
    metrics_artifact,
    metrics_numbers,
    normalize_inputs,
    recency_weight,
    repair_narrative,
    summary_cards,
    topic_table_rows,
    validate_narrative,
)
from services.academic.types import (
    BLOOM_MIX,
    FSRS_NEW_REVIEW,
    BloomLevel,
    CardType,
    DocumentStructure,
    ErrorTag,
    ExtractedTopic,
    FlashcardSpec,
    MCQOption,
    MCQSpec,
    MetricsReport,
    MetricsResponse,
    StructuredUnit,
    TopicCitation,
    not_generated_state,
)
from tests.eval.corpus import CORPUS

pytestmark = pytest.mark.asyncio


def _doc_chunks(doc_id):
    return [dict(r) for r in CORPUS if r["doc"] == doc_id]


def _chunk_texts(doc_id):
    return {r["id"]: r["text"] for r in _doc_chunks(doc_id)}


def _bio_structure():
    return extract_structure("doc-biology", "Photosynthesis primer",
                             _doc_chunks("doc-biology"))


def _linalg_structure():
    """doc-linalg yields 7 concepts under naming v2 (defined-term rule)."""
    return extract_structure("doc-linalg", "Linear algebra notes",
                             _doc_chunks("doc-linalg"))


def _flash_rows(doc_id):
    """Corpus rows with a second qualifying sentence per chunk.

    The flashcards eligibility rule requires a multi-chunk topic OR a
    chunk with >= 2 qualifying sentences.  The shared corpus is
    deliberately terse (one sentence per chunk), so flashcard tests use
    this realistic paragraph shape without touching the frozen corpus.
    """
    return [
        {**row, "text": row["text"] + " It is a core idea in this subject."}
        for row in _doc_chunks(doc_id)
    ]


def _linalg_flash_structure():
    return extract_structure("doc-linalg", "Linear algebra notes",
                             _flash_rows("doc-linalg"))


def _linalg_flash_texts():
    return {r["id"]: r["text"] for r in _flash_rows("doc-linalg")}


_RICH_TERMS = ["Polymorphism", "Inheritance", "Encapsulation", "Abstraction",
               "Recursion", "Iteration", "Compilation", "Interpretation",
               "Syntax", "Semantics", "Variable", "Constant"]


def _rich_rows():
    # Each chunk's text yields its concept under naming v2 (defined-term
    # rule) and carries two qualifying sentences so it satisfies the
    # flashcard evidence rule; a full-deck Bloom guard has enough material
    # (12 concepts deal to all six Bloom levels).  The section number keeps
    # each chunk's text distinct so CLOZE fronts do not collide.
    return [
        {"id": f"rich-{i:02d}",
         "text": f"{term} is the focus of section {i + 1} in this primer. "
                 f"It lets one interface serve many different types.",
         "heading": f"Topic {i + 1}", "page": 1, "position": i}
        for i, term in enumerate(_RICH_TERMS)
    ]


def _rich_structure():
    return extract_structure("doc-rich", "Rich primer", _rich_rows())


def _rich_texts():
    return {r["id"]: r["text"] for r in _rich_rows()}


def _hand_structure():
    def _topic(title, cid):
        return ExtractedTopic(title=title, pages=[1], chunks=[cid],
                              subtopics=[], concepts=[])

    return DocumentStructure(
        doc_id="d", title="D",
        units=[
            StructuredUnit(title="U1", topics=[_topic("Alpha", "c1")]),
            StructuredUnit(title="U2", topics=[_topic("Beta", "c2")]),
        ],
        fingerprint="f" * 64,
    )


# -- STEP 5: typed flashcards from the spine ----------------------------------


async def test_bloom_mix_sums_to_one():
    assert abs(sum(BLOOM_MIX.values()) - 1.0) < 1e-9
    assert set(BLOOM_MIX) == {level.value for level in BloomLevel}


async def test_flashcards_built_from_spine_concepts():
    struct = _linalg_flash_structure()
    index = concept_index(struct)
    assert len(index) == 7  # chunks with a usable title line or defined term
    texts = _linalg_flash_texts()
    deck = build_flashcards(struct, texts)
    assert len(deck.flashcards) == 7
    allowed = set(texts)
    types_seen = set()
    for card in deck.flashcards:
        assert card.concept_id in index  # spine-bound, never raw-PDF
        assert card.topic == index[card.concept_id]["topic"]
        assert {c.chunk_id for c in card.citations} <= allowed
        assert card.citations  # grounded, never empty-cited
        assert isinstance(card.card_type, CardType)
        assert isinstance(card.bloom_level, BloomLevel)
        assert card.metadata.get("provenance", "").startswith("spine concept")
        assert card.review == FSRS_NEW_REVIEW  # defaults, no transitions
        types_seen.add(card.card_type.value)
    assert len(types_seen) > 1  # type vocabulary actually rotates


async def test_flashcard_wire_aliases_and_extra_rejected():
    card = FlashcardSpec(
        id="card-0001", topic="T", concept_id="concept-0001",
        question="What is X?", answer="X is Y.",
        type="definition", difficulty="easy", cognitive="remember",
        cites=[TopicCitation(chunk_id="c1", span="x y")],
        metadata={"provenance": "spine"}, review=dict(FSRS_NEW_REVIEW),
    )
    assert card.card_type == CardType.DEFINITION
    assert card.bloom_level == BloomLevel.REMEMBER
    with pytest.raises(ValidationError):
        FlashcardSpec(
            id="c", topic="T", concept_id="k", question="Q?", answer="A",
            type="definition", cognitive="remember", cites=[],
            metadata={}, review={}, bogus=1,
        )


async def test_flashcard_model_rejects_trivial_and_long_fronts():
    with pytest.raises(ValidationError, match="differ from front"):
        FlashcardSpec(
            id="c", topic="T", concept_id="k",
            question="Same text here", answer="same text here!",
            type="definition", cognitive="remember", cites=[],
            metadata={}, review={},
        )
    with pytest.raises(ValidationError, match="30"):
        FlashcardSpec(
            id="c", topic="T", concept_id="k",
            question=" ".join(["word"] * 31), answer="short answer",
            type="definition", cognitive="remember", cites=[],
            metadata={}, review={},
        )


async def test_bloom_guard_passes_full_deck():
    struct = _rich_structure()
    deck = build_flashcards(struct, _rich_texts())
    report = validate_deck(deck.flashcards, set(_rich_texts()),
                           set(concept_index(struct)))
    assert report.ok is True and report.bloom_ok is True
    for level in BloomLevel:
        assert abs(report.bloom_realized[level.value]
                   - BLOOM_MIX[level.value]) <= 0.10


async def test_short_deck_reported_not_padded():
    struct = _linalg_flash_structure()
    texts = dict(list(_linalg_flash_texts().items())[:3])
    deck = build_flashcards(struct, texts)
    assert len(deck.flashcards) == 2  # short, never padded
    assert "skipped" in deck.provenance_note
    report = validate_deck(deck.flashcards, set(texts),
                           set(concept_index(struct)))
    assert report.bloom_ok is False  # honest shortfall ...
    assert "analyze" in report.shortfall  # ... naming the level


async def test_bloom_repair_once_then_honest():
    struct = _bio_structure()
    index = concept_index(struct)
    ids = sorted(index)
    chunks = _chunk_texts("doc-biology")
    cards = []
    for i in range(20):
        cid = ids[i % len(ids)]
        chunk = index[cid]["chunk_ids"][0]
        cards.append(FlashcardSpec(
            id=f"card-{i:04d}", topic=index[cid]["topic"], concept_id=cid,
            question=f"Question number {i} about {index[cid]['concept']}?",
            answer=f"Answer number {i}: {chunks[chunk]}",
            type="definition", cognitive="remember",
            cites=[TopicCitation(chunk_id=chunk, span="span words here")],
            metadata={"provenance": "spine"}, review=dict(FSRS_NEW_REVIEW),
        ))
    before = validate_deck(cards, set(chunks), set(ids))
    assert before.bloom_ok is False
    fixed, report, repaired = repair_bloom_once(cards)
    assert repaired is True
    assert report.bloom_ok is True and report.ok is True
    assert len(fixed) == 20


async def test_validate_deck_rejects_each_defect():
    struct = _rich_structure()
    index = concept_index(struct)
    chunks = _rich_texts()
    deck = build_flashcards(struct, chunks)
    good = deck.flashcards[0]
    dupe = good.model_copy(update={"id": "card-9999"})
    unknown = good.model_copy(update={"id": "card-9998",
                                      "concept_id": "concept-9999"})
    bad_cite = good.model_copy(update={
        "id": "card-9997",
        "citations": [TopicCitation(chunk_id="no-such-chunk", span="x")]})
    no_prov = good.model_copy(update={"id": "card-9996", "metadata": {}})
    no_fsrs = good.model_copy(update={"id": "card-9995", "review": {}})
    report = validate_deck([good, dupe, unknown, bad_cite, no_prov, no_fsrs],
                           set(chunks), set(index))
    codes = {e["code"] for e in report.errors}
    assert {"duplicate_front", "unknown_concept", "unsupported_cite",
            "missing_provenance", "missing_fsrs_field"} <= codes
    assert report.ok is False


async def test_deck_projection_keeps_flashcards_ui_contract():
    struct = _rich_structure()
    deck = build_flashcards(struct, _rich_texts())
    projected = to_deck_cards(deck.flashcards)
    assert len(projected) == len(deck.flashcards)
    for row in projected:
        assert set(row) == {"front", "back", "difficulty"}
        assert row["difficulty"] in ("easy", "medium", "hard")
        assert row["front"] and row["back"] and row["front"] != row["back"]


async def test_flashcards_artifact_shape():
    struct = _rich_structure()
    deck = build_flashcards(struct, _rich_texts())
    report = validate_deck(deck.flashcards, set(_rich_texts()),
                           set(concept_index(struct)))
    artifact = flashcards_artifact(deck, report)
    assert artifact["bloom_ok"] is True
    assert len(artifact["flashcards"]) == 12
    assert artifact["flashcards"][0]["type"] in {
        t.value for t in CardType}  # wire aliases preserved


async def test_live_typed_path_uses_shared_structured_generation():
    import json as _json

    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    calls = {"n": 0}

    class _StubProvider:
        async def generate(self, prompt, config=None):
            calls["n"] += 1
            from services.strategies.base import GenerationResult as _GR

            def _result(text):
                return _GR(text=text, model="stub", usage={},
                           finish_reason="stop", processing_time=0.0)

            if calls["n"] == 1:
                return _result('{"bogus": true}')  # schema fail -> 1 repair
            return _result(_json.dumps({
                "title": "Deck",
                "flashcards": [{
                    "id": "card-0001", "topic": "T",
                    "concept_id": "concept-0001",
                    "question": "What is X?", "answer": "X is Y.",
                    "card_type": "definition", "difficulty": "medium",
                    "bloom_level": "remember",
                    "citations": [{"chunk_id": "c1", "span": "x y"}],
                }],
                "citations": [],
            }))

    engine = TransformationEngine.__new__(TransformationEngine)
    engine.llm = _StubProvider()
    payload = await engine._generate_structured(
        GenerationMode.FLASHCARDS_TYPED, "prompt", {"c1"})
    assert payload["flashcards"][0]["question"] == "What is X?"
    assert calls["n"] == 2


async def test_live_typed_path_rejects_foreign_chunk_id():
    import json as _json

    from services.strategies.llm.structured import StructuredGenerationError
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    class _BadProvider:
        async def generate(self, prompt, config=None):
            from services.strategies.base import GenerationResult as _GR

            return _GR(text=_json.dumps({
                "title": "Deck",
                "flashcards": [{
                    "id": "card-0001", "topic": "T",
                    "concept_id": "concept-0001",
                    "question": "What is X?", "answer": "X is Y.",
                    "card_type": "definition", "difficulty": "medium",
                    "bloom_level": "remember",
                    "citations": [{"chunk_id": "11111111-2222-3333-4444-555555555555",
                                   "span": "x"}],
                }],
                "citations": [],
            }), model="stub", usage={}, finish_reason="stop",
                processing_time=0.0)

    engine = TransformationEngine.__new__(TransformationEngine)
    engine.llm = _BadProvider()
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        await engine._generate_structured(
            GenerationMode.FLASHCARDS_TYPED, "prompt", {"c1"})


# -- STEP 6: MCQ matrix ---------------------------------------------------------


async def test_matrix_weighting_three_modes_sum_exactly():
    struct = _hand_structure()
    uniform = build_matrix(struct, total=10)
    assert sum(c.target for c in uniform.cells) == 10
    alpha = sum(c.target for c in uniform.cells
                if c.topic_title == "Alpha")
    assert alpha == 5  # uniform split over two topics
    weighted = build_matrix(struct, total=10,
                            syllabus_weights={"U1": 75.0, "U2": 25.0})
    assert sum(c.target for c in weighted.cells) == 10
    alpha_w = sum(c.target for c in weighted.cells
                  if c.topic_title == "Alpha")
    beta_w = sum(c.target for c in weighted.cells
                 if c.topic_title == "Beta")
    assert (alpha_w, beta_w) == (8, 2)  # follows weights within rounding
    freq = build_matrix(struct, total=10,
                        pyq_frequency={"Alpha": 3, "Beta": 1})
    assert sum(c.target for c in freq.cells) == 10
    alpha_f = sum(c.target for c in freq.cells
                  if c.topic_title == "Alpha")
    assert alpha_f >= 7  # frequency signal shifts targets


async def test_unfilled_cells_reported_never_padded():
    struct = _bio_structure()
    plan = build_matrix(struct, total=12)
    mcq_set = build_mcqs(struct, {}, plan)  # no source material
    assert mcq_set.questions == []
    assert len(mcq_set.plan.unfilled) == 12
    assert all("unfilled" in u.reason and "never" not in u.reason.lower()
               or "invented" in u.reason for u in mcq_set.plan.unfilled)
    verdict = validate_set(mcq_set, set(), {t.title for u in struct.units
                                            for t in u.topics})
    assert verdict["plan_ok"] is True  # 0 filled == 0 questions


async def test_build_mcqs_rich_validated_and_plan_sums():
    # doc-linalg yields 7 concepts with distinct source facts, so every
    # cell finds 3 real distractors (doc-biology yields only 1 concept and
    # would leave every cell unfilled under the real-fact contract).
    struct = _linalg_structure()
    texts = _chunk_texts("doc-linalg")
    plan = build_matrix(struct, total=12)
    mcq_set = build_mcqs(struct, texts, plan)
    assert len(mcq_set.questions) == 12
    assert mcq_set.plan.unfilled == []
    topics = {t.title for u in struct.units for t in u.topics}
    verdict = validate_set(mcq_set, set(texts), topics)
    assert verdict == {"ok": True, "errors": [], "plan_ok": True}
    for question in mcq_set.questions:
        assert len(question.options) == 4
        assert sum(1 for o in question.options if o.is_correct) == 1
        distractors = [o for o in question.options if not o.is_correct]
        assert all(isinstance(o.error_tag, ErrorTag) for o in distractors)
        assert all(o.rationale.strip() for o in distractors)
        assert question.citations  # grounded in the prompt


async def test_mcq_model_rejects_structural_defects():
    def _option(text, correct=False, tag="misconception"):
        return {"text": text, "is_correct": correct,
                "error_tag": None if correct else tag,
                "rationale": "" if correct else "Why it is wrong, in one sentence."}

    base = {"id": "mcq-1", "question": "Which statement is correct?",
            "explanation": "Because the source says so.",
            "topic": "T", "topic_id": "topic-0001",
            "level": "remember", "cites": []}
    with pytest.raises(ValidationError):
        MCQSpec(**{**base, "options": [_option("A", True), _option("B")],
                   "correct_answer": 0})  # 2 options
    with pytest.raises(ValidationError):
        MCQSpec(**{**base, "options": [_option(f"O{i}") for i in range(5)],
                   "correct_answer": 0})  # 5 options
    with pytest.raises(ValidationError, match="duplicate"):
        MCQSpec(**{**base,
                   "options": [_option("Same", True), _option("same"),
                               _option("C"), _option("D")],
                   "correct_answer": 0})
    with pytest.raises(ValidationError, match="exactly 1"):
        MCQSpec(**{**base,
                   "options": [_option("A", True), _option("B", True),
                               _option("C"), _option("D")],
                   "correct_answer": 0})
    with pytest.raises(ValidationError, match="error_tag"):
        MCQSpec(**{**base,
                   "options": [_option("A", True),
                               {"text": "B", "is_correct": False,
                                "error_tag": None, "rationale": "R."},
                               _option("C"), _option("D")],
                   "correct_answer": 0})
    with pytest.raises(ValidationError):
        MCQSpec(**{**base, "options": [_option("A") for _ in range(4)],
                   "correct_answer": 0, "bogus": 1})  # extra=forbid


async def test_validate_mcq_catches_similar_leaky_and_partial_truth():
    key = ("Photosynthesis converts light energy into chemical energy "
           "stored in glucose daily")
    near = ("Photosynthesis converts light energy into chemical energy "
            "stored in glucose nightly")  # one final word differs
    assert shingle_similarity(key, near) >= 0.8
    assert shingle_similarity(key, "Unrelated moon geology facts.") < 0.8

    def _option(text, correct=False, tag="misconception"):
        return MCQOption(
            text=text, is_correct=correct,
            error_tag=None if correct else ErrorTag(tag),
            rationale="" if correct else "Why it is wrong, in one sentence.")

    similar = MCQSpec(
        id="m1", question="Which statement about the process is correct?",
        options=[_option(key, True), _option(near),
                 _option("Far away option one here."),
                 _option("Far away option two here.")],
        correct_answer=0, explanation="The source states the key.",
        topic="T", topic_id="topic-0001", level="remember", cites=[])
    errors = validate_mcq(similar, set(), {"T"})
    assert any(e["code"] == "similar_distractor" for e in errors)

    leaky = MCQSpec(
        id="m2", question=f"Is it true that {key}",
        options=[_option(key, True), _option("Far away option one here."),
                 _option("Far away option two here."),
                 _option("Far away option three here.")],
        correct_answer=0, explanation="The source states the key.",
        topic="T", topic_id="topic-0001", level="remember", cites=[])
    assert any(e["code"] == "leakage"
               for e in validate_mcq(leaky, set(), {"T"}))

    partial = MCQSpec(
        id="m3", question="Which statement about the topic is correct?",
        options=[_option(key, True),
                 _option("Partly right but missing the key detail.",
                         tag="partial_truth"),
                 _option("Far away option one here."),
                 _option("Far away option two here.")],
        correct_answer=0, explanation="The source states the key.",
        topic="T", topic_id="topic-0001", level="remember", cites=[])
    assert validate_mcq(partial, set(), {"T"}) == []  # accepted, tagged


async def test_key_rotation_balances_coverage_difficulty_bloom():
    struct = _linalg_structure()
    texts = _chunk_texts("doc-linalg")
    plan = build_matrix(struct, total=24)
    mcq_set = build_mcqs(struct, texts, plan)
    assert len(mcq_set.questions) == 24
    slots = {q.correct_answer for q in mcq_set.questions}
    assert len(slots) > 1  # key position rotates, not stuck at 0
    levels = {q.bloom_level for q in mcq_set.questions}
    assert len(levels) > 1  # Bloom matrix actually spans levels
    topics = {t.title for u in struct.units for t in u.topics}
    assert validate_set(mcq_set, set(texts), topics)["ok"] is True


async def test_rv2_projection_is_study_compatible():
    struct = _linalg_structure()
    texts = _chunk_texts("doc-linalg")
    mcq_set = build_mcqs(struct, texts, build_matrix(struct, total=12))
    projected = project_quiz_questions(mcq_set)
    assert len(projected) == 12
    for row in projected:
        assert set(row) >= {"question", "options", "correct_answer",
                            "explanation"}
        assert all(isinstance(o, str) for o in row["options"])
        assert isinstance(row["correct_answer"], int)
        # The study service renders options[correct_answer] as text:
        assert isinstance(row["options"][row["correct_answer"]], str)
        assert "error_tag" not in json.dumps(row)
    artifact = mcq_artifact(mcq_set)
    assert "error_tag" in json.dumps(artifact["questions"])  # rich stays put
    assert "plan" in artifact and "unfilled" in artifact
    assert artifact["projection"] == projected  # one source of truth


async def test_live_mcq_path_uses_shared_structured_generation():
    import json as _json

    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    class _StubProvider:
        async def generate(self, prompt, config=None):
            from services.strategies.base import GenerationResult as _GR

            return _GR(text=_json.dumps({
                "title": "Set",
                "questions": [{
                    "question": "Which statement is correct?",
                    "options": [
                        {"text": "Key", "is_correct": True,
                         "error_tag": None, "rationale": ""},
                        {"text": "D1", "is_correct": False,
                         "error_tag": "misconception", "rationale": "Wrong."},
                        {"text": "D2", "is_correct": False,
                         "error_tag": "unit_confusion", "rationale": "Wrong."},
                        {"text": "D3", "is_correct": False,
                         "error_tag": "overgeneralization",
                         "rationale": "Wrong."},
                    ],
                    "correct_answer": 0, "explanation": "Grounded.",
                    "topic": "T", "difficulty": "medium",
                    "bloom_level": "remember", "citations": [],
                }],
                "citations": [],
            }), model="stub", usage={}, finish_reason="stop",
                processing_time=0.0)

    engine = TransformationEngine.__new__(TransformationEngine)
    engine.llm = _StubProvider()
    payload = await engine._generate_structured(
        GenerationMode.MCQ_MATRIX, "prompt", set())
    assert payload["questions"][0]["correct_answer"] == 0


# -- STEP 7: deterministic intelligence ------------------------------------------


def _bio_inputs(**overrides):
    struct = _bio_structure()
    syllabus = [{"unit_id": "unit-1", "title": "Life processes",
                 "topics": ["Photosynthesis overview", "Calvin cycle"]}]
    pyqs = [
        {"question": "Photosynthesis overview converts light energy",
         "year": 2024, "marks": 5},
        {"question": "Chlorophyll pigments absorb red",
         "year": 2024, "marks": 5},
    ]
    questions = [{"question": "What does the Calvin cycle fix?"}]
    payload = {
        "structure": struct,
        "chunk_rows": _doc_chunks("doc-biology"),
        "pyqs": pyqs,
        "quiz_questions": questions,
        "syllabus_units": syllabus,
    }
    payload.update(overrides)
    return payload


async def test_recency_hand_computed():
    assert recency_weight(2026, 2026, 5.0) == 1.0
    assert recency_weight(2021, 2026, 5.0) == 0.5
    assert recency_weight(2030, 2026, 5.0) == 1.0  # future clamps


async def test_metrics_deterministic_and_shaped():
    first = compute_metrics(normalize_inputs(**_bio_inputs()))
    second = compute_metrics(normalize_inputs(**_bio_inputs()))
    assert (json.dumps(first.model_dump(mode="json"), sort_keys=True)
            == json.dumps(second.model_dump(mode="json"), sort_keys=True))
    assert isinstance(first, MetricsReport)
    assert first.half_life == 5.0 and first.current_year == 2026
    assert len(first.topics) == 12
    assert first.totals["pyq_questions"] == 2
    assert first.totals["total_marks"] == 10


async def test_metrics_values_and_provenance():
    report = compute_metrics(normalize_inputs(**_bio_inputs()))
    photo = next(r for r in report.topics
                 if r.topic_title == "Photosynthesis overview")
    assert photo.frequency == 1 and photo.recurrence == 1
    assert photo.marks_total == 5 and photo.marks_share == 50.0
    assert photo.recency_score == recency_weight(2024, 2026, 5.0)
    assert photo.in_syllabus and photo.in_documents and photo.in_pyq
    assert photo.evidence["chunks"] == ["bio-photosyn-01"]
    assert photo.evidence["pyqs"] == ["pyq-0"]
    calvin = next(r for r in report.topics
                  if r.topic_title == "Calvin cycle")
    assert calvin.frequency == 0 and calvin.in_questions is True
    assert "Calvin cycle" in report.never_asked  # syllabus gap, honest
    assert len(report.missing) == 10  # 10 topics with no PYQ
    assert report.totals["pyq_coverage"] == round(2 / 12, 4)
    assert report.hotspots[0] == "Photosynthesis overview"  # tie -> spine ord
    assert set(report.hotspots) == {"Photosynthesis overview",
                                    "Chlorophyll pigments"}
    units = {u.unit_title: u for u in report.units}
    assert sum(u.frequency for u in report.units) == 2
    assert all(u.topic_count > 0 for u in report.units)


async def test_hotspot_tiebreak_follows_spine_ord():
    report = compute_metrics(normalize_inputs(**_bio_inputs(pyqs=[])))
    assert [r.hotspot_rank for r in report.topics] == list(range(1, 13))
    assert report.hotspots == [] and report.never_asked != []


async def test_narrative_grounding_repair_then_withhold():
    report = compute_metrics(normalize_inputs(**_bio_inputs()))
    allowed = metrics_numbers(report)
    coverage = report.totals["pyq_coverage"]
    assert str(coverage) in allowed
    good = (f"PYQ coverage is {coverage} with 2 questions "
            f"worth 10 marks in 2026.")
    assert validate_narrative(good, report)["ok"] is True
    bad = "Coverage improved by 99.9% with 1000 questions."
    verdict = validate_narrative(bad, report)
    assert verdict["ok"] is False and "99.9%" in verdict["fabricated"]
    # Stubborn generator: repaired once, then withheld (metrics intact).
    calls = {"n": 0}

    def _stubborn(prompt):
        calls["n"] += 1
        return "Coverage soared by 99.9% this year."

    response = repair_narrative(report, _stubborn)
    assert calls["n"] == 2  # initial + exactly one repair
    assert response.narrative is None
    assert response.narrative_status == "withheld"
    assert response.metrics == report  # HTTP-200 shape: metrics survive

    def _honest(prompt):
        return good

    repaired = repair_narrative(report, _honest)
    assert repaired.narrative_status == "ok" and repaired.narrative == good


async def test_not_generated_names_the_call():
    state = not_generated_state("flashcards",
                                "POST /api/generation/flashcards/typed")
    assert state["state"] == "not_generated"
    assert "POST /api/generation/flashcards/typed" in state["message"]
    metrics_state = not_generated_state("metrics", "POST /metrics")
    assert metrics_state["generation_call"] == "POST /metrics"


async def test_tables_and_cards_backend_first():
    report = compute_metrics(normalize_inputs(**_bio_inputs()))
    rows = topic_table_rows(report)
    assert len(rows) == 12
    row = next(r for r in rows if r["topic"] == "Photosynthesis overview")
    assert row["frequency"] == 1 and row["marks_share"] == 50.0
    assert row["coverage"] == {"syllabus": True, "documents": True,
                               "pyq": True, "questions": False}
    assert row["evidence"]["chunks"] == ["bio-photosyn-01"]
    cards = summary_cards(report)
    by_card = {c["card"]: c for c in cards}
    assert set(by_card) == {"coverage", "hotspots", "never_asked",
                            "missing", "marks"}
    assert by_card["marks"]["total_marks"] == 10
    artifact = metrics_artifact(
        report, MetricsResponse(metrics=report, narrative="Good.",
                                narrative_status="ok"))
    assert artifact["narrative_status"] == "ok"
    assert len(artifact["tables"]) == 12 and len(artifact["cards"]) == 5


async def test_normalize_marks_unmapped_honest():
    inputs = _bio_inputs(pyqs=[
        {"question": "Totally unrelated tenancy law text here", "year": 2020,
         "marks": 2},
    ])
    report = compute_metrics(normalize_inputs(**inputs))
    assert all(r.frequency == 0 for r in report.topics)  # never forced
    assert report.totals["pyq_coverage"] == 0.0


# -- RV-1: new modes + schemas in all three places --------------------------------


async def test_rv1_step57_modes_registered_in_enum_prompts_and_engine():
    from services.transformation.engine import TransformationEngine
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import (
        FlashcardsTypedOutput,
        GenerationMode,
        MCQMatrixOutput,
        MetricsNarrativeOutput,
    )

    assert TransformationEngine.MODE_MODELS[GenerationMode.FLASHCARDS_TYPED] \
        is FlashcardsTypedOutput
    assert TransformationEngine.MODE_MODELS[GenerationMode.MCQ_MATRIX] \
        is MCQMatrixOutput
    assert TransformationEngine.MODE_MODELS[GenerationMode.METRICS_NARRATIVE] \
        is MetricsNarrativeOutput
    assert TransformationEngine.STRICT_MODELS[GenerationMode.FLASHCARDS_TYPED] \
        is FlashcardsTypedOutput
    assert TransformationEngine.STRICT_MODELS[GenerationMode.MCQ_MATRIX] \
        is MCQMatrixOutput
    assert TransformationEngine.STRICT_MODELS[GenerationMode.METRICS_NARRATIVE] \
        is MetricsNarrativeOutput
    cases = ((GenerationMode.FLASHCARDS_TYPED, {"bloom_mix": BLOOM_MIX}),
             (GenerationMode.MCQ_MATRIX, {"topic": "T", "bloom": "apply"}),
             (GenerationMode.METRICS_NARRATIVE, {}))
    for mode, kwargs in cases:
        prompt = PromptTemplates.get_prompt(
            mode, "[source chunk_id=x] body", "q", **kwargs)
        assert PromptTemplates.INJECTION_GUARD in prompt
        assert PromptTemplates.CITATION_RULES in prompt


async def test_wire_shapes_mirror_domain_models():
    from services.transformation.types import (
        FlashcardsTypedOutput,
        MCQMatrixOutput,
    )

    assert {"id", "topic", "concept_id", "question", "answer", "card_type",
            "difficulty", "bloom_level", "citations"} == set(
                FlashcardsTypedOutput.model_fields["flashcards"].annotation
                .__args__[0].model_fields)
    assert {"question", "options", "correct_answer", "explanation", "topic",
            "difficulty", "bloom_level", "citations"} == set(
                MCQMatrixOutput.model_fields["questions"].annotation
                .__args__[0].model_fields)


async def test_artifact_keys_reuse_node_metadata():
    from database.knowledge_repo import (
        FLASHCARDS_ARTIFACT_KEY,
        MCQ_ARTIFACT_KEY,
        METRICS_ARTIFACT_KEY,
    )

    assert FLASHCARDS_ARTIFACT_KEY == "flashcards_artifact"
    assert MCQ_ARTIFACT_KEY == "mcq_artifact"
    assert METRICS_ARTIFACT_KEY == "metrics_artifact"
