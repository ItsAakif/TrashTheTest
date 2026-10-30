"""GEN grounding fixes: honest empty output, citation fabrication gate,
typed-deck honesty, relaxed lexical recovery, and tutor context.

Offline, deterministic: no network, no Postgres, no providers.
"""

import json
import struct
import uuid
from types import SimpleNamespace as NS

import pytest

from services.academic import orchestrator
from services.academic.types import (
    FSRS_NEW_REVIEW,
    BloomLevel,
    CardDifficulty,
    CardType,
    FlashcardSetSpec,
    FlashcardSpec,
    TopicCitation,
)
from database import repositories
from services.strategies.base import GenerationResult
from services.strategies.llm.structured import StructuredGenerationError
from services.transformation.engine import TransformationEngine
from services.transformation.types import GenerationMode

pytestmark = pytest.mark.asyncio

CID_A = "11111111-2222-4333-8444-555555555555"
CID_B = "99999999-8888-4777-8666-555555555555"
# The short, non-UUID fragment the live notes job actually emitted.
SHORT_FRAG = "4dcf8b79"


# ---------------------------------------------------------------------------
# Citation gate: a cited id must be one of the retrieved ids, whatever shape
# it takes (full UUID, 8-hex fragment, or a UUID smuggled into prose).
# ---------------------------------------------------------------------------

async def test_check_citations_rejects_short_non_uuid_fragment():
    engine = TransformationEngine(llm_provider=NS(generate=None))
    allowed = {CID_A}
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        engine._check_citations({"citations": [{"chunk_id": SHORT_FRAG}]}, allowed)
    # Allowed ids given as UUID objects still match string citations.
    engine._check_citations(
        {"citations": [{"chunk_id": CID_A}]},
        {uuid.UUID(CID_A)},
    )
    # A UUID that appears only inside prose is still rejected.
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        engine._check_citations({"sections": [{"content": f"see {CID_B}"}]}, allowed)


class _CountingStub:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0
        self.prompts = []

    async def generate(self, prompt, config=None):
        self.prompts.append(prompt)
        self.calls += 1
        idx = min(self.calls - 1, len(self.texts) - 1)
        return GenerationResult(
            text=self.texts[idx], model="stub", usage={},
            finish_reason="stop", processing_time=0.0,
        )


def _notes_payload(chunk_id: str) -> str:
    return json.dumps({
        "title": "Bio Notes",
        "sections": [{
            "heading": "Photosynthesis",
            "content": "Light dependent reactions happen in the thylakoid.",
            "citations": [{"chunk_id": chunk_id, "span": "thylakoid"}],
        }],
    })


async def test_fabricated_cite_gets_bounded_repair_then_typed_error():
    stub = _CountingStub([_notes_payload(SHORT_FRAG)] * 3)
    engine = TransformationEngine(llm_provider=stub)
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        await engine._generate_structured(
            GenerationMode.NOTES, "prompt", {CID_A})
    # Initial attempt + exactly two corrective attempts (max_repairs=2).
    assert stub.calls == 3
    # The corrective prompt names the grounding failure for the model.
    assert any("grounding check" in p for p in stub.prompts[1:])
    assert SHORT_FRAG in stub.prompts[1]


async def test_fabricated_cite_is_repaired_within_the_bound():
    stub = _CountingStub([_notes_payload(SHORT_FRAG), _notes_payload(CID_A)])
    engine = TransformationEngine(llm_provider=stub)
    payload = await engine._generate_structured(
        GenerationMode.NOTES, "prompt", {CID_A})
    assert stub.calls == 2
    assert payload["sections"][0]["citations"][0]["chunk_id"] == CID_A


# ---------------------------------------------------------------------------
# Domain persist: an empty result is never written as a domain row.
# ---------------------------------------------------------------------------

async def test_persist_domain_rows_never_writes_empty_rows(monkeypatch):
    calls = {"notes": [], "decks": [], "cards": []}

    async def fake_note(db, project_id, title, content=None):
        calls["notes"].append(title)
        return NS(id=uuid.uuid4())

    async def fake_deck(db, project_id, name):
        calls["decks"].append(name)
        return NS(id=uuid.uuid4())

    async def fake_card(db, deck_id, front, back, difficulty="medium"):
        calls["cards"].append((front, back, difficulty))

    monkeypatch.setattr(repositories, "create_note", fake_note)
    monkeypatch.setattr(repositories, "create_flashcard_deck", fake_deck)
    monkeypatch.setattr(repositories, "create_flashcard", fake_card)

    engine = TransformationEngine.__new__(TransformationEngine)
    pid = uuid.uuid4()

    # Headings-only notes read back as an empty artifact: not persisted.
    await engine._persist_domain_rows(
        None, pid, GenerationMode.NOTES,
        {"title": "T", "sections": [{"heading": "Only a heading",
                                     "content": "   "}]},
    )
    # A zero-card deck would read back "ready": never created.
    await engine._persist_domain_rows(
        None, pid, GenerationMode.FLASHCARDS, {"title": "D", "flashcards": []})
    # Same for a card with no back.
    await engine._persist_domain_rows(
        None, pid, GenerationMode.FLASHCARDS,
        {"title": "D", "flashcards": [{"front": "Q?", "back": "  "}]})
    assert calls == {"notes": [], "decks": [], "cards": []}

    # Real content still persists exactly as before.
    await engine._persist_domain_rows(
        None, pid, GenerationMode.NOTES,
        {"title": "T", "sections": [{"heading": "H", "content": "Body"}]})
    await engine._persist_domain_rows(
        None, pid, GenerationMode.FLASHCARDS,
        {"title": "D", "flashcards": [{"front": "Q?", "back": "A.",
                                        "difficulty": "easy"}]})
    assert calls["notes"] == ["T"]
    assert calls["decks"] == ["D"]
    assert calls["cards"] == [("Q?", "A.", "easy")]


# ---------------------------------------------------------------------------
# Typed flashcards: never a "ready" deck with zero cards.
# ---------------------------------------------------------------------------

def _spine(chunk_ids=("c1",), root_ids=None):
    return NS(
        chunks=[{"id": cid, "text": f"Passage text for {cid}."}
                for cid in chunk_ids],
        structure=NS(title="Doc"),
        root_ids=[uuid.uuid4()] if root_ids is None else list(root_ids),
        model_version="stub-model",
        source_fingerprint="fp-1",
    )


def _card(n=1, chunk_id="c1", concept_id="c1"):
    return FlashcardSpec(
        id=f"card-{n:04d}",
        topic="Photosynthesis",
        concept_id=concept_id,
        question=f"Explain concept number {n} for the exam",
        answer=f"Concept number {n} has a distinct answer body {n}",
        type=CardType.CONCEPTUAL,
        difficulty=CardDifficulty.MEDIUM,
        cognitive=BloomLevel.UNDERSTAND,
        cites=[TopicCitation(chunk_id=chunk_id, span="Passage text")],
        metadata={"provenance": "spine concept (DocumentStructure)"},
        review=dict(FSRS_NEW_REVIEW),
    )


def _spec(cards, title="Typed deck"):
    return FlashcardSetSpec(
        title=title, flashcards=cards, bloom_realized={},
        provenance_note="stub provenance",
    )


async def _run_typed(monkeypatch, build_result, *, spine=None,
                      concepts=("c1",), decks=None, to_deck=None):
    """Drive generate_typed_flashcards with every external edge faked."""
    saved = []
    deck_lookups = []
    created_cards = []
    repairs = []
    real_repair = orchestrator.academic_flashcards.repair_bloom_once

    async def _spine_stub(*args, **kwargs):
        return spine or _spine()

    async def _save(db, root_id, key, payload):
        saved.append((root_id, key))

    async def _decks(db, pid):
        deck_lookups.append(pid)
        return list(decks or [])

    async def _create_deck(db, pid, name):
        return NS(id=uuid.uuid4(), name=name)

    async def _create_card(db, deck_id, front, back, difficulty="medium"):
        created_cards.append((front, back, difficulty))

    def _repair(cards, mix=None):
        repairs.append(len(cards))
        return real_repair(cards, mix)

    monkeypatch.setattr(orchestrator, "build_project_spine", _spine_stub)
    monkeypatch.setattr(orchestrator, "build_flashcards", build_result)
    monkeypatch.setattr(
        orchestrator.academic_flashcards, "concept_index",
        lambda structure: {cid: {"ord": 0} for cid in concepts})
    monkeypatch.setattr(
        orchestrator.academic_flashcards, "repair_bloom_once", _repair)
    monkeypatch.setattr(
        orchestrator.knowledge_repo, "save_academic_artifact", _save)
    monkeypatch.setattr(
        orchestrator.repositories, "get_flashcard_decks_by_project", _decks)
    monkeypatch.setattr(
        orchestrator.repositories, "create_flashcard_deck", _create_deck)
    monkeypatch.setattr(
        orchestrator.repositories, "create_flashcard", _create_card)
    if to_deck is not None:
        monkeypatch.setattr(orchestrator, "to_deck_cards", to_deck)

    result = await orchestrator.generate_typed_flashcards(object(), uuid.uuid4())
    return NS(result=result, saved=saved, lookups=deck_lookups,
              created=created_cards, repairs=repairs)


async def test_typed_flashcards_invalid_build_is_not_generated(monkeypatch):
    def _invalid(*args, **kwargs):
        # A real pydantic ValidationError, exactly what an empty build raises
        # (FlashcardSetSpec.flashcards has min_length=1).
        raise FlashcardSetSpec.model_validate({"title": "no cards"})

    out = await _run_typed(monkeypatch, _invalid, decks=[])
    # The failure happens before any deck/artifact work: no lookup, no write.
    assert out.result["state"] == "not_generated"
    assert out.result["artifact_id"] is None
    assert out.result["deck_id"] is None
    assert out.result["message"]
    assert out.saved == [] and out.lookups == [] and out.created == []


async def test_typed_flashcards_empty_projection_is_not_generated(monkeypatch):
    def _project(cards):
        # Every projected row is unusable (blank front): zero cards remain.
        return [{"front": " ", "back": "A body", "difficulty": "medium"}]

    out = await _run_typed(
        monkeypatch,
        lambda *a, **k: _spec([_card(1), _card(2)]),
        to_deck=_project,
    )
    assert out.result["state"] == "not_generated"
    assert "zero cards" in out.result["message"]
    assert out.saved == [] and out.lookups == [] and out.created == []


async def test_typed_flashcards_unsupported_cite_blocks_persist_before_repair(
        monkeypatch):
    out = await _run_typed(
        monkeypatch,
        # Card cites a chunk that is not in the retrieved set.
        lambda *a, **k: _spec([_card(1, chunk_id="ghost-chunk")]),
        concepts=("c1",),
    )
    # The gate answered BEFORE any Bloom repair ran (a repair re-validates
    # against the cards' own cites and would erase this verdict).
    assert out.repairs == []
    assert out.result["state"] == "not_generated"
    assert "nothing was persisted" in out.result["message"]
    codes = [e["code"] for e in out.result["validation"]["errors"]]
    assert "unsupported_cite" in codes
    assert out.saved == [] and out.lookups == [] and out.created == []


async def test_typed_flashcards_fills_placeholder_zero_card_deck(monkeypatch):
    deck = NS(id=uuid.uuid4(), name="Placeholder", card_count=0, flashcards=[])
    out = await _run_typed(
        monkeypatch,
        lambda *a, **k: _spec([_card(1), _card(2)]),
        decks=[deck],
    )
    assert out.result["state"] == "ready"
    assert out.result["deck_filled"] is True
    assert out.result["deck_id"] == str(deck.id)
    assert len(out.created) == 2
    assert len(out.saved) == 1


async def test_typed_flashcards_never_rewrites_a_deck_that_has_cards(monkeypatch):
    existing = NS(id=uuid.uuid4(), front="Existing question one",
                  back="Existing answer", difficulty="easy")
    deck = NS(id=uuid.uuid4(), name="Reviewed", card_count=1,
              flashcards=[existing])
    out = await _run_typed(
        monkeypatch,
        lambda *a, **k: _spec([_card(1), _card(2)]),
        decks=[deck],
    )
    assert out.result["state"] == "ready"
    assert out.result["deck_filled"] is False
    assert out.created == []  # no card added, removed, or reordered
    assert len(out.saved) == 1


async def test_typed_flashcards_projection_dedupes_and_skips_blanks(monkeypatch):
    def _project(cards):
        return [
            {"front": "Same front here", "back": "First back", "difficulty": "easy"},
            {"front": "  same   front HERE ", "back": "Second back",
             "difficulty": "hard"},
            {"front": "No back at all", "back": "   ", "difficulty": "easy"},
        ]

    out = await _run_typed(
        monkeypatch,
        lambda *a, **k: _spec([_card(1), _card(2)]),
        decks=[],
        to_deck=_project,
    )
    assert out.result["state"] == "ready"
    assert len(out.created) == 1
    assert out.created[0][0] == "Same front here"


async def test_read_flashcards_reports_zero_card_deck_as_empty(monkeypatch):
    from services.transformation.routes import read_flashcards

    deck = NS(id=uuid.uuid4(), name="Zero", card_count=0, flashcards=[])

    async def _decks(db, pid):
        return [deck]

    monkeypatch.setattr(repositories, "get_flashcard_decks_by_project", _decks)
    project = NS(id=uuid.uuid4())

    empty = await read_flashcards(project.id, project=project, db=object())
    assert empty["state"] == "empty"
    assert "no cards" in empty["message"]
    assert empty["total"] == 1

    deck.card_count = 1
    deck.flashcards = [NS(id=uuid.uuid4(), front="Q?", back="A.",
                          difficulty="easy")]
    ready = await read_flashcards(project.id, project=project, db=object())
    assert ready["state"] == "ready"
    assert ready["cards"][0]["front"] == "Q?"


# ---------------------------------------------------------------------------
# Retrieval: a scoped search that fused to zero rows gets ONE OR-joined
# lexical retry, reported in meta -- never a silent "no evidence".
# ---------------------------------------------------------------------------

class _EmbResult:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class _Embed:
    model_name = "test-embedding-model"
    _vec = [0.1, 0.2, 0.3, 0.4]

    async def embed_queries(self, texts, batch_size=32):
        return _EmbResult([list(self._vec) for _ in texts])

    async def embed_passages(self, texts, batch_size=32):
        return _EmbResult([list(self._vec) for _ in texts])

    async def generate_embedding(self, text):
        return list(self._vec)

    @property
    def dimensions(self):
        return len(self._vec)


def _chunk(content):
    from database.models import Chunk

    vec = list(_Embed._vec)
    return Chunk(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        content=content,
        embedding=struct.pack(f"{len(vec)}f", *vec),
        embedding_model=_Embed.model_name,
        embedding_dim=len(vec),
        page_start=1,
        page_end=1,
        section_path=["Ch 1", "Sec A"],
        heading="Sec A",
        chunk_metadata={"method": "native", "confidence": 0.99},
    )


class _FakeDB:
    async def rollback(self):
        return None


async def _service(monkeypatch, hybrid):
    from services.retrieval.service import VectorRetrievalService
    from services.retrieval.types import RetrievalConfig

    import database.repositories as repos

    async def _state(db, project_id=None, document_id=None,
                     embedding_model=None):
        return {"state": "ok", "indexed_chunks": 1, "total_chunks": 1}

    monkeypatch.setattr(repos, "get_indexing_state", _state)
    svc = VectorRetrievalService(
        embedding_provider=_Embed(),
        config=RetrievalConfig(embedding_model=_Embed.model_name),
    )
    monkeypatch.setattr(svc, "_hybrid_search", hybrid)
    return svc


async def test_relaxed_lexical_retry_runs_once_and_flags_meta(monkeypatch):
    chunk = _chunk("photosynthesis chlorophyll stroma thylakoid")
    calls = []

    async def _hybrid(db, query_text, query_embedding, project_id=None,
                      document_id=None, user_id=None, **kwargs):
        relaxed = bool(kwargs.get("relaxed"))
        calls.append(relaxed)
        if relaxed:
            return [(chunk, 0.9, "lexical")], "pgvector"
        return [], "pgvector"

    svc = await _service(monkeypatch, _hybrid)
    res = await svc.retrieve_with_db(
        "photosynthesis chlorophyll stroma", _FakeDB(),
        project_id=str(uuid.uuid4()), enable_rerank=False,
    )
    assert calls == [False, True]  # strict pass, then exactly one retry
    assert res.meta["recovery"] == "lexical_or"
    assert res.meta["backend"] == "pgvector"
    assert res.total_results == 1


async def test_no_relaxed_retry_without_scope_or_pgvector(monkeypatch):
    calls = []

    async def _pgvector_unscoped(db, query_text, query_embedding,
                                 project_id=None, document_id=None,
                                 user_id=None, **kwargs):
        # Unscoped search: there is no scope to relax against.
        calls.append(bool(kwargs.get("relaxed")))
        return [], "pgvector"

    svc = await _service(monkeypatch, _pgvector_unscoped)
    res = await svc.retrieve_with_db(
        "photosynthesis", _FakeDB(), enable_rerank=False)
    assert calls == [False]
    assert "recovery" not in res.meta

    async def _offline(db, query_text, query_embedding, project_id=None,
                       document_id=None, user_id=None, **kwargs):
        # Scoped, but the DB arm is not Postgres: nothing to OR-relax.
        calls.append(bool(kwargs.get("relaxed")))
        return [], "offline-fallback"

    calls.clear()
    svc2 = await _service(monkeypatch, _offline)
    res2 = await svc2.retrieve_with_db(
        "photosynthesis", _FakeDB(), project_id=str(uuid.uuid4()),
        enable_rerank=False)
    assert calls == [False]
    assert "recovery" not in res2.meta
    assert res2.total_results == 0


async def test_fused_sql_binds_the_relaxed_or_switch():
    import inspect

    from services.retrieval.hybrid import FUSED_SQL

    assert "CAST(:relaxed AS integer)" in FUSED_SQL
    # Same statement, same scope: only the lexical arm's join changes.
    assert "plainto_tsquery('english', :query_text)" in FUSED_SQL
    assert "&', '|'" in FUSED_SQL
    # Parameter style stays portable (no `::` bare-colon casts).
    assert ":relaxed::" not in FUSED_SQL
    assert ":(project|document|user)_id" not in FUSED_SQL

    sig = inspect.signature(repositories.search_similar_chunks_hybrid)
    assert "relaxed" in sig.parameters
    assert sig.parameters["relaxed"].default is False


# ---------------------------------------------------------------------------
# Tutor: a topic question gets ONE retrieval whose context is returned with
# the answer; a greeting gets none.
# ---------------------------------------------------------------------------

async def _tutor(monkeypatch, message, chunks):
    from services.provider_credentials import override
    from services.retrieval.service import retrieval_service
    from services.study import routes
    from services.study.routes import TutorMessageRequest

    user_id, project_id = uuid.uuid4(), uuid.uuid4()
    project = NS(id=project_id, user_id=user_id, name="Biology")
    calls = {"retrieval": 0}

    async def project_for_request(*args, **kwargs):
        return project

    async def retrieve(*args, **kwargs):
        if chunks is None:
            raise AssertionError("this message must not trigger retrieval")
        calls["retrieval"] += 1
        return NS(chunks=list(chunks))

    async def generate(self, prompt, config=None):
        return NS(text="Photosynthesis stores energy in glucose.")

    monkeypatch.setattr(repositories, "get_project_by_id", project_for_request)
    monkeypatch.setattr(retrieval_service, "retrieve_with_db", retrieve)
    monkeypatch.setattr(override.EffectiveLLMProvider, "generate", generate)

    result = await routes.tutor_message(
        TutorMessageRequest(project_id=str(project_id), message=message),
        current_user=NS(id=user_id),
        db=object(),
    )
    return result, calls


async def test_tutor_topic_answer_returns_the_server_side_context(monkeypatch):
    chunk = NS(
        chunk_id="c1",
        document_id="document-9",
        content="Photosynthesis converts light into chemical energy.",
        page=4,
        citation=NS(method="dense"),
    )
    result, calls = await _tutor(
        monkeypatch, "Explain photosynthesis", [chunk])
    # One retrieval only: the client must not run a second, divergent search.
    assert calls["retrieval"] == 1
    assert result["retrieval_used"] is True
    assert result["grounded"] is True
    # The returned context is exactly the text the answer was grounded on.
    assert result["context"] == chunk.content
    assert result["citations"][0]["document_id"] == "document-9"
    assert result["sources"] == ["c1"]


async def test_tutor_greeting_never_retrieves(monkeypatch):
    # chunks=None makes the fake retrieval raise: a greeting must not query.
    result, calls = await _tutor(monkeypatch, "Hi!", None)
    assert calls["retrieval"] == 0
    assert result["retrieval_used"] is False
    assert result["grounded"] is False
    assert "context" not in result
    assert result["sources"] == []


# ---------------------------------------------------------------------------
# Notes synthesis: a REAL chunk id quoted inexactly is re-aligned to text
# taken from that chunk; a foreign id, or an empty chunk, still fails.
# ---------------------------------------------------------------------------

from services.academic.notes_hier import _verbatim_span, synthesize_notes
from services.academic.types import (
    DocumentStructure,
    ExtractedTopic,
    StructuredUnit,
    TopicNotes,
    TopicShape,
)
from services.transformation.engine import normalize_text

# The shape the live per-topic model actually emitted: an elided quote whose
# halves are separated in the chunk, so it never occurs verbatim there.
ELIDED_SPAN = (
    "Module is just a single file containing code ... "
    "holds functions and classes"
)
MODULE_TEXT = (
    "A module is just a single file containing code. "
    "A single Python file (.py) holds functions and classes for one concern. "
    "Packages group modules together under a directory with an init file."
)


def _py_spine(topic_title="Modules", chunk_id=CID_A):
    return DocumentStructure(
        doc_id="doc-python",
        title="Python",
        units=[StructuredUnit(title="Basics", topics=[
            ExtractedTopic(title=topic_title, chunks=[chunk_id])])],
        fingerprint="f" * 64,
    )


def _py_note(span, chunk_id=CID_A, topic_title="Modules"):
    return TopicNotes(
        unit_title="Basics",
        topic_title=topic_title,
        heading=topic_title,
        body_paragraphs=["Modules are files that hold reusable Python code."],
        key_terms=["module"],
        shape=TopicShape.CONCEPT,
        citations=[TopicCitation(chunk_id=chunk_id, span=span)],
        provenance_note="stub per-topic notes",
    )


async def test_synthesis_realigns_an_elided_model_span_to_the_chunk_text():
    artifact = synthesize_notes(
        "Py notes", _py_spine(), [_py_note(ELIDED_SPAN)],
        chunk_texts={CID_A: MODULE_TEXT},
    )
    cite = artifact.sections[0].citations[0]
    assert cite.chunk_id == CID_A
    # The stored quote now literally occurs in the cited chunk.
    assert normalize_text(cite.span) in normalize_text(MODULE_TEXT)
    assert "re-aligned" in artifact.provenance_note
    # The elision is gone: no model-authored "..." survives into the artifact.
    assert "..." not in cite.span


async def test_synthesis_quotes_the_chunk_when_the_span_shares_nothing():
    artifact = synthesize_notes(
        "Py notes", _py_spine(), [_py_note("quantum entanglement of photons")],
        chunk_texts={CID_A: MODULE_TEXT},
    )
    cite = artifact.sections[0].citations[0]
    # Fallback quotes the chunk's own first words: real text, never invented.
    assert cite.span.split()[:3] == ["A", "module", "is"]
    assert normalize_text(cite.span) in normalize_text(MODULE_TEXT)
    assert "re-aligned" in artifact.provenance_note


async def test_synthesis_keeps_a_literal_span_untouched():
    literal = "single file containing code"
    artifact = synthesize_notes(
        "Py notes", _py_spine(), [_py_note(literal)],
        chunk_texts={CID_A: MODULE_TEXT},
    )
    assert artifact.sections[0].citations[0].span == literal
    assert "re-aligned" not in artifact.provenance_note


async def test_synthesis_still_rejects_a_foreign_chunk_id():
    # Re-alignment only ever touches real ids: fabrication still raises.
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        synthesize_notes(
            "Py notes", _py_spine(),
            [_py_note("foreign words quoted here", chunk_id="ghost-chunk")],
            chunk_texts={CID_A: MODULE_TEXT},
        )


async def test_synthesis_fails_honestly_when_the_cited_chunk_is_empty():
    # No text to quote from -> nothing is invented, the gate still raises.
    with pytest.raises(StructuredGenerationError, match="unsupported"):
        synthesize_notes(
            "Py notes", _py_spine(), [_py_note("anything at all")],
            chunk_texts={CID_A: "   "},
        )


async def test_verbatim_span_crosses_punctuation_only_tokens():
    body = "A module code - file layout keeps things simple to follow."
    got = _verbatim_span("module code file", body)
    assert got is not None
    assert normalize_text(got) in normalize_text(body)


# ---------------------------------------------------------------------------
# Topic vocabulary: a bare topic title from THIS project reaches retrieval
# instead of the canned clarifier; greetings still pay for nothing.
# ---------------------------------------------------------------------------

async def test_project_topic_titles_is_failure_tolerant():
    from services.study.routes import _project_topic_titles

    # A non-session (or dead) DB must degrade to an empty vocabulary, never
    # raise: the tutor still answers with its honest clarification.
    assert await _project_topic_titles(object(), uuid.uuid4()) == []


async def test_tutor_bare_topic_query_retrieves_and_cites(monkeypatch):
    from services.study import routes

    async def titles(db, project_uuid):
        return ["Photosynthesis", "Cell Division"]

    monkeypatch.setattr(routes, "_project_topic_titles", titles)

    chunk = NS(
        chunk_id="c1",
        document_id="document-9",
        content="Photosynthesis converts light into chemical energy.",
        page=4,
        citation=NS(method="dense"),
    )
    # "photosynthesis" alone: <=3 words, no academic verb -> ambiguous
    # BEFORE the project's topic titles are consulted.
    result, calls = await _tutor(monkeypatch, "photosynthesis", [chunk])
    assert calls["retrieval"] == 1
    assert result["intent"] == "project_specific"
    assert result["retrieval_used"] is True
    assert result["grounded"] is True
    assert result["citations"][0]["chunk_id"] == "c1"
    assert result["sources"] == ["c1"]
    assert result["response"] == "Photosynthesis stores energy in glucose."


async def test_tutor_unknown_short_message_still_asks(monkeypatch):
    from services.study import routes

    async def titles(db, project_uuid):
        return ["Photosynthesis"]

    monkeypatch.setattr(routes, "_project_topic_titles", titles)
    # Not a topic of this project -> the ambiguous clarifier, no retrieval
    # (chunks=None makes the fake retrieval raise if it is ever called).
    result, calls = await _tutor(monkeypatch, "quasars", None)
    assert calls["retrieval"] == 0
    assert result["intent"] == "ambiguous"
    assert result["retrieval_used"] is False
    assert result["sources"] == []


async def test_tutor_greeting_never_even_looks_up_topics(monkeypatch):
    from services.study import routes

    async def titles(db, project_uuid):
        raise AssertionError("a greeting must not resolve topic titles")

    monkeypatch.setattr(routes, "_project_topic_titles", titles)
    result, calls = await _tutor(monkeypatch, "Thank you.", None)
    assert calls["retrieval"] == 0
    assert result["intent"] == "conversational"


# ---------------------------------------------------------------------------
# Boilerplate is never persisted as notes, and the single-call response says
# "empty" instead of reporting a headings-only payload as completed.
# ---------------------------------------------------------------------------

async def test_persist_domain_rows_rejects_placeholder_only_notes(monkeypatch):
    async def fake_note(db, project_id, title, content=None):
        raise AssertionError("boilerplate must never reach the notes row")

    monkeypatch.setattr(repositories, "create_note", fake_note)
    engine = TransformationEngine.__new__(TransformationEngine)
    pid = uuid.uuid4()

    await engine._persist_domain_rows(
        None, pid, GenerationMode.NOTES,
        {"title": "T", "sections": [
            {"heading": "Topic A",
             "content": "No information available for this topic."},
            {"heading": "Topic B", "content": "   "},
        ]},
    )

    # A real section next to a declared gap still persists the payload.
    seen = {}

    async def real_note(db, project_id, title, content=None):
        seen["title"] = title
        return NS(id=uuid.uuid4())

    monkeypatch.setattr(repositories, "create_note", real_note)
    await engine._persist_domain_rows(
        None, pid, GenerationMode.NOTES,
        {"title": "T", "sections": [
            {"heading": "Topic A",
             "content": "No information available for this topic."},
            {"heading": "Topic B",
             "content": "Glycolysis splits glucose into two pyruvate molecules."},
        ]},
    )
    assert seen["title"] == "T"


async def test_single_call_completion_state_is_honest():
    from services.transformation.routes import _completion_state

    # Headings-only / blank / boilerplate -> empty, never "completed".
    assert _completion_state(
        "notes", {"sections": [{"heading": "H", "content": "  "}]},
    )["state"] == "empty"
    assert _completion_state(
        "notes", {"sections": [{"heading": "H",
                                "content": "No information available."}]},
    )["state"] == "empty"
    assert _completion_state("notes", {"sections": []})["state"] == "empty"
    # Real prose -> ready.
    assert _completion_state(
        "notes", {"sections": [{"heading": "H",
                                "content": "Glycolysis splits glucose."}]},
    )["state"] == "ready"
    # A zero-card deck is not a completed deck either.
    assert _completion_state("flashcards", {"flashcards": []})["state"] == "empty"
    assert _completion_state(
        "flashcards", {"flashcards": [{"front": "Q?", "back": "A."}]},
    )["state"] == "ready"
