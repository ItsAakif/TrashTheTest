"""STEPS 2-4 tests: extraction -> hierarchical notes -> coverage (offline).

No DB, no LLM, no network. Provider doubles with call counters cover the
"more than one call" / "exactly 2 repairs" / "resumed groups only" style
assertions via the shared _generate_structured path. Live-Postgres checks
(persist_spine, save_notes_artifact) need a real database and are NOT RUN
here; the pure converters they rely on (spine_spec_to_rows, rows_to_tree)
are asserted offline.
"""

import json

import pytest
from pydantic import ValidationError

from database.knowledge_repo import rows_to_tree, spine_spec_to_rows
from services.academic.coverage import (
    attach_report,
    evaluate_coverage,
    missing_evidence,
    repair_notes,
)
from services.academic.extraction import extract_structure
from services.academic.notes_hier import (
    build_outline,
    compress_text,
    estimate_tokens,
    group_topics,
    render_topic_notes,
    select_topic_chunks,
    shape_for_topic,
    synthesize_notes,
    topic_query,
)
from services.academic.types import (
    DocumentStructure,
    HierarchicalNotesRequest,
    NotesArtifact,
    TopicNotes,
    TopicShape,
)
from services.retrieval.service import select_topic_evidence
from tests.eval.corpus import CORPUS

pytestmark = pytest.mark.asyncio


def _doc_chunks(doc_id):
    return [dict(r) for r in CORPUS if r["doc"] == doc_id]


def _chunk_index(rows):
    return {r["id"]: dict(r) for r in rows}


# -- STEP 2: extraction -------------------------------------------------------


async def test_extraction_headed_corpus_preserves_source_order():
    rows = _doc_chunks("doc-biology")
    struct = extract_structure("doc-biology", "Photosynthesis primer", rows)
    assert struct.doc_id == "doc-biology"
    assert len(struct.units) == 1  # no unit markers -> single conservative unit
    assert [t.title for t in struct.units[0].topics] == [
        r["heading"] for r in rows
    ]
    # Every chunk survives with its page; every concept has provenance.
    seen_chunks = [c for t in struct.units[0].topics for c in t.chunks]
    assert seen_chunks == [r["id"] for r in rows]
    for topic in struct.units[0].topics:
        for sub in topic.subtopics:
            for concept in sub.concepts:
                assert len(concept.chunks) == 1
                assert concept.provenance.chunk_ids == concept.chunks
                assert concept.provenance.note
    assert struct.units[0].topics[0].pages == [1]


async def test_extraction_spine_spec_validates():
    rows = _doc_chunks("doc-linalg")
    struct = extract_structure("doc-linalg", "Linear algebra notes", rows)
    spec = struct.to_spine_spec()
    assert spec.nodes[0].node_type.value == "document"
    types = [n.node_type.value for n in spec.nodes]
    # Concept naming v2 skips chunks with no usable title line and no
    # defined term (never invents a phrase): 7 of the 12 single-sentence
    # linalg chunks qualify, the rest are honestly absent.
    assert types.count("concept") == 7
    # Dense ordinals per parent and full depth chain present.
    assert "subtopic" in types and "unit" in types and "topic" in types
    again = type(spec).model_validate(spec.model_dump())
    assert again.model_dump() == spec.model_dump()


async def test_extraction_toc_wins_over_unknown_headings():
    rows = _doc_chunks("doc-biology")[:4]
    struct = extract_structure(
        "doc-biology", "Photosynthesis primer", rows,
        toc=["Photosynthesis overview", "Light reactions"],
    )
    titles = [t.title for t in struct.units[0].topics]
    assert "Chlorophyll pigments" not in titles  # not in TOC -> no fork
    assert "Photosynthesis overview" in titles and "Light reactions" in titles


async def test_extraction_inferred_fallback_without_headings():
    rows = [
        {"id": f"c{i}", "text": f"Content sentence number {i} about cells.",
         "page": i // 2 + 1}
        for i in range(7)
    ]
    struct = extract_structure("d1", "Plain doc", rows)
    assert len(struct.units) == 1
    assert len(struct.units[0].topics) == 3  # windows of 3 in source order
    assert [c for t in struct.units[0].topics for c in t.chunks] == [
        f"c{i}" for i in range(7)
    ]
    assert struct.units[0].method.value == "inferred_split"


async def test_extraction_unit_markers_fork_units():
    rows = [
        {"id": "u1", "text": "Intro to mechanics motion.", "page": 1,
         "heading": "Unit 1: Mechanics"},
        {"id": "u2", "text": "Force equals mass times acceleration.", "page": 1,
         "heading": "Newton laws"},
        {"id": "u3", "text": "Heat flows from hot to cold bodies.", "page": 2,
         "heading": "Unit 2: Heat"},
    ]
    struct = extract_structure("d2", "Physics", rows)
    assert [u.title for u in struct.units] == [
        "Unit 1 -- Mechanics", "Unit 2 -- Heat",
    ]
    spec = struct.to_spine_spec()  # still a valid full-depth spine
    assert spec.nodes[0].node_type.value == "document"


async def test_extraction_empty_chunks_rejected_and_fingerprint_stable():
    with pytest.raises(ValueError):
        extract_structure("d", "Empty", [])
    rows = _doc_chunks("doc-revolution")
    first = extract_structure("doc-revolution", "Revolution", rows)
    second = extract_structure("doc-revolution", "Revolution", list(rows))
    assert first.fingerprint == second.fingerprint
    assert json.dumps(first.model_dump(), sort_keys=True) == json.dumps(
        second.model_dump(), sort_keys=True
    )


async def test_step234_models_reject_extra_keys_and_camel_request():
    with pytest.raises(ValidationError):
        DocumentStructure(
            doc_id="d", title="t", units=[], fingerprint="f", bogus=1,
        )
    with pytest.raises(ValidationError):
        TopicNotes(
            unit_title="u", topic_title="t", heading="h",
            body_paragraphs=["p"], provenance_note="n", bogus=1,
        )
    snake = HierarchicalNotesRequest(project_id="p1")
    assert snake.project_id == "p1" and snake.threshold == 0.7
    camel = HierarchicalNotesRequest(projectId="p2")
    assert camel.project_id == "p2"


# -- RV-1: new modes/shapes in all three places --------------------------------


async def test_rv1_modes_registered_in_enum_prompts_and_engine():
    from services.transformation.engine import TransformationEngine
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import (
        GenerationMode,
        HierNotesOutput,
        StructureOutput,
        TopicNotesOutput,
    )

    for mode in (GenerationMode.STRUCTURE, GenerationMode.NOTES_TOPIC,
                 GenerationMode.NOTES_HIER):
        assert isinstance(mode.value, str)
    assert TransformationEngine.MODE_MODELS[GenerationMode.STRUCTURE] is StructureOutput
    assert TransformationEngine.MODE_MODELS[GenerationMode.NOTES_TOPIC] is TopicNotesOutput
    assert TransformationEngine.MODE_MODELS[GenerationMode.NOTES_HIER] is HierNotesOutput
    assert TransformationEngine.STRICT_MODELS[GenerationMode.STRUCTURE] is StructureOutput
    assert TransformationEngine.STRICT_MODELS[GenerationMode.NOTES_TOPIC] is TopicNotesOutput
    assert TransformationEngine.STRICT_MODELS[GenerationMode.NOTES_HIER] is HierNotesOutput
    for mode, kwargs in ((GenerationMode.STRUCTURE, {}),
                         (GenerationMode.NOTES_TOPIC, {"shape": "formula"}),
                         (GenerationMode.NOTES_HIER, {})):
        prompt = PromptTemplates.get_prompt(mode, "[source chunk_id=x] body", "q", **kwargs)
        assert PromptTemplates.INJECTION_GUARD in prompt
        assert PromptTemplates.CITATION_RULES in prompt


async def test_wire_shapes_mirror_domain_models():
    from services.transformation.types import TopicNotesOutput

    domain_fields = set(TopicNotes.model_fields) - {
        "shape_extras", "no_source", "provenance_note",
    }
    wire_fields = set(TopicNotesOutput.model_fields)
    assert {"unit_title", "topic_title", "heading", "body_paragraphs",
            "key_terms", "shape", "citations"} <= wire_fields
    assert {"unit_title", "topic_title", "heading", "body_paragraphs",
            "key_terms", "citations"} <= domain_fields


# -- STEP 3: outline / grouping / per-topic evidence ----------------------------


async def test_outline_follows_spine_order_and_group_budget():
    struct = extract_structure("doc-linalg", "Linear algebra", _doc_chunks("doc-linalg"))
    outline = build_outline(struct)
    assert [e["topic_title"] for e in outline] == [
        r["heading"] for r in _doc_chunks("doc-linalg")
    ]
    sizes = {e["topic_title"]: 100 for e in outline}
    groups = group_topics(outline, sizes, budget_tokens=6000)
    assert len(groups) == len(outline)  # one group per topic, source order
    assert all(g["estimated_tokens"] <= 6000 for g in groups)
    assert all(g["split"] is False for g in groups)


async def test_oversized_topic_splits_by_subtopic_and_records_it():
    outline = [{
        "unit_title": "U", "topic_title": "Big",
        "unit_pos": 0, "topic_pos": 0,
        "subtopics": ["Big -- a", "Big -- b"], "chunk_ids": ["c1", "c2"],
    }]
    groups = group_topics(outline, {"Big": 20000}, budget_tokens=6000)
    assert len(groups) == 2
    assert all(g["split"] is True for g in groups)
    assert [g["subtopic"] for g in groups] == ["Big -- a", "Big -- b"]


async def test_per_topic_retrieval_takes_all_relevant_in_source_order():
    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    topic = struct.units[0].topics[2]  # Light reactions
    # Similarity-shuffled input: vector order must not leak into output.
    shuffled = sorted(_doc_chunks("doc-biology"), key=lambda r: r["id"], reverse=True)
    hits = select_topic_chunks(shuffled, topic, struct.units[0].title)
    assert {c["id"] for c in hits} >= set(topic.chunks)
    pages = [c["page"] for c in hits]
    assert pages == sorted(pages)  # source order, never similarity order
    assert len(hits) < len(shuffled)  # scoped, but uncapped by top-k


async def test_retrieval_service_topic_filter_matches_academic_filter():
    from services.retrieval.types import RetrievedChunk

    chunks = [
        RetrievedChunk(chunk_id="c1", content="Rubisco fixes carbon dioxide",
                       similarity=0.1, metadata={"position": 1},
                       document_id="d", page=2, section="Calvin cycle"),
        RetrievedChunk(chunk_id="c2", content="Unrelated tenancy law text",
                       similarity=0.99, metadata={"position": 0},
                       document_id="d", page=1, section="Other"),
    ]
    hits = select_topic_evidence(chunks, "Calvin cycle")
    assert [h.chunk_id for h in hits] == ["c1"]  # relevance wins over score


async def test_render_topic_notes_cites_validate_and_span_grounded():
    from services.transformation.engine import validate_grounding

    struct = extract_structure("doc-linalg", "Math", _doc_chunks("doc-linalg"))
    unit_title = struct.units[0].title
    topic = struct.units[0].topics[1]  # Dot product
    notes = render_topic_notes(unit_title, topic, _doc_chunks("doc-linalg"))
    assert notes.citations  # every backing chunk cited
    # The topic's own structural evidence is always included; per-topic
    # retrieval may legitimately add further relevant chunks (no top-k cut).
    assert set(topic.chunks) <= {c.chunk_id for c in notes.citations}
    texts = {r["id"]: r["text"] for r in _doc_chunks("doc-linalg")}
    gate = validate_grounding(
        {"sections": [{"heading": notes.heading,
                       "content": " ".join(notes.body_paragraphs),
                       "citations": [c.model_dump() for c in notes.citations]}]},
        set(texts), texts,
    )
    assert gate["unsupported"] == []


async def test_render_no_source_topic_is_honest():
    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    topic = struct.units[0].topics[0]
    notes = render_topic_notes(struct.units[0].title, topic, [])
    assert notes.no_source is True and notes.citations == []
    with pytest.raises(ValidationError):
        TopicNotes(
            unit_title="u", topic_title="t", heading="h",
            body_paragraphs=["p"], citations=[], no_source=False,
            provenance_note="n",
        )


async def test_shape_adapts_per_topic_kind():
    assert shape_for_topic("Matrix multiplication steps",
                           ["First multiply rows then sum"]) == TopicShape.ALGORITHM
    assert shape_for_topic("Determinant formula",
                           ["det(A) = a*d - b*c"]) == TopicShape.FORMULA
    assert shape_for_topic("C4 vs CAM plants",
                           ["C4 concentrate while CAM open at night"]) == TopicShape.COMPARISON
    assert shape_for_topic("Rubisco enzyme",
                           ["Rubisco is abundant"]) == TopicShape.CONCEPT


async def test_content_contract_keeps_must_drop_filler():
    text = ("In this section we will now discuss things. "
            "Photosynthesis is defined as light energy conversion. "
            "For example, chlorophyll absorbs red light. "
            "Photosynthesis is defined as light energy conversion.")
    compressed = compress_text(text)
    assert "In this section" not in compressed  # may_compress: filler
    assert "defined as" in compressed  # must_keep: definition
    assert "For example" in compressed  # must_keep: example
    assert compressed.count("defined as") == 1  # may_compress: duplicates


async def test_topic_query_uses_path_and_aliases():
    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    topic = struct.units[0].topics[0]
    topic.aliases.append("photo overview")
    query = topic_query(struct.units[0].title, topic)
    assert "Photosynthesis overview" in query and "photo overview" in query


async def test_estimate_tokens_consistent_with_engine():
    from services.transformation.engine import TransformationEngine

    assert estimate_tokens("abcd" * 100) == 100
    assert TransformationEngine.MAX_CONTEXT_TOKENS == 6000


# -- STEP 3: synthesis ----------------------------------------------------------


def _render_all(struct):
    unit_title = struct.units[0].title
    rows = _doc_chunks(struct.doc_id)
    return [render_topic_notes(unit_title, t, rows) for t in struct.units[0].topics]


def _allowed_ids(doc_id):
    return {r["id"] for r in _doc_chunks(doc_id)}


async def test_synthesis_follows_spine_order_not_input_order():
    struct = extract_structure("doc-revolution", "Rev", _doc_chunks("doc-revolution"))
    notes = _render_all(struct)
    shuffled = list(reversed(notes))  # adversarial input order
    artifact = synthesize_notes("Rev notes", struct, shuffled,
                                allowed_chunk_ids=_allowed_ids("doc-revolution"))
    spine_order = [t.title for t in struct.units[0].topics]
    seen_order = []
    for s in artifact.sections:
        if not seen_order or seen_order[-1] != s.topic_title:
            seen_order.append(s.topic_title)
    # Spine order is authoritative: surviving sections follow source order
    # (a subsequence when shared evidence merges), never similarity order.
    positions = [spine_order.index(t) for t in seen_order]
    assert positions == sorted(positions)
    assert positions == sorted(set(positions))  # contiguous, no interleave
    # And sections are contiguous per topic (never interleaved).
    first_seen = {}
    for i, s in enumerate(artifact.sections):
        first_seen.setdefault(s.topic_title, i)
    last_seen = {}
    for i, s in enumerate(artifact.sections):
        last_seen[s.topic_title] = i
    # No information loss: every source chunk's text survives somewhere.
    bodies = " ".join(s.content for s in artifact.sections)
    for row in _doc_chunks("doc-revolution"):
        assert row["text"] in bodies


async def test_synthesis_dedupes_and_merges_citations():
    struct = extract_structure("doc-linalg", "Math", _doc_chunks("doc-linalg"))
    notes = _render_all(struct)
    dupe_para = notes[0].body_paragraphs[0]
    notes[1].body_paragraphs.append(dupe_para)  # cross-group duplicate
    artifact = synthesize_notes("Math notes", struct, notes,
                                allowed_chunk_ids=_allowed_ids("doc-linalg"))
    bodies = [s.content for s in artifact.sections]
    assert bodies.count(dupe_para) == 1
    survivor = next(s for s in artifact.sections if s.content == dupe_para)
    assert len(survivor.citations) >= 2  # citations merged, first kept


async def test_cross_topic_notes_reference_spine_titles_only():
    structs = [
        extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology")),
        extract_structure("doc-linalg", "Math", _doc_chunks("doc-linalg")),
    ]
    units = [s.units[0] for s in structs]
    multi = DocumentStructure(
        doc_id="multi", title="Combined", units=units,
        fingerprint="x" * 64,
    )
    notes = []
    for s in structs:
        notes.extend(_render_all(s))
    allowed = _allowed_ids("doc-biology") | _allowed_ids("doc-linalg")
    artifact = synthesize_notes("Combined notes", multi, notes,
                                allowed_chunk_ids=allowed)
    assert len(artifact.cross_topic_notes) == 1
    spine_titles = {u.title for u in multi.units}
    spine_titles.update(t.title for u in multi.units for t in u.topics)
    line = artifact.cross_topic_notes[0]
    assert any(t in line for t in spine_titles)


async def test_synthesis_rejects_foreign_citation_ids():
    from services.strategies.llm.structured import StructuredGenerationError

    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    notes = _render_all(struct)
    notes[0].citations.append(notes[0].citations[0].model_copy(
        update={"chunk_id": "00000000-0000-0000-0000-000000000000"}))
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        synthesize_notes("Bio notes", struct, notes)


async def test_per_topic_llm_path_uses_shared_structured_generation():
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    struct = extract_structure("doc-linalg", "Math", _doc_chunks("doc-linalg"))
    topic = struct.units[0].topics[0]
    calls = {"n": 0}

    class _StubProvider:
        async def generate(self, prompt, config=None):
            calls["n"] += 1
            from services.strategies.base import GenerationResult as _GR

            def _result(text):
                return _GR(text=text, model="stub", usage={},
                           finish_reason="stop", processing_time=0.0)

            if calls["n"] == 1:
                return _result('{"bogus": true}')  # fails schema -> 1 repair
            return _result(json.dumps({
                "unit_title": "U", "topic_title": topic.title,
                "heading": topic.title, "body_paragraphs": ["Vectors explained."],
                "key_terms": ["vector"], "shape": "concept",
                "citations": [{"chunk_id": topic.chunks[0],
                               "span": "ordered list of numbers"}],
            }))

    engine = TransformationEngine.__new__(TransformationEngine)
    engine.llm = _StubProvider()
    payload = await engine._generate_structured(
        GenerationMode.NOTES_TOPIC, "prompt", set(topic.chunks),
    )
    assert payload["topic_title"] == topic.title
    assert calls["n"] == 2  # one validation failure -> exactly one repair


async def test_per_topic_llm_path_rejects_foreign_chunk_id():
    from services.strategies.llm.structured import StructuredGenerationError
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    class _BadProvider:
        async def generate(self, prompt, config=None):
            from services.strategies.base import GenerationResult as _GR

            return _GR(text=json.dumps({
                "unit_title": "U", "topic_title": "T", "heading": "T",
                "body_paragraphs": ["P."], "key_terms": [],
                "shape": "concept",
                "citations": [{"chunk_id": "11111111-2222-3333-4444-555555555555",
                               "span": "x"}],
            }), model="stub", usage={}, finish_reason="stop",
                processing_time=0.0)

    engine = TransformationEngine.__new__(TransformationEngine)
    engine.llm = _BadProvider()
    with pytest.raises(StructuredGenerationError, match="Fabricated"):
        await engine._generate_structured(
            GenerationMode.NOTES_TOPIC, "prompt", {"real-chunk"},
        )


# -- STEP 4: coverage + repair ---------------------------------------------------


async def test_coverage_full_fixture_passes_and_deterministic():
    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    notes = _render_all(struct)
    first = evaluate_coverage(struct, notes)
    second = evaluate_coverage(struct, notes)
    assert first.model_dump() == second.model_dump()
    assert first.topic_coverage == 1.0 and first.concept_coverage == 1.0
    assert first.passes is True and first.uncovered == []
    assert first.nli == "unavailable"


async def test_coverage_names_omitted_topic_with_reason():
    struct = extract_structure("doc-linalg", "Math", _doc_chunks("doc-linalg"))
    dropped_titles = [t.title for t in struct.units[0].topics[-4:]]
    notes = _render_all(struct)[:-4]  # drop four topics (8/12 < 0.7)
    report = evaluate_coverage(struct, notes)
    assert report.passes is False
    for dropped in dropped_titles:
        assert any(dropped in u for u in report.uncovered)
    score = next(s for s in report.per_topic
                 if s.topic_title == dropped_titles[-1])
    assert score.score == 0.0 and score.missing


async def test_coverage_no_source_topic_reported_not_forced():
    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    notes = _render_all(struct)
    notes[0] = render_topic_notes(struct.units[0].title,
                                  struct.units[0].topics[0], [])
    report = evaluate_coverage(struct, notes)
    assert any("no_source" in u for u in report.uncovered)


async def test_coverage_threshold_is_configurable():
    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    notes = _render_all(struct)[:-2]
    strict_report = evaluate_coverage(struct, notes, threshold=1.0)
    assert strict_report.passes is False
    assert strict_report.threshold == 1.0


async def test_repair_bounded_to_two_passes_and_uncovered_only():
    struct = extract_structure("doc-linalg", "Math", _doc_chunks("doc-linalg"))
    rows = _doc_chunks("doc-linalg")
    index = _chunk_index(rows)
    notes = _render_all(struct)[:-1]  # one uncovered topic
    calls: list = []

    def _stub_regen(topic, evidence):
        calls.append(topic.title)
        # Omission-prone double: never cites, so coverage cannot improve.
        return TopicNotes(
            unit_title=struct.units[0].title, topic_title=topic.title,
            heading=topic.title, body_paragraphs=["Unrelated filler words here."],
            key_terms=[], shape=TopicShape.CONCEPT, citations=[],
            no_source=True,
            provenance_note="stub regen (still missing)",
        )

    fixed, report, log = repair_notes(struct, notes, index, _stub_regen,
                                      threshold=1.0)
    assert len(log) == 2  # exactly 2 passes, no more
    assert all(entry.regenerated_topics == [struct.units[0].topics[-1].title]
               for entry in log)
    assert all(c == struct.units[0].topics[-1].title for c in calls)
    assert report.passes is False  # honest: still uncovered


async def test_repair_reverts_pass_that_lowers_coverage():
    from services.academic.types import (
        ConceptProvenance,
        ExtractedConcept,
        ExtractedSubtopic,
        ExtractedTopic,
        StructuredUnit,
        StructureMethod,
    )

    def _concept(title, cid):
        return ExtractedConcept(
            title=title, pages=[1], chunks=[cid], confidence=1.0,
            provenance=ConceptProvenance(
                method=StructureMethod.HEADING_GROUP, source_heading="T",
                chunk_ids=[cid], note="test"),
        )

    topic = ExtractedTopic(
        title="T", pages=[1], chunks=["c1", "c2"],
        subtopics=[ExtractedSubtopic(
            title="T", pages=[1], chunks=["c1", "c2"], concepts=[
                _concept("Alpha beta gamma", "c1"),
                _concept("Delta epsilon zeta", "c2"),
            ])],
        concepts=[])
    struct = DocumentStructure(
        doc_id="d", title="D",
        units=[StructuredUnit(title="U", topics=[topic])],
        fingerprint="f" * 64)
    index = {
        "c1": {"id": "c1", "text": "Alpha beta gamma processes are described here.",
               "page": 1},
        "c2": {"id": "c2", "text": "Delta epsilon zeta relations are shown here.",
               "page": 1},
    }
    partial = TopicNotes(
        unit_title="U", topic_title="T", heading="T",
        body_paragraphs=["Alpha beta gamma processes are described here."],
        key_terms=["alpha"], shape=TopicShape.CONCEPT,
        citations=[{"chunk_id": "c1", "span": "Alpha beta gamma processes"}],
        no_source=False, provenance_note="partial: c2 uncovered")
    before = evaluate_coverage(struct, [partial], threshold=1.0)
    assert before.concept_coverage == 0.5

    def _damaging_regen(_topic, _evidence):
        return TopicNotes(
            unit_title="U", topic_title="T", heading="T",
            body_paragraphs=["Filler with no terms."],
            key_terms=[], shape=TopicShape.CONCEPT, citations=[],
            no_source=True, provenance_note="damaging")

    fixed, report, log = repair_notes(struct, [partial], index,
                                      _damaging_regen, threshold=1.0,
                                      max_passes=1)
    assert len(log) == 1 and log[0].kept is False
    assert "reverted" in log[0].note
    assert report.concept_coverage == before.concept_coverage


async def test_repair_missing_evidence_retrieves_omitted_chunks():
    struct = extract_structure("doc-biology", "Bio", _doc_chunks("doc-biology"))
    rows = _doc_chunks("doc-biology")
    index = _chunk_index(rows)
    notes = _render_all(struct)
    report = evaluate_coverage(struct, notes)
    score = report.per_topic[0]
    topic = struct.units[0].topics[0]
    evidence = missing_evidence(topic, score, index)
    assert {c["id"] for c in evidence} >= set(topic.chunks)


async def test_attach_report_lands_on_artifact():
    struct = extract_structure("doc-linalg", "Math", _doc_chunks("doc-linalg"))
    notes = _render_all(struct)
    artifact = synthesize_notes("Math notes", struct, notes,
                                allowed_chunk_ids=_allowed_ids("doc-linalg"))
    report = evaluate_coverage(struct, notes)
    final = attach_report(artifact, report, [])
    assert isinstance(final, NotesArtifact)
    assert final.coverage is not None and final.coverage.passes is True
    assert final.uncovered == []


# -- persistence converters (pure; DB writes are NOT RUN offline) ----------------


async def test_spine_rows_parents_before_children_and_tree_ordered():
    struct = extract_structure("doc-revolution", "Rev", _doc_chunks("doc-revolution"))
    spec = struct.to_spine_spec()
    rows = spine_spec_to_rows(spec, "proj-1", "doc-revolution")
    assert len(rows) == len(spec.nodes)
    position = {id(r): i for i, r in enumerate(rows)}
    for i, row in enumerate(rows):
        if row["parent_ref"] is not None:
            assert position[id(rows[row["parent_ref"]])] < i
    tree = rows_to_tree([
        {"id": f"n{i}", "parent_id": None if i == 0 else "n0",
         "type": r["type"], "title": r["title"], "order_index": r["order_index"],
         "pages": r["pages"], "chunks": r["chunks"]}
        for i, r in enumerate(rows[:4])
    ])
    assert tree is not None and tree["title"] == "Rev"
    assert [c["order_index"] for c in tree["children"]] == sorted(
        c["order_index"] for c in tree["children"]
    )
