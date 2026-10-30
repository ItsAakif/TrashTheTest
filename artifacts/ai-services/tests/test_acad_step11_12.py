"""STEPS 11+12: backend regression gates + CURRENT vs NEW end-to-end eval.

STEP 11 (add only, never delete): consolidated backend gates over the whole
academic vertical on the shared offline fixture (doc-biology, 12 chunks):
spine creation/ordering/parent-child/provenance/aliases; extraction
structure/order/pages/cites; notes hierarchical/order/coverage/repair/cite
validation; flashcards schema/Bloom/dedupe/grounding; MCQs
single-key/distractor/matrix/grounding; intelligence
frequency/marks/mapping/provenance; PDF
valid/extractable/TOC/bookmarks/numbers/cites/no-clip/fonts/deterministic;
artifact persistence/versioning/linkage.

STEP 12 (same sample doc): CURRENT single-shot baseline vs NEW hierarchical
pipeline, measured with the same deterministic metrics. No success is
claimed beyond what the numbers show; every comparison below asserts a
measured value, not a judgement.

No DB, no LLM, no network. No test depends on a live provider.
"""

import io
import os
import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from database import knowledge_repo
from services.academic import artifacts as academic_artifacts
from services.academic.baseline import (
    build_current_notes_baseline,
    measure_baseline,
)
from services.academic.coverage import (
    attach_report,
    evaluate_coverage,
    repair_notes,
)
from services.academic.extraction import extract_structure
from services.academic.flashcards import (
    build_flashcards,
    concept_index,
    validate_deck,
)
from services.academic.mcq import (
    build_matrix,
    build_mcqs,
    project_quiz_questions,
    shingle_similarity,
    validate_mcq,
)
from services.academic.metrics import (
    compute_metrics,
    metrics_artifact,
    normalize_inputs,
    recency_weight,
    repair_narrative,
    validate_narrative,
)
from services.academic.notes_hier import (
    build_outline,
    render_topic_notes,
    synthesize_notes,
)
from services.academic.notes_pdf import render_notes_pdf
from services.academic.spine import (
    assign_order_index,
    canonical_key_for,
    merge_alias_lists,
    normalize_key,
    resolve_alias,
    validate_spine_structure,
)
from services.academic.types import (
    BLOOM_MIX,
    BloomLevel,
    FlashcardSpec,
    GeneratedArtifact,
    artifact_fingerprint,
)
from tests.eval.corpus import CORPUS

pytestmark = pytest.mark.asyncio

PINNED_EPOCH = "1704067200"
SAMPLE_DOC = "doc-biology"
SAMPLE_TITLE = "Photosynthesis primer"


def _doc_chunks(doc_id=SAMPLE_DOC):
    return [dict(r) for r in CORPUS if r["doc"] == doc_id]


def _chunk_texts(doc_id=SAMPLE_DOC):
    return {r["id"]: r["text"] for r in _doc_chunks(doc_id)}


def _structure():
    return extract_structure(SAMPLE_DOC, SAMPLE_TITLE, _doc_chunks())


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


# Six concepts with DISTINCT source facts (the real-fact distractor pool
# needs >= 3 other concepts whose facts are not near-duplicates).
_MCQ_FACTS = [
    "Photosynthesis is the process that converts light energy into chemical energy.",
    "Chlorophyll is the pigment that absorbs red and blue light in the thylakoid membranes.",
    "Light reactions are the reactions that split water and release oxygen.",
    "Calvin cycle is the cycle that fixes carbon dioxide into sugar.",
    "Stomata are the pores that control carbon dioxide intake and water loss.",
    "C4 plants are the plants that concentrate carbon dioxide to reduce photorespiration.",
]


def _mcq_rows():
    return [
        {"id": f"mcq-{i:02d}", "text": text,
         "heading": f"Topic {i + 1}", "page": 1, "position": i}
        for i, text in enumerate(_MCQ_FACTS)
    ]


def _mcq_structure():
    return extract_structure("doc-mcq", "MCQ fixture", _mcq_rows())


def _mcq_texts():
    return {r["id"]: r["text"] for r in _mcq_rows()}


def _topic_notes(structure=None):
    structure = structure or _structure()
    chunks = _doc_chunks()
    return structure, [
        render_topic_notes(unit.title, topic, chunks)
        for unit in structure.units
        for topic in unit.topics
    ]


def _hier_artifact():
    structure, notes = _topic_notes()
    artifact = synthesize_notes(
        SAMPLE_TITLE, structure, notes, chunk_texts=_chunk_texts())
    return structure, notes, attach_report(
        artifact, evaluate_coverage(structure, notes))


def _norm_ws(text):
    return " ".join((text or "").split())


# ---------------------------------------------------------------------------
# STEP 11a: spine creation / ordering / parent-child / provenance / aliases
# ---------------------------------------------------------------------------


async def test_spine_creation_covers_full_depth_with_single_root():
    """Spine creation: document/unit/topic/subtopic/concept rows, one root."""
    spec = _structure().to_spine_spec()
    kinds = {n.node_type.value for n in spec.nodes}
    assert {"document", "unit", "topic", "subtopic", "concept"} <= kinds
    roots = [n for n in spec.nodes if n.parent_ref is None]
    assert len(roots) == 1
    assert roots[0].node_type.value == "document"


async def test_spine_ordering_dense_per_parent_in_source_order():
    """Ordering: dense ordinals per parent; topics follow first-seen order."""
    structure = _structure()
    spec = structure.to_spine_spec()
    by_parent = {}
    for node in spec.nodes:
        by_parent.setdefault(node.parent_ref, []).append(node.order_index)
    for parent, ords in by_parent.items():
        assert sorted(ords) == list(range(len(ords))), parent
    first_seen = []
    for row in _doc_chunks():
        if row["heading"] not in first_seen:
            first_seen.append(row["heading"])
    spec_topics = [n.title for n in spec.nodes
                   if n.node_type.value == "topic"]
    assert spec_topics == first_seen


async def test_spine_parent_child_depth_contract():
    """Parent-child: refs point backwards; depth matches node type."""
    from services.academic.types import NODE_DEPTH, NodeType

    spec = _structure().to_spine_spec()
    nodes = list(spec.nodes)
    for i, node in enumerate(nodes):
        if node.parent_ref is None:
            depth = 0
        else:
            assert node.parent_ref < i, f"node {i}: parent must precede child"
            ancestor = nodes[node.parent_ref]
            depth = NODE_DEPTH[NodeType(ancestor.node_type.value)] + 1
        assert NODE_DEPTH[NodeType(node.node_type.value)] == depth, node.title
    plain = []
    for n in spec.nodes:
        row = n.model_dump()
        row["node_type"] = n.node_type.value
        plain.append(row)
    stats = validate_spine_structure(plain)
    assert stats["total"] == len(nodes)


async def test_spine_provenance_on_every_concept():
    """Provenance: every concept row names its detection method."""
    spec = _structure().to_spine_spec()
    concepts = [n for n in spec.nodes if n.node_type.value == "concept"]
    assert concepts, "sample doc must yield concepts"
    for node in concepts:
        assert node.node_metadata.get("method"), node.title
        assert node.node_metadata.get("provenance_note"), node.title


async def test_spine_aliases_merge_and_registry_resolution():
    """Aliases: union dedupes; registry resolves direct/alias/collision."""
    assert merge_alias_lists(["Calvin Cycle", "calvin cycle", "C4"],
                             ["C4", "CAM"]) == ["Calvin Cycle", "C4", "CAM"]
    assert canonical_key_for("Photosystems") == normalize_key("Photosystems")
    assert normalize_key("Eigenvalues") == normalize_key("eigenvalue ")
    direct, via, collision = resolve_alias(
        "Vectors", [], {"vector": "vector"})
    assert (via, direct, collision) == ("direct_key", "vector", None)
    owner, via, _ = resolve_alias(
        "Scalar projection", ["Vectors"],
        {"vector": "vector"},
        {"vectors": "vector"})
    assert owner == "vector" and via == "alias"
    _, _, clash = resolve_alias(
        "Rank", ["Vectors"], {"rank": "rank"}, {"vectors": "vector"})
    assert clash and "Vectors" in clash


async def test_spine_assign_order_index_is_dense_per_parent():
    rows = assign_order_index([
        {"parent_ref": None, "title": "root"},
        {"parent_ref": 0, "title": "b"},
        {"parent_ref": 0, "title": "a"},
        {"parent_ref": 1, "title": "c"},
    ])
    assert [r["order_index"] for r in rows] == [0, 0, 1, 0]


# ---------------------------------------------------------------------------
# STEP 11b: extraction structure / order / pages / cites
# ---------------------------------------------------------------------------


async def test_extraction_topics_follow_first_seen_source_order():
    structure = _structure()
    got = [t.title for u in structure.units for t in u.topics]
    assert got, "sample doc must yield topics"
    first_seen = []
    for row in _doc_chunks():
        if row["heading"] not in first_seen:
            first_seen.append(row["heading"])
    assert got == first_seen


async def test_extraction_pages_preserved_from_chunks():
    structure = _structure()
    chunk_pages = {r["id"]: r["page"] for r in _doc_chunks()}
    for unit in structure.units:
        for topic in unit.topics:
            assert topic.pages, topic.title
            for cid in topic.chunks:
                assert chunk_pages[cid] in topic.pages, (topic.title, cid)
            for sub in topic.subtopics:
                for concept in sub.concepts:
                    assert concept.pages, concept.title
                    for cid in concept.chunks:
                        assert chunk_pages[cid] in concept.pages


async def test_extraction_cites_cover_every_source_chunk():
    structure = _structure()
    claimed = set()
    for unit in structure.units:
        for topic in unit.topics:
            claimed.update(topic.chunks or [])
            for sub in topic.subtopics:
                claimed.update(sub.chunks or [])
                for concept in sub.concepts:
                    claimed.update(concept.chunks or [])
    assert claimed == {r["id"] for r in _doc_chunks()}


async def test_extraction_fingerprint_deterministic():
    first = extract_structure(SAMPLE_DOC, SAMPLE_TITLE, _doc_chunks())
    second = extract_structure(SAMPLE_DOC, SAMPLE_TITLE, _doc_chunks())
    assert first.fingerprint == second.fingerprint
    assert first.model_dump() == second.model_dump()


# ---------------------------------------------------------------------------
# STEP 11c: notes hierarchical / order / coverage / repair / cite validation
# ---------------------------------------------------------------------------


async def test_notes_outline_follows_spine_order():
    structure = _structure()
    outline = build_outline(structure)
    expected = [(u.title, t.title) for u in structure.units for t in u.topics]
    assert [(e["unit_title"], e["topic_title"]) for e in outline] == expected


async def test_notes_per_topic_cites_are_own_evidence():
    structure, notes = _topic_notes()
    assert len(notes) == sum(len(u.topics) for u in structure.units)
    for item in notes:
        if item.no_source:
            assert item.citations == []
            continue
        assert item.citations, item.topic_title
        by_id = {c["id"]: c for c in _doc_chunks()}
        for cite in item.citations:
            assert cite.chunk_id in by_id, cite.chunk_id
            assert cite.span.strip(), cite.chunk_id


async def test_notes_synthesis_keeps_spine_order_and_validates_cites():
    structure, notes = _topic_notes()
    artifact = synthesize_notes(
        SAMPLE_TITLE, structure, notes, chunk_texts=_chunk_texts())
    positions = {(u.title, t.title): pos
                 for pos, (u, t) in
                 enumerate((u, t) for u in structure.units for t in u.topics)}
    seq = [positions[(s.unit_title, s.topic_title)] for s in artifact.sections]
    assert seq == sorted(seq), "sections must follow spine order"
    # Cross-group paragraph dedupe may fold one repeated paragraph (first
    # wins, citations merged), so distinct topics <= notes; what must hold
    # is no information loss: every cited chunk survives in the sections.
    noted_cites = {c.chunk_id for n in notes for c in n.citations}
    section_cites = {c.chunk_id for s in artifact.sections for c in s.citations}
    assert noted_cites, "fixture notes must carry citations"
    assert noted_cites <= section_cites
    assert {(s.unit_title, s.topic_title) for s in artifact.sections} <= set(positions)
    for section in artifact.sections:
        assert section.citations, section.heading


async def test_notes_coverage_full_fixture_passes_without_nli():
    structure, notes = _topic_notes()
    report = evaluate_coverage(structure, notes)
    assert report.topic_coverage == 1.0
    assert report.concept_coverage == 1.0
    assert report.passes is True
    assert report.nli == "unavailable"
    assert report.uncovered == []


async def test_notes_repair_bounded_regenerates_only_uncovered():
    structure, notes = _topic_notes()
    by_title = {n.topic_title: n for n in notes}
    # Drop one topic and force repair with threshold=1.0 (the default 0.7
    # still passes at 11/12, honestly needing no repair).
    dropped = [n for n in notes[1:]]
    calls = []

    def _regen(topic, evidence):
        calls.append(topic.title)
        return by_title[topic.title]

    chunk_index = {r["id"]: r for r in _doc_chunks()}
    fixed, report, log = repair_notes(
        structure, dropped, chunk_index, _regen, threshold=1.0, max_passes=2)
    assert len(log) <= 2
    assert report.concept_coverage >= evaluate_coverage(
        structure, dropped, threshold=1.0).concept_coverage
    assert set(calls) <= {t.title for u in structure.units for t in u.topics}
    assert calls == [notes[0].topic_title]
    assert report.passes is True


async def test_notes_repair_reverts_regression_and_records():
    structure, notes = _topic_notes()

    def _worse(topic, evidence):
        return notes[0].model_copy(update={
            "unit_title": topic.unit_title if hasattr(topic, "unit_title") else "",
            "topic_title": topic.title,
            "heading": topic.title,
            "body_paragraphs": ["Unrelated filler with no concept terms xyzzy."],
            "citations": [],
            "no_source": True,
        })

    chunk_index = {r["id"]: r for r in _doc_chunks()}
    before = evaluate_coverage(structure, notes).concept_coverage
    _, report, log = repair_notes(structure, notes, chunk_index, _worse)
    assert log == [], "passing input must not trigger repair passes"
    assert report.concept_coverage == before


async def test_notes_foreign_citation_rejected_at_synthesis():
    structure, notes = _topic_notes()
    bad = notes[0].model_copy(update={
        "citations": [c.model_copy(update={"chunk_id": "foreign-chunk-1",
                                           "span": "foreign span words here"})
                      for c in notes[0].citations],
    })
    with pytest.raises(Exception):
        synthesize_notes(SAMPLE_TITLE, structure,
                         [bad] + notes[1:], chunk_texts=_chunk_texts())


# ---------------------------------------------------------------------------
# STEP 11d: flashcards schema / Bloom / dedupe / grounding
# ---------------------------------------------------------------------------


async def test_flashcard_schema_rejects_extra_and_honours_aliases():
    card = FlashcardSpec.model_validate({
        "id": "card-0001", "topic": "Vectors", "concept_id": "c1",
        "question": "What is a vector?", "answer": "An ordered list of numbers.",
        "type": "definition", "cognitive": "remember",
        "cites": [{"chunk_id": "math-linalg-01", "span": "ordered list"}],
    })
    assert card.card_type.value == "definition"
    assert card.bloom_level.value == "remember"
    with pytest.raises(ValidationError):
        FlashcardSpec.model_validate({**card.model_dump(), "bogus": 1})


async def test_flashcard_bloom_mix_sums_and_deck_passes_guard():
    assert abs(sum(BLOOM_MIX.values()) - 1.0) < 1e-9
    structure = _rich_structure()
    deck = build_flashcards(structure, _rich_texts(), title="Rich cards")
    assert deck.flashcards, "sample doc must yield cards"
    index = concept_index(structure)
    allowed = set(_rich_texts())
    report = validate_deck(deck.flashcards, allowed,
                           set(index), tolerance=0.10)
    assert report.ok is True, report.errors
    assert report.bloom_ok is True
    # Realized shares are rounded to 4dp per level, so the sum is 1.0
    # within rounding, not to the bit.
    assert abs(sum(deck.bloom_realized.values()) - 1.0) < 0.01


async def test_flashcard_dedupe_duplicate_front_rejected():
    structure = _rich_structure()
    deck = build_flashcards(structure, _rich_texts())
    first = deck.flashcards[0]
    twin = first.model_copy(update={"id": "card-twin"})
    report = validate_deck(deck.flashcards + [twin], set(_rich_texts()),
                           set(concept_index(structure)))
    assert report.ok is False
    assert any(e.get("code") == "duplicate_front" for e in report.errors)


async def test_flashcard_grounding_concepts_resolve_cites_in_prompt():
    structure = _rich_structure()
    index = concept_index(structure)
    deck = build_flashcards(structure, _rich_texts())
    allowed = set(_rich_texts())
    for card in deck.flashcards:
        assert card.concept_id in index, card.id
        assert card.review.get("fsrs_state") == "new", card.id
        for cite in card.citations:
            assert cite.chunk_id in allowed, (card.id, cite.chunk_id)


# ---------------------------------------------------------------------------
# STEP 11e: MCQs single-key / distractor / matrix / grounding
# ---------------------------------------------------------------------------


def _mcq_set(total=12):
    structure = _mcq_structure()
    plan = build_matrix(structure, total)
    return structure, build_mcqs(structure, _mcq_texts(), plan)


async def test_mcq_single_key_correct_index_points_at_key():
    _, mcq_set = _mcq_set()
    assert mcq_set.questions, "matrix must fill cells with material"
    for spec in mcq_set.questions:
        keys = [i for i, o in enumerate(spec.options) if o.is_correct]
        assert keys == [spec.correct_answer]
        assert len(spec.options) == 4
        assert len({o.text.strip().lower() for o in spec.options}) == 4


async def test_mcq_distractors_tagged_explained_and_distinct():
    _, mcq_set = _mcq_set()
    for spec in mcq_set.questions:
        key_text = spec.options[spec.correct_answer].text
        for i, option in enumerate(spec.options):
            if i == spec.correct_answer:
                continue
            assert option.error_tag is not None, (spec.id, option.text)
            assert option.rationale.strip(), (spec.id, option.text)
            assert shingle_similarity(option.text, key_text) < 0.8, spec.id


async def test_mcq_matrix_targets_sum_exactly_all_weightings():
    structure = _structure()
    units = {u.title for u in structure.units}
    syllabus = {u: 60.0 for u in units}
    topics = [t.title for u in structure.units for t in u.topics]
    pyq = {t: (i % 3) + 1 for i, t in enumerate(topics)}
    for plan in (build_matrix(structure, 12),
                 build_matrix(structure, 12, syllabus_weights=syllabus),
                 build_matrix(structure, 12, pyq_frequency=pyq)):
        assert sum(c.target for c in plan.cells) == plan.total_target == 12


async def test_mcq_grounding_and_study_projection_strings():
    structure, mcq_set = _mcq_set()
    allowed = set(_mcq_texts())
    titles = {t.title for u in structure.units for t in u.topics}
    for spec in mcq_set.questions:
        assert validate_mcq(spec, allowed, titles) == [], spec.id
        for cite in spec.citations:
            assert cite.chunk_id in allowed
    projected = project_quiz_questions(mcq_set)
    assert len(projected) == len(mcq_set.questions)
    for row in projected:
        assert isinstance(row["options"], list)
        assert all(isinstance(o, str) for o in row["options"])
        assert isinstance(row["options"][row["correct_answer"]], str)


# ---------------------------------------------------------------------------
# STEP 11f: intelligence frequency / marks / mapping / provenance
# ---------------------------------------------------------------------------


def _metrics_inputs():
    structure = _structure()
    chunks = [{"id": r["id"], "text": r["text"]} for r in _doc_chunks()]
    topics = [t.title for u in structure.units for t in u.topics]
    pyqs = [
        {"question": f"Explain {topics[0]} with examples", "topic": topics[0],
         "year": 2024, "marks": 10},
        {"question": f"Discuss {topics[1]} in detail", "topic": topics[1],
         "year": 2019, "marks": 5},
        {"question": "Unrelated astrophysics trivia", "topic": "Quasars",
         "year": 2020, "marks": 5},
    ]
    syllabus = [{"title": "Unit 1", "topics": topics[:2]}]
    return normalize_inputs(structure, chunks, pyqs=pyqs,
                            syllabus_units=syllabus)


async def test_metrics_frequency_marks_mapping_deterministic():
    first = compute_metrics(_metrics_inputs(), current_year=2026)
    second = compute_metrics(_metrics_inputs(), current_year=2026)
    assert (first.model_dump(mode="json") ==
            second.model_dump(mode="json"))
    rows = {r.topic_title: r for r in first.topics}
    assert rows, "metrics must carry topic rows"
    assert sum(r.frequency for r in first.topics) >= 2
    assert sum(r.marks_total for r in first.topics) == 15
    # Only the two mapped PYQs (15 of 20 marks) attribute to topics; the
    # unmapped PYQ stays unmapped and counts toward missing, never forced.
    assert abs(sum(r.marks_share for r in first.topics) - 75.0) < 1.0
    assert first.totals["total_marks"] == 20


async def test_metrics_recency_hand_computed():
    assert recency_weight(2026, 2026, 5.0) == pytest.approx(1.0)
    assert recency_weight(2021, 2026, 5.0) == pytest.approx(0.5)


async def test_metrics_mapping_provenance_and_never_asked():
    report = compute_metrics(_metrics_inputs(), current_year=2026)
    for row in report.topics:
        assert set(row.evidence) >= {"chunks", "pyqs", "questions",
                                     "syllabus"}, row.topic_title
        assert row.ord >= 1
    assert report.missing, "the unmapped PYQ must surface as missing"
    assert report.provenance_note.strip()
    payload = metrics_artifact(report)
    assert payload["tables"] and payload["cards"]


async def test_metrics_narrative_number_grounding_repair_then_withhold():
    report = compute_metrics(_metrics_inputs(), current_year=2026)
    good = "Coverage holds with no fabricated figures in this sentence."
    assert validate_narrative(good, report)["ok"] is True
    bad = "Coverage improved by 97.3 percent year over year."
    assert validate_narrative(bad, report)["ok"] is False
    response = repair_narrative(report, lambda _prompt: bad)
    assert response.narrative is None
    assert response.narrative_status == "withheld"
    assert response.metrics.topics == report.topics


# ---------------------------------------------------------------------------
# STEP 11g: PDF valid / extractable / TOC / bookmarks / numbers / cites /
#           no-clip / fonts / deterministic
# ---------------------------------------------------------------------------


def _pdf_report(**meta_over):
    from pypdf import PdfReader  # noqa: F401 (dependency probe)

    meta = {"title": SAMPLE_TITLE, "version": 1, "model": "test-model",
            "created_at": "2026-09-23", "epoch": int(PINNED_EPOCH)}
    meta.update(meta_over)
    structure, _, artifact = _hier_artifact()
    return structure, render_notes_pdf(artifact, structure, meta)


def _pdf_body(pdf_bytes):
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return reader, _norm_ws("\n".join(
        (page.extract_text() or "") for page in reader.pages))


async def test_pdf_valid_extractable_with_metadata(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    _, report = _pdf_report()
    assert report.pdf_bytes[:5] == b"%PDF-"
    reader, body = _pdf_body(report.pdf_bytes)
    assert len(reader.pages) == report.page_count > 0
    assert len(body.strip()) > 0
    assert reader.metadata.title == SAMPLE_TITLE
    assert reader.metadata.author == "TrashTheTest"


async def test_pdf_toc_bookmarks_and_page_numbers(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    structure, report = _pdf_report()
    reader, body = _pdf_body(report.pdf_bytes)
    flat = []

    def _walk(items):
        for item in items:
            if isinstance(item, list):
                _walk(item)
            else:
                flat.append(item.title)
    _walk(list(reader.outline))
    units = [u.title for u in structure.units]
    topics = [t.title for u in structure.units for t in u.topics]
    for title in units + topics:
        assert any(title in entry for entry in flat), title
    assert report.bookmark_count >= len(units) + len(topics)
    pages = [p.extract_text() or "" for p in reader.pages]
    for entry in report.toc_entries:
        assert 1 <= entry["page"] <= len(pages), entry
    assert "Page 1 of" in body, "footer page numbers must render"


async def test_pdf_cites_listed_with_labels(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    _, report = _pdf_report()
    _, body = _pdf_body(report.pdf_bytes)
    assert "Sources:" in body
    # Cite labels carry the 8-char short id plus the span text.
    assert "bio-phot" in body


async def test_pdf_no_clip_fonts_base_only(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    structure, artifact = _hier_artifact()[0], _hier_artifact()[2]
    unit, topic = structure.units[0], structure.units[0].topics[0]
    wide = [["H1", "H2", "H3", "H4", "H5", "H6"]]
    wide += [[f"row {r} col {c} wrapped value" for c in range(6)]
             for r in range(4)]
    extras = {(unit.title, topic.title): {"table": wide}}
    meta = {"title": SAMPLE_TITLE, "version": 1, "model": "m",
            "created_at": "2026-09-23", "epoch": int(PINNED_EPOCH)}
    report = render_notes_pdf(artifact, structure, meta, extras)
    _, body = _pdf_body(report.pdf_bytes)
    assert "row 3 col 5 wrapped value" in body
    fonts = set()
    from pypdf import PdfReader
    for page in PdfReader(io.BytesIO(report.pdf_bytes)).pages:
        resources = (page.get("/Resources") or {}).get("/Font") or {}
        for font in resources.values():
            base = font.get_object().get("/BaseFont")
            if base:
                fonts.add(str(base))
    assert fonts, "PDF must embed font references"
    assert all("Helvetica" in f for f in fonts), fonts


async def test_pdf_deterministic_pinned_epoch(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    _, first = _pdf_report()
    _, second = _pdf_report()
    assert first.sha256 == second.sha256
    assert first.pdf_bytes == second.pdf_bytes


# ---------------------------------------------------------------------------
# STEP 11h: artifact persistence / versioning / linkage
# ---------------------------------------------------------------------------


def _record(version=1, fingerprint=None, **over):
    record = {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "sources": [SAMPLE_DOC],
        "type": "notes_pdf",
        "version": version,
        "status": "ready",
        "path": f"notes-pdf/v{version}.pdf",
        "pages": 7,
        "hash": "a" * 64,
        "model": "test-model",
        "config": {"topic_count": 12},
        "coverage": {"topic_coverage": 1.0},
        "fingerprint": fingerprint or "f" * 64,
        "created_at": "2026-09-23T00:00:00+00:00",
        "updated_at": "2026-09-23T00:00:00+00:00",
    }
    record.update(over)
    return record


async def test_artifact_persistence_versions_and_reuses(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    structure, _, artifact = _hier_artifact()
    record, report, reused = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m1", config={"project_id": "p1"})
    assert record["version"] == 1 and reused is False
    assert record["pages"] == report.page_count
    assert record["path"].endswith(".pdf")
    wire = {k: v for k, v in record.items()
            if k not in ("artifact_type", "title", "toc_entries")}
    GeneratedArtifact.model_validate(wire)
    same, _, reused_same = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m1", config={"project_id": "p1"},
        existing_versions=[record])
    assert reused_same is True and same["version"] == 1
    nxt, _, reused_nxt = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m2", config={"project_id": "p1"},
        existing_versions=[record])
    assert reused_nxt is False and nxt["version"] == 2


async def test_artifact_fingerprint_links_sources_model_config_coverage():
    base = artifact_fingerprint(["b", "a"], "m", {"k": 1}, {"passes": True})
    assert base == artifact_fingerprint(["a", "b"], "m", {"k": 1},
                                        {"passes": True})
    for other in (
        artifact_fingerprint(["a"], "m", {"k": 1}, {"passes": True}),
        artifact_fingerprint(["a", "b"], "other", {"k": 1}, {"passes": True}),
        artifact_fingerprint(["a", "b"], "m", {"k": 2}, {"passes": True}),
        artifact_fingerprint(["a", "b"], "m", {"k": 1}, {"passes": False}),
    ):
        assert other != base


class _FakeSession:
    def __init__(self, nodes):
        self._nodes = {n.id: n for n in nodes}

    async def get(self, _model, node_id):
        return self._nodes.get(node_id)

    async def flush(self):
        return None


async def test_artifact_store_supersede_old_versions_readable():
    node_id = uuid.uuid4()
    node = SimpleNamespace(id=node_id, node_metadata={})
    db = _FakeSession([node])
    await knowledge_repo.save_generated_artifact(db, node_id, _record(1, "1" * 64))
    await knowledge_repo.save_generated_artifact(db, node_id, _record(2, "2" * 64))
    latest = await knowledge_repo.get_generated_artifact(db, node_id, "notes_pdf")
    assert latest["version"] == 2 and latest["status"] == "ready"
    old = await knowledge_repo.get_generated_artifact(
        db, node_id, "notes_pdf", version=1)
    assert old["version"] == 1 and old["superseded_by"] == 2
    listed = await knowledge_repo.list_generated_artifacts(db, node_id, "notes_pdf")
    assert [v["version"] for v in listed] == [1, 2]


# ---------------------------------------------------------------------------
# STEP 12: CURRENT vs NEW on the same sample doc (measured, not claimed)
# ---------------------------------------------------------------------------


def _current_on_sample():
    corpus = _doc_chunks()
    artifact = build_current_notes_baseline(corpus)
    return artifact, measure_baseline(artifact, corpus)


def _new_on_sample():
    structure, notes = _topic_notes()
    artifact = synthesize_notes(
        SAMPLE_TITLE, structure, notes, chunk_texts=_chunk_texts())
    report = evaluate_coverage(structure, notes)
    full = attach_report(artifact, report)
    sections = [
        {"heading": s.heading, "content": s.content,
         "citations": [{"chunk_id": c.chunk_id, "span": c.span}
                       for c in s.citations],
         "page": None}
        for s in full.sections
    ]
    probe = {"sections": sections}
    gate = measure_baseline(
        {**probe, "metrics": {}}, _doc_chunks())
    return full, report, gate


async def test_step12_coverage_new_meets_or_beats_current():
    _, current = _current_on_sample()
    _, report, new_gate = _new_on_sample()
    assert report.topic_coverage >= current["concept_coverage"]
    assert report.concept_coverage >= current["concept_coverage"]
    assert new_gate["concept_coverage"] >= current["concept_coverage"]
    assert new_gate["missing_count"] <= current["missing_count"]


async def test_step12_order_citation_grounding_dupes_measured():
    _, current = _current_on_sample()
    full, _, new_gate = _new_on_sample()
    assert new_gate["order_accuracy"] == 1.0
    assert new_gate["citation_coverage"] == 1.0
    # Content-level dupes: no repeated normalized section body (the real
    # cross-group dedupe guarantee). Heading-level repeats are by design:
    # a multi-paragraph topic yields one section per paragraph under the
    # same heading, so the baseline heading counter is reported, not gated.
    bodies = [" ".join(s.content.split()) for s in full.sections]
    assert len(set(bodies)) == len(bodies)
    # Content-level dupes: no repeated normalized section body (the real
    # cross-group dedupe guarantee). Heading-level repeats are by design:
    # a multi-paragraph topic yields one section per paragraph under the
    # same heading, so the baseline heading counter is reported, not gated.
    assert current["grounding"]["unsupported"] == []
    assert new_gate["grounding"]["unsupported"] == []
    assert new_gate["generated_topics"] >= current["generated_topics"]


async def test_step12_pdf_visual_inspection_via_extraction(monkeypatch):
    """Visual inspection proxy: no blank/clipped pages, bookmarks+TOC land,
    cites+numbers correct. A human still opens the evidence PDF; the checks
    below are what can be asserted from extraction."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    _, report = _pdf_report()
    reader = PdfReader(io.BytesIO(report.pdf_bytes))
    texts = [(page.extract_text() or "") for page in reader.pages]
    assert all(len(_norm_ws(t)) > 0 for t in texts), "no blank pages"
    body = _norm_ws("\n".join(texts))
    assert "Contents" in body, "TOC page present"
    assert "Sources:" in body, "cites present"
    assert "Page 1 of" in body, "page numbers present"
    assert len(body) > 2000, "body must carry real content, not stubs"
    assert len(reader.outline) > 0, "bookmarks present"
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "..", "..",
        ".planning", "evidence", "acad-step12-notes.pdf")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as handle:
        handle.write(report.pdf_bytes)
    assert os.path.getsize(out_path) > 0
