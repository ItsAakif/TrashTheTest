"""STEP 0 eval harness: CURRENT Notes pipeline baseline (deterministic).

Fixture: the synthetic corpus (tests/eval/corpus.py: 3 documents,
36 chunks) -- the only deterministic fixture available (the audit found
only injection_document.pdf plus this corpus). No LLM, no DB, no
network, no "looks-better" judgements: pinned numbers only.
"""

import json

import pytest

from services.academic.baseline import (
    BASELINE_FIXTURE,
    BASELINE_GENERATOR,
    BASELINE_STRATEGY,
    build_current_notes_baseline,
    measure_baseline,
    run_baseline,
)
from tests.eval.corpus import CORPUS

pytestmark = pytest.mark.asyncio


async def test_baseline_runs_deterministically():
    first = run_baseline(CORPUS)
    second = run_baseline(CORPUS)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


async def test_baseline_labels_current_pipeline():
    artifact = run_baseline(CORPUS)
    assert artifact["strategy"] == BASELINE_STRATEGY == "single_shot"
    assert artifact["generator"] == BASELINE_GENERATOR
    assert artifact["fixture"] == BASELINE_FIXTURE
    assert artifact["context"]["max_chunks"] == 10
    assert artifact["context"]["chunks_used"] <= 10
    assert artifact["context"]["truncated"] is True  # 36 chunks, 10-slot cap


async def test_baseline_pinned_numbers():
    """Pinned STEP 0 numbers (change only when the pipeline changes)."""
    metrics = run_baseline(CORPUS)["metrics"]
    assert metrics["source_topics"] == 36
    assert metrics["generated_topics"] == 10
    assert metrics["order_accuracy"] == 1.0
    assert metrics["concept_coverage"] == 0.2778
    assert metrics["citation_coverage"] == 1.0
    assert metrics["dupes"] == 0
    assert metrics["length_chars"] == 1030
    assert metrics["length_tokens_est"] == 257
    assert metrics["missing_count"] == 26
    assert metrics["grounding"]["unsupported"] == []


async def test_baseline_sections_cite_only_corpus_chunks_with_spans():
    artifact = run_baseline(CORPUS)
    allowed = {row["id"] for row in CORPUS}
    texts = {row["id"]: row["text"] for row in CORPUS}
    assert artifact["sections"], "baseline must produce sections"
    for section in artifact["sections"]:
        assert section["citations"], f"section {section['heading']!r} uncited"
        for cite in section["citations"]:
            assert cite["chunk_id"] in allowed
            from services.transformation.engine import normalize_text

            assert normalize_text(cite["span"]) in normalize_text(
                texts[cite["chunk_id"]]
            )


async def test_baseline_follows_source_order():
    artifact = run_baseline(CORPUS)
    order = [row["id"] for row in CORPUS]
    cited = [s["citations"][0]["chunk_id"] for s in artifact["sections"]]
    positions = [order.index(cid) for cid in cited]
    assert positions == sorted(positions)


async def test_current_notes_pipeline_runs_on_corpus(monkeypatch, dummy_db):
    """The CURRENT Notes pipeline executes end to end (stub LLM + corpus)."""
    from services.retrieval.types import RetrievalResult, RetrievedChunk
    from services.strategies.base import GenerationResult
    from services.transformation import engine as enginemod
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    ordered = sorted(CORPUS, key=lambda r: (r["page"], r["id"]))
    notes_payload = {
        "title": "Corpus notes",
        "sections": [
            {
                "heading": row["heading"],
                "content": row["text"],
                "citations": [{"chunk_id": row["id"], "span": row["text"][:40]}],
            }
            for row in ordered[:4]
        ],
    }

    class StubLLM:
        def __init__(self):
            self.calls = 0

        async def generate(self, prompt, config=None):
            self.calls += 1
            return GenerationResult(
                text=json.dumps(notes_payload), model="stub", usage={},
                finish_reason="stop", processing_time=0.0,
            )

    stub = StubLLM()
    engine = TransformationEngine(llm_provider=stub)
    chunks = [
        RetrievedChunk(
            chunk_id=row["id"], content=row["text"], similarity=0.9,
            metadata={"page_start": row["page"]},
            document_id=row["doc"], page=row["page"], section=row["heading"],
        )
        for row in CORPUS
    ]
    fake_rr = RetrievalResult(
        query="q", total_results=len(chunks), chunks=chunks, processing_time=0.0
    )

    async def _fake_retrieve(**kwargs):
        return fake_rr

    async def _noop_cache(*a, **k):
        return None

    monkeypatch.setattr(enginemod.retrieval_service, "retrieve_with_db", _fake_retrieve)
    monkeypatch.setattr(engine, "_check_cache", _noop_cache)
    monkeypatch.setattr(engine, "_save_to_cache", _noop_cache)

    import uuid

    result = await engine.generate_notes(
        project_id=str(uuid.uuid4()), db=dummy_db, query="general content"
    )
    assert stub.calls == 1  # single-shot: exactly one provider call
    assert result.mode == GenerationMode.NOTES
    assert result.content["grounding"]["support"] == "retrieved"
    assert [s["heading"] for s in result.content["sections"]] == [
        s["heading"] for s in notes_payload["sections"]
    ]


async def test_baseline_summary_report():
    metrics = run_baseline(CORPUS)["metrics"]
    print(
        "\nSTEP0 baseline (current single-shot notes, synthetic corpus): "
        f"source_topics={metrics['source_topics']} "
        f"generated_topics={metrics['generated_topics']} "
        f"order_accuracy={metrics['order_accuracy']} "
        f"concept_coverage={metrics['concept_coverage']} "
        f"citation_coverage={metrics['citation_coverage']} "
        f"dupes={metrics['dupes']} "
        f"length_tokens_est={metrics['length_tokens_est']} "
        f"missing_count={metrics['missing_count']}; "
        "live_llm=NOT RUN; injection_pdf=NOT RUN (injection fixture, not academic)"
    )
