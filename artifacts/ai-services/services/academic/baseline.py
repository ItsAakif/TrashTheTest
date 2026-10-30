"""STEP 0 baseline: run the CURRENT Notes pipeline deterministically.

Only injection_document.pdf plus the synthetic corpus exist as fixtures,
so the synthetic corpus (tests/eval/corpus.py: 3 documents, 36 chunks)
is the deterministic fixture. This module reuses the CURRENT pipeline
pieces (TransformationEngine._build_context with document_order=True,
_trim_context with MAX_CONTEXT_TOKENS, validate_grounding) instead of
reimplementing them, then measures source topics, generated topics,
order accuracy, concept/citation coverage, dupes, length, missing, and
grounding. No LLM, no DB, no judgements of "looks better": numbers only.
"""

from typing import Any, Dict, List

from services.transformation.engine import (
    TransformationEngine,
    normalize_text,
    validate_grounding,
)
from services.retrieval.types import RetrievedChunk, RetrievalResult

BASELINE_STRATEGY = "single_shot"
BASELINE_GENERATOR = "current-notes-pipeline"
BASELINE_FIXTURE = "synthetic-corpus-36"


def corpus_to_retrieval(corpus: List[Dict[str, Any]]) -> RetrievalResult:
    """Wrap corpus rows as retrieval chunks with a relevance ranking that
    deliberately differs from document order (alternating similarity), so
    the document-order path is genuinely exercised, not trivially passed.
    """
    chunks: List[RetrievedChunk] = []
    for i, row in enumerate(corpus):
        similarity = 0.95 - (0.02 * (i % 7)) - (0.001 * i)
        chunks.append(
            RetrievedChunk(
                chunk_id=row["id"],
                content=row["text"],
                similarity=similarity,
                metadata={"page_start": row.get("page"), "position": i},
                document_id=row.get("doc"),
                page=row.get("page"),
                section=row.get("heading"),
            )
        )
    chunks.sort(key=lambda c: -c.similarity)
    return RetrievalResult(
        query="general content", total_results=len(chunks),
        chunks=chunks, processing_time=0.0,
    )


def build_current_notes_baseline(
    corpus: List[Dict[str, Any]],
    engine: TransformationEngine = None,
) -> Dict[str, Any]:
    """Emulate the CURRENT single-shot Notes pipeline offline.

    Mirrors engine.generate(): retrieve -> _build_context(document_order)
    -> _trim_context(MAX_CONTEXT_TOKENS) -> one structured-notes payload.
    Section prose is extractive (chunk text verbatim) and labelled as such;
    a live LLM would paraphrase, but section ORDER, chunk cap, and token
    ceiling are exactly the current pipeline's.
    """
    engine = engine or TransformationEngine.__new__(TransformationEngine)
    retrieval = corpus_to_retrieval(corpus)
    context = engine._build_context(retrieval, document_order=True)
    trimmed = engine._trim_context(context)
    # Which chunk ids survived the context cap (at most 10 blocks)?
    # Match the full delimited block header so id prefixes (e.g. *-01
    # vs *-10) can never false-positive on each other.
    used_ids: List[str] = []
    for chunk in retrieval.chunks:
        if f"[source chunk_id={chunk.chunk_id} " in trimmed:
            used_ids.append(chunk.chunk_id)
    by_id = {c.chunk_id: c for c in retrieval.chunks}
    # Preserve document order among the surviving chunks.
    ordered = sorted(used_ids, key=lambda cid: corpus_index(corpus, cid))
    sections: List[Dict[str, Any]] = []
    for cid in ordered:
        chunk = by_id[cid]
        row = corpus_by_id(corpus, cid)
        span = " ".join((chunk.content or "").split()[:8])
        sections.append({
            "heading": row.get("heading") or cid,
            "content": f"[extractive-offline] {chunk.content}",
            "citations": [{"chunk_id": cid, "span": span}],
            "page": row.get("page"),
        })
    return {
        "strategy": BASELINE_STRATEGY,
        "generator": BASELINE_GENERATOR,
        "fixture": BASELINE_FIXTURE,
        "title": "Baseline notes (current single-shot pipeline, offline)",
        "sections": sections,
        "context": {
            "chunks_used": len(used_ids),
            "chunks_total": len(corpus),
            "truncated": len(used_ids) < len(corpus),
            "max_context_tokens": TransformationEngine.MAX_CONTEXT_TOKENS,
            "max_chunks": 10,
        },
    }


def corpus_index(corpus: List[Dict[str, Any]], cid: str) -> int:
    for i, row in enumerate(corpus):
        if row["id"] == cid:
            return i
    return len(corpus)


def corpus_by_id(corpus: List[Dict[str, Any]], cid: str) -> Dict[str, Any]:
    for row in corpus:
        if row["id"] == cid:
            return row
    return {}


def measure_baseline(
    artifact: Dict[str, Any], corpus: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Deterministic metrics for a baseline notes artifact."""
    source_topics: List[str] = []
    for row in corpus:
        heading = row.get("heading") or row["id"]
        if heading not in source_topics:
            source_topics.append(heading)
    sections = artifact.get("sections") or []
    generated_topics = [s.get("heading", "") for s in sections]
    # Order accuracy: fraction of adjacent generated-heading pairs whose
    # source positions are non-decreasing (source order authoritative).
    pos = {h: i for i, h in enumerate(source_topics)}
    pairs = [
        (generated_topics[i], generated_topics[i + 1])
        for i in range(len(generated_topics) - 1)
        if generated_topics[i] in pos and generated_topics[i + 1] in pos
    ]
    if pairs:
        ordered = sum(1 for a, b in pairs if pos[a] <= pos[b])
        order_accuracy = round(ordered / len(pairs), 4)
    else:
        order_accuracy = 1.0
    # Concept coverage: distinct cited chunk ids over total chunks.
    cited: set = set()
    for section in sections:
        for cite in section.get("citations") or []:
            if isinstance(cite, dict) and cite.get("chunk_id"):
                cited.add(str(cite["chunk_id"]))
    all_ids = {row["id"] for row in corpus}
    concept_coverage = round(len(cited & all_ids) / len(all_ids), 4) if all_ids else 1.0
    # Citation coverage: sections carrying at least one citation.
    cited_sections = sum(1 for s in sections if (s.get("citations") or []))
    citation_coverage = (
        round(cited_sections / len(sections), 4) if sections else 0.0
    )
    # Dupes: repeated normalized headings beyond the first occurrence.
    seen: set = set()
    dupes = 0
    for heading in generated_topics:
        key = normalize_text(heading)
        if key in seen:
            dupes += 1
        else:
            seen.add(key)
    # Length: characters + engine-consistent token estimate (4 chars/token).
    length_chars = sum(len(s.get("content") or "") for s in sections)
    length_tokens_est = length_chars // 4
    # Missing: source headings with zero cited chunks.
    heading_chunks: Dict[str, set] = {}
    for row in corpus:
        heading_chunks.setdefault(row.get("heading") or row["id"], set()).add(row["id"])
    missing = sorted(
        h for h, cids in heading_chunks.items() if not (cids & cited)
    )
    # Grounding gate over the real engine validator (raises on fabrication).
    chunk_texts = {row["id"]: row["text"] for row in corpus}
    payload = {"sections": sections}
    gate = validate_grounding(payload, all_ids, chunk_texts)
    return {
        "source_topics": len(source_topics),
        "generated_topics": len(generated_topics),
        "order_accuracy": order_accuracy,
        "concept_coverage": concept_coverage,
        "citation_coverage": citation_coverage,
        "dupes": dupes,
        "length_chars": length_chars,
        "length_tokens_est": length_tokens_est,
        "missing_count": len(missing),
        "missing": missing,
        "grounding": gate,
    }


def run_baseline(corpus: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the artifact and attach its measurements (fully deterministic)."""
    artifact = build_current_notes_baseline(corpus)
    artifact["metrics"] = measure_baseline(artifact, corpus)
    return artifact


__all__ = [
    "BASELINE_STRATEGY",
    "BASELINE_GENERATOR",
    "BASELINE_FIXTURE",
    "corpus_to_retrieval",
    "build_current_notes_baseline",
    "corpus_index",
    "corpus_by_id",
    "measure_baseline",
    "run_baseline",
]
