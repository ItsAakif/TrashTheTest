"""Eval report writer (T308).

Runs the offline quality harness and writes
.planning/evidence/eval-phase3.md with the metrics table, fixture
inventory, index/plan notes, and the rerank latency measurement.

Usage (from artifacts/ai-services):
    python -m scripts.eval_report --out ../../.planning/evidence/eval-phase3.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.eval.corpus import CORPUS, DOCUMENTS
from tests.eval.test_retrieval_quality import (
    GOLDEN_PATH,
    NDCG_GATE,
    RECALL_GATE,
    evaluate,
    load_golden,
    rank_all,
)


def measure_rerank_latency_ms(queries, repeats: int = 3) -> float:
    from tests.eval.scorers import rerank_score
    from tests.eval.corpus import CORPUS as _C

    by_id = {c["id"]: c for c in _C}
    best = None
    for _ in range(max(1, repeats)):
        start = time.time()
        for entry in queries:
            fused = rank_all(entry["query"])["fused"][:50]
            sorted(fused, key=lambda cid: (-rerank_score(entry["query"], by_id[cid]), cid))[:8]
        elapsed_ms = (time.time() - start) * 1000.0 / len(queries)
        best = elapsed_ms if best is None else min(best, elapsed_ms)
    return best or 0.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 offline eval report (T308)")
    parser.add_argument("--out", default=".planning/evidence/eval-phase3.md")
    args = parser.parse_args(argv)

    queries = load_golden()
    summary = evaluate(queries)
    rerank_ms = measure_rerank_latency_ms(queries)
    fused, reranked = summary["fused"], summary["reranked"]
    delta = reranked["nDCG@10"] - fused["nDCG@10"]
    verdict = (
        "PASS"
        if reranked["recall@10"] >= RECALL_GATE and reranked["nDCG@10"] >= NDCG_GATE
        else "FAIL"
    )

    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        golden_size = len(json.load(fh)["queries"])

    lines = [
        "# Phase 3 retrieval eval (T308)",
        "",
        f"Verdict: **{verdict}** (gates: recall@10 >= {RECALL_GATE:.2f}, "
        f"nDCG@10 >= {NDCG_GATE:.2f}, fused+rerank row).",
        "",
        "## Metrics (offline deterministic harness, no providers)",
        "",
        "| mode | recall@10 | nDCG@10 | MRR |",
        "| --- | --- | --- | --- |",
    ]
    for mode in ("vector", "lexical", "fused", "reranked"):
        row = summary[mode]
        lines.append(
            f"| {mode} | {row['recall@10']:.3f} | {row['nDCG@10']:.3f} | {row['MRR']:.3f} |"
        )
    lines += [
        "",
        f"Rerank delta (reranked minus fused nDCG@10): **{delta:+.4f}**.",
        f"Rerank latency (offline proxy scorer, per query over 50 candidates): "
        f"{rerank_ms:.2f} ms. Live bge-reranker-v2-m3 latency is measured at "
        "the acceptance gate against real weights and recorded in "
        ".planning/evidence/phase3-acceptance.md.",
        "",
        "## Fixture inventory",
        "",
        f"- Documents: {len(DOCUMENTS)} ({', '.join(d['id'] for d in DOCUMENTS)})",
        f"- Chunks: {len(CORPUS)} (12 per document, pages 1-6)",
        f"- Golden queries: {golden_size} (exact-term + paraphrase mix; "
        "see tests/eval/golden_queries.json)",
        "",
        "## Index / plan notes (live Postgres required)",
        "",
        "- Vector arm: ORDER BY embedding_v <=> :q LIMIT :k over the HNSW "
        "index ix_chunks_embedding_v_hnsw (built post-backfill by "
        "scripts/build_vector_index.py).",
        "- Lexical arm: ts_rank over the tsv generated column (GIN "
        "ix_chunks_tsv). Fusion: single-statement RRF (FULL OUTER JOIN).",
        "- Filtered queries set hnsw.iterative_scan=strict_order per session.",
        "- Live pgvector/HNSW EXPLAIN: NOT RUN in this environment (no live "
        "DB service). Do not treat offline proxy numbers as index proof.",
        "",
        "## BM25-extension decision (ADR-005 s5 trigger)",
        "",
        f"Lexical-only recall@10 = {summary['lexical']['recall@10']:.3f}, "
        f"nDCG@10 = {summary['lexical']['nDCG@10']:.3f}. If the lexical row "
        "is the bottleneck on the real corpus at the acceptance gate, file "
        "a follow-up task to adopt a real BM25 extension; otherwise keep "
        "native tsvector.",
        "",
        "## LLM-judge layer (opt-in, never in CI)",
        "",
        "- Not run by default. Optional command (requires provider keys):",
        "  `python -m scripts.llm_judge --in <generated answers> --out "
        ".planning/evidence/eval-phase3-judge.md` (script not shipped; "
        "record output per run when executed).",
        "",
    ]
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out_path}: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
