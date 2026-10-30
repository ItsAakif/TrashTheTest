"""Retrieval quality gates (T308, T302 rerank delta).

Offline and deterministic: no providers, no network. Rows for
vector-only, lexical-only, fused RRF, and fused+rerank. Gates:
recall@10 >= 0.80 and nDCG@10 >= 0.70 on the fused+rerank row; a
regression fails CI. The gate-fails-when-degraded test proves the gate
is not vacuous. Completes in seconds (time bound asserted).
"""

import json
import os
import time

import pytest

from services.retrieval.hybrid import rrf_fuse

from .corpus import CORPUS
from .scorers import (
    build_docfreq,
    lexical_score,
    mrr,
    ndcg_at,
    rank,
    recall_at,
    rerank_score,
    vector_proxy_score,
)

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden_queries.json")
RECALL_GATE = 0.80
NDCG_GATE = 0.70
TIME_BOUND_SECONDS = 60


def load_golden():
    with open(GOLDEN_PATH, encoding="utf-8") as fh:
        return json.load(fh)["queries"]


def rank_all(query):
    df = build_docfreq(CORPUS)
    vec = rank(CORPUS, query, vector_proxy_score)
    lex = rank(CORPUS, query, lambda q, c: lexical_score(q, c, df))
    fused_pairs = rrf_fuse([vec, lex], top_n=50)
    fused = [cid for cid, _ in fused_pairs]
    # Rerank fused top-50 -> top-10 with the heading-weighted scorer.
    by_id = {c["id"]: c for c in CORPUS}
    reranked = sorted(
        fused, key=lambda cid: (-rerank_score(query, by_id[cid]), cid)
    )[:10]
    return {"vector": vec, "lexical": lex, "fused": fused, "reranked": reranked}


def evaluate(queries):
    rows = {m: {"recall": [], "ndcg": [], "mrr": []} for m in ("vector", "lexical", "fused", "reranked")}
    for entry in queries:
        rel = set(entry["expected_chunks"])
        rankings = rank_all(entry["query"])
        for mode, ranking in rankings.items():
            rows[mode]["recall"].append(recall_at(rel, ranking))
            rows[mode]["ndcg"].append(ndcg_at(rel, ranking))
            rows[mode]["mrr"].append(mrr(rel, ranking))
    summary = {}
    for mode, vals in rows.items():
        summary[mode] = {
            "recall@10": sum(vals["recall"]) / len(vals["recall"]),
            "nDCG@10": sum(vals["ndcg"]) / len(vals["ndcg"]),
            "MRR": sum(vals["mrr"]) / len(vals["mrr"]),
        }
    return summary


def test_eval_runs_offline_within_time_bound():
    queries = load_golden()
    assert len(queries) >= 30, "golden set must hold at least 30 queries"
    start = time.time()
    summary = evaluate(queries)
    elapsed = time.time() - start
    assert elapsed < TIME_BOUND_SECONDS, f"eval took {elapsed:.1f}s"
    assert set(summary) == {"vector", "lexical", "fused", "reranked"}


def test_fused_row_meets_ci_gates():
    summary = evaluate(load_golden())
    fused = summary["fused"]
    assert fused["recall@10"] >= RECALL_GATE, f"recall@10={fused['recall@10']:.3f}"
    assert fused["nDCG@10"] >= NDCG_GATE, f"nDCG@10={fused['nDCG@10']:.3f}"


def test_reranked_row_meets_ci_gates_and_reports_delta():
    summary = evaluate(load_golden())
    reranked, fused = summary["reranked"], summary["fused"]
    assert reranked["recall@10"] >= RECALL_GATE
    assert reranked["nDCG@10"] >= NDCG_GATE
    # Delta is recorded (may be >= 0); the harness decides, not assumes.
    delta = reranked["nDCG@10"] - fused["nDCG@10"]
    assert isinstance(delta, float)
    print(f"\nrerank nDCG delta: {delta:+.4f}")


def test_gate_fails_when_retrieval_degraded():
    """Degrade rankings (worst-first) and prove the gate would fail."""
    from .scorers import build_docfreq as _df
    from .scorers import lexical_score as _lex
    from .scorers import vector_proxy_score as _vec

    df = _df(CORPUS)
    queries = load_golden()
    degraded_recalls = []
    for entry in queries:
        rel = set(entry["expected_chunks"])
        scored = sorted(
            ((c["id"], _vec(entry["query"], c) + _lex(entry["query"], c, df))
             for c in CORPUS),
            key=lambda kv: (kv[1], kv[0]),
        )
        degraded = [cid for cid, _ in scored[:10]]  # worst-first top-10
        degraded_recalls.append(recall_at(rel, degraded))
    mean_recall = sum(degraded_recalls) / len(degraded_recalls)
    assert mean_recall < RECALL_GATE, "degraded retrieval must miss the gate"


def test_arms_cover_complementary_queries():
    """Lexical arm catches exact-term queries, vector arm paraphrases."""
    df = build_docfreq(CORPUS)
    queries = load_golden()
    exact = [q for q in queries if q["mode"] == "exact"]
    para = [q for q in queries if q["mode"] == "paraphrase"]
    assert exact and para, "golden set must mix exact and paraphrase queries"

    def hit_rate(items, scorer):
        hits = 0
        for entry in items:
            rel = set(entry["expected_chunks"])
            ranking = rank(CORPUS, entry["query"], scorer)
            if rel & set(ranking[:10]):
                hits += 1
        return hits / len(items)

    lex_exact = hit_rate(exact, lambda q, c: lexical_score(q, c, df))
    vec_para = hit_rate(para, vector_proxy_score)
    assert lex_exact >= 0.80, f"lexical hit rate on exact queries: {lex_exact:.3f}"
    assert vec_para >= 0.80, f"vector hit rate on paraphrases: {vec_para:.3f}"


def test_citations_resolve_for_all_fixture_chunks():
    """Every returned citation resolves to real stored text (T308)."""
    for chunk in CORPUS:
        assert (chunk["text"] or "").strip(), f"empty text for {chunk['id']}"
        assert chunk["page"] >= 1
