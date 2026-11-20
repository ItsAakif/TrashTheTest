"""S2 retrieval evaluation: metrics + DB-backed strategy comparison.

Labels are ANCHOR STRINGS resolved against the project's stored chunks, so
the dataset is independent of random chunk UUIDs (which change on
rechunking). Metrics are pure functions and unit-tested; the runner needs a
real DB + embedding provider and is executed by scripts/eval_retrieval.py.

Provenance: dataset queries and anchors are manually authored against the
demo corpus. They are SYNTHETIC-MANUAL, not independently human-validated.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class EvalQuery:
    id: str
    query: str
    anchors: List[str]
    category: str = "general"
    difficulty: str = "medium"
    provenance: str = "synthetic-manual"


@dataclass
class QueryResult:
    id: str
    query: str
    relevant_ids: List[str]
    retrieved_ids: List[str]
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit: float
    latency_ms: float
    category: str = "general"
    difficulty: str = "medium"


@dataclass
class ConfigResult:
    config: str
    mode: str
    rerank: bool
    top_k: int
    aggregates: Dict[str, float]
    per_query: List[QueryResult] = field(default_factory=list)


def _norm(text: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", t).strip().casefold()


def resolve_anchors(chunks: Iterable[Any], anchors: Sequence[str]) -> Set[str]:
    """Map anchor strings to every chunk id whose text contains them."""
    normalised = [(str(c.id), _norm(getattr(c, "content", ""))) for c in chunks]
    out: Set[str] = set()
    for anchor in anchors:
        target = _norm(anchor)
        if not target:
            continue
        for cid, body in normalised:
            if target in body:
                out.add(cid)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / float(k)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    for i, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    gains = [1.0 if cid in relevant else 0.0 for cid in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def hit_rate(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


async def evaluate_config(
    db: Any,
    queries: Sequence[EvalQuery],
    project_id: str,
    *,
    mode: str,
    rerank: bool,
    top_k: int = 5,
    service: Any = None,
    chunks: Iterable[Any] = (),
) -> ConfigResult:
    """Run one retrieval configuration and aggregate the metrics."""
    from services.retrieval.service import retrieval_service

    svc = service or retrieval_service
    rows: List[QueryResult] = []
    latencies: List[float] = []
    for q in queries:
        # Anchors are resolved against the live corpus, never hardcoded UUIDs.
        relevant = resolve_anchors(chunks, q.anchors)
        started = time.time()
        result = await svc.retrieve_with_db(
            query=q.query,
            db=db,
            project_id=project_id,
            top_k=top_k,
            min_similarity=0.0,
            enable_rerank=rerank,
            retrieval_mode=mode,
        )
        latency = (time.time() - started) * 1000.0
        retrieved = [str(c.chunk_id) for c in result.chunks]
        latencies.append(latency)
        rows.append(QueryResult(
            id=q.id,
            query=q.query,
            relevant_ids=sorted(relevant),
            retrieved_ids=retrieved,
            recall=recall_at_k(retrieved, relevant, top_k),
            precision=precision_at_k(retrieved, relevant, top_k),
            mrr=reciprocal_rank(retrieved, relevant),
            ndcg=ndcg_at_k(retrieved, relevant, top_k),
            hit=hit_rate(retrieved, relevant, top_k),
            latency_ms=latency,
            category=q.category,
            difficulty=q.difficulty,
        ))

    def mean(key: str) -> float:
        return sum(getattr(r, key) for r in rows) / len(rows) if rows else 0.0

    aggregates = {
        "recall_at_k": mean("recall"),
        "precision_at_k": mean("precision"),
        "mrr": mean("mrr"),
        "ndcg_at_k": mean("ndcg"),
        "hit_rate_at_k": mean("hit"),
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "queries": float(len(rows)),
    }
    return ConfigResult(
        config=f"{mode}{'+rerank' if rerank else ''}",
        mode=mode,
        rerank=rerank,
        top_k=top_k,
        aggregates=aggregates,
        per_query=rows,
    )


def load_dataset(path: str) -> List[EvalQuery]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalQuery(**row) for row in payload["queries"]]


def to_dict(result: ConfigResult) -> Dict[str, Any]:
    return asdict(result)


__all__ = [
    "EvalQuery",
    "QueryResult",
    "ConfigResult",
    "resolve_anchors",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "hit_rate",
    "percentile",
    "evaluate_config",
    "load_dataset",
    "to_dict",
]
