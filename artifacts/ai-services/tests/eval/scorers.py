"""Deterministic offline scorers shared by the eval tests and the report.

Modes mirror the production arms without any model weights:
- vector_proxy: Jaccard overlap on content tokens (stands in for the
  dense arm on this tiny fixture, where paraphrases keep keywords).
- lexical: rare-term match rate (stands in for the tsvector arm; rare =
  appears in <= 2 chunks, like an exact scientific term).
- fused: RRF over the two rank lists (production rrf_fuse).
- reranked: heading-weighted coverage reorder of the fused top list
  (stands in for the cross-encoder; deterministic and offline).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the and or of in on to for with is are was were be by as at from "
    "that this it its into when how what which who not no does do did than "
    "then there their they them he she his her we you our your".split()
)


def tokens(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP]


def build_docfreq(corpus: List[Dict]) -> Counter:
    df: Counter = Counter()
    for chunk in corpus:
        df.update(set(tokens(chunk["text"] + " " + chunk.get("heading", ""))))
    return df


def vector_proxy_score(query: str, chunk: Dict) -> float:
    q, c = set(tokens(query)), set(tokens(chunk["text"]))
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)


def lexical_score(query: str, chunk: Dict, docfreq: Counter) -> float:
    q = [t for t in tokens(query) if docfreq.get(t, 99) <= 2]
    if not q:
        q = tokens(query)
    if not q:
        return 0.0
    c = set(tokens(chunk["text"]))
    return sum(1 for t in q if t in c) / len(q)


def rerank_score(query: str, chunk: Dict) -> float:
    """Heading-weighted coverage: query tokens in heading count double."""
    q = set(tokens(query))
    if not q:
        return 0.0
    body = set(tokens(chunk["text"]))
    head = set(tokens(chunk.get("heading", "")))
    return (len(q & body) + 2.0 * len(q & head)) / (len(q) * 2.0)


def rank(corpus: List[Dict], query: str, scorer) -> List[str]:
    scored = sorted(
        ((c["id"], scorer(query, c)) for c in corpus),
        key=lambda kv: (-kv[1], kv[0]),
    )
    return [cid for cid, s in scored if s > 0]


def recall_at(relevant: set, ranking: List[str], k: int = 10) -> float:
    if not relevant:
        return 1.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def ndcg_at(relevant: set, ranking: List[str], k: int = 10) -> float:
    import math

    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ranking[:k]) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 1.0


def mrr(relevant: set, ranking: List[str]) -> float:
    for i, cid in enumerate(ranking):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0
