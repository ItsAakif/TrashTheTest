"""Topic source sufficiency and bounded evidence recovery.

The notes pipeline must not confuse "the retriever returned no rows" with
"the uploaded source has no material for this topic".  This module keeps
that decision deterministic and explicit.  It is a thin policy layer over
the existing chunk/retrieval contracts: it never creates citations and never
invents source text.

Recovery is deliberately additive.  A topic starts with the structure's
claimed chunks, then tries broader lexical/semantic evidence, unit/parent
chunks, aliases, and nearby pages.  The final list is still source ordered;
ranking is used only to decide which evidence to retain, never to reorder a
document.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "how", "in", "is", "it", "of", "on", "or", "that",
        "the", "their", "this", "to", "was", "were", "what", "when", "where",
        "which", "with", "you", "your",
    }
)


class SufficiencyStatus(str, Enum):
    """Honest source states used by notes and tutor adapters."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    MISSING = "MISSING"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class SufficiencyResult:
    """Result of checking whether evidence can support a topic."""

    status: SufficiencyStatus
    reason: str
    evidence: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    recovery_steps: Tuple[str, ...] = field(default_factory=tuple)
    scores: Dict[str, float] = field(default_factory=dict)

    @property
    def sufficient(self) -> bool:
        return self.status is SufficiencyStatus.SUFFICIENT

    @property
    def supported(self) -> bool:
        return self.status in {
            SufficiencyStatus.SUFFICIENT,
            SufficiencyStatus.AMBIGUOUS,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "evidence": [dict(row) for row in self.evidence],
            "recovery_steps": list(self.recovery_steps),
            "scores": dict(self.scores),
        }


def _get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def _row_id(row: Any) -> str:
    return str(_get(row, "id", _get(row, "chunk_id", "")) or "")


def _row_text(row: Any) -> str:
    return str(_get(row, "text", _get(row, "content", "")) or "")


def _row_heading(row: Any) -> str:
    return str(_get(row, "heading", _get(row, "section", "")) or "").strip().lower()


def _row_page(row: Any) -> Optional[int]:
    value = _get(row, "page", _get(row, "page_start", None))
    if value is None:
        metadata = _get(row, "metadata", None) or _get(row, "chunk_metadata", None) or {}
        value = metadata.get("page", metadata.get("page_start")) if isinstance(metadata, Mapping) else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _row_position(row: Any) -> int:
    value = _get(row, "position", 0)
    metadata = _get(row, "metadata", None) or {}
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get("position", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _tokens(text: str) -> Set[str]:
    return {
        token for token in _TOKEN_RE.findall((text or "").lower())
        if token not in _STOPWORDS
    }


def _topic_text(topic: Any, unit_title: str = "", aliases: Optional[Sequence[str]] = None) -> str:
    title = str(_get(topic, "title", topic) or "")
    bits = [unit_title, title]
    bits.extend(str(value) for value in (aliases or _get(topic, "aliases", []) or []) if value)
    for sub in _get(topic, "subtopics", []) or []:
        bits.append(str(_get(sub, "title", sub) or ""))
    return " -- ".join(bit for bit in bits if bit)


def _source_order(row: Any) -> Tuple[int, int, int, str]:
    page = _row_page(row)
    return (
        0 if page is not None else 1,
        page if page is not None else 0,
        _row_position(row),
        _row_id(row),
    )


def _relevance(row: Any, vocabulary: Set[str], names: Set[str]) -> float:
    heading = _row_heading(row)
    if heading and any(heading == name or heading in name or name in heading for name in names if name):
        return 1.0
    tokens = _tokens(_row_text(row))
    if not tokens or not vocabulary:
        return 0.0
    return len(tokens & vocabulary) / float(len(vocabulary))


def _maybe_await(value: Any) -> Any:
    """Accept sync or async rerank callbacks without a new dependency."""

    if inspect.isawaitable(value):
        # A synchronous policy helper cannot await.  The caller can pass an
        # already-materialised score map; do not silently run an event loop.
        return None
    return value


def recover_topic_evidence(
    all_chunks: Sequence[Any],
    topic: Any,
    unit_title: str = "",
    unit_chunk_ids: Optional[Sequence[str]] = None,
    aliases: Optional[Sequence[str]] = None,
    nearby_pages: int = 2,
    semantic_scores: Optional[Mapping[str, float]] = None,
    rerank: Optional[Callable[..., Any]] = None,
    min_overlap: float = 0.20,
) -> List[Dict[str, Any]]:
    """Recover all useful topic evidence, in source order.

    ``semantic_scores`` is a provider/retriever supplied id -> score map;
    ``rerank`` may be a small callback accepting ``(query, rows)`` or
    ``(query, rows, scores)``.  No callback is required for offline use.
    """

    rows = [dict(row) if isinstance(row, Mapping) else {
        "id": _row_id(row), "chunk_id": _row_id(row), "text": _row_text(row),
        "content": _row_text(row), "heading": _row_heading(row),
        "page": _row_page(row), "position": _row_position(row),
    } for row in (all_chunks or [])]
    by_id = {_row_id(row): row for row in rows if _row_id(row)}
    names = {
        str(_get(topic, "title", topic) or "").strip().lower(),
        *[str(value or "").strip().lower() for value in (aliases or _get(topic, "aliases", []) or [])],
    }
    names.discard("")
    vocabulary = _tokens(_topic_text(topic, unit_title, aliases))
    scores: Dict[str, float] = {}
    for row in rows:
        rid = _row_id(row)
        score = _relevance(row, vocabulary, names)
        if semantic_scores and rid in semantic_scores:
            try:
                score = max(score, float(semantic_scores[rid]))
            except (TypeError, ValueError):
                pass
        scores[rid] = round(float(score), 6)

    selected: Set[str] = {
        str(value) for value in (_get(topic, "chunks", []) or []) if str(value) in by_id
    }
    selected.update(str(value) for value in (unit_chunk_ids or []) if str(value) in by_id)

    # Broader lexical/alias/semantic recovery.  The threshold is deliberately
    # lower than the strict selector: recovery runs before unsupported status.
    for row in rows:
        rid = _row_id(row)
        if rid in selected:
            continue
        if scores.get(rid, 0.0) >= min_overlap:
            selected.add(rid)

    # Nearby pages are useful when a heading/chunk boundary split a concept.
    if nearby_pages > 0 and selected:
        seed_pages = [p for p in (_row_page(by_id[rid]) for rid in selected) if p is not None]
        if seed_pages:
            lo, hi = min(seed_pages) - int(nearby_pages), max(seed_pages) + int(nearby_pages)
            for row in rows:
                page = _row_page(row)
                if page is not None and lo <= page <= hi:
                    selected.add(_row_id(row))

    if rerank is not None:
        candidate_rows = [by_id[rid] for rid in selected if rid in by_id]
        try:
            ranked = rerank(_topic_text(topic, unit_title, aliases), candidate_rows)
            if ranked is None:
                try:
                    ranked = rerank(_topic_text(topic, unit_title, aliases), candidate_rows, scores)
                except TypeError:
                    ranked = None
            ranked = _maybe_await(ranked)
            if ranked:
                ranked_ids: List[str] = []
                for value in ranked:
                    if isinstance(value, Mapping):
                        rid = str(value.get("id") or value.get("chunk_id") or "")
                    else:
                        rid = str(value)
                    if rid in by_id and rid not in ranked_ids:
                        ranked_ids.append(rid)
                if ranked_ids:
                    # Keep the callback's positive set, then append any
                    # source-claimed rows that the callback omitted.
                    selected = set(ranked_ids) | {
                        str(value) for value in (_get(topic, "chunks", []) or [])
                        if str(value) in by_id
                    }
        except Exception:
            # Reranking is an enhancement; lexical recovery remains usable.
            pass

    return sorted((by_id[rid] for rid in selected if rid in by_id), key=_source_order)


def assess_source_sufficiency(
    topic: Any,
    evidence: Sequence[Any],
    all_chunks: Optional[Sequence[Any]] = None,
    unit_title: str = "",
    unit_chunk_ids: Optional[Sequence[str]] = None,
    aliases: Optional[Sequence[str]] = None,
    semantic_scores: Optional[Mapping[str, float]] = None,
    ambiguous: bool = False,
    rerank: Optional[Callable[..., Any]] = None,
) -> SufficiencyResult:
    """Classify topic evidence as sufficient/insufficient/missing/ambiguous.

    Retrieval is an optimisation, not a source-scope decision.  In
    particular, a non-empty but irrelevant result must not short-circuit
    the bounded recovery pass: a relevant row may exist in ``all_chunks``.
    Recovery is merged with the direct rows and the merged set is source
    sorted before any status is returned.
    """

    corpus = list(all_chunks or ())
    rows = [dict(row) if isinstance(row, Mapping) else {
        "id": _row_id(row), "text": _row_text(row), "heading": _row_heading(row),
    } for row in (evidence or [])]
    rows = [row for row in rows if _row_text(row).strip() or _row_heading(row)]

    vocab = _tokens(_topic_text(topic, unit_title, aliases))
    names = {
        str(_get(topic, "title", topic) or "").strip().lower(),
        *[str(value or "").strip().lower() for value in (aliases or _get(topic, "aliases", []) or [])],
    }

    # A direct hit avoids unnecessary recovery.  Everything else, including
    # an irrelevant non-empty retrieval result, must go through recovery.
    direct = any(_relevance(row, vocab, names) >= 0.20 for row in rows)
    if not direct:
        recovered = recover_topic_evidence(
            corpus, topic, unit_title=unit_title,
            unit_chunk_ids=unit_chunk_ids, aliases=aliases,
            semantic_scores=semantic_scores, rerank=rerank,
        )
        merged: Dict[str, Dict[str, Any]] = {}
        for row in [*rows, *recovered]:
            rid = _row_id(row)
            # Rows without an id are still retained; use a stable synthetic
            # key rather than dropping source evidence.
            key = rid or f"__row_{len(merged)}"
            merged.setdefault(key, row)
        rows = list(merged.values())

    rows.sort(key=_source_order)
    if not rows:
        status = SufficiencyStatus.MISSING if not corpus else SufficiencyStatus.INSUFFICIENT
        return SufficiencyResult(
            status=status,
            reason=(
                "no extractable chunks in the selected source scope"
                if status is SufficiencyStatus.MISSING
                else "no topic-matching chunks after lexical, alias, parent, and nearby-page recovery"
            ),
            recovery_steps=("topic", "aliases", "unit", "nearby_pages", "semantic", "rerank"),
        )
    if ambiguous:
        return SufficiencyResult(
            status=SufficiencyStatus.AMBIGUOUS,
            reason="multiple conflicting source matches require a narrower topic/alias",
            evidence=tuple(rows),
            recovery_steps=("alias", "parent", "rerank"),
            scores={_row_id(row): 1.0 for row in rows if _row_id(row)},
        )

    if not any(_relevance(row, vocab, names) >= 0.20 for row in rows):
        return SufficiencyResult(
            status=SufficiencyStatus.INSUFFICIENT,
            reason="recovered chunks do not match the topic vocabulary",
            evidence=tuple(rows),
            recovery_steps=("broader_topic", "aliases", "parent", "nearby_pages", "rerank"),
        )
    return SufficiencyResult(
        status=SufficiencyStatus.SUFFICIENT,
        reason="topic evidence exists after source-scoped recovery",
        evidence=tuple(rows),
        recovery_steps=("topic", "aliases", "unit", "nearby_pages", "lexical", "semantic", "rerank"),
        scores={_row_id(row): _relevance(row, vocab, names) for row in rows if _row_id(row)},
    )


# Friendly aliases used by callers/tests.
classify_source_sufficiency = assess_source_sufficiency
recover_topic_chunks = recover_topic_evidence

__all__ = [
    "SufficiencyStatus",
    "SufficiencyResult",
    "recover_topic_evidence",
    "recover_topic_chunks",
    "assess_source_sufficiency",
    "classify_source_sufficiency",
]
