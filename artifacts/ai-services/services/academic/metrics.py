"""STEP 7 intelligence: deterministic metrics FIRST, LLM explains only.

normalize_inputs() aligns syllabus <-> documents <-> PYQs <-> questions <->
marks <-> evidence into one topic universe (the spine is authoritative for
identity; syllabus titles map onto it by normalized match). compute_metrics()
is pure and deterministic: per-topic frequency, marks share, recency-weighted
score (documented half-life, injectable current_year), unit -> topic
distribution, never-asked topics, missing coverage, and a hotspot ranking
with a documented tie-break (score desc, then spine ord). Two runs over the
same stored rows produce byte-identical MetricsReport JSON.

The LLM narrates the results only: validate_narrative() rejects any number
token absent from the metric payload (normalized numeric comparison),
repair_narrative() retries once, then the response is narrative=None with
narrative_status="withheld" (metrics still returned, HTTP 200 by the route).

Every claim carries provenance: each TopicMetricRow.evidence lists the
chunk ids / PYQ refs / question ids behind its numbers. Frontend tables and
cards render backend-first from topic_table_rows() and summary_cards();
no frontend change is required to consume this step.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from services.academic.flashcards import _normalize
from services.academic.types import (
    MetricsReport,
    MetricsResponse,
    TopicMetricRow,
    UnitMetricRow,
)

DEFAULT_HALF_LIFE = 5.0
DEFAULT_CURRENT_YEAR = 2026

# A number token in narrative prose: integers, decimals, percentages.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def recency_weight(year: int, current_year: int, half_life: float) -> float:
    """Recency weight 0.5 ** ((current_year - year) / half_life).

    year == current_year -> 1.0; year == current_year - half_life -> 0.5.
    Future years clamp to 1.0 (never boost above the present).
    """
    age = max(0, current_year - int(year))
    return round(0.5 ** (age / float(half_life)), 4)


def normalize_inputs(
    structure: Any,
    chunk_rows: List[Dict[str, Any]],
    pyqs: Optional[List[Dict[str, Any]]] = None,
    quiz_questions: Optional[List[Dict[str, Any]]] = None,
    syllabus_units: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Align syllabus <-> docs <-> PYQs <-> questions into one universe.

    Returns {"topics": [{topic_id, topic_title, unit_title, ord,
    chunk_ids, syllabus_titles}], "pyqs": [...], "questions": [...]}.
    Pure and deterministic (spine order throughout).
    """
    topics: List[Dict[str, Any]] = []
    ordinal = 0
    for unit in structure.units:
        for topic in unit.topics:
            ordinal += 1
            chunk_ids = list(topic.chunks or [])
            for sub in topic.subtopics or []:
                for cid in sub.chunks or []:
                    if cid not in chunk_ids:
                        chunk_ids.append(cid)
                for concept in sub.concepts or []:
                    for cid in concept.chunks or []:
                        if cid not in chunk_ids:
                            chunk_ids.append(cid)
            topics.append({
                "topic_id": f"topic-{ordinal:04d}",
                "topic_title": topic.title,
                "unit_title": unit.title,
                "ord": ordinal,
                "chunk_ids": chunk_ids,
                "syllabus_titles": _syllabus_hits(
                    topic.title, syllabus_units or []),
            })
    return {
        "topics": topics,
        "pyqs": [dict(q) for q in (pyqs or [])],
        "questions": [dict(q) for q in (quiz_questions or [])],
        "chunk_rows": [dict(c) for c in (chunk_rows or [])],
    }


def _syllabus_hits(
    topic_title: str,
    syllabus_units: List[Dict[str, Any]],
) -> List[str]:
    """Syllabus unit titles whose topic list names this topic (normalized)."""
    wanted = _normalize(topic_title)
    hits: List[str] = []
    for unit in syllabus_units or []:
        for name in (unit.get("topics") or []):
            if _normalize(str(name)) == wanted:
                title = str(unit.get("title") or unit.get("unit_id") or "")
                if title and title not in hits:
                    hits.append(title)
    return hits


def _topic_match_scores(
    text: str,
    topics: List[Dict[str, Any]],
) -> List[Tuple[int, float]]:
    """Token-overlap scores of a text against every topic (deterministic)."""
    tokens = set(_normalize(text).split())
    scores: List[Tuple[int, float]] = []
    for pos, entry in enumerate(topics):
        vocab = set(_normalize(entry["topic_title"]).split())
        if not tokens or not vocab:
            scores.append((pos, 0.0))
            continue
        scores.append((pos, len(tokens & vocab) / len(tokens)))
    return scores


def compute_metrics(
    normalized: Dict[str, Any],
    current_year: int = DEFAULT_CURRENT_YEAR,
    half_life: float = DEFAULT_HALF_LIFE,
) -> MetricsReport:
    """Deterministic MetricsReport over the normalized universe.

    - frequency: PYQs mapped per topic (best token-overlap >= 0.15, the
      same floor as the engine's map_questions_to_units; below stays
      unmapped and counts toward missing, never force-assigned).
    - marks share: topic marks / total PYQ marks * 100.
    - recurrence: distinct PYQ years per topic.
    - recency_score: sum of recency_weight(year) over the topic's PYQs.
    - hotspot_score: recency_score * 2 + frequency (documented blend);
      ranking tie-break is (score desc, spine ord asc).
    - coverage: syllabus (topic named in a syllabus unit), doc (topic has
      a chunk), PYQ (frequency > 0), questions (a quiz question names it).
    """
    topics = normalized["topics"]
    pyqs = normalized.get("pyqs") or []
    questions = normalized.get("questions") or []
    per_topic: List[TopicMetricRow] = []
    total_marks = sum(max(0, int(q.get("marks", 0) or 0)) for q in pyqs)
    for entry in topics:
        matched: List[Dict[str, Any]] = []
        for qpos, pyq in enumerate(pyqs):
            text = f"{pyq.get('question', '')} {pyq.get('topic') or ''}"
            scores = _topic_match_scores(text, topics)
            best_pos, best = max(scores, key=lambda t: (t[1], -t[0]))
            if topics[best_pos]["topic_id"] == entry["topic_id"] and best >= 0.15:
                matched.append({"pos": qpos, "pyq": pyq, "score": round(best, 4)})
        years = sorted({int(m["pyq"].get("year", current_year) or current_year)
                        for m in matched})
        marks = sum(max(0, int(m["pyq"].get("marks", 0) or 0)) for m in matched)
        recency = round(sum(
            recency_weight(int(m["pyq"].get("year", current_year) or current_year),
                           current_year, half_life)
            for m in matched
        ), 4)
        frequency = len(matched)
        hotspot = round(recency * 2.0 + frequency, 4)
        q_hits = [
            str(q.get("question", ""))[:80]
            for q in questions
            if _normalize(entry["topic_title"]) in _normalize(str(q.get("question", "")))
        ]
        chunk_ids = entry["chunk_ids"]
        item = TopicMetricRow(
            topic_id=entry["topic_id"],
            topic_title=entry["topic_title"],
            unit_title=entry["unit_title"],
            ord=entry["ord"],
            frequency=frequency,
            recurrence=len(years),
            marks_total=marks,
            marks_share=(round(marks / total_marks * 100.0, 2)
                         if total_marks else 0.0),
            recency_score=recency,
            hotspot_score=hotspot,
            hotspot_rank=0,  # assigned after the ranking pass below
            in_syllabus=bool(entry["syllabus_titles"]),
            in_documents=bool(chunk_ids),
            in_pyq=frequency > 0,
            in_questions=bool(q_hits),
            evidence={
                "chunks": list(chunk_ids),
                "pyqs": [f"pyq-{m['pos']}" for m in matched],
                "questions": q_hits[:5],
                "syllabus": list(entry["syllabus_titles"]),
            },
        )
        per_topic.append(item)
    ranked = sorted(per_topic, key=lambda r: (-r.hotspot_score, r.ord))
    for rank, row in enumerate(ranked, start=1):
        row.hotspot_rank = rank
    hotspots = [row.topic_title for row in ranked if row.frequency > 0]
    never_asked = [row.topic_title for row in per_topic
                   if row.frequency == 0 and row.in_syllabus]
    missing = [row.topic_title for row in per_topic
               if not row.in_documents or not row.in_pyq]
    units: List[UnitMetricRow] = []
    for unit_title in sorted({row.unit_title for row in per_topic}):
        members = [r for r in per_topic if r.unit_title == unit_title]
        units.append(UnitMetricRow(
            unit_title=unit_title,
            topic_count=len(members),
            frequency=sum(r.frequency for r in members),
            marks_total=sum(r.marks_total for r in members),
            topics=[r.topic_title for r in members],
        ))
    n_topics = len(per_topic)
    totals = {
        "topics": n_topics,
        "pyq_questions": len(pyqs),
        "total_marks": total_marks,
        "syllabus_coverage": (round(
            sum(1 for r in per_topic if r.in_syllabus) / n_topics, 4)
            if n_topics else 1.0),
        "doc_coverage": (round(
            sum(1 for r in per_topic if r.in_documents) / n_topics, 4)
            if n_topics else 1.0),
        "pyq_coverage": (round(
            sum(1 for r in per_topic if r.in_pyq) / n_topics, 4)
            if n_topics else 1.0),
        "question_coverage": (round(
            sum(1 for r in per_topic if r.in_questions) / n_topics, 4)
            if n_topics else 1.0),
        "missing_count": len(missing),
        "never_asked_count": len(never_asked),
    }
    return MetricsReport(
        topics=sorted(per_topic, key=lambda r: r.ord),
        units=units,
        never_asked=never_asked,
        missing=missing,
        hotspots=hotspots,
        totals=totals,
        half_life=float(half_life),
        current_year=int(current_year),
        provenance_note=(
            "deterministic metrics over stored rows: spine topics, "
            "document chunks, extracted PYQs and quiz questions; "
            "no LLM number originates here"
        ),
    )


def metrics_numbers(report: MetricsReport) -> set:
    """Every numeric claim in the payload, normalized for comparison."""
    payload = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    found = set(_NUMBER_RE.findall(payload))
    normalized: set = set()
    for token in found:
        bare = token.rstrip("%")
        try:
            value = float(bare)
        except ValueError:
            continue
        normalized.add(f"{value:g}")
        normalized.add(f"{value:g}%")
        normalized.add(bare)
        normalized.add(token)
    years = {str(report.current_year)}
    return normalized | years


def validate_narrative(narrative: str, report: MetricsReport) -> Dict[str, Any]:
    """Ground a narrative: every number must occur in the metric payload."""
    allowed = metrics_numbers(report)
    fabricated: List[str] = []
    for token in _NUMBER_RE.findall(narrative or ""):
        bare = token.rstrip("%")
        try:
            value = float(bare)
        except ValueError:
            continue
        candidates = {f"{value:g}", f"{value:g}%", bare, token}
        if not (candidates & allowed):
            fabricated.append(token)
    return {"ok": not fabricated, "fabricated": fabricated}


def repair_narrative(
    report: MetricsReport,
    generate: Any,
    max_repairs: int = 1,
) -> MetricsResponse:
    """One bounded narrative repair; then withhold (metrics still returned).

    generate(prompt) -> narrative text. After max_repairs failures the
    response is narrative=None with narrative_status="withheld".
    """
    from services.academic.metrics_narrative import narrative_prompt

    prompt = narrative_prompt(report)
    attempts = 0
    while True:
        text = generate(prompt) if callable(generate) else str(generate)
        verdict = validate_narrative(text, report)
        if verdict["ok"]:
            return MetricsResponse(
                metrics=report, narrative=text,
                narrative_status=("ok" if attempts == 0 else "repaired"),
            )
        attempts += 1
        if attempts > max_repairs:
            return MetricsResponse(
                metrics=report, narrative=None,
                narrative_status="withheld",
            )


def topic_table_rows(report: MetricsReport) -> List[Dict[str, Any]]:
    """Backend-first table rows: one flat row per topic (provenance inline)."""
    return [{
        "topic": row.topic_title,
        "unit": row.unit_title,
        "frequency": row.frequency,
        "recurrence": row.recurrence,
        "marks_total": row.marks_total,
        "marks_share": row.marks_share,
        "recency_score": row.recency_score,
        "hotspot_rank": row.hotspot_rank,
        "coverage": {
            "syllabus": row.in_syllabus,
            "documents": row.in_documents,
            "pyq": row.in_pyq,
            "questions": row.in_questions,
        },
        "evidence": dict(row.evidence),
    } for row in report.topics]


def summary_cards(report: MetricsReport) -> List[Dict[str, Any]]:
    """Backend-first summary cards (totals + hotspots + gaps)."""
    totals = dict(report.totals)
    return [
        {"card": "coverage",
         "syllabus_coverage": totals.get("syllabus_coverage"),
         "doc_coverage": totals.get("doc_coverage"),
         "pyq_coverage": totals.get("pyq_coverage"),
         "question_coverage": totals.get("question_coverage")},
        {"card": "hotspots", "topics": list(report.hotspots[:5])},
        {"card": "never_asked", "topics": list(report.never_asked),
         "count": totals.get("never_asked_count")},
        {"card": "missing", "topics": list(report.missing),
         "count": totals.get("missing_count")},
        {"card": "marks",
         "total_marks": totals.get("total_marks"),
         "pyq_questions": totals.get("pyq_questions")},
    ]


def metrics_artifact(
    report: MetricsReport,
    response: Optional[MetricsResponse] = None,
) -> Dict[str, Any]:
    """Artifact payload: metrics + narrative verdict + table/card views."""
    return {
        "metrics": report.model_dump(mode="json"),
        "narrative": response.narrative if response else None,
        "narrative_status": response.narrative_status if response else "not_requested",
        "tables": topic_table_rows(report),
        "cards": summary_cards(report),
        "provenance_note": report.provenance_note,
    }


__all__ = [
    "DEFAULT_HALF_LIFE",
    "DEFAULT_CURRENT_YEAR",
    "recency_weight",
    "normalize_inputs",
    "compute_metrics",
    "metrics_numbers",
    "validate_narrative",
    "repair_narrative",
    "topic_table_rows",
    "summary_cards",
    "metrics_artifact",
]
