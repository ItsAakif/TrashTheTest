"""STEP 4 coverage evaluator: source concepts -> represented -> missing.

evaluate_coverage(structure, topic_notes) scores every topic (represented
source concepts / total) plus aggregate topic/concept coverage, names the
missing topics/concepts with reason codes, and returns a pass/fail verdict
against a configured threshold (no page-count metric anywhere).

repair_notes(...) performs threshold repair: for uncovered topics only, it
retrieves the missing evidence, regenerates JUST that section, merges it
back, and revalidates -- at most MAX_REPAIR_PASSES passes, and a pass that
lowers net coverage is reverted and recorded. Observable via RepairRecord
logs (kept on the artifact's repair_log).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from services.academic.types import (
    CoverageReport,
    DocumentStructure,
    ExtractedTopic,
    NotesArtifact,
    PerTopicScore,
    RepairRecord,
    TopicNotes,
)

DEFAULT_COVERAGE_THRESHOLD = 0.7
MAX_REPAIR_PASSES = 2

# Deterministic representation signals for one concept:
#  - its chunk id is cited by the topic notes, or
#  - ≥2 distinctive concept tokens appear in the section text.


def _concept_tokens(title: str) -> set:
    import re

    stop = {"the", "and", "for", "with", "from", "that", "this", "with",
            "are", "was", "were", "have", "has", "which", "their", "into"}
    return {t for t in re.findall(r"[a-z0-9]+", (title or "").lower())
            if len(t) > 3 and t not in stop}


def _topic_concepts(topic: ExtractedTopic) -> List[Dict[str, Any]]:
    """All source concepts of a topic (subtopic concepts + topic-level)."""
    out: List[Dict[str, Any]] = []
    for sub in topic.subtopics or []:
        for concept in sub.concepts or []:
            out.append({"title": concept.title, "chunks": list(concept.chunks)})
    for concept in topic.concepts or []:
        if all(c["title"] != concept.title for c in out):
            out.append({"title": concept.title, "chunks": list(concept.chunks)})
    return out


def evaluate_coverage(
    structure: DocumentStructure,
    topic_notes: List[TopicNotes],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> CoverageReport:
    """Deterministic coverage verdict over the spine (no NLI, no synthesis)."""
    by_topic = {(n.unit_title, n.topic_title): n for n in topic_notes}
    per_topic: List[PerTopicScore] = []
    total_concepts = 0
    represented_concepts = 0
    covered_topics = 0
    all_topics = 0
    uncovered: List[str] = []
    for unit in structure.units:
        for topic in unit.topics:
            all_topics += 1
            concepts = _topic_concepts(topic)
            notes = by_topic.get((unit.title, topic.title))
            if notes is None or notes.no_source:
                per_topic.append(PerTopicScore(
                    unit_title=unit.title,
                    topic_title=topic.title,
                    score=0.0,
                    represented_concepts=0,
                    total_concepts=len(concepts),
                    missing=[c["title"] for c in concepts] or [topic.title],
                    citations_ok=False,
                ))
                uncovered.append(
                    f"{topic.title} (no notes generated)" if notes is None
                    else f"{topic.title} (no_source: no retrievable chunk)"
                )
                total_concepts += len(concepts)
                continue
            cited = {c.chunk_id for c in notes.citations}
            body = " ".join(notes.body_paragraphs).lower()
            missing: List[str] = []
            represented = 0
            for concept in concepts:
                chunks = set(concept["chunks"] or [])
                cited_hit = bool(chunks & cited)
                tokens = _concept_tokens(concept["title"]) | _concept_tokens(topic.title)
                term_hit = sum(1 for t in tokens if t in body) >= 2
                if cited_hit or term_hit or not tokens:
                    represented += 1
                else:
                    missing.append(concept["title"])
            total = len(concepts)
            score = (represented / total) if total else 1.0
            citations_ok = bool(notes.citations)
            if not citations_ok and total:
                missing = [c["title"] for c in concepts]
                represented = 0
                score = 0.0
            if score >= 1.0:
                covered_topics += 1
            else:
                uncovered.append(
                    f"{topic.title} (score {round(score, 2)}: "
                    f"missing {len(missing)}/{total})"
                )
            total_concepts += total
            represented_concepts += represented
            per_topic.append(PerTopicScore(
                unit_title=unit.title,
                topic_title=topic.title,
                score=round(score, 4),
                represented_concepts=represented,
                total_concepts=total,
                missing=missing,
                citations_ok=citations_ok,
            ))
    topic_coverage = round(covered_topics / all_topics, 4) if all_topics else 1.0
    concept_coverage = (
        round(represented_concepts / total_concepts, 4) if total_concepts else 1.0
    )
    passes = topic_coverage >= threshold and concept_coverage >= threshold
    return CoverageReport(
        topic_coverage=topic_coverage,
        concept_coverage=concept_coverage,
        per_topic=per_topic,
        uncovered=uncovered,
        passes=passes,
        threshold=threshold,
        nli="unavailable",
    )


def missing_evidence(
    topic: ExtractedTopic,
    score: PerTopicScore,
    chunk_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Retrieve the chunks backing still-missing concepts (repair evidence)."""
    missing_titles = set(score.missing)
    out: List[Dict[str, Any]] = []
    for sub in topic.subtopics or []:
        for concept in sub.concepts or []:
            if concept.title in missing_titles:
                for cid in concept.chunks or []:
                    if cid in chunk_index and chunk_index[cid] not in out:
                        out.append(chunk_index[cid])
    if not out:
        # Fall back to every topic chunk: the section may have cited none.
        for cid in topic.chunks or []:
            if cid in chunk_index and chunk_index[cid] not in out:
                out.append(chunk_index[cid])
    return out


def repair_notes(
    structure: DocumentStructure,
    topic_notes: List[TopicNotes],
    chunk_index: Dict[str, Dict[str, Any]],
    regen: Callable[[ExtractedTopic, List[Dict[str, Any]]], TopicNotes],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    max_passes: int = MAX_REPAIR_PASSES,
) -> Tuple[List[TopicNotes], CoverageReport, List[RepairRecord]]:
    """Bounded threshold repair over uncovered topics only.

    Each pass regenerates ONLY topics with score < 1.0 (callers assert this
    via the regen call record), merges the result, and re-evaluates. A pass
    that lowers net concept coverage is reverted and recorded (kept=False).
    Stops early when the report passes. At most max_passes passes.
    """
    current = list(topic_notes)
    report = evaluate_coverage(structure, current, threshold)
    log: List[RepairRecord] = []
    topic_by_key = {
        (u.title, t.title): t for u in structure.units for t in u.topics
    }
    for pass_no in range(1, max_passes + 1):
        if report.passes:
            break
        uncovered_keys = [
            (s.unit_title, s.topic_title) for s in report.per_topic
            if s.score < 1.0
        ]
        if not uncovered_keys:
            break
        before = report.concept_coverage
        regenerated: List[str] = []
        candidate = list(current)
        by_key = {(n.unit_title, n.topic_title): i for i, n in enumerate(candidate)}
        for key in uncovered_keys:
            topic = topic_by_key.get(key)
            if topic is None:
                continue
            score = next(
                (s for s in report.per_topic
                 if (s.unit_title, s.topic_title) == key),
                None,
            )
            evidence = missing_evidence(
                topic, score, chunk_index,
            ) if score is not None else []
            fresh = regen(topic, evidence)
            regenerated.append(topic.title)
            if key in by_key:
                candidate[by_key[key]] = fresh
            else:
                candidate.append(fresh)
        new_report = evaluate_coverage(structure, candidate, threshold)
        kept = new_report.concept_coverage >= before
        if kept:
            current, report = candidate, new_report
        log.append(RepairRecord(
            pass_no=pass_no,
            regenerated_topics=regenerated,
            coverage_before=before,
            coverage_after=new_report.concept_coverage,
            kept=kept,
            note=("kept: coverage improved or held" if kept
                  else "reverted: repair lowered net coverage"),
        ))
        if kept and new_report.passes:
            break
    return current, report, log


def attach_report(
    artifact: NotesArtifact,
    report: CoverageReport,
    repair_log: Optional[List[RepairRecord]] = None,
) -> NotesArtifact:
    """Attach the coverage verdict + repair log to the final artifact."""
    data = artifact.model_dump()
    data["coverage"] = report.model_dump()
    data["uncovered"] = list(report.uncovered)
    if repair_log is not None:
        data["repair_log"] = [r.model_dump() for r in repair_log]
    return NotesArtifact.model_validate(data)


__all__ = [
    "DEFAULT_COVERAGE_THRESHOLD",
    "MAX_REPAIR_PASSES",
    "evaluate_coverage",
    "missing_evidence",
    "repair_notes",
    "attach_report",
]
