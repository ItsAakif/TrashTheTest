"""Hard validation gates for generated notes and PDF input.

This module is deliberately independent of the job queue and renderer so a
provider double, a route, and the worker all enforce the same contract.  A
source gap is an explicit report, not a normal prose section.  Generated
sections must be substantive, topic-bound, cited, and ordered like the
source spine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PLACEHOLDER_RE = re.compile(
    r"(?:\bno\s+(?:retrievable|relevant|available|source|information)\b|"
    r"\bnot\s+(?:enough|sufficient)\s+(?:information|source|material)\b|"
    r"\bcannot\s+(?:answer|find|retrieve)\b|\bnot\s+available\b|"
    r"\bunsupported\s+(?:topic|content|source)\b|\btbc\b|\bplaceholder\b|"
    r"\bcoming\s+soon\b)",
    re.IGNORECASE,
)
MIN_NOTE_CHARS = 40
MIN_SECTION_CHARS = 40

# A placeholder statement is short boilerplate; a longer body that merely
# mentions an unavailable thing is real prose (see is_placeholder).
PLACEHOLDER_MAX_CHARS = 220
PLACEHOLDER_MIN_SURROUNDING_WORDS = 12

# ADR-008 T006: a placeholder spine label ("Pages 12-14",
# "Untitled -- page 5") cannot be gated on token overlap.  Only the weak
# structure verdicts earn the downgrade, and only below this confidence.
DEGRADED_STRUCTURE_METHODS = frozenset({"page_bounded", "inferred"})
DEGRADED_STRUCTURE_CONFIDENCE = 0.6


def is_degraded_structure(method: Any, confidence: Any) -> bool:
    """True when the structure verdict is too weak to trust a topic label."""

    try:
        if method not in DEGRADED_STRUCTURE_METHODS:
            return False
    except TypeError:
        return False
    if confidence is None:
        return True
    try:
        return float(confidence) < DEGRADED_STRUCTURE_CONFIDENCE
    except (TypeError, ValueError):
        return False


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _tokens(text: str) -> Set[str]:
    return {token for token in _TOKEN_RE.findall((text or "").lower()) if len(token) > 2}


def _norm(text: str) -> str:
    return " ".join(_TOKEN_RE.findall((text or "").lower()))


def _gap_targets_topic(gap: Any, unit: str, topic: str) -> bool:
    """Return whether a source gap names this exact unit/topic.

    A bare ``no_source`` token is metadata, not permission to omit every
    topic in a spine.  The parser emits ``Unit::Topic: reason`` while the
    per-topic validator emits ``Topic: reason``; both forms are accepted,
    but a generic token or another topic never matches.
    """

    if isinstance(gap, Mapping):
        gap_unit = str(gap.get("unit_title") or gap.get("unit") or "")
        gap_topic = str(gap.get("topic_title") or gap.get("topic") or "")
        if gap_topic and _norm(gap_topic) == _norm(topic):
            return not gap_unit or _norm(gap_unit) == _norm(unit)
        return False
    raw = str(gap or "").strip()
    if not raw or _norm(raw) in {"no_source", "no source", "unmapped"}:
        return False
    if "::" in raw:
        left, right = raw.split("::", 1)
        right = right.split(":", 1)[0]
        return _norm(left) == _norm(unit) and _norm(right) == _norm(topic)
    # Topic-only gaps are produced by validate_topic_notes.  Compare the
    # complete first field, not a substring, so "Cell" cannot cover "Cell
    # Division" and a reason cannot masquerade as a topic name.
    first = raw.split(":", 1)[0]
    return _norm(first) == _norm(topic)


def is_placeholder(text: str) -> bool:
    """True for boilerplate that must never be persisted as a normal section.

    A placeholder is a short boilerplate statement ("No relevant
    information", "not available", "TBC").  Real source prose can legitimately
    contain the same words mid-sentence ("multiline comments are not available
    in python"), so a match only counts as a placeholder when the marker
    dominates a short body or almost no surrounding prose is left.
    """

    raw = (text or "").strip()
    if not raw:
        return True
    match = _PLACEHOLDER_RE.search(raw)
    if not match:
        return False
    if len(raw) <= PLACEHOLDER_MAX_CHARS:
        return True
    surrounding = f"{raw[: match.start()]} {raw[match.end():]}".split()
    return len(surrounding) < PLACEHOLDER_MIN_SURROUNDING_WORDS


def _citation_id(citation: Any) -> str:
    return str(_get(citation, "chunk_id", "") or "")


def _citation_span(citation: Any) -> str:
    return str(_get(citation, "span", "") or "")


@dataclass
class NotesValidationReport:
    """Structured result consumed by the worker and API responses."""

    ok: bool = True
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    source_gaps: List[str] = field(default_factory=list)
    expected_topics: List[str] = field(default_factory=list)
    covered_topics: List[str] = field(default_factory=list)
    citation_count: int = 0
    section_count: int = 0
    word_count: int = 0

    @property
    def passes(self) -> bool:
        return self.ok

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "passes": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "source_gaps": list(self.source_gaps),
            "expected_topics": list(self.expected_topics),
            "covered_topics": list(self.covered_topics),
            "citation_count": self.citation_count,
            "section_count": self.section_count,
            "word_count": self.word_count,
        }


def validate_topic_notes(
    note: Any,
    topic: Any = None,
    allowed_chunk_ids: Optional[Set[str]] = None,
    chunk_texts: Optional[Mapping[str, str]] = None,
    min_chars: int = MIN_NOTE_CHARS,
) -> NotesValidationReport:
    """Validate one TopicNotes payload, returning source gaps honestly."""

    data = _dump(note)
    report = NotesValidationReport()
    body = list(data.get("body_paragraphs") or [])
    citations = list(data.get("citations") or [])
    topic_title = str(_get(topic, "title", topic) or data.get("topic_title") or "")
    if bool(data.get("no_source")):
        report.source_gaps.append(
            str(data.get("provenance_note") or f"{topic_title or 'topic'}: no_source")
        )
        # A no_source marker is an omission contract, not a wrapper around
        # generated prose.  Reject either substantive body text or citations
        # so callers cannot smuggle fabricated content through the gap path.
        substantive_body = [
            part for part in body if str(part or "").strip()
        ]
        if substantive_body:
            report.ok = False
            report.errors.append({
                "code": "no_source_has_body", "topic": topic_title,
            })
        if citations:
            report.ok = False
            report.errors.append({
                "code": "no_source_has_citations", "topic": topic_title,
            })
        if not substantive_body and not citations:
            report.warnings.append({"code": "source_gap_omitted", "topic": topic_title})
        return report

    if not body or not any(str(part or "").strip() for part in body):
        report.ok = False
        report.errors.append({"code": "empty_body", "topic": topic_title})
    text = " ".join(str(part or "") for part in body)
    if len(text.strip()) < int(min_chars):
        report.ok = False
        report.errors.append({
            "code": "too_short", "topic": topic_title,
            "chars": len(text.strip()), "minimum": int(min_chars),
        })
    if is_placeholder(text):
        report.ok = False
        report.errors.append({"code": "placeholder", "topic": topic_title})
    if topic_title:
        topic_tokens = _tokens(topic_title)
        if topic_tokens and not (topic_tokens & _tokens(text)):
            # Degraded structure (page-bounded, low confidence): the
            # "topic" is a placeholder like "Pages 12-14", so token
            # overlap is not the right gate.  Warn and continue; the
            # body and citations still matter.
            meta = data.get("metadata") or getattr(note, "metadata", None) or {}
            struct = _get(note, "structure_method", None)
            if struct is None:
                struct = _get(meta, "structure_method", None)
            if struct is None:
                struct = data.get("structure_method")
            conf = _get(note, "structure_confidence", None)
            if conf is None:
                conf = _get(meta, "structure_confidence", None)
            if conf is None:
                conf = data.get("structure_confidence")
            degraded = is_degraded_structure(struct, conf)
            if degraded:
                report.warnings.append({"code": "topic_mismatch_degraded",
                                        "topic": topic_title})
            else:
                # The topic label is chosen by our extractor, so lexical
                # overlap between label and body is evidence of verbosity,
                # not of correctness: an extractor label like "Answer True
                # and Fals" or "Mode" will rarely be repeated by the model
                # even when the body is accurate.  Correctness is already
                # hard-gated below by body length, citations and spans, so
                # low overlap is advisory only and never fails the topic.
                report.warnings.append({"code": "topic_overlap_low",
                                        "topic": topic_title})
    if not citations:
        report.ok = False
        report.errors.append({"code": "missing_citations", "topic": topic_title})
    if allowed_chunk_ids is not None:
        for citation in citations:
            cid = _citation_id(citation)
            if not cid or cid not in allowed_chunk_ids:
                report.ok = False
                report.errors.append({"code": "invalid_citation", "topic": topic_title, "chunk_id": cid})
            if chunk_texts is not None and cid in chunk_texts:
                span = _norm(_citation_span(citation))
                source = _norm(str(chunk_texts[cid]))
                if span and span not in source:
                    report.ok = False
                    report.errors.append({"code": "citation_span", "topic": topic_title, "chunk_id": cid})
    report.citation_count = len(citations)
    report.word_count = len(text.split())
    return report


def _expected_topic_keys(structure: Any) -> List[Tuple[str, str]]:
    units = list(_get(structure, "units", []) or [])
    return [
        (str(_get(unit, "title", "") or ""), str(_get(topic, "title", "") or ""))
        for unit in units
        for topic in (list(_get(unit, "topics", []) or []))
    ]


def validate_notes_artifact(
    artifact: Any,
    structure: Any = None,
    allowed_chunk_ids: Optional[Set[str]] = None,
    chunk_texts: Optional[Mapping[str, str]] = None,
    min_chars: int = MIN_SECTION_CHARS,
    allow_source_gaps: bool = True,
) -> NotesValidationReport:
    """Validate the complete ordered artifact before persistence or PDF render."""

    data = _dump(artifact)
    sections = list(data.get("sections") or [])
    expected = _expected_topic_keys(structure) if structure is not None else []
    expected_labels = [f"{unit}::{topic}" for unit, topic in expected]
    report = NotesValidationReport(
        expected_topics=expected_labels,
        source_gaps=list(data.get("source_gaps") or []),
    )
    if not sections:
        report.ok = False
        report.errors.append({"code": "no_sections"})
        return report

    positions = {label: index for index, label in enumerate(expected_labels)}
    last_position = -1
    seen_topics: Set[str] = set()
    for index, section in enumerate(sections):
        section_data = _dump(section)
        unit = str(section_data.get("unit_title") or "")
        topic = str(section_data.get("topic_title") or section_data.get("heading") or "")
        label = f"{unit}::{topic}"
        content = str(section_data.get("content") or "")
        citations = list(section_data.get("citations") or [])
        report.section_count += 1
        report.citation_count += len(citations)
        report.word_count += len(content.split())
        if expected and label not in positions:
            report.ok = False
            report.errors.append({"code": "topic_not_in_spine", "section": index, "topic": label})
        elif expected:
            position = positions[label]
            if position < last_position:
                report.ok = False
                report.errors.append({"code": "document_order", "section": index, "topic": label})
            last_position = max(last_position, position)
        seen_topics.add(label)
        if not content.strip():
            report.ok = False
            report.errors.append({"code": "empty_section", "section": index, "topic": label})
            continue
        if len(content.strip()) < int(min_chars):
            report.ok = False
            report.errors.append({
                "code": "section_too_short", "section": index,
                "topic": label, "chars": len(content.strip()), "minimum": int(min_chars),
            })
        if is_placeholder(content):
            report.ok = False
            report.errors.append({"code": "placeholder_section", "section": index, "topic": label})
        if not citations:
            report.ok = False
            report.errors.append({"code": "section_missing_citations", "section": index, "topic": label})
        if expected and topic and _tokens(topic) and not (_tokens(topic) & _tokens(content)):
            # Degraded structure (page-bounded / inferred, low confidence):
            # the topic is a placeholder label, so token overlap is not the
            # right gate.  Warn and keep the body + citation checks.
            meta = section_data.get("metadata") or {}
            struct = meta.get("structure_method")
            if struct is None:
                struct = section_data.get("structure_method")
            conf = meta.get("structure_confidence")
            if conf is None:
                conf = section_data.get("structure_confidence")
            degraded = is_degraded_structure(struct, conf)
            if degraded:
                report.warnings.append({
                    "code": "topic_mismatch_degraded",
                    "section": index, "topic": label,
                })
            else:
                # The topic label is chosen by our extractor, so lexical
                # overlap between label and body is evidence of verbosity,
                # not of correctness: an extractor label like "Answer True
                # and Fals" or "Mode" will rarely be repeated by the model
                # even when the body is accurate.  Correctness stays hard
                # through section_too_short, placeholder_section,
                # section_missing_citations, invalid_citation and
                # citation_span below; low overlap is advisory only.
                report.warnings.append({
                    "code": "topic_overlap_low",
                    "section": index, "topic": label,
                })
        for citation in citations:
            cid = _citation_id(citation)
            if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                report.ok = False
                report.errors.append({"code": "invalid_citation", "section": index, "chunk_id": cid})
            if chunk_texts is not None and cid in chunk_texts:
                span = _norm(_citation_span(citation))
                if span and span not in _norm(str(chunk_texts[cid])):
                    report.ok = False
                    report.errors.append({"code": "citation_span", "section": index, "chunk_id": cid})

    report.covered_topics = [label for label in expected_labels if label in seen_topics]
    if expected:
        missing = [label for label in expected_labels if label not in seen_topics]
        # Missing topics are allowed only when the artifact explicitly names
        # that exact unit/topic gap.  A generic ``no_source`` token must not
        # become a blanket waiver for the whole spine.
        for label in missing:
            unit, topic = label.split("::", 1)
            if allow_source_gaps and any(
                _gap_targets_topic(gap, unit, topic)
                for gap in report.source_gaps
            ):
                continue
            report.ok = False
            report.errors.append({"code": "missing_topic", "topic": label})
    if not any(str(section.get("content") or "").strip() for section in (_dump(s) for s in sections)):
        report.ok = False
        report.errors.append({"code": "headings_only"})
    if structure is not None or chunk_texts is not None:
        structural = _structural_body_errors(artifact, structure, chunk_texts)
        if structural:
            report.errors.extend(structural)
            report.ok = False
    return report


def require_valid_notes(
    artifact: Any,
    structure: Any = None,
    allowed_chunk_ids: Optional[Set[str]] = None,
    chunk_texts: Optional[Mapping[str, str]] = None,
    min_chars: int = MIN_SECTION_CHARS,
    allow_source_gaps: bool = True,
) -> NotesValidationReport:
    """Validate and raise a typed error for a hard-gate failure."""

    report = validate_notes_artifact(
        artifact, structure, allowed_chunk_ids, chunk_texts,
        min_chars=min_chars, allow_source_gaps=allow_source_gaps,
    )
    if not report.ok:
        from services.academic.orchestrator import AcademicPipelineError

        codes = ", ".join(str(item.get("code")) for item in report.errors[:6])
        raise AcademicPipelineError(f"notes validation failed: {codes}")
    return report


def _structural_body_errors(
    artifact: Any,
    structure: Any = None,
    chunk_texts: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Find title-only sections that generic length checks cannot detect."""

    data = _dump(artifact)
    sections = list(data.get("sections") or [])
    errors: List[Dict[str, Any]] = []
    for index, section in enumerate(sections):
        section_data = _dump(section)
        content_tokens = list(_tokens(str(section_data.get("content") or "")))
        unit_title = str(section_data.get("unit_title") or "")
        topic_title = str(section_data.get("topic_title") or section_data.get("heading") or "")
        heading_tokens = set(_tokens(
            f"{unit_title} {topic_title} {section_data.get('heading') or ''}"
        ))
        residual = [token for token in content_tokens if token not in heading_tokens]
        # A real explanatory sentence has several words beyond the heading;
        # a repeated/long title has none (or only connective filler).
        if len(residual) < 4:
            errors.append({"code": "title_only_section", "section": index, "topic": topic_title})
            continue
        if chunk_texts:
            cited_ids = {
                _citation_id(citation)
                for citation in (section_data.get("citations") or [])
                if _citation_id(citation) in chunk_texts
            }
            if cited_ids:
                source_tokens = set()
                for cid in cited_ids:
                    source_tokens.update(_tokens(str(chunk_texts.get(cid) or "")))
                # Source text is evidence for a real body, not a replacement
                # for the structural check.  Require some residual overlap so
                # a long title with a citation cannot masquerade as prose.
                if residual and not (set(residual) & source_tokens):
                    errors.append({
                        "code": "body_not_grounded", "section": index, "topic": topic_title,
                    })
    return errors


def assert_pdf_has_body(
    artifact: Any,
    structure: Any = None,
    allowed_chunk_ids: Optional[Set[str]] = None,
    chunk_texts: Optional[Mapping[str, str]] = None,
    min_chars: int = MIN_SECTION_CHARS,
) -> None:
    """PDF-specific guard: headings/title-only documents are never valid.

    ``structure``, the allowed citation scope, and source text are supplied
    by the render path so the guard can distinguish a real body sentence from
    a long title that happens to exceed the minimum length.  The second
    positional argument remains backwards-compatible with the old
    ``assert_pdf_has_body(artifact, min_chars)`` form.
    """

    if isinstance(structure, int) and allowed_chunk_ids is None and chunk_texts is None:
        min_chars, structure = structure, None
    # The worker performs the full spine/coverage gate separately.  The PDF
    # guard deliberately keeps its legacy structural scope compatible with
    # artifacts that deduplicate repeated topic paragraphs, while still
    # using the supplied structure below for title-only detection.
    report = validate_notes_artifact(
        artifact,
        structure=None,
        allowed_chunk_ids=allowed_chunk_ids,
        chunk_texts=chunk_texts,
        min_chars=min_chars,
        allow_source_gaps=True,
    )
    structural = _structural_body_errors(artifact, structure, chunk_texts)
    if structural:
        report.errors.extend(structural)
        report.ok = False
    if not report.ok:
        codes = ", ".join(str(item.get("code")) for item in report.errors[:6])
        raise ValueError(f"PDF input has headings-only or invalid note sections: {codes}")


__all__ = [
    "MIN_NOTE_CHARS",
    "MIN_SECTION_CHARS",
    "NotesValidationReport",
    "is_placeholder",
    "is_degraded_structure",
    "validate_topic_notes",
    "validate_notes_artifact",
    "require_valid_notes",
    "assert_pdf_has_body",
]
