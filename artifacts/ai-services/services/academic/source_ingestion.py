"""Deterministic source-type ingestion for study material.

Uploaded source files remain ordinary documents.  This module classifies
and normalises the two structured academic source families without making
the LLM responsible for arithmetic or ordering:

* syllabus: units -> topics -> canonical topic keys;
* question paper: sections -> questions -> marks/parts -> topic mapping.

The parsers are intentionally conservative.  A line is a topic only when
it is under a recognised unit/section heading or is explicitly marked as a
topic.  Unknown text remains in a deterministic ``unmapped`` bucket rather
than being invented into a curriculum.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from services.academic.spine import normalize_key


class SourceType(str, Enum):
    STUDY_MATERIAL = "study_material"
    SYLLABUS = "syllabus"
    QUESTION_PAPER = "question_paper"
    QUESTION_BANK = "question_bank"
    REFERENCE = "reference"


class CanonicalTopic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    aliases: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)

    @field_validator("aliases")
    @classmethod
    def _dedupe_aliases(cls, values: List[str]) -> List[str]:
        seen = set()
        result = []
        for value in values:
            key = normalize_key(value)
            if key and key not in seen:
                seen.add(key)
                result.append(value)
        return result


class SyllabusUnitIngested(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    topics: List[str] = Field(default_factory=list)
    weight_percent: Optional[float] = Field(default=None, ge=0, le=100)
    order: int = Field(ge=0)
    source_chunk_ids: List[str] = Field(default_factory=list)

    @field_validator("topics")
    @classmethod
    def _nonempty_topics(cls, values: List[str]) -> List[str]:
        if any(not str(value or "").strip() for value in values):
            raise ValueError("syllabus topics must be non-empty")
        return values


class SyllabusIngested(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    units: List[SyllabusUnitIngested] = Field(min_length=1)
    canonical_topics: List[CanonicalTopic] = Field(default_factory=list)
    unmapped: List[str] = Field(default_factory=list)
    source_type: str = SourceType.SYLLABUS.value


class QuestionPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    text: str = Field(min_length=1)
    marks: Optional[int] = Field(default=None, ge=0, le=100)


class QuestionPaperQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    number: int = Field(ge=0)
    text: str = Field(min_length=1)
    year: int = Field(ge=1950, le=2100)
    marks: int = Field(gt=0, le=100)
    section: str = Field(min_length=1)
    parts: List[QuestionPart] = Field(default_factory=list)
    topic: Optional[str] = None
    unit_id: Optional[str] = None
    source_chunk_ids: List[str] = Field(default_factory=list)
    mapping_method: str = "unmapped"

    @field_validator("parts")
    @classmethod
    def _valid_parts(cls, values: List[QuestionPart]) -> List[QuestionPart]:
        for part in values:
            if not part.text.strip():
                raise ValueError("question part text must be non-empty")
        return values


class QuestionPaperIngested(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    year: int = Field(ge=1950, le=2100)
    questions: List[QuestionPaperQuestion] = Field(default_factory=list)
    sections: List[str] = Field(default_factory=list)
    unmapped: List[str] = Field(default_factory=list)
    source_type: str = SourceType.QUESTION_PAPER.value


_UNIT_RE = re.compile(
    r"^\s*(unit|chapter|module|part|section)\s*([0-9]+|[ivx]+)\s*[:.\-]?\s*(.*)$",
    re.IGNORECASE,
)
# Question papers label their sections with letters ("Section A: Arrays") as
# often as with numbers.  _UNIT_RE only matches numeric/roman labels, so the
# letter form needs its own pattern or every Section A/B heading is dropped
# into `unmapped` and all questions land in a single "General" section.
_QP_SECTION_RE = re.compile(
    r"^\s*(unit|chapter|module|part|section)\s*([0-9]+|[a-z]|[ivx]+)\s*[:.\-]\s*(.+)$",
    re.IGNORECASE,
)
_TOPIC_RE = re.compile(r"^\s*(?:topic\s*)?[-*]?\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(.+?)\s*$", re.I)
_MARKS_RE = re.compile(r"(?:\(|\[)?\s*(\d{1,3})\s*marks?\s*(?:\)|\])?", re.I)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_NUMBER_RE = re.compile(r"^\s*(?:question\s*)?(\d+)[.):]?\s+(.*)$", re.I)
# Sub-parts appear as "(a)", "a.", "a)", "i.", and "- bullet".  A single
# optional-paren form is not enough: making only "(" optional left the ")"
# mandatory, so "a. State the invariant" fell through to body prose.  The
# bare-letter branch requires a "." or ")" separator so ordinary words such
# as "algorithm design" are never mistaken for a part label.
_PART_RE = re.compile(
    r"^\s*(?:\(([a-zivx]{1,3})\)|([a-zivx]{1,3})\s*[.)]|[-*])\s*(.+)$",
    re.I,
)
_TOPIC_MARK_RE = re.compile(r"^\s*(?:topic|syllabus)\s*[:\-]\s*(.+?)\s*$", re.I)


def _line_records(text: str, chunk_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    current_chunk = None
    ids = list(chunk_ids or [])
    out: List[Dict[str, Any]] = []
    for index, raw in enumerate((text or "").splitlines()):
        line = " ".join(raw.split())
        if not line:
            continue
        if ids:
            # A caller may provide a flat chunk list; associate a line to a
            # deterministic chunk by source order when no explicit map exists.
            current_chunk = ids[min(index, len(ids) - 1)]
        out.append({"line": line, "chunk_id": current_chunk})
    return out


def _year_from(text: str, default: int) -> int:
    match = _YEAR_RE.search(text or "")
    if match:
        value = int(match.group(1))
        if 1950 <= value <= 2100:
            return value
    return int(default)


def parse_syllabus_text(
    source_id: str,
    title: str,
    text: str,
    chunk_ids: Optional[Sequence[str]] = None,
) -> SyllabusIngested:
    """Parse units/topics in first-seen source order.

    ``topic:`` lines are preferred.  If a unit has no explicit topic lines,
    its first substantive following line is retained as a conservative
    fallback, and no unrelated line is promoted globally.
    """

    records = _line_records(text, chunk_ids)
    units: List[List[Any]] = []
    current: Optional[Dict[str, Any]] = None
    unmapped: List[str] = []
    for record in records:
        line = record["line"]
        match = _UNIT_RE.match(line)
        if match:
            number = match.group(2).upper()
            label = f"{match.group(1).title()} {number}"
            suffix = match.group(3).strip()
            current = {
                "unit_id": f"unit-{len(units) + 1:02d}-{normalize_key(label).replace(' ', '-')}",
                "title": f"{label} -- {suffix}" if suffix else label,
                "topics": [], "source_chunk_ids": [], "order": len(units),
            }
            units.append(current)
            if record.get("chunk_id"):
                current["source_chunk_ids"].append(record["chunk_id"])
            continue
        if current is None:
            unmapped.append(line)
            continue
        topic_match = _TOPIC_MARK_RE.match(line)
        topic = topic_match.group(1) if topic_match else None
        if topic is None:
            # A bullet or a short heading is a topic only in a syllabus
            # source; long prose is left in unmapped for an explicit model.
            if len(line) <= 160 and not line.endswith((".", "!", "?")):
                topic = re.sub(r"^[-*]\s*", "", line).strip()
        if topic:
            topic = re.sub(r"^[-*]\s*", "", topic).strip(" -:")
            if topic and normalize_key(topic) not in {normalize_key(t) for t in current["topics"]}:
                current["topics"].append(topic)
                if record.get("chunk_id"):
                    current["source_chunk_ids"].append(record["chunk_id"])
        elif len(line) > 160:
            unmapped.append(line)
    if not units:
        # One implicit unit keeps the result usable for plain-text syllabi.
        lines = [record["line"] for record in records if len(record["line"]) <= 160]
        if not lines:
            raise ValueError("syllabus source has no usable unit or topic lines")
        units.append({
            "unit_id": "unit-01-course",
            "title": title or "Course syllabus",
            "topics": lines,
            "source_chunk_ids": [r["chunk_id"] for r in records if r.get("chunk_id")],
            "order": 0,
        })
    canonical: List[CanonicalTopic] = []
    seen: Dict[str, CanonicalTopic] = {}
    for unit in units:
        for topic in unit["topics"]:
            key = normalize_key(topic)
            if not key:
                continue
            item = seen.get(key)
            if item is None:
                item = CanonicalTopic(key=key, title=topic, aliases=[], source_ids=[source_id])
                seen[key] = item
                canonical.append(item)
            elif topic not in item.aliases:
                item.aliases.append(topic)
    parsed_units = [SyllabusUnitIngested(**unit) for unit in units]
    return SyllabusIngested(
        source_id=str(source_id), title=str(title or "Syllabus"),
        units=parsed_units, canonical_topics=canonical,
        unmapped=unmapped, source_type=SourceType.SYLLABUS.value,
    )


def _parse_parts(lines: Sequence[str]) -> Tuple[str, List[QuestionPart]]:
    text_lines: List[str] = []
    parts: List[QuestionPart] = []
    for line in lines:
        match = _PART_RE.match(line)
        if match and (match.group(1) or match.group(2) or line.lstrip().startswith("-")):
            part_text = match.group(3).strip()
            marks_match = _MARKS_RE.search(part_text)
            marks = int(marks_match.group(1)) if marks_match else None
            if marks_match:
                # Keep the explanatory text, not the arithmetic annotation.
                part_text = _MARKS_RE.sub("", part_text).strip(" ()[]")
            parts.append(QuestionPart(
                label=(match.group(1) or match.group(2) or "-").upper(),
                text=part_text or match.group(3).strip(),
                marks=marks,
            ))
        elif line:
            text_lines.append(line)
    return " ".join(text_lines), parts


def parse_question_paper_text(
    source_id: str,
    title: str,
    text: str,
    year: Optional[int] = None,
    chunk_ids: Optional[Sequence[str]] = None,
    syllabus_topics: Optional[Sequence[str]] = None,
) -> QuestionPaperIngested:
    """Parse question number, year, marks, section, subparts, and topic mapping."""

    records = _line_records(text, chunk_ids)
    paper_year = _year_from(title + "\n" + text, year or datetime_year())
    sections: List[str] = []
    questions: List[QuestionPaperQuestion] = []
    current_section = "General"
    current: Optional[Dict[str, Any]] = None
    unmapped: List[str] = []

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        body, parts = _parse_parts(current["lines"])
        # A mark annotation on the numbered question line is the question
        # total.  If only sub-parts carry marks, sum those parts instead of
        # accidentally treating the first part as the whole question.
        first_line = str(current["lines"][0] if current["lines"] else "")
        question_marks_match = _MARKS_RE.search(first_line)
        continuation_marks_match = None
        if question_marks_match is None and not any(part.marks is not None for part in parts):
            continuation_marks_match = _MARKS_RE.search(" ".join(current["lines"][1:]))
        if current.get("marks"):
            marks = int(current["marks"])
        elif question_marks_match:
            marks = int(question_marks_match.group(1))
        elif continuation_marks_match:
            marks = int(continuation_marks_match.group(1))
        else:
            marks = sum(part.marks or 0 for part in parts) or 1
        # If a part-level total is present and no question total is, use it.
        if not current.get("marks") and not question_marks_match and not continuation_marks_match:
            part_total = sum(part.marks or 0 for part in parts)
            marks = max(1, part_total or 1)
        body = _TOPIC_MARK_RE.sub(r"", body).strip()
        topic = None
        method = "unmapped"
        if syllabus_topics:
            qtokens = set(normalize_key(body).split())
            ranked = sorted(
                ((len(qtokens & set(normalize_key(candidate).split())), candidate)
                 for candidate in syllabus_topics),
                key=lambda item: (-item[0], normalize_key(item[1])),
            )
            if ranked and ranked[0][0] > 0:
                topic, method = ranked[0][1], "token_overlap"
        questions.append(QuestionPaperQuestion(
            number=current["number"], text=body or f"Question {current['number']}",
            year=current.get("year", paper_year), marks=max(1, min(100, marks)),
            section=current_section, parts=parts, topic=topic,
            mapping_method=method, source_chunk_ids=current["chunk_ids"],
        ))
        if not topic and syllabus_topics:
            unmapped.append(f"question-{current['number']:03d}")
        current = None

    for record in records:
        line = record["line"]
        section_match = _QP_SECTION_RE.match(line)
        if section_match and not _NUMBER_RE.match(line):
            label = f"{section_match.group(1).title()} {section_match.group(2).upper()}"
            current_section = f"{label} -- {section_match.group(3).strip()}"
            if current_section not in sections:
                sections.append(current_section)
            continue
        number_match = _NUMBER_RE.match(line)
        if number_match:
            flush()
            body = number_match.group(2).strip()
            marks_match = _MARKS_RE.search(body)
            body = _MARKS_RE.sub("", body).strip(" ()[]")
            current = {
                "number": int(number_match.group(1)),
                "lines": [body] if body else [],
                "chunk_ids": [record["chunk_id"]] if record.get("chunk_id") else [],
                "marks": int(marks_match.group(1)) if marks_match else None,
            }
            explicit_year = _YEAR_RE.search(line)
            if explicit_year:
                current["year"] = int(explicit_year.group(1))
            continue
        if current is not None:
            current["lines"].append(line)
            if record.get("chunk_id"):
                current["chunk_ids"].append(record["chunk_id"])
        elif line and not line.lower().startswith(("session", "time allowed", "instructions")):
            unmapped.append(line)
    flush()
    if not questions:
        raise ValueError("question paper has no numbered questions")
    return QuestionPaperIngested(
        source_id=str(source_id), title=str(title or "Question paper"),
        year=paper_year, questions=questions,
        sections=sections or ["General"], unmapped=unmapped,
        source_type=SourceType.QUESTION_PAPER.value,
    )


def datetime_year() -> int:
    """Small injectable default that avoids importing wall-clock time in tests."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).year


def map_questions_to_syllabus(
    questions: Sequence[QuestionPaperQuestion],
    syllabus: Sequence[SyllabusUnitIngested],
    min_overlap: int = 1,
) -> List[Dict[str, Any]]:
    """Deterministic question -> unit/topic mapping with explicit unmapped rows."""

    topics = [(unit.unit_id, unit.title, topic) for unit in syllabus for topic in unit.topics]
    mapped: List[Dict[str, Any]] = []
    for question in questions:
        qtokens = set(normalize_key(question.text).split())
        best: Optional[Tuple[int, str, str, str]] = None
        for unit_id, unit_title, topic in topics:
            overlap = len(qtokens & set(normalize_key(f"{unit_title} {topic}").split()))
            if overlap >= int(min_overlap) and (
                best is None or (-overlap, normalize_key(topic), unit_id) < (-best[0], normalize_key(best[2]), best[1])
            ):
                best = (overlap, unit_id, topic, unit_title)
        selected_topic = best[2] if best else None
        selected_unit = best[1] if best else None
        selected_method = "token_overlap" if best else "unmapped"
        # Keep the typed question and the standalone mapping helper in sync;
        # metrics can then consume either path without a second, divergent
        # mapping decision.
        question.topic = selected_topic
        question.unit_id = selected_unit
        question.mapping_method = selected_method
        mapped.append({
            "number": question.number,
            "unit_id": selected_unit,
            "unit_title": best[3] if best else None,
            "topic": selected_topic,
            "overlap": best[0] if best else 0,
            "method": selected_method,
        })
    return mapped


def compute_source_metrics(
    syllabus: Sequence[SyllabusUnitIngested],
    papers: Sequence[QuestionPaperIngested],
    document_topics: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute deterministic frequency, marks, coverage, and missing topics."""

    topic_rows: List[Dict[str, Any]] = []
    docs = {normalize_key(value) for value in (document_topics or [])}
    for unit in syllabus:
        for ord_value, topic in enumerate(unit.topics, start=1):
            key = normalize_key(topic)
            rows = [q for paper in papers for q in paper.questions if normalize_key(q.topic or "") == key]
            frequency = len(rows)
            marks = sum(q.marks for q in rows)
            topic_rows.append({
                "unit_id": unit.unit_id,
                "unit_title": unit.title,
                "topic": topic,
                "topic_key": key,
                "ord": ord_value,
                "frequency": frequency,
                "marks_total": marks,
                "marks_share": round(marks / sum(q.marks for p in papers for q in p.questions) * 100, 2)
                if sum(q.marks for p in papers for q in p.questions) else 0.0,
                "years": sorted({q.year for q in rows}),
                "in_documents": key in docs,
                "in_question_papers": frequency > 0,
            })
    total_marks = sum(q.marks for paper in papers for q in paper.questions)
    mapped = sum(1 for q in (question for paper in papers for question in paper.questions) if q.topic)
    total_questions = sum(len(paper.questions) for paper in papers)
    return {
        "topics": topic_rows,
        "totals": {
            "units": len(syllabus),
            "topics": len(topic_rows),
            "question_papers": len(papers),
            "questions": total_questions,
            "mapped_questions": mapped,
            "unmapped_questions": total_questions - mapped,
            "total_marks": total_marks,
            "coverage_pct": round(mapped / total_questions * 100, 2) if total_questions else 0.0,
        },
        "missing": [row["topic"] for row in topic_rows if not row["in_question_papers"]],
        "never_asked": [row["topic"] for row in topic_rows if row["frequency"] == 0],
        "unmapped_questions": [
            f"paper-{paper.source_id}:q-{q.number:03d}"
            for paper in papers for q in paper.questions if not q.topic
        ],
    }


__all__ = [
    "SourceType",
    "CanonicalTopic",
    "SyllabusUnitIngested",
    "SyllabusIngested",
    "QuestionPart",
    "QuestionPaperQuestion",
    "QuestionPaperIngested",
    "parse_syllabus_text",
    "parse_question_paper_text",
    "map_questions_to_syllabus",
    "compute_source_metrics",
]
