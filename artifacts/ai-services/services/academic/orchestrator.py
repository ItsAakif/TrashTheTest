"""Production wiring for the academic generation vertical.

The modules in this package are deliberately deterministic and testable in
isolation.  This module is the small production seam that connects them to
the existing Postgres, retrieval, provider, storage, job, and compatibility
paths:

    processed chunks -> extraction -> persisted spine -> per-topic notes ->
    bounded coverage repair -> synthesis -> PDF -> storage -> artifact and
    compatibility projections

No new queue, retrieval implementation, or database access pattern is
introduced.  The job handler stores its resume state in the existing
``jobs.output_data`` JSONB column and reports progress through
``services.jobs.queue``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import ValidationError

from database import knowledge_repo, repositories
from services.academic.coverage import (
    DEFAULT_COVERAGE_THRESHOLD,
    MAX_REPAIR_PASSES,
    attach_report,
    evaluate_coverage,
    missing_evidence,
)
from services.academic.sufficiency import (
    SufficiencyStatus,
    assess_source_sufficiency,
    recover_topic_evidence,
)
from services.academic.validation import (
    MIN_SECTION_CHARS,
    is_degraded_structure,
    validate_notes_artifact,
)
from services.academic.extraction import extract_structure, merge_adjacent_topics
from services.academic.source_ingestion import (
    SourceType,
    QuestionPaperIngested,
    QuestionPaperQuestion,
    SyllabusIngested,
    SyllabusUnitIngested,
    compute_source_metrics,
    map_questions_to_syllabus,
    parse_question_paper_text,
    parse_syllabus_text,
)
from services.academic.flashcards import (
    build_flashcards,
    flashcards_artifact,
    to_deck_cards,
    validate_deck,
)
from services.academic.mcq import (
    mcq_artifact,
    project_quiz_questions,
    validate_set,
)
from services.academic.metrics import (
    compute_metrics,
    metrics_artifact,
    normalize_inputs,
    repair_narrative,
)
from services.academic.notes_hier import (
    DEFAULT_TOPIC_BATCH_TOKENS,
    NOTES_TOPIC_MAX_OUTPUT_TOKENS,
    build_outline,
    group_topics,
    render_topic_notes,
    select_topic_chunks,
    synthesize_notes,
    topic_query,
)
from services.academic.spine import normalize_key
from services.academic.types import (
    DocumentStructure,
    ExtractedConcept,
    ExtractedSubtopic,
    ExtractedTopic,
    NotesArtifact,
    NotesGenerationConfig,
    RepairRecord,
    StructuredUnit,
    StructureMethod,
    TopicNotes,
    TopicShape,
    not_generated_state,
)
from services.academic import artifacts as academic_artifacts
from services.academic import flashcards as academic_flashcards
from services.academic import mcq as academic_mcq

logger = logging.getLogger(__name__)


# These names are intentionally separate from the document-processing
# worker's stage vocabulary.  A job records the exact stage that is running
# and every write merges into output_data, so a restart can skip completed
# groups without losing the stage ledger.
NOTES_HIERARCHICAL_STAGES: Tuple[str, ...] = (
    "model",
    "outline",
    "group",
    "per_group",
    "coverage",
    "repair",
    "synthesis",
    "validate",
    "persist",
    "pdf",
)

NOTES_HIERARCHICAL_STAGE_PROGRESS_RANGES: Dict[str, Tuple[int, int]] = {
    "model": (0, 10),
    "outline": (10, 15),
    "group": (15, 20),
    "per_group": (20, 65),
    "coverage": (65, 70),
    "repair": (70, 80),
    "synthesis": (80, 84),
    "validate": (84, 88),
    "persist": (88, 94),
    "pdf": (94, 100),
}

# Friendly aliases used by callers/tests that expect the shorter contract.
NOTES_HIERARCHICAL_STAGES_DICT = NOTES_HIERARCHICAL_STAGE_PROGRESS_RANGES
STAGE_PROGRESS_RANGES = NOTES_HIERARCHICAL_STAGE_PROGRESS_RANGES
JOB_TYPE = "notes_hierarchical"

# Per-topic generation bounds.  A slow provider (free router) must never let
# one topic hold the whole job open: each topic call is bounded, and topics
# are generated with a small bounded pool instead of strictly sequentially.
# 300s default: a free router routinely needs more than 150s for a dense
# topic, and the earlier bound turned a slow-but-working provider into a
# hard failure.  The bound is still finite so one topic can never hang a job.
DEFAULT_PER_TOPIC_TIMEOUT_S = 120.0
DEFAULT_TOPIC_CONCURRENCY = 4
MIN_PER_TOPIC_TIMEOUT_S = 30.0
MAX_PER_TOPIC_TIMEOUT_S = 1800.0

# A document that resolves to a great many tiny topics makes one bounded LLM
# call per topic; merging adjacent topics into page-window topics keeps the
# call count (and the wall-clock) proportional to the document, not to the
# extractor's topic granularity.  Deterministic and source-ordered.
MAX_TOPICS_PER_JOB = 8
MIN_MAX_TOPICS_PER_JOB = 4
MAX_MAX_TOPICS_PER_JOB = 30


def _topic_timeout_s() -> float:
    raw = os.getenv("NOTES_PER_TOPIC_TIMEOUT_S")
    try:
        value = float(raw) if raw else DEFAULT_PER_TOPIC_TIMEOUT_S
    except (TypeError, ValueError):
        value = DEFAULT_PER_TOPIC_TIMEOUT_S
    return max(MIN_PER_TOPIC_TIMEOUT_S, min(MAX_PER_TOPIC_TIMEOUT_S, value))


def _max_topics_per_job() -> int:
    raw = os.getenv("NOTES_MAX_TOPICS")
    try:
        value = int(raw) if raw else MAX_TOPICS_PER_JOB
    except (TypeError, ValueError):
        value = MAX_TOPICS_PER_JOB
    return max(MIN_MAX_TOPICS_PER_JOB, min(MAX_MAX_TOPICS_PER_JOB, value))


def _gaps_provider_slow(gaps: Sequence[Mapping[str, Any]]) -> bool:
    """True when any recorded gap is a per-topic generation timeout."""
    return any(
        "timed out" in str(gap.get("reason") or "") for gap in (gaps or [])
    )


def _topic_concurrency() -> int:
    raw = os.getenv("NOTES_TOPIC_CONCURRENCY")
    try:
        value = int(raw) if raw else DEFAULT_TOPIC_CONCURRENCY
    except (TypeError, ValueError):
        value = DEFAULT_TOPIC_CONCURRENCY
    return max(1, min(4, value))


# A single AsyncSession is not safe for concurrent use, but per-topic note
# generation deliberately overlaps the LLM calls.  Every retrieval and every
# progress/commit write therefore passes through this lock, while the provider
# calls themselves stay concurrent (they never touch the session).
_DB_LOCK: Optional[asyncio.Lock] = None


def _db_lock() -> asyncio.Lock:
    """Serialise DB access for concurrent per-topic tasks.

    An AsyncSession is not safe for concurrent use, so every retrieval /
    progress write performed inside a concurrent topic task takes this lock.
    LLM calls stay concurrent (they do not touch the session).

    The lock is created lazily and re-created when the running event loop
    changes (for example a new pytest-asyncio loop per test).  ``Lock.loop``
    was removed in Python 3.10, so the bound loop is probed defensively.
    """
    global _DB_LOCK
    loop = asyncio.get_running_loop()
    if _DB_LOCK is None:
        _DB_LOCK = asyncio.Lock()
        return _DB_LOCK
    bound = getattr(_DB_LOCK, "_loop", None)
    if bound is None:
        try:
            bound = _DB_LOCK._get_loop()  # type: ignore[attr-defined]
        except RuntimeError:
            # Already bound to a different (closed) loop.
            bound = None
        except Exception:
            bound = loop
    if bound is not None and bound is not loop:
        _DB_LOCK = asyncio.Lock()
    return _DB_LOCK


class AcademicPipelineError(RuntimeError):
    """Base error for an honest terminal academic-generation failure."""


class NoExtractableTextError(AcademicPipelineError):
    """Raised when a document scope contains no usable chunk text."""


@dataclass
class SpineBuildResult:
    """Result of extraction plus optional spine persistence."""

    structure: DocumentStructure
    source_structures: List[DocumentStructure]
    source_ids: List[str]
    source_fingerprint: str
    model_version: int
    root_ids: List[Any]
    chunks: List[Dict[str, Any]]
    reused: bool = False
    model_header_id: Optional[Any] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "source_fingerprint": self.source_fingerprint,
            "source_ids": list(self.source_ids),
            "root_ids": [str(value) for value in self.root_ids],
            "reused": self.reused,
            "stats": spine_stats(self.structure),
        }


# ---------------------------------------------------------------------------
# Generic row conversion and source loading
# ---------------------------------------------------------------------------


def _get(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _as_uuid(value: Any) -> Optional[uuid.UUID]:
    if value is None or value == "":
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError):
        return None


def _page_value(row: Any, metadata: Dict[str, Any]) -> Optional[int]:
    for candidate in (
        _get(row, "page_start"),
        _get(row, "page"),
        metadata.get("page_start"),
        metadata.get("page"),
        metadata.get("page_number"),
    ):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def chunk_to_dict(row: Any, document: Any = None, position: int = 0) -> Dict[str, Any]:
    """Convert an ORM chunk/retrieval row to the academic evidence shape."""
    metadata = dict(_get(row, "chunk_metadata", None) or _get(row, "metadata", None) or {})
    text = _get(row, "content", None)
    if text is None:
        text = _get(row, "text", "")
    heading = _get(row, "heading", None) or metadata.get("heading")
    section = _get(row, "section", None) or metadata.get("section")
    if heading is None and section:
        heading = section
    if heading is None:
        path = _get(row, "section_path", None) or metadata.get("section_path") or []
        heading = " / ".join(str(value) for value in path if value) or None
    document_id = _get(row, "document_id", None)
    if document_id is None and document is not None:
        document_id = _get(document, "id", None)
    source_type = _get(document, "source_type", None) if document is not None else None
    if hasattr(source_type, "value"):
        source_type = source_type.value
    return {
        "id": str(_get(row, "id", None) or _get(row, "chunk_id", None) or ""),
        "chunk_id": str(_get(row, "chunk_id", None) or _get(row, "id", None) or ""),
        "document_id": str(document_id or ""),
        "text": str(text or ""),
        "content": str(text or ""),
        "page": _page_value(row, metadata),
        "position": _get(row, "position", metadata.get("position", position)),
        "heading": str(heading or ""),
        "section": str(section or ""),
        "section_path": list(_get(row, "section_path", None) or metadata.get("section_path") or []),
        "metadata": metadata,
        "document_title": str(
            _get(document, "file_name", None)
            or _get(document, "title", None)
            or _get(document, "name", None)
            or ""
        ),
        "source_type": str(source_type or "study_material"),
        "similarity": _get(row, "similarity", _get(row, "score", 0.0)),
        "score": _get(row, "similarity", _get(row, "score", 0.0)),
    }


def _forward_structure_verdicts(
    sections: Sequence[Any],
    chunk_index: Mapping[str, Dict[str, Any]],
) -> int:
    """ADR-008 T006: copy a degraded structure verdict onto the sections.

    ``validate_notes_artifact`` downgrades ``topic_mismatch`` to a warning
    only when ``section.metadata.structure_method`` /
    ``structure_confidence`` say the spine label is a placeholder built by
    page-bounded or inferred detection.  Chunks carry that verdict inside
    their metadata JSON, but nothing forwarded it onto the synthesized
    artifact, so the gate stayed hard for exactly the documents the degrade
    rule was written for.  Only degraded verdicts are copied: healthy
    evidence leaves the section untouched and the gate stays strict.
    Returns the number of sections stamped.
    """

    forwarded = 0
    for section in sections:
        verdict = None
        for citation in _get(section, "citations", None) or []:
            cid = str(_get(citation, "chunk_id", "") or "")
            row = chunk_index.get(cid)
            if not row:
                continue
            row_meta = _get(row, "metadata", None) or {}
            method = _get(row_meta, "structure_method", None)
            confidence = _get(row_meta, "structure_confidence", None)
            if is_degraded_structure(method, confidence):
                verdict = (method, confidence)
                break
        if verdict is None:
            continue
        method, confidence = verdict
        meta = dict(_get(section, "metadata", None) or {})
        meta["structure_method"] = method
        meta["structure_confidence"] = confidence
        if isinstance(section, Mapping):
            section["metadata"] = meta
        else:
            section.metadata = meta
        forwarded += 1
    return forwarded


async def load_processed_chunks(
    db: Any,
    project_id: Any,
    document_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Load processed chunks through existing repositories, source order first.

    No relationship is touched here.  Documents are sorted by their stored
    creation time/name and chunks by the repository's source order; the
    extraction helper applies the final page/position ordering.
    """
    pid = _as_uuid(project_id) or project_id
    documents = await repositories.get_documents_by_project(db, pid)
    documents = list(documents or [])
    wanted = {str(value) for value in (document_ids or [])}
    if wanted:
        documents = [doc for doc in documents if str(_get(doc, "id")) in wanted]
    documents.sort(
        key=lambda doc: (
            str(_get(doc, "created_at", "") or ""),
            str(_get(doc, "file_name", "") or ""),
            str(_get(doc, "id", "") or ""),
        )
    )
    out: List[Dict[str, Any]] = []
    for document in documents:
        did = _get(document, "id", None)
        rows = await repositories.get_chunks_by_document(db, did)
        for position, row in enumerate(rows or []):
            item = chunk_to_dict(row, document=document, position=position)
            if item["text"].strip():
                out.append(item)
    # A test/double may provide pre-shaped chunk rows without a Document
    # relation.  Keep this fallback narrow and deterministic.
    if not out and not documents:
        try:
            rows = await repositories.get_chunks_by_project(db, pid)
        except Exception:
            rows = []
        for position, row in enumerate(rows or []):
            item = chunk_to_dict(row, position=position)
            if item["text"].strip():
                out.append(item)
    if not out:
        raise NoExtractableTextError(
            "no extractable text: processed chunks are empty for this project"
        )
    return out


# ---------------------------------------------------------------------------
# Deterministic multi-document structure merge
# ---------------------------------------------------------------------------


def _merge_unique(*values: Iterable[Any]) -> List[Any]:
    seen: Set[str] = set()
    out: List[Any] = []
    for value in values:
        if value is None:
            continue
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            if item is None:
                continue
            key = str(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def _merge_concepts(
    left: Sequence[ExtractedConcept], right: Sequence[ExtractedConcept]
) -> List[ExtractedConcept]:
    out: List[ExtractedConcept] = []
    positions: Dict[str, int] = {}
    for concept in [*left, *right]:
        key = normalize_key(concept.title)
        if key in positions:
            old = out[positions[key]]
            merged = old.model_copy(update={
                "pages": _merge_unique(old.pages, concept.pages),
                "chunks": _merge_unique(old.chunks, concept.chunks),
                "confidence": max(
                    old.confidence if old.confidence is not None else 0.0,
                    concept.confidence if concept.confidence is not None else 0.0,
                ) or None,
            })
            out[positions[key]] = merged
        else:
            positions[key] = len(out)
            out.append(concept.model_copy(deep=True))
    return out


def _merge_subtopics(
    left: Sequence[ExtractedSubtopic], right: Sequence[ExtractedSubtopic]
) -> List[ExtractedSubtopic]:
    out: List[ExtractedSubtopic] = []
    positions: Dict[str, int] = {}
    for sub in [*left, *right]:
        key = normalize_key(sub.title)
        if key in positions:
            old = out[positions[key]]
            out[positions[key]] = old.model_copy(update={
                "pages": _merge_unique(old.pages, sub.pages),
                "chunks": _merge_unique(old.chunks, sub.chunks),
                "concepts": _merge_concepts(old.concepts, sub.concepts),
            })
        else:
            positions[key] = len(out)
            out.append(sub.model_copy(deep=True))
    return out


def _merge_topics(
    left: Sequence[ExtractedTopic], right: Sequence[ExtractedTopic]
) -> List[ExtractedTopic]:
    out: List[ExtractedTopic] = []
    positions: Dict[str, int] = {}
    for topic in [*left, *right]:
        key = normalize_key(topic.title)
        if key in positions:
            old = out[positions[key]]
            out[positions[key]] = old.model_copy(update={
                "pages": _merge_unique(old.pages, topic.pages),
                "chunks": _merge_unique(old.chunks, topic.chunks),
                "aliases": _merge_unique(old.aliases, topic.aliases),
                "subtopics": _merge_subtopics(old.subtopics, topic.subtopics),
                "concepts": _merge_concepts(old.concepts, topic.concepts),
            })
        else:
            positions[key] = len(out)
            out.append(topic.model_copy(deep=True))
    return out


def merge_document_structures(
    structures: Sequence[DocumentStructure],
    project_title: str = "Academic project",
) -> DocumentStructure:
    """Merge per-document structures in source order without LLM calls."""
    if not structures:
        raise NoExtractableTextError("no extractable text: no document structures")
    units: List[StructuredUnit] = []
    unit_positions: Dict[str, int] = {}
    for structure in structures:
        for unit in structure.units:
            key = normalize_key(unit.title)
            if key in unit_positions:
                old = units[unit_positions[key]]
                units[unit_positions[key]] = old.model_copy(update={
                    "pages": _merge_unique(old.pages, unit.pages),
                    "topics": _merge_topics(old.topics, unit.topics),
                    "aliases": _merge_unique(old.aliases, unit.aliases),
                })
            else:
                unit_positions[key] = len(units)
                units.append(unit.model_copy(deep=True))
    basis = json.dumps(
        [structure.fingerprint for structure in structures],
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(basis.encode("utf-8")).hexdigest()
    source_types = {
        str(getattr(getattr(structure, "source_type", None), "value", getattr(structure, "source_type", "study_material")))
        for structure in structures
    }
    source_metadata: Dict[str, Any] = {
        "source_types": sorted(source_types),
        "documents": {},
    }
    for structure in structures:
        doc_id = str(getattr(structure, "doc_id", ""))
        metadata = dict(getattr(structure, "source_metadata", None) or {})
        if doc_id:
            source_metadata["documents"][doc_id] = metadata
        # Preserve the common single-source fields for callers that consume
        # the merged structure directly.  A documents map remains the
        # lossless multi-source representation.
        for key in ("source_ingestion", "syllabus", "question_paper", "mapping", "questions"):
            if key in metadata:
                if key in source_metadata:
                    existing = source_metadata[key]
                    incoming = metadata[key]
                    if isinstance(existing, list) and isinstance(incoming, list):
                        existing.extend(incoming)
                    elif isinstance(existing, list):
                        existing.append(incoming)
                    else:
                        source_metadata[key] = [existing, incoming]
                else:
                    source_metadata[key] = metadata[key]
    return DocumentStructure(
        doc_id="project",
        title=project_title or "Academic project",
        units=units,
        fingerprint=fingerprint,
        source_type=(next(iter(source_types)) if len(source_types) == 1 else "study_material"),
        source_metadata=source_metadata,
    )


def _unique_row_ids(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values or []:
        cid = str(value or "")
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def _pages_for_ids(rows: Sequence[Dict[str, Any]], ids: Sequence[str]) -> List[int]:
    wanted = set(ids)
    pages: List[int] = []
    for row in rows:
        if str(row.get("id") or row.get("chunk_id") or "") not in wanted:
            continue
        page = row.get("page")
        try:
            page_value = int(page) if page is not None else None
        except (TypeError, ValueError):
            page_value = None
        if page_value is not None and page_value not in pages:
            pages.append(page_value)
    return pages


def _ids_mentioning(rows: Sequence[Dict[str, Any]], title: str) -> List[str]:
    """Best-effort source citation ids for a parsed topic label."""

    needle = normalize_key(title)
    if not needle:
        return []
    ids: List[str] = []
    for row in rows:
        text = str(row.get("text") or row.get("content") or "")
        heading = str(row.get("heading") or row.get("section") or "")
        if needle in normalize_key(f"{heading} {text}"):
            cid = str(row.get("id") or row.get("chunk_id") or "")
            if cid and cid not in ids:
                ids.append(cid)
    return ids


def _ingestion_fingerprint(doc_id: str, title: str, payload: Any) -> str:
    body = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(f"{doc_id}\n{title}\n{body}".encode("utf-8")).hexdigest()


def _source_type_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "study_material").strip().lower()


def _structure_from_syllabus(
    doc_id: str,
    title: str,
    rows: Sequence[Dict[str, Any]],
    parsed: SyllabusIngested,
) -> DocumentStructure:
    all_ids = _unique_row_ids(
        row.get("id") or row.get("chunk_id") for row in rows
    )
    units: List[StructuredUnit] = []
    for parsed_unit in parsed.units:
        unit_ids = _unique_row_ids(parsed_unit.source_chunk_ids) or all_ids
        topics: List[ExtractedTopic] = []
        seen_topics: Set[str] = set()
        for topic_title in parsed_unit.topics:
            key = normalize_key(topic_title)
            if not key or key in seen_topics:
                continue
            seen_topics.add(key)
            topic_ids = _ids_mentioning(rows, topic_title) or unit_ids
            topics.append(ExtractedTopic(
                title=topic_title,
                pages=_pages_for_ids(rows, topic_ids),
                chunks=topic_ids,
                subtopics=[],
                concepts=[],
                method=StructureMethod.TOC_HEADING,
            ))
        if not topics:
            continue
        units.append(StructuredUnit(
            title=parsed_unit.title,
            aliases=[],
            pages=_pages_for_ids(rows, unit_ids),
            topics=topics,
            method=StructureMethod.TOC_HEADING,
        ))
    if not units:
        raise ValueError("syllabus parser produced no usable units")
    parsed_payload = parsed.model_dump(mode="json")
    metadata = {
        "document_id": doc_id,
        "source_type": SourceType.SYLLABUS.value,
        "source_ingestion": parsed_payload,
        "syllabus": parsed_payload,
    }
    return DocumentStructure(
        doc_id=doc_id,
        title=title,
        units=units,
        fingerprint=_ingestion_fingerprint(doc_id, title, parsed_payload),
        source_type=SourceType.SYLLABUS.value,
        source_metadata=metadata,
    )


def _write_back_question_mapping(
    paper: QuestionPaperIngested,
    syllabus_units: Sequence[SyllabusUnitIngested],
) -> List[Dict[str, Any]]:
    """Make standalone parsing and metrics consume the same mapping."""

    rows = map_questions_to_syllabus(paper.questions, syllabus_units)
    by_number = {int(row.get("number", -1)): row for row in rows}
    for question in paper.questions:
        mapped = by_number.get(int(question.number), {})
        question.topic = mapped.get("topic") or question.topic
        question.unit_id = mapped.get("unit_id") or question.unit_id
        question.mapping_method = (
            str(mapped.get("method") or question.mapping_method or "unmapped")
        )
    return rows


def _structure_from_question_paper(
    doc_id: str,
    title: str,
    rows: Sequence[Dict[str, Any]],
    parsed: QuestionPaperIngested,
    syllabus_units: Sequence[SyllabusUnitIngested],
) -> DocumentStructure:
    mapping = _write_back_question_mapping(parsed, syllabus_units)
    all_ids = _unique_row_ids(
        row.get("id") or row.get("chunk_id") for row in rows
    )
    section_order: List[str] = []
    questions_by_section: Dict[str, List[QuestionPaperQuestion]] = {}
    for question in parsed.questions:
        section = str(question.section or "General")
        if section not in questions_by_section:
            section_order.append(section)
            questions_by_section[section] = []
        questions_by_section[section].append(question)
    if not section_order:
        section_order = list(parsed.sections) or ["General"]
        questions_by_section.setdefault(section_order[0], [])

    units: List[StructuredUnit] = []
    for section in section_order:
        questions = questions_by_section.get(section, [])
        if not questions:
            continue
        topic_groups: Dict[str, Dict[str, Any]] = {}
        for question in questions:
            topic_title = str(
                question.topic or f"Question {question.number}"
            )
            key = normalize_key(topic_title)
            if key not in topic_groups:
                topic_groups[key] = {
                    "title": topic_title,
                    "ids": [],
                }
            topic_groups[key]["ids"].extend(question.source_chunk_ids or [])
        topics: List[ExtractedTopic] = []
        for group in topic_groups.values():
            topic_ids = _unique_row_ids(group["ids"]) or all_ids
            topics.append(ExtractedTopic(
                title=group["title"],
                pages=_pages_for_ids(rows, topic_ids),
                chunks=topic_ids,
                subtopics=[],
                concepts=[],
                method=StructureMethod.TOC_HEADING,
            ))
        if not topics:
            continue
        section_ids = _unique_row_ids(
            cid for question in questions for cid in (question.source_chunk_ids or [])
        ) or all_ids
        units.append(StructuredUnit(
            title=section,
            aliases=[],
            pages=_pages_for_ids(rows, section_ids),
            topics=topics,
            method=StructureMethod.TOC_HEADING,
        ))
    if not units:
        raise ValueError("question-paper parser produced no usable questions")

    paper_payload = parsed.model_dump(mode="json")
    metadata = {
        "document_id": doc_id,
        "source_type": SourceType.QUESTION_PAPER.value,
        "source_ingestion": paper_payload,
        "question_paper": paper_payload,
        "questions": [question.model_dump(mode="json") for question in parsed.questions],
        "mapping": mapping,
    }
    return DocumentStructure(
        doc_id=doc_id,
        title=title,
        units=units,
        fingerprint=_ingestion_fingerprint(doc_id, title, paper_payload),
        source_type=SourceType.QUESTION_PAPER.value,
        source_metadata=metadata,
    )


def _boilerplate_lines_for_rows(rows: Sequence[Dict[str, Any]]) -> set:
    """Running-header / page-number lines across a document's chunk rows.

    Mirrors the chunker pipeline: group chunk texts by page, then flag
    lines that repeat on enough pages (find_boilerplate_lines).  Returns an
    empty set when the rows carry no page information.
    """
    from services.text_utils import find_boilerplate_lines

    pages: Dict[Any, List[str]] = {}
    for row in rows:
        pages.setdefault(row.get("page"), []).append(
            str(row.get("text") or row.get("content") or "")
        )
    if not pages:
        return set()
    ordered = [pages[key] for key in sorted(pages, key=lambda k: (k is None, k))]
    return find_boilerplate_lines(["\n".join(parts) for parts in ordered])


def _build_dispatch_structure(
    doc_id: str,
    title: str,
    rows: Sequence[Dict[str, Any]],
    syllabus_units: Sequence[SyllabusUnitIngested],
) -> Tuple[DocumentStructure, Optional[SyllabusIngested], Optional[QuestionPaperIngested]]:
    source_type = _source_type_value(rows[0].get("source_type"))
    text = "\n".join(
        str(row.get("text") or row.get("content") or "") for row in rows
    )
    if source_type == SourceType.SYLLABUS.value:
        parsed = parse_syllabus_text(
            doc_id, title, text,
            chunk_ids=[str(row.get("id") or row.get("chunk_id") or "") for row in rows],
        )
        return _structure_from_syllabus(doc_id, title, rows, parsed), parsed, None
    if source_type == SourceType.QUESTION_PAPER.value:
        topics = [topic for unit in syllabus_units for topic in unit.topics]
        parsed = parse_question_paper_text(
            doc_id, title, text,
            chunk_ids=[str(row.get("id") or row.get("chunk_id") or "") for row in rows],
            syllabus_topics=topics or None,
        )
        return (
            _structure_from_question_paper(doc_id, title, rows, parsed, syllabus_units),
            None,
            parsed,
        )
    return (
        extract_structure(
            doc_id, title, list(rows),
            source_type=source_type or "study_material",
            source_metadata={"document_id": doc_id, "source_type": source_type},
            boilerplate=_boilerplate_lines_for_rows(rows),
        ),
        None,
        None,
    )


async def build_source_structures(
    db: Any,
    project_id: Any,
    document_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[DocumentStructure]]:
    chunks = await load_processed_chunks(db, project_id, document_ids=document_ids)
    by_document: Dict[str, List[Dict[str, Any]]] = {}
    titles: Dict[str, str] = {}
    for row in chunks:
        did = str(row.get("document_id") or "")
        by_document.setdefault(did, []).append(row)
        titles.setdefault(did, str(row.get("document_title") or did))
    ordered_document_ids = list(dict.fromkeys(
        str(row.get("document_id") or "") for row in chunks
    ))

    # Parse syllabus sources first so a question paper in the same project can
    # receive its deterministic topic/unit mapping before either structure is
    # merged or metrics are computed.
    parsed_by_id: Dict[str, Tuple[DocumentStructure, Optional[SyllabusIngested], Optional[QuestionPaperIngested]]] = {}
    syllabus_units: List[SyllabusUnitIngested] = []
    for did in ordered_document_ids:
        rows = by_document.get(did, [])
        if not rows or _source_type_value(rows[0].get("source_type")) != SourceType.SYLLABUS.value:
            continue
        try:
            built = _build_dispatch_structure(did, titles.get(did) or did, rows, syllabus_units)
        except (TypeError, ValueError) as exc:
            raise NoExtractableTextError(
                f"no extractable text: invalid processed chunk scope ({exc})"
            ) from exc
        parsed_by_id[did] = built
        if built[1] is not None:
            syllabus_units.extend(built[1].units)

    for did in ordered_document_ids:
        if did in parsed_by_id:
            continue
        rows = by_document.get(did, [])
        if not rows:
            continue
        try:
            parsed_by_id[did] = _build_dispatch_structure(
                did, titles.get(did) or did, rows, syllabus_units
            )
        except (TypeError, ValueError) as exc:
            raise NoExtractableTextError(
                f"no extractable text: invalid processed chunk scope ({exc})"
            ) from exc

    structures = [parsed_by_id[did][0] for did in ordered_document_ids if did in parsed_by_id]
    return chunks, structures


def _source_fingerprint(structures: Sequence[DocumentStructure]) -> str:
    payload = [
        {
            "doc_id": s.doc_id,
            "fingerprint": s.fingerprint,
            "title": s.title,
            "source_type": s.source_type,
            "source_metadata": s.source_metadata,
        }
        for s in structures
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def spine_stats(structure: DocumentStructure) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for unit in structure.units:
        counts["unit"] = counts.get("unit", 0) + 1
        for topic in unit.topics:
            counts["topic"] = counts.get("topic", 0) + 1
            for sub in topic.subtopics or []:
                counts["subtopic"] = counts.get("subtopic", 0) + 1
                counts["concept"] = counts.get("concept", 0) + len(sub.concepts or [])
            counts["concept"] = counts.get("concept", 0) + len(topic.concepts or [])
    counts.setdefault("document", 1)
    return {
        "node_count": counts,
        "unresolved_count": 0,
        "canonical_collisions": 0,
    }


async def _existing_model_version(db: Any, project_id: Any) -> Tuple[int, Optional[Any], Optional[str]]:
    """Read the version header when migration 010 is present.

    JSONB node metadata is the compatibility fallback for databases that have
    not run the optional header migration yet.  This keeps the live path
    additive during rolling deploys.
    """
    try:
        rows = await knowledge_repo.list_knowledge_models(db, _as_uuid(project_id) or project_id)
        if rows:
            latest = max(rows, key=lambda row: int(_get(row, "version", 0) or 0))
            version = int(_get(latest, "version", 0) or 0)
            # A failed/in-flight header is not a reusable model. Keep its
            # version for monotonic numbering, but only an explicitly READY
            # header can short-circuit the next build.
            latest_status = _get(latest, "status", "ready")
            latest_status = getattr(latest_status, "value", latest_status)
            fingerprint = (
                str(_get(latest, "source_fingerprint", "") or "")
                if str(latest_status) == "ready"
                else ""
            )
            return version, latest, fingerprint
    except Exception:
        pass
    try:
        nodes = await knowledge_repo.list_knowledge_nodes(
            db, project_id=_as_uuid(project_id) or project_id
        )
    except Exception:
        nodes = []
    versions: List[int] = []
    fingerprint = ""
    for node in nodes or []:
        metadata = dict(_get(node, "node_metadata", None) or {})
        try:
            value = metadata.get("model_version") or _get(node, "model_version", None)
            if value is not None:
                versions.append(int(value))
        except (TypeError, ValueError):
            pass
        if not fingerprint and metadata.get("source_fingerprint"):
            fingerprint = str(metadata["source_fingerprint"])
    return (max(versions) if versions else 0), None, fingerprint


async def build_project_spine(
    db: Any,
    project_id: Any,
    document_ids: Optional[Sequence[str]] = None,
    project_title: str = "Academic project",
    force: bool = False,
    persist: bool = True,
) -> SpineBuildResult:
    """Extract and persist the canonical project spine.

    Existing rows with the same source fingerprint are reused.  Changed
    sources receive a new header version; old node versions are retained and
    marked superseded rather than deleted.
    """
    chunks, source_structures = await build_source_structures(
        db, project_id, document_ids=document_ids
    )
    merged = merge_document_structures(source_structures, project_title=project_title)
    fingerprint = _source_fingerprint(source_structures)
    current_version, header, old_fingerprint = await _existing_model_version(db, project_id)
    source_ids = [str(s.doc_id) for s in source_structures]

    if not force and old_fingerprint == fingerprint:
        root_ids: List[Any] = []
        try:
            nodes = await knowledge_repo.list_knowledge_nodes(
                db, project_id=_as_uuid(project_id) or project_id
            )
            # Ids, not row objects: callers store root_ids in the resumable job
            # state and pass them to knowledge_repo lookups.  Returning the ORM
            # rows made ``str(node)`` (their repr) land in that state, and the
            # next DB call failed with an asyncpg invalid-UUID error - i.e. a
            # second notes run on an unchanged project could never complete.
            root_ids = [
                _get(n, "id", None) for n in (nodes or [])
                if _get(n, "parent_id", None) is None
                and (
                    not current_version
                    or int(
                        (dict(_get(n, "node_metadata", None) or {})).get("model_version")
                        or _get(n, "model_version", 0)
                        or 0
                    ) == int(current_version)
                )
            ]
            root_ids = [value for value in root_ids if value is not None]
        except Exception:
            root_ids = []
        return SpineBuildResult(
            structure=merged,
            source_structures=source_structures,
            source_ids=source_ids,
            source_fingerprint=fingerprint,
            model_version=int(current_version or 1),
            root_ids=root_ids,
            chunks=chunks,
            reused=True,
            model_header_id=_get(header, "id", None),
        )

    version = int(current_version or 0) + 1
    header_id: Optional[Any] = None
    if persist:
        try:
            header = await knowledge_repo.create_knowledge_model(
                db,
                _as_uuid(project_id) or project_id,
                version,
                fingerprint,
                status="building",
            )
            header_id = _get(header, "id", None)
        except Exception as exc:
            # A rolling deployment may not have migration 010 yet.  The node
            # metadata version remains sufficient and the build continues.
            logger.info("knowledge model header unavailable; using JSONB metadata: %s", exc)
            header = None

    try:
        root_ids = []
        if persist:
            for source_structure in source_structures:
                spec = source_structure.to_spine_spec()
                for node in spec.nodes:
                    metadata = dict(node.node_metadata or {})
                    metadata.update({
                        "model_version": version,
                        "source_fingerprint": fingerprint,
                        "model_status": "ready",
                    })
                    if getattr(node.node_type, "value", node.node_type) == "document":
                        metadata.update({
                            "source_type": source_structure.source_type,
                            "source_metadata": source_structure.source_metadata,
                        })
                    node.node_metadata = metadata
                created = await knowledge_repo.persist_spine(
                    db,
                    _as_uuid(project_id) or project_id,
                    _as_uuid(source_structure.doc_id) or source_structure.doc_id,
                    spec,
                    model_version=version,
                )
                roots_for_doc = [
                    row for row in (created or [])
                    if _get(row, "parent_id", None) is None
                ]
                if roots_for_doc:
                    root_ids.append(_get(roots_for_doc[0], "id", None))
                else:
                    root_ids.extend(_get(row, "id", None) for row in (created or [])[:1])
            try:
                await knowledge_repo.supersede_knowledge_models(
                    db, _as_uuid(project_id) or project_id, version
                )
            except Exception:
                pass
            if header is not None:
                node_count = sum(
                    sum(
                        int(value)
                        for value in spine_stats(source)["node_count"].values()
                    )
                    for source in source_structures
                ) or 1
                await knowledge_repo.set_knowledge_model_status(
                    db,
                    _get(header, "id", version),
                    "ready",
                    node_count=node_count,
                )
            await db.flush()
        return SpineBuildResult(
            structure=merged,
            source_structures=source_structures,
            source_ids=source_ids,
            source_fingerprint=fingerprint,
            model_version=version,
            root_ids=root_ids,
            chunks=chunks,
            reused=False,
            model_header_id=header_id,
        )
    except Exception as exc:
        if header is not None:
            try:
                await knowledge_repo.set_knowledge_model_status(
                    db, _get(header, "id", None), "failed", error=str(exc)
                )
            except Exception:
                pass
        raise


# ---------------------------------------------------------------------------
# Topic evidence and live/deterministic note generation
# ---------------------------------------------------------------------------


def _context_for_rows(rows: Sequence[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for row in rows:
        cid = str(row.get("id") or row.get("chunk_id") or "")
        if not cid:
            continue
        page = row.get("page")
        locator = f" page={page}" if page is not None else ""
        blocks.append(
            f"[source chunk_id={cid}{locator}]\n"
            f"{row.get('text') or row.get('content') or ''}\n"
            f"[/source chunk_id={cid}]"
        )
    return "\n\n---\n\n".join(blocks)


def _topic_rows_from_structure(
    topic: ExtractedTopic,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return select_topic_chunks(list(rows), topic)


def _topic_from_structure(
    structure: DocumentStructure,
    unit_title: str,
    topic_title: str,
) -> ExtractedTopic:
    for unit in structure.units:
        if unit.title != unit_title:
            continue
        for topic in unit.topics:
            if topic.title == topic_title:
                return topic
    raise KeyError(f"topic {topic_title!r} is not present in the spine")


def _payload_to_topic_notes(
    payload: Any,
    unit_title: str,
    topic: ExtractedTopic,
    provenance_note: str,
) -> TopicNotes:
    # The spine is authoritative for the unit/topic/heading labels.  The
    # model may echo a paraphrase (or a wrong label) back; it is kept in
    # metadata only, never used as the persisted title, so section labels
    # always come from the spine.
    if isinstance(payload, TopicNotes):
        raw = payload.model_dump(mode="json")
    else:
        raw = dict(payload or {})
    citations = []
    for cite in raw.get("citations", raw.get("cites", [])) or []:
        if isinstance(cite, dict):
            citations.append({
                "chunk_id": str(cite.get("chunk_id") or ""),
                "span": str(cite.get("span") or ""),
            })
        elif hasattr(cite, "chunk_id"):
            citations.append({
                "chunk_id": str(cite.chunk_id),
                "span": str(getattr(cite, "span", "") or ""),
            })
    body = raw.get("body_paragraphs") or raw.get("paragraphs") or [raw.get("content") or ""]
    body = [str(value) for value in body if str(value).strip()]
    no_source = bool(raw.get("no_source", not citations or not body))
    shape_extras = dict(raw.get("shape_extras") or {})
    llm_unit = str(raw.get("unit_title") or "")
    llm_topic = str(raw.get("topic_title") or "")
    llm_heading = str(raw.get("heading") or "")
    if llm_unit and llm_unit != unit_title:
        shape_extras["llm_unit_title"] = llm_unit
    if llm_topic and llm_topic != topic.title:
        shape_extras["llm_topic_title"] = llm_topic
    if llm_heading and llm_heading != topic.title:
        shape_extras["llm_heading"] = llm_heading
    return TopicNotes(
        unit_title=unit_title,
        topic_title=topic.title,
        heading=topic.title,
        body_paragraphs=[] if no_source else body,
        key_terms=list(raw.get("key_terms") or []),
        shape=raw.get("shape") or "concept",
        shape_extras=shape_extras,
        citations=[] if no_source else citations,
        no_source=no_source,
        sufficiency=str(raw.get("sufficiency") or (
            SufficiencyStatus.MISSING.value if no_source else SufficiencyStatus.SUFFICIENT.value
        )),
        source_gap_reason=raw.get("source_gap_reason") or (
            None if not no_source else "generation returned no cited prose"
        ),
        provenance_note=str(raw.get("provenance_note") or provenance_note),
    )


async def retrieve_topic_rows(
    db: Any,
    project_id: Any,
    structure: DocumentStructure,
    unit_title: str,
    topic: ExtractedTopic,
    all_rows: Sequence[Dict[str, Any]],
    use_retrieval: bool = True,
    rerank: Optional[Callable[..., Any]] = None,
) -> List[Dict[str, Any]]:
    """Use the existing hybrid/rerank retrieval seam, then source-sort it."""
    selected = _topic_rows_from_structure(topic, all_rows)
    unit_chunk_ids: List[str] = []
    for unit in structure.units:
        if unit.title == unit_title:
            unit_chunk_ids = [str(cid) for other in unit.topics for cid in (other.chunks or [])]
            break
    if not use_retrieval or db is None:
        recovered = recover_topic_evidence(
            all_rows, topic, unit_title=unit_title,
            unit_chunk_ids=unit_chunk_ids, aliases=list(topic.aliases or []),
            rerank=rerank,
        )
        if not selected:
            selected = recovered
        return selected
    if selected and len(selected) >= 2:
        # The spine already carries this topic's source chunks in source
        # order.  A per-topic embedding + rerank pass is redundant, and on
        # a CPU worker it dominates the whole notes run.  Only widen the
        # evidence set when the spine is thin.
        return selected
    semantic_scores: Dict[str, float] = {}
    try:
        from services.retrieval.service import retrieval_service

        result = await retrieval_service.retrieve_topic_evidence(
            topic.title,
            db,
            project_id=str(project_id),
            aliases=list(topic.aliases or []),
            extra_titles=[unit_title] + [s.title for s in topic.subtopics or []],
            enable_rerank=True,
        )
        retrieved = [chunk_to_dict(chunk) for chunk in (result.chunks or [])]
        for row in retrieved:
            try:
                semantic_scores[str(row.get("id") or row.get("chunk_id"))] = float(
                    row.get("similarity", row.get("score", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                pass
        by_id = {str(row.get("id") or row.get("chunk_id")): row for row in selected}
        for row in retrieved:
            by_id.setdefault(str(row.get("id") or row.get("chunk_id")), row)
        selected = list(by_id.values())
    except Exception as exc:
        # Retrieval is an evidence enhancement, not a replacement for the
        # source-order structure.  A provider/index outage must not silently
        # change ordering; fall back to the claimed chunks and log the reason.
        logger.info("topic retrieval fallback for %s: %s", topic.title, exc)
    sufficiency = assess_source_sufficiency(
        topic, selected, all_chunks=all_rows, unit_title=unit_title,
        unit_chunk_ids=unit_chunk_ids, aliases=list(topic.aliases or []),
        semantic_scores=semantic_scores, rerank=rerank,
    )
    if sufficiency.status is not SufficiencyStatus.SUFFICIENT:
        recovered = recover_topic_evidence(
            all_rows, topic, unit_title=unit_title,
            unit_chunk_ids=unit_chunk_ids, aliases=list(topic.aliases or []),
            semantic_scores=semantic_scores, rerank=rerank,
        )
        by_id = {str(row.get("id") or row.get("chunk_id")): row for row in selected}
        for row in recovered:
            by_id.setdefault(str(row.get("id") or row.get("chunk_id")), row)
        selected = list(by_id.values())
    return sorted(
        selected,
        key=lambda row: (
            int(row.get("page") is None),
            int(row.get("page") or 0),
            int(row.get("position") or 0),
            str(row.get("id") or row.get("chunk_id") or ""),
        ),
    )


async def generate_topic_live_or_deterministic(
    db: Any,
    project_id: Any,
    engine: Any,
    structure: DocumentStructure,
    unit_title: str,
    topic: ExtractedTopic,
    all_rows: Sequence[Dict[str, Any]],
    use_live: bool = True,
    use_retrieval: bool = True,
    max_output_tokens: Optional[int] = None,
    batch_tokens: Optional[int] = None,
) -> TopicNotes:
    """Generate one topic through the shared structured path when requested.

    The topic's evidence is split into source-order batches
    (``split_topic_evidence``) and each batch is generated through the shared
    NOTES_TOPIC structured call with a per-topic output budget, then merged.
    One oversized call used to be cut off mid-JSON, which failed the whole
    job; bounded batches plus the per-call budget keep every answer inside
    the output ceiling.
    """
    # Locked DB region: topic-evidence retrieval reads through the shared
    # AsyncSession, so it must never overlap another topic's session use.  The
    # LLM calls below are intentionally left unlocked.
    async with _db_lock():
        evidence = await retrieve_topic_rows(
            db,
            project_id,
            structure,
            unit_title,
            topic,
            all_rows,
            use_retrieval=use_retrieval,
        )
    if not evidence:
        return render_topic_notes(unit_title, topic, [])
    if not use_live or engine is None:
        return render_topic_notes(unit_title, topic, evidence)
    from services.academic.notes_hier import (
        generate_topic_notes,
        merge_topic_payloads,
        split_topic_evidence,
    )

    allowed = {str(row.get("id") or row.get("chunk_id")) for row in evidence}
    batches = split_topic_evidence(
        evidence,
        int(batch_tokens) if batch_tokens else DEFAULT_TOPIC_BATCH_TOKENS,
    )
    payloads: List[Dict[str, Any]] = []
    for batch in batches:
        payloads.append(
            await generate_topic_notes(
                engine,
                unit_title,
                topic,
                _context_for_rows(batch),
                allowed,
                query=topic_query(unit_title, topic),
                max_tokens=int(max_output_tokens) if max_output_tokens else None,
            )
        )
    payload = merge_topic_payloads(payloads) if len(payloads) > 1 else payloads[0]
    return _payload_to_topic_notes(
        payload,
        unit_title,
        topic,
        provenance_note=(
            "shared NOTES_TOPIC structured generation over "
            f"{len(batches)} batch(es); citations limited to this topic prompt"
        ),
    )


def _honest_gap_note(
    unit_title: str,
    topic: ExtractedTopic,
    reason: str,
    provenance_note: str,
) -> TopicNotes:
    """The single honest gap shape used whenever a topic has no source prose.

    Same fields as the schema-failure gap: no citations, no_source=True, an
    INSUFFICIENT sufficiency verdict, and a named reason -- never a silent
    empty note and never a fabricated one.
    """
    return TopicNotes(
        unit_title=unit_title,
        topic_title=topic.title,
        heading=topic.title,
        body_paragraphs=[],
        key_terms=[],
        shape=TopicShape.CONCEPT,
        shape_extras={},
        citations=[],
        no_source=True,
        sufficiency=SufficiencyStatus.INSUFFICIENT.value,
        source_gap_reason=reason,
        provenance_note=provenance_note,
    )


async def _generate_topic_with_honest_error(
    db: Any,
    project_id: Any,
    engine: Any,
    structure: DocumentStructure,
    unit_title: str,
    topic: ExtractedTopic,
    all_rows: Sequence[Dict[str, Any]],
    use_live: bool = True,
    use_retrieval: bool = True,
    max_output_tokens: Optional[int] = None,
    batch_tokens: Optional[int] = None,
) -> TopicNotes:
    """One topic, with a failure message that names the real cause.

    A cut-off provider answer used to surface as a bare
    "LLM output JSON was truncated" with no topic context. The topic path is
    now: bounded evidence batches + a per-topic output budget + bounded
    truncated-JSON recovery inside the structured helper. If it still fails,
    the job fails with the topic and the reason - never a silent fallback to
    headings-only notes.
    """
    try:
        return await generate_topic_live_or_deterministic(
            db,
            project_id,
            engine,
            structure,
            unit_title,
            topic,
            all_rows,
            use_live=use_live,
            use_retrieval=use_retrieval,
            max_output_tokens=max_output_tokens,
            batch_tokens=batch_tokens,
        )
    except AcademicPipelineError:
        raise
    except ValidationError as exc:
        # Schema validation failed even after coerce_enum_errors: return an
        # honest gap TopicNotes instead of propagating a hard failure.
        from services.strategies.llm.structured import coerce_enum_errors
        payload = exc.errors() or [{}]
        first = payload[0]
        source_gap_reason = (
            "schema validation failed: "
            + str(first.get("loc"))
            + ": "
            + str(first.get("msg"))
        )
        warnings = []
        warnings.append({"topic": topic.title, "reason": source_gap_reason})
        return _honest_gap_note(
            unit_title, topic, source_gap_reason, "schema validation failed"
        )
    except Exception as exc:
        raise AcademicPipelineError(
            f"topic {topic.title!r} ({unit_title}) could not be generated: {exc}"
        ) from exc


def notes_config_identity(config: Any) -> Dict[str, Any]:
    """Stable notes-config identity used in the artifact fingerprint (pure).

    Advisory recommendation metadata is dropped: it never changes the
    generated content, so it must not create a new artifact version.
    """
    payload: Dict[str, Any] = {}
    if config is None:
        return payload
    if hasattr(config, "persisted_dict"):
        payload = dict(config.persisted_dict())
    elif isinstance(config, dict):
        payload = dict(config)
    payload.pop("recommendation", None)
    return {key: payload[key] for key in sorted(payload)}


async def _repair_notes_async(
    structure: DocumentStructure,
    topic_notes: List[TopicNotes],
    chunk_index: Dict[str, Dict[str, Any]],
    regen: Callable[[ExtractedTopic, List[Dict[str, Any]]], Any],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    max_passes: int = MAX_REPAIR_PASSES,
) -> Tuple[List[TopicNotes], Any, List[RepairRecord]]:
    """Async adapter for the existing bounded repair contract.

    The pure ``coverage.repair_notes`` helper remains the reference
    implementation for offline callers.  The worker needs an async callback
    for the live provider, so this adapter preserves its exact two-pass,
    uncovered-only, and revert-on-regression rules without duplicating any
    other repair policy.
    """
    current = list(topic_notes)
    report = evaluate_coverage(structure, current, threshold=threshold)
    log: List[RepairRecord] = []
    topic_by_key = {
        (unit.title, topic.title): topic
        for unit in structure.units
        for topic in unit.topics
    }
    for pass_no in range(1, max(0, int(max_passes)) + 1):
        if report.passes:
            break
        uncovered_keys = [
            (score.unit_title, score.topic_title)
            for score in report.per_topic
            if score.score < 1.0
        ]
        if not uncovered_keys:
            break
        before = report.concept_coverage
        candidate = list(current)
        positions = {
            (note.unit_title, note.topic_title): index
            for index, note in enumerate(candidate)
        }
        regenerated: List[str] = []
        for key in uncovered_keys:
            topic = topic_by_key.get(key)
            if topic is None:
                continue
            score = next(
                (item for item in report.per_topic
                 if (item.unit_title, item.topic_title) == key),
                None,
            )
            evidence = missing_evidence(topic, score, chunk_index) if score else []
            fresh = await regen(topic, evidence)
            if hasattr(fresh, "model_dump"):
                fresh = TopicNotes.model_validate(fresh.model_dump())
            regenerated.append(topic.title)
            if key in positions:
                candidate[positions[key]] = fresh
            else:
                candidate.append(fresh)
        new_report = evaluate_coverage(structure, candidate, threshold=threshold)
        kept = new_report.concept_coverage >= before
        if kept:
            current, report = candidate, new_report
        log.append(RepairRecord(
            pass_no=pass_no,
            regenerated_topics=regenerated,
            coverage_before=before,
            coverage_after=new_report.concept_coverage,
            kept=kept,
            note=(
                "kept: coverage improved or held"
                if kept
                else "reverted: repair lowered net coverage"
            ),
        ))
        if kept and new_report.passes:
            break
    return current, report, log


async def _generate_topic_notes_bounded(
    groups: Sequence[Mapping[str, Any]],
    structure: DocumentStructure,
    generate: Callable[[str, ExtractedTopic], Awaitable[TopicNotes]],
    *,
    timeout_s: float,
    concurrency: int,
    existing: Sequence[TopicNotes] = (),
    pending_indexes: Optional[Sequence[int]] = None,
    on_complete: Optional[Callable[[int, TopicNotes], Awaitable[None]]] = None,
) -> Tuple[List[TopicNotes], List[Dict[str, Any]]]:
    """Generate pending topics with a bounded pool, preserving spine order.

    Every per-topic call is bounded by ``timeout_s``: a timeout becomes an
    honest gap ``TopicNotes`` plus a ``gaps`` entry instead of hanging the
    job.  At most ``concurrency`` calls are in flight, and the returned list
    is rebuilt by iterating the structure in spine order, so synthesis order
    is independent of completion order.
    """
    notes_by_key: Dict[Tuple[str, str], TopicNotes] = {
        (note.unit_title, note.topic_title): note for note in existing
    }
    gaps: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    indexes = (
        list(pending_indexes)
        if pending_indexes is not None
        else list(range(len(groups)))
    )

    async def _one(index: int) -> Tuple[int, TopicNotes]:
        group = groups[index]
        async with semaphore:
            topic = _topic_from_structure(
                structure, str(group["unit_title"]), str(group["topic_title"])
            )
            try:
                note = await asyncio.wait_for(
                    generate(str(group["unit_title"]), topic),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                reason = f"generation timed out after {timeout_s:.0f}s"
                gaps.append({"topic": topic.title, "reason": reason})
                note = _honest_gap_note(
                    str(group["unit_title"]), topic, reason, "generation timed out"
                )
            return index, note

    tasks = [asyncio.ensure_future(_one(index)) for index in indexes]
    try:
        for future in asyncio.as_completed(tasks):
            index, note = await future
            notes_by_key[(note.unit_title, note.topic_title)] = note
            if on_complete is not None:
                await on_complete(index, note)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    ordered = [
        notes_by_key[(unit.title, topic.title)]
        for unit in structure.units
        for topic in unit.topics
        if (unit.title, topic.title) in notes_by_key
    ]
    return ordered, gaps


# ---------------------------------------------------------------------------
# Resumable job handler
# ---------------------------------------------------------------------------


def _stage_fraction(stage: str, fraction: float) -> int:
    lo, hi = NOTES_HIERARCHICAL_STAGE_PROGRESS_RANGES[stage]
    return int(lo + (hi - lo) * max(0.0, min(1.0, float(fraction))))


def _safe_json(value: Any) -> Any:
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if hasattr(value, "model_dump"):
        return _safe_json(value.model_dump(mode="json"))
    return value


def _state(job: Any) -> Dict[str, Any]:
    output = dict(_get(job, "output_data", None) or {})
    return dict(output.get("academic") or {})


async def _install_job_provider_selection(
    db: Any,
    project_id: Any,
    data: Dict[str, Any],
) -> Any:
    """Rehydrate a worker's request-scoped provider without storing a key."""

    user_id = _as_uuid(data.get("user_id"))
    if user_id is None:
        return None
    try:
        user = await repositories.get_user_by_id(db, user_id)
        if user is None:
            return None
        from services.provider_credentials.override import resolve_for_request

        return await resolve_for_request(db, user, project_id)
    except Exception as exc:
        logger.info("academic provider selection unavailable: %s", exc)
        return None


async def _report(
    db: Any,
    job: Any,
    stage: str,
    fraction: float,
    message: str,
    state: Dict[str, Any],
    done: Set[str],
) -> None:
    progress = _stage_fraction(stage, fraction)
    previous = int(_get(job, "progress", 0) or 0)
    progress = max(previous, max(0, min(100, progress)))
    output = dict(_get(job, "output_data", None) or {})
    lifecycle = list(output.get("_lifecycle") or [])
    if stage == "coverage" and "validating" not in lifecycle:
        lifecycle.append("validating")
    if stage == "repair" and "repairing" not in lifecycle:
        lifecycle.append("repairing")
    if lifecycle:
        output["_lifecycle"] = lifecycle
    output["_stage"] = stage
    output["_stage_message"] = message
    output["_stages_done"] = sorted(done)
    output["academic"] = _safe_json(state)
    job.output_data = output
    # Keep the in-memory row monotonic as well as the persisted row. This is
    # important for test doubles and for a worker whose next report is issued
    # before a session refresh.
    job.progress = progress
    job.stage = stage
    job.stage_message = message
    try:
        from services.jobs import queue as job_queue

        # The collecting loop commits once per completed topic while the pool
        # still has tasks running; take the same lock as the per-topic
        # retrieval so a commit can never overlap a task's session use.
        async with _db_lock():
            await job_queue.report_progress(
                db, job.id, progress, stage=stage, stage_message=message
            )
            await db.commit()
    except Exception as exc:
        logger.warning("academic progress commit failed for %s: %s", job.id, exc)
        try:
            async with _db_lock():
                await db.rollback()
        except Exception:
            pass


def _completed_stages(job: Any) -> Set[str]:
    output = dict(_get(job, "output_data", None) or {})
    done = {str(value) for value in output.get("_stages_done", [])}
    stage = str(_get(job, "stage", "") or "")
    if stage in NOTES_HIERARCHICAL_STAGES:
        # A worker can die while the current stage is in flight.  Only stages
        # strictly before it are safe to infer as complete; the explicit
        # _stages_done ledger is the authority for the current stage.
        done.update(NOTES_HIERARCHICAL_STAGES[: NOTES_HIERARCHICAL_STAGES.index(stage)])
    return done


def _root_for_state(state: Dict[str, Any]) -> Optional[Any]:
    roots = state.get("root_ids") or []
    if roots:
        return roots[0]
    return None


async def _persist_compat_notes(
    db: Any,
    project_id: Any,
    payload: Dict[str, Any],
    title: str,
    artifact_id: Optional[str] = None,
) -> Optional[Any]:
    """Write the existing notes compatibility row, idempotently when possible."""
    try:
        existing = await repositories.get_notes_by_project(db, project_id)
    except Exception:
        existing = []
    for note in existing or []:
        content = _get(note, "content", None)
        if isinstance(content, dict) and content.get("artifact_id") and content.get("artifact_id") == artifact_id:
            return note
    try:
        return await repositories.create_note(db, project_id, title, payload)
    except Exception as exc:
        logger.warning("compat notes projection failed: %s", exc)
        return None


async def _save_notes_payload(
    db: Any,
    root_id: Any,
    payload: Dict[str, Any],
) -> None:
    if root_id is None:
        return
    await knowledge_repo.save_notes_artifact(db, root_id, payload)


async def build_and_store_pdf(
    db: Any,
    project_id: Any,
    structure: DocumentStructure,
    notes_artifact: Any,
    root_id: Any,
    source_ids: Sequence[str],
    model_version: Optional[int] = None,
    chunk_pages: Optional[Dict[str, int]] = None,
    chunk_texts: Optional[Dict[str, str]] = None,
    notes_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Render, store, and commit a notes PDF artifact.

    Fingerprint matching happens inside ``build_notes_pdf_artifact`` before
    ReportLab is called.  Only a new record reaches the storage backend.

    ``notes_config`` is part of the fingerprint identity: the same sources
    rendered with different settings (mode/detail/budgets) are a different
    version, while an identical rebuild reuses the stored one.  Advisory
    recommendation metadata is excluded so a wording change in a suggestion
    never manufactures a new version.
    """
    if root_id is None:
        raise AcademicPipelineError("cannot persist PDF without a spine document root")
    existing = []
    try:
        existing = await knowledge_repo.list_generated_artifacts(
            db, root_id, academic_artifacts.NOTES_PDF_TYPE
        )
    except Exception:
        existing = []
    config = {
        "project_id": str(project_id),
        "source_ids": list(source_ids),
        "model_version": model_version,
        "source_fingerprint": getattr(structure, "fingerprint", None),
        "notes_config": notes_config_identity(notes_config),
    }
    record, report, reused = academic_artifacts.build_notes_pdf_artifact(
        notes_artifact,
        structure,
        model="academic-notes-hierarchical",
        config=config,
        existing_versions=existing,
        chunk_pages=chunk_pages or {},
        chunk_texts=chunk_texts,
        allowed_chunk_ids=set(chunk_texts) if chunk_texts else None,
    )
    if reused:
        return {**record, "reused": True, "pdf_report": report}

    backend = None
    try:
        from services.storage import get_default_backend

        # StorageBackend requires {user_id}/{project_id}/{generated_name}.
        # The pure artifact builder returns only a display filename, so scope
        # it here immediately before the write. Previously this bare filename
        # was rejected and the job retried at the lock timeout forever.
        pid = _as_uuid(project_id) or project_id
        project = await repositories.get_project_by_id(db, pid)
        if project is None or not _get(project, "user_id", None):
            raise AcademicPipelineError("PDF storage failed: project owner is missing")
        safe_name = PurePosixPath(str(record.get("path") or "")).name
        if not safe_name:
            raise AcademicPipelineError("PDF storage failed: generated filename is empty")
        scoped_hint = f"{_get(project, 'user_id')}/{pid}/{safe_name}"
        backend = get_default_backend()
        stored_key = await backend.save(scoped_hint, report.pdf_bytes)
        record["path"] = str(stored_key or scoped_hint)
        record["sha256"] = report.sha256
        record["hash"] = report.sha256
    except Exception as exc:
        # Do not commit an artifact claiming bytes that are not in storage.
        raise AcademicPipelineError(f"PDF storage failed: {exc}") from exc
    finally:
        # ``backend`` is intentionally kept in this scope for test doubles and
        # future streaming writers; bytes are never put in the database.
        _ = backend

    await knowledge_repo.save_generated_artifact(db, root_id, record)
    return {**record, "reused": False, "pdf_report": report}


async def _record_hierarchical_failure(
    ctx: Any, job_id: Any, error: str
) -> None:
    """Persist one terminal failure even after the handler transaction failed."""
    from services.jobs import queue as job_queue

    db = ctx.db
    try:
        await db.rollback()
    except Exception:
        pass

    session_factory = getattr(ctx, "session_factory", None)
    if session_factory is not None:
        async with session_factory() as fail_db:
            try:
                from database.models.job import Job
                failed_row = await fail_db.get(Job, job_id)
                if failed_row is not None:
                    failed_output = dict(getattr(failed_row, "output_data", None) or {})
                    failed_output["_lifecycle"] = list(failed_output.get("_lifecycle") or []) + ["failed"]
                    failed_row.output_data = failed_output
            except Exception:
                pass
            await job_queue.fail(fail_db, job_id, error, retry=False)
            await fail_db.commit()
        return

    # Inline/dev callers do not provide a factory. Expire the identity-map row
    # so q.fail loads the persisted state instead of reusing stale attributes.
    expire_all = getattr(db, "expire_all", None)
    if callable(expire_all):
        expire_all()
    try:
        from database.models.job import Job
        failed_row = await db.get(Job, job_id)
        if failed_row is not None:
            failed_output = dict(getattr(failed_row, "output_data", None) or {})
            failed_output["_lifecycle"] = list(failed_output.get("_lifecycle") or []) + ["failed"]
            failed_row.output_data = failed_output
    except Exception:
        pass
    await job_queue.fail(db, job_id, error, retry=False)
    await db.commit()


async def process_hierarchical_job(ctx: Any, job: Any) -> None:
    """Worker entry point for the resumable hierarchical notes job."""
    from services.jobs import queue as job_queue

    db = ctx.db
    job_id = _get(job, "id", None)
    data = dict(_get(job, "input_data", None) or {})
    project_id = data.get("project_id") or _get(job, "project_id", None)
    document_ids = list(data.get("document_ids") or [])
    threshold = float(data.get("threshold", DEFAULT_COVERAGE_THRESHOLD))
    use_live = bool(data.get("use_live", True))
    use_retrieval = bool(data.get("use_retrieval", True))
    provider_token = await _install_job_provider_selection(db, project_id, data)
    try:
        notes_config = NotesGenerationConfig.model_validate(data.get("config") or {})
    except Exception:
        notes_config = NotesGenerationConfig()
    done = _completed_stages(job)
    state = _state(job)

    try:
        if not project_id:
            raise AcademicPipelineError("hierarchical notes job has no project_id")

        # -- model -------------------------------------------------------
        if "model" not in done:
            result = await build_project_spine(
                db,
                project_id,
                document_ids=document_ids,
                project_title=str(data.get("project_title") or "Academic project"),
                force=bool(data.get("force", False)),
            )
            merged_structure = merge_adjacent_topics(
                result.structure, _max_topics_per_job()
            )
            result.structure = merged_structure
            state.update({
                "structure": merged_structure.model_dump(mode="json"),
                "source_structures": [
                    source.model_dump(mode="json") for source in result.source_structures
                ],
                "source_ids": result.source_ids,
                "source_fingerprint": result.source_fingerprint,
                "model_version": result.model_version,
                "root_ids": [str(value) for value in result.root_ids],
                "spine_reused": result.reused,
                "config": notes_config.persisted_dict(),
            })
            done.add("model")
            await _report(db, job, "model", 1.0, "Spine extracted and persisted", state, done)
        else:
            if not state.get("structure"):
                raise AcademicPipelineError("resume state has no persisted spine")
            result = None

        structure = DocumentStructure.model_validate(state["structure"])
        # Fix 4: a document that resolves to many tiny topics would otherwise
        # cost one bounded LLM call per label.  Merge adjacent topics in
        # source order into page-window topics before any generation so a
        # slow provider is asked for fewer calls.  Deterministic and
        # idempotent, so a resumed job converges on the same plan.
        merged_structure = merge_adjacent_topics(structure, _max_topics_per_job())
        if merged_structure is not structure:
            # A resume of a pre-merge job stored an unmerged structure, so
            # every derived stage no longer matches.  Rebuild the whole
            # downstream plan from the merged spine instead of indexing
            # topics that no longer exist.
            structure = merged_structure
            for stage in (
                "outline", "group", "per_group", "coverage", "repair",
                "synthesis", "validate", "persist", "pdf",
            ):
                done.discard(stage)
            for key in (
                "outline", "groups", "completed_group_indexes", "topic_notes",
                "coverage", "repair_log", "notes_artifact", "validation",
                "notes_artifact_id", "pdf_artifact_id", "pdf_path",
                "pdf_version", "pdf_pages", "pdf_ready",
                "gaps", "failed_topic_count", "provider_slow",
            ):
                state.pop(key, None)
        state.setdefault("config", notes_config.persisted_dict())
        if result is not None:
            chunks = result.chunks
        else:
            chunks, _ = await build_source_structures(db, project_id, document_ids)
        chunk_index = {str(row.get("id") or row.get("chunk_id")): row for row in chunks}
        root_id = _root_for_state(state)

        # -- outline -----------------------------------------------------
        if "outline" not in done:
            state["outline"] = build_outline(structure)
            done.add("outline")
            await _report(db, job, "outline", 1.0, f"Outline built ({len(state['outline'])} topics)", state, done)

        # -- grouping ----------------------------------------------------
        if "group" not in done:
            sizes: Dict[str, int] = {}
            for unit in structure.units:
                for topic in unit.topics:
                    sizes[topic.title] = sum(
                        len(str(row.get("text") or "")) // 4
                        for row in _topic_rows_from_structure(topic, chunks)
                    )
            state["groups"] = group_topics(
                state.get("outline") or [], sizes,
                budget_tokens=int(notes_config.topic_budget_tokens),
            )
            state["config"] = notes_config.persisted_dict()
            done.add("group")
            await _report(db, job, "group", 1.0, f"Topic groups planned ({len(state['groups'])})", state, done)

        groups = list(state.get("groups") or [])
        topic_notes: List[Dict[str, Any]] = list(state.get("topic_notes") or [])
        completed_group_indexes = {
            int(value) for value in state.get("completed_group_indexes", [])
            if str(value).isdigit()
        }
        gaps: List[Dict[str, Any]] = list(state.get("gaps") or [])
        engine = None
        if use_live:
            try:
                from services.transformation.engine import transformation_engine

                engine = transformation_engine
            except Exception:
                engine = None

        # -- per-group map ----------------------------------------------
        # One topic is the largest unit asked of the model. Its evidence is
        # split into bounded batches and each batch gets the per-topic output
        # budget, so no single call can be cut off mid-JSON.
        topic_output_tokens = int(
            notes_config.topic_output_tokens or NOTES_TOPIC_MAX_OUTPUT_TOKENS
        )
        topic_batch_tokens = int(
            notes_config.topic_batch_tokens
            or max(1, int(notes_config.topic_budget_tokens) // 2)
            or DEFAULT_TOPIC_BATCH_TOKENS
        )
        # Per-topic bound shared by the map step and the repair callback so a
        # slow provider call can never hold the job open, whichever stage runs.
        timeout_s = float(notes_config.topic_timeout_s or _topic_timeout_s())
        if "per_group" not in done:
            concurrency = int(
                notes_config.topic_concurrency or _topic_concurrency()
            )
            existing_notes: List[TopicNotes] = []
            for value in topic_notes:
                try:
                    existing_notes.append(TopicNotes.model_validate(value))
                except Exception:
                    continue
            pending_indexes = [
                index for index in range(len(groups))
                if index not in completed_group_indexes
            ]
            reported = len(completed_group_indexes)

            async def _generate(unit_title: str, topic: ExtractedTopic) -> TopicNotes:
                return await _generate_topic_with_honest_error(
                    db,
                    project_id,
                    engine,
                    structure,
                    unit_title,
                    topic,
                    chunks,
                    use_live=use_live,
                    use_retrieval=use_retrieval,
                    max_output_tokens=topic_output_tokens,
                    batch_tokens=topic_batch_tokens,
                )

            async def _on_complete(index: int, note: TopicNotes) -> None:
                nonlocal reported
                completed_group_indexes.add(index)
                reported += 1
                state["completed_group_indexes"] = sorted(completed_group_indexes)
                await _report(
                    db,
                    job,
                    "per_group",
                    reported / max(1, len(groups)),
                    f"Generated topic {reported}/{len(groups)}: {note.topic_title}",
                    state,
                    done,
                )

            ordered_notes, new_gaps = await _generate_topic_notes_bounded(
                groups,
                structure,
                _generate,
                timeout_s=timeout_s,
                concurrency=concurrency,
                existing=existing_notes,
                pending_indexes=pending_indexes,
                on_complete=_on_complete,
            )
            gaps.extend(new_gaps)
            topic_notes = [note.model_dump(mode="json") for note in ordered_notes]
            state["topic_notes"] = topic_notes
            state["gaps"] = gaps
            state["failed_topic_count"] = len(gaps)
            # Fix 3: tell the read path honestly when the provider was slow
            # enough that some topics hit the per-topic bound, so the UI can
            # say so instead of presenting a partial run as a clean one.
            state["provider_slow"] = _gaps_provider_slow(gaps)
            done.add("per_group")
            await _report(db, job, "per_group", 1.0, "All topic groups generated", state, done)

        notes_models = [TopicNotes.model_validate(value) for value in topic_notes]

        # -- coverage ----------------------------------------------------
        if "coverage" not in done:
            report = evaluate_coverage(structure, notes_models, threshold=threshold)
            state["coverage"] = report.model_dump(mode="json")
            done.add("coverage")
            await _report(db, job, "coverage", 1.0, "Coverage measured", state, done)
        else:
            from services.academic.types import CoverageReport

            report = CoverageReport.model_validate(state["coverage"])

        # -- bounded repair ---------------------------------------------
        repair_log: List[Any] = []
        if "repair" not in done:
            if report.passes:
                repair_log: List[Any] = []
            else:
                async def _regenerate(topic: ExtractedTopic, evidence: List[Dict[str, Any]]) -> TopicNotes:
                    # The existing repair helper is synchronous; run the live
                    # structured call in this worker and return its typed
                    # value.  Only uncovered topics reach this callback.
                    unit = next(
                        (u for u in structure.units if topic in u.topics), None
                    )
                    unit_title = unit.title if unit is not None else "Unit"
                    rows = evidence or _topic_rows_from_structure(topic, chunks)
                    fake_structure = structure
                    try:
                        return await asyncio.wait_for(
                            _generate_topic_with_honest_error(
                                db,
                                project_id,
                                engine,
                                fake_structure,
                                unit_title,
                                topic,
                                rows,
                                use_live=use_live,
                                use_retrieval=use_retrieval,
                                max_output_tokens=topic_output_tokens,
                                batch_tokens=topic_batch_tokens,
                            ),
                            timeout=timeout_s,
                        )
                    except asyncio.TimeoutError:
                        reason = f"generation timed out after {timeout_s:.0f}s"
                        return _honest_gap_note(
                            unit_title, topic, reason, "repair timed out"
                        )

                # The adapter preserves the existing two-pass bound and
                # revert rule while allowing the live callback to be async.
                notes_models, report, repair_log = await _repair_notes_async(
                    structure,
                    notes_models,
                    chunk_index,
                    _regenerate,
                    threshold,
                    MAX_REPAIR_PASSES,
                )
            state["topic_notes"] = [note.model_dump(mode="json") for note in notes_models]
            state["coverage"] = report.model_dump(mode="json")
            state["repair_log"] = [item.model_dump(mode="json") for item in repair_log]
            done.add("repair")
            await _report(
                db,
                job,
                "repair",
                1.0,
                f"Coverage repair complete ({len(repair_log)} pass(es))",
                state,
                done,
            )

        # -- synthesis ---------------------------------------------------
        if "synthesis" not in done:
            artifact_title = str(
                data.get("project_title") or "Study Notes"
            ).strip() or "Study Notes"
            synthesized = synthesize_notes(
                artifact_title,
                structure,
                notes_models,
                chunk_texts={cid: str(row.get("text") or "") for cid, row in chunk_index.items()},
            )
            synthesized = attach_report(synthesized, report, repair_log)
            synthesized = synthesized.model_copy(update={
                "config": dict(state.get("config") or notes_config.persisted_dict()),
                "observability": {
                    "job_type": JOB_TYPE,
                    "project_id": str(project_id),
                    "document_ids": list(document_ids),
                    "mode": str(notes_config.mode.value),
                    "detail": str(notes_config.detail.value),
                    "topic_count": sum(len(unit.topics) for unit in structure.units),
                    "section_count": len(synthesized.sections),
                    "citation_count": sum(len(section.citations) for section in synthesized.sections),
                    "coverage": report.model_dump(mode="json"),
                    "repair_attempts": len(repair_log),
                    "word_count": sum(len(section.content.split()) for section in synthesized.sections),
                    # Truncation guard rails: evidence per model call and the
                    # per-call output budget behind every topic answer.
                    "topic_batch_tokens": topic_batch_tokens,
                    "topic_output_tokens": topic_output_tokens,
                },
            })
            state["notes_artifact"] = synthesized.model_dump(mode="json")
            done.add("synthesis")
            await _report(db, job, "synthesis", 1.0, "Notes synthesized in spine order", state, done)
        else:
            synthesized = NotesArtifact.model_validate(state["notes_artifact"])

        # -- validation --------------------------------------------------
        if "validate" not in done:
            # This is the same gate used by the PDF renderer and tests.  A
            # heading-only or boilerplate artifact is never persisted.
            #
            # ADR-008 T006: forward the cited chunks' structure verdict so a
            # placeholder spine label ("Pages 12-14") degrades to a warning
            # instead of a hard topic_mismatch.  The payload is refreshed so
            # the verdict travels with the persisted notes as well.
            if _forward_structure_verdicts(synthesized.sections, chunk_index):
                state["notes_artifact"] = synthesized.model_dump(mode="json")
            allowed_ids = set(chunk_index)
            chunk_texts = {
                str(cid): str(row.get("text") or row.get("content") or "")
                for cid, row in chunk_index.items()
            }
            validation = validate_notes_artifact(
                synthesized, structure, allowed_ids, chunk_texts,
                min_chars=MIN_SECTION_CHARS, allow_source_gaps=True,
            )
            if not validation.ok:
                codes = ", ".join(str(item.get("code")) for item in validation.errors[:8])
                raise AcademicPipelineError(
                    "validation failed: synthesized notes failed hard gates: " + codes
                )
            state["validation"] = validation.as_dict()
            state["validation"]["grounding"] = "shared_validate_grounding"
            done.add("validate")
            await _report(db, job, "validate", 1.0, "Citations, coverage, order, and grounding validated", state, done)

        # -- persist -----------------------------------------------------
        if "persist" not in done:
            notes_payload = dict(state.get("notes_artifact") or synthesized.model_dump(mode="json"))
            notes_payload["model_version"] = state.get("model_version")
            notes_payload["source_fingerprint"] = state.get("source_fingerprint")
            # Run-level partial-success metadata travels with the artifact so
            # a later read can report honest gaps/provider_slow without the
            # job row.
            notes_payload["gaps"] = list(gaps)
            notes_payload["failed_topic_count"] = len(gaps)
            notes_payload["provider_slow"] = bool(state.get("provider_slow") or False)
            await _save_notes_payload(db, root_id, notes_payload)
            state["notes_artifact_id"] = str(root_id) if root_id is not None else None
            done.add("persist")
            await _report(db, job, "persist", 1.0, "Notes artifact and compatibility row persisted", state, done)

        # -- PDF + final artifact ---------------------------------------
        if "pdf" not in done:
            chunk_pages = {
                cid: int(row.get("page"))
                for cid, row in chunk_index.items()
                if row.get("page") is not None
            }
            pdf_record = await build_and_store_pdf(
                db,
                project_id,
                structure,
                synthesized,
                root_id,
                state.get("source_ids") or [],
                model_version=int(state.get("model_version") or 1),
                chunk_pages=chunk_pages,
                chunk_texts={
                    str(cid): str(row.get("text") or row.get("content") or "")
                    for cid, row in chunk_index.items()
                },
                notes_config=notes_config.persisted_dict(),
            )
            state["pdf_artifact_id"] = str(pdf_record.get("id") or "")
            state["pdf_path"] = str(pdf_record.get("path") or "")
            state["pdf_version"] = pdf_record.get("version")
            state["pdf_pages"] = pdf_record.get("pages")
            state["pdf_ready"] = True
            # Include the durable artifact identity in the compatibility
            # payload: the read list and the viewer need id + version + pages
            # to open THIS note instead of only the newest one.
            compat = dict(state.get("notes_artifact") or synthesized.model_dump(mode="json"))
            compat.update({
                "artifact_id": state["pdf_artifact_id"],
                "pdf_path": state["pdf_path"],
                "pdf_version": pdf_record.get("version"),
                "pdf_pages": pdf_record.get("pages"),
                "pdf_sha256": pdf_record.get("sha256") or pdf_record.get("hash"),
                "model": pdf_record.get("model"),
                "artifact_status": pdf_record.get("status"),
                "created_at": pdf_record.get("created_at"),
                "model_version": state.get("model_version"),
                # Partial-success metadata on the durable compatibility row.
                "gaps": list(gaps),
                "failed_topic_count": len(gaps),
                "provider_slow": bool(state.get("provider_slow") or False),
            })
            await _save_notes_payload(db, root_id, compat)
            await _persist_compat_notes(
                db,
                project_id,
                compat,
                synthesized.title,
                artifact_id=state["pdf_artifact_id"],
            )
            done.add("pdf")
            await _report(db, job, "pdf", 1.0, "PDF stored and artifact committed", state, done)

        state["state"] = "ready"
        state["gaps"] = gaps
        state["failed_topic_count"] = len(gaps)
        state.setdefault("provider_slow", _gaps_provider_slow(gaps))
        lifecycle = list(state.setdefault("lifecycle", []))
        if "completed" not in lifecycle:
            lifecycle.append("completed")
        state["lifecycle"] = lifecycle
        state["config"] = notes_config.persisted_dict()
        state.setdefault("observability", dict(state.get("notes_artifact", {}).get("observability") or {}))
        state["pdf_ready"] = True
        output = dict(_get(job, "output_data", None) or {})
        output.update({
            "academic": _safe_json(state),
            "_stage": "pdf",
            "_stages_done": sorted(done),
            "artifact_id": state.get("pdf_artifact_id"),
            "model_version": state.get("model_version"),
            "pdf_ready": True,
        })
        await job_queue.complete(db, job.id, output_data=output)
        await db.commit()
    except Exception as exc:
        logger.exception("hierarchical academic generation failed")
        try:
            await _record_hierarchical_failure(
                ctx,
                job_id,
                str(exc) or type(exc).__name__,
            )
        except Exception:
            # Let the worker boundary retry the terminal write in a clean
            # session. Swallowing this error would leave the job PROCESSING
            # until JOB_LOCK_TIMEOUT_SECONDS and recreate the same timeout.
            logger.exception("could not persist terminal notes failure")
            raise
    finally:
        if provider_token is not None:
            try:
                from services.provider_credentials.override import reset_request_llm

                reset_request_llm(provider_token)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Typed practice and metrics production entry points
# ---------------------------------------------------------------------------


async def _latest_notes_payload(db: Any, project_id: Any) -> Optional[Dict[str, Any]]:
    try:
        nodes = await knowledge_repo.list_knowledge_nodes(
            db, project_id=_as_uuid(project_id) or project_id
        )
    except Exception:
        return None
    for node in sorted(nodes or [], key=lambda n: str(_get(n, "created_at", "") or ""), reverse=True):
        try:
            payload = await knowledge_repo.get_notes_artifact(db, _get(node, "id", None))
        except Exception:
            payload = None
        if payload:
            return dict(payload)
    return None


async def build_pdf_for_project(
    db: Any,
    project_id: Any,
    document_ids: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Build a PDF from the latest persisted hierarchical notes payload."""
    chunks, source_structures = await build_source_structures(
        db, project_id, document_ids=document_ids
    )
    structure = merge_document_structures(source_structures, project_title="Academic project")
    payload = await _latest_notes_payload(db, project_id)
    if not payload:
        raise AcademicPipelineError(
            "no hierarchical notes artifact: run POST /api/generation/notes/hierarchical first"
        )
    try:
        nodes = await knowledge_repo.list_knowledge_nodes(
            db, project_id=_as_uuid(project_id) or project_id
        )
    except Exception:
        nodes = []
    root = next((node for node in nodes or [] if _get(node, "parent_id", None) is None), None)
    if root is None:
        result = await build_project_spine(
            db, project_id, document_ids=document_ids, project_title="Academic project",
            force=force,
        )
        root_id = result.root_ids[0] if result.root_ids else None
        model_version = result.model_version
        source_ids = result.source_ids
    else:
        root_id = _get(root, "id", None)
        metadata = dict(_get(root, "node_metadata", None) or {})
        model_version = int(metadata.get("model_version") or 1)
        source_ids = [str(s.doc_id) for s in source_structures]
    notes_payload = {
        key: payload[key]
        for key in NotesArtifact.model_fields
        if key in payload
    }
    notes = NotesArtifact.model_validate(notes_payload)
    record = await build_and_store_pdf(
        db,
        project_id,
        structure,
        notes,
        root_id,
        source_ids,
        model_version=model_version,
        chunk_pages={
            str(row.get("id") or row.get("chunk_id")): int(row.get("page"))
            for row in chunks if row.get("page") is not None
        },
        chunk_texts={
            str(row.get("id") or row.get("chunk_id")): str(row.get("text") or row.get("content") or "")
            for row in chunks
        },
        # The stored notes artifact carries the settings it was generated
        # with, so a rebuild fingerprints identically to the job that produced
        # it instead of manufacturing a duplicate version.
        notes_config=dict(notes.config) if getattr(notes, "config", None) else None,
    )
    return {
        "state": "ready",
        "project_id": str(project_id),
        "artifact_id": str(record.get("id") or ""),
        "version": record.get("version"),
        "path": record.get("path"),
        "pages": record.get("pages"),
        "sha256": record.get("sha256") or record.get("hash"),
        "reused": bool(record.get("reused")),
    }


async def generate_typed_flashcards(
    db: Any,
    project_id: Any,
    document_ids: Optional[Sequence[str]] = None,
    max_cards: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Build, validate, and persist a typed deck plus rich artifact.

    The state is honest: `ready` is reported only for a non-empty deck with
    no unsupported citation, and only after the cards actually reached the
    domain rows. An empty build (the spine has no card-able concept with a
    retrievable chunk), an invalid build, or a deck that would cite outside
    its evidence returns `not_generated` with a message -- no artifact, no
    deck, never a "ready" deck with zero cards.
    """
    from pydantic import ValidationError

    result = await build_project_spine(
        db, project_id, document_ids=document_ids, force=force,
        project_title="Academic project",
    )
    chunk_texts = {
        str(row.get("id") or row.get("chunk_id")): str(row.get("text") or "")
        for row in result.chunks
    }

    def _empty(reason: str, validation: Any = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "state": "not_generated",
            "project_id": str(project_id),
            "artifact_id": None,
            "deck_id": None,
            "message": reason,
        }
        if validation is not None:
            payload["validation"] = validation
        return payload

    try:
        deck = build_flashcards(
            result.structure,
            chunk_texts,
            max_cards=max_cards,
            title="Academic typed flashcards",
        )
    except ValidationError as exc:
        # FlashcardSetSpec requires at least one card: an empty build is an
        # honest source gap, not a zero-card deck to persist.
        return _empty(
            "No typed flashcards could be built from this project's sources "
            f"yet ({str(exc).splitlines()[-1] if str(exc) else 'empty deck'}). "
            "Upload material with extractable concepts and try again."
        )

    allowed = set(chunk_texts)
    concept_ids = set(academic_flashcards.concept_index(result.structure))
    report = validate_deck(deck.flashcards, allowed, concept_ids)
    # Grounding is checked on the pristine report: the Bloom repair below
    # re-validates against the cards' own cites and would otherwise erase an
    # unsupported_cite verdict, letting an ungrounded card persist.
    unsupported = [
        err for err in report.errors if err.get("code") == "unsupported_cite"
    ]
    if unsupported:
        return _empty(
            f"{len(unsupported)} typed card(s) cited a chunk outside this "
            "project's retrieved evidence; nothing was persisted.",
            validation=report.model_dump(mode="json"),
        )
    if not report.ok and report.bloom_ok is False and deck.flashcards:
        # One bounded Bloom repair is allowed before an honest short report.
        deck.flashcards, report, _ = academic_flashcards.repair_bloom_once(deck.flashcards)

    # Projection for the UI rows: non-empty fronts/backs only, deduped so a
    # repeated concept never writes a duplicate card.
    def _front_key(text: str) -> str:
        return " ".join(str(text or "").split()).casefold()

    seen_fronts: set = set()
    deck_rows: List[Dict[str, str]] = []
    for card in to_deck_cards(deck.flashcards):
        if not str(card.get("front") or "").strip():
            continue
        if not str(card.get("back") or "").strip():
            continue
        key = _front_key(card["front"])
        if key in seen_fronts:
            continue
        seen_fronts.add(key)
        deck_rows.append(card)

    if not deck_rows:
        return _empty(
            "The typed deck projected to zero cards; nothing was persisted.",
            validation=report.model_dump(mode="json"),
        )

    payload = flashcards_artifact(deck, report)
    stamp = datetime.now(timezone.utc).isoformat()
    payload.update({
        "model_version": result.model_version,
        "source_fingerprint": result.source_fingerprint,
        "state": "ready",
        "created_at": stamp,
        "updated_at": stamp,
    })
    if result.root_ids:
        await knowledge_repo.save_academic_artifact(
            db, result.root_ids[0], knowledge_repo.FLASHCARDS_ARTIFACT_KEY, payload
        )

    deck_id = None
    deck_filled = False
    try:
        decks = await repositories.get_flashcard_decks_by_project(
            db, _as_uuid(project_id) or project_id
        )
        if decks:
            existing = decks[0]
            deck_id = _get(existing, "id", None)
            existing_cards = list(getattr(existing, "flashcards", None) or [])
            # An existing deck WITH cards (user reviews, FSRS state) is
            # never rewritten; an existing deck with ZERO cards is filled,
            # so the earlier empty-run placeholder becomes a real deck.
            if existing_cards or int(_get(existing, "card_count", 0) or 0) > 0:
                # Reused as-is: no card is deleted, added, or reordered.
                pass
            else:
                for card in deck_rows:
                    await repositories.create_flashcard(
                        db, deck_id, card["front"], card["back"], card["difficulty"]
                    )
                deck_filled = True
        else:
            compat_deck = await repositories.create_flashcard_deck(
                db, _as_uuid(project_id) or project_id, deck.title
            )
            deck_id = _get(compat_deck, "id", None)
            for card in deck_rows:
                await repositories.create_flashcard(
                    db, deck_id, card["front"], card["back"], card["difficulty"]
                )
            deck_filled = True
    except Exception as exc:
        logger.warning("typed flashcard compatibility deck failed: %s", exc)
    return {
        "state": "ready",
        "project_id": str(project_id),
        "artifact_id": str(result.root_ids[0]) if result.root_ids else None,
        "deck_id": str(deck_id) if deck_id is not None else None,
        "deck_filled": deck_filled,
        "deck": deck.model_dump(mode="json"),
        "artifact": payload,
        "validation": report.model_dump(mode="json"),
    }


async def generate_mcq_matrix(
    db: Any,
    project_id: Any,
    document_ids: Optional[Sequence[str]] = None,
    total: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Build, validate, and persist a Topic x Bloom MCQ set."""
    result = await build_project_spine(
        db, project_id, document_ids=document_ids, force=force,
        project_title="Academic project",
    )
    chunk_texts = {
        str(row.get("id") or row.get("chunk_id")): str(row.get("text") or "")
        for row in result.chunks
    }
    topic_count = sum(len(unit.topics) for unit in result.structure.units)
    requested = int(total) if total is not None else max(topic_count, 1)
    plan = academic_mcq.build_matrix(result.structure, requested)
    mcq_set = academic_mcq.build_mcqs(
        result.structure, chunk_texts, plan, title="Academic MCQ matrix",
        boilerplate=_boilerplate_lines_for_rows(result.chunks),
    )
    topic_titles = {topic.title for unit in result.structure.units for topic in unit.topics}
    verdict = validate_set(mcq_set, set(chunk_texts), topic_titles)
    payload = mcq_artifact(mcq_set)
    stamp = datetime.now(timezone.utc).isoformat()
    payload.update({
        "model_version": result.model_version,
        "source_fingerprint": result.source_fingerprint,
        "state": "ready",
        "validation": verdict,
        "created_at": stamp,
        "updated_at": stamp,
    })
    if result.root_ids:
        await knowledge_repo.save_academic_artifact(
            db, result.root_ids[0], knowledge_repo.MCQ_ARTIFACT_KEY, payload
        )
    quiz_id = None
    try:
        quizzes = await repositories.get_quizzes_by_project(
            db, _as_uuid(project_id) or project_id
        )
        if quizzes:
            quiz_id = _get(quizzes[0], "id", None)
        else:
            compat_quiz = await repositories.create_quiz(
                db, _as_uuid(project_id) or project_id,
                mcq_set.title, project_quiz_questions(mcq_set),
            )
            quiz_id = _get(compat_quiz, "id", None)
    except Exception as exc:
        logger.warning("MCQ compatibility quiz failed: %s", exc)
    return {
        "state": "ready",
        "project_id": str(project_id),
        "artifact_id": str(result.root_ids[0]) if result.root_ids else None,
        "quiz_id": str(quiz_id) if quiz_id is not None else None,
        "questions": [q.model_dump(mode="json") for q in mcq_set.questions],
        "plan": mcq_set.plan.model_dump(mode="json"),
        "unfilled": [u.model_dump(mode="json") for u in mcq_set.plan.unfilled],
        "artifact": payload,
        "validation": verdict,
    }


async def _load_cached_academic_rows(db: Any, project_id: Any, output_type: str) -> Dict[str, Any]:
    """Read an existing academic cache row without a new output enum."""
    try:
        from sqlalchemy import select
        from database.models.generated_content import GeneratedContentCache

        result = await db.execute(
            select(GeneratedContentCache)
            .where(
                GeneratedContentCache.project_id == (_as_uuid(project_id) or project_id),
                GeneratedContentCache.output_type == output_type,
            )
            .order_by(GeneratedContentCache.created_at.desc())
        )
        rows = list(result.scalars().all())
        return dict(rows[0].output_content or {}) if rows else {}
    except Exception:
        return {}


def _metadata_values(metadata: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = metadata.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def _source_metric_inputs(
    structures: Sequence[DocumentStructure],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Expose deterministic parser outputs to the metrics stage."""

    syllabus_units: List[Dict[str, Any]] = []
    papers: List[Dict[str, Any]] = []
    mapping: List[Dict[str, Any]] = []
    document_topics: List[str] = []
    for structure in structures or []:
        for unit in getattr(structure, "units", []) or []:
            document_topics.extend(
                str(getattr(topic, "title", "")).strip()
                for topic in (getattr(unit, "topics", []) or [])
                if str(getattr(topic, "title", "")).strip()
            )
        metadata = dict(getattr(structure, "source_metadata", None) or {})
        for raw in _metadata_values(metadata, "syllabus"):
            try:
                parsed = SyllabusIngested.model_validate(raw)
            except Exception:
                continue
            syllabus_units.extend(unit.model_dump(mode="json") for unit in parsed.units)
        for raw in _metadata_values(metadata, "question_paper"):
            try:
                parsed_paper = QuestionPaperIngested.model_validate(raw)
            except Exception:
                continue
            papers.append(parsed_paper.model_dump(mode="json"))
            # Keep a flat, metrics-friendly projection alongside the typed
            # parser payload.  The mapping has already been written back to
            # each question by _write_back_question_mapping.
        for raw in _metadata_values(metadata, "mapping"):
            if isinstance(raw, list):
                mapping.extend(item for item in raw if isinstance(item, dict))
            else:
                mapping.append(raw)
    flat_pyqs: List[Dict[str, Any]] = []
    for paper in papers:
        for question in paper.get("questions") or []:
            flat_pyqs.append({
                "question": question.get("text") or question.get("question") or "",
                "year": question.get("year"),
                "marks": question.get("marks"),
                "topic": question.get("topic"),
                "unit_id": question.get("unit_id"),
                "source_id": paper.get("source_id"),
                "question_number": question.get("number"),
            })
    source_metrics: Dict[str, Any] = {}
    if syllabus_units or papers:
        try:
            typed_units = [SyllabusUnitIngested.model_validate(unit) for unit in syllabus_units]
            typed_papers = [QuestionPaperIngested.model_validate(paper) for paper in papers]
            source_metrics = compute_source_metrics(
                typed_units, typed_papers, document_topics=document_topics,
            )
        except Exception:
            # The normal metrics path remains authoritative if a legacy
            # metadata row is incomplete; never fail a job on a best-effort
            # secondary projection.
            source_metrics = {}
    return syllabus_units, flat_pyqs, mapping, {"papers": papers, "metrics": source_metrics}


async def generate_metrics_report(
    db: Any,
    project_id: Any,
    document_ids: Optional[Sequence[str]] = None,
    current_year: Optional[int] = None,
    half_life: float = 5.0,
    include_narrative: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Compute and persist deterministic metrics for the insights read path."""
    result = await build_project_spine(
        db, project_id, document_ids=document_ids, force=force,
        project_title="Academic project",
    )
    syllabus = await _load_cached_academic_rows(db, project_id, "syllabus")
    pyq = await _load_cached_academic_rows(db, project_id, "pyq")
    mapping = await _load_cached_academic_rows(db, project_id, "mapping")
    parsed_syllabus_units, parsed_pyqs, parsed_mapping, parsed_projection = _source_metric_inputs(
        result.source_structures
    )
    questions: List[Dict[str, Any]] = list(parsed_pyqs)
    try:
        quizzes = await repositories.get_quizzes_by_project(
            db, _as_uuid(project_id) or project_id
        )
        for quiz in quizzes or []:
            questions.extend(list(_get(quiz, "questions", None) or []))
    except Exception:
        pass
    normalized = normalize_inputs(
        result.structure,
        result.chunks,
        pyqs=list(parsed_pyqs or pyq.get("questions") or []),
        quiz_questions=questions,
        syllabus_units=list(parsed_syllabus_units or syllabus.get("units") or []),
    )
    normalized["source_mapping"] = list(parsed_mapping or mapping.get("mappings") or [])
    normalized["source_ingestion"] = parsed_projection
    report = compute_metrics(
        normalized,
        current_year=int(current_year or datetime.now(timezone.utc).year),
        half_life=float(half_life),
    )
    response = None
    if include_narrative:
        # Narrative is optional; without a provider it is explicitly withheld
        # rather than replaced with invented prose.
        try:
            response = repair_narrative(
                report,
                lambda _prompt: "No grounded narrative provider is configured.",
            )
        except Exception:
            response = None
    payload = metrics_artifact(report, response)
    payload["source_ingestion"] = _safe_json(parsed_projection)
    payload["source_mapping"] = list(normalized.get("source_mapping") or [])
    stamp = datetime.now(timezone.utc).isoformat()
    payload.update({
        "model_version": result.model_version,
        "source_fingerprint": result.source_fingerprint,
        "state": "ready",
        "created_at": stamp,
        "updated_at": stamp,
    })
    if result.root_ids:
        await knowledge_repo.save_academic_artifact(
            db, result.root_ids[0], knowledge_repo.METRICS_ARTIFACT_KEY, payload
        )
    return {
        "state": "ready",
        "project_id": str(project_id),
        "artifact_id": str(result.root_ids[0]) if result.root_ids else None,
        "metrics": report.model_dump(mode="json"),
        "narrative": payload.get("narrative"),
        "narrative_status": payload.get("narrative_status", "not_requested"),
        "tables": payload.get("tables", []),
        "cards": payload.get("cards", []),
        "artifact": payload,
    }


# ---------------------------------------------------------------------------
# Read helpers used by routes
# ---------------------------------------------------------------------------


async def read_spine(db: Any, project_id: Any) -> Dict[str, Any]:
    try:
        nodes = await knowledge_repo.list_knowledge_nodes(
            db, project_id=_as_uuid(project_id) or project_id
        )
    except Exception:
        nodes = []
    nodes = list(nodes or [])
    latest_version, _, _ = await _existing_model_version(db, project_id)
    if latest_version:
        latest = []
        for node in nodes:
            metadata = dict(_get(node, "node_metadata", None) or {})
            value = metadata.get("model_version") or _get(node, "model_version", 0) or 0
            try:
                if int(value) == int(latest_version):
                    latest.append(node)
            except (TypeError, ValueError):
                continue
    else:
        latest = nodes
    roots = [n for n in latest if _get(n, "parent_id", None) is None]
    if not roots:
        return {
            "state": "not_built",
            "model_version": latest_version,
            "tree": None,
            "stats": {"node_count": {}, "unresolved_count": 0, "canonical_collisions": 0},
            "generation_call": "POST /api/knowledge/model",
        }
    trees = []
    node_ids = {str(_get(node, "id", None)): node for node in latest}
    for root in roots:
        root_id = str(_get(root, "id", None))
        members = []
        for node in latest:
            cursor = node
            seen = set()
            belongs = False
            while cursor is not None:
                current_id = str(_get(cursor, "id", None))
                if current_id in seen:
                    break
                seen.add(current_id)
                if current_id == root_id:
                    belongs = True
                    break
                parent_id = _get(cursor, "parent_id", None)
                cursor = node_ids.get(str(parent_id)) if parent_id is not None else None
            if belongs:
                members.append(node)
        tree = knowledge_repo.rows_to_tree(members)
        if tree is not None:
            trees.append(tree)
    return {
        "state": "ready",
        "model_version": latest_version,
        "tree": trees[0] if trees else None,
        "trees": trees,
        "stats": {
            "node_count": {
                key: sum(1 for n in latest if str(_get(n, "type")) == key)
                for key in ("document", "unit", "topic", "subtopic", "concept")
            },
            "unresolved_count": 0,
            "canonical_collisions": 0,
        },
    }


async def read_typed_artifact(
    db: Any,
    project_id: Any,
    key: str,
    kind: str,
    generation_call: str,
) -> Dict[str, Any]:
    try:
        nodes = await knowledge_repo.list_knowledge_nodes(
            db, project_id=_as_uuid(project_id) or project_id
        )
    except Exception:
        nodes = []
    for node in sorted(nodes or [], key=lambda n: str(_get(n, "created_at", "") or ""), reverse=True):
        try:
            value = await knowledge_repo.get_academic_artifact(db, _get(node, "id", None), key)
        except Exception:
            value = None
        if value:
            return {
                "state": "ready",
                "project_id": str(project_id),
                "artifact_id": str(_get(node, "id", None)),
                **dict(value),
                "artifact": value,
            }
    state = not_generated_state(kind, generation_call)
    state["project_id"] = str(project_id)
    return state


# Compatibility names for callers that use the shorter production contract.
async def build_spine(
    project_id: Any = None,
    db: Any = None,
    source_document_ids: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> SpineBuildResult:
    """Compatibility wrapper for the original ``build_spine`` contract.

    The task contract used ``build_spine(project_id, db,
    source_document_ids=...)`` while the production orchestrator uses the
    repository-friendly ``build_project_spine(db, project_id, ...)`` order.
    Accept both forms so rolling callers do not need a flag-day migration.
    """
    # A session-like first argument means the caller used the new order.
    if (
        hasattr(project_id, "execute")
        or hasattr(project_id, "flush")
        or hasattr(project_id, "add")
    ):
        project_id, db = db, project_id
    document_ids = kwargs.pop("document_ids", None)
    if document_ids is None:
        document_ids = source_document_ids
    project_title = kwargs.pop("project_title", "Academic project")
    force = bool(kwargs.pop("force", False))
    persist = bool(kwargs.pop("persist", True))
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"build_spine() got an unexpected keyword argument {unexpected!r}")
    return await build_project_spine(
        db,
        project_id,
        document_ids=document_ids,
        project_title=project_title,
        force=force,
        persist=persist,
    )


run_hierarchical_job = process_hierarchical_job
persist_pdf_artifact = build_and_store_pdf


__all__ = [
    "AcademicPipelineError",
    "NoExtractableTextError",
    "SpineBuildResult",
    "NOTES_HIERARCHICAL_STAGES",
    "NOTES_HIERARCHICAL_STAGE_PROGRESS_RANGES",
    "NOTES_HIERARCHICAL_STAGES_DICT",
    "STAGE_PROGRESS_RANGES",
    "JOB_TYPE",
    "chunk_to_dict",
    "load_processed_chunks",
    "merge_document_structures",
    "build_source_structures",
    "spine_stats",
    "build_project_spine",
    "build_spine",
    "retrieve_topic_rows",
    "generate_topic_live_or_deterministic",
    "build_and_store_pdf",
    "persist_pdf_artifact",
    "build_pdf_for_project",
    "process_hierarchical_job",
    "run_hierarchical_job",
    "generate_typed_flashcards",
    "generate_mcq_matrix",
    "generate_metrics_report",
    "read_spine",
    "read_typed_artifact",
]
