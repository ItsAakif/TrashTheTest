"""STEP 2 extraction: processed doc + chunks + pages -> DocumentStructure.

Runs BEFORE generation (no LLM, no provider calls). Detection is
conservative:

- TOC/headings win when present: distinct chunk headings in first-seen
  (source) order become topics; headings matching unit markers
  ("unit N", "chapter N", "part N", "module N") become units.
- Otherwise conservative inference: consecutive chunks are grouped into
  topics of at most MAX_INFERRED_TOPIC_CHUNKS, still in source order.

Source order is authoritative everywhere: chunks sort by (page, position),
units/topics/concepts keep first-seen order, order_index is dense per
parent, and every concept carries provenance (method + chunk ids + note).
Retrieval stays the evidence path; structure is only the ordering path.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from services.academic.types import (
    ConceptProvenance,
    DocumentStructure,
    ExtractedConcept,
    ExtractedSubtopic,
    ExtractedTopic,
    StructuredUnit,
    StructureMethod,
)
from services.text_utils import (
    appears_early,
    is_front_back_admissible,
    is_junk_title,
    is_sentence_backed,
    is_usable_title_candidate,
    normalise_text,
    remove_boilerplate,
)

logger = logging.getLogger(__name__)

MAX_INFERRED_TOPIC_CHUNKS = 3

# A detected topic never holds more than this many chunks; larger sections
# split into page-window subtopics (FIX E) so one section cannot swallow a
# quarter of the book.
MAX_TOPIC_CHUNKS = 8

_UNIT_RE = re.compile(
    r"^\s*(unit|chapter|part|module|section)\s+(\d+|[ivx]+)\b\s*[:.\-]?\s*(.*)$",
    re.IGNORECASE,
)
_SUB_SPLIT_RE = re.compile(r"\s*(?:>>|//|\s/\s)\s*")

_CONFIDENCE = {
    StructureMethod.TOC_HEADING: 1.0,
    StructureMethod.HEADING_GROUP: 0.85,
    StructureMethod.INFERRED_SPLIT: 0.6,
    StructureMethod.SINGLE_CHUNK: 0.7,
}


def _chunk_sort_key(chunk: Dict[str, Any]) -> tuple:
    """Source order: (page, position); unknowns sort last, stably."""
    page = chunk.get("page")
    try:
        page_key = (0, int(page)) if page is not None else (1, 0)
    except (TypeError, ValueError):
        page_key = (1, 0)
    position = chunk.get("position", chunk.get("order", 0))
    try:
        pos_key = int(position) if position is not None else 0
    except (TypeError, ValueError):
        pos_key = 0
    return (page_key[0], page_key[1], pos_key)


def _ordered_pages(chunks: List[Dict[str, Any]]) -> List[int]:
    seen: List[int] = []
    for chunk in chunks:
        page = chunk.get("page")
        if isinstance(page, int) and page not in seen:
            seen.append(page)
    return seen


def _page_windows(chunks: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """Map chunk id -> (page position, distinct page count) for R2.

    Front/back matter is PAGE matter: a chunk on one of the first/last
    10% of the document's distinct ordered pages is inside the window.
    Chunks without an integer page (or documents with < 2 distinct pages)
    get no entry and are never gated.
    """
    pages = sorted({c.get("page") for c in chunks if isinstance(c.get("page"), int)})
    if len(pages) < 2:
        return {}
    index_of = {page: i for i, page in enumerate(pages)}
    total = len(pages)
    out: Dict[str, Tuple[int, int]] = {}
    for chunk in chunks:
        page = chunk.get("page")
        if isinstance(page, int) and page in index_of:
            out[str(chunk.get("id") or chunk.get("chunk_id"))] = (
                index_of[page], total)
    return out


_CONCEPT_STOP_TERMS = ("is", "are", "means", "refers to", "is defined as",
                       "is called", "are called", "is known as")
_CONCEPT_MAX_WORDS = 8
_CONCEPT_MAX_CHARS = 80


def _is_usable_concept_name(
    name: str,
    topic_title: str,
    document_title: str = "",
    unit_title: str = "",
    boilerplate: Optional[set] = None,
) -> bool:
    """A concept name must be a short noun phrase, never a sentence or junk.

    R3 (``is_junk_title``) rejects title restatements and credits/nav
    markers; the document/unit titles and boilerplate travel in as
    parameters so this never reads module globals.
    """
    if not name:
        return False
    candidate = name.strip().strip("-").strip()
    if len(candidate) < 2 or len(candidate) > _CONCEPT_MAX_CHARS:
        return False
    words = candidate.split()
    if len(words) > _CONCEPT_MAX_WORDS:
        return False
    if candidate[-1:] in {".", "!", "?", ";", ",", ":"}:
        return False
    if candidate.lower() == (topic_title or "").strip().lower():
        return False
    if " -- " in candidate:
        return False
    if is_junk_title(candidate, document_title, unit_title, boilerplate):
        return False
    return True


def _concept_title(
    chunk: Dict[str, Any],
    heading: Optional[str],
    boilerplate: Optional[set] = None,
    document_title: str = "",
    unit_title: str = "",
) -> str:
    """Short concept name for one chunk (concept naming v2 + R3/R4).

    Candidates are the chunk's LINES that pass is_usable_title_candidate
    after normalisation and boilerplate/page-number removal; the best
    scoring candidate wins (title-ish +3, early in the chunk +2, 2-5 words
    +1, redundant prefix of another candidate -2; ties -> earliest).  R4
    is a hard gate: a candidate must start within the first 40% of the
    chunk.  R3 drops credits/nav/brand/title-restatement lines.  With no
    usable line, a defined term ("<Term> is/are/means ...") is tried.
    Returns "" when nothing qualifies -- the caller skips the concept and
    never invents a sentence.
    """
    boilerplate = {str(b) for b in (boilerplate or set())}
    topic = (heading or "").strip()
    text = remove_boilerplate(
        normalise_text(str(chunk.get("text") or chunk.get("content") or "")),
        boilerplate,
    )
    forbidden = {topic.lower()} | {b.lower() for b in boilerplate}
    early_limit = 0.4 * max(1, len(text))

    # Candidate lines with their character offsets (for the 40% rule).
    candidates: List[Dict[str, Any]] = []
    offset = 0
    for line in text.split("\n"):
        start = offset
        offset += len(line) + 1  # +1 for the consumed newline
        if not line.strip():
            continue
        if not is_usable_title_candidate(line, forbidden):
            continue
        # R4: a name that first appears after the first 40% is not anchored.
        if start > early_limit:
            continue
        # R3: credits/nav/brand/title restatements never name a concept.
        if is_junk_title(line, document_title, unit_title, boilerplate):
            continue
        words = line.split()
        score = 0
        if words and sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.5:
            score += 3
        if start <= early_limit:
            score += 2
        if 2 <= len(words) <= 5:
            score += 1
        candidates.append(
            {"score": score, "start": start, "line": line, "words": words}
        )

    # -2 when the line equals the first 3+ words of another candidate
    # (a redundant prefix of a longer candidate).
    for cand in candidates:
        words = cand["words"]
        if len(words) < 3:
            continue
        for other in candidates:
            other_words = other["words"]
            if other is cand or len(other_words) <= len(words):
                continue
            if other_words[: len(words)] == words:
                cand["score"] -= 2
                break

    if candidates:
        best = max(candidates, key=lambda c: (c["score"], -c["start"]))
        return best["line"].strip()

    # No usable line: a defined term "<Term> is/are/means/refers to ...".
    flat = " ".join(text.split())
    lowered = flat.lower()
    for stop in _CONCEPT_STOP_TERMS:
        marker = f" {stop} "
        index = lowered.find(marker)
        if index > 0:
            term = flat[:index].strip(" .,:;-")
            if not appears_early(text, term):
                continue
            if is_junk_title(term, document_title, unit_title, boilerplate):
                continue
            if is_usable_title_candidate(term, forbidden):
                return term
    return ""


def _record_skip(stats: Optional[Dict[str, Any]], reason: str) -> None:
    """Count one skipped concept (for the per-document audit log line)."""
    if stats is None:
        return
    stats["skipped"] = int(stats.get("skipped", 0)) + 1
    reasons = stats.setdefault("reasons", {})
    reasons[reason] = int(reasons.get(reason, 0)) + 1


def _make_concept(
    chunk: Dict[str, Any],
    heading: Optional[str],
    method: StructureMethod,
    note: str,
    boilerplate: Optional[set] = None,
    document_title: str = "",
    unit_title: str = "",
    window_index: Optional[int] = None,
    window_total: Optional[int] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> Optional[ExtractedConcept]:
    cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
    text = normalise_text(str(chunk.get("text") or chunk.get("content") or ""))
    # R1: a chunk that is not sentence-backed may not seed a concept.
    if not is_sentence_backed(text):
        _record_skip(stats, "not_sentence_backed")
        return None
    # R2: front/back-matter chunks need >= 2 qualifying sentences.  The
    # window is measured over the document's ordered PAGES (front/back
    # matter is page matter); ``window_index``/``window_total`` are the
    # chunk's page position among distinct ordered pages.
    if window_index is not None and window_total is not None:
        if not is_front_back_admissible(window_index, window_total, text):
            _record_skip(stats, "front_back_matter")
            return None
    title = _concept_title(
        chunk, heading, boilerplate=boilerplate,
        document_title=document_title, unit_title=unit_title,
    )
    if not title:
        _record_skip(stats, "no_title")
        return None
    pages = [chunk["page"]] if isinstance(chunk.get("page"), int) else []
    return ExtractedConcept(
        title=title[:500],
        pages=pages,
        chunks=[cid],
        confidence=_CONFIDENCE[method],
        provenance=ConceptProvenance(
            method=method,
            source_heading=heading,
            chunk_ids=[cid],
            note=note,
        ),
    )


def _split_subtopic(heading: str) -> Optional[tuple]:
    """Split 'Topic // Sub' style headings once (conservative subtopics)."""
    parts = _SUB_SPLIT_RE.split(heading, maxsplit=1)
    if len(parts) == 2 and all(p.strip() for p in parts):
        return parts[0].strip(), parts[1].strip()
    return None


def _structure_fingerprint(
    doc_id: str, title: str, chunks: List[Dict[str, Any]]
) -> str:
    ordered = sorted(chunks, key=_chunk_sort_key)
    basis = "\n".join(
        f"{c.get('id') or c.get('chunk_id')}|{c.get('page')}|"
        f"{c.get('heading') or ''}|{str(c.get('text') or c.get('content') or '')[:200]}"
        for c in ordered
    )
    return hashlib.sha256(f"{doc_id}\n{title}\n{basis}".encode()).hexdigest()


def extract_structure(
    doc_id: str,
    doc_title: str,
    chunks: List[Dict[str, Any]],
    toc: Optional[List[str]] = None,
    source_type: str = "study_material",
    source_metadata: Optional[Dict[str, Any]] = None,
    boilerplate: Optional[Set[str]] = None,
) -> DocumentStructure:
    """Build the ordered DocumentStructure for one processed document.

    chunks: processed-doc rows with id, text/content, page and optional
    heading/position. toc: optional table-of-contents headings (win over
    chunk headings when given). boilerplate: optional running-header /
    page-number lines (from find_boilerplate_lines) removed before concept
    naming; None keeps every existing call working. Raises ValueError on
    empty chunk input.
    """
    if not chunks:
        raise ValueError("extract_structure requires at least one chunk")
    ordered = sorted(chunks, key=_chunk_sort_key)
    stats: Dict[str, Any] = {"skipped": 0, "reasons": {}}

    # TOC entries constrain the heading vocabulary when supplied.
    toc_set = {h.strip().lower() for h in (toc or []) if (h or "").strip()}
    headed = sum(1 for c in ordered if str(c.get("heading") or "").strip())

    if toc_set or headed >= max(1, len(ordered) // 2):
        method = (
            StructureMethod.TOC_HEADING if toc_set
            else StructureMethod.HEADING_GROUP
        )
        structure = _extract_from_headings(
            doc_id, doc_title, ordered, toc_set, method,
            source_type=source_type, source_metadata=source_metadata,
            boilerplate=boilerplate, stats=stats,
        )
    else:
        structure = _extract_inferred(
            doc_id, doc_title, ordered,
            source_type=source_type, source_metadata=source_metadata,
            boilerplate=boilerplate, stats=stats,
        )
    logger.info(
        "academic.junk_filtered concepts_skipped=%d reasons=%s",
        int(stats.get("skipped", 0)),
        dict(sorted((stats.get("reasons") or {}).items())),
    )
    return structure


def _split_topic_into_page_windows(
    topic: ExtractedTopic,
    chunks: List[Dict[str, Any]],
    method: StructureMethod,
) -> ExtractedTopic:
    """Split an over-large topic into <= MAX_TOPIC_CHUNKS-chunk windows.

    Each window becomes a subtopic titled "<original title> (p. <first>-
    <last>)" using the chunks' page numbers; source order is kept so one
    section can never swallow a quarter of the book.
    """
    chunk_ids = [str(c.get("id") or c.get("chunk_id")) for c in chunks]
    concepts_by_chunk: Dict[str, ExtractedConcept] = {}
    for sub in topic.subtopics or []:
        for concept in sub.concepts or []:
            for cid in concept.chunks or []:
                concepts_by_chunk.setdefault(str(cid), concept)
    windows: List[ExtractedSubtopic] = []
    for start in range(0, len(chunk_ids), MAX_TOPIC_CHUNKS):
        window_ids = chunk_ids[start:start + MAX_TOPIC_CHUNKS]
        window_id_set = set(window_ids)
        window_chunks = [
            c for c in chunks
            if str(c.get("id") or c.get("chunk_id")) in window_id_set
        ]
        pages = sorted({c["page"] for c in window_chunks
                        if isinstance(c.get("page"), int)})
        first_page = pages[0] if pages else "?"
        last_page = pages[-1] if pages else "?"
        window_concepts = [
            concepts_by_chunk[cid] for cid in window_ids
            if cid in concepts_by_chunk
        ]
        windows.append(ExtractedSubtopic(
            title=f"{topic.title} (p. {first_page}-{last_page})"[:500],
            pages=_ordered_pages(window_chunks),
            chunks=window_ids,
            concepts=window_concepts,
            method=method,
        ))
    return ExtractedTopic(
        title=topic.title[:500],
        pages=topic.pages,
        chunks=topic.chunks,
        subtopics=windows,
        concepts=[],
        method=method,
    )


def _extract_from_headings(
    doc_id: str,
    doc_title: str,
    ordered: List[Dict[str, Any]],
    toc_set: set,
    method: StructureMethod,
    source_type: str = "study_material",
    source_metadata: Optional[Dict[str, Any]] = None,
    boilerplate: Optional[set] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> DocumentStructure:
    units: List[StructuredUnit] = []
    current_unit: Optional[Dict[str, Any]] = None
    chunk_windows = _page_windows(ordered)

    def _ensure_unit(title: str) -> Dict[str, Any]:
        nonlocal current_unit
        if current_unit is None or current_unit["title"] != title:
            current_unit = {"title": title, "topics": []}
            units.append(current_unit)  # type: ignore[arg-type]
        return current_unit

    default_unit = f"Unit 1 -- {doc_title}"
    _ensure_unit(default_unit)
    seen_unit_marker = False

    # Group consecutive chunks by heading, preserving source order.
    groups: List[Dict[str, Any]] = []
    for chunk in ordered:
        heading = str(chunk.get("heading") or "").strip() or None
        if heading and toc_set and heading.strip().lower() not in toc_set:
            heading = None  # TOC wins: unknown headings do not fork topics
        if groups and groups[-1]["heading"] == heading:
            groups[-1]["chunks"].append(chunk)
        else:
            groups.append({"heading": heading, "chunks": [chunk]})

    for group in groups:
        heading = group["heading"]
        unit_match = _UNIT_RE.match(heading or "")
        if unit_match:
            label = unit_match.group(1).capitalize()
            number = unit_match.group(2)
            rest = unit_match.group(3).strip()
            title = f"{label} {number}" + (f" -- {rest}" if rest else "")
            _ensure_unit(title[:500])
            seen_unit_marker = True
            # A unit-marker chunk still contributes its concept to the unit:
            # attach it to an implicit topic so no chunk is ever dropped.
            topic_title = rest or title
            topic = _build_topic(
                topic_title, group["chunks"], method,
                note="unit-marker chunk kept as its own topic (no chunk dropped)",
                boilerplate=boilerplate,
                document_title=doc_title,
                unit_title=title,
                chunk_windows=chunk_windows,
                stats=stats,
            )
            _ensure_unit(title[:500])["topics"].append(topic)
            continue
        if heading is None:
            topic_title = f"Untitled -- page {group['chunks'][0].get('page', '?')}"
        else:
            topic_title = heading
        topic = _build_topic(topic_title, group["chunks"], method,
                             note="heading group in source order",
                             boilerplate=boilerplate,
                             document_title=doc_title,
                             unit_title=(current_unit or {}).get("title", default_unit),
                             chunk_windows=chunk_windows,
                             stats=stats)
        if len(topic.chunks) > MAX_TOPIC_CHUNKS:
            topic = _split_topic_into_page_windows(
                topic, group["chunks"], method)
        assert current_unit is not None
        current_unit["topics"].append(topic)

    if not seen_unit_marker:
        # Single conservative unit keeps the required depth chain valid.
        units[0]["title"] = default_unit[:500]

    built_units = [
        StructuredUnit(
            title=u["title"][:500],
            pages=_ordered_pages([c for t in u["topics"] for c in _topic_chunks(t)]),
            topics=u["topics"],
            method=method,
        )
        for u in units
        if u["topics"]
    ]
    if not built_units:
        raise ValueError("extract_structure found no topics in headed chunks")
    return DocumentStructure(
        doc_id=doc_id,
        title=doc_title,
        units=built_units,
        fingerprint=_structure_fingerprint(doc_id, doc_title, ordered),
        source_type=source_type,
        source_metadata=dict(source_metadata or {}),
    )


def _topic_chunks(topic: ExtractedTopic) -> List[Dict[str, Any]]:
    return [{"page": p} for p in topic.pages]


def _build_topic(
    title: str,
    chunks: List[Dict[str, Any]],
    method: StructureMethod,
    note: str,
    boilerplate: Optional[set] = None,
    document_title: str = "",
    unit_title: str = "",
    chunk_windows: Optional[Dict[str, Tuple[int, int]]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> ExtractedTopic:
    windows = chunk_windows or {}
    concepts = [
        concept for concept in (
            _make_concept(
                c, title, method, f"{note}; one concept per chunk",
                boilerplate=boilerplate,
                document_title=document_title,
                unit_title=unit_title,
                window_index=windows.get(
                    str(c.get("id") or c.get("chunk_id")), (None, None))[0],
                window_total=windows.get(
                    str(c.get("id") or c.get("chunk_id")), (None, None))[1],
                stats=stats,
            )
            for c in chunks
        ) if concept is not None
    ]
    chunk_by_id = {str(c.get("id") or c.get("chunk_id")): c for c in chunks}
    sub_map: Dict[str, List[ExtractedConcept]] = {}
    sub_order: List[str] = []
    plain: List[ExtractedConcept] = []
    for concept in concepts:
        chunk = chunk_by_id.get(concept.chunks[0], {})
        split = _split_subtopic(title)
        if split:
            _, sub_title = split
            if sub_title not in sub_map:
                sub_map[sub_title] = []
                sub_order.append(sub_title)
            sub_map[sub_title].append(concept)
        else:
            plain.append(concept)
    subtopics: List[ExtractedSubtopic] = []
    if sub_map:
        for sub_title in sub_order:
            subs = sub_map[sub_title]
            pages = sorted({p for s in subs for p in s.pages})
            subtopics.append(ExtractedSubtopic(
                title=sub_title[:500],
                pages=pages,
                chunks=[cid for s in subs for cid in s.chunks],
                concepts=subs,
                method=method,
            ))
        if plain:
            subtopics.insert(0, ExtractedSubtopic(
                title=title[:500],
                pages=_ordered_pages(chunks),
                chunks=[str(c.get("id") or c.get("chunk_id")) for c in chunks],
                concepts=plain,
                method=method,
            ))
    else:
        subtopics = [ExtractedSubtopic(
            title=title[:500],
            pages=_ordered_pages(chunks),
            chunks=[str(c.get("id") or c.get("chunk_id")) for c in chunks],
            concepts=plain,
            method=method,
        )]
    return ExtractedTopic(
        title=title[:500],
        pages=_ordered_pages(chunks),
        chunks=[str(c.get("id") or c.get("chunk_id")) for c in chunks],
        subtopics=subtopics,
        concepts=[],
        method=method,
    )


def _extract_inferred(
    doc_id: str,
    doc_title: str,
    ordered: List[Dict[str, Any]],
    source_type: str = "study_material",
    source_metadata: Optional[Dict[str, Any]] = None,
    boilerplate: Optional[set] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> DocumentStructure:
    """Conservative fallback: window consecutive chunks into topics."""
    chunk_windows = _page_windows(ordered)
    unit_title = f"Unit 1 -- {doc_title}"
    topics: List[ExtractedTopic] = []
    for start in range(0, len(ordered), MAX_INFERRED_TOPIC_CHUNKS):
        window = ordered[start:start + MAX_INFERRED_TOPIC_CHUNKS]
        first_page = window[0].get("page", "?")
        title = f"{doc_title} -- part {len(topics) + 1} (p. {first_page})"
        topics.append(_build_topic(
            title[:500], window, StructureMethod.INFERRED_SPLIT,
            note="no headings: conservative window in source order",
            boilerplate=boilerplate,
            document_title=doc_title,
            unit_title=unit_title,
            chunk_windows=chunk_windows,
            stats=stats,
        ))
    unit = StructuredUnit(
        title=unit_title[:500],
        pages=_ordered_pages(ordered),
        topics=topics,
        method=StructureMethod.INFERRED_SPLIT,
    )
    return DocumentStructure(
        doc_id=doc_id,
        title=doc_title,
        units=[unit],
        fingerprint=_structure_fingerprint(doc_id, doc_title, ordered),
        source_type=source_type,
        source_metadata=dict(source_metadata or {}),
    )


def _merge_key(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _union_in_order(values) -> List[Any]:
    """Order-preserving unique union (first occurrence wins)."""
    seen: Set[str] = set()
    out: List[Any] = []
    for value in values:
        key = str(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _merge_concept_lists(
    left: List[ExtractedConcept], right: List[ExtractedConcept]
) -> List[ExtractedConcept]:
    out: List[ExtractedConcept] = []
    positions: Dict[str, int] = {}
    for concept in [*left, *right]:
        key = _merge_key(concept.title)
        if key in positions:
            old = out[positions[key]]
            out[positions[key]] = old.model_copy(update={
                "pages": _union_in_order([*old.pages, *concept.pages]),
                "chunks": _union_in_order([*old.chunks, *concept.chunks]),
                "confidence": max(
                    old.confidence or 0.0, concept.confidence or 0.0
                ) or None,
            })
        else:
            positions[key] = len(out)
            out.append(concept.model_copy(deep=True))
    return out


def _merge_subtopic_lists(
    left: List[ExtractedSubtopic], right: List[ExtractedSubtopic]
) -> List[ExtractedSubtopic]:
    out: List[ExtractedSubtopic] = []
    positions: Dict[str, int] = {}
    for sub in [*left, *right]:
        key = _merge_key(sub.title)
        if key in positions:
            old = out[positions[key]]
            out[positions[key]] = old.model_copy(update={
                "pages": _union_in_order([*old.pages, *sub.pages]),
                "chunks": _union_in_order([*old.chunks, *sub.chunks]),
                "concepts": _merge_concept_lists(old.concepts, sub.concepts),
            })
        else:
            positions[key] = len(out)
            out.append(sub.model_copy(deep=True))
    return out


def _merge_two_topics(left: ExtractedTopic, right: ExtractedTopic) -> ExtractedTopic:
    """Merge two adjacent topics into one page-window topic (source order)."""
    pages = _union_in_order([*left.pages, *right.pages])
    window = sorted({p for p in pages if isinstance(p, int)})
    first_page = window[0] if window else "?"
    last_page = window[-1] if window else "?"
    title = f"{left.title} - {right.title} (p. {first_page}-{last_page})"[:500]
    return ExtractedTopic(
        title=title,
        aliases=_union_in_order([*left.aliases, *right.aliases]),
        pages=pages,
        chunks=_union_in_order([*left.chunks, *right.chunks]),
        subtopics=_merge_subtopic_lists(left.subtopics, right.subtopics),
        concepts=_merge_concept_lists(left.concepts, right.concepts),
        method=left.method,
    )


def merge_adjacent_topics(
    structure: DocumentStructure, max_topics: int
) -> DocumentStructure:
    """Merge adjacent topics until the spine fits ``max_topics``.

    Deterministic and idempotent: the smallest (fewest chunks) adjacent pair
    in source order is merged first, ties broken by earliest position, until
    the count fits.  Merged topics keep the union of their pages/chunks/
    concepts in source order and are retitled
    ``"<first> - <last> (p. <first>-<last>)"``.  Nothing else is reordered,
    and a structure already within the limit is returned untouched.
    """
    limit = int(max_topics)
    before = sum(len(unit.topics) for unit in structure.units)
    if limit < 1 or before <= limit:
        return structure
    entries: List[List[Any]] = [
        [unit_index, topic]
        for unit_index, unit in enumerate(structure.units)
        for topic in unit.topics
    ]
    while len(entries) > limit:
        best_index = 0
        best_size: Optional[int] = None
        for i in range(len(entries) - 1):
            size = len(entries[i][1].chunks or []) + len(
                entries[i + 1][1].chunks or []
            )
            if best_size is None or size < best_size:
                best_size = size
                best_index = i
        unit_index, left = entries[best_index]
        _, right = entries[best_index + 1]
        entries[best_index] = [unit_index, _merge_two_topics(left, right)]
        del entries[best_index + 1]

    ordered_unit_indexes: List[int] = []
    grouped: Dict[int, List[ExtractedTopic]] = {}
    for unit_index, topic in entries:
        if unit_index not in grouped:
            grouped[unit_index] = []
            ordered_unit_indexes.append(unit_index)
        grouped[unit_index].append(topic)

    units: List[StructuredUnit] = []
    for unit_index in ordered_unit_indexes:
        template = structure.units[unit_index]
        topics = grouped[unit_index]
        units.append(StructuredUnit(
            title=template.title,
            aliases=list(template.aliases),
            pages=_union_in_order([p for t in topics for p in (t.pages or [])]),
            topics=topics,
            method=template.method,
        ))
    after = len(entries)
    logger.info(
        "notes.topics_merged before=%d after=%d limit=%d", before, after, limit
    )
    return structure.model_copy(update={"units": units})


__all__ = [
    "MAX_INFERRED_TOPIC_CHUNKS",
    "MAX_TOPIC_CHUNKS",
    "extract_structure",
    "merge_adjacent_topics",
]
