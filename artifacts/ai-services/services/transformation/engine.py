"""
Transformation Engine Service

Unified transformation engine that generates various content types
from retrieved document context using LLM generation.
"""

import hashlib
import json
import time
import uuid
from typing import Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from services.retrieval.service import retrieval_service
from services.research.service import ResearchAugmentationService, research_service
from .types import (
    GenerationMode,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    FlashcardList,
    FlashcardSet,
    FlashcardsTypedOutput,
    Quiz,
    MCQSet,
    MCQMatrixOutput,
    MetricsNarrativeOutput,
    StructuredNotes,
    StructureOutput,
    KnowledgeSpineOutput,
    TopicNotesOutput,
    HierNotesOutput,
    Note,
    Explanation,
    Answer,
    SummaryOutput,
    WebsiteOutput,
    Syllabus,
    PYQSet,
    MappingResult,
    AnalysisResult,
    constraints_for_marks,
)
from .prompts import PromptTemplates, augment_with_research
from database.models import GeneratedContentCache, OutputType
from database import repositories


# ---------------------------------------------------------------------------
# Phase-4 shared helpers (used by the engine AND the LangGraph nodes, so the
# graphs reuse the exact same grounding logic - no decorative wrappers).
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Canonical form for duplicate detection (case/space/punct folded)."""
    import re as _re
    import unicodedata

    folded = unicodedata.normalize("NFKC", text or "").lower()
    folded = _re.sub(r"[^a-z0-9\s]", " ", folded)
    return " ".join(folded.split())


def _document_order_key(chunk) -> tuple:
    """Sort key placing retrieval chunks in source document order.

    (page_start, position): chunks with unknown pagination sort after
    paginated ones but keep their relative relevance order (Python's
    sort is stable, so equal keys preserve the input ranking).
    """
    meta = dict(getattr(chunk, "metadata", None) or {})
    page = getattr(chunk, "page", None)
    if page is None:
        page = meta.get("page_start", meta.get("page"))
    try:
        page_key = (0, int(page)) if page is not None else (1, 0)
    except (TypeError, ValueError):
        page_key = (1, 0)
    position = meta.get("position")
    try:
        pos_key = int(position) if position is not None else 0
    except (TypeError, ValueError):
        pos_key = 0
    return (page_key[0], page_key[1], pos_key)


def dedupe_flashcards(cards: list) -> list:
    """Drop duplicate flashcards by normalized front (keep first)."""
    seen: set = set()
    out = []
    for card in cards:
        front = card.get("front", "") if isinstance(card, dict) else getattr(card, "front", "")
        key = normalize_text(str(front))
        if key and key not in seen:
            seen.add(key)
            out.append(card)
    return out


def dedupe_mcqs(questions: list) -> list:
    """Drop duplicate MCQ/quiz questions by normalized question text."""
    seen: set = set()
    out = []
    for q in questions:
        text = q.get("question", "") if isinstance(q, dict) else getattr(q, "question", "")
        key = normalize_text(str(text))
        if key and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def dedupe_pyqs(questions: list) -> list:
    """Drop duplicate PYQs by normalized (question, year, marks) triple.

    Same question text reappearing across years is kept once per year;
    exact repeats (same text + year + marks) collapse to the first.
    """
    seen: set = set()
    out = []
    for q in questions:
        if isinstance(q, dict):
            text = str(q.get("question", ""))
            year = q.get("year", "")
            marks = q.get("marks", "")
        else:
            text = str(getattr(q, "question", ""))
            year = getattr(q, "year", "")
            marks = getattr(q, "marks", "")
        key = (normalize_text(text), str(year), str(marks))
        if key[0] and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def sanitise_display_text(text: str) -> str:
    """Strip HTML/script markup from extracted text for safe display.

    Storage keeps the raw extracted text; this is applied at response
    time so rendered views never execute uploaded markup. Stdlib only.
    """
    import re as _re
    from html import escape as _escape

    cleaned = _re.sub(r"(?is)<script.*?</script\s*>", " ", text or "")
    cleaned = _re.sub(r"(?is)<style.*?</style\s*>", " ", cleaned)
    cleaned = _re.sub(r"(?s)<!--.*?-->", " ", cleaned)
    cleaned = _re.sub(r"<[^>]*>", " ", cleaned)
    cleaned = _re.sub(r"(?i)\bon\w+\s*=\s*('[^']*'|\"[^\"]*\"|[^\s>]+)", " ", cleaned)
    cleaned = _re.sub(r"\s+", " ", cleaned).strip()
    return _escape(cleaned, quote=False)


def _unit_corpus(unit: dict) -> set:
    """Token set for a syllabus unit (title + topics + outcomes)."""
    parts = [str(unit.get("title", ""))]
    parts.extend(str(t) for t in (unit.get("topics") or []))
    parts.extend(str(o) for o in (unit.get("outcomes") or []))
    return set(normalize_text(" ".join(parts)).split())


def score_unit_for_question(question_text: str, unit: dict) -> float:
    """Token-overlap score of a question against one syllabus unit.

    Returns overlap / question-tokens in 0..1 (0 when either side is
    empty). Pure + deterministic so mapping is reproducible.
    """
    qtokens = set(normalize_text(question_text or "").split())
    if not qtokens:
        return 0.0
    corpus = _unit_corpus(unit)
    if not corpus:
        return 0.0
    return len(qtokens & corpus) / len(qtokens)


def map_questions_to_units(
    questions: list,
    units: list,
    min_score: float = 0.15,
) -> Dict[str, Any]:
    """Deterministically map each PYQ to its best syllabus unit.

    - explicit unit_id on the question wins when it names a known unit
      (method "explicit_unit", confidence 1.0)
    - otherwise the highest token-overlap unit at >= min_score wins
      (method "token_overlap", confidence = overlap score)
    - anything below min_score is unmapped (reported, never forced)

    Returns {"mappings": [...], "unmapped_count": int, "coverage_pct": float}.
    """
    by_id = {str(u.get("unit_id")): u for u in units if u.get("unit_id")}
    mappings: list = []
    for idx, q in enumerate(questions):
        text = str(q.get("question", "") if isinstance(q, dict) else "")
        explicit = (q.get("unit_id") if isinstance(q, dict) else None) or None
        if explicit is not None and str(explicit) in by_id:
            unit = by_id[str(explicit)]
            mappings.append({
                "question_index": idx,
                "question": text,
                "unit_id": str(explicit),
                "unit_title": str(unit.get("title", "")),
                "confidence": 1.0,
                "method": "explicit_unit",
            })
            continue
        best_id: Optional[str] = None
        best_title: Optional[str] = None
        best_score = 0.0
        for uid, unit in by_id.items():
            score = score_unit_for_question(
                f"{text} {q.get('topic', '') if isinstance(q, dict) else ''}", unit
            )
            if score > best_score:
                best_score, best_id = score, uid
                best_title = str(unit.get("title", ""))
        if best_id is not None and best_score >= min_score:
            mappings.append({
                "question_index": idx,
                "question": text,
                "unit_id": best_id,
                "unit_title": best_title,
                "confidence": round(best_score, 4),
                "method": "token_overlap",
            })
        else:
            mappings.append({
                "question_index": idx,
                "question": text,
                "unit_id": None,
                "unit_title": None,
                "confidence": round(best_score, 4),
                "method": "unmapped",
            })
    total = len(mappings)
    unmapped = sum(1 for m in mappings if m["unit_id"] is None)
    coverage = round((total - unmapped) / total * 100.0, 2) if total else 0.0
    return {"mappings": mappings, "unmapped_count": unmapped, "coverage_pct": coverage}


def compute_frequency(
    mappings: list,
    units: list,
    questions: list,
) -> Dict[str, Any]:
    """Per-unit question frequency + marks alignment (deterministic).

    Returns {"total_questions", "mapped_questions", "unmapped_questions",
    "frequency": [{unit_id, unit_title, count, marks_total, share_pct}],
    "total_marks", "coverage_pct"}. Units with zero questions are
    included with count 0 (honest gaps, no fake rows).
    """
    marks_by_idx = {
        m.get("question_index"): int((questions[m["question_index"]].get("marks", 0))
                                     if isinstance(questions[m["question_index"]], dict)
                                     else 0)
        for m in mappings
        if 0 <= m.get("question_index", -1) < len(questions)
    }
    mapped = [m for m in mappings if m.get("unit_id") is not None]
    total = len(mappings)
    unmapped = total - len(mapped)
    total_marks = sum(marks_by_idx.get(m["question_index"], 0) for m in mapped)
    frequency: list = []
    for unit in units:
        uid = str(unit.get("unit_id", ""))
        unit_maps = [m for m in mapped if m.get("unit_id") == uid]
        count = len(unit_maps)
        marks_total = sum(marks_by_idx.get(m["question_index"], 0) for m in unit_maps)
        share = round(count / len(mapped) * 100.0, 2) if mapped else 0.0
        frequency.append({
            "unit_id": uid,
            "unit_title": str(unit.get("title", "")),
            "count": count,
            "marks_total": marks_total,
            "share_pct": share,
        })
    frequency.sort(key=lambda f: (-f["count"], -f["marks_total"], f["unit_id"]))
    coverage = round(len(mapped) / total * 100.0, 2) if total else 0.0
    return {
        "total_questions": total,
        "mapped_questions": len(mapped),
        "unmapped_questions": unmapped,
        "frequency": frequency,
        "total_marks": total_marks,
        "coverage_pct": coverage,
    }


def collect_cited_ids(payload: Dict[str, Any]) -> set:
    """All chunk ids cited anywhere in a validated payload."""
    found: set = set()
    stack = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            cits = node.get("citations")
            if isinstance(cits, list):
                for c in cits:
                    if isinstance(c, dict) and c.get("chunk_id"):
                        found.add(str(c["chunk_id"]))
                    elif isinstance(c, str):
                        found.add(c)
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return found


def validate_grounding(
    payload: Dict[str, Any],
    allowed_ids: set,
    chunk_texts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Grounding gate for Phase-4 payloads (raises, never silent).

    - every cited chunk id must be in allowed_ids (fabrication -> error)
    - when chunk_texts is given, every cited span must occur (normalized)
      in its chunk text (unsupported span -> error detail, not silent drop)

    Returns {"cited": [...], "unsupported": [...]}; unsupported is non-empty
    only when spans are supplied and a span does not occur.
    """
    from services.strategies.llm.structured import StructuredGenerationError

    cited = collect_cited_ids(payload)
    fabricated = sorted(i for i in cited if i not in allowed_ids)
    if fabricated:
        raise StructuredGenerationError(
            f"Fabricated citation ids rejected: {fabricated}"
        )
    unsupported: list = []
    if chunk_texts is not None:
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                cits = node.get("citations")
                if isinstance(cits, list):
                    for c in cits:
                        if isinstance(c, dict) and c.get("chunk_id") and c.get("span"):
                            body = normalize_text(chunk_texts.get(str(c["chunk_id"]), ""))
                            span = normalize_text(str(c["span"]))
                            if span and span not in body:
                                unsupported.append(
                                    {"chunk_id": str(c["chunk_id"]), "span": str(c["span"])[:120]}
                                )
                stack.extend(node.values())
            elif isinstance(node, (list, tuple)):
                stack.extend(node)
    return {"cited": sorted(cited), "unsupported": unsupported}


def check_hallucination(
    payload: Dict[str, Any],
    allowed_ids: set,
    has_retrieval: bool,
) -> Dict[str, Any]:
    """Hallucination/unsupported guard (pure, testable).

    - citations to ids outside the retrieval set -> fabricated
    - non-empty factual payload with retrieval backing but zero citations
      -> ungenerated-citation warning (grounding.support stays honest)
    - no retrieval at all -> unsupported (never rendered as cited)
    """
    cited = collect_cited_ids(payload)
    fabricated = sorted(i for i in cited if i not in allowed_ids)
    unwarned = bool(has_retrieval) and not cited and bool(allowed_ids)
    return {
        "fabricated": fabricated,
        "missing_citations": unwarned,
        "unsupported": (not has_retrieval),
        "ok": not fabricated and not (not has_retrieval and cited),
    }


async def _scoped_document_id(db: AsyncSession, project_id: str, source_type: str) -> Optional[str]:
    """First document of a given source type in the project (syllabus/PYQ)."""
    try:
        from sqlalchemy import text as _text

        row = (await db.execute(
            _text("select id from documents where project_id = :pid "
                  "and lower(source_type::text) = :stype "
                  "order by created_at desc limit 1"),
            {"pid": str(project_id), "stype": source_type.lower()},
        )).first()
        return str(row[0]) if row else None
    except Exception:
        return None

class TransformationEngine:
    """
    Unified Transformation Engine for generating educational content.
    
    Features:
    - Multiple output modes (notes, flashcards, quiz, etc.)
    - RAG integration with vector retrieval
    - Content caching to avoid recomputation
    - Research augmentation support via ResearchAugmentationService
    - Output validation and retry logic
    - Extensible mode system
    - Context trimming for token limits
    """
    
    # Maximum tokens allowed in context (leaving room for output)
    MAX_CONTEXT_TOKENS = 6000
    
    def __init__(
        self,
        llm_provider=None,
        research_service: Optional[ResearchAugmentationService] = None
    ):
        if llm_provider is not None:
            self.llm = llm_provider
        else:
            # Request selection wins (user Settings > AI Provider), else
            # the factory-owned router (T305). The effective provider
            # delegates per call, so one singleton serves every user.
            from services.provider_credentials.override import EffectiveLLMProvider

            self.llm = EffectiveLLMProvider()
        if research_service is not None:
            self.research = research_service
        else:
            from services.research.service import research_service as _default_research

            self.research = _default_research
    
    def _compute_content_hash(
        self,
        project_id: str,
        mode: GenerationMode,
        chunk_ids: list[str],
        config: Dict[str, Any]
    ) -> str:
        """Compute content hash for caching"""
        hash_input = f"{project_id}:{mode.value}:{','.join(sorted(chunk_ids))}:{json.dumps(config, sort_keys=True)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _build_context(
        self,
        retrieval_result,
        document_order: bool = False,
    ) -> str:
        """Build context string from retrieved chunks.

        Each chunk travels in a delimited per-source block carrying its
        chunk id (T307): the model may cite only ids listed here, and the
        citation check validates exactly that. Deduplicated, relevance
        ordered, at most 10 chunks.

        document_order=True sorts the blocks by (page_start, position)
        so notes generation follows the source document instead of the
        relevance ranking. Each block then carries its page + section
        path so the prompt order rule has concrete anchors.
        """
        # Deduplicate and sort by relevance
        seen_content = set()
        pending = []

        for chunk in retrieval_result.chunks:
            # Simple deduplication
            content_preview = chunk.content[:200]
            if content_preview not in seen_content:
                seen_content.add(content_preview)
                pending.append(chunk)

        if document_order:
            pending.sort(key=_document_order_key)

        context_parts = []
        for chunk in pending:
            cid = getattr(chunk, "chunk_id", "?")
            page = getattr(chunk, "page", None)
            section = getattr(chunk, "section", None)
            locator = f" page={page}" if page is not None else ""
            if section:
                locator += f" section={section}"
            context_parts.append(
                f"[source chunk_id={cid} relevance={chunk.similarity:.2f}{locator}]\n"
                f"{chunk.content}\n"
                f"[/source chunk_id={cid}]"
            )

        # Combine into context
        return "\n\n---\n\n".join(context_parts[:10])  # Max 10 chunks in context
    
    def _trim_context(
        self,
        context: str,
        max_tokens: int = None
    ) -> str:
        """
        Trim context to fit within token limits.
        
        Uses a rough estimate of 4 characters per token.
        Removes least relevant chunks first.
        """
        max_tokens = max_tokens or self.MAX_CONTEXT_TOKENS
        max_chars = max_tokens * 4  # Rough estimate
        
        if len(context) <= max_chars:
            return context
        
        # Split by chunk separator and trim
        chunks = context.split("\n\n---\n\n")
        
        # Keep more relevant (earlier) chunks, trim the end
        result_parts = []
        current_length = 0
        
        for chunk in chunks:
            if current_length + len(chunk) + len("\n\n---\n\n") > max_chars:
                # Truncate last chunk if needed
                remaining = max_chars - current_length - len("\n\n---\n\n")
                if remaining > 100:  # At least keep some content
                    result_parts.append(chunk[:remaining])
                break
            result_parts.append(chunk)
            current_length += len(chunk) + len("\n\n---\n\n")
        
        return "\n\n---\n\n".join(result_parts)
    
    async def _check_cache(
        self,
        db: AsyncSession,
        content_hash: str,
        project_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Check if content is cached"""
        # This would query the generated_content_cache table
        # For now, we'll implement inline
        from sqlalchemy import select
        
        result = await db.execute(
            select(GeneratedContentCache).where(
                GeneratedContentCache.content_hash == content_hash,
                GeneratedContentCache.project_id == project_id
            )
        )
        cached = result.scalar_one_or_none()
        
        if cached and cached.output_content:
            return cached.output_content
        
        return None
    
    async def _save_to_cache(
        self,
        db: AsyncSession,
        content_hash: str,
        project_id: uuid.UUID,
        output_type: OutputType,
        content: Dict[str, Any]
    ):
        """Save generated content to cache"""
        cached = GeneratedContentCache(
            id=uuid.uuid4(),
            project_id=project_id,
            content_hash=content_hash,
            output_type=output_type,
            output_content=content
        )
        db.add(cached)
        await db.commit()

    async def _persist_domain_rows(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        mode: GenerationMode,
        content: Dict[str, Any],
    ) -> None:
        """Write the domain rows the read endpoints serve (Phase B).

        notes -> notes row (full content; reads render content.sections);
        flashcards -> deck + cards; quiz/mcq -> quiz row with questions.
        Other modes are cache-served and persist nothing here. Shapes are
        already validated (strict models, min lengths), so non-empty lists
        are simply stored. No commit: the cache save commits the session.

        Empty output is NEVER persisted as a domain row: a deck with zero
        cards or a note with zero (or heading-only) sections read back as
        "ready" and rendered as an empty artifact. Such a result stays in
        the cache where it belongs, and the read endpoints keep reporting
        not_generated until there is something real to serve.
        """
        if mode == GenerationMode.NOTES:
            # Lazy import: services.academic pulls this engine in at package
            # import time, so the shared boilerplate detector is resolved
            # here instead of at module scope (circular otherwise).
            from services.academic.validation import is_placeholder

            sections = [
                s for s in (content.get("sections") or [])
                if isinstance(s, dict)
            ]
            # Headings-only is empty; boilerplate ("no information
            # available...") is a source gap reported elsewhere. Neither is
            # a notes artifact, so neither may become a "ready" row.
            substantive = [
                s for s in sections
                if not is_placeholder(str(s.get("content") or ""))
            ]
            if not substantive:
                return
            title = str(content.get("title") or "Study Notes")
            await repositories.create_note(db, project_id, title, content)
        elif mode == GenerationMode.FLASHCARDS:
            cards = [
                c for c in (content.get("flashcards") or [])
                if isinstance(c, dict)
                and str(c.get("front") or "").strip()
                and str(c.get("back") or "").strip()
            ]
            if not cards:
                # Never create an empty deck: it would read back "ready".
                return
            deck = await repositories.create_flashcard_deck(
                db, project_id, str(content.get("title") or "Flashcards"))
            for card in cards:
                await repositories.create_flashcard(
                    db, deck.id, str(card.get("front") or ""),
                    str(card.get("back") or ""),
                    difficulty=str(card.get("difficulty") or "medium"))
        elif mode in (GenerationMode.QUIZ, GenerationMode.MCQ):
            questions = content.get("questions") or []
            await repositories.create_quiz(
                db, project_id, str(content.get("title") or "Practice Quiz"),
                [q for q in questions if isinstance(q, dict)])
    
    MODE_MODELS = {
        GenerationMode.FLASHCARDS: FlashcardList,
        GenerationMode.QUIZ: Quiz,
        GenerationMode.MCQ: MCQSet,
        GenerationMode.EXPLANATION: Explanation,
        GenerationMode.ANSWER: Answer,
        GenerationMode.NOTES: StructuredNotes,
        GenerationMode.SUMMARY: SummaryOutput,
        GenerationMode.WEBSITE: WebsiteOutput,
        GenerationMode.SYLLABUS: Syllabus,
        GenerationMode.PYQ: PYQSet,
        GenerationMode.STRUCTURE: StructureOutput,
        GenerationMode.KNOWLEDGE_SPINE: KnowledgeSpineOutput,
        GenerationMode.NOTES_TOPIC: TopicNotesOutput,
        GenerationMode.NOTES_HIER: HierNotesOutput,
        GenerationMode.FLASHCARDS_TYPED: FlashcardsTypedOutput,
        GenerationMode.MCQ_MATRIX: MCQMatrixOutput,
        GenerationMode.METRICS_NARRATIVE: MetricsNarrativeOutput,
    }

    # Phase-4 strict production schemas (grounded: citations + provenance).
    # Compat aliases above are kept for existing callers/tests.
    STRICT_MODELS = {
        GenerationMode.NOTES: Note,
        GenerationMode.FLASHCARDS: FlashcardSet,
        GenerationMode.QUIZ: Quiz,
        GenerationMode.MCQ: MCQSet,
        GenerationMode.EXPLANATION: Explanation,
        GenerationMode.ANSWER: Answer,
        GenerationMode.SYLLABUS: Syllabus,
        GenerationMode.PYQ: PYQSet,
        GenerationMode.MAPPING: MappingResult,
        GenerationMode.ANALYSIS: AnalysisResult,
        GenerationMode.STRUCTURE: StructureOutput,
        GenerationMode.KNOWLEDGE_SPINE: KnowledgeSpineOutput,
        GenerationMode.NOTES_TOPIC: TopicNotesOutput,
        GenerationMode.NOTES_HIER: HierNotesOutput,
        GenerationMode.FLASHCARDS_TYPED: FlashcardsTypedOutput,
        GenerationMode.MCQ_MATRIX: MCQMatrixOutput,
        GenerationMode.METRICS_NARRATIVE: MetricsNarrativeOutput,
    }

    def _parse_output(
        self,
        mode: GenerationMode,
        raw_output: str
    ) -> Dict[str, Any]:
        """Legacy entry point kept for compatibility; validates strictly.

        New code should use generate_structured via _generate_structured.
        Raises StructuredGenerationError on schema failure (never loosely
        parsed text, T306).
        """
        from services.strategies.llm.structured import _extract_json

        model = self.MODE_MODELS.get(mode)
        data = _extract_json(raw_output)
        if model is None:
            return dict(data) if isinstance(data, dict) else {"raw_output": raw_output}
        return model.model_validate(data).model_dump()

    def _check_citations(self, payload: Dict[str, Any], allowed_ids: set) -> None:
        """Reject fabricated citation ids in model output (T307).

        Two independent checks, both must pass:

        1. every id the payload actually CITES (the `citations` arrays, at
           any depth) must be one of the retrieved chunk ids -- this also
           catches short, non-UUID ids such as the 8-hex strings a model
           invents, which the UUID scan below cannot see;
        2. every UUID-like token anywhere in a string must be one of the
           retrieved chunk ids, so an id smuggled into prose is rejected too.

        Raises StructuredGenerationError on fabrication.
        """
        import re as _re

        from services.strategies.llm.structured import StructuredGenerationError

        # Ids may arrive as str or UUID; compare as strings either way.
        _allowed = {str(i) for i in allowed_ids}

        cited_not_allowed = sorted(
            str(chunk_id)
            for chunk_id in collect_cited_ids(payload)
            if str(chunk_id) not in _allowed
        )
        if cited_not_allowed:
            raise StructuredGenerationError(
                f"Fabricated citation ids rejected: {cited_not_allowed}"
            )

        uuid_like = _re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        )
        stack = [payload]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, (list, tuple)):
                stack.extend(node)
            elif isinstance(node, str):
                for token in uuid_like.findall(node):
                    if token not in _allowed:
                        raise StructuredGenerationError(
                            f"Fabricated citation id rejected: {token}"
                        )

    async def _generate_structured(
        self,
        mode: GenerationMode,
        prompt: str,
        allowed_chunk_ids: set,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        """Schema-enforced generation: strict mode, JSON fallback, <=2 repairs (T306).

        ``max_tokens`` raises the per-call output budget for callers whose
        payload is a bounded unit of work (one academic topic). Omit it to keep
        the shared 4096-token default. A length-limited answer is recovered by
        the structured helper's bounded truncated-JSON repair, so a cut-off
        response fails the job only after the recovery and repair loops are
        both exhausted.
        """
        from services.strategies.llm.structured import (
            StructuredGenerationError,
            generate_structured,
        )

        model = self.MODE_MODELS.get(mode)
        if model is None:
            # EXPLANATION has no strict model: single JSON-mode attempt,
            # parsed via the structured helper (validated shape, T306).
            from services.strategies.base import GenerationConfig as _GC
            from services.strategies.llm.structured import _extract_json

            result = await self.llm.generate(
                prompt, _GC(response_format={"type": "json_object"})
            )
            try:
                payload = _extract_json(result.text)
                if not isinstance(payload, dict):
                    raise ValueError("top-level JSON must be an object")
            except Exception as exc:
                raise StructuredGenerationError(f"Explanation output invalid: {exc}")
            self._check_citations(payload, allowed_chunk_ids)
            return payload
        # Grounding runs INSIDE the bounded repair loop: a fabricated id is a
        # correctable failure (the model gets the corrective prompt naming the
        # bad ids) and, when the bound is exhausted, the same typed
        # StructuredGenerationError propagates to the caller unchanged.
        def _grounded(obj: Any) -> None:
            self._check_citations(obj.model_dump(), allowed_chunk_ids)

        validated = await generate_structured(
            self.llm, model, prompt, max_repairs=2, max_tokens=max_tokens,
            extra_validator=_grounded,
        )
        payload = validated.model_dump()
        self._check_citations(payload, allowed_chunk_ids)
        return payload

    async def _generate_with_retry(
        self,
        prompt: str,
        config: GenerationConfig,
        max_retries: int = 2
    ) -> str:
        """Generate content with retry on failure (kept for non-mode callers)."""
        for attempt in range(max_retries):
            try:
                result = await self.llm.generate(
                    prompt=prompt,
                    config=None  # Use default
                )
                return result.text
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Generation failed after {max_retries} attempts: {e}")
                time.sleep(1)  # Brief delay before retry

        return ""
    
    async def _perform_research(
        self,
        query: str,
        document_context: str
    ) -> Optional[str]:
        """
        Perform research augmentation using the dedicated service.
        
        This uses the ResearchAugmentationService which provides:
        - Query extraction
        - Tavily search
        - Jina content extraction
        - Summarization
        - Context merging with document priority
        """
        try:
            # Use the research service to augment context
            merged_context = await self.research.augment(
                user_query=query,
                document_context=document_context
            )
            
            if merged_context.has_research:
                return merged_context.research_context
            
            return None
            
        except Exception:
            return None
    
    async def generate(
        self,
        request: GenerationRequest,
        db: AsyncSession
    ) -> GenerationResult:
        """
        Generate content using the transformation engine.
        
        Args:
            request: Generation request with mode and parameters
            db: Database session
            
        Returns:
            GenerationResult with structured content
        """
        start_time = time.time()
        
        # Get configuration
        config = request.config or GenerationConfig(mode=request.mode)
        project_uuid = uuid.UUID(request.project_id)
        
        # Step 1: Retrieve relevant chunks
        retrieval_result = await retrieval_service.retrieve_with_db(
            query=request.query or "general content",
            db=db,
            project_id=request.project_id,
            document_id=request.document_id,
            top_k=config.max_context_chunks
        )
        
        # Extract chunk IDs for caching
        chunk_ids = [c.chunk_id for c in retrieval_result.chunks]
        
        # Step 2: Compute content hash for caching
        content_hash = self._compute_content_hash(
            project_id=request.project_id,
            mode=config.mode,
            chunk_ids=chunk_ids,
            config=config.model_dump()
        )
        
        # Step 3: Check cache
        cached_content = await self._check_cache(db, content_hash, project_uuid)
        if cached_content:
            return GenerationResult(
                project_id=request.project_id,
                mode=config.mode,
                content=cached_content,
                cached=True,
                used_research=False,
                tokens_used=0,
                processing_time=time.time() - start_time
            )
        
        # Step 4: Build context (notes follow the source document order:
        # page_start/position, so sections come out page 1 -> last instead
        # of relevance-ranked; other modes keep relevance order).
        context = self._build_context(
            retrieval_result,
            document_order=(config.mode == GenerationMode.NOTES),
        )
        
        # Step 4b: Trim context to fit token limits
        context = self._trim_context(context)
        
        # Step 5: Get prompt template (marks/style thread through for ANSWER).
        custom = dict(getattr(config, "custom_params", None) or {})
        prompt = PromptTemplates.get_prompt(
            mode=config.mode,
            context=context,
            query=request.query,
            marks=custom.get("marks", 5),
            style=custom.get("style", "detailed"),
        )
        
        # Step 6: Optionally augment with research using the dedicated service
        used_research = False
        research_note = ""
        if config.enable_research or request.enable_research:
            # Use the dedicated ResearchAugmentationService for proper:
            # - Query extraction
            # - Tavily search
            # - Jina content extraction
            # - Summarization
            # - Context merging with document priority
            research_result = await self._perform_research(
                request.query or "general overview",
                context
            )
            if research_result:
                # Append the research context with priority note
                prompt = augment_with_research(prompt, research_result)
                used_research = True
        
        # Step 7: Generate content (schema-enforced, bounded repair, T306)
        # with citation grounding (only ids present in the prompt, T307).
        allowed_ids = {c.chunk_id for c in retrieval_result.chunks}
        content = await self._generate_structured(config.mode, prompt, allowed_ids)

        # Step 7a: Phase-4 grounding gate + dedupe (strict, never silent).
        chunk_texts = {c.chunk_id: (c.content or "") for c in retrieval_result.chunks}
        gate = validate_grounding(content, allowed_ids, chunk_texts)
        content["_grounding_gate"] = gate
        if config.mode == GenerationMode.FLASHCARDS and isinstance(content.get("flashcards"), list):
            content["flashcards"] = dedupe_flashcards(content["flashcards"])
        if config.mode in (GenerationMode.QUIZ, GenerationMode.MCQ):
            key = "questions"
            if isinstance(content.get(key), list):
                content[key] = dedupe_mcqs(content[key])
        if config.mode == GenerationMode.ANSWER:
            words = len(str(content.get("answer_text", "")).split())
            content["_length_check"] = {
                "words": words,
                "min": content.get("target_min_words"),
                "max": content.get("target_max_words"),
                "ok": bool(
                    content.get("target_min_words") is not None
                    and content.get("target_max_words") is not None
                    and content["target_min_words"] <= words <= content["target_max_words"]
                ),
            }

        # Step 7b: Source attribution (T307/T303). Distinguish retrieved
        # (cited chunk ids backing this answer), generated (model-authored
        # structuring around retrieved spans), and unsupported (no
        # retrieval backing at all - never silently rendered as cited).
        if retrieval_result.chunks:
            support = "retrieved"
            support_note = (
                f"{len(allowed_ids)} retrieved chunk(s) back this answer; "
                "structuring text is model-generated."
            )
        else:
            support = "unsupported"
            support_note = (
                "No retrieved chunks backed this answer; it is model-generated "
                "and carries no citations."
            )
        content["citations"] = retrieval_result.citations()
        content["grounding"] = {
            "support": support,
            "note": support_note,
            "retrieved_chunk_ids": sorted(allowed_ids),
            "generated_without_citation": not bool(allowed_ids),
        }

        # Step 8: token accounting (prompt estimate + provider attribution).
        prompt_tokens = max(1, len(prompt) // 4)
        provider_name = getattr(self.llm, "last_provider", None) or getattr(
            self.llm, "model_name", "unknown"
        )
        content["attribution"] = {
            "provider": str(provider_name),
            "prompt_tokens_estimate": prompt_tokens,
        }

        # Step 8b: Persist domain rows served by the read endpoints and
        # study sessions (notes / flashcard decks / quizzes). The cache
        # alone is invisible to GET /api/generation/{notes,flashcards,quiz}
        # (Phase B live find: reads stayed not_generated after 200s).
        # Runs BEFORE the cache commit so a persist failure retries clean.
        await self._persist_domain_rows(db, project_uuid, config.mode, content)

        # Step 9: Save to cache
        output_type = OutputType(config.mode.value)
        await self._save_to_cache(
            db, content_hash, project_uuid, output_type, content
        )
        processing_time = time.time() - start_time

        return GenerationResult(
            project_id=request.project_id,
            mode=config.mode,
            content=content,
            cached=False,
            used_research=used_research,
            tokens_used=content.get("attribution", {}).get("prompt_tokens_estimate", 0),
            processing_time=processing_time
        )
    
    async def generate_notes(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False
    ) -> GenerationResult:
        """Convenience method for generating notes"""
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.NOTES,
            query=query,
            enable_research=enable_research
        )
        return await self.generate(request, db)
    
    async def generate_flashcards(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False
    ) -> GenerationResult:
        """Convenience method for generating flashcards"""
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.FLASHCARDS,
            query=query,
            enable_research=enable_research
        )
        return await self.generate(request, db)
    
    async def generate_quiz(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False
    ) -> GenerationResult:
        """Convenience method for generating quiz"""
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.QUIZ,
            query=query,
            enable_research=enable_research
        )
        return await self.generate(request, db)
    
    async def generate_explanation(
        self,
        project_id: str,
        db: AsyncSession,
        query: str,
        enable_research: bool = False
    ) -> GenerationResult:
        """Convenience method for generating explanation"""
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.EXPLANATION,
            query=query,
            enable_research=enable_research
        )
        return await self.generate(request, db)
    
    async def generate_summary(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False
    ) -> GenerationResult:
        """Convenience method for generating summary"""
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.SUMMARY,
            query=query,
            enable_research=enable_research
        )
        return await self.generate(request, db)
    
    async def generate_website(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False
    ) -> GenerationResult:
        """Convenience method for generating website content"""
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.WEBSITE,
            query=query,
            enable_research=enable_research
        )
        return await self.generate(request, db)

    async def generate_answer(
        self,
        project_id: str,
        db: AsyncSession,
        query: str,
        marks: int = 5,
        style: str = "detailed",
        enable_research: bool = False,
    ) -> GenerationResult:
        """Phase-4 marks-aware grounded answer (constraints adapt by marks)."""
        band = constraints_for_marks(int(marks))
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.ANSWER,
            query=query,
            enable_research=enable_research,
            config=GenerationConfig(
                mode=GenerationMode.ANSWER,
                custom_params={"marks": int(marks), "style": style,
                               "min_words": band["min_words"],
                               "max_words": band["max_words"]},
            ),
        )
        return await self.generate(request, db)

    async def generate_mcq(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False,
    ) -> GenerationResult:
        """Phase-4 strict MCQ set (exactly 4 options, validated)."""
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.MCQ,
            query=query,
            enable_research=enable_research,
        )
        return await self.generate(request, db)

    # -- Academic Intelligence vertical -----------------------------------

    async def _latest_cached_output(
        self,
        db: AsyncSession,
        project_uuid,
        output_type: "OutputType",
    ) -> Optional[Dict[str, Any]]:
        """Latest cached payload for an academic output type (or None)."""
        from sqlalchemy import select

        from database.models import GeneratedContentCache

        rows = (
            await db.execute(
                select(GeneratedContentCache)
                .where(
                    GeneratedContentCache.project_id == project_uuid,
                    GeneratedContentCache.output_type == output_type,
                )
                .order_by(GeneratedContentCache.created_at.desc())
                .limit(1)
            )
        ).scalars().all()
        if not rows or not rows[0].output_content:
            return None
        content = dict(rows[0].output_content)
        content.pop("_grounding_gate", None)
        content.pop("_length_check", None)
        content.pop("attribution", None)
        content.pop("grounding", None)
        content.pop("citations", None)
        return content


    async def generate_syllabus(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False,
    ) -> GenerationResult:
        """Academic: extract syllabus units+topics (grounded, deduplicated)."""
        scoped = await _scoped_document_id(db, project_id, "syllabus")
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.SYLLABUS,
            query=query,
            document_id=scoped,
            enable_research=enable_research,
        )
        result = await self.generate(request, db)
        result.content = self._validate_academic(Syllabus, result.content)
        return result

    async def generate_pyq(
        self,
        project_id: str,
        db: AsyncSession,
        query: str = None,
        enable_research: bool = False,
    ) -> GenerationResult:
        """Academic: extract PYQs with year/marks (grounded, deduplicated)."""
        scoped = await _scoped_document_id(db, project_id, "question_paper")
        request = GenerationRequest(
            project_id=project_id,
            mode=GenerationMode.PYQ,
            query=query,
            document_id=scoped,
            enable_research=enable_research,
        )
        result = await self.generate(request, db)
        if isinstance(result.content.get("questions"), list):
            result.content["questions"] = dedupe_pyqs(result.content["questions"])
        result.content = self._validate_academic(PYQSet, result.content)
        return result

    @staticmethod
    def _validate_academic(model, content: Dict[str, Any]) -> Dict[str, Any]:
        """Bridge the shared envelope to strict academic schemas.

        generate() attaches the standard envelope (retrieval-shaped
        top-level citations, grounding, attribution, gates); strict
        academic models are extra=forbid, so the envelope is normalised
        (retrieval citations -> CitationRef shape, grounding -> provenance)
        before validation and re-attached afterwards. Nested per-unit /
        per-question model citations are untouched.
        """
        raw = dict(content)
        extra: Dict[str, Any] = {}
        for key in ("_grounding_gate", "_length_check", "attribution", "grounding"):
            if key in raw:
                extra[key] = raw.pop(key)
        raw_cits = raw.pop("citations", []) or []
        grounding = extra.get("grounding") or {}
        norm_cits = []
        for c in raw_cits:
            if isinstance(c, dict) and c.get("chunk_id"):
                norm_cits.append({
                    "chunk_id": str(c["chunk_id"]),
                    "document_id": c.get("document_id") or c.get("doc_id"),
                    "page": c.get("page"),
                    "span": str(c.get("span", "") or ""),
                })
        payload = dict(raw)
        payload.setdefault("citations", norm_cits)
        if not payload.get("provenance"):
            payload["provenance"] = {
                "support": grounding.get("support", "generated"),
                "retrieved_chunk_ids": grounding.get("retrieved_chunk_ids", []),
                "generated_without_citation": grounding.get(
                    "generated_without_citation", True),
                "note": grounding.get("note", "Academic extraction."),
            }
        validated = model.model_validate(payload)
        merged = validated.model_dump()
        merged.update(extra)
        return merged

    async def generate_mapping(
        self,
        project_id: str,
        db: AsyncSession,
        syllabus: Optional[Dict[str, Any]] = None,
        questions: Optional[list] = None,
        min_score: float = 0.15,
    ) -> GenerationResult:
        """Academic: deterministic PYQ -> syllabus unit mapping.

        Uses the project's latest cached syllabus/PYQ when not supplied
        inline. No LLM call: token-overlap scoring (reproducible); below
        threshold stays unmapped (honest, never forced).
        """
        import time as _time

        start = _time.time()
        project_uuid = uuid.UUID(project_id)
        if syllabus is None:
            from database.models import OutputType as _OT

            syllabus = await self._latest_cached_output(db, project_uuid, _OT("syllabus"))
        if questions is None:
            from database.models import OutputType as _OT

            pyq = await self._latest_cached_output(db, project_uuid, _OT("pyq"))
            questions = (pyq or {}).get("questions", [])
        if not syllabus or not syllabus.get("units"):
            raise ValueError("No syllabus available for this project: extract a syllabus first.")
        if not questions:
            raise ValueError("No PYQs available for this project: extract PYQs first.")
        units = syllabus["units"]
        mapped = map_questions_to_units(questions, units, min_score=min_score)
        payload = {
            "syllabus_title": str(syllabus.get("title", "")),
            "mappings": mapped["mappings"],
            "unmapped_count": mapped["unmapped_count"],
            "coverage_pct": mapped["coverage_pct"],
            "citations": [],
            "provenance": {
                "support": "generated",
                "retrieved_chunk_ids": [],
                "generated_without_citation": True,
                "note": (
                    "Deterministic token-overlap mapping over the project's "
                    "extracted syllabus and PYQs; no retrieval backing."
                ),
            },
        }
        validated = MappingResult.model_validate(payload)
        content = validated.model_dump()
        from database.models import OutputType as _OT2

        await self._save_to_cache(
            db,
            self._compute_content_hash(project_id, GenerationMode.MAPPING, [],
                                       {"min_score": min_score}),
            project_uuid, _OT2("mapping"), content,
        )
        return GenerationResult(
            project_id=project_id, mode=GenerationMode.MAPPING, content=content,
            cached=False, used_research=False, tokens_used=0,
            processing_time=_time.time() - start,
        )

    async def generate_analysis(
        self,
        project_id: str,
        db: AsyncSession,
        syllabus: Optional[Dict[str, Any]] = None,
        questions: Optional[list] = None,
        mappings: Optional[list] = None,
        min_score: float = 0.15,
    ) -> GenerationResult:
        """Academic: frequency + marks alignment over mapped PYQs.

        Deterministic over the project's latest cached mapping (or a
        fresh mapping when none is cached). No LLM call, no fake rows:
        zero-question units are included with count 0.
        """
        import time as _time

        start = _time.time()
        project_uuid = uuid.UUID(project_id)
        from database.models import OutputType as _OT

        if syllabus is None:
            syllabus = await self._latest_cached_output(db, project_uuid, _OT("syllabus"))
        if questions is None:
            pyq = await self._latest_cached_output(db, project_uuid, _OT("pyq"))
            questions = (pyq or {}).get("questions", [])
        if mappings is None:
            cached_map = await self._latest_cached_output(db, project_uuid, _OT("mapping"))
            mappings = (cached_map or {}).get("mappings")
        if not syllabus or not syllabus.get("units"):
            raise ValueError("No syllabus available for this project: extract a syllabus first.")
        if not questions:
            raise ValueError("No PYQs available for this project: extract PYQs first.")
        if not mappings:
            mapped = map_questions_to_units(questions, syllabus["units"], min_score=min_score)
            mappings = mapped["mappings"]
        freq = compute_frequency(mappings, syllabus["units"], questions)
        payload = {
            "syllabus_title": str(syllabus.get("title", "")),
            "total_questions": freq["total_questions"],
            "mapped_questions": freq["mapped_questions"],
            "unmapped_questions": freq["unmapped_questions"],
            "frequency": freq["frequency"],
            "total_marks": freq["total_marks"],
            "coverage_pct": freq["coverage_pct"],
            "citations": [],
            "provenance": {
                "support": "generated",
                "retrieved_chunk_ids": [],
                "generated_without_citation": True,
                "note": (
                    "Deterministic frequency/marks aggregation over the "
                    "project's mapped PYQs; no retrieval backing."
                ),
            },
        }
        validated = AnalysisResult.model_validate(payload)
        content = validated.model_dump()
        await self._save_to_cache(
            db,
            self._compute_content_hash(project_id, GenerationMode.ANALYSIS, [],
                                       {"min_score": min_score}),
            project_uuid, _OT("analysis"), content,
        )
        return GenerationResult(
            project_id=project_id, mode=GenerationMode.ANALYSIS, content=content,
            cached=False, used_research=False, tokens_used=0,
            processing_time=_time.time() - start,
        )

    async def generate_strict(
        self,
        project_id: str,
        db: AsyncSession,
        mode: GenerationMode,
        query: str = None,
        enable_research: bool = False,
        **prompt_kwargs,
    ) -> GenerationResult:
        """Phase-4 strict production path: validates against STRICT_MODELS.

        Falls back to the compat MODE_MODELS shape when a mode has no
        strict schema. Raises StructuredGenerationError on failure
        (max 2 repairs), never loosely parsed text.
        """
        # Reuse the full pipeline (retrieval/cache/grounding) via generate,
        # then re-validate the payload against the strict schema so the
        # caller gets a typed model instance, not a bare dict.
        request = GenerationRequest(
            project_id=project_id, mode=mode, query=query,
            enable_research=enable_research,
        )
        result = await self.generate(request, db)
        model = self.STRICT_MODELS.get(mode)
        if model is None:
            return result
        validated = model.model_validate(result.content)
        result.content = validated.model_dump()
        return result


# Default engine instance
transformation_engine = TransformationEngine()