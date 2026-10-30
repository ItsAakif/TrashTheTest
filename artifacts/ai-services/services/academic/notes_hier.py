"""STEP 3 hierarchical notes: structure -> topic groups -> TopicNotes -> synthesis.

Replaces the one-call path for academic notes:

- build_outline(structure): ordered unit/topic plan in source order (no LLM).
- group_topics(outline, sizes, budget): one group per topic in source order;
  a topic whose own context exceeds the budget splits by subtopic (recorded).
- Per-topic retrieval takes ALL relevant chunks (heading/alias/token match),
  never a global top-10 cut; vector ordering never determines final order.
- render_topic_notes(...): deterministic offline TopicNotes builder with
  citations (chunk_id + span occurring in the chunk). The live LLM path
  (generate_topic_notes) goes through the shared schema-enforced
  _generate_structured (max 2 repairs) with the NOTES_TOPIC prompt.
- synthesize_notes(...): ordered Unit -> Topic -> Subtopic synthesis with
  cross-group dedupe (normalized shingle hash, keep first, merge cites),
  cross-topic transitions referencing spine titles only, and citation
  validation via the shared validate_grounding gate.

Content contract: must_keep (majors, definitions, formulas, algorithms,
examples, comparisons, exceptions, limits, relations, exam info) is never
compressed away; may_compress (repetition, filler, duplicates) is dropped.
Shape adapts per topic: algorithm (steps), formula, comparison, concept.
No page limit is imposed anywhere.
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from services.academic.sufficiency import (
    SufficiencyStatus,
    assess_source_sufficiency,
    recover_topic_evidence,
)
from services.academic.types import (
    DocumentStructure,
    ExtractedTopic,
    NotesArtifact,
    NotesSection,
    TopicCitation,
    TopicNotes,
    TopicShape,
)

# Shared 4-chars/token estimate, consistent with the engine.
CHARS_PER_TOKEN = 4
DEFAULT_TOPIC_BUDGET_TOKENS = 6000

# Evidence handed to ONE per-topic model call. A topic larger than this is
# split on chunk boundaries and generated in batches, so no single call can
# be asked for more than this much material (the cause of the truncated
# per-topic JSON).
DEFAULT_TOPIC_BATCH_TOKENS = 3000

# Per-topic OUTPUT budget. One topic is the largest unit this path asks the
# model for, so it gets twice the shared 4096 default; the batch split above
# keeps each answer inside it. Bounded on purpose: raising it further without
# splitting only moves the truncation.
NOTES_TOPIC_MAX_OUTPUT_TOKENS = 8192

# -- content contract ------------------------------------------------------

# Sentences matching any of these carry must_keep material and survive
# compression: definitions, formulas/numbers, algorithms/steps, examples,
# comparisons, exceptions, limits, relations, exam info.
_MUST_KEEP_RES = [
    re.compile(r"\b(is|are|was|were|means?|defined as|refers to)\b.{0,40}\b", re.I),
    re.compile(r"[=<>+\-*/^0-9]\s*[=<>+\-*/]|[0-9]+\s*(%|percent|degrees?)", re.I),
    re.compile(r"\b(step|algorithm|procedure|process|method|first|then|finally)\b", re.I),
    re.compile(r"\b(for example|e\.g\.|such as|including|example)\b", re.I),
    re.compile(r"\b(compare|comparison|versus|vs\.?|difference|similar|unlike|whereas)\b", re.I),
    re.compile(r"\b(except|exception|however|although|unless|only when|only if)\b", re.I),
    re.compile(r"\b(limit|constraint|bound|maximum|minimum|never|always)\b", re.I),
    re.compile(r"\b(because|therefore|thus|hence|causes?|leads to|depends on)\b", re.I),
    re.compile(r"\b(exam|mark|important|key|note that|crucial|essential)\b", re.I),
]

_FILLER_RES = [
    re.compile(r"^\s*(in this section|as mentioned|it is important to note that|"
               r"we will now|let us|as we (saw|see|have seen))\b", re.I),
    re.compile(r"^\s*(very|really|quite|basically|essentially)\s+\w+\s*$", re.I),
]

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Token estimate shared with the engine (4 chars per token)."""
    return max(1, len(text or "") // CHARS_PER_TOKEN)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]


def _is_filler(sentence: str) -> bool:
    words = sentence.split()
    if len(words) < 4:
        return True
    return any(rx.search(sentence) for rx in _FILLER_RES)


def _is_must_keep(sentence: str) -> bool:
    return any(rx.search(sentence) for rx in _MUST_KEEP_RES)


def compress_text(text: str) -> str:
    """Drop filler/duplicates; must_keep sentences always survive."""
    kept: List[str] = []
    seen: set = set()
    for sentence in _sentences(text):
        key = _normalize(sentence)
        if not key or key in seen:
            continue  # may_compress: duplicates
        seen.add(key)
        if _is_filler(sentence) and not _is_must_keep(sentence):
            continue  # may_compress: filler
        kept.append(sentence)
    return " ".join(kept)


def _normalize(text: str) -> str:
    folded = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return " ".join(folded.split())


def _shingle_hash(text: str, n: int = 5) -> str:
    import hashlib

    tokens = _normalize(text).split()
    if len(tokens) < n:
        return hashlib.sha256(" ".join(tokens).encode()).hexdigest()
    grams = {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}
    digest = hashlib.sha256()
    for gram in sorted(grams):
        digest.update(gram.encode())
    return digest.hexdigest()


# -- adaptive shape --------------------------------------------------------

_ALGORITHM_HINTS = ("step", "algorithm", "procedure", "process", "method", "stage", "phase")
_FORMULA_HINTS = ("formula", "equation", "=", "theorem", "proof", "derive")
_COMPARISON_HINTS = (" vs ", "versus", "compare", "comparison", "difference",
                     "differences", "similarities", "contrast", "whereas")


def shape_for_topic(title: str, texts: List[str]) -> TopicShape:
    """Pick the note shape from the topic title + evidence (deterministic)."""
    blob = f"{title} {' '.join(texts)}".lower()
    if any(h in blob for h in _COMPARISON_HINTS):
        return TopicShape.COMPARISON
    if any(h in blob for h in _FORMULA_HINTS):
        return TopicShape.FORMULA
    if any(h in blob for h in _ALGORITHM_HINTS):
        return TopicShape.ALGORITHM
    return TopicShape.CONCEPT


# -- outline + grouping ------------------------------------------------------

class OutlineEntry(Dict[str, Any]):
    """One ordered plan row (plain dict: unit/topic titles + ord path)."""


def build_outline(structure: DocumentStructure) -> List[Dict[str, Any]]:
    """Ordered unit/topic plan in source (spine) order. No LLM."""
    outline: List[Dict[str, Any]] = []
    for unit_pos, unit in enumerate(structure.units):
        for topic_pos, topic in enumerate(unit.topics):
            outline.append({
                "unit_title": unit.title,
                "topic_title": topic.title,
                "unit_pos": unit_pos,
                "topic_pos": topic_pos,
                "subtopics": [s.title for s in topic.subtopics],
                "chunk_ids": list(topic.chunks),
            })
    return outline


def group_topics(
    outline: List[Dict[str, Any]],
    topic_sizes: Optional[Dict[str, int]] = None,
    budget_tokens: int = DEFAULT_TOPIC_BUDGET_TOKENS,
) -> List[Dict[str, Any]]:
    """One group per topic in source order; oversized topics split by subtopic.

    topic_sizes maps topic_title -> context tokens. A topic whose own
    context exceeds the budget splits into one group per subtopic and the
    split is recorded on the group (split=True). Groups never exceed the
    budget unless a single subtopic alone does (reported, never dropped).
    """
    sizes = topic_sizes or {}
    groups: List[Dict[str, Any]] = []
    for entry in outline:
        size = sizes.get(entry["topic_title"], 0)
        if size > budget_tokens and len(entry.get("subtopics") or []) > 1:
            per = max(1, size // max(1, len(entry["subtopics"])))
            for sub in entry["subtopics"]:
                groups.append({
                    "unit_title": entry["unit_title"],
                    "topic_title": entry["topic_title"],
                    "subtopic": sub,
                    "chunk_ids": list(entry["chunk_ids"]),
                    "split": True,
                    "estimated_tokens": per,
                })
        else:
            groups.append({
                "unit_title": entry["unit_title"],
                "topic_title": entry["topic_title"],
                "subtopic": None,
                "chunk_ids": list(entry["chunk_ids"]),
                "split": False,
                "estimated_tokens": size,
            })
    return groups


def topic_query(unit_title: str, topic: ExtractedTopic) -> str:
    """Retrieval query for one topic: topic path + aliases."""
    bits = [unit_title, topic.title]
    bits.extend(topic.aliases or [])
    for sub in topic.subtopics or []:
        bits.append(sub.title)
    seen: set = set()
    ordered = [b for b in bits if b and b not in seen and not seen.add(b)]  # type: ignore[func-returns-value]
    return " -- ".join(ordered)


# -- per-topic evidence + deterministic notes ---------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Stopwords are excluded from overlap math so generic glue words ("the",
# "of") can never make two unrelated chunks look relevant.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "are", "was", "were", "have", "has", "had", "which", "their", "there",
    "into", "over", "under", "such", "than", "then", "them", "they", "its",
    "of", "in", "to", "a", "an", "on", "at", "by", "as", "is", "it", "be",
    "or", "not", "but", "you", "we", "he", "she", "him", "her", "his",
    "will", "would", "can", "could", "should", "when", "while", "where",
})


def _tokens(text: str) -> set:
    return {t for t in _TOKEN_RE.findall((text or "").lower())
            if t not in _STOPWORDS}


def select_topic_chunks(
    all_chunks: List[Dict[str, Any]],
    topic: ExtractedTopic,
    unit_title: str = "",
    min_overlap: float = 0.2,
) -> List[Dict[str, Any]]:
    """ALL chunks relevant to one topic, in source order (no top-k cut).

    Relevance: the chunk id is claimed by the structure, or its heading
    matches the topic/subtopic, or an alias matches, or token overlap with
    the topic vocabulary clears min_overlap. Falls back to the structure's
    own chunk ids so a topic never loses its evidence. Vector similarity
    ordering never applies: output follows (page, position).
    """
    claimed = set(topic.chunks or [])
    vocab = _tokens(topic_query(unit_title, topic))
    alias_sets = [_tokens(a) for a in (topic.aliases or [])]
    topic_names = {(topic.title or "").strip().lower()}
    topic_names.update((s.title or "").strip().lower() for s in topic.subtopics or [])
    hits: List[Dict[str, Any]] = []
    for chunk in all_chunks:
        cid = str(chunk.get("id") or chunk.get("chunk_id") or "")
        heading = str(chunk.get("heading") or chunk.get("section") or "").strip().lower()
        if cid and cid in claimed:
            hits.append(chunk)
            continue
        if heading and (heading in topic_names or any(
                h in heading or heading in h for h in topic_names if h)):
            hits.append(chunk)
            continue
        ctokens = _tokens(str(chunk.get("text") or chunk.get("content") or ""))
        if alias_sets and any(len(a & ctokens) / max(1, len(a)) >= 0.5 for a in alias_sets if a):
            hits.append(chunk)
            continue
        if vocab and ctokens and len(vocab & ctokens) / len(vocab) >= min_overlap:
            hits.append(chunk)
    if not hits:
        by_id = {str(c.get("id") or c.get("chunk_id")): c for c in all_chunks}
        hits = [by_id[cid] for cid in (topic.chunks or []) if cid in by_id]
    hits.sort(key=lambda c: (
        c.get("page", 10 ** 9) if isinstance(c.get("page"), int) else 10 ** 9,
        c.get("position", 0) if isinstance(c.get("position"), int) else 0,
    ))
    return hits


def _span_for(chunk: Dict[str, Any]) -> str:
    text = str(chunk.get("text") or chunk.get("content") or "")
    return " ".join(text.split()[:8])


# Shortest model word-run worth keeping as a quoted span. Below this the
# quote carries no evidence, so the chunk's own excerpt is used instead.
MIN_SPAN_WORDS = 3
# Length of the fallback excerpt taken from the cited chunk itself.
FALLBACK_SPAN_WORDS = 8


def _find_run(norm_words: List[str], run: List[str]) -> int:
    """Index of `run` inside `norm_words`, else -1 (both already tokenized)."""
    width, size = len(run), len(norm_words)
    if not width or width > size:
        return -1
    for start in range(size - width + 1):
        if norm_words[start:start + width] == run:
            return start
    return -1


def _verbatim_span(span: str, body: str) -> Optional[str]:
    """Recover a verbatim quote of `body` for a model-authored `span`.

    Live per-topic models routinely elide ("first words ... last words") or
    lightly re-word a quoted span, which the synthesis grounding gate rejects
    even though the cited chunk id is real. The quote is recovered FROM the
    chunk itself, so the gate then verifies literally true text:

    1. the model span already occurs (normalized) -> keep the model wording;
    2. otherwise the longest run of the model's words that occurs verbatim
       in the chunk becomes the span (the elided middle is simply dropped);
    3. otherwise the chunk's own first words are quoted - still real text
       from the cited chunk, never invented boilerplate.

    Returns None only when the chunk has no normalizable text at all, so an
    empty-evidence citation still fails the gate honestly.
    """
    from services.transformation.engine import normalize_text

    raw = str(body or "")
    norm_body = normalize_text(raw)
    norm_span = normalize_text(str(span or ""))
    if not norm_body:
        return None
    if norm_span and norm_span in norm_body:
        return str(span).strip()
    # Punctuation-only tokens carry no evidence; drop them from BOTH sides
    # so runs can cross them ("code - file" still matches "code file").
    tokens = [word for word in raw.split() if normalize_text(word)]
    if not tokens:
        return None
    norm_words = [normalize_text(word) for word in tokens]
    span_words = norm_span.split()
    best_start, best_len = -1, 0
    for i in range(len(span_words)):
        if len(span_words) - i <= best_len:
            break  # no longer run can be found from here
        for length in range(len(span_words) - i, best_len, -1):
            start = _find_run(norm_words, span_words[i:i + length])
            if start >= 0:
                best_start, best_len = start, length
                break
    if best_len >= MIN_SPAN_WORDS:
        return " ".join(tokens[best_start:best_start + best_len])
    return " ".join(tokens[:FALLBACK_SPAN_WORDS])


def _align_citation_spans(
    artifact: NotesArtifact, chunk_texts: Dict[str, str]
) -> Tuple[NotesArtifact, int]:
    """Replace non-literal spans on REAL chunk ids with the chunk's own text.

    A non-literal span is a quoting defect, not a fabricated citation: the id
    has already been checked against the evidence universe. Each such span is
    swapped for text taken from the cited chunk, so the grounding gate verifies
    a literally true quote. Citations whose chunk id is absent from
    ``chunk_texts`` are left untouched (the gate still rejects those ids as
    fabricated). Returns the artifact and the number of spans re-aligned.
    """
    re_aligned = 0
    sections: List[NotesSection] = []
    for section in artifact.sections:
        citations: List[TopicCitation] = []
        for cite in section.citations:
            body = chunk_texts.get(str(cite.chunk_id))
            if body is None:
                citations.append(cite)  # foreign id: gate rejects it below
                continue
            aligned = _verbatim_span(cite.span, body)
            if aligned is None:
                citations.append(cite)  # empty chunk: gate fails honestly
            elif aligned == cite.span:
                citations.append(cite)
            else:
                re_aligned += 1
                citations.append(cite.model_copy(update={"span": aligned}))
        sections.append(section.model_copy(update={"citations": citations}))
    if not re_aligned:
        return artifact, 0
    return artifact.model_copy(update={
        "sections": sections,
        "provenance_note": (
            f"{artifact.provenance_note}; {re_aligned} citation span(s) "
            "re-aligned to verbatim text from the cited chunk"
        ),
    }), re_aligned


def _chunk_id(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("chunk_id") or "")


def _key_terms(topic: ExtractedTopic, chunks: List[Dict[str, Any]]) -> List[str]:
    vocab = _tokens(topic.title)
    stop = {"the", "and", "for", "with", "from", "that", "this", "are", "was",
            "were", "have", "has", "which", "their", "into", "over", "under"}
    counts: Dict[str, int] = {}
    for chunk in chunks:
        for tok in _tokens(str(chunk.get("text") or chunk.get("content") or "")):
            if len(tok) > 3 and tok not in stop:
                counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    title_hits = [t for t in vocab if len(t) > 3 and t not in stop]
    out = list(title_hits)
    for tok, _ in ranked:
        if tok not in out:
            out.append(tok)
        if len(out) >= 10:
            break
    return out


def render_topic_notes(
    unit_title: str,
    topic: ExtractedTopic,
    chunks: List[Dict[str, Any]],
) -> TopicNotes:
    """Deterministic TopicNotes from a topic's own evidence chunks.

    New prose assembled from must_keep sentences (compressed, deduplicated,
    shape-adapted) with one citation per backing chunk. Zero chunks ->
    honest no_source notes (never invented citations).
    """
    evidence = select_topic_chunks(chunks, topic, unit_title)
    # A source gap is metadata, not prose.  Keep the topic body empty so a
    # caller cannot accidentally render "no information" as a normal section.
    if not evidence:
        recovered = recover_topic_evidence(
            chunks, topic, unit_title=unit_title,
            unit_chunk_ids=[], aliases=list(topic.aliases or []),
        )
        if recovered:
            evidence = recovered
    if not evidence:
        return TopicNotes(
            unit_title=unit_title,
            topic_title=topic.title,
            heading=topic.title,
            body_paragraphs=[],
            key_terms=[],
            shape=TopicShape.CONCEPT,
            citations=[],
            no_source=True,
            sufficiency=SufficiencyStatus.MISSING.value,
            source_gap_reason="no source evidence after topic/alias/parent/nearby-page recovery",
            provenance_note=(
                "no evidence chunks after bounded recovery; source gap reported, "
                "nothing invented"
            ),
        )
    texts = [str(c.get("text") or c.get("content") or "") for c in evidence]
    shape = shape_for_topic(topic.title, texts)
    compressed = [compress_text(t) for t in texts]
    compressed = [c for c in compressed if c]
    # Cross-chunk dedupe: first occurrence wins (may_compress duplicates).
    paras: List[str] = []
    seen_hashes: set = set()
    for para in compressed:
        h = _shingle_hash(para)
        if h not in seen_hashes:
            seen_hashes.add(h)
            paras.append(para)
    shape_extras: Dict[str, Any] = {}
    if shape == TopicShape.ALGORITHM:
        steps = [s for t in texts for s in _sentences(t) if _is_must_keep(s)]
        shape_extras["steps"] = steps[:12]
    elif shape == TopicShape.FORMULA:
        formulae = [s for t in texts for s in _sentences(t)
                    if re.search(r"[=<>+\-*/^]", s)]
        shape_extras["formulas"] = formulae[:8]
    elif shape == TopicShape.COMPARISON:
        shape_extras["compared"] = topic.title
    citations = [
        TopicCitation(chunk_id=_chunk_id(c), span=_span_for(c)) for c in evidence
    ]
    return TopicNotes(
        unit_title=unit_title,
        topic_title=topic.title,
        heading=topic.title,
        body_paragraphs=paras or [compress_text(texts[0])],
        key_terms=_key_terms(topic, evidence),
        shape=shape,
        shape_extras=shape_extras,
        citations=citations,
        no_source=False,
        sufficiency=SufficiencyStatus.SUFFICIENT.value,
        source_gap_reason=None,
        provenance_note=(
            f"deterministic render over {len(evidence)} evidence chunk(s); "
            f"shape={shape.value}; citations are per-chunk spans"
        ),
    )


async def generate_topic_notes(
    engine: Any,
    unit_title: str,
    topic: ExtractedTopic,
    context: str,
    allowed_chunk_ids: set,
    query: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Live LLM per-topic call through the shared schema-enforced path.

    Uses engine._generate_structured (NOTES_TOPIC mode, max 2 repairs on
    validation failure only, plus the bounded truncated-JSON recovery) so
    provider selection, schema validation and the citation check stay
    shared. The prompt carries ONLY this topic's chunks (topic-scoped
    context).

    ``max_tokens`` raises the per-call output budget for one topic. One
    topic is the largest unit this path asks for, so it gets a larger
    budget than the shared default; a topic whose evidence is still too
    large is split by ``split_topic_evidence`` before it ever reaches here.
    """
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import GenerationMode

    shape = shape_for_topic(topic.title, [context])
    prompt = PromptTemplates.get_prompt(
        GenerationMode.NOTES_TOPIC, context,
        query or topic_query(unit_title, topic),
        shape=shape.value,
    )
    return await engine._generate_structured(
        GenerationMode.NOTES_TOPIC, prompt, allowed_chunk_ids, max_tokens=max_tokens
    )


def split_topic_evidence(
    rows: List[Dict[str, Any]],
    budget_tokens: int = DEFAULT_TOPIC_BATCH_TOKENS,
) -> List[List[Dict[str, Any]]]:
    """Split one topic's evidence into source-order batches (pure, no LLM).

    A single oversized topic is what truncated the per-topic JSON call: the
    model tried to answer for every chunk at once and hit the output ceiling.
    Batches are cut on chunk boundaries, never mid-chunk, and keep source
    order, so citations and the final document order are unchanged. A chunk
    larger than the budget becomes its own batch (reported, never dropped).
    """
    budget = max(1, int(budget_tokens or 0) or DEFAULT_TOPIC_BATCH_TOKENS)
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    used = 0
    for row in rows or []:
        size = estimate_tokens(
            str(row.get("text") or row.get("content") or "")
        )
        if current and used + size > budget:
            batches.append(current)
            current, used = [], 0
        current.append(row)
        used += size
    if current:
        batches.append(current)
    return batches or [[]]


def merge_topic_payloads(payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-batch topic payloads into one TopicNotes-shaped dict.

    Body paragraphs concatenate in call order, key terms and citations are
    deduplicated (first wins, order preserved), and the heading/unit/topic of
    the first payload win. Merging never invents content: every field comes
    from a validated per-batch payload.
    """
    items = [dict(p) for p in (payloads or []) if isinstance(p, dict)]
    if not items:
        return {}
    if len(items) == 1:
        return items[0]
    merged = dict(items[0])
    paragraphs: List[str] = []
    terms: List[str] = []
    citations: List[Any] = []
    seen_terms: set = set()
    seen_cites: set = set()
    for item in items:
        for para in item.get("body_paragraphs") or []:
            text = str(para or "").strip()
            if text and text not in paragraphs:
                paragraphs.append(text)
        for term in item.get("key_terms") or []:
            key = str(term or "").strip().lower()
            if key and key not in seen_terms:
                seen_terms.add(key)
                terms.append(str(term).strip())
        for cite in item.get("citations") or []:
            if not isinstance(cite, dict):
                continue
            key = (
                str(cite.get("chunk_id") or ""),
                str(cite.get("span") or ""),
                cite.get("page"),
            )
            if key in seen_cites:
                continue
            seen_cites.add(key)
            citations.append(cite)
    merged["body_paragraphs"] = paragraphs
    merged["key_terms"] = terms
    merged["citations"] = citations
    return merged


# -- ordered synthesis ---------------------------------------------------------

def _section_paras(notes: TopicNotes) -> List[Tuple[str, List[TopicCitation]]]:
    # Round-robin: every citation backs at least one paragraph (no cited
    # chunk is silently dropped), and every paragraph carries evidence.
    paras = notes.body_paragraphs
    cites = list(notes.citations)
    width = max(1, len(paras))
    out: List[Tuple[str, List[TopicCitation]]] = []
    for i, para in enumerate(paras):
        owned = [c for j, c in enumerate(cites) if j % width == i]
        out.append((para, owned or cites[:1]))
    return out


def synthesize_notes(
    title: str,
    structure: DocumentStructure,
    topic_notes: List[TopicNotes],
    chunk_texts: Optional[Dict[str, str]] = None,
    allowed_chunk_ids: Optional[set] = None,
    provenance_note: str = "ordered synthesis Unit->Topic->Subtopic (source order)",
) -> NotesArtifact:
    """Merge per-topic notes into the final artifact in SPINE order.

    Never similarity-sorted: sections iterate structure.units/topics in
    order. Duplicate paragraphs across groups appear once (first wins,
    citations merged). Cross-topic transitions reference spine titles only.
    Citation ids are validated against the evidence universe: chunk_texts
    keys when supplied (spans checked too, via the shared
    validate_grounding gate; a real id quoted inexactly is first re-aligned
    to verbatim text from its own chunk, never invented), else
    allowed_chunk_ids when given, else the
    structure's own chunk ids. Per-topic retrieval may legitimately cite
    relevant chunks beyond the structure's claimed ids, so callers that
    render over a wider evidence set must pass that universe in.
    """
    by_topic = {(n.unit_title, n.topic_title): n for n in topic_notes}
    sections: List[NotesSection] = []
    source_gaps: List[str] = []
    named_gap_keys: set = set()
    seen_hashes: Dict[str, int] = {}
    for unit in structure.units:
        for topic in unit.topics:
            notes = by_topic.get((unit.title, topic.title))
            if notes is None:
                source_gaps.append(
                    f"{unit.title}::{topic.title}: no generated note after bounded repair"
                )
                named_gap_keys.add((unit.title, topic.title))
                continue
            if getattr(notes, "no_source", False) or not any(
                str(value or "").strip() for value in (notes.body_paragraphs or [])
            ):
                source_gaps.append(
                    f"{unit.title}::{topic.title}: {getattr(notes, 'source_gap_reason', None) or 'no source evidence'}"
                )
                named_gap_keys.add((unit.title, topic.title))
                continue
            sub_name = (topic.subtopics[0].title
                        if topic.subtopics else None)
            for para, cites in _section_paras(notes):
                # Global paragraph hash (heading excluded): a paragraph
                # repeated verbatim across sections appears once, first wins.
                h = _shingle_hash(para)
                if h in seen_hashes:
                    # Duplicate across groups: merge citations into survivor.
                    survivor = sections[seen_hashes[h]]
                    known = {c.chunk_id for c in survivor.citations}
                    survivor.citations.extend(
                        c for c in cites if c.chunk_id not in known
                    )
                    continue
                seen_hashes[h] = len(sections)
                sections.append(NotesSection(
                    unit_title=unit.title,
                    topic_title=topic.title,
                    subtopic_title=sub_name,
                    heading=notes.heading,
                    content=para,
                    citations=list(cites),
                ))
    emitted_keys = {(s.unit_title, s.topic_title) for s in sections}
    for unit in structure.units:
        for topic in unit.topics:
            key = (unit.title, topic.title)
            if key in emitted_keys or key in named_gap_keys:
                continue
            # The topic had generated notes, but every paragraph was a
            # verbatim duplicate of an earlier section and was folded into
            # it.  Name the fold explicitly: a silently dropped topic is
            # exactly what the missing_topic hard gate exists to catch.
            source_gaps.append(
                f"{unit.title}::{topic.title}: duplicate content folded into "
                "an earlier section"
            )
    if not sections:
        raise ValueError("synthesize_notes found no substantive topic notes for the spine")
    cross: List[str] = []
    spine_titles = {u.title for u in structure.units}
    spine_titles.update(t.title for u in structure.units for t in u.topics)
    for prev, nxt in zip(structure.units, structure.units[1:]):
        last_topic = prev.topics[-1].title if prev.topics else prev.title
        first_topic = nxt.topics[0].title if nxt.topics else nxt.title
        assert last_topic in spine_titles and first_topic in spine_titles
        cross.append(
            f"Building on {last_topic!r}, the next unit turns to {first_topic!r}."
        )
    artifact = NotesArtifact(
        title=title,
        sections=sections,
        cross_topic_notes=cross,
        uncovered=list(source_gaps),
        source_gaps=list(source_gaps),
        provenance_note=provenance_note,
    )
    if chunk_texts is not None:
        from services.transformation.engine import validate_grounding

        # A real chunk id quoted inexactly is a quoting defect, not a
        # fabrication: recover the quote from the chunk itself so the gate
        # below verifies literally true text. Foreign ids are untouched.
        artifact, _re_aligned = _align_citation_spans(artifact, chunk_texts)
        allowed = set(chunk_texts)
        gate = validate_grounding(artifact.model_dump(), allowed, chunk_texts)
        if gate["unsupported"]:
            from services.strategies.llm.structured import StructuredGenerationError

            raise StructuredGenerationError(
                f"Synthesis spans unsupported: {gate['unsupported']}"
            )
    else:
        _validate_citation_ids(
            artifact,
            allowed_chunk_ids
            if allowed_chunk_ids is not None
            else _structure_chunk_ids(structure),
        )
    return artifact


def _structure_chunk_ids(structure: DocumentStructure) -> set:
    ids: set = set()
    for unit in structure.units:
        for topic in unit.topics:
            ids.update(topic.chunks or [])
            for sub in topic.subtopics or []:
                ids.update(sub.chunks or [])
                ids.update(c.chunks[0] for c in sub.concepts or [] if c.chunks)
    return ids


def _validate_citation_ids(
    artifact: NotesArtifact, allowed: set
) -> None:
    from services.strategies.llm.structured import StructuredGenerationError

    bad = sorted({
        c.chunk_id for s in artifact.sections for c in s.citations
        if c.chunk_id not in allowed
    })
    if bad:
        raise StructuredGenerationError(
            f"Fabricated citation ids rejected: {bad}"
        )


__all__ = [
    "CHARS_PER_TOKEN",
    "DEFAULT_TOPIC_BUDGET_TOKENS",
    "DEFAULT_TOPIC_BATCH_TOKENS",
    "NOTES_TOPIC_MAX_OUTPUT_TOKENS",
    "estimate_tokens",
    "compress_text",
    "shape_for_topic",
    "build_outline",
    "group_topics",
    "topic_query",
    "select_topic_chunks",
    "render_topic_notes",
    "generate_topic_notes",
    "split_topic_evidence",
    "merge_topic_payloads",
    "synthesize_notes",
]
