"""STEP 6 MCQs: Topic x Bloom matrix with error-tagged distractors.

Each question is built from Topic -> Concepts -> source facts: the key is
the concept's own cleaned source fact (first sentence of its chunk in the
8..35-word band), and the three distractors are REAL cleaned source facts
from OTHER concepts (same topic first, then same unit, then the rest of
the document) -- never templated filler. Every distractor carries a
closed-vocabulary error_tag plus a one-sentence rationale. Matrix
planning (build_matrix) allocates Topic x Bloom cells with
largest-remainder weighting (syllabus weights, then PYQ frequency, else
uniform); cells with no usable key fact or fewer than 3 valid distractors
land in unfilled[] and are NEVER padded with invented content.

RV-2: quizzes.questions keeps the study-compatible string projection
(options: list[str], correct_answer: int) via project_quiz_questions();
the rich {text, is_correct, error_tag, rationale} options live ONLY in
the artifact payload (node_metadata["mcq_artifact"]). One source of
truth, two projections, no change to the study service.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from services.academic.flashcards import (
    _normalize,
    _span_for,
)
from services.academic.types import (
    BloomLevel,
    CardDifficulty,
    DocumentStructure,
    ErrorTag,
    ExtractedConcept,
    ExtractedTopic,
    MCQCell,
    MCQOption,
    MCQPlan,
    MCQSetSpec,
    MCQSpec,
    TopicCitation,
    UnfilledCell,
)
from services.chunking.chunker import BANNED_HEADINGS  # type: ignore
from services.text_utils import (
    clean_display_text,
    collapse_letter_spacing,
    collapse_spaces,
    is_page_number_line,
)

BLOOM_LEVELS = [BloomLevel(l) for l in (
    "remember", "understand", "apply", "analyze", "evaluate", "create",
)]

# Shingle-similarity ceiling: a distractor >= 0.8 similar to the key is
# rejected (near-duplicate, not a distractor).
DISTRACTOR_SIM_CEILING = 0.8

# Stem ceiling: keeps question stems answerable in one read (mirrors the
# flashcard front ceiling).
MCQ_STEM_MAX_WORDS = 24

# Key/distractor fact length band (words). A source fact shorter than
# FACT_MIN_WORDS or longer than FACT_MAX_WORDS is never emitted as an
# option; a cell whose concept has no fact in this band is left unfilled.
FACT_MIN_WORDS = 8
FACT_MAX_WORDS = 35

# Option ceiling: a chosen fact longer than this is trimmed at the last
# sentence boundary before the ceiling (never mid-word, no ellipsis).
OPTION_MAX_WORDS = 40


def _is_valid_topic_title(title: str) -> bool:
    """Local copy of the chunker rule so mcq has zero cross-package
    coupling beyond the banned-set (T008)."""
    if not title:
        return False
    t = title.strip()
    if not t or len(t) > 120 or len(t) < 2:
        return False
    words = t.split()
    if not words or len(words) > 8:
        return False
    if t[-1:] in {".", "!", "?", ";", ","}:
        return False
    if t.lower() in BANNED_HEADINGS or t.lower().startswith("untitled"):
        return False
    return True

_BLOOM_STEM = {
    BloomLevel.REMEMBER: "Which statement about {concept} is correct?",
    BloomLevel.UNDERSTAND: "What does {concept} mean in the context of {topic}?",
    BloomLevel.APPLY: "How does {concept} apply within {topic}?",
    BloomLevel.ANALYZE: "What distinguishes {concept} from related ideas in {topic}?",
    BloomLevel.EVALUATE: "Why is {concept} significant for {topic}?",
    BloomLevel.CREATE: "What follows from {concept} within {topic}?",
}


def _shingles(text: str, n: int = 3) -> set:
    tokens = _normalize(text).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def shingle_similarity(a: str, b: str) -> float:
    """Word-3-gram Jaccard similarity in 0..1 (deterministic)."""
    left, right = _shingles(a), _shingles(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _topic_cells(structure: DocumentStructure) -> List[Dict[str, Any]]:
    """Ordered Topic x Bloom cells (spine order, Bloom order)."""
    cells: List[Dict[str, Any]] = []
    ordinal = 0
    for unit in structure.units:
        for topic in unit.topics:
            ordinal += 1
            for level in BLOOM_LEVELS:
                cells.append({
                    "topic_id": f"topic-{ordinal:04d}",
                    "topic_title": topic.title,
                    "unit_title": unit.title,
                    "bloom_level": level,
                    "ord": ordinal,
                })
    return cells


def build_matrix(
    structure: DocumentStructure,
    total: int,
    syllabus_weights: Optional[Dict[str, float]] = None,
    pyq_frequency: Optional[Dict[str, int]] = None,
    per_cell_target: int = 1,
) -> MCQPlan:
    """Allocate Topic x Bloom targets with largest-remainder math.

    Weighting (deterministic, documented rounding):
    - syllabus weight_percent present -> topic share follows unit weights;
    - else PYQ frequency present -> share follows per-topic PYQ counts;
    - else uniform over topics.
    Within a topic the target splits evenly over the six Bloom levels
    (largest-remainder again). Cell targets sum EXACTLY to total.
    """
    topics: List[Dict[str, Any]] = []
    seen: set = set()
    for unit in structure.units:
        for topic in unit.topics:
            if topic.title in seen:
                continue
            seen.add(topic.title)
            topics.append({"topic": topic.title, "unit": unit.title})
    if not topics or total <= 0:
        return MCQPlan(cells=[], unfilled=[], total_target=max(0, total))
    if syllabus_weights:
        total_w = sum(max(0.0, syllabus_weights.get(t["unit"], 0.0))
                      for t in topics)
        if total_w > 0:
            shares = [max(0.0, syllabus_weights.get(t["unit"], 0.0)) / total_w
                      for t in topics]
        else:
            shares = [1.0 / len(topics)] * len(topics)
    elif pyq_frequency:
        counts = [max(0, int(pyq_frequency.get(t["topic"], 0))) for t in topics]
        if sum(counts) > 0:
            shares = [c / sum(counts) for c in counts]
        else:
            shares = [1.0 / len(topics)] * len(topics)
    else:
        shares = [1.0 / len(topics)] * len(topics)
    exact = [total * share for share in shares]
    per_topic = [int(value) for value in exact]
    remainder = total - sum(per_topic)
    order = sorted(range(len(topics)),
                   key=lambda i: (exact[i] - per_topic[i]), reverse=True)
    for i in order:
        if remainder <= 0:
            break
        per_topic[i] += 1
        remainder -= 1
    cells: List[MCQCell] = []
    for entry_pos, entry in enumerate(topics, start=1):
        topic_id = f"topic-{entry_pos:04d}"
        count = per_topic[topics.index(entry)]
        # Even split of this topic's target over the six Bloom levels.
        per_level = [count // len(BLOOM_LEVELS)] * len(BLOOM_LEVELS)
        leftover = count - sum(per_level)
        for k in range(leftover):
            per_level[k] += 1
        for level_pos, level in enumerate(BLOOM_LEVELS):
            cells.append(MCQCell(
                topic_id=topic_id,
                topic_title=entry["topic"],
                bloom_level=level,
                target=max(0, per_level[level_pos]) * max(1, per_cell_target),
                filled=0,
            ))
    total_target = sum(c.target for c in cells)
    plan = MCQPlan(cells=cells, unfilled=[], total_target=total_target)
    return plan


def _topic_evidence(
    topic: ExtractedTopic,
    chunk_texts: Dict[str, str],
) -> List[Tuple[str, str]]:
    """(chunk_id, text) evidence for one topic (structure-claimed chunks)."""
    out: List[Tuple[str, str]] = []
    for cid in topic.chunks or []:
        text = chunk_texts.get(cid, "")
        if text.strip():
            out.append((cid, text))
    return out


def _key_fact(text: str, boilerplate: Optional[set] = None) -> str:
    """The concept's source fact: first sentence of its chunk in the band.

    The chunk text is cleaned with ``clean_display_text``; boilerplate
    lines, page-number lines and bullet lines are dropped; the surviving
    lines are flattened to one line and split into sentences. The first
    sentence with ``FACT_MIN_WORDS..FACT_MAX_WORDS`` words is the fact.
    Returns "" when no sentence qualifies -- the caller leaves the cell
    unfilled (nothing is ever invented).
    """
    cleaned = clean_display_text(text or "")
    kept: List[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("•"):
            continue
        if is_page_number_line(stripped):
            continue
        if boilerplate:
            norm = collapse_spaces(collapse_letter_spacing(stripped))
            if norm in boilerplate:
                continue
        kept.append(stripped)
    flat = " ".join(kept)
    # No bullet glyph reaches an emitted option: drop any inline "•" left
    # by the PDF text layer and re-collapse the whitespace it leaves.
    flat = " ".join(flat.replace("•", "").split())
    for part in re.split(r"(?<=[.!?])\s+", flat):
        sentence = part.strip()
        if not sentence:
            continue
        # A fact must be a COMPLETE sentence: it ends with terminal
        # punctuation. The PDF text layer often truncates mid-word or
        # drops the final period; those fragments are not sentences and
        # never become options (the cell is left unfilled instead).
        if sentence[-1] not in ".!?":
            continue
        words = sentence.split()
        if FACT_MIN_WORDS <= len(words) <= FACT_MAX_WORDS:
            return sentence
    return ""


def _trim_fact(text: str, max_words: int = OPTION_MAX_WORDS) -> str:
    """Trim at the last sentence boundary before ``max_words``.

    Never cuts mid-word and appends no ellipsis: whole sentences that fit
    within the ceiling are kept; if none fits, the text is cut at the word
    boundary. (Key/distractor facts are already 8..35 words, so this is a
    safety net against mid-word truncation, not a normal path.)
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    kept: List[str] = []
    count = 0
    for part in re.split(r"(?<=[.!?])\s+", text):
        n = len(part.split())
        if count + n > max_words:
            break
        kept.append(part)
        count += n
    if kept:
        return " ".join(kept).strip()
    return " ".join(words[:max_words])


def _heuristic_boilerplate(chunk_texts: Dict[str, str]) -> set:
    """Lines repeated across many chunks (find_boilerplate_lines semantics).

    Fallback for when no page-derived boilerplate set is supplied: a line
    that repeats on >= 25% of chunks (and >= 4 chunks) counts as
    boilerplate. The running header appears on nearly every chunk, so it
    is caught; a line unique to one chunk never is.
    """
    texts = [str(t or "") for t in (chunk_texts or {}).values()]
    if not texts:
        return set()
    counts: Dict[str, int] = {}
    for text in texts:
        seen: set = set()
        for line in text.split("\n"):
            norm = collapse_spaces(collapse_letter_spacing(line))
            if norm and norm not in seen:
                seen.add(norm)
        for norm in seen:
            counts[norm] = counts.get(norm, 0) + 1
    total = len(texts)
    return {line for line, count in counts.items()
            if count >= 4 and count / total >= 0.25}


def _concept_name(title: str) -> str:
    """Short display name for a spine concept title.

    Extraction titles may embed the source sentence
    ("<heading> -- <sentence>"); the name is the heading part only, so
    stems and distractors never leak the key sentence verbatim.
    """
    name = (title or "").split(" -- ")[0].strip()
    words = name.split()
    if len(words) > 6:
        name = " ".join(words[:6])
    return name or (title or "").strip()


def _all_concepts(
    structure: DocumentStructure,
) -> List[Tuple[str, str, ExtractedConcept]]:
    """(unit_title, topic_title, concept) in source order, deduped by chunk.

    A concept is identified by its single source chunk, so deduping by
    chunk id keeps the enumeration stable even when a topic carries both
    subtopic-level and topic-level concept lists.
    """
    out: List[Tuple[str, str, ExtractedConcept]] = []
    seen: set = set()
    for unit in structure.units:
        for topic in unit.topics:
            for sub in topic.subtopics or []:
                for concept in sub.concepts or []:
                    cid = concept.chunks[0] if concept.chunks else None
                    if cid and cid in seen:
                        continue
                    if cid:
                        seen.add(cid)
                    out.append((unit.title, topic.title, concept))
            for concept in topic.concepts or []:
                cid = concept.chunks[0] if concept.chunks else None
                if cid and cid in seen:
                    continue
                if cid:
                    seen.add(cid)
                out.append((unit.title, topic.title, concept))
    return out


def _concept_for_chunk(
    topic: ExtractedTopic,
    cid: str,
) -> Optional[ExtractedConcept]:
    """The spine concept whose single source chunk is ``cid`` (or None)."""
    for sub in topic.subtopics or []:
        for concept in sub.concepts or []:
            if cid in (concept.chunks or []):
                return concept
    for concept in topic.concepts or []:
        if cid in (concept.chunks or []):
            return concept
    return None


def _candidate_facts(
    structure: DocumentStructure,
    chunk_texts: Dict[str, str],
    boilerplate: set,
) -> List[Tuple[str, str, str, str, str]]:
    """Every concept's key fact, in source order.

    Returns (unit_title, topic_title, concept_title, chunk_id, fact) for
    each concept whose chunk yields a fact in the 8..35-word band.
    """
    out: List[Tuple[str, str, str, str, str]] = []
    for unit_title, topic_title, concept in _all_concepts(structure):
        cid = concept.chunks[0] if concept.chunks else None
        if not cid:
            continue
        fact = _key_fact(chunk_texts.get(cid, ""), boilerplate)
        if fact:
            out.append((unit_title, topic_title, concept.title, cid, fact))
    return out


def _ordered_candidates(
    all_facts: List[Tuple[str, str, str, str, str]],
    current_cid: str,
    current_topic: str,
    current_unit: str,
) -> List[Tuple[str, str, str]]:
    """Other concepts' facts: same topic first, then same unit, then rest.

    Each entry is (source_concept, source_topic, fact); the current
    concept (by chunk id) is excluded.
    """
    same_topic: List[Tuple[str, str, str]] = []
    same_unit: List[Tuple[str, str, str]] = []
    rest: List[Tuple[str, str, str]] = []
    for unit_title, topic_title, concept_title, cid, fact in all_facts:
        if cid == current_cid:
            continue
        entry = (concept_title, topic_title, fact)
        if topic_title == current_topic:
            same_topic.append(entry)
        elif unit_title == current_unit:
            same_unit.append(entry)
        else:
            rest.append(entry)
    return same_topic + same_unit + rest


def _distractors(
    concept: str,
    topic_title: str,
    key_text: str,
    candidates: List[Tuple[str, str, str]],
) -> Optional[List[MCQOption]]:
    """Up to 3 real source facts as distractors, or None when fewer exist.

    Every candidate is another concept's cleaned source fact. A candidate
    is accepted only when ALL hold: 8..35 words; not equal to the key
    (normalised); shingle-similarity to the key below the ceiling; no
    newline, bullet or page-number-only content; >= 3 alphabetic words;
    first 6 words not identical to the key's; not already chosen. Never
    padded, never invented: fewer than 3 valid candidates -> None (the
    caller leaves the cell unfilled).
    """
    key_norm = _normalize(key_text)
    key_first6 = " ".join(key_text.split()[:6]).lower()
    chosen: List[MCQOption] = []
    chosen_norms = {key_norm}
    for source_concept, source_topic, fact in candidates:
        if len(chosen) >= 3:
            break
        words = fact.split()
        if not (FACT_MIN_WORDS <= len(words) <= FACT_MAX_WORDS):
            continue
        if _normalize(fact) in chosen_norms:
            continue
        if shingle_similarity(fact, key_text) >= DISTRACTOR_SIM_CEILING:
            continue
        if "\n" in fact or "•" in fact:
            continue
        if is_page_number_line(fact.strip()):
            continue
        alpha_words = [w for w in words if w.isalpha()]
        if len(alpha_words) < 3:
            continue
        if " ".join(words[:6]).lower() == key_first6:
            continue
        if source_topic == topic_title:
            error_tag = ErrorTag.MISCONCEPTION
            rationale = (
                f"A true statement about {source_concept} in "
                f"{topic_title}, not {concept}; it does not answer "
                f"this question."
            )
        else:
            error_tag = ErrorTag.UNIT_CONFUSION
            rationale = (
                f"A unit confusion: {source_concept} belongs to "
                f"{source_topic}, so this true statement answers the "
                f"wrong question."
            )
        chosen.append(MCQOption(
            text=fact,
            is_correct=False,
            error_tag=error_tag,
            rationale=rationale,
        ))
        chosen_norms.add(_normalize(fact))
    if len(chosen) < 3:
        return None
    return chosen


def build_mcqs(
    structure: DocumentStructure,
    chunk_texts: Dict[str, str],
    plan: MCQPlan,
    title: Optional[str] = None,
    boilerplate: Optional[set] = None,
) -> MCQSetSpec:
    """Deterministic MCQs for every fillable matrix cell (offline, no LLM).

    The key is the concept's own cleaned source fact (first sentence of its
    chunk in the 8..35-word band); the three distractors are REAL cleaned
    source facts from OTHER concepts. A cell whose concept has no usable
    key fact, or fewer than 3 valid distractors, moves to unfilled[] with
    a reason and is NEVER padded. Filled counts on the returned plan sum
    to len(questions). Options rotate the key position deterministically
    (round-robin over 0..3) for balanced key coverage.

    ``boilerplate`` is the page-derived running-header set; when omitted, a
    heuristic over the chunk texts flags lines repeated across >= 25% of
    chunks so the header is still dropped.
    """
    by_topic: Dict[str, ExtractedTopic] = {}
    topic_unit: Dict[str, str] = {}
    for unit in structure.units:
        for topic in unit.topics:
            by_topic.setdefault(topic.title, topic)
            topic_unit[topic.title] = unit.title
    if boilerplate is None:
        boilerplate = _heuristic_boilerplate(chunk_texts)
    boilerplate = {str(b) for b in (boilerplate or set())}
    all_facts = _candidate_facts(structure, chunk_texts, boilerplate)
    questions: List[MCQSpec] = []
    unfilled: List[UnfilledCell] = []
    filled_counts: Dict[Tuple[str, str], int] = {}
    ordinal = 0
    for cell in plan.cells:
        topic = by_topic.get(cell.topic_title)
        if topic is not None and not _is_valid_topic_title(topic.title):
            if cell.target > 0:
                unfilled.append(UnfilledCell(
                    topic_id=cell.topic_id,
                    topic_title=cell.topic_title,
                    bloom_level=cell.bloom_level,
                    reason=("topic title rejected by _is_valid_topic_title; "
                            "cell left unfilled, nothing invented"),
                ))
            continue
        evidence = _topic_evidence(topic, chunk_texts) if topic is not None else []
        if topic is None or not evidence:
            if cell.target > 0:
                unfilled.append(UnfilledCell(
                    topic_id=cell.topic_id,
                    topic_title=cell.topic_title,
                    bloom_level=cell.bloom_level,
                    reason=("no source material: topic has no retrievable chunk; "
                            "cell left unfilled, nothing invented"),
                ))
            continue
        current_unit = topic_unit.get(topic.title, "")
        for _ in range(max(0, cell.target)):
            ordinal += 1
            cid, text = evidence[(ordinal - 1) % len(evidence)]
            # The stem names the concept that owns the evidence chunk, so
            # the key fact and the concept in the stem are the same fact.
            concept = _concept_for_chunk(topic, cid)
            if concept is not None:
                raw_concept = concept.title
            elif topic.subtopics and topic.subtopics[0].concepts:
                raw_concept = topic.subtopics[0].concepts[0].title
            else:
                raw_concept = topic.title
            # A sentence-like concept name (the old "<heading> -- <sentence>"
            # titles) must never reach a stem: leave the cell unfilled with
            # a reason instead of generating a question from a garbage name.
            # The check runs on the RAW spine title because _concept_name
            # strips the " -- " marker and truncates to 6 words, which would
            # otherwise hide the garbage it is meant to catch.
            if " -- " in raw_concept or len(raw_concept.split()) > 8:
                if cell.target > 0:
                    unfilled.append(UnfilledCell(
                        topic_id=cell.topic_id,
                        topic_title=cell.topic_title,
                        bloom_level=cell.bloom_level,
                        reason=("concept name rejected (sentence-like title); "
                                "cell left unfilled, nothing invented"),
                    ))
                continue
            concept_title = _concept_name(raw_concept)
            key = _trim_fact(_key_fact(text, boilerplate))
            if not key:
                if cell.target > 0:
                    unfilled.append(UnfilledCell(
                        topic_id=cell.topic_id,
                        topic_title=cell.topic_title,
                        bloom_level=cell.bloom_level,
                        reason=("no usable key fact (8..35 words) for concept; "
                                "cell left unfilled, nothing invented"),
                    ))
                continue
            candidates = _ordered_candidates(
                all_facts, cid, topic.title, current_unit)
            distractors = _distractors(
                concept_title, topic.title, key, candidates)
            if distractors is None or len(distractors) < 3:
                if cell.target > 0:
                    unfilled.append(UnfilledCell(
                        topic_id=cell.topic_id,
                        topic_title=cell.topic_title,
                        bloom_level=cell.bloom_level,
                        reason=("insufficient distinct source facts for "
                                "distractors; cell left unfilled, nothing invented"),
                    ))
                continue
            key_option = MCQOption(text=key, is_correct=True)
            # Deterministic key rotation: balanced correct_answer coverage.
            slot = (ordinal - 1) % 4
            options = list(distractors)
            options.insert(slot, key_option)
            # Control chars from the PDF text layer must never reach the
            # DB/UI: sanitise every option text before validation.
            options = [
                option.model_copy(update={"text": clean_display_text(option.text)})
                for option in options
            ]
            stem = clean_display_text(_BLOOM_STEM[cell.bloom_level].format(
                concept=concept_title, topic=topic.title))
            # No leakage: the stem never contains the full key text.
            assert _normalize(key) not in _normalize(stem)
            questions.append(MCQSpec(
                id=f"mcq-{ordinal:04d}",
                question=stem,
                options=options,
                correct_answer=slot,
                explanation=clean_display_text(
                    f"{key} This follows from the cited source for {concept_title}."),
                topic=topic.title,
                topic_id=cell.topic_id,
                difficulty=(CardDifficulty.EASY if cell.bloom_level in
                            (BloomLevel.REMEMBER, BloomLevel.UNDERSTAND)
                            else CardDifficulty.MEDIUM),
                level=cell.bloom_level,
                cites=[TopicCitation(chunk_id=cid, span=_span_for(text))],
            ))
            filled_counts[(cell.topic_title, cell.bloom_level.value)] = (
                filled_counts.get((cell.topic_title, cell.bloom_level.value), 0) + 1
            )
    cells = [cell.model_copy(update={
        "filled": filled_counts.get((cell.topic_title, cell.bloom_level.value), 0),
    }) for cell in plan.cells]
    filled_plan = MCQPlan(cells=cells, unfilled=unfilled,
                          total_target=plan.total_target)
    return MCQSetSpec(
        title=title or f"{structure.title} practice set",
        questions=questions,
        plan=filled_plan,
        provenance_note=(
            f"deterministic Topic x Bloom set from spine evidence; "
            f"{len(questions)} filled, {len(unfilled)} unfilled cells reported"
        ),
    )


def validate_mcq(
    spec: MCQSpec,
    allowed_chunk_ids: set,
    topic_titles: set,
) -> List[Dict[str, Any]]:
    """Validate one rich MCQ (single key/plausible/dupes/leakage/grounded)."""
    errors: List[Dict[str, Any]] = []
    if len(spec.options) != 4:
        errors.append({"question_id": spec.id, "code": "option_count",
                       "detail": f"expected 4 options, found {len(spec.options)}"})
        return errors
    keys = [i for i, o in enumerate(spec.options) if o.is_correct]
    if len(keys) != 1:
        errors.append({"question_id": spec.id, "code": "key_count",
                       "detail": f"exactly 1 key required, found {len(keys)}"})
    elif spec.correct_answer != keys[0]:
        errors.append({"question_id": spec.id, "code": "key_mismatch",
                       "detail": "correct_answer does not point at the key"})
    seen: set = set()
    for option in spec.options:
        key = _normalize(option.text)
        if key in seen:
            errors.append({"question_id": spec.id, "code": "duplicate_option",
                           "detail": f"duplicate option {option.text!r}"})
        seen.add(key)
        if not option.is_correct:
            if option.error_tag is None:
                errors.append({"question_id": spec.id, "code": "missing_error_tag",
                               "detail": f"distractor {option.text!r} untagged"})
            if not (option.rationale or "").strip():
                errors.append({"question_id": spec.id, "code": "missing_rationale",
                               "detail": f"distractor {option.text!r} has no rationale"})
    key_text = next((o.text for o in spec.options if o.is_correct), "")
    for option in spec.options:
        if option.is_correct:
            continue
        if shingle_similarity(option.text, key_text) >= DISTRACTOR_SIM_CEILING:
            errors.append({"question_id": spec.id, "code": "similar_distractor",
                           "detail": f"distractor {option.text!r} too close to the key"})
    # No leakage: the stem must not give away the key verbatim, and the
    # explanation must stay on the question's topic.
    if key_text and _normalize(key_text) in _normalize(spec.question):
        errors.append({"question_id": spec.id, "code": "leakage",
                       "detail": "question stem contains the key verbatim"})
    if not spec.question.rstrip().endswith("?") and len(spec.question.split()) < 4:
        errors.append({"question_id": spec.id, "code": "ambiguous",
                       "detail": "stem too short to be unambiguous"})
    if len((spec.question or "").split()) > MCQ_STEM_MAX_WORDS:
        errors.append({"question_id": spec.id, "code": "stem_too_long",
                       "detail": f"stem exceeds {MCQ_STEM_MAX_WORDS} words"})
    if spec.topic not in topic_titles:
        errors.append({"question_id": spec.id, "code": "unknown_topic",
                       "detail": f"topic {spec.topic!r} not in the spine"})
    for cite in spec.citations or []:
        cid = cite.chunk_id if hasattr(cite, "chunk_id") else cite.get("chunk_id")
        if cid not in allowed_chunk_ids:
            errors.append({"question_id": spec.id, "code": "unsupported_cite",
                           "detail": f"citation {cid!r} outside the prompt"})
    return errors


def validate_set(
    mcq_set: MCQSetSpec,
    allowed_chunk_ids: set,
    topic_titles: set,
) -> Dict[str, Any]:
    """Validate a whole set: per-question errors + plan/filled accounting."""
    errors: List[Dict[str, Any]] = []
    for question in mcq_set.questions:
        errors.extend(validate_mcq(question, allowed_chunk_ids, topic_titles))
        # T008: a banned literal or shape-broken topic never validates
        # as a quiz item (the set fails honestly, nothing is padded).
        if not _is_valid_topic_title(question.topic):
            errors.append({
                "question_id": question.id,
                "code": "card_topic_invalid",
                "detail": f"topic {question.topic!r} failed "
                          f"_is_valid_topic_title",
            })
    seen_questions: Dict[str, str] = {}
    for question in mcq_set.questions:
        key = _normalize(question.question)
        if key in seen_questions:
            errors.append({"question_id": question.id, "code": "duplicate_question",
                           "detail": f"duplicates {seen_questions[key]}"})
        else:
            seen_questions[key] = question.id
    filled_total = sum(c.filled for c in mcq_set.plan.cells)
    plan_ok = filled_total == len(mcq_set.questions)
    if not plan_ok:
        errors.append({"question_id": "*", "code": "plan_mismatch",
                       "detail": f"filled counts sum to {filled_total} "
                                 f"but {len(mcq_set.questions)} questions exist"})
    return {"ok": not errors, "errors": errors, "plan_ok": plan_ok}


def project_quiz_questions(mcq_set: MCQSetSpec) -> List[Dict[str, Any]]:
    """Study-compatible projection for quizzes.questions (RV-2).

    options is list[str] and correct_answer is an int index, so the study
    service renders options[correct_answer] as display text (a dict there
    would render as "{'text': ...}"). Rich options stay in the artifact.
    """
    projected: List[Dict[str, Any]] = []
    for question in mcq_set.questions:
        options = [o.text if hasattr(o, "text") else o.get("text")
                   for o in question.options]
        assert all(isinstance(o, str) for o in options)
        projected.append({
            "question": question.question,
            "options": options,
            "correct_answer": int(question.correct_answer),
            "explanation": question.explanation,
        })
    return projected


def mcq_artifact(mcq_set: MCQSetSpec) -> Dict[str, Any]:
    """Artifact payload: rich set + plan + unfilled (DB: node_metadata)."""
    return {
        "title": mcq_set.title,
        "questions": [q.model_dump(by_alias=True, mode="json")
                      for q in mcq_set.questions],
        "plan": mcq_set.plan.model_dump(mode="json"),
        "unfilled": [u.model_dump(mode="json") for u in mcq_set.plan.unfilled],
        "projection": project_quiz_questions(mcq_set),
        "provenance_note": mcq_set.provenance_note,
    }


async def generate_mcq_cell(
    engine: Any,
    context: str,
    allowed_chunk_ids: set,
    topic_title: str,
    bloom_level: BloomLevel,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Live LLM per-cell MCQ call through the shared schema-enforced path."""
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import GenerationMode

    prompt = PromptTemplates.get_prompt(
        GenerationMode.MCQ_MATRIX, context,
        query or f"{topic_title} [{bloom_level.value}]",
        topic=topic_title, bloom=bloom_level.value,
    )
    return await engine._generate_structured(
        GenerationMode.MCQ_MATRIX, prompt, allowed_chunk_ids
    )


__all__ = [
    "BLOOM_LEVELS",
    "DISTRACTOR_SIM_CEILING",
    "MCQ_STEM_MAX_WORDS",
    "shingle_similarity",
    "build_matrix",
    "build_mcqs",
    "validate_mcq",
    "validate_set",
    "project_quiz_questions",
    "mcq_artifact",
    "generate_mcq_cell",
]
