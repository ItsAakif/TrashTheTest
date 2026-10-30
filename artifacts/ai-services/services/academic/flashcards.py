"""STEP 5 flashcards: typed cards FROM THE SPINE (never raw PDF text).

Every card is bound to a concept from a DocumentStructure concept index:
card.concept_id resolves into that index (same project + model version by
construction), card.topic names the spine topic, and citations are a subset
of the card's prompt chunk ids. Bloom allocation follows BLOOM_MIX with a
guard (realized mix within 0.10 absolute per level, >= 1 card per level;
repaired once, then honestly reported). FSRS-ready review fields are stored
with documented defaults; the scheduler is deferred (no state transitions
here -- grep must show no review-state writes outside FSRS_NEW_REVIEW).

The Flashcards UI contract is unchanged: to_deck_cards() projects typed
cards to {front, back, difficulty} so decks persist through the existing
deck repository; the rich typed set lives in the document root's
node_metadata["flashcards_artifact"] (same additive pattern as notes).
"""

import copy
import re
import string
from typing import Any, Callable, Dict, List, Optional, Tuple

from services.academic.types import (
    BLOOM_MIX,
    FLASHCARD_FRONT_MAX_WORDS,
    FSRS_NEW_REVIEW,
    BloomLevel,
    CardDifficulty,
    CardType,
    DocumentStructure,
    ExtractedTopic,
    FlashcardDeckReport,
    FlashcardSetSpec,
    FlashcardSpec,
    TopicCitation,
)
from services.chunking.chunker import BANNED_HEADINGS  # type: ignore
from services.text_utils import (
    clean_display_text,
    count_qualifying_sentences,
    is_front_back_admissible,
    is_junk_title,
    is_sentence_backed,
)

# T007: validate_deck rejects fronts over 24 words.  The model-level
# construction ceiling stays FLASHCARD_FRONT_MAX_WORDS (30) so existing
# typed cards keep validating at construction time.
FLASHCARD_FRONT_MAX_WORDS_VALIDATED = 24


def _is_valid_topic_title(title: str) -> bool:
    """Local copy of the chunker rule so flashcards has zero
    cross-package coupling beyond the banned-set."""
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


def _collapse_letter_spacing(text: str) -> str:
    if not text:
        return text
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        tokens = [t for t in line.split(" ") if t]
        single = [t for t in tokens if len(t) == 1 and t.isalpha()]
        if tokens and len(single) / len(tokens) >= 0.6:
            out.append(re.sub(r"(?<=\b\w) (?=\w\b)", "", line))
        else:
            out.append(line)
    return "\n".join(out)

BLOOM_LEVELS = [BloomLevel(l) for l in (
    "remember", "understand", "apply", "analyze", "evaluate", "create",
)]

CARD_TYPES = [CardType(t) for t in (
    "definition", "conceptual", "process", "formula",
    "comparison", "cause-effect", "application", "misconception",
)]

# Bloom verbs per level (question templates stay topic-grounded: every
# question names its concept and topic, so cards are never ambiguous
# about what is being asked).
_BLOOM_QUESTION = {
    BloomLevel.REMEMBER: "What is {concept}?",
    BloomLevel.UNDERSTAND: "What does {concept} mean in {topic}?",
    BloomLevel.APPLY: "How is {concept} applied in {topic}?",
    BloomLevel.ANALYZE: "What distinguishes {concept} within {topic}?",
    BloomLevel.EVALUATE: "Why does {concept} matter for {topic}?",
    BloomLevel.CREATE: "What follows from {concept} in {topic}?",
}

# Card-type question frames (each names concept + topic: no ambiguity).
_TYPE_QUESTION = {
    CardType.DEFINITION: "Define {concept} ({topic}).",
    CardType.CONCEPTUAL: "What is the key idea of {concept} in {topic}?",
    CardType.PROCESS: "What is the process involving {concept} in {topic}?",
    CardType.FORMULA: "What relation governs {concept} in {topic}?",
    CardType.COMPARISON: "How does {concept} compare within {topic}?",
    CardType.CAUSE_EFFECT: "What causes {concept} in {topic}, and with what effect?",
    CardType.APPLICATION: "Where is {concept} used in {topic}?",
    CardType.MISCONCEPTION: "What is a common misunderstanding about {concept} ({topic})?",
}

_DIFFICULTY_BY_BLOOM = {
    BloomLevel.REMEMBER: CardDifficulty.EASY,
    BloomLevel.UNDERSTAND: CardDifficulty.EASY,
    BloomLevel.APPLY: CardDifficulty.MEDIUM,
    BloomLevel.ANALYZE: CardDifficulty.MEDIUM,
    BloomLevel.EVALUATE: CardDifficulty.HARD,
    BloomLevel.CREATE: CardDifficulty.HARD,
}


def concept_index(structure: DocumentStructure) -> Dict[str, Dict[str, Any]]:
    """Index every spine concept in source order (pure, offline).

    Returns {concept_id: {concept_id, unit, topic, concept, chunk_ids,
    ord}} with dense ordinals in spine order. concept_id values are the
    only valid FlashcardSpec.concept_id values for this structure.
    """
    index: Dict[str, Dict[str, Any]] = {}
    ordinal = 0
    for unit in structure.units:
        for topic in unit.topics:
            for sub in topic.subtopics or []:
                for concept in sub.concepts or []:
                    ordinal += 1
                    cid = f"concept-{ordinal:04d}"
                    index[cid] = {
                        "concept_id": cid,
                        "unit": unit.title,
                        "topic": topic.title,
                        "concept": concept.title,
                        "chunk_ids": list(concept.chunks or []),
                        "pages": list(concept.pages or []),
                        "ord": ordinal,
                    }
            for concept in topic.concepts or []:
                if any(e["concept"] == concept.title and e["topic"] == topic.title
                       for e in index.values()):
                    continue
                ordinal += 1
                cid = f"concept-{ordinal:04d}"
                index[cid] = {
                    "concept_id": cid,
                    "unit": unit.title,
                    "topic": topic.title,
                    "concept": concept.title,
                    "chunk_ids": list(concept.chunks or []),
                    "pages": list(concept.pages or []),
                    "ord": ordinal,
                }
    return index


def bloom_targets(total: int, mix: Optional[Dict[str, float]] = None) -> Dict[str, int]:
    """Largest-remainder per-level card targets summing exactly to total."""
    blend = dict(mix or BLOOM_MIX)
    if total <= 0:
        return {level.value: 0 for level in BLOOM_LEVELS}
    exact = {level.value: total * blend.get(level.value, 0.0)
             for level in BLOOM_LEVELS}
    targets = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(targets.values())
    order = sorted(exact, key=lambda k: (exact[k] - targets[k]), reverse=True)
    for key in order:
        if remainder <= 0:
            break
        targets[key] += 1
        remainder -= 1
    return targets


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()).split())


def _is_usable_concept_name(
    name: str,
    document_title: str = "",
    unit_title: str = "",
    boilerplate=None,
) -> bool:
    """Concept-name floor for card fronts (never emit a garbage name).

    Skips a card when the concept name contains a bullet glyph, starts with
    punctuation or a digit, is a single word shorter than 6 characters
    ("False", "File"), or is a single lowercase word. Multi-word names and
    capitalised single words of >= 6 characters ("Classes", "Objects")
    pass.  R3 (``is_junk_title``) additionally rejects title restatements
    and credits/nav/brand names ("Sheryians AI", "Downloading Python").
    """
    if not name:
        return False
    if "•" in name:
        return False
    stripped = (name or "").strip()
    if not stripped:
        return False
    if stripped[0] in string.punctuation or stripped[0].isdigit():
        return False
    words = stripped.split()
    if len(words) == 1:
        if len(words[0]) < 6:
            return False
        if words[0].islower():
            return False
    if is_junk_title(stripped, document_title, unit_title, boilerplate):
        return False
    return True


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", _collapse_letter_spacing(
        (text or "").strip()))
    return next((p.strip() for p in parts if p.strip()), "")


def _span_for(text: str) -> str:
    return " ".join((text or "").split()[:8])


_YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
_AND_RE = re.compile(r"\band\b", re.IGNORECASE)
_DEF_CUE = r"(?:is|are|means|refers to|consists of)"
_DEF_CUE_RE = re.compile(r"\b" + _DEF_CUE + r"\b", re.IGNORECASE)


def _sentence_count(text: str) -> int:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return len([p for p in parts if p.strip()])


def _has_fact_list(answer: str) -> bool:
    """True when the answer packs >= 3 comma items or > 1 'and' fact."""
    if len([p for p in (answer or "").split(",") if p.strip()]) >= 3:
        return True
    return len(_AND_RE.findall(answer or "")) > 1


def _answer_is_acceptable(answer: str, question: str) -> bool:
    """Atomicity / yes-no / length / substring gates for a card back."""
    if len(answer or "") < 12:
        return False
    if _sentence_count(answer) > 1:
        return False
    if _has_fact_list(answer):
        return False
    if _YES_NO_RE.match(answer or ""):
        return False
    if answer and _normalize(answer) in _normalize(question):
        return False
    return True


def _definitional_sentence(concept: str, chunk_text: str) -> str:
    """The first sentence of ``chunk_text`` that defines ``concept``.

    A definitional sentence contains ``<concept> is/are/means/refers to/
    consists of`` (the requirement that catches bare proper-noun/product
    names with no definitional source span).
    """
    name = (concept or "").strip()
    if not name:
        return ""
    pattern = re.compile(
        r"\b" + re.escape(name) + r"\b\s+" + _DEF_CUE + r"\b", re.IGNORECASE)
    # Bullet-glyph control codes become "•"; treat them as word breaks so
    # "Polymorphism\x0f is" still matches the definitional span.
    text = clean_display_text(chunk_text or "").replace("•", " ")
    text = " ".join(text.split())
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if sentence.strip() and pattern.search(sentence):
            return sentence.strip()
    return ""


def _has_definitional_span(concept: str, chunk_text: str) -> bool:
    return bool(_definitional_sentence(concept, chunk_text))


def _cloze_pair(concept: str, chunk_text: str) -> Tuple[str, str]:
    """(front, back) for a cloze card from the concept's own definition.

    front = the definitional sentence with the concept replaced by
    ``_____``; back = the concept plus one clarifying clause from the same
    sentence.  Returns ``("", "")`` when no definitional sentence exists.
    """
    name = (concept or "").strip()
    sentence = _definitional_sentence(name, chunk_text)
    if not sentence or not name:
        return "", ""
    pattern = re.compile(re.escape(name), re.IGNORECASE)
    if not pattern.search(sentence):
        return "", ""
    front = pattern.sub("_____", sentence, count=1)
    cue = re.search(r"\b" + _DEF_CUE + r"\b\s+(.*)$", sentence, re.IGNORECASE)
    clause = (cue.group(1).strip() if cue else sentence).rstrip(".!?")
    clause = clause.split(",")[0].strip() or sentence.rstrip(".!?")
    back = f"{name}: {clause}."
    if _normalize(front) == _normalize(back):
        return "", ""
    return front, back


def _topic_chunk_counts(structure: DocumentStructure) -> Dict[str, set]:
    """topic title -> set of chunk ids referenced by that topic."""
    counts: Dict[str, set] = {}
    for unit in structure.units:
        for topic in unit.topics:
            bucket = counts.setdefault(topic.title, set())
            for cid in topic.chunks or []:
                bucket.add(str(cid))
            for sub in topic.subtopics or []:
                for cid in sub.chunks or []:
                    bucket.add(str(cid))
    return counts


def _page_positions(entries: List[Dict[str, Any]]) -> Tuple[Dict[int, int], int]:
    """Distinct ordered page -> position, plus the page count (R2 window)."""
    pages = sorted({p for e in entries for p in (e.get("pages") or [])
                    if isinstance(p, int)})
    return {page: i for i, page in enumerate(pages)}, len(pages)


def _answer_for(concept: str, card_type: CardType, chunk_text: str,
                 question: str = "") -> str:
    """Answer text from the chunk (never equal to the question by design).

    Returns "" when the answer would be useless: shorter than 12 characters
    or equal to the question after normalisation.  The caller skips the
    card instead of emitting a trivial back.
    """
    sentence = _first_sentence(chunk_text)
    if not sentence:
        sentence = f"{concept} is covered in the source material."
    if card_type == CardType.MISCONCEPTION:
        answer = (
            f"A common error is to misstate {concept}; correctly, "
            f"{sentence[0].lower() + sentence[1:] if len(sentence) > 1 else sentence}"
        )
    else:
        answer = sentence
    if len(answer) < 12:
        return ""
    if question and _normalize(answer) == _normalize(question):
        return ""
    return answer


def build_flashcards(
    structure: DocumentStructure,
    chunk_texts: Dict[str, str],
    max_cards: Optional[int] = None,
    mix: Optional[Dict[str, float]] = None,
    title: Optional[str] = None,
) -> FlashcardSetSpec:
    """Deterministic typed deck from spine concepts (offline, no LLM).

    One card per spine concept in source order (capped by max_cards),
    card types rotating across the 8-type vocabulary, Bloom levels dealt
    per bloom_targets() so the realized mix matches the configured mix.
    Cards whose concept has no retrievable chunk, or whose topic fails
    _is_valid_topic_title (spine placeholder / banned literal), are
    SKIPPED and named in provenance (reported short, never padded with
    unsupported cards).
    """
    index = concept_index(structure)
    entries = sorted(index.values(), key=lambda e: e["ord"])
    if max_cards is not None:
        entries = entries[:max(0, max_cards)]
    targets = bloom_targets(len(entries), mix)
    # Deal Bloom levels deterministically: round-robin over levels that
    # still have quota, in BLOOM_LEVELS order (source order preserved:
    # concept i takes the next level with remaining quota).
    quotas = dict(targets)
    levels: List[BloomLevel] = []
    for _ in entries:
        pick = next(
            (level for level in BLOOM_LEVELS if quotas.get(level.value, 0) > 0),
            BLOOM_LEVELS[0],
        )
        quotas[pick.value] -= 1
        levels.append(pick)
    topic_chunks = _topic_chunk_counts(structure)
    page_pos, page_total = _page_positions(entries)
    doc_title = str(getattr(structure, "title", "") or "")
    seen_fronts: set = set()
    cards: List[FlashcardSpec] = []
    skipped: List[str] = []
    for pos, entry in enumerate(entries):
        # T007: a garbage/spine-placeholder topic is skipped honestly
        # (named in provenance) instead of becoming a padded card.  The
        # check runs before the chunk lookup so the reported reason is
        # the topic itself, even when the chunk is missing too.
        topic = entry["topic"]
        if not _is_valid_topic_title(topic):
            skipped.append(f"{topic!r}:invalid_topic")
            continue
        texts = [chunk_texts.get(cid, "") for cid in entry["chunk_ids"]]
        texts = [t for t in texts if t.strip()]
        if not texts:
            skipped.append(entry["concept"])
            continue
        card_type = CARD_TYPES[pos % len(CARD_TYPES)]
        bloom = levels[pos]
        concept = entry["concept"]
        primary = texts[0]
        # R3: a credits/nav/brand or title-restatement name never cards.
        if is_junk_title(concept, doc_title, entry["unit"]):
            skipped.append(f"{concept!r}:junk_concept")
            continue
        # A sentence-like concept name (the old "<heading> -- <sentence>"
        # titles) must never reach a card front: skip the card honestly
        # (named in provenance) instead of emitting a garbage question.
        # The concept-name floor (bullet glyph, leading punctuation/digit,
        # single short word, single lowercase word) applies on top.
        if (not _is_valid_topic_title(concept)) or " -- " in concept \
                or len(concept.split()) > 8 \
                or not _is_usable_concept_name(concept, doc_title, entry["unit"]):
            skipped.append(f"{concept!r}:invalid_concept")
            continue
        # R1: the source chunk must be sentence-backed.
        if not is_sentence_backed(primary):
            skipped.append(f"{concept!r}:not_sentence_backed")
            continue
        # R2: front/back-matter pages need >= 2 qualifying sentences.
        concept_page = next(
            (p for p in (entry.get("pages") or []) if isinstance(p, int)), None)
        if concept_page is not None and page_total >= 2:
            if not is_front_back_admissible(
                    page_pos.get(concept_page, 0), page_total, primary):
                skipped.append(f"{concept!r}:front_back_matter")
                continue
        # A topic needs >= 2 chunks OR its chunk >= 2 qualifying sentences.
        if len(topic_chunks.get(topic, set())) < 2 \
                and count_qualifying_sentences(primary) < 2:
            skipped.append(f"{concept!r}:single_chunk_topic")
            continue
        # Definitional source span is required (catches bare product names).
        if not _has_definitional_span(concept, primary):
            skipped.append(f"{concept!r}:no_definitional_span")
            continue
        # Prefer CLOZE for definitions: front = definitional sentence with
        # the term blanked, back = the term plus one clarifying clause.
        question = answer = ""
        is_cloze = False
        if card_type == CardType.DEFINITION:
            cloze_front, cloze_back = _cloze_pair(concept, primary)
            if cloze_front and len(cloze_front.split()) <= FLASHCARD_FRONT_MAX_WORDS_VALIDATED:
                if _answer_is_acceptable(cloze_back, cloze_front):
                    question = clean_display_text(cloze_front)
                    answer = clean_display_text(cloze_back)
                    is_cloze = True
        if not is_cloze:
            frame = _TYPE_QUESTION[card_type].format(concept=concept, topic=topic)
            bloom_q = _BLOOM_QUESTION[bloom].format(concept=concept, topic=topic)
            question = clean_display_text(frame if pos % 2 == 0 else bloom_q)
            answer = _answer_for(concept, card_type, primary, question=question)
            if not answer:
                skipped.append(f"{concept!r}:no_answer")
                continue
            # Control chars from the PDF text layer must never reach DB/UI.
            answer = clean_display_text(answer)
            if _normalize(question) == _normalize(answer):
                answer = f"In {topic}, {answer[0].lower() + answer[1:] if len(answer) > 1 else answer}"
            if not _answer_is_acceptable(answer, question):
                skipped.append(f"{concept!r}:weak_answer")
                continue
        # Duplicate fronts are rejected case/punctuation-insensitively.
        front_key = _normalize(question)
        if front_key in seen_fronts:
            skipped.append(f"{concept!r}:duplicate_front")
            continue
        seen_fronts.add(front_key)
        chunk_id = entry["chunk_ids"][0]
        metadata = {
            "unit": entry["unit"],
            "concept": concept,
            "provenance": "spine concept (DocumentStructure), not raw PDF text",
        }
        if is_cloze:
            metadata["cloze"] = True
        cards.append(FlashcardSpec(
            id=f"card-{entry['ord']:04d}",
            topic=topic,
            concept_id=entry["concept_id"],
            question=question,
            answer=answer,
            type=card_type,
            difficulty=_DIFFICULTY_BY_BLOOM[bloom],
            cognitive=bloom,
            cites=[TopicCitation(chunk_id=chunk_id, span=_span_for(primary))],
            metadata=metadata,
            review=dict(FSRS_NEW_REVIEW),
        ))
    realized = _realized_mix(cards)
    provenance = (
        f"deterministic deck from {len(index)} spine concepts; "
        f"{len(cards)} cards, {len(skipped)} skipped "
        f"(no retrievable chunk, invalid topic, or unusable concept name)"
    )
    if skipped:
        provenance += f": {', '.join(skipped[:8])}"
        if len(skipped) > 8:
            provenance += f" (+{len(skipped) - 8} more)"
    return FlashcardSetSpec(
        title=title or f"{structure.title} flashcards",
        flashcards=cards,
        bloom_realized=realized,
        provenance_note=provenance,
    )


def _realized_mix(cards: List[FlashcardSpec]) -> Dict[str, float]:
    total = len(cards)
    realized = {level.value: 0.0 for level in BLOOM_LEVELS}
    if not total:
        return realized
    for card in cards:
        level = card.bloom_level.value if isinstance(card.bloom_level, BloomLevel) \
            else str(card.bloom_level)
        realized[level] = realized.get(level, 0.0) + 1.0 / total
    return {key: round(value, 4) for key, value in realized.items()}


def validate_deck(
    cards: List[FlashcardSpec],
    allowed_chunk_ids: set,
    concept_ids: set,
    mix: Optional[Dict[str, float]] = None,
    tolerance: float = 0.10,
) -> FlashcardDeckReport:
    """Validate a deck: dupes/trivial/length/concept/cites/Bloom/provenance."""
    blend = dict(mix or BLOOM_MIX)
    errors: List[Dict[str, Any]] = []
    seen_fronts: Dict[str, str] = {}
    for card in cards:
        front = _normalize(card.question)
        if front in seen_fronts:
            errors.append({"card_id": card.id, "code": "duplicate_front",
                           "detail": f"front duplicates card {seen_fronts[front]}"})
        else:
            seen_fronts[front] = card.id
        if _normalize(card.question) == _normalize(card.answer):
            errors.append({"card_id": card.id, "code": "trivial",
                           "detail": "back equals front"})
        front_words = len((card.question or "").split())
        if front_words > FLASHCARD_FRONT_MAX_WORDS_VALIDATED:
            errors.append({"card_id": card.id, "code": "front_too_long",
                           "detail": f"front exceeds {FLASHCARD_FRONT_MAX_WORDS_VALIDATED} words "
                                     f"(model ceiling {FLASHCARD_FRONT_MAX_WORDS})"})
        if len((card.question or "").split()) < 3:
            errors.append({"card_id": card.id, "code": "ambiguous",
                           "detail": "front too short to identify the concept"})
        if not _is_valid_topic_title(card.topic):
            errors.append({"card_id": card.id, "code": "card_topic_invalid",
                           "detail": f"topic {card.topic!r} failed "
                                     f"_is_valid_topic_title"})
        if card.concept_id not in concept_ids:
            errors.append({"card_id": card.id, "code": "unknown_concept",
                           "detail": f"concept_id {card.concept_id!r} resolves nowhere"})
        for cite in card.citations or []:
            cid = cite.chunk_id if hasattr(cite, "chunk_id") else cite.get("chunk_id")
            if cid not in allowed_chunk_ids:
                errors.append({"card_id": card.id, "code": "unsupported_cite",
                               "detail": f"citation {cid!r} outside the prompt"})
        if not (card.metadata or {}).get("provenance"):
            errors.append({"card_id": card.id, "code": "missing_provenance",
                           "detail": "metadata.provenance is required"})
        review = card.review or {}
        for key, default in FSRS_NEW_REVIEW.items():
            if key not in review:
                errors.append({"card_id": card.id, "code": "missing_fsrs_field",
                               "detail": f"review.{key} missing (FSRS-ready required)"})
                break
    realized = _realized_mix(cards)
    # Bloom guard: within tolerance per level AND >= 1 card per level that
    # has material (levels with zero configured share are exempt).
    bloom_ok = True
    shortfalls: List[str] = []
    counts: Dict[str, int] = {}
    for card in cards:
        level = card.bloom_level.value if isinstance(card.bloom_level, BloomLevel) \
            else str(card.bloom_level)
        counts[level] = counts.get(level, 0) + 1
    for level in BLOOM_LEVELS:
        want = blend.get(level.value, 0.0)
        got = realized.get(level.value, 0.0)
        if abs(got - want) > tolerance + 1e-9:
            bloom_ok = False
            shortfalls.append(f"{level.value}: want {want:.2f}, got {got:.2f}")
        if want > 0 and counts.get(level.value, 0) < 1:
            bloom_ok = False
            shortfalls.append(f"{level.value}: no card at this level")
    ok = not errors and bloom_ok
    return FlashcardDeckReport(
        ok=ok,
        total=len(cards),
        bloom_realized=realized,
        bloom_ok=bloom_ok,
        errors=errors,
        shortfall="; ".join(shortfalls),
    )


def repair_bloom_once(
    cards: List[FlashcardSpec],
    mix: Optional[Dict[str, float]] = None,
) -> Tuple[List[FlashcardSpec], FlashcardDeckReport, bool]:
    """One Bloom repair pass: re-deal levels toward the mix, then report.

    Returns (cards, report, repaired): repaired is True when the single
    pass ran; a still-violating deck is reported honestly (shortfall
    named), never silently passed.
    """
    blend = dict(mix or BLOOM_MIX)
    targets = bloom_targets(len(cards), blend)
    quotas = dict(targets)
    fixed = []
    for card in cards:
        pick = next(
            (level for level in BLOOM_LEVELS if quotas.get(level.value, 0) > 0),
            None,
        )
        if pick is None:
            fixed.append(card)
            continue
        quotas[pick.value] -= 1
        payload = card.model_dump(by_alias=True)
        # Preserve the "type"/"cognitive"/"cites" wire aliases on rebuild.
        payload["cognitive"] = pick.value
        payload.pop("bloom_level", None)
        payload["difficulty"] = _DIFFICULTY_BY_BLOOM[pick].value
        fixed.append(FlashcardSpec.model_validate(payload))
    # Concept/cite universes for the report: derive from the cards so the
    # repair verdict measures the Bloom guard only (concept/cite validity
    # is the caller's validate_deck pass, unchanged by re-dealing).
    allowed = {c.chunk_id for card in fixed for c in (card.citations or [])}
    concepts = {card.concept_id for card in fixed}
    report = validate_deck(fixed, allowed, concepts, blend)
    return fixed, report, True


def to_deck_cards(cards: List[FlashcardSpec]) -> List[Dict[str, str]]:
    """Project typed cards to the Flashcards UI contract (front/back/difficulty).

    Decks persist through the existing flashcard-deck repository with this
    projection only; the rich typed set lives in the artifact payload.
    """
    return [{
        "front": card.question,
        "back": card.answer,
        "difficulty": (card.difficulty.value
                       if isinstance(card.difficulty, CardDifficulty)
                       else str(card.difficulty)),
    } for card in cards]


def flashcards_artifact(
    deck: FlashcardSetSpec,
    report: FlashcardDeckReport,
) -> Dict[str, Any]:
    """Artifact payload: rich typed set + Bloom verdict (DB: node_metadata)."""
    return {
        "title": deck.title,
        "flashcards": [card.model_dump(by_alias=True, mode="json")
                       for card in deck.flashcards],
        "bloom_realized": dict(deck.bloom_realized),
        "bloom_ok": report.bloom_ok,
        "validation_errors": list(report.errors),
        "shortfall": report.shortfall,
        "provenance_note": deck.provenance_note,
    }


async def generate_flashcards_typed(
    engine: Any,
    context: str,
    allowed_chunk_ids: set,
    query: Optional[str] = None,
    bloom_mix: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Live LLM typed-flashcards call through the shared schema-enforced path.

    Uses engine._generate_structured (FLASHCARDS_TYPED mode, max 2 repairs
    on validation failure only) so provider selection, schema validation
    and the citation check stay shared.
    """
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import GenerationMode

    prompt = PromptTemplates.get_prompt(
        GenerationMode.FLASHCARDS_TYPED, context, query or "typed flashcards",
        bloom_mix=bloom_mix or BLOOM_MIX,
    )
    return await engine._generate_structured(
        GenerationMode.FLASHCARDS_TYPED, prompt, allowed_chunk_ids
    )


__all__ = [
    "BLOOM_LEVELS",
    "CARD_TYPES",
    "concept_index",
    "bloom_targets",
    "build_flashcards",
    "validate_deck",
    "repair_bloom_once",
    "to_deck_cards",
    "flashcards_artifact",
    "generate_flashcards_typed",
]
