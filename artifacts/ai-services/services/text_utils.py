"""Shared text hygiene helpers (PDF text-layer cleanup).

One place for the cleanup rules every pipeline stage needs:

- ``strip_control_chars`` turns the PDF bullet glyphs (private-use Cc
  codes) into readable bullets and drops the rest of the control range.
- ``collapse_letter_spacing`` repairs letter-spaced text layers
  ("s h e r y i a n s" -> "sheryians") without touching normal prose.
- ``collapse_spaces`` / ``normalise_text`` give every stage the same
  whitespace-normalised text.
- ``find_boilerplate_lines`` / ``remove_boilerplate`` keep running
  headers and page numbers out of chunk structure and concept names.
- ``is_usable_title_candidate`` is the shared "may this line be a
  title?" gate used by concept naming.
- ``clean_display_text`` sanitises generated card/MCQ display strings
  before validation so control characters never reach the DB or UI.
"""

import re
import unicodedata
from typing import Sequence

# Private-use control codes PDFs use for bullet glyphs.  They carry no
# semantic text, so they are replaced with a readable bullet, not dropped.
BULLET_CODES = {
    "\x0b", "\x0e", "\x0f", "\x10", "\x11", "\x12", "\x13", "\x14",
    "\x15", "\x16", "\x17", "\x18", "\x19", "\x1a", "\x1b",
}

# Terminal punctuation a title candidate may never end with.
_TITLE_STOP_SUFFIXES = {".", ",", ";", ":", "!", "?"}

# A standalone numeric token (page number glued to a title, list index).
_STANDALONE_NUMBER_RE = re.compile(r"(?:^|\s)\d{1,4}(?:\s|$)")

# Roman-numeral / arabic page-number lines.
_PAGE_NUMBER_RE = re.compile(r"^\d{1,3}$")
_ROMAN_NUMBER_RE = re.compile(r"^[ivxlc]{1,5}$", re.IGNORECASE)


def strip_control_chars(text: str) -> str:
    """Replace bullet-glyph control codes, drop other Cc/Cf chars.

    Bullet codes become ``"• "``; other control/format characters
    (Unicode category Cc or Cf) are dropped except ``\\n`` and ``\\t``.
    Runs of spaces are collapsed and each line is right-stripped;
    newlines are kept.
    """
    if not text:
        return ""
    out = []
    for ch in text:
        if ch in BULLET_CODES:
            out.append("• ")
        elif ch in ("\n", "\t"):
            out.append(ch)
        elif unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        else:
            out.append(ch)
    joined = "".join(out)
    lines = [re.sub(r" {2,}", " ", line).rstrip(" ") for line in joined.split("\n")]
    return "\n".join(lines)


def collapse_letter_spacing(text: str) -> str:
    """Collapse letter-spaced lines ("s h e r y" -> "shery").

    Only lines where >= 60% of the non-empty space-separated tokens are
    single alphabetic characters are rewritten, so normal prose keeps its
    spaces.  Lines are joined back with ``"\\n"``.
    """
    if not text:
        return ""
    lines = []
    for line in text.split("\n"):
        tokens = [t for t in line.split(" ") if t]
        singles = [t for t in tokens if len(t) == 1 and t.isalpha()]
        if tokens and len(singles) / len(tokens) >= 0.6:
            line = re.sub(r"(?<=\b\w) (?=\w\b)", "", line)
        lines.append(line)
    return "\n".join(lines)


def collapse_spaces(text: str) -> str:
    """Collapse intra-line whitespace; keep paragraph breaks; strip ends."""
    collapsed = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


def normalise_text(text: str) -> str:
    """Full normalisation: letter spacing, control chars, whitespace."""
    return collapse_spaces(strip_control_chars(collapse_letter_spacing(text or "")))


def is_page_number_line(line: str) -> bool:
    """True for bare page-number lines: ``^\\d{1,3}$`` or a roman numeral."""
    s = (line or "").strip()
    return bool(_PAGE_NUMBER_RE.match(s) or _ROMAN_NUMBER_RE.match(s))


def find_boilerplate_lines(
    texts: Sequence[str],
    threshold: float = 0.25,
    min_count: int = 4,
) -> set:
    """Lines that repeat across many PAGES (running headers/footers).

    ``texts`` holds one raw string per PDF page.  Each line is normalised
    with ``collapse_spaces(collapse_letter_spacing(line))`` and counted
    once per page.  A line is boilerplate when it appears on at least
    ``min_count`` pages AND on at least ``threshold`` fraction of all
    pages (with 70 pages, a 69-page header hits this; a 3-page line
    does not).
    """
    pages = [str(t or "") for t in (texts or [])]
    if not pages:
        return set()
    counts: dict = {}
    for page in pages:
        seen_here = set()
        for line in page.split("\n"):
            norm = collapse_spaces(collapse_letter_spacing(line))
            if norm and norm not in seen_here:
                seen_here.add(norm)
        for norm in seen_here:
            counts[norm] = counts.get(norm, 0) + 1
    total = len(pages)
    return {
        line for line, count in counts.items()
        if count >= min_count and count / total >= threshold
    }


def remove_boilerplate(text: str, boilerplate: set) -> str:
    """Drop boilerplate lines and page-number lines from ``text``."""
    forbidden = {str(b) for b in (boilerplate or set())}
    kept = []
    for line in str(text or "").split("\n"):
        norm = collapse_spaces(collapse_letter_spacing(line))
        if norm in forbidden or is_page_number_line(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def is_usable_title_candidate(line: str, forbidden: set = frozenset()) -> bool:
    """True when ``line`` may be used as a title/concept name.

    Rejects empty/oversized lines, lines without alphabetic characters,
    lines ending in sentence punctuation, page-number lines, lines in
    ``forbidden`` (normalised, case-insensitive), lines with a standalone
    numeric token (page number glued to a title), and ``" -- "`` titles.
    """
    s = collapse_spaces(collapse_letter_spacing(line)).strip()
    if not s:
        return False
    if len(s) < 2 or len(s) > 60:
        return False
    words = s.split()
    if len(words) < 1 or len(words) > 8:
        return False
    if not any(ch.isalpha() for ch in s):
        return False
    if s[-1:] in _TITLE_STOP_SUFFIXES:
        return False
    if is_page_number_line(s):
        return False
    forbidden_norm = {str(f).lower() for f in (forbidden or set())}
    if s.lower() in forbidden_norm:
        return False
    if _STANDALONE_NUMBER_RE.search(s):
        return False
    if " -- " in s:
        return False
    return True


def clean_display_text(text: str) -> str:
    """Sanitise generated display text (card/MCQ strings) before validation."""
    return collapse_spaces(strip_control_chars(text or ""))


# ---------------------------------------------------------------------------
# Notes-quality rules (R1-R4): sentence backing, front/back matter,
# junk titles and concept anchoring.  Shared by extraction (concept
# seeding) and flashcards (card eligibility) so both stages agree.
# ---------------------------------------------------------------------------

# Sentence-ish split: whitespace after terminal punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Cover/credits/navigation markers that never name a concept.  Word-start
# anchored (not word-end) so "download" catches "Downloading" and
# "sheryians" catches the letter-spaced header "sheryianscodingschool".
_JUNK_TITLE_PATTERNS = (
    r"sheryians", r"much love", r"answer true", r"download", r"install",
    r"setup", r"follow", r"subscribe", r"thank you", r"questions",
    r"credits", r"references", r"agenda", r"outline", r"click",
    r"page \d+", r"slide \d+", r"http", r"www",
)
_JUNK_TITLE_RE = re.compile(
    r"(?:\b(?:" + "|".join(_JUNK_TITLE_PATTERNS) + r")|@)", re.IGNORECASE
)


def split_sentences(text: str) -> list:
    """Split on whitespace following ``[.!?]`` (sentence-ish units)."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(str(text or "")) if s.strip()]


def count_qualifying_sentences(text: str, min_words: int = 8) -> int:
    """Count sentences that end in ``[.!?]`` and hold >= ``min_words`` words."""
    count = 0
    for sentence in split_sentences(text):
        if sentence[-1:] in ".!?" and len(sentence.split()) >= min_words:
            count += 1
    return count


def is_sentence_backed(text: str, min_words: int = 8) -> bool:
    """R1: text holds >= 1 sentence of >= ``min_words`` words ending in [.!?].

    A chunk that is not sentence-backed may not seed a concept or a card.
    """
    return count_qualifying_sentences(text, min_words) >= 1


def is_front_back_matter(index: int, total: int, fraction: float = 0.1) -> bool:
    """R2 zone: is ``index`` in the first/last ``fraction`` of ``total`` chunks?

    The window size is ``int(total * fraction)`` (floored), so documents
    with fewer than ``1/fraction`` chunks have an empty window: 10% of a
    3-chunk document is not a meaningful front/back-matter window.
    """
    if total <= 0:
        return False
    edge = int(total * fraction)
    if edge <= 0:
        return False
    return index < edge or index >= total - edge


def is_front_back_admissible(
    index: int,
    total: int,
    text: str,
    min_sentences: int = 2,
    fraction: float = 0.1,
) -> bool:
    """R2: edge chunks are admissible only with >= ``min_sentences`` sentences."""
    if not is_front_back_matter(index, total, fraction):
        return True
    return count_qualifying_sentences(text) >= min_sentences


def _token_set(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if t}


def is_junk_title(
    text: str,
    document_title: str = "",
    unit_title: str = "",
    boilerplate=None,
) -> bool:
    """R3: True when ``text`` may never be a concept/topic name.

    Rejects: empty text; the normalised token set equal to or a subset of
    the document/unit title tokens; the exact token set of any
    running-header/boilerplate line; and credits/cover/navigation markers
    (``sheryians``, ``much love``, ``answer true``, ``download`` ...).
    """
    norm = " ".join(str(text or "").split())
    if not norm:
        return True
    tokens = _token_set(norm)
    if not tokens:
        return True
    title_tokens = _token_set(document_title) | _token_set(unit_title)
    if title_tokens and tokens <= title_tokens:
        return True
    for line in (boilerplate or ()):
        line_tokens = _token_set(line)
        if line_tokens and tokens == line_tokens:
            return True
    if _JUNK_TITLE_RE.search(norm):
        return True
    return False


def appears_early(text: str, needle: str, fraction: float = 0.4) -> bool:
    """R4: ``needle`` starts within the first ``fraction`` of ``text`` chars."""
    hay = str(text or "")
    target = str(needle or "")
    if not target:
        return False
    index = hay.find(target)
    if index < 0:
        return False
    return index <= fraction * max(1, len(hay))


__all__ = [
    "BULLET_CODES",
    "strip_control_chars",
    "collapse_letter_spacing",
    "collapse_spaces",
    "normalise_text",
    "is_page_number_line",
    "find_boilerplate_lines",
    "remove_boilerplate",
    "is_usable_title_candidate",
    "clean_display_text",
    "split_sentences",
    "count_qualifying_sentences",
    "is_sentence_backed",
    "is_front_back_matter",
    "is_front_back_admissible",
    "is_junk_title",
    "appears_early",
]
