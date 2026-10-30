"""Deterministic tutor intent gating.

Tutor messages are not all academic questions.  Greetings and thanks must
not trigger retrieval, unrelated requests must not receive an academic dump,
and short follow-ups need a clarifying question.  The classifier is pure so
route tests and the frontend can share the same policy without a provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Sequence


class TutorIntent(str, Enum):
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    PROJECT_SPECIFIC = "project_specific"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class TutorIntentResult:
    intent: TutorIntent
    confidence: float
    reason: str
    retrieval_required: bool
    clarification: Optional[str] = None

    @property
    def needs_retrieval(self) -> bool:
        return self.retrieval_required

    def as_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "confidence": round(float(self.confidence), 3),
            "reason": self.reason,
            "retrieval_required": self.retrieval_required,
            "clarification": self.clarification,
        }


_GREETING_RE = re.compile(
    r"^(?:hi|hey|hello|yo|good\s+(?:morning|afternoon|evening|night)|"
    r"how\s+are\s+you|how's\s+it\s+going|nice\s+to\s+meet\s+you|"
    r"thanks?|thank\s+you|you're\s+welcome|bye|goodbye|see\s+you|"
    r"good\s+night)[.!]?$",
    re.IGNORECASE,
)
_ACADEMIC_RE = re.compile(
    r"\b(?:explain|define|definition|describe|summari[sz]e|outline|derive|"
    r"calculate|solve|compare|contrast|give\s+examples?|why|how\s+does|how\s+do|"
    r"what\s+is|what\s+are|quiz|flashcards?|notes?|revision|exam|formula|"
    r"theorem|proof|topic|concept|homework|assignment|practice)\b",
    re.IGNORECASE,
)
_PROJECT_RE = re.compile(
    r"\b(?:this\s+project|my\s+project|uploaded|document|documents|"
    r"source|sources|syllabus|textbook|chapter|unit|page\s+\d|according\s+to|"
    r"in\s+the\s+material|from\s+the\s+notes|topic\s+\d+)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_RE = re.compile(
    r"\b(?:weather|temperature|forecast|news|stock|share\s+price|football|"
    r"cricket|celebrity|movie\s+review|recipe|restaurant|restaurant|"
    r"current\s+events|political|election|medical\s+advice|diagnos(?:e|is))\b",
    re.IGNORECASE,
)
_FOLLOWUP_RE = re.compile(
    r"^(?:why|how|what|explain|more|tell\s+me\s+more|that|this|it|those|"
    r"those\s+steps|continue|go\s+on)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"[?？]\s*$")


def classify_tutor_intent(
    message: str,
    project_name: Optional[str] = None,
    topic_titles: Optional[Sequence[str]] = None,
    history: Optional[Iterable[str]] = None,
) -> TutorIntentResult:
    """Classify one message without retrieval or an LLM call."""

    text = " ".join((message or "").strip().split())
    if not text:
        return TutorIntentResult(
            TutorIntent.AMBIGUOUS, 1.0,
            "empty message", False,
            "What would you like help with?",
        )
    if _GREETING_RE.match(text):
        return TutorIntentResult(
            TutorIntent.CONVERSATIONAL, 1.0,
            "greeting, thanks, or social phrase", False,
        )

    # Explicit unrelated requests take precedence over generic question words.
    if _UNSUPPORTED_RE.search(text):
        return TutorIntentResult(
            TutorIntent.UNSUPPORTED, 0.98,
            "outside uploaded academic material", False,
            "I can only answer from your uploaded study material. I do not have a source for that.",
        )

    haystack = text.lower()
    topic_hit = any(
        len(normalized) > 2 and normalized in haystack
        for normalized in ((title or "").strip().lower() for title in (topic_titles or []))
    )
    project_hit = bool(project_name and str(project_name).strip().lower() in haystack)
    if _PROJECT_RE.search(text) or topic_hit or project_hit:
        return TutorIntentResult(
            TutorIntent.PROJECT_SPECIFIC, 0.94,
            "question names the active project, a source, or a known topic", True,
        )
    if _FOLLOWUP_RE.match(text) and any((item or "").strip() for item in (history or [])):
        return TutorIntentResult(
            TutorIntent.PROJECT_SPECIFIC, 0.86,
            "academic follow-up to an earlier grounded exchange", True,
        )
    if _ACADEMIC_RE.search(text) or _QUESTION_RE.search(text):
        # A short noun/question with no academic verb is ambiguous rather
        # than an invitation to dump the whole textbook.
        words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text)
        if len(words) <= 2 and not _ACADEMIC_RE.search(text):
            return TutorIntentResult(
                TutorIntent.AMBIGUOUS, 0.76,
                "short question without a clear academic operation", False,
                "Which topic or concept should I explain?",
            )
        return TutorIntentResult(
            TutorIntent.ACADEMIC, 0.90,
            "academic question", True,
        )
    if len(text.split()) <= 3:
        return TutorIntentResult(
            TutorIntent.AMBIGUOUS, 0.70,
            "message is too short to identify an academic target", False,
            "Could you name the topic or paste the question you want explained?",
        )
    return TutorIntentResult(
        TutorIntent.AMBIGUOUS, 0.58,
        "no clear academic or project intent", False,
        "I can help with a concept from your uploaded material. Which topic should we start?",
    )


def greeting_response(message: str) -> str:
    """Natural, deterministic response for a social message."""

    text = (message or "").strip().lower()
    if re.search(r"thank", text):
        return "You're welcome. Ask me about any topic in your study material whenever you need a hand."
    if re.search(r"\b(bye|goodbye|see you)\b", text):
        return "Goodbye. Come back when you want to revise a topic."
    if re.search(r"good (morning|afternoon|evening|night)", text):
        return f"Good {re.search(r'good (morning|afternoon|evening|night)', text).group(1)}! What would you like to study?"
    return "Hi! I can help you understand topics from your uploaded study material. What would you like to learn?"


__all__ = [
    "TutorIntent",
    "TutorIntentResult",
    "classify_tutor_intent",
    "greeting_response",
]
