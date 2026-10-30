"""
Transformation Engine Types and Models

Pydantic models for generation input, output, configuration, and caching.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum
import uuid


class GenerationMode(str, Enum):
    """Available generation modes"""
    NOTES = "notes"
    FLASHCARDS = "flashcards"
    QUIZ = "quiz"
    EXPLANATION = "explanation"
    SUMMARY = "summary"
    WEBSITE = "website"
    # Phase 4: first-class structured outputs (additive; existing values kept).
    ANSWER = "answer"  # marks-aware grounded answer
    MCQ = "mcq"  # strict multiple-choice set
    # Academic Intelligence vertical (additive; existing values kept).
    SYLLABUS = "syllabus"  # syllabus extraction (units + topics)
    PYQ = "pyq"  # previous-year-question extraction
    MAPPING = "mapping"  # PYQ -> syllabus unit mapping
    ANALYSIS = "analysis"  # frequency / marks alignment analysis
    # Hierarchical academic notes, Steps 2-4 (additive; existing kept).
    STRUCTURE = "structure"  # STEP 2 deterministic structure extraction
    KNOWLEDGE_SPINE = "knowledge_spine"  # typed canonical spine extraction
    NOTES_TOPIC = "notes_topic"  # STEP 3 per-topic notes with cites
    NOTES_HIER = "notes_hierarchical"  # STEP 3 ordered synthesis
    # Practice + intelligence, Steps 5-7 (additive; existing kept).
    FLASHCARDS_TYPED = "flashcards_typed"  # STEP 5 typed spine-bound cards
    MCQ_MATRIX = "mcq_matrix"  # STEP 6 Topic x Bloom matrix MCQs
    METRICS_NARRATIVE = "metrics_narrative"  # STEP 7 explanation-only narrative


class GenerationConfig(BaseModel):
    """Configuration for content generation"""
    mode: GenerationMode
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    enable_research: bool = False
    max_context_chunks: int = 10
    custom_params: Dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(BaseModel):
    """Request for content generation"""
    project_id: str
    mode: GenerationMode
    query: Optional[str] = None  # Optional query for targeted generation
    document_id: Optional[str] = None  # Scope retrieval to one document (syllabus/PYQ)
    config: Optional[GenerationConfig] = None
    enable_research: bool = False


class CitationRef(BaseModel):
    """Provenance pointer: one model-cited chunk (must resolve to prompt ids)."""
    chunk_id: str = Field(min_length=1)
    document_id: Optional[str] = None
    page: Optional[int] = None
    span: str = ""  # short quoted span from the chunk (grounding check)

    model_config = {"extra": "forbid"}


class Provenance(BaseModel):
    """Grounding envelope attached to every Phase-4 resource."""
    support: Literal["retrieved", "generated", "unsupported"]
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    generated_without_citation: bool = False
    note: str = Field(min_length=1)

    model_config = {"extra": "forbid"}


class Flashcard(BaseModel):
    """A single flashcard"""
    front: str = Field(min_length=1)
    back: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    # Phase 4 provenance: chunk ids backing this card (subset of prompt ids).
    citations: List["CitationRef"] = Field(default_factory=list)


class QuizQuestion(BaseModel):
    """A single quiz question"""
    question: str = Field(min_length=1)
    options: List[str]
    correct_answer: int  # Index of correct option
    explanation: Optional[str] = None
    # Phase 4 provenance: chunk ids backing this question.
    citations: List["CitationRef"] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def _options_non_empty(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError("quiz question needs at least 2 options")
        if any(not (o or "").strip() for o in v):
            raise ValueError("quiz options must be non-empty strings")
        return v

    @field_validator("correct_answer")
    @classmethod
    def _correct_index_is_int(cls, v: int) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("correct_answer must be an integer index")
        return v

    @model_validator(mode="after")
    def _correct_index_in_range(self) -> "QuizQuestion":
        if not (0 <= self.correct_answer < len(self.options)):
            raise ValueError("correct_answer index out of range for options")
        return self


class NoteSection(BaseModel):
    """A section of structured notes"""
    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)
    subsections: List["NoteSection"] = Field(default_factory=list)
    # Phase 4 provenance: chunk ids backing this section.
    citations: List["CitationRef"] = Field(default_factory=list)


class WebsiteContent(BaseModel):
    """Website generation output"""
    title: str
    sections: List[Dict[str, Any]]  # Structured HTML sections
    styles: Dict[str, str] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Result from content generation"""
    project_id: str
    mode: GenerationMode
    content: Dict[str, Any]  # Mode-specific structured content
    cached: bool = False
    used_research: bool = False
    tokens_used: int = 0
    processing_time: float


class CachedContent(BaseModel):
    """Cached content entry"""
    content_hash: str
    output_type: str
    content: Dict[str, Any]
    created_at: datetime


# Validation models for output parsing
class FlashcardList(BaseModel):
    """Validated flashcard list"""
    flashcards: List[Flashcard]


class Quiz(BaseModel):
    """Validated quiz"""
    title: str
    questions: List[QuizQuestion]


class StructuredNotes(BaseModel):
    """Validated structured notes"""
    title: str
    sections: List[NoteSection]


class SummaryOutput(BaseModel):
    """Validated summary"""
    title: str
    summary: str
    key_points: List[str]


class WebsiteOutput(BaseModel):
    """Validated website content"""
    title: str
    content: str  # HTML content
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 4 strict schemas: Note, Flashcard set, MCQ, Explanation, Answer.
# Every grounded resource carries citations + provenance; validation is
# strict (no silent malformed accept); marks-aware answers adapt constraints
# by marks via MARKS_CONSTRAINTS / constraints_for_marks.
# ---------------------------------------------------------------------------

AnswerStyle = Literal["concise", "detailed", "exam", "tutorial"]

# marks -> (min_words, max_words): calibrated bands, interpolated between.
MARKS_CONSTRAINTS: Dict[int, tuple] = {
    1: (15, 60),
    2: (40, 120),
    3: (80, 200),
    5: (150, 350),
    10: (300, 700),
}


def constraints_for_marks(marks: int) -> Dict[str, int]:
    """Word-count band for a marks value (nearest-band, interpolated)."""
    if marks <= 0:
        raise ValueError("marks must be a positive integer")
    keys = sorted(MARKS_CONSTRAINTS)
    if marks <= keys[0]:
        lo, hi = MARKS_CONSTRAINTS[keys[0]]
    elif marks >= keys[-1]:
        lo, hi = MARKS_CONSTRAINTS[keys[-1]]
    else:
        lower = max(k for k in keys if k <= marks)
        upper = min(k for k in keys if k >= marks)
        if lower == upper:
            lo, hi = MARKS_CONSTRAINTS[lower]
        else:
            t = (marks - lower) / (upper - lower)
            llo, lhi = MARKS_CONSTRAINTS[lower]
            ulo, uhi = MARKS_CONSTRAINTS[upper]
            lo, hi = (round(llo + (ulo - llo) * t), round(lhi + (uhi - lhi) * t))
    return {"marks": marks, "min_words": lo, "max_words": hi}


class Note(BaseModel):
    """Phase-4 strict structured note document (grounded)."""
    title: str = Field(min_length=1)
    sections: List[NoteSection] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    model_config = {"extra": "forbid"}


class FlashcardSet(BaseModel):
    """Phase-4 strict flashcard set (grounded, deduplicated by engine)."""
    title: str = Field(min_length=1)
    flashcards: List[Flashcard] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    model_config = {"extra": "forbid"}


class MCQ(BaseModel):
    """Phase-4 strict single multiple-choice question (exactly 4 options)."""
    question: str = Field(min_length=1)
    options: List[str] = Field(min_length=4, max_length=4)
    correct_answer: int
    explanation: str = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)

    @field_validator("options")
    @classmethod
    def _mcq_options_distinct(cls, v: List[str]) -> List[str]:
        if any(not (o or "").strip() for o in v):
            raise ValueError("MCQ options must be non-empty strings")
        if len({o.strip().lower() for o in v}) != 4:
            raise ValueError("MCQ options must be 4 distinct choices")
        return v

    @model_validator(mode="after")
    def _mcq_index_in_range(self) -> "MCQ":
        if not isinstance(self.correct_answer, int) or isinstance(self.correct_answer, bool):
            raise ValueError("correct_answer must be an integer index")
        if not (0 <= self.correct_answer < 4):
            raise ValueError("correct_answer must be in 0..3")
        return self

    model_config = {"extra": "forbid"}


class MCQSet(BaseModel):
    """Phase-4 strict MCQ set (grounded)."""
    title: str = Field(min_length=1)
    questions: List[MCQ] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    model_config = {"extra": "forbid"}


class Explanation(BaseModel):
    """Phase-4 strict concept explanation (grounded)."""
    explanation: str = Field(min_length=1)
    examples: List[str] = Field(default_factory=list)
    related_concepts: List[str] = Field(default_factory=list)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    @field_validator("examples", "related_concepts")
    @classmethod
    def _entries_non_empty(cls, v: List[str]) -> List[str]:
        if any(not (e or "").strip() for e in v):
            raise ValueError("examples/related_concepts must be non-empty strings")
        return v

    model_config = {"extra": "forbid"}


class Answer(BaseModel):
    """Phase-4 strict marks-aware answer (grounded where possible)."""
    question: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    marks: int = Field(gt=0, le=100)
    style: AnswerStyle = "detailed"
    target_min_words: int = Field(gt=0)
    target_max_words: int = Field(gt=0)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    @model_validator(mode="after")
    def _answer_bands_consistent(self) -> "Answer":
        if self.target_min_words > self.target_max_words:
            raise ValueError("target_min_words must not exceed target_max_words")
        band = constraints_for_marks(self.marks)
        # The model-declared band must cover the canonical marks band:
        # generators may widen, never silently narrow.
        if self.target_min_words > band["min_words"] or self.target_max_words < band["max_words"]:
            raise ValueError(
                f"target band [{self.target_min_words},{self.target_max_words}] "
                f"must cover marks band [{band['min_words']},{band['max_words']}]"
            )
        return self

    def word_count(self) -> int:
        return len(self.answer_text.split())

    def length_ok(self) -> bool:
        n = self.word_count()
        return self.target_min_words <= n <= self.target_max_words

    model_config = {"extra": "forbid"}


# Resolve forward refs (self-nested NoteSection, CitationRef attachments).
NoteSection.model_rebuild()
Flashcard.model_rebuild()
QuizQuestion.model_rebuild()
Note.model_rebuild()
FlashcardSet.model_rebuild()
MCQSet.model_rebuild()
Explanation.model_rebuild()
Answer.model_rebuild()


# ---------------------------------------------------------------------------
# Academic Intelligence vertical: syllabus extraction + PYQ extraction +
# frequency/unit mapping + marks alignment. Every extracted resource carries
# citations + provenance; validation is strict (extra=forbid, no silent
# malformed accept); mapping/analysis are deterministic over extracted
# resources (token-overlap, engine helpers) so results are reproducible and
# honest (unmapped questions are reported, never force-assigned).
# ---------------------------------------------------------------------------


class SyllabusUnit(BaseModel):
    """One syllabus unit: title + topic list (+ optional outcomes/weight)."""
    unit_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    topics: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
    weight_percent: Optional[float] = None
    citations: List[CitationRef] = Field(default_factory=list)

    @field_validator("topics", "outcomes")
    @classmethod
    def _entries_non_empty(cls, v: List[str]) -> List[str]:
        if any(not (e or "").strip() for e in v):
            raise ValueError("topics/outcomes must be non-empty strings")
        return v

    @field_validator("weight_percent")
    @classmethod
    def _weight_in_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0 <= v <= 100):
            raise ValueError("weight_percent must be in 0..100")
        return v

    model_config = {"extra": "forbid"}


class Syllabus(BaseModel):
    """Extracted syllabus: ordered units grounded in the uploaded document."""
    title: str = Field(min_length=1)
    course: Optional[str] = None
    units: List[SyllabusUnit] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    @model_validator(mode="after")
    def _unit_ids_unique(self) -> "Syllabus":
        ids = [u.unit_id for u in self.units]
        if len(set(ids)) != len(ids):
            raise ValueError("syllabus unit_id values must be unique")
        return self

    model_config = {"extra": "forbid"}


class PYQQuestion(BaseModel):
    """One extracted previous-year question (marks + year + unit mapping)."""
    question: str = Field(min_length=1)
    year: int = Field(ge=1950, le=2100)
    marks: int = Field(gt=0, le=100)
    unit_id: Optional[str] = None  # syllabus unit assignment (None = unmapped)
    topic: Optional[str] = None
    paper: Optional[str] = None
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class PYQSet(BaseModel):
    """Extracted PYQ collection grounded in the uploaded document."""
    title: str = Field(min_length=1)
    questions: List[PYQQuestion] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    model_config = {"extra": "forbid"}


class UnitMappingEntry(BaseModel):
    """One PYQ -> syllabus unit assignment (deterministic, scored)."""
    question_index: int = Field(ge=0)
    question: str = Field(min_length=1)
    unit_id: Optional[str] = None  # None = unmapped (reported, never forced)
    unit_title: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["token_overlap", "explicit_unit", "unmapped"] = "token_overlap"

    model_config = {"extra": "forbid"}


class MappingResult(BaseModel):
    """PYQ-to-unit mapping over a syllabus (honest unmapped reporting)."""
    syllabus_title: str = Field(min_length=1)
    mappings: List[UnitMappingEntry] = Field(default_factory=list)
    unmapped_count: int = Field(ge=0)
    coverage_pct: float = Field(ge=0.0, le=100.0)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    @model_validator(mode="after")
    def _counts_consistent(self) -> "MappingResult":
        unmapped = sum(1 for m in self.mappings if m.unit_id is None)
        if unmapped != self.unmapped_count:
            raise ValueError("unmapped_count must equal mappings with null unit_id")
        total = len(self.mappings)
        expected = round((total - unmapped) / total * 100.0, 2) if total else 0.0
        if abs(expected - self.coverage_pct) > 0.01:
            raise ValueError("coverage_pct inconsistent with mapped/total ratio")
        return self

    model_config = {"extra": "forbid"}


class UnitFrequency(BaseModel):
    """Per-unit question frequency + marks share."""
    unit_id: str = Field(min_length=1)
    unit_title: str = Field(min_length=1)
    count: int = Field(ge=0)
    marks_total: int = Field(ge=0)
    share_pct: float = Field(ge=0.0, le=100.0)

    model_config = {"extra": "forbid"}


class AnalysisResult(BaseModel):
    """Frequency + marks alignment analysis over mapped PYQs."""
    syllabus_title: str = Field(min_length=1)
    total_questions: int = Field(ge=0)
    mapped_questions: int = Field(ge=0)
    unmapped_questions: int = Field(ge=0)
    frequency: List[UnitFrequency] = Field(default_factory=list)
    total_marks: int = Field(ge=0)
    coverage_pct: float = Field(ge=0.0, le=100.0)
    citations: List[CitationRef] = Field(default_factory=list)
    provenance: Optional[Provenance] = None

    @model_validator(mode="after")
    def _totals_consistent(self) -> "AnalysisResult":
        if self.mapped_questions + self.unmapped_questions != self.total_questions:
            raise ValueError("mapped + unmapped must equal total_questions")
        if sum(f.count for f in self.frequency) != self.mapped_questions:
            raise ValueError("frequency counts must sum to mapped_questions")
        if sum(f.marks_total for f in self.frequency) != self.total_marks:
            raise ValueError("frequency marks must sum to total_marks")
        return self

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Hierarchical academic notes, Steps 2-4: wire shapes for the three new
# modes. Self-contained (no import of services.academic: the academic
# package imports the engine, so the dependency runs one way only). The
# academic domain models mirror these field-for-field; tests assert parity.
# ---------------------------------------------------------------------------


class StructureUnitShape(BaseModel):
    """One extracted unit: title + ordered topic titles."""

    title: str = Field(min_length=1)
    topics: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class StructureOutput(BaseModel):
    """STEP 2 wire shape: ordered structure (extraction before generation)."""

    title: str = Field(min_length=1)
    units: List[StructureUnitShape] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class KnowledgeSpineNodeOutput(BaseModel):
    """Wire node for the typed canonical spine mode.

    The persisted spine is validated by ``KnowledgeSpineSpec`` in the
    academic package.  This wire model keeps the generation layer strict
    without importing that package (and therefore avoids an import cycle).
    """

    node_type: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    aliases: List[str] = Field(default_factory=list)
    order_index: int = Field(ge=0)
    parent_ref: Optional[int] = Field(default=None, ge=0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    chunks: List[str] = Field(default_factory=list)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class KnowledgeSpineOutput(BaseModel):
    """Strict output shape for ``GenerationMode.KNOWLEDGE_SPINE``."""

    title: str = Field(min_length=1)
    nodes: List[KnowledgeSpineNodeOutput] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class TopicNotesOutput(BaseModel):
    """STEP 3 wire shape: per-topic notes with chunk cites."""

    unit_title: str = Field(min_length=1)
    topic_title: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    body_paragraphs: List[str] = Field(min_length=1)
    key_terms: List[str] = Field(default_factory=list)
    shape: str = "concept"
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class HierNotesOutput(BaseModel):
    """STEP 3 wire shape: ordered synthesis sections in spine order."""

    title: str = Field(min_length=1)
    sections: List[NoteSection] = Field(min_length=1)
    cross_topic_notes: List[str] = Field(default_factory=list)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


SyllabusUnit.model_rebuild()
Syllabus.model_rebuild()
PYQQuestion.model_rebuild()
PYQSet.model_rebuild()
UnitMappingEntry.model_rebuild()
MappingResult.model_rebuild()
UnitFrequency.model_rebuild()
AnalysisResult.model_rebuild()
StructureUnitShape.model_rebuild()
StructureOutput.model_rebuild()
KnowledgeSpineNodeOutput.model_rebuild()
KnowledgeSpineOutput.model_rebuild()
TopicNotesOutput.model_rebuild()
HierNotesOutput.model_rebuild()


# ---------------------------------------------------------------------------
# Practice + intelligence, Steps 5-7: wire shapes for the three new modes.
# Self-contained (no import of services.academic: the academic package
# imports the engine, so the dependency runs one way only). The academic
# domain models mirror these field-for-field; tests assert parity.
# ---------------------------------------------------------------------------


class FlashcardTypedWire(BaseModel):
    """STEP 5 wire shape: one typed spine-bound flashcard."""

    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    card_type: str = Field(min_length=1)
    difficulty: str = "medium"
    bloom_level: str = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class FlashcardsTypedOutput(BaseModel):
    """STEP 5 wire shape: typed deck with chunk cites."""

    title: str = Field(min_length=1)
    flashcards: List[FlashcardTypedWire] = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class MCQOptionWire(BaseModel):
    """STEP 6 wire shape: one rich option with error tag + rationale."""

    text: str = Field(min_length=1)
    is_correct: bool = False
    error_tag: Optional[str] = None
    rationale: str = ""

    model_config = {"extra": "forbid"}


class MCQWire(BaseModel):
    """STEP 6 wire shape: one matrix MCQ with rich options."""

    question: str = Field(min_length=1)
    options: List[MCQOptionWire] = Field(min_length=4, max_length=4)
    correct_answer: int
    explanation: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    difficulty: str = "medium"
    bloom_level: str = Field(min_length=1)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class MCQMatrixOutput(BaseModel):
    """STEP 6 wire shape: matrix questions with rich options."""

    title: str = Field(min_length=1)
    questions: List[MCQWire] = Field(default_factory=list)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class MetricsNarrativeOutput(BaseModel):
    """STEP 7 wire shape: explanation-only narrative over computed metrics."""

    narrative: str = Field(min_length=1)
    key_numbers: List[str] = Field(default_factory=list)
    citations: List[CitationRef] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


FlashcardTypedWire.model_rebuild()
FlashcardsTypedOutput.model_rebuild()
MCQOptionWire.model_rebuild()
MCQWire.model_rebuild()
MCQMatrixOutput.model_rebuild()
MetricsNarrativeOutput.model_rebuild()


__all__ = [
    "GenerationMode",
    "GenerationConfig",
    "GenerationRequest",
    "CitationRef",
    "Provenance",
    "AnswerStyle",
    "MARKS_CONSTRAINTS",
    "constraints_for_marks",
    "Flashcard",
    "QuizQuestion",
    "NoteSection",
    "WebsiteContent",
    "GenerationResult",
    "CachedContent",
    "FlashcardList",
    "FlashcardSet",
    "Quiz",
    "MCQ",
    "MCQSet",
    "StructuredNotes",
    "Note",
    "Explanation",
    "Answer",
    "SummaryOutput",
    "WebsiteOutput",
    "SyllabusUnit",
    "Syllabus",
    "PYQQuestion",
    "PYQSet",
    "UnitMappingEntry",
    "MappingResult",
    "UnitFrequency",
    "AnalysisResult",
    "StructureUnitShape",
    "StructureOutput",
    "KnowledgeSpineNodeOutput",
    "KnowledgeSpineOutput",
    "TopicNotesOutput",
    "HierNotesOutput",
    "FlashcardTypedWire",
    "FlashcardsTypedOutput",
    "MCQOptionWire",
    "MCQWire",
    "MCQMatrixOutput",
    "MetricsNarrativeOutput",
]