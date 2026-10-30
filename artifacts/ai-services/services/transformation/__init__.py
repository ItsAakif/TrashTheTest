"""
Transformation Engine Module

Unified transformation engine for generating educational content
(notes, flashcards, quiz, explanation, summary, website) from
retrieved document context using LLM generation.
"""

from .types import (
    GenerationMode,
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    Flashcard,
    QuizQuestion,
    NoteSection,
    WebsiteContent,
    CachedContent,
    FlashcardList,
    Quiz,
    StructuredNotes,
    SummaryOutput,
    WebsiteOutput,
    SyllabusUnit,
    Syllabus,
    PYQQuestion,
    PYQSet,
    UnitMappingEntry,
    MappingResult,
    UnitFrequency,
    AnalysisResult,
    KnowledgeSpineNodeOutput,
    KnowledgeSpineOutput,
    StructureOutput,
    TopicNotesOutput,
    HierNotesOutput,
    FlashcardsTypedOutput,
    MCQMatrixOutput,
    MetricsNarrativeOutput,
)

from .prompts import PromptTemplates, augment_with_research

from .engine import (
    TransformationEngine,
    transformation_engine,
    normalize_text,
    dedupe_flashcards,
    dedupe_mcqs,
    dedupe_pyqs,
    sanitise_display_text,
    score_unit_for_question,
    map_questions_to_units,
    compute_frequency,
    collect_cited_ids,
    validate_grounding,
    check_hallucination,
)

from .routes import router


__all__ = [
    # Types
    "GenerationMode",
    "GenerationConfig",
    "GenerationRequest",
    "GenerationResult",
    "Flashcard",
    "QuizQuestion",
    "NoteSection",
    "WebsiteContent",
    "CachedContent",
    "FlashcardList",
    "Quiz",
    "StructuredNotes",
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
    "KnowledgeSpineNodeOutput",
    "KnowledgeSpineOutput",
    "StructureOutput",
    "TopicNotesOutput",
    "HierNotesOutput",
    "FlashcardsTypedOutput",
    "MCQMatrixOutput",
    "MetricsNarrativeOutput",
    # Prompts
    "PromptTemplates",
    "augment_with_research",
    # Engine
    "TransformationEngine",
    "transformation_engine",
    "normalize_text",
    "dedupe_flashcards",
    "dedupe_mcqs",
    "dedupe_pyqs",
    "sanitise_display_text",
    "score_unit_for_question",
    "map_questions_to_units",
    "compute_frequency",
    "collect_cited_ids",
    "validate_grounding",
    "check_hallucination",
    # Routes
    "router",
]