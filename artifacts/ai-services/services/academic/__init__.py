"""Academic Knowledge Spine package (STEP 0 baseline + STEP 1 spine + STEPS 2-4).

STEP 0: services.academic.baseline runs the CURRENT single-shot Notes
pipeline deterministically over the synthetic corpus and measures it.
STEP 1: services.academic.types + spine provide the typed spine and the
shared-registry alias resolution (no graph database).
STEP 2: services.academic.extraction detects the ordered document structure
BEFORE generation (TOC/headings win, conservative inference otherwise).
STEP 3: services.academic.notes_hier renders per-topic TopicNotes and
synthesises them in spine order (never similarity-sorted).
STEP 4: services.academic.coverage evaluates concept coverage and repairs
only uncovered topics (bounded passes, revert-on-regression).
STEP 5: services.academic.flashcards builds typed spine-bound cards with
a Bloom guard and FSRS-ready review defaults (scheduler deferred).
STEP 6: services.academic.mcq plans a Topic x Bloom matrix and builds
error-tagged MCQs; the study-compatible string projection is separate.
STEP 7: services.academic.metrics computes deterministic intelligence
metrics; the LLM explains the results only (grounded narrative).
"""

from services.academic.baseline import measure_baseline, run_baseline
from services.academic.coverage import (
    attach_report,
    evaluate_coverage,
    missing_evidence,
    repair_notes,
)
from services.academic.extraction import extract_structure
from services.academic.flashcards import (
    bloom_targets,
    build_flashcards,
    concept_index,
    flashcards_artifact,
    repair_bloom_once,
    to_deck_cards,
    validate_deck,
)
from services.academic.mcq import (
    build_matrix,
    build_mcqs,
    mcq_artifact,
    project_quiz_questions,
    shingle_similarity,
    validate_mcq,
    validate_set,
)
from services.academic.metrics import (
    compute_metrics,
    metrics_artifact,
    metrics_numbers,
    normalize_inputs,
    recency_weight,
    repair_narrative,
    summary_cards,
    topic_table_rows,
    validate_narrative,
)
from services.academic.notes_pdf import (
    NotesPdfReport,
    notes_pdf_storage_hint,
    render_notes_pdf,
)
from services.academic.types import (
    ArtifactStatus,
    GeneratedArtifact,
    NotesGenerationConfig,
    NotesGenerationMode,
    NotesDetailLevel,
    NotesScope,
    artifact_fingerprint,
    recommend_notes_config,
)
from services.academic.sufficiency import (
    SufficiencyStatus,
    assess_source_sufficiency,
    recover_topic_evidence,
)
from services.academic.validation import (
    NotesValidationReport,
    validate_notes_artifact,
    validate_topic_notes,
)
from services.academic.source_ingestion import (
    SourceType,
    compute_source_metrics,
    map_questions_to_syllabus,
    parse_question_paper_text,
    parse_syllabus_text,
)

from services.academic.baseline import measure_baseline, run_baseline
from services.academic.coverage import (
    attach_report,
    evaluate_coverage,
    missing_evidence,
    repair_notes,
)
from services.academic.extraction import extract_structure
from services.academic.notes_hier import (
    build_outline,
    compress_text,
    estimate_tokens,
    group_topics,
    render_topic_notes,
    select_topic_chunks,
    shape_for_topic,
    synthesize_notes,
    topic_query,
)
from services.academic.spine import (
    assign_order_index,
    canonical_key_for,
    merge_alias_lists,
    normalize_key,
    resolve_alias,
    validate_spine_structure,
)
from services.academic.orchestrator import (
    JOB_TYPE as HIERARCHICAL_JOB_TYPE,
    NOTES_HIERARCHICAL_STAGES,
    NOTES_HIERARCHICAL_STAGE_PROGRESS_RANGES,
    AcademicPipelineError,
    NoExtractableTextError,
    build_and_store_pdf,
    build_project_spine,
    build_spine,
    generate_mcq_matrix,
    generate_metrics_report,
    generate_typed_flashcards,
    process_hierarchical_job,
    read_spine,
    read_typed_artifact,
)

__all__ = [
    "measure_baseline",
    "run_baseline",
    "extract_structure",
    "build_outline",
    "compress_text",
    "estimate_tokens",
    "group_topics",
    "render_topic_notes",
    "select_topic_chunks",
    "shape_for_topic",
    "synthesize_notes",
    "topic_query",
    "evaluate_coverage",
    "missing_evidence",
    "repair_notes",
    "attach_report",
    "assign_order_index",
    "canonical_key_for",
    "merge_alias_lists",
    "normalize_key",
    "resolve_alias",
    "validate_spine_structure",
    "concept_index",
    "bloom_targets",
    "build_flashcards",
    "validate_deck",
    "repair_bloom_once",
    "to_deck_cards",
    "flashcards_artifact",
    "build_matrix",
    "build_mcqs",
    "validate_mcq",
    "validate_set",
    "project_quiz_questions",
    "shingle_similarity",
    "mcq_artifact",
    "normalize_inputs",
    "compute_metrics",
    "recency_weight",
    "metrics_numbers",
    "validate_narrative",
    "repair_narrative",
    "topic_table_rows",
    "summary_cards",
    "metrics_artifact",
    "NotesPdfReport",
    "notes_pdf_storage_hint",
    "render_notes_pdf",
    "ArtifactStatus",
    "GeneratedArtifact",
    "artifact_fingerprint",
    "NotesGenerationConfig",
    "NotesGenerationMode",
    "NotesDetailLevel",
    "NotesScope",
    "recommend_notes_config",
    "SufficiencyStatus",
    "assess_source_sufficiency",
    "recover_topic_evidence",
    "NotesValidationReport",
    "validate_notes_artifact",
    "validate_topic_notes",
    "SourceType",
    "parse_syllabus_text",
    "parse_question_paper_text",
    "map_questions_to_syllabus",
    "compute_source_metrics",
    "HIERARCHICAL_JOB_TYPE",
    "NOTES_HIERARCHICAL_STAGES",
    "NOTES_HIERARCHICAL_STAGE_PROGRESS_RANGES",
    "AcademicPipelineError",
    "NoExtractableTextError",
    "build_project_spine",
    "build_spine",
    "build_and_store_pdf",
    "process_hierarchical_job",
    "generate_typed_flashcards",
    "generate_mcq_matrix",
    "generate_metrics_report",
    "read_spine",
    "read_typed_artifact",
]
