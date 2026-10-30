"""Graph states (TypedDicts). Single source of shape for nodes/edges/tests."""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class IngestionState(TypedDict, total=False):
    document_id: str
    file_path: str
    file_type: str
    page_count: int
    classification: str  # born-digital | scanned | mixed | table-heavy
    pages: list  # per-page dicts {page, text, method, confidence}
    gate: dict  # quality-gate signals per page
    ocr_route: str  # ppocr | puter-offered | none
    puter_authorised: bool
    cleaned_text: str
    structure: list
    chunks: list
    embeddings_done: bool
    persist_ok: bool
    validation: dict
    attempts: dict  # node -> attempt count (bounded retries)
    error: Optional[str]


class TutorState(TypedDict, total=False):
    query: str
    intent: str
    transformed_query: str
    project_id: Optional[str]
    document_id: Optional[str]
    candidates: list
    fused: list
    reranked: list
    answer: str
    citations: list
    valid: bool
    refine_rounds: int
    attempts: dict
    error: Optional[str]


class ResearchState(TypedDict, total=False):
    question: str
    plan: list
    search_results: list
    deduped: list
    eval_scores: dict
    synthesis: str
    citations: list
    attempts: dict
    error: Optional[str]


class StudyState(TypedDict, total=False):
    content: str
    check: dict
    eval_score: float
    difficulty: str  # easy | medium | hard
    next_step: str
    attempts: dict
    error: Optional[str]


__all__ = ["IngestionState", "TutorState", "ResearchState", "StudyState"]
