"""
Vector Retrieval Types and Models

Pydantic models for retrieval input, output, and configuration.
Phase 3 (T301-T304): citations with resolvable provenance, hybrid/RRF
metadata, honest indexing states, and frontend request aliases.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class RetrievalQuery(BaseModel):
    """Query for vector retrieval"""
    query: str
    project_id: Optional[str] = None
    document_id: Optional[str] = None
    top_k: int = 5
    min_similarity: float = 0.0  # Minimum similarity threshold
    include_metadata: bool = True
    # Phase 3: rerank switch for A/B comparison, pagination
    enable_rerank: Optional[bool] = None
    offset: int = 0


class CitationPayload(BaseModel):
    """Resolvable citation for one retrieved chunk (T303)."""
    chunk_id: str
    document_id: str
    doc_id: Optional[str] = None
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    page: Optional[int] = None
    section_path: List[str] = Field(default_factory=list)
    section: Optional[str] = None
    heading: Optional[str] = None
    method: Optional[str] = None
    confidence: Optional[float] = None
    span: str = ""


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with similarity score"""
    chunk_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_id: Optional[str] = None
    # Phase 3 provenance (T301/T303): page/section/score/source/citation
    page: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None  # 'vector' | 'lexical' | 'both' | 'reranked'
    citation: Optional[CitationPayload] = None


class RetrievalResult(BaseModel):
    """Result from vector retrieval"""
    query: str
    total_results: int
    chunks: List[RetrievedChunk]
    processing_time: float
    query_embedding_time: float = 0.0
    retrieval_time: float = 0.0
    # Phase 3: honest corpus state + fusion/rerank metadata (T301/T302/T304)
    state: Literal["ok", "indexing_in_progress", "unindexed", "empty_corpus"] = "ok"
    indexed_chunks: int = 0
    total_chunks: int = 0
    rerank_applied: bool = False
    reranker: Optional[str] = None
    rerank_latency_ms: Optional[float] = None
    truncated_query: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def average_similarity(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.similarity for c in self.chunks) / len(self.chunks)

    def citations(self) -> List[Dict[str, Any]]:
        """Generated-answer citation list [{doc_id, page, chunk_id}] (T303)."""
        out = []
        for c in self.chunks:
            if c.citation is not None:
                d = c.citation.model_dump()
                out.append({
                    "doc_id": d.get("document_id"),
                    "page": d.get("page_number"),
                    "chunk_id": d.get("chunk_id"),
                })
            else:
                out.append({
                    "doc_id": c.document_id,
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                })
        return out


class RetrievalStats(BaseModel):
    """Statistics about retrieval for a project"""
    project_id: str
    total_chunks: int
    chunks_with_embeddings: int
    average_similarity: float
    top_retrieved_topics: List[str] = Field(default_factory=list)


# Configuration for retrieval
class RetrievalConfig(BaseModel):
    """Configuration for vector retrieval (Phase 3 hybrid defaults)."""
    default_top_k: int = 5
    max_top_k: int = 50
    min_similarity_threshold: float = 0.0
    embedding_dimensions: int = 1024  # Qwen3-Embedding-0.6B (ADR-005)
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Query preprocessing
    query_max_length: int = 1000
    enable_query_expansion: bool = False

    # Hybrid fusion (T301)
    per_arm_candidates: int = 50
    enable_hybrid: bool = True
    rrf_k: int = 60

    # Rerank (T302): switchable off for A/B comparison
    enable_rerank: bool = True
    rerank_candidates: int = 50
    rerank_top_k: int = 8
    rerank_budget_ms: float = 2000.0

    # Multi-query settings (for future)
    enable_multi_query: bool = False
    multi_query_count: int = 3


# For research augmentation integration
class ResearchRetrievalResult(BaseModel):
    """Retrieval result prepared for research augmentation"""
    chunks: List[RetrievedChunk]
    context_summary: str
    source_document_ids: List[str]


__all__ = [
    "RetrievalQuery",
    "CitationPayload",
    "RetrievedChunk",
    "RetrievalResult",
    "RetrievalStats",
    "RetrievalConfig",
    "ResearchRetrievalResult",
]
