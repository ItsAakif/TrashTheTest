"""
Vector Retrieval Module

Retrieves relevant chunks from the vector database using cosine similarity
with pgvector. Used by downstream features for RAG and context retrieval.
"""

from .types import (
    RetrievalQuery,
    CitationPayload,
    RetrievedChunk,
    RetrievalResult,
    RetrievalStats,
    RetrievalConfig,
    ResearchRetrievalResult,
)

from .service import VectorRetrievalService, retrieval_service
from .hybrid import rrf_fuse, build_tsquery, normalize_query, FUSED_SQL
from .citations import (
    Citation,
    build_citation_from_chunk,
    resolve_citation,
    validate_citation_ids,
)

from .routes import router


__all__ = [
    # Types
    "RetrievalQuery",
    "CitationPayload",
    "RetrievedChunk",
    "RetrievalResult",
    "RetrievalStats",
    "RetrievalConfig",
    "ResearchRetrievalResult",
    # Service
    "VectorRetrievalService",
    "retrieval_service",
    # Hybrid + citations
    "rrf_fuse",
    "build_tsquery",
    "normalize_query",
    "FUSED_SQL",
    "Citation",
    "build_citation_from_chunk",
    "resolve_citation",
    "validate_citation_ids",
    # Routes
    "router",
]