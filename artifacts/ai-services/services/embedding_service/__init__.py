"""
Embedding Service Module

Generates vector embeddings for document chunks using local models.
Supports batch processing, caching, and incremental generation.
"""

from .types import (
    EmbeddingChunkInput,
    EmbeddingBatchInput,
    EmbeddingProgress,
    EmbeddingResult,
    EmbeddingJobInput,
    EmbeddingStats,
    EmbeddingConfig,
)

from .service import LocalEmbeddingService, embedding_service

from .routes import router


__all__ = [
    # Types
    "EmbeddingChunkInput",
    "EmbeddingBatchInput",
    "EmbeddingProgress",
    "EmbeddingResult",
    "EmbeddingJobInput",
    "EmbeddingStats",
    "EmbeddingConfig",
    # Service
    "LocalEmbeddingService",
    "embedding_service",
    # Routes
    "router",
]