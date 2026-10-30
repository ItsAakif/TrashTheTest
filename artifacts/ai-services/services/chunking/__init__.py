"""
Semantic Chunking Module

Transforms extracted document text into semantically meaningful chunks
for downstream embedding and retrieval.
"""

from .types import (
    ChunkMetadata,
    ChunkResult,
    DocumentChunkingInput,
    ChunkingProgress,
    ChunkingResult,
    ChunkingJobInput,
    ChunkingConfig,
)

from .chunker import SemanticChunker, semantic_chunker

from .service import ChunkingService, chunking_service

from .routes import router


__all__ = [
    # Types
    "ChunkMetadata",
    "ChunkResult",
    "DocumentChunkingInput",
    "ChunkingProgress",
    "ChunkingResult",
    "ChunkingJobInput",
    "ChunkingConfig",
    # Chunker
    "SemanticChunker",
    "semantic_chunker",
    # Service
    "ChunkingService",
    "chunking_service",
    # Routes
    "router",
]