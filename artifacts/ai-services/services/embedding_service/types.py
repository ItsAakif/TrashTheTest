"""
Embedding Generation Types and Models

Pydantic models for embedding input, output, and progress tracking.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class EmbeddingChunkInput(BaseModel):
    """Input for embedding a single chunk"""
    chunk_id: str
    content: str
    content_hash: str  # For caching
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingBatchInput(BaseModel):
    """Input for embedding a batch of chunks"""
    document_id: str
    project_id: str
    chunks: List[EmbeddingChunkInput]


class EmbeddingProgress(BaseModel):
    """Progress tracking for embedding generation"""
    document_id: str
    current_batch: int
    total_batches: int
    chunks_processed: int
    total_chunks: int
    status: str  # "embedding", "processing", "completed", "failed"
    progress_percent: float
    message: Optional[str] = None
    cached_count: int = 0  # How many were cached


class EmbeddingResult(BaseModel):
    """Complete result for embedding generation"""
    document_id: str
    project_id: str
    total_chunks: int
    embedded_chunks: int
    cached_chunks: int  # Reused from cache
    total_tokens: int
    processing_time: float
    is_complete: bool = True
    
    @property
    def new_embeddings_generated(self) -> int:
        return self.embedded_chunks - self.cached_chunks


class EmbeddingJobInput(BaseModel):
    """Input for embedding job"""
    document_id: str
    project_id: str
    batch_size: int = 32  # Chunks per batch
    force_regenerate: bool = False  # Force regenerate even if cached


class EmbeddingStats(BaseModel):
    """Statistics about embeddings for a project"""
    project_id: str
    total_chunks: int
    chunks_with_embeddings: int
    chunks_without_embeddings: int
    average_tokens_per_chunk: float


# Configuration for embedding
class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation"""
    batch_size: int = 32
    model_dimensions: int = 384  # BGE-small dimension
    cache_enabled: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


__all__ = [
    "EmbeddingChunkInput",
    "EmbeddingBatchInput",
    "EmbeddingProgress",
    "EmbeddingResult",
    "EmbeddingJobInput",
    "EmbeddingStats",
    "EmbeddingConfig",
]