"""
Semantic Chunking Types and Models

Pydantic models for chunking input, output, and progress tracking.
"""

from typing import Optional, List, Dict, Any, Set
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ChunkMetadata(BaseModel):
    """Metadata for a single chunk (citation-grade provenance)."""
    page_number: int
    section: Optional[str] = None
    heading: Optional[str] = None
    position: int  # Position within document
    document_id: str
    # T207 enrichment: derived from document_pages, never a batch index.
    project_id: Optional[str] = None
    user_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[List[str]] = None
    source: Optional[str] = None  # storage key / file name
    method: Optional[str] = None  # extraction method of the source page
    confidence: Optional[float] = None
    # ADR-008 / T003: how the section was detected + how sure we are.
    structure_method: Optional[str] = None
    structure_confidence: Optional[float] = None


class ChunkResult(BaseModel):
    """Result for a single chunk"""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: ChunkMetadata
    token_count: int
    content_hash: str  # For caching embeddings


class DocumentChunkingInput(BaseModel):
    """Input for document chunking"""
    document_id: str
    project_id: str
    full_text: str  # Complete document text from extraction
    structural_elements: List[Dict[str, Any]] = Field(default_factory=list)
    # Structural elements from extraction (headings, sections, etc.)
    # T207 provenance inputs (optional; service fills from document_pages).
    user_id: Optional[str] = None
    source: Optional[str] = None
    page_spans: Optional[List[Dict[str, Any]]] = None  # [{page, start, end, method, confidence, heading}]
    # Running-header / page-number lines (from find_boilerplate_lines over the
    # per-page texts); the structure detector drops them before heading
    # candidacy so they never shape the chunk structure.
    boilerplate_lines: Set[str] = Field(default_factory=set)


class ChunkingProgress(BaseModel):
    """Progress tracking for chunking"""
    document_id: str
    current_batch: int
    total_batches: int
    chunks_created: int
    status: str  # "chunking", "processing", "completed", "failed"
    progress_percent: float
    message: Optional[str] = None
    last_processed_position: Optional[int] = None


class ChunkingResult(BaseModel):
    """Complete result for document chunking"""
    document_id: str
    project_id: str
    total_chunks: int
    chunks: List[ChunkResult]
    total_tokens: int
    processing_time: float
    is_complete: bool = True
    
    @property
    def average_tokens_per_chunk(self) -> float:
        if self.total_chunks == 0:
            return 0
        return self.total_tokens / self.total_chunks


class ChunkingJobInput(BaseModel):
    """Input for chunking job"""
    document_id: str
    project_id: str
    start_position: Optional[int] = None  # For resume capability
    batch_size: int = 50  # Pages per batch for large documents


# Configuration for chunking
class ChunkingConfig(BaseModel):
    """Configuration for semantic chunking"""
    min_chunk_size: int = 300  # Minimum characters
    max_chunk_size: int = 1000  # Maximum characters
    overlap_size: int = 100  # Characters to overlap between chunks
    batch_size: int = 50  # Pages per batch for large documents
    
    # Structural priority - split by these boundaries first
    split_on_headings: bool = True
    split_on_paragraphs: bool = True
    split_on_sentences: bool = True  # Fallback


__all__ = [
    "ChunkMetadata",
    "ChunkResult",
    "DocumentChunkingInput",
    "ChunkingProgress",
    "ChunkingResult",
    "ChunkingJobInput",
    "ChunkingConfig",
]