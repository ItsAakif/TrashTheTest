"""
Research Augmentation Types and Models

Pydantic models for research input, output, and configuration.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class ResearchQuery(BaseModel):
    """Input for research augmentation"""
    query: str
    project_id: Optional[str] = None
    max_sources: int = 3
    extract_full_content: bool = False


class ResearchSource(BaseModel):
    """A single research source"""
    url: str
    title: str
    content: str
    relevance_score: float
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchContext(BaseModel):
    """Structured research context"""
    original_query: str
    sources: List[ResearchSource]
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    processing_time: float
    cached: bool = False


class MergedContext(BaseModel):
    """Document context merged with research context"""
    document_context: str
    research_context: Optional[str]
    has_research: bool
    priority_note: str = "Document content has priority over research"


class ResearchConfig(BaseModel):
    """Configuration for research augmentation"""
    enabled: bool = True
    max_sources: int = 3
    max_content_length: int = 2000  # Characters per source
    min_relevance_score: float = 0.5
    cache_enabled: bool = True
    
    # Query extraction
    auto_extract_query: bool = True
    query_max_length: int = 100


class ResearchResult(BaseModel):
    """Result from research augmentation"""
    query: str
    context: MergedContext
    sources: List[ResearchSource]
    processing_time: float
    used_cache: bool


__all__ = [
    "ResearchQuery",
    "ResearchSource",
    "ResearchContext",
    "MergedContext",
    "ResearchConfig",
    "ResearchResult",
]