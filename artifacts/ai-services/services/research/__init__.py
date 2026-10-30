"""
Research Augmentation Module

Structured research system that augments document-based RAG
with real-time web data using Tavily and Jina Reader.
"""

from .types import (
    ResearchQuery,
    ResearchSource,
    ResearchContext,
    MergedContext,
    ResearchConfig,
    ResearchResult,
)

from .service import ResearchAugmentationService, research_service

from .routes import router


__all__ = [
    # Types
    "ResearchQuery",
    "ResearchSource",
    "ResearchContext",
    "MergedContext",
    "ResearchConfig",
    "ResearchResult",
    # Service
    "ResearchAugmentationService",
    "research_service",
    # Routes
    "router",
]