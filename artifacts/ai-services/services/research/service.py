"""
Research Augmentation Service

Structured research system that augments document-based RAG
with real-time web data using Tavily and Jina Reader.
"""

import asyncio
import hashlib
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from services.strategies.search.tavily import TavilyProvider
from .types import (
    ResearchQuery,
    ResearchSource,
    ResearchContext,
    MergedContext,
    ResearchConfig,
    ResearchResult,
)


class ResearchAugmentationService:
    """
    Structured research augmentation service.
    
    Pipeline (explicit and traceable):
    1. Extract research query (from user input or document chunks)
    2. Perform web search with Tavily
    3. Extract content with Jina Reader
    4. Summarize into structured context (LLM-based)
    5. Merge with document context (document priority)
    
    Features:
    - Query extraction
    - Caching for repeated queries
    - Graceful fallbacks on failures
    - Configurable limits
    - Pipeline stage tracking
    - Edge case handling
    """
    
    def __init__(
        self,
        search_provider: Optional[TavilyProvider] = None,
        llm_provider=None,
        config: Optional[ResearchConfig] = None
    ):
        self.search = search_provider or TavilyProvider()
        if llm_provider is not None:
            self.llm = llm_provider
        else:
            from factory import get_llm as _get_llm  # factory-owned router (T305)

            self.llm = _get_llm()
        self.config = config or ResearchConfig()
        
        # In-memory cache for research results
        self._cache: dict[str, ResearchContext] = {}
        
        # Pipeline stage tracking
        self._stage_timings: dict[str, float] = {}
    
    def _compute_query_hash(self, query: str) -> str:
        """Compute hash for caching research results"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    def _extract_query(
        self,
        user_query: Optional[str],
        document_topics: Optional[List[str]] = None
    ) -> str:
        """
        Extract a clean research query.
        
        Can come from user input or be derived from document topics.
        """
        if user_query and user_query.strip():
            # Use and truncate user query
            query = user_query.strip()
            if len(query) > self.config.query_max_length:
                query = query[:self.config.query_max_length]
            return query
        
        if document_topics:
            # Derive query from document topics
            return " ".join(document_topics[:3])  # Use top 3 topics
        
        return "overview"  # Default fallback
    
    async def _search(
        self,
        query: str,
        max_results: int
    ) -> List[ResearchSource]:
        """Perform web search with Tavily"""
        try:
            response = await self.search.search(query, max_results)
            
            sources = []
            for result in response.results:
                if result.relevance_score >= self.config.min_relevance_score:
                    sources.append(ResearchSource(
                        url=result.url,
                        title=result.title,
                        content=result.content or "",
                        relevance_score=result.relevance_score
                    ))
            
            return sources
            
        except Exception:
            return []
    
    async def _extract_content(
        self,
        source: ResearchSource
    ) -> ResearchSource:
        """Extract content from URL using Jina Reader"""
        try:
            extraction = await self.search.extract_content(source.url)
            
            if extraction.content and extraction.content != f"Error extracting content: ":
                source.content = extraction.content[:self.config.max_content_length]
                if extraction.title:
                    source.title = extraction.title
                    
        except Exception:
            pass  # Keep original content if extraction fails
        
        return source
    
    async def _summarize(
        self,
        sources: List[ResearchSource]
    ) -> tuple[str, List[str]]:
        """
        Summarize research sources into concise context.
        
        First tries LLM-based summarization, falls back to heuristic if LLM fails.
        Returns (summary, key_findings)
        """
        if not sources:
            return "", []
        
        # Try LLM-based summarization for better quality
        try:
            return await self._summarize_with_llm(sources)
        except Exception:
            # Fallback to heuristic summarization
            return self._summarize_heuristic(sources)
    
    async def _summarize_with_llm(
        self,
        sources: List[ResearchSource]
    ) -> tuple[str, List[str]]:
        """
        Use LLM to generate high-quality summary and key findings.
        """
        # Prepare source content for LLM
        source_texts = []
        for s in sources[:3]:  # Limit to top 3 sources
            source_texts.append(f"Source: {s.title}\nURL: {s.url}\nContent: {s.content[:1000]}")
        
        combined_text = "\n\n---\n\n".join(source_texts)
        
        # Create summarization prompt
        prompt = f"""You are a research assistant. Summarize the following web sources into a concise research context.

Sources:
{combined_text}

Generate a JSON response with this structure:
{{
  "summary": "2-3 paragraph summary of key information",
  "key_findings": ["finding 1", "finding 2", "finding 3", "finding 4"]
}}

Focus on: factual information, key concepts, and useful details for educational purposes."""
        
        try:
            from services.strategies.llm.structured import generate_structured

            from pydantic import BaseModel, Field

            class _ResearchSummary(BaseModel):
                summary: str = ""
                key_findings: List[str] = Field(default_factory=list)

            # T306: route JSON through schema-validated generate_structured
            # (max 2 repairs, typed failure) instead of bare parsing.
            parsed = await generate_structured(self.llm, _ResearchSummary, prompt)
            return (
                parsed.summary or "",
                list(parsed.key_findings or []),
            )
        except Exception:
            pass
        
        # Fallback to heuristic if LLM fails
        return self._summarize_heuristic(sources)
    
    def _summarize_heuristic(
        self,
        sources: List[ResearchSource]
    ) -> tuple[str, List[str]]:
        """
        Fallback heuristic-based summarization.
        """
        # Combine unique content
        content_parts = []
        seen_content = set()
        
        for source in sources:
            # Take first 500 chars as preview
            preview = source.content[:500] if source.content else ""
            if preview and preview not in seen_content:
                seen_content.add(preview)
                content_parts.append(f"- {source.title}: {preview}")
        
        summary = "\n".join(content_parts[:5])  # Max 5 sources in summary
        
        # Extract key findings (simple heuristic)
        key_findings = []
        for source in sources[:3]:
            if source.content:
                # Take first sentence as a key finding
                sentences = source.content.split('.')
                if sentences and sentences[0]:
                    key_findings.append(sentences[0].strip()[:200])
        
        return summary, key_findings[:5]  # Max 5 key findings
    
    async def _perform_research(
        self,
        query: str,
        max_sources: int
    ) -> Optional[ResearchContext]:
        """
        Perform full research pipeline with stage tracking.
        
        Pipeline stages:
        1. Query extraction (handled in augment method)
        2. Web search with Tavily
        3. Content extraction with Jina
        4. Summarization with LLM
        
        Returns None on failure to allow graceful fallback.
        """
        # Track stage timings
        stage_timings = {}
        start_time = time.time()
        
        # Stage 1: Search
        stage_start = time.time()
        sources = await self._search(query, max_sources)
        stage_timings["search"] = time.time() - stage_start
        
        if not sources:
            return None
        
        # Stage 2: Content extraction
        stage_start = time.time()
        extracted_sources = []
        for source in sources[:max_sources]:
            extracted = await self._extract_content(source)
            if extracted.content:  # Only keep sources with content
                extracted_sources.append(extracted)
        stage_timings["content_extraction"] = time.time() - stage_start
        
        if not extracted_sources:
            return None
        
        # Stage 3: Summarization
        stage_start = time.time()
        summary, key_findings = await self._summarize(extracted_sources)
        stage_timings["summarization"] = time.time() - stage_start
        
        processing_time = time.time() - start_time
        stage_timings["total"] = processing_time
        
        # Store stage timings for debugging/monitoring
        self._stage_timings = stage_timings
        
        return ResearchContext(
            original_query=query,
            sources=extracted_sources,
            summary=summary,
            key_findings=key_findings,
            processing_time=processing_time,
            cached=False
        )
    
    def _merge_contexts(
        self,
        document_context: str,
        research_context: Optional[ResearchContext]
    ) -> MergedContext:
        """
        Merge document context with research context.
        
        Document content ALWAYS has priority.
        """
        if not research_context:
            return MergedContext(
                document_context=document_context,
                research_context=None,
                has_research=False
            )
        
        # Format research context
        research_formatted = f"""
Additional Information from Web Research:
{research_context.summary}

Key Findings:
{chr(10).join(f"- {f}" for f in research_context.key_findings)}

Sources: {', '.join(s.title for s in research_context.sources[:3])}
"""
        
        return MergedContext(
            document_context=document_context,
            research_context=research_formatted,
            has_research=True,
            priority_note="IMPORTANT: Document content has priority over web research. Use web content only to supplement or verify document content."
        )
    
    async def augment(
        self,
        user_query: Optional[str],
        document_context: str,
        document_topics: Optional[List[str]] = None
    ) -> MergedContext:
        """
        Augment document context with research.
        
        This is the main entry point for research augmentation.
        """
        # Check if research is enabled
        if not self.config.enabled:
            return MergedContext(
                document_context=document_context,
                research_context=None,
                has_research=False
            )
        
        # Step 1: Extract research query
        research_query = self._extract_query(user_query, document_topics)
        
        # Step 2: Check cache
        query_hash = self._compute_query_hash(research_query)
        
        if self.config.cache_enabled and query_hash in self._cache:
            cached_context = self._cache[query_hash]
            cached_context.cached = True
            return self._merge_contexts(document_context, cached_context)
        
        # Step 3: Perform research
        research_context = await self._perform_research(
            research_query,
            self.config.max_sources
        )
        
        # Step 4: Cache result
        if research_context and self.config.cache_enabled:
            self._cache[query_hash] = research_context
        
        # Step 5: Merge contexts
        return self._merge_contexts(document_context, research_context)
    
    async def research_only(
        self,
        query: str,
        max_sources: Optional[int] = None
    ) -> Optional[ResearchContext]:
        """
        Perform research only (without document context).
        
        Useful for standalone research queries.
        """
        if not self.config.enabled:
            return None
        
        # Check cache
        query_hash = self._compute_query_hash(query)
        
        if self.config.cache_enabled and query_hash in self._cache:
            cached = self._cache[query_hash]
            cached.cached = True
            return cached
        
        # Perform research
        max_sources = max_sources or self.config.max_sources
        research_context = await self._perform_research(query, max_sources)
        
        # Cache result
        if research_context and self.config.cache_enabled:
            self._cache[query_hash] = research_context
        
        return research_context
    
    def clear_cache(self):
        """Clear research cache"""
        self._cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache and performance statistics"""
        return {
            "cached_queries": len(self._cache),
            "cache_enabled": self.config.cache_enabled,
            "stage_timings": self._stage_timings
        }
    
    def get_stage_timings(self) -> dict:
        """Get pipeline stage timing information"""
        return self._stage_timings.copy()


# Default service instance
research_service = ResearchAugmentationService()