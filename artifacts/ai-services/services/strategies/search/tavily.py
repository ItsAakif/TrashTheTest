"""
Tavily + Jina Search Provider

Web search and content extraction for research augmentation.
Uses Tavily for search and Jina Reader for content extraction.
"""

import time
from typing import Optional

import aiohttp

from ..base import SearchStrategy, SearchResult, SearchResponse, ContentExtractionResult
from config.settings import get_search_config


class TavilyProvider(SearchStrategy):
    """
    Tavily search implementation for web search.
    
    Features:
    - Structured web search
    - Content extraction with Jina Reader
    - Relevance scoring
    - Toggle for research augmentation
    """
    
    def __init__(self):
        config = get_search_config()
        self.api_key = config.api_key
        self.max_results = config.max_results
        self.jina_enabled = config.jina_enabled
    
    async def search(
        self, 
        query: str, 
        num_results: int = 5
    ) -> SearchResponse:
        """
        Perform a web search
        
        Args:
            query: Search query string
            num_results: Number of results to return
            
        Returns:
            SearchResponse with results
        """
        start_time = time.time()
        
        if not self.api_key:
            # Return empty results if no API key
            return SearchResponse(
                results=[],
                query=query,
                total_results=0,
                processing_time=time.time() - start_time,
            )
        
        try:
            # Call Tavily API
            results = await self._search_tavily(query, num_results)
            
            processing_time = time.time() - start_time
            
            return SearchResponse(
                results=results,
                query=query,
                total_results=len(results),
                processing_time=processing_time,
            )
        except Exception as e:
            # Return empty on error
            return SearchResponse(
                results=[],
                query=query,
                total_results=0,
                processing_time=time.time() - start_time,
            )
    
    async def _search_tavily(
        self, 
        query: str, 
        num_results: int
    ) -> list[SearchResult]:
        """Call Tavily API"""
        session = aiohttp.ClientSession()
        
        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": num_results,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    results = []
                    for item in data.get("results", []):
                        results.append(SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            content=item.get("content", ""),
                            relevance_score=item.get("score", 0.0),
                        ))
                    
                    return results
                else:
                    return []
        except Exception:
            return []
        finally:
            await session.close()
    
    async def extract_content(self, url: str) -> ContentExtractionResult:
        """
        Extract readable content from a URL using Jina Reader
        
        Args:
            url: URL to extract content from
            
        Returns:
            ContentExtractionResult with extracted content
        """
        start_time = time.time()
        
        try:
            # Use Jina Reader API
            content, title, summary = await self._extract_with_jina(url)
            
            processing_time = time.time() - start_time
            
            return ContentExtractionResult(
                url=url,
                title=title,
                content=content,
                summary=summary,
                processing_time=processing_time,
            )
        except Exception as e:
            processing_time = time.time() - start_time
            
            return ContentExtractionResult(
                url=url,
                title=None,
                content=f"Error extracting content: {str(e)}",
                summary=None,
                processing_time=processing_time,
            )
    
    async def _extract_with_jina(
        self, 
        url: str
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Extract content using Jina Reader"""
        session = aiohttp.ClientSession()
        
        # Jina Reader endpoint
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            async with session.get(jina_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    content = data.get("content", "")
                    title = data.get("title")
                    summary = data.get("summary")
                    
                    return content, title, summary
                else:
                    raise Exception(f"Jina extraction failed: {resp.status}")
        finally:
            await session.close()
    
    async def search_and_extract(
        self, 
        query: str, 
        num_results: int = 3
    ) -> list[ContentExtractionResult]:
        """
        Perform search and extract content from top results
        
        Args:
            query: Search query
            num_results: Number of results to process
            
        Returns:
            List of ContentExtractionResult
        """
        # First search
        search_response = await self.search(query, num_results)
        
        # Then extract content from each result
        results = []
        for search_result in search_response.results[:num_results]:
            extraction = await self.extract_content(search_result.url)
            results.append(extraction)
        
        return results


class SerperProvider(SearchStrategy):
    """
    Serper search as alternative to Tavily.
    
    Placeholder for future implementation.
    """
    
    def __init__(self):
        config = get_search_config()
        self.api_key = config.api_key or os.getenv("SERPER_API_KEY")
    
    async def search(
        self, 
        query: str, 
        num_results: int = 5
    ) -> SearchResponse:
        """Search using Serper"""
        # Placeholder
        raise NotImplementedError("Serper provider not yet implemented")
    
    async def extract_content(self, url: str) -> ContentExtractionResult:
        """Extract content from URL"""
        raise NotImplementedError("Serper provider not yet implemented")
    
    async def search_and_extract(
        self, 
        query: str, 
        num_results: int = 3
    ) -> list[ContentExtractionResult]:
        """Search and extract"""
        raise NotImplementedError("Serper provider not yet implemented")


# Factory function
def create_search_provider(provider_name: Optional[str] = None) -> SearchStrategy:
    """
    Create a search provider based on configuration
    
    Args:
        provider_name: Optional override ("tavily" or "serper")
        
    Returns:
        SearchStrategy implementation
    """
    config = get_search_config()
    provider = provider_name or config.provider
    
    if provider == "serper":
        return SerperProvider()
    else:
        # Default to Tavily
        return TavilyProvider()


# Convenience function
def create_tavily_provider() -> TavilyProvider:
    """Create and return a Tavily provider instance"""
    return TavilyProvider()


__all__ = [
    "TavilyProvider",
    "SerperProvider",
    "create_search_provider",
    "create_tavily_provider",
]