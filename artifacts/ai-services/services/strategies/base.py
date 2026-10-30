"""
Base Protocol Interfaces for Model Strategy Layer

This module defines abstract base classes (Protocols) for each AI/ML component:
- OCRStrategy: Text extraction from files
- EmbeddingStrategy: Generate vector embeddings
- LLMStrategy: Text generation with language models
- TTSStrategy: Text-to-speech synthesis
- SearchStrategy: Web search and content extraction
"""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from typing import Any


# ============================================================
# OCR Protocols and Data Classes
# ============================================================

@dataclass
class OCRResult:
    """Result from OCR processing"""
    text: str
    confidence: float
    page_number: int | None = None
    method: str = "ppocr"
    metadata: dict | None = None


@dataclass
class ExtractionResult:
    """Complete extraction result from a document"""
    pages: list[OCRResult]
    total_pages: int
    method: str
    processing_time: float
    
    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


@runtime_checkable
class OCRStrategy(Protocol):
    """Abstract interface for OCR providers"""
    
    async def extract_text(
        self, 
        file_path: str, 
        pages: list[int] | None = None
    ) -> ExtractionResult:
        """
        Extract text from a file (PDF, image, etc.)
        
        Args:
            file_path: Path to the file
            pages: Optional list of page numbers to process
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        ...
    
    async def extract_page(self, file_path: str, page_number: int) -> OCRResult:
        """
        Extract text from a single page
        
        Args:
            file_path: Path to the file
            page_number: Page number to extract
            
        Returns:
            OCRResult for the single page
        """
        ...


# ============================================================
# Embedding Protocols and Data Classes
# ============================================================

@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embeddings: list[list[float]]
    model: str
    dimensions: int
    processing_time: float


@runtime_checkable
class EmbeddingStrategy(Protocol):
    """Abstract interface for embedding providers"""
    
    async def generate_embeddings(
        self, 
        texts: list[str], 
        batch_size: int = 32
    ) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        ...
    
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            List of floats representing the embedding
        """
        ...
    
    @property
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings"""
        ...
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        ...


# ============================================================
# LLM Protocols and Data Classes
# ============================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    response_format: dict | None = None  # For structured outputs


@dataclass
class GenerationResult:
    """Result from text generation"""
    text: str
    model: str
    usage: dict  # tokens used
    finish_reason: str
    processing_time: float


@dataclass
class ChatMessage:
    """A message in a chat conversation"""
    role: str  # "system", "user", "assistant"
    content: str


@runtime_checkable
class LLMStrategy(Protocol):
    """Abstract interface for LLM providers"""
    
    async def generate(
        self, 
        prompt: str, 
        config: GenerationConfig | None = None
    ) -> GenerationResult:
        """
        Generate text from a single prompt
        
        Args:
            prompt: The prompt text
            config: Optional generation configuration
            
        Returns:
            GenerationResult with generated text
        """
        ...
    
    async def chat(
        self, 
        messages: list[ChatMessage], 
        config: GenerationConfig | None = None
    ) -> GenerationResult:
        """
        Generate response in a chat conversation
        
        Args:
            messages: List of chat messages
            config: Optional generation configuration
            
        Returns:
            GenerationResult with assistant's response
        """
        ...
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        ...


# ============================================================
# TTS Protocols and Data Classes
# ============================================================

@dataclass
class TTSResult:
    """Result from TTS synthesis"""
    audio_data: bytes | None = None
    audio_path: str | None = None
    format: str = "wav"
    sample_rate: int = 24000
    duration_ms: int = 0
    voice: str = ""
    processing_time: float = 0.0


@runtime_checkable
class TTSStrategy(Protocol):
    """Abstract interface for TTS providers"""
    
    async def synthesize(
        self, 
        text: str, 
        voice: str | None = None
    ) -> TTSResult:
        """
        Synthesize text to speech
        
        Args:
            text: Text to convert to speech
            voice: Optional voice identifier
            
        Returns:
            TTSResult with audio data or path
        """
        ...
    
    async def synthesize_stream(
        self, 
        text: str, 
        voice: str | None = None
    ) -> bytes:
        """
        Synthesize text and return streaming audio bytes
        
        Args:
            text: Text to convert to speech
            voice: Optional voice identifier
            
        Returns:
            Audio bytes for streaming
        """
        ...
    
    @property
    def available_voices(self) -> list[str]:
        """Return list of available voice identifiers"""
        ...
    
    @property
    def default_voice(self) -> str:
        """Return the default voice identifier"""
        ...


# ============================================================
# Search Protocols and Data Classes
# ============================================================

@dataclass
class SearchResult:
    """A single search result"""
    title: str
    url: str
    content: str | None = None
    relevance_score: float = 0.0


@dataclass
class SearchResponse:
    """Response from a web search"""
    results: list[SearchResult]
    query: str
    total_results: int
    processing_time: float


@dataclass
class ContentExtractionResult:
    """Result from content extraction"""
    url: str
    title: str | None
    content: str
    summary: str | None = None
    raw_html: str | None = None
    processing_time: float = 0.0


@runtime_checkable
class SearchStrategy(Protocol):
    """Abstract interface for search providers"""
    
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
        ...
    
    async def extract_content(self, url: str) -> ContentExtractionResult:
        """
        Extract readable content from a URL
        
        Args:
            url: URL to extract content from
            
        Returns:
            ContentExtractionResult with extracted content
        """
        ...
    
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
        ...


# ============================================================
# Factory Type for Strategy Registry
# ============================================================

class StrategyType:
    """Enum-like class for strategy types"""
    OCR = "ocr"
    EMBEDDING = "embedding"
    LLM = "llm"
    TTS = "tts"
    SEARCH = "search"


# Export all for convenience
__all__ = [
    # Data classes
    "OCRResult",
    "ExtractionResult",
    "EmbeddingResult",
    "GenerationConfig",
    "GenerationResult",
    "ChatMessage",
    "TTSResult",
    "SearchResult",
    "SearchResponse",
    "ContentExtractionResult",
    # Protocols
    "OCRStrategy",
    "EmbeddingStrategy",
    "LLMStrategy",
    "TTSStrategy",
    "SearchStrategy",
    # Utilities
    "StrategyType",
]