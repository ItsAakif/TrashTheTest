"""
Strategy Factory / Registry System

Central factory that returns the correct strategy implementation
based on configuration. Used by all future modules.
"""

from typing import Optional

from services.strategies.base import (
    OCRStrategy,
    EmbeddingStrategy,
    LLMStrategy,
    TTSStrategy,
    SearchStrategy,
    StrategyType,
)
from services.strategies.ocr.ppocr_onnx import create_ocr_provider as create_ppocr_provider
from services.strategies.embedding.qwen3_embedding import create_embedding_provider
from services.strategies.embedding.bge_small import BGESmallProvider as _BGELegacy
from services.strategies.llm.groq_rotating import create_llm_provider
from services.strategies.tts.kokoro import create_tts_provider
from services.strategies.search.tavily import create_search_provider
from config.settings import get_settings


class StrategyFactory:
    """
    Factory for creating strategy instances based on configuration.
    
    This provides a central registry for all AI/ML components,
    allowing easy switching between providers without modifying
    business logic.
    """
    
    def __init__(self):
        self._ocr_provider: Optional[OCRStrategy] = None
        self._embedding_provider: Optional[EmbeddingStrategy] = None
        self._llm_provider: Optional[LLMStrategy] = None
        self._tts_provider: Optional[TTSStrategy] = None
        self._search_provider: Optional[SearchStrategy] = None
    
    def get_ocr(self) -> OCRStrategy:
        """
        Get OCR strategy instance.

        The factory honors OCR_PROVIDER: PP-OCRv6 CPU ONNX is the single
        local path (ADR-005 as amended). Any other value falls back to
        the PP-OCR provider (never a cloud OCR).

        Returns:
            OCRStrategy implementation based on configuration
        """
        if self._ocr_provider is None:
            self._ocr_provider = create_ppocr_provider()
        return self._ocr_provider
    
    def get_embedding(self) -> EmbeddingStrategy:
        """
        Get embedding strategy instance
        
        Returns:
            EmbeddingStrategy implementation based on configuration
        """
        if self._embedding_provider is None:
            self._embedding_provider = create_embedding_provider()
        return self._embedding_provider
    
    def get_llm(self) -> LLMStrategy:
        """
        Get LLM router instance (primary Groq + OpenRouter fallback).

        The factory is the only LLM construction site (T305): service
        code must call factory.get_llm(), never GroqProvider() directly,
        so key rotation (groq_rotating) and fallback are always in effect.
        """
        if self._llm_provider is None:
            from services.strategies.llm.router import get_llm_router

            self._llm_provider = get_llm_router()
        return self._llm_provider
    
    def get_tts(self) -> TTSStrategy:
        """
        Get TTS strategy instance
        
        Returns:
            TTSStrategy implementation based on configuration
        """
        if self._tts_provider is None:
            self._tts_provider = create_tts_provider()
        return self._tts_provider
    
    def get_search(self) -> SearchStrategy:
        """
        Get search strategy instance
        
        Returns:
            SearchStrategy implementation based on configuration
        """
        if self._search_provider is None:
            self._search_provider = create_search_provider()
        return self._search_provider
    
    def get_strategy(self, strategy_type: str) -> any:
        """
        Get strategy by type name
        
        Args:
            strategy_type: One of "ocr", "embedding", "llm", "tts", "search"
            
        Returns:
            Strategy implementation
            
        Raises:
            ValueError: If strategy_type is unknown
        """
        mapping = {
            StrategyType.OCR: self.get_ocr,
            StrategyType.EMBEDDING: self.get_embedding,
            StrategyType.LLM: self.get_llm,
            StrategyType.TTS: self.get_tts,
            StrategyType.SEARCH: self.get_search,
        }
        
        getter = mapping.get(strategy_type)
        if getter is None:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return getter()
    
    def reset(self):
        """Reset all cached providers"""
        self._ocr_provider = None
        self._embedding_provider = None
        self._llm_provider = None
        self._tts_provider = None
        self._search_provider = None
    
    def reload_config(self):
        """Reload configuration and reset providers"""
        from config.settings import settings as _settings

        _settings.reload()
        self.reset()


# Global factory instance
_strategy_factory: Optional[StrategyFactory] = None


def get_strategy_factory() -> StrategyFactory:
    """
    Get the global strategy factory instance
    
    Returns:
        StrategyFactory singleton
    """
    global _strategy_factory
    if _strategy_factory is None:
        _strategy_factory = StrategyFactory()
    return _strategy_factory


def get_ocr() -> OCRStrategy:
    """Convenience function to get OCR strategy"""
    return get_strategy_factory().get_ocr()


def get_embedding() -> EmbeddingStrategy:
    """Convenience function to get embedding strategy"""
    return get_strategy_factory().get_embedding()


def get_llm() -> LLMStrategy:
    """Convenience function to get LLM strategy"""
    return get_strategy_factory().get_llm()


def get_tts() -> TTSStrategy:
    """Convenience function to get TTS strategy"""
    return get_strategy_factory().get_tts()


def get_search() -> SearchStrategy:
    """Convenience function to get search strategy"""
    return get_strategy_factory().get_search()


# Export all
__all__ = [
    "StrategyFactory",
    "get_strategy_factory",
    "get_ocr",
    "get_embedding",
    "get_llm",
    "get_tts",
    "get_search",
]