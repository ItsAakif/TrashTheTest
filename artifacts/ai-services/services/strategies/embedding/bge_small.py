"""
BGE-small Embedding Provider

Local embedding provider using BGE-small (or E5-small) for cost-efficient
vector embeddings without API costs.
"""

import time
from typing import Optional
import numpy as np

from ..base import EmbeddingStrategy, EmbeddingResult
from config.settings import get_embedding_config


class BGESmallProvider(EmbeddingStrategy):
    """
    BGE-small implementation for local embeddings.
    
    Uses sentence-transformers to run BGE-small locally.
    No API costs, fast inference, 384 dimensions.
    """
    
    def __init__(self):
        config = get_embedding_config()
        self.model_name = config.model
        self.dimensions = config.dimensions
        self.device = config.device
        self.normalize = config.normalize_embeddings
        self._model = None
        self._tokenizer = None
    
    async def _ensure_model_loaded(self):
        """Load the model if not already loaded"""
        if self._model is None:
            # Import here to avoid loading at import time
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self.model_name, device=self.device)
    
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
            EmbeddingResult with embeddings
        """
        await self._ensure_model_loaded()
        
        start_time = time.time()
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self._model.encode(
                batch,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            
            all_embeddings.extend(embeddings.tolist())
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model_name,
            dimensions=self.dimensions,
            processing_time=processing_time,
        )
    
    async def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            List of floats representing the embedding
        """
        result = await self.generate_embeddings([text])
        return result.embeddings[0]
    
    @property
    def dimensions(self) -> int:
        """Return the dimensionality of embeddings"""
        return self._dimensions
    
    @dimensions.setter
    def dimensions(self, value: int):
        self._dimensions = value
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value
    
    def unload(self):
        """Unload model from memory"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None


class E5SmallProvider(EmbeddingStrategy):
    """
    E5-small alternative embedding provider.
    
    Another local model option with 384 dimensions.
    """
    
    def __init__(self):
        config = get_embedding_config()
        self.model_name = "intfloat/e5-small-v2"
        self.dimensions = 384
        self.device = config.device
        self.normalize = config.normalize_embeddings
        self._model = None
    
    async def _ensure_model_loaded(self):
        """Load the model if not already loaded"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self.model_name, device=self.device)
    
    async def generate_embeddings(
        self, 
        texts: list[str], 
        batch_size: int = 32
    ) -> EmbeddingResult:
        """Generate embeddings using E5-small"""
        await self._ensure_model_loaded()
        
        start_time = time.time()
        
        # E5 requires "passage: " prefix for passage embedding
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        all_embeddings = []
        
        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i + batch_size]
            
            embeddings = self._model.encode(
                batch,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            
            all_embeddings.extend(embeddings.tolist())
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model_name,
            dimensions=self.dimensions,
            processing_time=processing_time,
        )
    
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text"""
        result = await self.generate_embeddings([text])
        return result.embeddings[0]
    
    @property
    def dimensions(self) -> int:
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        return self._model_name


# Factory functions
def create_embedding_provider(provider_name: Optional[str] = None) -> EmbeddingStrategy:
    """
    Create an embedding provider based on configuration
    
    Args:
        provider_name: Optional override for provider ("bge-small" or "e5-small")
        
    Returns:
        EmbeddingStrategy implementation
    """
    config = get_embedding_config()
    provider = provider_name or config.provider
    
    if provider == "e5-small":
        return E5SmallProvider()
    else:
        # Default to BGE-small
        return BGESmallProvider()


# Convenience function matching other modules
def create_bge_provider() -> BGESmallProvider:
    """Create and return a BGE-small provider instance"""
    return BGESmallProvider()


__all__ = [
    "BGESmallProvider",
    "E5SmallProvider",
    "create_embedding_provider",
    "create_bge_provider",
]