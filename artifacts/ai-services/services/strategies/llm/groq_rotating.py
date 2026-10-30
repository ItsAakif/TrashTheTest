"""
Groq LLM Provider with API Key Rotation

Primary LLM provider using Groq API with automatic key rotation
to handle rate limits across multiple API keys.
"""

import os
import time
import random
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from groq import AsyncGroq, RateLimitError

from ..base import LLMStrategy, GenerationConfig, GenerationResult, ChatMessage


def _configured_primary_model() -> str:
    """Model id from config only (fail-soft: settings fail-fast governs boot)."""
    try:
        from config.settings import get_llm_config

        return get_llm_config().primary_model
    except Exception:
        return "qwen/qwen3-32b"


@dataclass
class KeyStatus:
    """Track status of an API key"""
    key: str
    last_used: datetime
    error_count: int = 0
    is_rate_limited: bool = False
    rate_limit_reset: Optional[datetime] = None


class GroqRotatingProvider(LLMStrategy):
    """
    Groq API implementation with automatic key rotation.
    
    Features:
    - Multiple API key support for rate limit handling
    - Automatic failover to next key on rate limit
    - Tracks key health and rotates accordingly
    - Configurable temperature, max_tokens
    - Structured outputs (JSON)
    """
    
    def __init__(self, default_model: Optional[str] = None):
        self.api_keys: List[KeyStatus] = []
        self.current_key_index = 0
        # Model id comes only from config (config/settings.py), never a literal.
        self.default_model = default_model or _configured_primary_model()
        self.default_temperature = 0.7
        self.default_max_tokens = 4096
        self._clients: dict = {}

        # Load API keys from environment
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load all available API keys from environment"""
        # Primary key
        primary_key = os.getenv("GROQ_API_KEY")
        if primary_key:
            self.api_keys.append(KeyStatus(
                key=primary_key,
                last_used=datetime.min
            ))
        
        # Additional keys for rotation (GROQ_API_KEY_1, GROQ_API_KEY_2, etc.)
        for i in range(1, 10):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key:
                self.api_keys.append(KeyStatus(
                    key=key,
                    last_used=datetime.min
                ))
        
        if not self.api_keys:
            raise ValueError("At least one GROQ_API_KEY is required")
        
        # Shuffle keys for random initial selection
        random.shuffle(self.api_keys)
    
    def _get_available_key(self) -> KeyStatus:
        """Get the next available (non-rate-limited) API key"""
        now = datetime.now()
        
        # First, try to find a key that's not rate limited
        for i in range(len(self.api_keys)):
            idx = (self.current_key_index + i) % len(self.api_keys)
            key_status = self.api_keys[idx]
            
            # Check if rate limit has expired
            if key_status.is_rate_limited and key_status.rate_limit_reset:
                if now >= key_status.rate_limit_reset:
                    key_status.is_rate_limited = False
                    key_status.rate_limit_reset = None
            
            if not key_status.is_rate_limited:
                self.current_key_index = idx
                return key_status
        
        # All keys are rate limited, wait for the first one to reset
        soonest_reset = min(
            (ks.rate_limit_reset for ks in self.api_keys if ks.rate_limit_reset),
            default=now + timedelta(seconds=60)
        )
        
        wait_seconds = max(0, (soonest_reset - now).total_seconds())
        if wait_seconds > 0:
            raise RuntimeError(f"All API keys rate limited. Wait {wait_seconds:.0f}s before retrying.")
        
        # Fallback to first key
        return self.api_keys[0]
    
    def _get_client(self, key: str) -> AsyncGroq:
        """Get or create Groq client for a specific key"""
        if key not in self._clients:
            self._clients[key] = AsyncGroq(api_key=key)
        return self._clients[key]
    
    def _mark_key_rate_limited(self, key_status: KeyStatus, retry_after: Optional[int] = None):
        """Mark a key as rate limited"""
        key_status.is_rate_limited = True
        key_status.error_count += 1
        
        # Set reset time (default 60s if not specified)
        wait_seconds = retry_after or 60
        key_status.rate_limit_reset = datetime.now() + timedelta(seconds=wait_seconds)
    
    async def _try_generate(
        self,
        messages: List[dict],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Try to generate with current key, handle rate limits"""
        gen_config = config or GenerationConfig()
        key_status = self._get_available_key()
        client = self._get_client(key_status.key)
        
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(
                model=gen_config.model or self.default_model,
                messages=messages,
                temperature=gen_config.temperature or self.default_temperature,
                max_tokens=gen_config.max_tokens or self.default_max_tokens,
                top_p=gen_config.top_p or 1.0,
                frequency_penalty=gen_config.frequency_penalty or 0,
                presence_penalty=gen_config.presence_penalty or 0,
                stop=gen_config.stop,
                response_format=gen_config.response_format,
            )
            
            key_status.last_used = datetime.now()
            processing_time = time.time() - start_time
            
            return GenerationResult(
                text=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason or "unknown",
                processing_time=processing_time,
            )
            
        except RateLimitError as e:
            # Extract retry-after if available
            retry_after = None
            if hasattr(e, 'response') and hasattr(e.response, 'headers'):
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    retry_after = int(retry_after)
            
            self._mark_key_rate_limited(key_status, retry_after)
            
            # Try with next key
            if len(self.api_keys) > 1:
                return await self._try_generate(messages, config)
            else:
                raise RuntimeError(f"Rate limited and no fallback keys available. Wait before retrying.")
                
        except Exception as e:
            key_status.error_count += 1
            # DNS/transport failure: provider-level, not key-level. Do NOT
            # rotate the key (rotation is reserved for 401/403/429) and keep
            # the original type so the router can classify it TRANSIENT and
            # retry this same provider with backoff.
            from .router import is_transient_connection_error

            if is_transient_connection_error(e):
                raise
            raise RuntimeError(f"Groq generation failed: {str(e)}")
    
    async def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate text from a single prompt
        
        Args:
            prompt: The prompt text
            config: Optional generation configuration
            
        Returns:
            GenerationResult with generated text
        """
        messages = [{"role": "user", "content": prompt}]
        return await self._try_generate(messages, config)
    
    async def chat(
        self, 
        messages: List[ChatMessage], 
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate response in a chat conversation
        
        Args:
            messages: List of chat messages
            config: Optional generation configuration
            
        Returns:
            GenerationResult with assistant's response
        """
        # Convert messages to Groq format
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        return await self._try_generate(groq_messages, config)
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self.default_model
    
    def get_key_status(self) -> List[dict]:
        """Get status of all API keys (for monitoring)"""
        now = datetime.now()
        return [
            {
                "index": i,
                "last_used": ks.last_used.isoformat() if ks.last_used != datetime.min else None,
                "error_count": ks.error_count,
                "is_rate_limited": ks.is_rate_limited,
                "rate_limit_reset": ks.rate_limit_reset.isoformat() if ks.rate_limit_reset else None,
                "key_preview": ks.key[:8] + "..." + ks.key[-4:],
            }
            for i, ks in enumerate(self.api_keys)
        ]


# Factory function
def create_llm_provider(provider_name: Optional[str] = None) -> LLMStrategy:
    """
    Create an LLM provider based on configuration
    
    Args:
        provider_name: Optional override
        
    Returns:
        LLMStrategy implementation
    """
    return GroqRotatingProvider()


def create_groq_provider() -> GroqRotatingProvider:
    """Create and return a Groq provider instance with key rotation"""
    return GroqRotatingProvider()


__all__ = [
    "GroqRotatingProvider",
    "create_llm_provider",
    "create_groq_provider",
]
