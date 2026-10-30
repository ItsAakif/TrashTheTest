"""
Groq LLM Provider

Primary LLM provider using Groq API for fast inference.
Also supports OpenRouter as alternative.
"""

import os
import time
from typing import Optional

from groq import AsyncGroq

from ..base import LLMStrategy, GenerationConfig, GenerationResult, ChatMessage
from config.settings import get_llm_config


class GroqProvider(LLMStrategy):
    """
    Groq API implementation for text generation.
    
    Supports:
    - Single prompt generation
    - Chat completion
    - Configurable temperature, max_tokens
    - Structured outputs (JSON)
    - Future integration point for local 7B model
    """
    
    def __init__(self):
        config = get_llm_config()
        self.api_key = config.api_key
        self.default_model = config.model
        self.default_temperature = config.default_temperature
        self.default_max_tokens = config.default_max_tokens
        self._client: AsyncGroq | None = None
    
    async def _get_client(self) -> AsyncGroq:
        """Get or create Groq client"""
        if self._client is None:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is required")
            self._client = AsyncGroq(api_key=self.api_key)
        return self._client
    
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
        start_time = time.time()
        
        # Set up generation parameters
        gen_config = config or GenerationConfig()
        
        client = await self._get_client()
        
        # Build request
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=gen_config.temperature,
                max_tokens=gen_config.max_tokens,
                top_p=gen_config.top_p,
                frequency_penalty=gen_config.frequency_penalty,
                presence_penalty=gen_config.presence_penalty,
                stop=gen_config.stop,
                response_format=gen_config.response_format,
            )
            
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
        except Exception as e:
            # Preserve DNS/transport errors: the router classifies them as
            # TRANSIENT and retries this provider instead of failing the job.
            from .router import is_transient_connection_error

            if is_transient_connection_error(e):
                raise
            raise RuntimeError(f"Groq generation failed: {str(e)}")
    
    async def chat(
        self, 
        messages: list[ChatMessage], 
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
        start_time = time.time()
        
        gen_config = config or GenerationConfig()
        
        client = await self._get_client()
        
        # Convert messages to Groq format
        groq_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        try:
            response = await client.chat.completions.create(
                model=self.default_model,
                messages=groq_messages,
                temperature=gen_config.temperature,
                max_tokens=gen_config.max_tokens,
                top_p=gen_config.top_p,
                frequency_penalty=gen_config.frequency_penalty,
                presence_penalty=gen_config.presence_penalty,
                stop=gen_config.stop,
                response_format=gen_config.response_format,
            )
            
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
        except Exception as e:
            # Same transport rule as generate(): keep DNS/connection errors
            # typed so the router retries instead of treating them as model
            # failures.
            from .router import is_transient_connection_error

            if is_transient_connection_error(e):
                raise
            raise RuntimeError(f"Groq chat failed: {str(e)}")
    
    @property
    def model_name(self) -> str:
        """Return the model name"""
        return self.default_model


class OpenRouterProvider(LLMStrategy):
    """
    OpenRouter fallback provider (T305).

    Single key, many upstream models, OpenAI-compatible /chat/completions
    with native json_schema support. Only constructed by the factory-owned
    LLMRouter; service code must use factory.get_llm().
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 base_url: Optional[str] = None):
        try:
            config = get_llm_config()
            self.api_key = api_key or config.fallback_api_key or os.getenv("OPENROUTER_API_KEY")
            self.default_model = model or config.fallback_model
            self.default_temperature = config.default_temperature
            self.default_max_tokens = config.default_max_tokens
        except Exception:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
            self.default_model = model or "openai/gpt-oss-20b"
            self.default_temperature = 0.7
            self.default_max_tokens = 2048
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL")
                         or "https://openrouter.ai/api/v1").rstrip("/")

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, messages: list, gen_config) -> dict:
        body: dict = {
            "model": gen_config.model if getattr(gen_config, "model", None) else self.default_model,
            "messages": messages,
            "temperature": gen_config.temperature if gen_config.temperature is not None else self.default_temperature,
            "max_tokens": gen_config.max_tokens if gen_config.max_tokens is not None else self.default_max_tokens,
        }
        fmt = getattr(gen_config, "response_format", None)
        if fmt:
            body["response_format"] = fmt
        return body

    async def _post(self, messages: list, gen_config) -> GenerationResult:
        import httpx as _httpx

        start_time = time.time()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for the fallback provider")
        async with _httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                self.base_url + "/chat/completions",
                headers=self._headers(),
                json=self._payload(messages, gen_config),
            )
        if resp.status_code == 429:
            err = RuntimeError(f"OpenRouter rate limited (429): {resp.text[:200]}")
            setattr(err, "status_code", 429)
            raise err
        if 500 <= resp.status_code < 600:
            err = RuntimeError(f"OpenRouter server error ({resp.status_code}): {resp.text[:200]}")
            setattr(err, "status_code", resp.status_code)
            raise err
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter request failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        choices = data.get("choices", [])
        text = (choices[0].get("message", {}).get("content", "") if choices else "") or ""
        usage = data.get("usage", {}) or {}
        return GenerationResult(
            text=text,
            model=data.get("model", self.default_model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            finish_reason=(choices[0].get("finish_reason", "unknown") if choices else "unknown"),
            processing_time=time.time() - start_time,
        )

    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate text using OpenRouter"""
        from ..base import GenerationConfig as _GC

        return await self._post(
            [{"role": "user", "content": prompt}], config or _GC()
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Chat using OpenRouter"""
        from ..base import GenerationConfig as _GC

        wire = [{"role": m.role, "content": m.content} for m in messages]
        return await self._post(wire, config or _GC())

    @property
    def model_name(self) -> str:
        return f"openrouter/{self.default_model}"


class Local7BProvider(LLMStrategy):
    """
    Future integration point for local fine-tuned 7B model.
    
    This is a placeholder for the future integration of a fine-tuned
    7B model trained locally for exam-style answering, structured
    outputs, and domain-specific reasoning.
    """
    
    def __init__(self):
        config = get_llm_config()
        self.model_path = config.future_model  # Path to local model
        self._model = None
    
    async def generate(
        self, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Generate using local 7B model"""
        # Placeholder for future implementation
        # Would use a local inference runtime (e.g. llama.cpp or transformers)
        raise NotImplementedError(
            "Local 7B model integration not yet implemented. "
            "This is a future integration point."
        )
    
    async def chat(
        self, 
        messages: list[ChatMessage], 
        config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        """Chat using local 7B model"""
        raise NotImplementedError(
            "Local 7B model integration not yet implemented."
        )
    
    @property
    def model_name(self) -> str:
        return f"local/{self.model_path}"


# Factory function
def create_llm_provider(provider_name: Optional[str] = None) -> LLMStrategy:
    """
    Create an LLM provider based on configuration
    
    Args:
        provider_name: Optional override ("groq", "openrouter", or "local-7b")
        
    Returns:
        LLMStrategy implementation
    """
    config = get_llm_config()
    provider = provider_name or config.provider
    
    if provider == "openrouter":
        return OpenRouterProvider()
    elif provider == "local-7b":
        return Local7BProvider()
    else:
        # Default to Groq
        return GroqProvider()


# Convenience function
def create_groq_provider() -> GroqProvider:
    """Create and return a Groq provider instance"""
    return GroqProvider()


__all__ = [
    "GroqProvider",
    "OpenRouterProvider", 
    "Local7BProvider",
    "create_llm_provider",
    "create_groq_provider",
]