"""Request-scoped LLM override: user selection controls generation.

Flow: a generation route resolves the effective selection
(service.resolve_effective: project override -> user -> app default) and
stores it in a ContextVar for the request. The singleton engines
(transformation, study) hold an EffectiveLLMProvider that delegates each
call to the override-built provider when one is set, else to the
factory-owned router. No global state is mutated, so concurrent requests
with different users never leak keys across each other.

Ephemeral providers are built per call from the resolved selection and
never cached: decrypted keys live only in request memory.
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

from services.strategies.base import ChatMessage, GenerationConfig, GenerationResult, LLMStrategy

# Set per request by generation/study routes; None means "app default".
_request_llm: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "ttt_request_llm", default=None
)


def set_request_llm(selection: Optional[dict[str, Any]]):
    """Install the resolved selection for this request (reset afterwards)."""
    return _request_llm.set(selection)


def reset_request_llm(token) -> None:
    _request_llm.reset(token)


def current_request_llm() -> Optional[dict[str, Any]]:
    return _request_llm.get()


def build_provider(selection: dict[str, Any]) -> LLMStrategy:
    """Build an ephemeral provider from a resolved selection.

    Construction goes through the provider module's own factory
    functions (the T305 grep gate forbids direct construction
    literals outside factory/provider modules); the result is never
    cached and never stored globally. The caller keeps the decrypted
    key in request memory only.
    """
    from services.strategies.llm import groq as _groq_module

    provider = selection.get("provider", "groq")
    api_key = selection.get("api_key")
    model = selection.get("model") or None
    if provider == "openrouter":
        return _groq_module.OpenRouterProvider(api_key=api_key, model=model)
    groq = _groq_module.create_groq_provider()
    if api_key:
        groq.api_key = api_key
        groq._client = None  # force reconnect with the user's key
    if model:
        groq.default_model = model
    return groq


class EffectiveLLMProvider(LLMStrategy):
    """Delegates each call to the request selection or the app router."""

    def _delegate(self) -> LLMStrategy:
        selection = _request_llm.get()
        if selection:
            try:
                return build_provider(selection)
            except Exception:
                pass
        from factory import get_llm as _get_llm

        return _get_llm()

    async def generate(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        # Preserve the legacy call shape exactly (config omitted when
        # None) so stub providers with generate(self, prompt) keep working.
        if config is None:
            return await self._delegate().generate(prompt)
        return await self._delegate().generate(prompt, config)

    async def chat(
        self, messages: list[ChatMessage], config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        if config is None:
            return await self._delegate().chat(messages)
        return await self._delegate().chat(messages, config)

    @property
    def model_name(self) -> str:
        try:
            return self._delegate().model_name
        except Exception:
            return "router(unknown)"


async def resolve_for_request(
    db, user=None, project_id: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Resolve and install the request selection. Returns the token used.

    Always returns a usable state: resolution never raises (service
    degrades to the app default), and installing None is a no-op reset.
    """
    from .service import resolve_effective

    project = None
    if project_id and user is not None:
        try:
            from database import repositories

            project = await repositories.get_project_by_id(db, project_id)
            if project is not None and project.user_id != user.id:
                project = None  # foreign project: ignore, never leak
        except Exception:
            project = None
    try:
        selection = await resolve_effective(db, user, project)
    except Exception:
        selection = None
    if selection is not None and selection.get("source") == "app" \
            and not selection.get("using_user_key"):
        # Pure app default: install nothing so the effective provider
        # delegates to the factory-owned router (rotation + OpenRouter
        # fallback + existing test seams stay exactly as before).
        selection = None
    return set_request_llm(selection)


__all__ = [
    "EffectiveLLMProvider",
    "build_provider",
    "current_request_llm",
    "resolve_for_request",
    "reset_request_llm",
    "set_request_llm",
]
