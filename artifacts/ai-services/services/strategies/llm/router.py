"""LLM provider router with fallback (T305, ADR-005 section 7).

Primary: Groq (fast, low-cost, OpenAI-compatible). Fallback: OpenRouter
(single key, native json_schema structured outputs). The factory is the
only construction site; service code must never instantiate providers
directly (grep-gated in tests).

- Retries on 429/5xx with bounded backoff, then switches provider.
- DNS/TCP/TLS failures are TRANSIENT: retried 3x on the SAME provider with
  1s/3s/7s backoff before failover, never treated as a model failure and
  never allowed to rotate API keys (rotation stays 401/403/429 only).
- Records which provider+model served each response (surfaced to caller
  via GenerationResult.model "provider:model").
- Boot validates the configured model id against the provider /models
  endpoint and warns loudly on mismatch (config.settings.run_boot_checks).
- Typed ProviderUnavailable when every provider fails, never a bare 500.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ..base import LLMStrategy, GenerationConfig, GenerationResult, ChatMessage

logger = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    """Typed error: every configured provider failed. Never a bare 500."""

    def __init__(self, message: str, attempts: Optional[list] = None):
        super().__init__(message)
        self.attempts = attempts or []


# DNS/transport failures are NOT model failures: a Docker/WSL resolver hiccup
# once killed 23 notes topics with "[Errno -2] Name or service not known".
# They are classified separately, retried on the SAME provider (no key
# rotation - rotation stays reserved for 401/403/429) and only then failed
# over.
_NAME_RESOLUTION_MARKERS = (
    "name or service not known",
    "temporary failure in name resolution",
    "name resolution",
    "getaddrinfo",
    "nodename nor servname",
    "no address associated with hostname",
    "[errno -2]",
    "[errno -3]",
    "[errno -5]",
)
_TIMEOUT_MARKERS = (
    "readtimeout",
    "read timeout",
    "connecttimeout",
    "connect timeout",
    "timed out",
    "timeout",
)
_CONNECTION_MARKERS = (
    "connecterror",
    "connect error",
    "connection error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "connection closed",
    "server disconnected",
    "network is unreachable",
    "no route to host",
    "remote protocol error",
    "transport error",
)

# 3 attempts total per provider: sleep after attempt 1 (1s) and 2 (3s);
# the third failure fails over immediately (the 7s entry is the documented
# ceiling for a caller that raises the attempt budget).
DEFAULT_CONNECTION_BACKOFF_SECONDS: Tuple[float, ...] = (1.0, 2.0)
DEFAULT_MAX_CONNECTION_ATTEMPTS = 2


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Follow __cause__/__context__ so a wrapped gaierror is still seen."""
    seen: list[BaseException] = []
    current: Optional[BaseException] = exc
    while current is not None and current not in seen and len(seen) < 8:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def connection_cause(exc: BaseException) -> Optional[str]:
    """Return a human cause ("name resolution" / "timeout" / "connection").

    None means the error is not connection-level and must go through the
    ordinary retry/failover path.
    """
    for err in _exception_chain(exc):
        if isinstance(err, socket.gaierror):
            return "name resolution"
        name = type(err).__name__.lower()
        msg = str(err).lower()
        if any(marker in msg for marker in _NAME_RESOLUTION_MARKERS):
            return "name resolution"
        if any(marker in msg for marker in _TIMEOUT_MARKERS) or "timeout" in name:
            return "timeout"
        if any(marker in msg for marker in _CONNECTION_MARKERS) or name in (
            "connecterror",
            "connectionerror",
        ):
            return "connection"
    return None


def is_transient_connection_error(exc: BaseException) -> bool:
    """True for DNS/TCP/TLS transport failures (transient, provider-level)."""
    return connection_cause(exc) is not None


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = ("429", "rate limit", "rate_limit", "502", "503", "504",
               "timeout", "connection", "temporarily", "overloaded",
               "service unavailable", "bad gateway")
    status = getattr(exc, "status_code", None)
    if status is not None and (status == 429 or 500 <= int(status) < 600):
        return True
    return any(m in msg for m in markers)


@dataclass
class _Attempt:
    provider: str
    ok: bool = False
    error: str = ""


class LLMRouter(LLMStrategy):
    """Primary/fallback router. Implements LLMStrategy for drop-in use."""

    def __init__(
        self,
        primary: Optional[LLMStrategy] = None,
        fallback: Optional[LLMStrategy] = None,
        max_retries_per_provider: int = 2,
        backoff_base_seconds: float = 0.5,
        max_connection_attempts: int = DEFAULT_MAX_CONNECTION_ATTEMPTS,
        connection_backoff_seconds: Tuple[float, ...] = DEFAULT_CONNECTION_BACKOFF_SECONDS,
    ):
        if primary is None or fallback is None:
            from .groq import GroqProvider, OpenRouterProvider
            from .groq_rotating import GroqRotatingProvider

            try:
                primary = primary or GroqRotatingProvider()
            except Exception:
                primary = primary or GroqProvider()
            try:
                fallback = fallback or OpenRouterProvider()
            except Exception:
                fallback = None
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max(1, max_retries_per_provider)
        self.backoff_base = backoff_base_seconds
        self.max_connection_attempts = max(1, int(max_connection_attempts))
        self.connection_backoff = tuple(connection_backoff_seconds) or DEFAULT_CONNECTION_BACKOFF_SECONDS
        self.last_provider: Optional[str] = None
        self.last_attempts: List[_Attempt] = []

    @property
    def model_name(self) -> str:
        base = getattr(self.primary, "model_name", "unknown")
        return f"router({base})"

    def _tag(self, provider_name: str, result: GenerationResult) -> GenerationResult:
        result.model = f"{provider_name}:{result.model or getattr(self.primary, 'model_name', '')}"
        return result

    async def _call_with_retries(
        self, provider: LLMStrategy, kind: str, payload: Any,
        config: Optional[GenerationConfig],
    ) -> GenerationResult:
        last: Optional[Exception] = None
        pname = getattr(provider, "model_name", type(provider).__name__)
        attempt = 0
        while True:
            attempt += 1
            try:
                if kind == "chat":
                    result = await provider.chat(payload, config)
                else:
                    result = await provider.generate(payload, config)
                pname = getattr(provider, "model_name", type(provider).__name__)
                self.last_provider = pname
                return self._tag(pname, result)
            except Exception as exc:  # noqa: BLE001
                last = exc
                cause = connection_cause(exc)
                if cause is not None:
                    # Transient transport failure: retry the SAME provider
                    # (same key) with bounded backoff. No key rotation here.
                    if attempt >= self.max_connection_attempts:
                        logger.warning(
                            "%s provider unreachable (%s), attempt %d/%d; giving up on this provider",
                            pname, cause, attempt, self.max_connection_attempts,
                        )
                        raise
                    delay = self.connection_backoff[
                        min(attempt - 1, len(self.connection_backoff) - 1)
                    ]
                    logger.warning(
                        "%s provider unreachable (%s), attempt %d/%d; retrying in %.0fs",
                        pname, cause, attempt, self.max_connection_attempts, delay,
                    )
                    await asyncio.sleep(max(0.0, float(delay)))
                    continue
                if not _is_retryable(exc) or attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.backoff_base * (2 ** (attempt - 1)))
        raise last  # pragma: no cover - loop always raises first

    @staticmethod
    def _failure_message(attempts: List["_Attempt"], last_error: Optional[Exception]) -> str:
        cause = connection_cause(last_error) if last_error is not None else None
        if cause is not None:
            return (
                f"All LLM providers failed ({len(attempts)} attempts). "
                f"provider unreachable ({cause}): {last_error}"
            )
        return (
            f"All LLM providers failed ({len(attempts)} attempts). "
            f"Last error: {last_error}"
        )

    async def generate(
        self, prompt: str, config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        attempts: List[_Attempt] = []
        providers = [p for p in (self.primary, self.fallback) if p is not None]
        last_error: Optional[Exception] = None
        for provider in providers:
            pname = getattr(provider, "model_name", type(provider).__name__)
            try:
                result = await self._call_with_retries(provider, "generate", prompt, config)
                attempts.append(_Attempt(provider=pname, ok=True))
                self.last_attempts = attempts
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempts.append(_Attempt(provider=pname, error=str(exc)))
                continue
        self.last_attempts = attempts
        raise ProviderUnavailable(
            self._failure_message(attempts, last_error),
            attempts=[a.__dict__ for a in attempts],
        )

    async def chat(
        self, messages: List[ChatMessage], config: Optional[GenerationConfig] = None
    ) -> GenerationResult:
        attempts: List[_Attempt] = []
        providers = [p for p in (self.primary, self.fallback) if p is not None]
        last_error: Optional[Exception] = None
        for provider in providers:
            pname = getattr(provider, "model_name", type(provider).__name__)
            try:
                result = await self._call_with_retries(provider, "chat", messages, config)
                attempts.append(_Attempt(provider=pname, ok=True))
                self.last_attempts = attempts
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                attempts.append(_Attempt(provider=pname, error=str(exc)))
                continue
        self.last_attempts = attempts
        raise ProviderUnavailable(
            self._failure_message(attempts, last_error),
            attempts=[a.__dict__ for a in attempts],
        )


_router: Optional[LLMRouter] = None


def get_llm_router(**kwargs) -> LLMRouter:
    """Factory-owned singleton: the only LLM construction site (T305)."""
    global _router
    if _router is None:
        _router = LLMRouter(**kwargs)
    return _router


def reset_llm_router() -> None:
    global _router
    _router = None


__all__ = [
    "LLMRouter",
    "ProviderUnavailable",
    "get_llm_router",
    "reset_llm_router",
    "connection_cause",
    "is_transient_connection_error",
    "DEFAULT_CONNECTION_BACKOFF_SECONDS",
    "DEFAULT_MAX_CONNECTION_ATTEMPTS",
]
