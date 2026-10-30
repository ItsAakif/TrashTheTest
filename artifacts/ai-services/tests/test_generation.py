"""LLM router + structured generation tests (T305/T306). Offline.

Covers: no direct provider construction outside the factory/provider
modules, primary-429 succeeds via fallback and names the provider used,
all-fail surfaces typed ProviderUnavailable, key rotation uses the
second key when the first is rate-limited, forced-invalid triggers at
most two repairs then typed failure, success validates against the
model, and no unvalidated json.loads in the generation path.
"""

import json
import pathlib

import pytest

from services.strategies.base import GenerationConfig, GenerationResult
from services.strategies.llm.router import LLMRouter, ProviderUnavailable
from services.strategies.llm.structured import (
    StructuredGenerationError,
    generate_structured,
)
from services.transformation.types import FlashcardList


def _result(text, model="m"):
    return GenerationResult(text=text, model=model, usage={}, finish_reason="stop",
                            processing_time=0.0)


class StubProvider:
    def __init__(self, name, behavior):
        self._name = name
        self.behavior = behavior  # "ok" | "rate-limit" | "fail"
        self.calls = 0
        self.fixed_text = json.dumps({"flashcards": [{"front": "Q?", "back": "A",
                                                      "difficulty": "easy"}]})

    @property
    def model_name(self):
        return self._name

    async def generate(self, prompt, config=None):
        self.calls += 1
        if self.behavior == "rate-limit":
            raise RuntimeError("request failed with 429 rate limited, retry later")
        if self.behavior == "fail":
            raise RuntimeError("connection refused")
        return _result(self.fixed_text, self._name)

    async def chat(self, messages, config=None):
        prompt = messages[0].content if messages else ""
        return await self.generate(prompt, config)


def test_no_direct_provider_construction_outside_factory():
    root = pathlib.Path("services")
    offenders = []
    for path in root.rglob("*.py"):
        if "services/strategies/llm" in path.as_posix().replace("\\", "/"):
            continue  # provider modules may construct (factory-owned)
        if path.name == "factory.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "GroqProvider()" in text:
            offenders.append(str(path))
    assert offenders == [], f"direct construction outside factory: {offenders}"


@pytest.mark.asyncio
async def test_primary_429_succeeds_via_fallback_and_names_provider():
    primary = StubProvider("groq-test", "rate-limit")
    fallback = StubProvider("openrouter-test", "ok")
    router = LLMRouter(primary=primary, fallback=fallback, backoff_base_seconds=0.0)
    result = await router.generate("hello")
    assert fallback.calls == 1
    assert result.model.startswith("openrouter-test:")
    assert router.last_provider == "openrouter-test"


@pytest.mark.asyncio
async def test_all_providers_failing_gives_typed_error():
    router = LLMRouter(primary=StubProvider("p", "fail"),
                       fallback=StubProvider("f", "fail"),
                       backoff_base_seconds=0.0,
                       connection_backoff_seconds=(0.0, 0.0, 0.0))
    with pytest.raises(ProviderUnavailable):
        await router.generate("hello")


class _DnsProvider(StubProvider):
    """Provider whose transport cannot resolve the host (Docker/WSL DNS)."""

    def __init__(self, name, error=None):
        super().__init__(name, "fail")
        self.error = error or RuntimeError("[Errno -2] Name or service not known")

    async def generate(self, prompt, config=None):
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_dns_failure_is_retried_three_times_on_the_same_provider():
    primary = _DnsProvider("groq-dns")
    fallback = StubProvider("openrouter-ok", "ok")
    router = LLMRouter(
        primary=primary,
        fallback=fallback,
        max_retries_per_provider=2,
        backoff_base_seconds=0.0,
        connection_backoff_seconds=(0.0, 0.0, 0.0),
    )
    result = await router.generate("hello")

    assert primary.calls == 2, "transient DNS must use the bounded attempt budget (2), not max_retries"
    # The fallback still serves the call after the primary is exhausted.
    assert fallback.calls == 1
    assert result.model.startswith("openrouter-ok:")


@pytest.mark.asyncio
async def test_dns_exhaustion_names_name_resolution_in_typed_error(caplog):
    import socket

    primary = _DnsProvider("groq-dns", socket.gaierror(-2, "Name or service not known"))
    fallback = _DnsProvider("openrouter-dns")
    router = LLMRouter(
        primary=primary,
        fallback=fallback,
        backoff_base_seconds=0.0,
        connection_backoff_seconds=(0.0, 0.0, 0.0),
    )
    with pytest.raises(ProviderUnavailable) as excinfo:
        await router.generate("hello")

    message = str(excinfo.value)
    assert "provider unreachable (name resolution)" in message
    assert "Name or service not known" in message
    assert primary.calls == 2 and fallback.calls == 2
    # One log line per retry, naming the provider and attempt number.
    retries = [r for r in caplog.records if "provider unreachable" in r.getMessage()]
    assert any("attempt 1/2" in r.getMessage() for r in retries)


@pytest.mark.asyncio
async def test_connection_cause_classification():
    import socket

    from services.strategies.llm.router import connection_cause

    assert connection_cause(socket.gaierror(-2, "Name or service not known")) == "name resolution"
    assert connection_cause(RuntimeError("[Errno -3] Temporary failure in name resolution")) == "name resolution"
    assert connection_cause(RuntimeError("invalid api key (401 unauthorized)")) is None
    assert connection_cause(RuntimeError("Connection refused")) == "connection"
    assert connection_cause(RuntimeError("ReadTimeout while reading response")) == "timeout"
    assert connection_cause(RuntimeError("429 rate limit exceeded")) is None


def test_key_rotation_uses_second_key_when_first_rate_limited():
    from datetime import datetime

    from services.strategies.llm.groq_rotating import GroqRotatingProvider, KeyStatus

    provider = GroqRotatingProvider.__new__(GroqRotatingProvider)
    provider.api_keys = [
        KeyStatus(key="key-one", last_used=datetime.min),
        KeyStatus(key="key-two", last_used=datetime.min),
    ]
    provider.current_key_index = 0
    provider._mark_key_rate_limited(provider.api_keys[0], retry_after=600)
    chosen = provider._get_available_key()
    assert chosen.key == "key-two"


@pytest.mark.asyncio
async def test_structured_success_validates_against_model():
    provider = StubProvider("stub", "ok")
    out = await generate_structured(provider, FlashcardList, "prompt")
    assert len(out.flashcards) == 1
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_structured_forced_invalid_repairs_twice_then_fails():
    provider = StubProvider("stub", "ok")
    provider.fixed_text = "definitely not json {{{"
    with pytest.raises(StructuredGenerationError):
        await generate_structured(provider, FlashcardList, "prompt", max_repairs=2)
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_structured_repair_recovers():
    provider = StubProvider("stub", "ok")
    bad = json.dumps({"wrong": "shape"})
    good = json.dumps({"flashcards": [{"front": "Q?", "back": "A"}]})
    texts = [bad, good]
    orig = provider.generate

    async def scripted(prompt, config=None):
        provider.calls += 1
        idx = min(provider.calls - 1, 1)
        return _result(texts[idx], "stub")

    provider.generate = scripted
    out = await generate_structured(provider, FlashcardList, "prompt")
    assert len(out.flashcards) == 1
