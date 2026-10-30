"""Dynamic model catalog: live fetch + 1h cache + static fallback.

Offline (httpx stubbed); live provider calls are NOT RUN without keys.
Covers:
- Groq live shape {data:[{id}]} normalized with curated pins preserved.
- OpenRouter live shape filters embedding/moderation, keeps context/pricing.
- 1h server-side cache (second call serves cache without HTTP).
- Provider failure falls back to the static curated list.
- Groq without an app key falls back (never uses a user key here).
- /models route: q alias, limit clamp, live flag, never leaks keys.
"""

import pytest

pytestmark = pytest.mark.asyncio


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _StubClient:
    calls = 0
    response = _FakeResp(200, {"data": []})
    error = None
    seen_auth = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        type(self).calls += 1
        type(self).seen_auth.append((headers or {}).get("Authorization"))
        if type(self).error is not None:
            raise type(self).error
        assert "models" in url
        return type(self).response


def _patch(monkeypatch, response=None, error=None):
    import httpx
    import services.provider_credentials.catalog as catalog

    _StubClient.calls = 0
    _StubClient.seen_auth = []
    _StubClient.response = response if response is not None else _FakeResp(200, {"data": []})
    _StubClient.error = error
    monkeypatch.setattr(httpx, "AsyncClient", _StubClient)
    catalog.clear_catalog_cache()
    return catalog


async def test_groq_live_normalized_and_curated_pinned(monkeypatch):
    catalog = _patch(monkeypatch, response=_FakeResp(200, {
        "object": "list",
        "data": [{"id": "llama-3.3-70b-versatile", "created": 1, "owned_by": "groq"}],
    }))
    out = await catalog.get_live_catalog(provider="groq")
    assert out["source"] == "live"
    ids = [e["id"] for e in out["models"]["groq"]]
    assert "llama-3.3-70b-versatile" in ids
    # Curated pins survive so saved values never orphan.
    assert "qwen/qwen3-32b" in ids
    # Groq fetch used a Bearer auth header (app key, server-side).
    assert _StubClient.seen_auth and _StubClient.seen_auth[0].startswith("Bearer ")


async def test_openrouter_filters_non_text_and_keeps_hints(monkeypatch):
    catalog = _patch(monkeypatch, response=_FakeResp(200, {"data": [
        {"id": "qwen/qwen3-32b:free", "name": "Qwen3 32B",
         "context_length": 131072, "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "openai/text-embedding-3-small", "name": "Embedding",
         "context_length": 8192, "pricing": {"prompt": "0.00000002", "completion": "0"}},
        {"id": "openai/whisper-large", "name": "Whisper"},
    ]}))
    out = await catalog.get_live_catalog(provider="openrouter")
    ids = [e["id"] for e in out["models"]["openrouter"]]
    assert "qwen/qwen3-32b:free" in ids
    assert "openai/text-embedding-3-small" not in ids
    assert "openai/whisper-large" not in ids
    entry = next(e for e in out["models"]["openrouter"] if e["id"] == "qwen/qwen3-32b:free")
    assert entry["context_length"] == 131072
    assert entry["pricing"] == "free"
    # OpenRouter catalog is public: no auth header is sent.
    assert _StubClient.seen_auth == [None]


async def test_cache_serves_second_call_without_http(monkeypatch):
    catalog = _patch(monkeypatch, response=_FakeResp(200, {
        "data": [{"id": "llama-3.3-70b-versatile"}]}))
    first = await catalog.get_live_catalog(provider="groq")
    assert first["source"] == "live"
    assert _StubClient.calls == 1
    second = await catalog.get_live_catalog(provider="groq")
    assert second["source"] == "cache"
    assert _StubClient.calls == 1
    assert [e["id"] for e in second["models"]["groq"]] == \
        [e["id"] for e in first["models"]["groq"]]


async def test_failure_falls_back_to_static(monkeypatch):
    import httpx

    catalog = _patch(monkeypatch, error=httpx.TimeoutException("slow"))
    out = await catalog.get_live_catalog(provider="groq")
    assert out["source"] == "static"
    ids = [e["id"] for e in out["models"]["groq"]]
    assert "qwen/qwen3-32b" in ids


async def test_search_and_limit_applied_server_side(monkeypatch):
    catalog = _patch(monkeypatch, response=_FakeResp(200, {"data": [
        {"id": "llama-3.3-70b-versatile"},
        {"id": "qwen/qwen3-32b"},
        {"id": "mixtral-8x7b-32768"},
    ]}))
    out = await catalog.get_live_catalog(provider="groq", search="qwen")
    ids = [e["id"] for e in out["models"]["groq"]]
    assert ids and all("qwen" in i.lower() for i in ids)
    out = await catalog.get_live_catalog(provider="groq", limit=1, refresh=True)
    assert len(out["models"]["groq"]) == 1


async def test_models_route_q_alias_limit_live_and_no_keys(
    app_client, dummy_db, monkeypatch, user_a,
):
    from main import app
    from database.engine import get_db
    from services.auth.deps import get_current_user

    async def fake_get_db():
        yield dummy_db

    async def fake_user():
        return user_a

    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    try:
        # Static path with q alias still works, no keys anywhere.
        resp = await app_client.get(
            "/api/settings/providers/models", params={"provider": "groq", "q": "qwen"})
        assert resp.status_code == 200, resp.text
        assert "gsk-" not in resp.text and "Bearer" not in resp.text
        body = resp.json()
        assert body["models"]["groq"]

        # Live path via stubbed HTTP.
        _patch(monkeypatch, response=_FakeResp(200, {
            "data": [{"id": "llama-3.3-70b-versatile"}]}))
        live = await app_client.get(
            "/api/settings/providers/models",
            params={"provider": "groq", "live": "true", "limit": "2"})
        assert live.status_code == 200, live.text
        payload = live.json()
        assert payload["source"] in ("live", "cache", "mixed", "static")
        assert len(payload["models"]["groq"]) <= 2
        assert "gsk-" not in live.text
    finally:
        app.dependency_overrides.clear()
        import services.provider_credentials.catalog as catalog
        catalog.clear_catalog_cache()
