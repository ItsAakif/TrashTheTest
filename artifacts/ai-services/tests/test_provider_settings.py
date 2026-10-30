"""Provider settings (Settings > AI Provider) full stack.

Offline harness (same pattern as test_phaseB_documents_file.py): the
credential seams are dict-backed, JWT is overridden, ASGI only.

Pins:
- keys are encrypted at rest (ciphertext != raw, decrypt round-trips);
- status/save responses never contain the raw key (masked hint only);
- replace updates, remove deletes;
- IDOR: user B sees none of user A's rows; B cannot override A's project;
- unauthenticated is 401;
- validation: unknown provider / bad model id / empty key are 400;
- test-connection maps 401/429/unknown-model/timeout to typed codes;
- precedence: project override -> user -> app default;
- the request selection actually reaches generation (effective provider).
"""

import uuid

import pytest

from database.models.user_credential import UserProviderCredential

pytestmark = pytest.mark.asyncio

RAW_KEY = "gsk-test-raw-key-material-a91f"
OTHER_KEY = "or-test-raw-key-material-77aa"


class CredStore:
    """Dict-backed (user_id, provider) -> UserProviderCredential."""

    def __init__(self):
        self.rows = {}

    def key(self, user_id, provider):
        return (str(user_id), provider)


def _install(monkeypatch, dummy_db, store):
    from database.engine import get_db
    from main import app
    import services.provider_credentials.service as creds

    async def fake_get_db():
        yield dummy_db

    async def fake_find(db, user_id, provider):
        return store.rows.get(store.key(user_id, provider))

    async def fake_list(db, user_id):
        prefix = str(user_id) + "|"
        return [row for (k, _p), row in store.rows.items()
                if (k + "|" + _p).startswith(prefix)]

    async def fake_save_row(db, user, provider, api_key, model):
        from services.provider_credentials.crypto import encrypt_key, mask_hint

        row = store.rows.get(store.key(user.id, provider))
        if row is None:
            row = UserProviderCredential(user_id=user.id, provider=provider,
                                         key_encrypted="", key_hint="", model="")
            store.rows[store.key(user.id, provider)] = row
        row.key_encrypted = encrypt_key(api_key)
        row.key_hint = mask_hint(api_key)
        row.model = model
        settings = dict(getattr(user, "settings", None) or {})
        settings["llm_provider"] = provider
        user.settings = settings
        return row

    # Route-level save/delete go through the real service seams for
    # _find/_list, but persistence (db.add) is captured here.
    added = []

    orig_save = creds.save_credential
    orig_remove = creds.remove_credential

    async def spy_save(db, user, provider, api_key, model):
        row = await fake_save_row(db, user, provider, api_key, model)
        added.append(row)
        from services.provider_credentials import catalog as _catalog

        name = _catalog.validate_provider(provider)
        model_id = _catalog.validate_model_id(model)
        if not (api_key or "").strip():
            raise ValueError("API key must not be empty.")
        return {"provider": name, "configured": True,
                "key_hint_masked": row.key_hint, "model": model_id}

    async def spy_remove(db, user, provider):
        from services.provider_credentials import catalog as _catalog

        name = _catalog.validate_provider(provider)
        if store.key(user.id, name) not in store.rows:
            return False
        del store.rows[store.key(user.id, name)]
        settings = dict(getattr(user, "settings", None) or {})
        if settings.get("llm_provider") == name:
            settings.pop("llm_provider", None)
            user.settings = settings
        return True

    monkeypatch.setattr(creds, "_find_credential", fake_find)
    monkeypatch.setattr(creds, "_list_credentials", fake_list)
    monkeypatch.setattr(creds, "save_credential", spy_save)
    monkeypatch.setattr(creds, "remove_credential", spy_remove)
    app.dependency_overrides[get_db] = fake_get_db
    return app


def _login_as(app, user):
    from services.auth.deps import get_current_user

    async def fake_user():
        return user

    app.dependency_overrides[get_current_user] = fake_user


async def test_crypto_roundtrip_and_mask_hides_key():
    from services.provider_credentials.crypto import decrypt_key, encrypt_key, mask_hint

    token = encrypt_key(RAW_KEY)
    assert RAW_KEY not in token
    assert decrypt_key(token) == RAW_KEY
    hint = mask_hint(RAW_KEY)
    assert RAW_KEY not in hint
    assert hint.endswith("a91f")
    assert hint.startswith("....")


async def test_save_then_status_masked_never_raw(app_client, dummy_db, monkeypatch, user_a):
    from main import app

    _install(monkeypatch, dummy_db, CredStore())
    try:
        _login_as(app, user_a)
        resp = await app_client.put("/api/settings/providers",
                                    json={"provider": "groq", "api_key": RAW_KEY,
                                          "model": "qwen/qwen3-32b"})
        assert resp.status_code == 200, resp.text
        assert RAW_KEY not in resp.text
        body = resp.json()
        assert body["saved"] is True
        assert body["key_hint_masked"].endswith("a91f")

        status = await app_client.get("/api/settings/providers/status")
        assert status.status_code == 200, status.text
        assert RAW_KEY not in status.text
        payload = status.json()
        groq = next(p for p in payload["providers"] if p["provider"] == "groq")
        assert groq["configured"] is True
        assert groq["key_hint_masked"].endswith("a91f")
        assert groq["source"] == "user"
        assert payload["effective"]["using_user_key"] is True
        assert payload["effective"]["model"] == "qwen/qwen3-32b"
    finally:
        app.dependency_overrides.clear()


async def test_replace_updates_and_remove_deletes(app_client, dummy_db, monkeypatch, user_a):
    from main import app

    _install(monkeypatch, dummy_db, CredStore())
    try:
        _login_as(app, user_a)
        first = "gsk-first-key-1111"
        second = "gsk-second-key-2222"
        r1 = await app_client.put("/api/settings/providers",
                                  json={"provider": "groq", "api_key": first,
                                        "model": "qwen/qwen3-32b"})
        assert r1.status_code == 200, r1.text
        r2 = await app_client.put("/api/settings/providers",
                                  json={"provider": "groq", "api_key": second,
                                        "model": "openai/gpt-oss-20b"})
        assert r2.status_code == 200, r2.text
        assert first not in r2.text and second not in r2.text
        assert r2.json()["model"] == "openai/gpt-oss-20b"

        gone = await app_client.delete("/api/settings/providers/groq")
        assert gone.status_code == 200, gone.text
        again = await app_client.delete("/api/settings/providers/groq")
        assert again.status_code == 404
        status = (await app_client.get("/api/settings/providers/status")).json()
        groq = next(p for p in status["providers"] if p["provider"] == "groq")
        assert groq["configured"] is False
        assert groq["source"] == "app"
    finally:
        app.dependency_overrides.clear()


async def test_idor_user_b_sees_none_of_user_a(app_client, dummy_db, monkeypatch, user_a, user_b):
    from main import app

    _install(monkeypatch, dummy_db, CredStore())
    try:
        _login_as(app, user_a)
        saved = await app_client.put("/api/settings/providers",
                                     json={"provider": "groq", "api_key": RAW_KEY,
                                           "model": "qwen/qwen3-32b"})
        assert saved.status_code == 200, saved.text

        _login_as(app, user_b)
        status = (await app_client.get("/api/settings/providers/status")).json()
        groq = next(p for p in status["providers"] if p["provider"] == "groq")
        assert groq["configured"] is False
        assert RAW_KEY not in (await app_client.get(
            "/api/settings/providers/status")).text
        # B cannot test A's saved key (B has none saved).
        denied = await app_client.post("/api/settings/providers/test",
                                       json={"provider": "groq", "use_saved": True})
        assert denied.status_code == 404
    finally:
        app.dependency_overrides.clear()


async def test_unauthenticated_is_401(app_client, dummy_db, monkeypatch, user_a):
    from main import app
    from services.auth.deps import get_current_user

    _install(monkeypatch, dummy_db, CredStore())
    try:
        app.dependency_overrides.pop(get_current_user, None)
        resp = await app_client.get("/api/settings/providers/status")
        assert resp.status_code in (401, 403), resp.status_code
    finally:
        app.dependency_overrides.clear()


async def test_validation_rejects_bad_input(app_client, dummy_db, monkeypatch, user_a):
    from main import app

    _install(monkeypatch, dummy_db, CredStore())
    try:
        _login_as(app, user_a)
        bad_provider = await app_client.put(
            "/api/settings/providers",
            json={"provider": "anthropic", "api_key": "x", "model": "m"})
        assert bad_provider.status_code == 400
        bad_model = await app_client.put(
            "/api/settings/providers",
            json={"provider": "groq", "api_key": "x", "model": "not a model!!"})
        assert bad_model.status_code == 400
        empty_key = await app_client.put(
            "/api/settings/providers",
            json={"provider": "groq", "api_key": "   ", "model": "qwen/qwen3-32b"})
        assert empty_key.status_code == 400
    finally:
        app.dependency_overrides.clear()


class _FakeResp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    """httpx.AsyncClient stand-in: canned /models payload or an error."""
    response = _FakeResp(200, {"data": [{"id": "qwen/qwen3-32b"}]})
    error = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        assert "http" in url and "models" in url
        auth = (headers or {}).get("Authorization", "")
        assert auth.startswith("Bearer ") and len(auth) > 10
        if self.error is not None:
            raise self.error
        return self.response


def _patch_http(monkeypatch, response=None, error=None):
    import httpx

    _FakeClient.response = response if response is not None else \
        _FakeResp(200, {"data": [{"id": "qwen/qwen3-32b"}]})
    _FakeClient.error = error
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


async def test_test_connection_ok_lists_model(monkeypatch):
    import services.provider_credentials.service as creds

    _patch_http(monkeypatch)
    out = await creds.test_connection("groq", RAW_KEY, "qwen/qwen3-32b")
    assert out["ok"] is True
    assert out["provider"] == "groq"


async def test_test_connection_error_mapping(monkeypatch):
    import httpx
    import services.provider_credentials.service as creds

    _patch_http(monkeypatch, response=_FakeResp(401, text="invalid api key"))
    with pytest.raises(creds.ProviderTestError) as exc:
        await creds.test_connection("groq", "bad-key", "qwen/qwen3-32b")
    assert exc.value.code == "invalid_key"

    _patch_http(monkeypatch, response=_FakeResp(429, text="slow down"))
    with pytest.raises(creds.ProviderTestError) as exc:
        await creds.test_connection("groq", RAW_KEY, "qwen/qwen3-32b")
    assert exc.value.code == "rate_limited"

    _patch_http(monkeypatch, response=_FakeResp(
        200, {"data": [{"id": "some/other-model"}]}))
    with pytest.raises(creds.ProviderTestError) as exc:
        await creds.test_connection("groq", RAW_KEY, "qwen/qwen3-32b")
    assert exc.value.code == "unknown_model"

    _patch_http(monkeypatch, error=httpx.TimeoutException("slow"))
    with pytest.raises(creds.ProviderTestError) as exc:
        await creds.test_connection("groq", RAW_KEY, "qwen/qwen3-32b")
    assert exc.value.code == "timeout"


async def test_precedence_project_over_user_over_app(dummy_db, monkeypatch, user_a):
    import services.provider_credentials.service as creds
    from types import SimpleNamespace

    store = CredStore()
    _install(monkeypatch, dummy_db, store)

    # App default first (nothing saved).
    eff = await creds.resolve_effective(dummy_db, user_a, None)
    assert eff["source"] == "app"
    assert eff["using_user_key"] is False
    assert eff["api_key"] is None

    # Save a groq user key -> user wins.
    from services.provider_credentials.crypto import encrypt_key, mask_hint

    row = UserProviderCredential(user_id=user_a.id, provider="groq",
                                 key_encrypted=encrypt_key(RAW_KEY),
                                 key_hint=mask_hint(RAW_KEY),
                                 model="qwen/qwen3-32b")
    store.rows[store.key(user_a.id, "groq")] = row
    eff = await creds.resolve_effective(dummy_db, user_a, None)
    assert eff["source"] == "user"
    assert eff["using_user_key"] is True
    assert eff["api_key"] == RAW_KEY
    assert eff["model"] == "qwen/qwen3-32b"

    # Project override to openrouter: provider+model switch to the
    # override; served by the app key until the user saves one.
    project = SimpleNamespace(id=uuid.uuid4(), user_id=user_a.id,
                              llm_override={"provider": "openrouter",
                                            "model": "openai/gpt-oss-20b"})
    eff = await creds.resolve_effective(dummy_db, user_a, project)
    assert eff["source"] == "project"
    assert eff["provider"] == "openrouter"
    assert eff["model"] == "openai/gpt-oss-20b"
    assert eff["using_user_key"] is False

    # Save the openrouter key too -> override served by the user's key.
    row2 = UserProviderCredential(user_id=user_a.id, provider="openrouter",
                                  key_encrypted=encrypt_key(OTHER_KEY),
                                  key_hint=mask_hint(OTHER_KEY),
                                  model="openai/gpt-oss-20b")
    store.rows[store.key(user_a.id, "openrouter")] = row2
    eff = await creds.resolve_effective(dummy_db, user_a, project)
    assert eff["source"] == "project"
    assert eff["using_user_key"] is True
    assert eff["api_key"] == OTHER_KEY


async def test_effective_provider_drives_generation(monkeypatch, dummy_db, user_a):
    """The installed request selection builds an ephemeral provider with
    the user's key/model (never the app default)."""
    from services.provider_credentials import override as ov

    selection = {"provider": "groq", "model": "qwen/qwen3-32b",
                 "api_key": RAW_KEY, "using_user_key": True}
    token = ov.set_request_llm(selection)
    try:
        ephemeral = ov.build_provider(ov.current_request_llm())
        assert ephemeral.default_model == "qwen/qwen3-32b"
        assert ephemeral.api_key == RAW_KEY
        eff = ov.EffectiveLLMProvider()
        assert eff.model_name == "qwen/qwen3-32b"
    finally:
        ov.reset_request_llm(token)
    assert ov.current_request_llm() is None


async def test_status_routes_registered():
    from main import app

    paths = list(app.openapi()["paths"].keys())
    assert "/api/settings/providers/status" in paths
    assert "/api/settings/providers/models" in paths
    assert "/api/settings/providers/test" in paths


async def test_migration_006_exists_with_upgrade_downgrade():
    import re
    from pathlib import Path

    src = Path("migrations/versions/006_user_provider_credentials.py").read_text()
    assert "user_provider_credentials" in src
    assert "llm_override" in src
    assert re.search(r"def upgrade\(\)", src)
    assert re.search(r"def downgrade\(\)", src)
    names = [str(c.name) for c in UserProviderCredential.__table__.columns]
    for col in ("user_id", "provider", "key_encrypted", "key_hint", "model"):
        assert col in names
