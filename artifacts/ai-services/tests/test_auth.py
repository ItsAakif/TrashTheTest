"""T104/T105: JWT auth (register/login/refresh/revocation) + dependencies."""

import uuid
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest

from database import repositories
from services.auth import service as auth

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Pure service unit tests (no DB)
# ---------------------------------------------------------------------------

async def test_password_hash_is_argon2id_and_verifies():
    hashed = auth.hash_password("correct-horse-123")
    assert hashed != "correct-horse-123"
    assert "$argon2id$" in hashed
    assert auth.verify_password(hashed, "correct-horse-123") is True
    assert auth.verify_password(hashed, "wrong-password") is False


async def test_decode_rejects_wrong_secret(user_a):
    from services.auth.service import issue_token_pair

    pair = issue_token_pair(user_a.id, 0)
    with pytest.raises(pyjwt.InvalidSignatureError):
        pyjwt.decode(pair["access_token"], "wrong-secret", algorithms=["HS256"])
    # Service-level: token signed with another secret is 401.
    bad = pyjwt.encode(
        {"sub": str(user_a.id), "tv": 0, "type": "access", "iss": "trashthetest",
         "aud": "trashthetest-api",
         "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        "wrong-secret",
        algorithm="HS256",
    )
    with pytest.raises(auth.AuthError):
        auth.decode_token(bad, expected_type="access")


async def test_decode_rejects_expired_and_wrong_type(user_a):
    from config.settings import get_auth_config

    cfg = get_auth_config()
    expired = pyjwt.encode(
        {"sub": str(user_a.id), "tv": 0, "type": "access", "iss": cfg.issuer,
         "aud": cfg.audience,
         "exp": datetime.now(timezone.utc) - timedelta(minutes=1)},
        cfg.secret,
        algorithm="HS256",
    )
    with pytest.raises(auth.AuthError):
        auth.decode_token(expired, expected_type="access")

    refresh = auth.create_refresh_token(user_a.id, 0)
    with pytest.raises(auth.AuthError):
        auth.decode_token(refresh, expected_type="access")


async def test_token_for_missing_user_and_stale_version_rejected(dummy_db, user_a, monkeypatch):
    from services.auth.service import issue_token_pair

    pair = issue_token_pair(user_a.id, 0)

    async def none_user(db, uid):
        return None

    monkeypatch.setattr(repositories, "get_user_by_id", none_user)
    with pytest.raises(auth.AuthError):
        await auth.user_from_access_token(dummy_db, pair["access_token"])

    stale_user = user_a
    stale_user.token_version = 3

    async def stale_lookup(db, uid):
        return stale_user

    monkeypatch.setattr(repositories, "get_user_by_id", stale_lookup)
    with pytest.raises(auth.AuthError):
        await auth.user_from_access_token(dummy_db, pair["access_token"])


# ---------------------------------------------------------------------------
# API tests with faked repositories (no Postgres)
# ---------------------------------------------------------------------------

def _install_auth_fakes(monkeypatch, dummy_db, users_by_email):
    from database.engine import get_db
    from main import app

    async def fake_get_db():
        yield dummy_db

    async def fake_by_email(db, email):
        return users_by_email.get(email.strip().lower())

    async def fake_by_email_ci(db, email):
        return users_by_email.get(email.strip().lower())

    async def fake_by_id(db, uid):
        for u in users_by_email.values():
            if u.id == uid:
                return u
        return None

    async def fake_create(db, email, password_hash, name=None, settings=None):
        from tests.conftest import make_user

        user = make_user(email)
        user.password_hash = password_hash
        user.name = name
        users_by_email[email] = user
        return user

    monkeypatch.setattr(repositories, "get_user_by_email", fake_by_email)
    monkeypatch.setattr(repositories, "get_user_by_email_ci", fake_by_email_ci)
    monkeypatch.setattr(repositories, "get_user_by_id", fake_by_id)
    monkeypatch.setattr(repositories, "create_user", fake_create)
    app.dependency_overrides[get_db] = fake_get_db
    return app


async def test_register_login_refresh_me_flow(app_client, dummy_db, monkeypatch):
    from main import app

    store = {}
    _install_auth_fakes(monkeypatch, dummy_db, store)
    try:
        reg = await app_client.post(
            "/api/auth/register",
            json={"email": "new@example.com", "password": "supersecret-1", "name": "New"},
        )
        assert reg.status_code == 201, reg.text
        body = reg.json()
        assert body["access_token"] and body["refresh_token"]
        assert "$argon2id$" in store["new@example.com"].password_hash
        assert "supersecret-1" not in store["new@example.com"].password_hash

        dup = await app_client.post(
            "/api/auth/register",
            json={"email": "NEW@example.com", "password": "supersecret-1"},
        )
        assert dup.status_code == 409

        login = await app_client.post(
            "/api/auth/login",
            json={"email": "new@example.com", "password": "supersecret-1"},
        )
        assert login.status_code == 200, login.text

        bad = await app_client.post(
            "/api/auth/login",
            json={"email": "new@example.com", "password": "nope-nope-nope"},
        )
        assert bad.status_code == 401

        me = await app_client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {login.json()['access_token']}"},
        )
        assert me.status_code == 200
        assert me.json()["email"] == "new@example.com"

        refreshed = await app_client.post(
            "/api/auth/refresh",
            json={"refresh_token": login.json()["refresh_token"]},
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["access_token"]

        invalid_refresh = await app_client.post(
            "/api/auth/refresh", json={"refresh_token": "not-a-jwt"}
        )
        assert invalid_refresh.status_code == 401

        wrong_type = await app_client.post(
            "/api/auth/refresh",
            json={"refresh_token": login.json()["access_token"]},
        )
        assert wrong_type.status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_me_rejects_bad_tokens(app_client, dummy_db, monkeypatch, user_a):
    from main import app

    store = {user_a.email: user_a}
    _install_auth_fakes(monkeypatch, dummy_db, store)
    try:
        assert (await app_client.get("/api/auth/me")).status_code == 401

        demo = await app_client.get(
            "/api/auth/me", headers={"X-Demo-Mode": "true", "Authorization": "Bearer anything"}
        )
        assert demo.status_code == 401

        assert (
            await app_client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"})
        ).status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_password_change_invalidates_old_tokens(app_client, dummy_db, monkeypatch):
    from main import app
    from services.auth.service import issue_token_pair

    user = None
    store = {}

    async def fake_by_email(db, email):
        return store.get(email.strip().lower())

    async def fake_by_email_ci(db, email):
        return store.get(email.strip().lower())

    async def fake_by_id(db, uid):
        return store.get("u")

    async def fake_create(db, email, password_hash, name=None, settings=None):
        from tests.conftest import make_user

        u = make_user(email)
        u.password_hash = password_hash
        store["u"] = u
        store[email] = u
        return u

    from database.engine import get_db

    async def fake_get_db():
        yield dummy_db

    monkeypatch.setattr(repositories, "get_user_by_email", fake_by_email)
    monkeypatch.setattr(repositories, "get_user_by_email_ci", fake_by_email_ci)
    monkeypatch.setattr(repositories, "get_user_by_id", fake_by_id)
    monkeypatch.setattr(repositories, "create_user", fake_create)
    app.dependency_overrides[get_db] = fake_get_db
    try:
        reg = await app_client.post(
            "/api/auth/register",
            json={"email": "rot@example.com", "password": "old-password-1"},
        )
        assert reg.status_code == 201
        old_access = reg.json()["access_token"]

        chg = await app_client.post(
            "/api/auth/change-password",
            json={"current_password": "old-password-1", "new_password": "new-password-2"},
            headers={"Authorization": f"Bearer {old_access}"},
        )
        assert chg.status_code == 200, chg.text

        # Old token now fails token_version check.
        assert (
            await app_client.get("/api/auth/me", headers={"Authorization": f"Bearer {old_access}"})
        ).status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_no_demo_auth_remains():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for base, _, files in os.walk(root):
        if "__pycache__" in base or os.path.basename(base) == "tests":
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            for needle in ("demo" + "-user-001", "X-Demo" + "-Mode", "verify" + "_token"):
                if needle in text:
                    offenders.append(f"{path}:{needle}")
    assert offenders == [], f"demo auth remains: {offenders}"
