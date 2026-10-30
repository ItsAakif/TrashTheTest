"""Password whitespace normalisation regression (live login-after-signup bug).

Live reproduction before the fix:
  POST /api/auth/register {"password": "Secret123! "}  -> 201
  POST /api/auth/login    {"password": "Secret123!"}   -> 401

Registration hashed the raw string while login verified the string as
typed. ``normalise_password`` now trims once at every flow boundary:
register, authenticate and change-password.
"""

import pytest

from database import repositories
from services.auth import service as auth

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

async def test_normalise_password_trims_only_outer_whitespace():
    assert auth.normalise_password(" Pass123! ") == "Pass123!"
    assert auth.normalise_password("Pass123!") == "Pass123!"
    assert auth.normalise_password("\tPass 123!\n") == "Pass 123!"
    assert auth.normalise_password("") == ""
    assert auth.normalise_password(None) == ""


async def test_hash_and_verify_primitives_stay_byte_exact():
    # Trimming lives in the flows, never in the primitives.
    hashed = auth.hash_password("Pass123!")
    assert auth.verify_password(hashed, "Pass123!") is True
    assert auth.verify_password(hashed, " Pass123! ") is False


# ---------------------------------------------------------------------------
# API regression: register padded -> login unpadded (and padded)
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


async def test_register_padded_login_unpadded_succeeds(app_client, dummy_db, monkeypatch):
    from main import app

    store = {}
    _install_auth_fakes(monkeypatch, dummy_db, store)
    try:
        reg = await app_client.post(
            "/api/auth/register",
            json={"email": "padded@example.com", "password": " Pass123! "},
        )
        assert reg.status_code == 201, reg.text

        # Hash covers the trimmed password, not the pasted whitespace.
        assert auth.verify_password(store["padded@example.com"].password_hash, "Pass123!")
        assert not auth.verify_password(store["padded@example.com"].password_hash, " Pass123! ")

        unpadded = await app_client.post(
            "/api/auth/login",
            json={"email": "padded@example.com", "password": "Pass123!"},
        )
        assert unpadded.status_code == 200, unpadded.text
        assert unpadded.json()["access_token"]

        padded = await app_client.post(
            "/api/auth/login",
            json={"email": "padded@example.com", "password": " Pass123! "},
        )
        assert padded.status_code == 200, padded.text

        wrong = await app_client.post(
            "/api/auth/login",
            json={"email": "padded@example.com", "password": "Pass123"},
        )
        assert wrong.status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_change_password_normalises_current_and_new(app_client, dummy_db, monkeypatch):
    from main import app

    store = {}
    _install_auth_fakes(monkeypatch, dummy_db, store)
    try:
        reg = await app_client.post(
            "/api/auth/register",
            json={"email": "rotate@example.com", "password": "OldPass!1"},
        )
        assert reg.status_code == 201, reg.text
        access = reg.json()["access_token"]

        chg = await app_client.post(
            "/api/auth/change-password",
            json={"current_password": " OldPass!1 ", "new_password": " NewPass!2 "},
            headers={"Authorization": f"Bearer {access}"},
        )
        assert chg.status_code == 200, chg.text

        login = await app_client.post(
            "/api/auth/login",
            json={"email": "rotate@example.com", "password": "NewPass!2"},
        )
        assert login.status_code == 200, login.text
    finally:
        app.dependency_overrides.clear()


async def test_whitespace_only_password_is_rejected_after_trim(
    app_client, dummy_db, monkeypatch
):
    from main import app

    store = {}
    _install_auth_fakes(monkeypatch, dummy_db, store)
    try:
        reg = await app_client.post(
            "/api/auth/register",
            json={"email": "blank@example.com", "password": "        "},
        )
        assert reg.status_code == 401, reg.text
        assert store == {}
    finally:
        app.dependency_overrides.clear()
