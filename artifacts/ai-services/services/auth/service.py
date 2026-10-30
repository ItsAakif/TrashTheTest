"""JWT auth service: Argon2id passwords, HS256 access+refresh tokens.

Revocation: every token carries a ``tv`` (token_version) claim compared
against ``users.token_version``. Changing the password or calling logout
increments the column and invalidates all previously issued tokens.

Verification checks signature, exp, iss, aud, user existence, is_active,
and token_version. Any failure raises AuthError (mapped to 401).
"""

import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHash
from sqlalchemy.ext.asyncio import AsyncSession

_ph = PasswordHasher()  # Argon2id by default


class AuthError(Exception):
    """Authentication/authorization failure (HTTP 401, never leaks why)."""


# ---------------------------------------------------------------------------
# Passwords (Argon2id; constant-time compare happens inside argon2 verify)
# ---------------------------------------------------------------------------

def normalise_password(raw: str) -> str:
    """One password normalisation rule for every flow.

    Registration used to hash the raw string while login verified the
    string as typed, so a pasted trailing space made an account permanently
    unreachable. Trim exactly once here, at the flow boundary; the hash and
    verify primitives stay byte-exact so they never mask other differences.
    """
    return (raw or "").strip()


def hash_password(plaintext: str) -> str:
    return _ph.hash(plaintext)


def verify_password(password_hash: str, plaintext: str) -> bool:
    try:
        return _ph.verify(password_hash, plaintext)
    except (VerifyMismatchError, InvalidHash):
        # Fall back to a constant-time comparison so timing does not
        # distinguish "wrong password" from "corrupt hash".
        return hmac.compare_digest(password_hash or "", plaintext or "") and False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _ph.check_needs_rehash(password_hash)
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def _auth_config():
    from config.settings import get_auth_config

    return get_auth_config()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _issue_token(user_id: uuid.UUID, token_version: int, kind: str, ttl_seconds: int) -> str:
    cfg = _auth_config()
    now = _now()
    payload = {
        "sub": str(user_id),
        "tv": int(token_version),
        "type": kind,
        "iss": cfg.issuer,
        "aud": cfg.audience,
        "iat": now,
        "exp": now + timedelta(seconds=ttl_seconds),
        "jti": secrets.token_hex(16),
    }
    return jwt.encode(payload, cfg.secret, algorithm="HS256")


def create_access_token(user_id: uuid.UUID, token_version: int) -> str:
    cfg = _auth_config()
    return _issue_token(user_id, token_version, "access", cfg.access_ttl_seconds)


def create_refresh_token(user_id: uuid.UUID, token_version: int) -> str:
    cfg = _auth_config()
    return _issue_token(user_id, token_version, "refresh", cfg.refresh_ttl_seconds)


def issue_token_pair(user_id: uuid.UUID, token_version: int) -> dict[str, str]:
    return {
        "access_token": create_access_token(user_id, token_version),
        "refresh_token": create_refresh_token(user_id, token_version),
        "token_type": "bearer",
    }


def decode_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    """Decode + verify signature/exp/iss/aud/type. Raises AuthError."""
    cfg = _auth_config()
    try:
        payload = jwt.decode(
            token,
            cfg.secret,
            algorithms=["HS256"],
            issuer=cfg.issuer,
            audience=cfg.audience,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid or expired token") from exc
    if payload.get("type") != expected_type:
        raise AuthError("Invalid or expired token")
    return payload


async def _load_token_user(db: AsyncSession, payload: dict[str, Any]):
    from database import repositories

    try:
        user_id = uuid.UUID(str(payload.get("sub")))
    except (ValueError, TypeError) as exc:
        raise AuthError("Invalid or expired token") from exc
    user = await repositories.get_user_by_id(db, user_id)
    if user is None or not getattr(user, "is_active", True):
        raise AuthError("Invalid or expired token")
    if int(payload.get("tv", -1)) != int(getattr(user, "token_version", 0) or 0):
        raise AuthError("Invalid or expired token")
    return user


async def user_from_access_token(db: AsyncSession, token: str):
    payload = decode_token(token, expected_type="access")
    return await _load_token_user(db, payload)


# ---------------------------------------------------------------------------
# Flows (DB-backed)
# ---------------------------------------------------------------------------

async def register_user(
    db: AsyncSession, email: str, password: str, name: Optional[str] = None
):
    """Create a user; duplicate email -> AuthError('exists') (route maps 409)."""
    from database import repositories

    password = normalise_password(password)
    if len(password) < 8:
        # Trimming can turn an 8-space "password" into an empty one.
        # Keep the route's documented 8-character floor meaningful.
        raise AuthError("invalid password")
    normalized = email.strip().lower()
    existing = await repositories.get_user_by_email(db, normalized)
    if existing is None:
        # Case-insensitive fallback for rows predating the lower() index.
        existing = await repositories.get_user_by_email_ci(db, normalized)
    if existing is not None:
        raise AuthError("exists")
    user = await repositories.create_user(
        db=db,
        email=normalized,
        password_hash=hash_password(password),
        name=name,
    )
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate_user(db: AsyncSession, email: str, password: str):
    """Return (user, token_pair); invalid credentials -> AuthError."""
    from database import repositories

    password = normalise_password(password)
    normalized = email.strip().lower()
    user = await repositories.get_user_by_email(db, normalized)
    if user is None:
        user = await repositories.get_user_by_email_ci(db, normalized)
    if user is None or not getattr(user, "is_active", True):
        raise AuthError("Invalid email or password")
    ok = verify_password(user.password_hash, password)
    # Constant-time dummy verify so unknown-email vs wrong-password timing
    # does not leak account existence.
    if user is None:  # pragma: no cover - defensive
        verify_password(_ph.hash("dummy"), password)
    if not ok:
        raise AuthError("Invalid email or password")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
        await db.commit()
    pair = issue_token_pair(user.id, int(getattr(user, "token_version", 0) or 0))
    return user, pair


async def refresh_access_token(db: AsyncSession, refresh_token: str) -> dict[str, str]:
    payload = decode_token(refresh_token, expected_type="refresh")
    user = await _load_token_user(db, payload)
    return {
        "access_token": create_access_token(
            user.id, int(getattr(user, "token_version", 0) or 0)
        ),
        "token_type": "bearer",
    }


async def revoke_all_tokens(db: AsyncSession, user) -> None:
    """Increment token_version: every previously issued token fails."""
    user.token_version = int(getattr(user, "token_version", 0) or 0) + 1
    await db.commit()


async def change_password(
    db: AsyncSession, user, current_password: str, new_password: str
) -> None:
    current_password = normalise_password(current_password)
    new_password = normalise_password(new_password)
    if len(new_password) < 8:
        raise AuthError("invalid password")
    if not verify_password(user.password_hash, current_password):
        raise AuthError("Invalid email or password")
    user.password_hash = hash_password(new_password)
    user.token_version = int(getattr(user, "token_version", 0) or 0) + 1
    await db.commit()


__all__ = [
    "AuthError",
    "normalise_password",
    "hash_password",
    "verify_password",
    "needs_rehash",
    "create_access_token",
    "create_refresh_token",
    "issue_token_pair",
    "decode_token",
    "user_from_access_token",
    "register_user",
    "authenticate_user",
    "refresh_access_token",
    "revoke_all_tokens",
    "change_password",
]
