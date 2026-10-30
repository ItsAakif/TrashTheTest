"""User provider credential service.

Encrypted store + masked status + effective-resolution + live test.

Security contract (never violated):
- Raw keys live only in request bodies inbound and in process memory
  outbound to the provider call. They are never logged, never returned
  by any endpoint, never placed in URLs, and never cached.
- Status endpoints return only masked hints (....last4).
- Each user sees only their own rows (user_id is always the JWT id).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.user import User
from database.models.user_credential import UserProviderCredential

from .catalog import PROVIDERS, app_default, validate_model_id, validate_provider
from .crypto import decrypt_key, encrypt_key, mask_hint


class ProviderTestError(Exception):
    """Typed test-connection failure with a stable machine code."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# Small DB seams (kept module-level so tests can stub them).
# ---------------------------------------------------------------------------

async def _find_credential(
    db: AsyncSession, user_id: uuid.UUID, provider: str
) -> Optional[UserProviderCredential]:
    result = await db.execute(
        select(UserProviderCredential).where(
            UserProviderCredential.user_id == user_id,
            UserProviderCredential.provider == provider,
        )
    )
    try:
        return result.scalars().first()
    except AttributeError:
        return None


async def _list_credentials(db: AsyncSession, user_id: uuid.UUID) -> list[UserProviderCredential]:
    result = await db.execute(
        select(UserProviderCredential).where(UserProviderCredential.user_id == user_id)
    )
    try:
        return list(result.scalars().all())
    except AttributeError:
        return []


def _active_provider(user: User) -> Optional[str]:
    settings = getattr(user, "settings", None) or {}
    active = settings.get("llm_provider") if isinstance(settings, dict) else None
    return active if active in PROVIDERS else None


def _set_active_provider(user: User, provider: str) -> None:
    settings = dict(getattr(user, "settings", None) or {})
    settings["llm_provider"] = provider
    user.settings = settings


# ---------------------------------------------------------------------------
# CRUD + masked status.
# ---------------------------------------------------------------------------

async def save_credential(
    db: AsyncSession,
    user: User,
    provider: str,
    api_key: str,
    model: str,
) -> dict[str, Any]:
    """Validate, encrypt, and upsert. Returns the masked status (never raw)."""
    name = validate_provider(provider)
    model_id = validate_model_id(model)
    if not (api_key or "").strip():
        raise ValueError("API key must not be empty.")

    token = encrypt_key(api_key)
    hint = mask_hint(api_key)

    row = await _find_credential(db, user.id, name)
    if row is None:
        row = UserProviderCredential(
            user_id=user.id, provider=name,
            key_encrypted=token, key_hint=hint, model=model_id,
        )
        db.add(row)
    else:
        row.key_encrypted = token
        row.key_hint = hint
        row.model = model_id
    _set_active_provider(user, name)
    try:
        await db.flush()
    except AttributeError:
        pass
    return _masked_row(name, hint, model_id, configured=True)


def _masked_row(provider: str, hint: str, model: str, configured: bool) -> dict[str, Any]:
    return {
        "provider": provider,
        "configured": configured,
        "key_hint_masked": hint if configured else None,
        "model": model if configured else None,
    }


async def get_status(db: AsyncSession, user: User) -> dict[str, Any]:
    """Masked per-provider status + effective resolution (never raw keys)."""
    rows = {r.provider: r for r in await _list_credentials(db, user.id)}
    providers = []
    for name in PROVIDERS:
        row = rows.get(name)
        default = app_default(name)
        if row is not None:
            entry = _masked_row(name, row.key_hint, row.model, True)
            entry["source"] = "user"
            entry["effective_model"] = row.model
        else:
            entry = _masked_row(name, "", "", False)
            entry["source"] = "app"
            entry["effective_model"] = default["model"]
        entry["app_model"] = default["model"]
        entry["app_key_available"] = default["has_key"]
        providers.append(entry)
    effective = await resolve_effective(db, user)
    return {
        "active_provider": effective["provider"],
        "providers": providers,
        "effective": {
            "provider": effective["provider"],
            "model": effective["model"],
            "source": effective["source"],
            # "Using your key" vs "app-provided" display signal.
            "using_user_key": effective["using_user_key"],
            "key_hint_masked": effective["key_hint"],
        },
    }


async def remove_credential(db: AsyncSession, user: User, provider: str) -> bool:
    name = validate_provider(provider)
    row = await _find_credential(db, user.id, name)
    if row is None:
        return False
    try:
        await db.delete(row)
        await db.flush()
    except AttributeError:
        pass
    if _active_provider(user) == name:
        settings = dict(getattr(user, "settings", None) or {})
        settings.pop("llm_provider", None)
        user.settings = settings
    return True


# ---------------------------------------------------------------------------
# Project override (precedence head).
# ---------------------------------------------------------------------------

def get_project_override(project: Any) -> Optional[dict[str, str]]:
    raw = getattr(project, "llm_override", None)
    if not isinstance(raw, dict):
        return None
    try:
        provider = validate_provider(str(raw.get("provider", "")))
        model = validate_model_id(str(raw.get("model", "")))
    except ValueError:
        return None
    return {"provider": provider, "model": model}


async def set_project_override(
    db: AsyncSession, project: Any, provider: str, model: str
) -> dict[str, str]:
    override = {
        "provider": validate_provider(provider),
        "model": validate_model_id(model),
    }
    project.llm_override = override
    try:
        await db.flush()
    except AttributeError:
        pass
    return override


async def clear_project_override(db: AsyncSession, project: Any) -> None:
    project.llm_override = None
    try:
        await db.flush()
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Effective resolution: project override -> user -> app default.
# ---------------------------------------------------------------------------

async def resolve_effective(
    db: AsyncSession,
    user: Optional[User] = None,
    project: Any = None,
) -> dict[str, Any]:
    """Resolve the concrete LLM selection for a request.

    Returns {provider, model, source, using_user_key, api_key, key_hint}.
    `api_key` is the decrypted user key or None (caller falls back to the
    env/app key). Never raises: failures degrade to the app default.
    """
    try:
        from config.settings import get_llm_config

        cfg = get_llm_config()
        app_provider = (cfg.provider or "groq").strip().lower()
        if app_provider not in PROVIDERS:
            app_provider = "groq"
        app_model = cfg.primary_model if app_provider == "groq" else cfg.fallback_model
    except Exception:
        app_provider, app_model = "groq", ""

    provider, model, source = app_provider, app_model, "app"
    api_key: Optional[str] = None
    key_hint: Optional[str] = None
    using_user_key = False

    try:
        if project is not None:
            override = get_project_override(project)
            if override is not None:
                provider, model, source = override["provider"], override["model"], "project"
        if user is not None:
            if source == "project":
                # Project override fixes the provider: only the user's key
                # for THAT provider may serve it (never a foreign key).
                row = await _find_credential(db, user.id, provider)
                if row is not None:
                    try:
                        api_key = decrypt_key(row.key_encrypted)
                        key_hint, using_user_key = row.key_hint, True
                    except ValueError:
                        api_key, key_hint, using_user_key = None, None, False
            else:
                # User level: the active provider's credential wins; when
                # nothing is marked active, the first configured one does.
                names = [p for p in ([_active_provider(user)] if _active_provider(user) else [])
                         + [p for p in PROVIDERS]]
                seen: set[str] = set()
                for name in names:
                    if name in seen or name not in PROVIDERS:
                        continue
                    seen.add(name)
                    row = await _find_credential(db, user.id, name)
                    if row is None:
                        continue
                    try:
                        api_key = decrypt_key(row.key_encrypted)
                    except ValueError:
                        continue
                    key_hint, using_user_key = row.key_hint, True
                    provider, model, source = name, row.model, "user"
                    break
        # A project override without a user key still controls provider +
        # model (served by the app key); the display stays "app-provided".
    except Exception:
        provider, model, source = app_provider, app_model, "app"
        api_key, key_hint, using_user_key = None, None, False

    return {
        "provider": provider,
        "model": model,
        "source": source,
        "using_user_key": using_user_key,
        "api_key": api_key,
        "key_hint": key_hint,
    }


# ---------------------------------------------------------------------------
# Live test-connection: minimal real call with mapped errors.
# ---------------------------------------------------------------------------

def _provider_base(provider: str) -> str:
    try:
        from config.settings import get_llm_config, groq_models_base

        cfg = get_llm_config()
        if provider == "groq":
            # Host-only GROQ_BASE_URL must be normalized for /models,
            # else the Test-connection probe 404s on a valid key.
            return groq_models_base(cfg.base_url)
        return (cfg.fallback_base_url or "https://openrouter.ai/api/v1").rstrip("/")
    except Exception:
        return "https://api.groq.com/openai/v1" if provider == "groq" \
            else "https://openrouter.ai/api/v1"


def _map_http_error(status_code: int, body: str) -> ProviderTestError:
    snippet = (body or "")[:200]
    if status_code == 401:
        return ProviderTestError("invalid_key",
                                 "Key rejected (401). Check the key and try again.", 400)
    if status_code == 403:
        return ProviderTestError("forbidden",
                                 f"Provider refused the key (403). {snippet}".strip(), 400)
    if status_code == 404:
        return ProviderTestError("unknown_model",
                                 f"Model or endpoint not found (404). {snippet}".strip(), 400)
    if status_code == 429:
        return ProviderTestError("rate_limited",
                                 "Rate limited (429). Wait a minute and retry.", 502)
    if 500 <= status_code < 600:
        return ProviderTestError("provider_error",
                                 f"Provider error ({status_code}). {snippet}".strip(), 502)
    return ProviderTestError("request_failed",
                             f"Request failed ({status_code}). {snippet}".strip(), 400)


async def test_connection(
    provider: str,
    api_key: str,
    model: str,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Minimal live check: GET provider /models with the key, then verify
    the model id is listed. No tokens are spent and nothing is stored."""
    name = validate_provider(provider)
    key = (api_key or "").strip()
    if not key:
        raise ProviderTestError("missing_key", "API key is required for a connection test.", 400)
    model_id = validate_model_id(model)
    base = _provider_base(name)

    started = time.time()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.get(f"{base}/models",
                                    headers={"Authorization": f"Bearer {key}"})
    except httpx.TimeoutException as exc:
        raise ProviderTestError("timeout",
                                "Connection timed out. Check your network and retry.", 502) from exc
    except httpx.RequestError as exc:
        raise ProviderTestError("unreachable",
                                f"Provider unreachable: {type(exc).__name__}.", 502) from exc

    if resp.status_code != 200:
        try:
            body = resp.text
        except Exception:
            body = ""
        raise _map_http_error(resp.status_code, body)

    ids: list[str] = []
    try:
        data = resp.json()
        items = data.get("data", []) if isinstance(data, dict) else []
        ids = [m.get("id", "") for m in items if isinstance(m, dict)]
    except Exception:
        ids = []
    latency_ms = int((time.time() - started) * 1000)
    if ids and model_id not in ids:
        raise ProviderTestError(
            "unknown_model",
            f"Key works, but model '{model_id}' is not listed by the provider. "
            "Check the id (advanced models must match the provider exactly).",
            400,
        )
    result: dict[str, Any] = {"ok": True, "provider": name, "model": model_id,
                              "latency_ms": latency_ms}
    if not ids:
        result["warning"] = "Provider did not list models; the id could not be verified."
    return result


__all__ = [
    "ProviderTestError",
    "save_credential",
    "get_status",
    "remove_credential",
    "get_project_override",
    "set_project_override",
    "clear_project_override",
    "resolve_effective",
    "test_connection",
]
