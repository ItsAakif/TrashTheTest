"""Model catalog for the Settings > AI Provider surface.

Single source of truth for selectable models. Curated static entries are
the guaranteed fallback; the live provider catalogs (Groq + OpenRouter)
are fetched server-side, cached for 1h, and merged in when available.

Live sources (server-side only, app key or public catalog):
- Groq: GET {GROQ_BASE_URL}/models with ``Authorization: Bearer <app key>``.
  OpenAI-compatible shape: {object: "list", data: [{id, created, owned_by}]}.
- OpenRouter: GET {OPENROUTER_BASE_URL}/models (public, no auth required).
  Shape: {data: [{id, name, description, context_length,
  pricing: {prompt, completion}, supported_parameters}]}.
  Text-capable filter excludes embedding/moderation/audio entries.

Security contract:
- Raw keys are never logged, never returned, never cached, never in URLs.
- The live Groq fetch uses the server app key only; user keys are only
  ever used by POST /test (test_connection) with an explicit user gesture.
- On any provider failure the static curated list is returned instead.

"Advanced" models: any well-formed custom id is accepted and validated
live by the test-connection call (unknown ids fail honestly with
`unknown_model` instead of failing silently at generation time).
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

PROVIDERS: tuple[str, ...] = ("groq", "openrouter")

# Curated dropdown entries. Labels are display-only; ids are sent to the
# provider. The `:free` suffix is OpenRouter's free-tier convention;
# `openrouter/auto` is OpenRouter's auto router ("Free Router").
# These are the guaranteed fallback when live providers are unreachable.
_CATALOG: dict[str, list[dict[str, str]]] = {
    "groq": [
        {"id": "qwen/qwen3.8-27b", "label": "Qwen3.8 27B"},
        {"id": "qwen/qwen3-32b", "label": "Qwen3 32B"},
        {"id": "openai/gpt-oss-20b", "label": "GPT-OSS 20B"},
        {"id": "openai/gpt-oss-120b", "label": "GPT-OSS 120B"},
    ],
    "openrouter": [
        {"id": "openrouter/auto", "label": "Free Router (auto)"},
        {"id": "qwen/qwen3-32b:free", "label": "Qwen3 Free"},
        {"id": "openai/gpt-oss-20b:free", "label": "GPT-OSS 20B Free"},
    ],
}

_CUSTOM_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-/:]{0,127}$")

# Groq hosts OpenAI-compatible /models; OpenRouter too (base URLs come
# from config, never literals here).
_PROVIDER_BASE_ENV = {"groq": "GROQ_BASE_URL", "openrouter": "OPENROUTER_BASE_URL"}

# ---------------------------------------------------------------------------
# Live catalog: fetch + 1h server-side cache + static fallback.
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS = 3600
DEFAULT_LIMIT = 100
MAX_LIMIT = 500

# provider -> {"expires": float, "entries": list[dict]}
_CACHE: dict[str, dict[str, Any]] = {}

# OpenRouter ids that are not text-chat models (excluded "where practical").
_NON_TEXT_HINTS = (
    "embedding",
    "embed",
    "moderation",
    "moderate",
    "whisper",
    "tts",
    "text-to-speech",
    "speech",
    "image",
    "vision-embed",
    "clip",
)


def validate_provider(provider: str) -> str:
    name = (provider or "").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
    return name


def validate_model_id(model: str) -> str:
    cleaned = (model or "").strip()
    if not cleaned or len(cleaned) > 128 or not _CUSTOM_MODEL_RE.match(cleaned):
        raise ValueError(
            "Invalid model id. Use a provider model id such as "
            "'qwen/qwen3-32b' (letters, digits, '.', '-', '_', '/', ':')."
        )
    return cleaned


def get_catalog(provider: Optional[str] = None, search: Optional[str] = None) -> dict[str, Any]:
    """Curated models merged with the configured app defaults.

    `search` filters case-insensitively across ids and labels (drives the
    advanced searchable dropdown). Custom ids are always allowed via the
    advanced input; this catalog only seeds the dropdown.
    """
    try:
        from config.settings import get_llm_config

        cfg = get_llm_config()
        app_defaults = {"groq": cfg.primary_model, "openrouter": cfg.fallback_model}
    except Exception:
        app_defaults = {}

    out: dict[str, Any] = {"providers": list(PROVIDERS), "models": {}}
    for name in PROVIDERS:
        if provider is not None and name != provider:
            continue
        entries = [dict(e) for e in _CATALOG[name]]
        seen = {e["id"] for e in entries}
        default_id = (app_defaults.get(name) or "").strip()
        if default_id and default_id not in seen:
            entries.insert(0, {"id": default_id, "label": f"{default_id} (app default)"})
        if search:
            needle = search.strip().lower()
            entries = [e for e in entries
                       if needle in e["id"].lower() or needle in e["label"].lower()]
        out["models"][name] = entries
    return out


def app_default(provider: str) -> dict[str, str]:
    """App-provided (env) model + availability for one provider."""
    name = validate_provider(provider)
    try:
        from config.settings import get_llm_config

        cfg = get_llm_config()
        if name == "groq":
            return {"model": cfg.primary_model, "has_key": bool(cfg.api_key)}
        return {"model": cfg.fallback_model, "has_key": bool(cfg.fallback_api_key)}
    except Exception:
        return {"model": "", "has_key": False}


# ---------------------------------------------------------------------------
# Live helpers (never log / return / cache keys).
# ---------------------------------------------------------------------------

def _provider_models_url(provider: str) -> str:
    try:
        from config.settings import get_llm_config, groq_models_base

        cfg = get_llm_config()
        if provider == "groq":
            # GROQ_BASE_URL is host-only (the SDK appends /openai/v1
            # itself); normalize for the raw /models fetch, else 404.
            return groq_models_base(cfg.base_url)
        return (cfg.fallback_base_url or "https://openrouter.ai/api/v1").rstrip("/")
    except Exception:
        return "https://api.groq.com/openai/v1" if provider == "groq" \
            else "https://openrouter.ai/api/v1"


def _app_key(provider: str) -> Optional[str]:
    """Server-side app key for catalog fetch (never returned to callers)."""
    try:
        from config.settings import get_llm_config

        cfg = get_llm_config()
        if provider == "groq":
            return cfg.api_key or None
        return cfg.fallback_api_key or None
    except Exception:
        return None


def _is_non_text_openrouter(model_id: str) -> bool:
    lowered = (model_id or "").lower()
    return any(hint in lowered for hint in _NON_TEXT_HINTS)


def _normalize_groq_item(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    model_id = str(item.get("id", "")).strip()
    if not model_id:
        return None
    return {"id": model_id, "label": model_id}


def _format_price(hint: Any) -> Optional[str]:
    try:
        value = float(hint)
    except (TypeError, ValueError):
        return None
    # OpenRouter pricing is per-token in USD; show per-1M tokens.
    per_m = value * 1_000_000
    if per_m == 0:
        return "free"
    if per_m < 0.01:
        return f"${per_m:.4f}/1M"
    if per_m < 10:
        return f"${per_m:.2f}/1M"
    return f"${per_m:.1f}/1M"


def _normalize_openrouter_item(item: Any) -> Optional[dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    model_id = str(item.get("id", "")).strip()
    if not model_id or _is_non_text_openrouter(model_id):
        return None
    name = str(item.get("name", "") or "").strip() or model_id
    entry: dict[str, Any] = {"id": model_id, "label": name, "name": name}
    try:
        ctx = item.get("context_length", None)
        if ctx is not None and int(ctx) > 0:
            entry["context_length"] = int(ctx)
    except (TypeError, ValueError):
        pass
    pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else None
    if pricing is not None:
        hint = _format_price(pricing.get("prompt"))
        if hint is not None:
            entry["pricing"] = hint
    return entry


def _get_cached(provider: str) -> Optional[list[dict[str, Any]]]:
    slot = _CACHE.get(provider)
    if not slot:
        return None
    if slot.get("expires", 0) <= time.time():
        _CACHE.pop(provider, None)
        return None
    entries = slot.get("entries")
    return [dict(e) for e in entries] if isinstance(entries, list) else None


def _set_cache(provider: str, entries: list[dict[str, Any]]) -> None:
    _CACHE[provider] = {
        "expires": time.time() + CACHE_TTL_SECONDS,
        "entries": [dict(e) for e in entries],
    }


def clear_catalog_cache(provider: Optional[str] = None) -> None:
    """Test seam: drop cached live entries (keys are never cached)."""
    if provider is None:
        _CACHE.clear()
        return
    _CACHE.pop(validate_provider(provider), None)


def _ensure_curated_and_default(
    provider: str, live_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pin curated ids + the app default so saved values never orphan."""
    seen = {str(e.get("id", "")) for e in live_entries}
    pinned: list[dict[str, Any]] = []
    try:
        from config.settings import get_llm_config

        cfg = get_llm_config()
        default_id = (cfg.primary_model if provider == "groq"
                      else cfg.fallback_model).strip()
    except Exception:
        default_id = ""
    if default_id and default_id not in seen:
        pinned.append({"id": default_id, "label": f"{default_id} (app default)"})
        seen.add(default_id)
    for curated in _CATALOG.get(provider, []):
        if curated["id"] not in seen:
            pinned.append(dict(curated))
            seen.add(curated["id"])
    return pinned + live_entries


def _apply_search_and_limit(
    entries: list[dict[str, Any]],
    search: Optional[str],
    limit: int,
) -> list[dict[str, Any]]:
    needle = (search or "").strip().lower()
    if needle:
        entries = [e for e in entries
                   if needle in str(e.get("id", "")).lower()
                   or needle in str(e.get("label", "")).lower()
                   or needle in str(e.get("name", "") or "").lower()]
    # Clamp limit: at least 1, at most MAX_LIMIT.
    try:
        count = int(limit)
    except (TypeError, ValueError):
        count = DEFAULT_LIMIT
    count = max(1, min(MAX_LIMIT, count))
    return entries[:count]


async def _fetch_live_entries(provider: str, timeout_seconds: float = 10.0) -> list[dict[str, Any]]:
    """Fetch the provider's live /models list. Raises on any failure."""
    import httpx

    name = validate_provider(provider)
    base = _provider_models_url(name)
    headers: dict[str, str] = {}
    if name == "groq":
        # Groq catalog requires auth: server-side app key only.
        key = _app_key("groq")
        if not key:
            raise RuntimeError("Groq app key not configured; cannot fetch live catalog.")
        headers = {"Authorization": f"Bearer {key}"}
    # OpenRouter catalog is public; no auth header is sent.

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.get(f"{base}/models", headers=headers or None)
    if resp.status_code != 200:
        raise RuntimeError(f"Provider /models returned HTTP {resp.status_code}.")
    try:
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError("Provider /models returned invalid JSON.") from exc
    items = payload.get("data", []) if isinstance(payload, dict) else []
    if not isinstance(items, list):
        raise RuntimeError("Provider /models returned an unexpected shape.")

    entries: list[dict[str, Any]] = []
    for item in items:
        entry = _normalize_groq_item(item) if name == "groq" \
            else _normalize_openrouter_item(item)
        if entry is not None:
            entries.append(entry)
    # Stable alphabetical order so pagination is deterministic.
    entries.sort(key=lambda e: str(e.get("id", "")))
    return _ensure_curated_and_default(name, entries)


async def get_live_catalog(
    provider: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    refresh: bool = False,
) -> dict[str, Any]:
    """Live catalog with 1h cache and static fallback.

    Returns {providers, models, source} where source is one of
    "live" (all fresh), "cache" (all served from cache),
    "mixed" (some providers fell back), or "static" (all fell back).
    `refresh=True` bypasses the cache. Never includes keys.
    """
    names: list[str]
    if provider is not None:
        names = [validate_provider(provider)]
    else:
        names = list(PROVIDERS)

    models: dict[str, list[dict[str, Any]]] = {}
    live_hits = 0
    cache_hits = 0
    for name in names:
        cached = None if refresh else _get_cached(name)
        if cached is not None:
            merged = _ensure_curated_and_default(name, cached)
            models[name] = _apply_search_and_limit(merged, search, limit)
            cache_hits += 1
            continue
        try:
            fresh = await _fetch_live_entries(name)
            # Cache the live portion only (curated pins are re-added on read
            # so a renamed app default never goes stale in the cache).
            curated_ids = {c["id"] for c in _CATALOG.get(name, [])}
            try:
                from config.settings import get_llm_config

                cfg = get_llm_config()
                default_id = (cfg.primary_model if name == "groq"
                              else cfg.fallback_model).strip()
                if default_id:
                    curated_ids.add(default_id)
            except Exception:
                pass
            _set_cache(name, [e for e in fresh if e.get("id") not in curated_ids])
            models[name] = _apply_search_and_limit(fresh, search, limit)
            live_hits += 1
        except Exception:
            # Graceful fallback: static curated list (+ app default).
            fallback = get_catalog(provider=name)["models"][name]
            models[name] = _apply_search_and_limit(
                [dict(e) for e in fallback], search, limit)
    if live_hits == len(names):
        source = "live"
    elif cache_hits == len(names):
        source = "cache"
    elif live_hits == 0 and cache_hits == 0:
        source = "static"
    else:
        source = "mixed"
    return {"providers": names, "models": models, "source": source}


__all__ = [
    "PROVIDERS",
    "CACHE_TTL_SECONDS",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "validate_provider",
    "validate_model_id",
    "get_catalog",
    "get_live_catalog",
    "clear_catalog_cache",
    "app_default",
]
