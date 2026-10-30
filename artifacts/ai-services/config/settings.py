"""
Central Configuration System for Model Strategy Layer.

Uses pydantic-settings so every value comes from the environment (or .env).
Secrets have NO defaults: importing this module without JWT_SECRET or
DATABASE_URL set raises immediately and names the missing variable.

Model identifiers come only from this config - service code must read them
via the get_*_config() helpers, never from string literals.
"""

import os
from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Legacy shape: AppConfig + sub-configs are preserved so existing callers
# (factory, services) keep working. Values are now sourced from _Env.
# ---------------------------------------------------------------------------

class OCRConfig(BaseModel):
    """OCR configuration - PP-OCRv6 CPU ONNX tiny det/rec, no API key."""

    provider: str = "ppocr-onnx"
    det_model: str = "PP-OCRv6_mobile_det_tiny"
    rec_model: str = "PP-OCRv6_mobile_rec_tiny"
    engine: str = "onnxruntime"
    device: str = "cpu"
    lang: str = "en"


class EmbeddingConfig(BaseModel):
    """Embedding configuration (default: 1024d multilingual, ADR-005)."""

    provider: str = "qwen3-embedding"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    dimensions: int = 1024
    device: str = "cpu"  # or "cuda"
    normalize_embeddings: bool = True
    cache_dir: str = "./.cache/embeddings"


class LLMConfig(BaseModel):
    """LLM configuration - primary + fallback routers, ids from config only."""

    provider: str = "groq"
    api_key: str | None = None
    base_url: str | None = None
    model: str = "qwen/qwen3-32b"
    primary_model: str = "qwen/qwen3-32b"
    fallback_model: str = "openai/gpt-oss-20b"
    fallback_api_key: str | None = None
    fallback_base_url: str | None = None
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    # Future integration point for local 7B model
    future_model: str = "local-7b-finetuned"


class TTSConfig(BaseModel):
    """TTS configuration"""

    provider: str = "kokoro"
    voice: str = "af_sarah"
    output_format: str = "wav"
    sample_rate: int = 24000


class SearchConfig(BaseModel):
    """Search configuration"""

    provider: str = "tavily"
    api_key: str | None = None
    enabled: bool = False  # Research augmentation toggle
    max_results: int = 5
    # Jina Reader for content extraction
    jina_enabled: bool = True


class ServerConfig(BaseModel):
    """Server configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    env: str = "dev"  # dev | prod (HSTS only when prod)


class StorageConfig(BaseModel):
    """Storage backend configuration"""

    backend: str = "local"  # local | s3
    path: str = "./storage"
    s3_endpoint: str | None = None
    s3_bucket: str | None = None
    s3_region: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None


class AuthConfig(BaseModel):
    """JWT auth configuration. Secret is required and has no default."""

    secret: str
    issuer: str = "trashthetest"
    audience: str = "trashthetest-api"
    access_ttl_seconds: int = 3600
    refresh_ttl_seconds: int = 604800


class WorkerConfig(BaseModel):
    """Background worker configuration"""

    concurrency: int = 2
    lock_timeout_seconds: int = 600
    redis_url: str = "redis://localhost:6379/0"


class PuterConfig(BaseModel):
    """Puter optional heavy-document path (default OFF, explicit gesture)."""

    enabled: bool = False
    heavy_page_threshold: int = 100


class RateLimitConfig(BaseModel):
    """Rate limit configuration (per window, sliding in-memory)."""

    auth_requests: int = 20
    auth_window_seconds: int = 60
    upload_requests: int = 30
    upload_window_seconds: int = 60


class RerankConfig(BaseModel):
    """Reranking configuration"""

    model: str = "BAAI/bge-reranker-v2-m3"


class AppConfig(BaseModel):
    """Main application configuration"""

    ocr: OCRConfig = OCRConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    llm: LLMConfig = LLMConfig()
    tts: TTSConfig = TTSConfig()
    search: SearchConfig = SearchConfig()
    server: ServerConfig = ServerConfig()
    storage: StorageConfig = StorageConfig()
    auth: AuthConfig | None = None
    worker: WorkerConfig = WorkerConfig()
    puter: PuterConfig = PuterConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    rerank: RerankConfig = RerankConfig()
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    database_url: str = ""

    # Environment override prefix
    env_prefix: str = ""


class _Env(BaseSettings):
    """Raw environment parsing with fail-fast required secrets.

    JWT_SECRET and DATABASE_URL have no defaults on purpose: if either is
    missing, instantiation raises a ValidationError naming the variable.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    JWT_SECRET: str
    DATABASE_URL: str

    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    JWT_ISSUER: str = "trashthetest"
    JWT_AUDIENCE: str = "trashthetest-api"
    JWT_ACCESS_TTL_SECONDS: int = 3600
    JWT_REFRESH_TTL_SECONDS: int = 604800

    STORAGE_BACKEND: str = "local"
    STORAGE_PATH: str = "./storage"
    S3_ENDPOINT: Optional[str] = None
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None

    EMBEDDING_MODEL: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIMENSIONS: int = 1024
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_CACHE_DIR: str = "./.cache/embeddings"

    RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"

    LLM_PROVIDER: str = "groq"
    LLM_PRIMARY: str = "qwen/qwen3-32b"
    LLM_FALLBACK: str = "openai/gpt-oss-20b"
    LLM_MODEL: Optional[str] = None  # legacy override for LLM_PRIMARY
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2048

    OCR_PROVIDER: str = "ppocr-onnx"
    OCR_DET_MODEL: str = "PP-OCRv6_mobile_det_tiny"
    OCR_REC_MODEL: str = "PP-OCRv6_mobile_rec_tiny"
    OCR_ENGINE: str = "onnxruntime"
    OCR_DEVICE: str = "cpu"
    OCR_LANG: str = "en"

    WORKER_CONCURRENCY: int = 2
    JOB_LOCK_TIMEOUT_SECONDS: int = 600
    REDIS_URL: str = "redis://localhost:6379/0"

    PUTER_ENABLED: bool = False
    PUTER_HEAVY_PAGE_THRESHOLD: int = 100

    RATE_LIMIT_AUTH_REQUESTS: int = 20
    RATE_LIMIT_AUTH_WINDOW_SECONDS: int = 60
    RATE_LIMIT_UPLOAD_REQUESTS: int = 30
    RATE_LIMIT_UPLOAD_WINDOW_SECONDS: int = 60

    TTS_PROVIDER: str = "kokoro"
    KOKORO_VOICE: str = "af_sarah"
    SEARCH_PROVIDER: str = "tavily"
    TAVILY_API_KEY: Optional[str] = None
    SEARCH_API_KEY: Optional[str] = None
    SEARCH_ENABLED: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    APP_ENV: str = "dev"

    @model_validator(mode="after")
    def _refuse_wildcard_cors_with_credentials(self):
        origins = parse_cors_origins(self.CORS_ORIGINS)
        if "*" in origins:
            raise ValueError(
                "CORS_ORIGINS must not be '*' when credentials are enabled; "
                "set an explicit allowlist instead."
            )
        return self


def parse_cors_origins(raw: Any) -> list[str]:
    """Accept a JSON list or a comma-separated string."""
    if isinstance(raw, list):
        return [o.strip() for o in raw if o and o.strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(o).strip() for o in parsed if str(o).strip()]
            except ValueError:
                pass
        return [o.strip() for o in text.split(",") if o.strip()]
    return []


def _load_env() -> _Env:
    try:
        return _Env()
    except Exception as exc:  # pragma: no cover - fail-fast path
        # Re-raise with the missing variable names prominent.
        raise RuntimeError(
            f"Missing required configuration: {exc}. "
            "Set JWT_SECRET and DATABASE_URL (see .env.example)."
        ) from exc


# Fail fast at import: names JWT_SECRET / DATABASE_URL when absent.
_ENV = _load_env()


def _build_app_config(env: _Env) -> AppConfig:
    primary = env.LLM_MODEL or env.LLM_PRIMARY
    return AppConfig(
        ocr=OCRConfig(
            provider=env.OCR_PROVIDER,
            det_model=env.OCR_DET_MODEL,
            rec_model=env.OCR_REC_MODEL,
            engine=env.OCR_ENGINE,
            device=env.OCR_DEVICE,
            lang=env.OCR_LANG,
        ),
        embedding=EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "qwen3-embedding"),
            model=env.EMBEDDING_MODEL,
            dimensions=env.EMBEDDING_DIMENSIONS,
            device=env.EMBEDDING_DEVICE,
            cache_dir=env.EMBEDDING_CACHE_DIR,
        ),
        llm=LLMConfig(
            provider=env.LLM_PROVIDER,
            api_key=env.GROQ_API_KEY or env.LLM_API_KEY,
            base_url=env.GROQ_BASE_URL,
            model=primary,
            primary_model=primary,
            fallback_model=env.LLM_FALLBACK,
            fallback_api_key=env.OPENROUTER_API_KEY,
            fallback_base_url=env.OPENROUTER_BASE_URL,
            default_temperature=env.LLM_TEMPERATURE,
            default_max_tokens=env.LLM_MAX_TOKENS,
        ),
        tts=TTSConfig(
            provider=env.TTS_PROVIDER,
            voice=env.KOKORO_VOICE,
        ),
        search=SearchConfig(
            provider=env.SEARCH_PROVIDER,
            api_key=env.TAVILY_API_KEY or env.SEARCH_API_KEY,
            enabled=env.SEARCH_ENABLED,
        ),
        server=ServerConfig(
            host=env.HOST,
            port=env.PORT,
            reload=os.getenv("RELOAD", "false").lower() == "true",
            env=env.APP_ENV,
        ),
        storage=StorageConfig(
            backend=env.STORAGE_BACKEND,
            path=env.STORAGE_PATH,
            s3_endpoint=env.S3_ENDPOINT,
            s3_bucket=env.S3_BUCKET,
            s3_region=env.S3_REGION,
            s3_access_key=env.S3_ACCESS_KEY,
            s3_secret_key=env.S3_SECRET_KEY,
        ),
        auth=AuthConfig(
            secret=env.JWT_SECRET,
            issuer=env.JWT_ISSUER,
            audience=env.JWT_AUDIENCE,
            access_ttl_seconds=env.JWT_ACCESS_TTL_SECONDS,
            refresh_ttl_seconds=env.JWT_REFRESH_TTL_SECONDS,
        ),
        worker=WorkerConfig(
            concurrency=env.WORKER_CONCURRENCY,
            lock_timeout_seconds=env.JOB_LOCK_TIMEOUT_SECONDS,
            redis_url=env.REDIS_URL,
        ),
        puter=PuterConfig(
            enabled=env.PUTER_ENABLED,
            heavy_page_threshold=env.PUTER_HEAVY_PAGE_THRESHOLD,
        ),
        rate_limit=RateLimitConfig(
            auth_requests=env.RATE_LIMIT_AUTH_REQUESTS,
            auth_window_seconds=env.RATE_LIMIT_AUTH_WINDOW_SECONDS,
            upload_requests=env.RATE_LIMIT_UPLOAD_REQUESTS,
            upload_window_seconds=env.RATE_LIMIT_UPLOAD_WINDOW_SECONDS,
        ),
        rerank=RerankConfig(model=env.RERANK_MODEL),
        cors_origins=parse_cors_origins(env.CORS_ORIGINS),
        database_url=env.DATABASE_URL,
    )


class Settings:
    """
    Settings manager that loads configuration from environment variables
    and provides a centralized config object.
    """

    def __init__(self):
        self._config: AppConfig | None = None

    def _load_from_env(self) -> AppConfig:
        """Load configuration from environment variables (fail-fast)."""
        global _ENV
        _ENV = _load_env()
        return _build_app_config(_ENV)

    @property
    def config(self) -> AppConfig:
        """Get the current configuration, loading from env if needed"""
        if self._config is None:
            self._config = _build_app_config(_ENV)
        return self._config

    def reload(self) -> AppConfig:
        """Force reload configuration from environment"""
        self._config = self._load_from_env()
        return self._config

    def get_provider(self, strategy_type: str) -> str:
        """Get the configured provider for a strategy type"""
        mapping = {
            "ocr": self.config.ocr.provider,
            "embedding": self.config.embedding.provider,
            "llm": self.config.llm.provider,
            "tts": self.config.tts.provider,
            "search": self.config.search.provider,
        }
        return mapping.get(strategy_type, "unknown")


async def run_boot_checks() -> None:
    """Startup validation.

    (a) Configured embedding dimension vs DB-declared dimension: hard failure
        with remediation on mismatch.
    (b) pgvector extension present and >= 0.8: warning, not a crash.
    (c) LLM model id resolvable via provider /models: warning, not a crash.
    All checks degrade to warnings when the DB/provider is unreachable so a
    clean checkout with only .env.example values still boots.
    """
    import logging

    logger = logging.getLogger(__name__)
    cfg = get_settings()

    # (a) + (b): database checks
    try:
        from sqlalchemy import text as _text

        from database.engine import engine as _engine

        async with _engine.connect() as conn:
            try:
                ver = (
                    await conn.execute(
                        _text("SELECT extversion FROM pg_extension WHERE extname='vector'")
                    )
                ).scalar()
                if not ver:
                    logger.warning(
                        "boot check: pgvector extension not installed; "
                        "run CREATE EXTENSION vector (requires >= 0.8)."
                    )
                else:
                    from packaging.version import Version as _V

                    if _V(str(ver)) < _V("0.8"):
                        logger.warning(
                            "boot check: pgvector %s < 0.8; upgrade pgvector to >= 0.8.",
                            ver,
                        )
            except Exception as exc:
                logger.warning("boot check: pgvector version unreadable: %s", exc)

            try:
                db_dim = (
                    await conn.execute(
                        _text(
                            "SELECT embedding_dim FROM chunks "
                            "WHERE embedding_dim IS NOT NULL LIMIT 1"
                        )
                    )
                ).scalar()
            except Exception:
                db_dim = None  # fresh DB / pre-Phase-3 schema: nothing to compare
            if db_dim is not None and int(db_dim) != int(cfg.embedding.dimensions):
                raise RuntimeError(
                    f"Embedding dimension mismatch: config EMBEDDING_DIMENSIONS="
                    f"{cfg.embedding.dimensions} but DB chunks.embedding_dim={db_dim}. "
                    f"Remediation: set EMBEDDING_DIMENSIONS={db_dim} to match the "
                    "indexed data, or re-embed all chunks for the new model."
                )
    except RuntimeError:
        raise
    except Exception as exc:
        logger.warning("boot check: database unreachable, skipping DB checks: %s", exc)

    # (c): LLM model resolvability (warning only)
    try:
        import httpx as _httpx

        base = (cfg.llm.base_url or "").rstrip("/")
        key = cfg.llm.api_key
        if base and key:
            # Host-only GROQ_BASE_URL (SDK appends /openai/v1 itself)
            # must be normalized for the raw /models fetch, else 404.
            url = groq_models_base(base) + "/models"
            try:
                resp = await _httpx.AsyncClient(timeout=5).get(
                    url, headers={"Authorization": f"Bearer {key}"}
                )
                if resp.status_code == 200:
                    ids: list[str] = []
                    try:
                        data = resp.json().get("data", [])
                        ids = [m.get("id", "") for m in data if isinstance(m, dict)]
                    except Exception:
                        ids = []
                    if ids and cfg.llm.primary_model not in ids:
                        logger.warning(
                            "boot check: LLM_PRIMARY=%s not listed by provider /models; "
                            "check the model id.",
                            cfg.llm.primary_model,
                        )
                else:
                    logger.warning(
                        "boot check: provider /models returned %s; skipping model check.",
                        resp.status_code,
                    )
            except Exception as exc:
                logger.warning("boot check: LLM provider unreachable: %s", exc)
        else:
            logger.warning(
                "boot check: no LLM base_url/api_key configured; skipping model check."
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("boot check: LLM check skipped: %s", exc)


# Global settings instance
settings = Settings()


def get_settings() -> AppConfig:
    """Get the current application settings"""
    return settings.config


def get_ocr_config() -> OCRConfig:
    """Get OCR configuration"""
    return settings.config.ocr


def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration"""
    return settings.config.embedding


def get_llm_config() -> LLMConfig:
    """Get LLM configuration"""
    return settings.config.llm


def groq_models_base(raw_base: str | None) -> str:
    """Normalize a Groq base URL for OpenAI-compatible /models fetches.

    GROQ_BASE_URL is intentionally host-only (https://api.groq.com)
    because the Groq SDK appends /openai/v1 itself; using that raw value
    for ``{base}/models`` would request https://api.groq.com/models and
    404. Append /openai/v1 only when the configured base carries no /v1
    path segment, so full bases (and non-Groq OpenAI-compat proxies such
    as https://proxy.example.com/v1) pass through unchanged. The SDK
    chat path is untouched by this helper.
    """
    base = (raw_base or "https://api.groq.com/openai/v1").rstrip("/")
    path = base.split("://", 1)[-1]
    if "/v1" not in path:
        base += "/openai/v1"
    return base


def get_tts_config() -> TTSConfig:
    """Get TTS configuration"""
    return settings.config.tts


def get_search_config() -> SearchConfig:
    """Get search configuration"""
    return settings.config.search


def get_storage_config() -> StorageConfig:
    """Get storage configuration"""
    return settings.config.storage


def get_auth_config() -> AuthConfig:
    """Get auth configuration (fail-fast: JWT_SECRET is required)"""
    auth = settings.config.auth
    assert auth is not None  # guaranteed by _load_env fail-fast
    return auth


def get_worker_config() -> WorkerConfig:
    """Get worker configuration"""
    return settings.config.worker


def get_puter_config() -> PuterConfig:
    """Get Puter optional-path configuration (default OFF)."""
    return settings.config.puter


def get_rate_limit_config() -> RateLimitConfig:
    """Get rate limit configuration"""
    return settings.config.rate_limit


def get_cors_origins() -> list[str]:
    """Get CORS allowlist (never ['*'] with credentials)."""
    return settings.config.cors_origins


# Export convenience
__all__ = [
    "AppConfig",
    "OCRConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "TTSConfig",
    "SearchConfig",
    "ServerConfig",
    "StorageConfig",
    "AuthConfig",
    "WorkerConfig",
    "PuterConfig",
    "RateLimitConfig",
    "RerankConfig",
    "Settings",
    "settings",
    "get_settings",
    "get_ocr_config",
    "get_embedding_config",
    "get_llm_config",
    "groq_models_base",
    "get_tts_config",
    "get_search_config",
    "get_storage_config",
    "get_auth_config",
    "get_worker_config",
    "get_puter_config",
    "get_rate_limit_config",
    "get_cors_origins",
    "run_boot_checks",
    "parse_cors_origins",
]
