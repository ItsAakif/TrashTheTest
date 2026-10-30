"""Middleware: correlation ids, structured logging, errors, rate limits, headers."""

import json
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .exceptions import TrashTheTestException, get_status_code

logger = logging.getLogger(__name__)

CORRELATION_HEADER = "X-Request-ID"
# Header names whose values must never appear in logs.
_REDACTED_HEADERS = {"authorization", "x-api-key", "cookie", "set-cookie"}
# Substrings of secrets configured on this host (compared case-insensitively).
_SECRET_SUBSTRINGS: set[str] = set()


def _secret_substrings() -> set[str]:
    global _SECRET_SUBSTRINGS
    if not _SECRET_SUBSTRINGS:
        try:
            from config.settings import get_settings

            cfg = get_settings()
            candidates = [
                getattr(getattr(cfg, "auth", None), "secret", None),
                getattr(getattr(cfg, "llm", None), "api_key", None),
                getattr(getattr(cfg, "llm", None), "fallback_api_key", None),
                getattr(getattr(cfg, "search", None), "api_key", None),
            ]
            _SECRET_SUBSTRINGS = {c for c in candidates if c and len(c) >= 6}
        except Exception:
            _SECRET_SUBSTRINGS = set()
    return _SECRET_SUBSTRINGS


def _redact(text: str) -> str:
    for secret in _secret_substrings():
        if secret in text:
            text = text.replace(secret, "***redacted***")
    return text


def _safe_headers(headers) -> dict:
    return {
        k.lower(): ("***redacted***" if k.lower() in _REDACTED_HEADERS else _redact(v))
        for k, v in headers.items()
    }


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Generate (or echo) a per-request correlation id, propagated to jobs."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[CORRELATION_HEADER] = correlation_id
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured JSON request/response logging with secret redaction."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        correlation_id = getattr(request.state, "correlation_id", "-")
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client": request.client.host if request.client else "unknown",
                    "headers": _safe_headers(request.headers),
                }
            )
        )
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                json.dumps(
                    {
                        "event": "response",
                        "correlation_id": correlation_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_s": round(process_time, 3),
                    }
                )
            )
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as exc:  # pragma: no cover - handled downstream too
            process_time = time.time() - start_time
            logger.error(
                json.dumps(
                    {
                        "event": "error",
                        "correlation_id": correlation_id,
                        "method": request.method,
                        "path": request.url.path,
                        "error": _redact(type(exc).__name__),
                        "duration_s": round(process_time, 3),
                    }
                )
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Typed error envelope: no stack traces, correlation id echoed."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = getattr(request.state, "correlation_id", None) or str(uuid.uuid4())
        try:
            return await call_next(request)
        except TrashTheTestException as exc:
            status_code = get_status_code(exc)
            return JSONResponse(
                status_code=status_code,
                content={
                    "success": False,
                    "error": exc.message,
                    "error_code": exc.error_code,
                    "details": exc.details,
                    "correlation_id": correlation_id,
                    "timestamp": time.time(),
                },
                headers={CORRELATION_HEADER: correlation_id},
            )
        except Exception:  # noqa: BLE001 - envelope must never leak internals
            logger.exception("Unhandled exception (correlation_id=%s)", correlation_id)
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "correlation_id": correlation_id,
                    "timestamp": time.time(),
                },
                headers={CORRELATION_HEADER: correlation_id},
            )


def _rate_limit_settings():
    try:
        from config.settings import get_rate_limit_config

        cfg = get_rate_limit_config()
        return (
            cfg.auth_requests,
            cfg.auth_window_seconds,
            cfg.upload_requests,
            cfg.upload_window_seconds,
        )
    except Exception:
        return 20, 60, 30, 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory sliding-window rate limiter (no external service required).

    Applies per authenticated user id, per IP when unauthenticated, on
    /api/auth/* and */upload* routes. 429 carries Retry-After.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._hits: dict[str, list[float]] = {}

    def _key(self, request: Request, scope: str) -> str:
        user_id = self._best_effort_user_id(request)
        if user_id:
            return f"{scope}:user:{user_id}"
        ip = request.client.host if request.client else "unknown"
        return f"{scope}:ip:{ip}"

    @staticmethod
    def _best_effort_user_id(request: Request) -> str | None:
        """Decode the JWT sub without verifying (identity hint only)."""
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        token = auth[7:].strip()
        try:
            import jwt as _jwt

            payload = _jwt.decode(token, options={"verify_signature": False})
            sub = payload.get("sub")
            return str(sub) if sub else None
        except Exception:
            return None

    def _allowed(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = time.time()
        hits = [t for t in self._hits.get(key, []) if now - t < window]
        if len(hits) >= limit:
            retry_after = int(window - (now - hits[0])) + 1 if hits else window
            self._hits[key] = hits
            return False, max(retry_after, 1)
        hits.append(now)
        self._hits[key] = hits
        return True, 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # CORS preflight must never consume quota or 429: browsers send
        # OPTIONS before the real request, and CORSMiddleware (innermost)
        # answers it downstream. A 429 here would surface as a CORS failure.
        if request.method == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        auth_limit, auth_window, upload_limit, upload_window = _rate_limit_settings()

        scope: str | None = None
        if path.startswith("/api/auth"):
            scope, limit, window = "auth", auth_limit, auth_window
        elif "upload" in path:
            scope, limit, window = "upload", upload_limit, upload_window

        if scope is None:
            return await call_next(request)

        allowed, retry_after = self._allowed(self._key(request, scope), limit, window)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "Rate limit exceeded",
                    "error_code": "RATE_LIMIT_ERROR",
                },
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers; HSTS only in prod; restrictive CSP on website routes."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Published study sites are meant to be embedded (the app shows them
        # in an iframe and users share them). Only that public HTML endpoint
        # may be framed; everything else stays DENY.
        public_site_html = (
            request.url.path.startswith("/api/website/public/")
            and request.url.path.endswith("/html")
        )
        if not public_site_html:
            response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        try:
            from config.settings import get_settings

            is_prod = get_settings().server.env == "prod"
        except Exception:
            is_prod = False
        if is_prod:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )
        if request.url.path.startswith("/api/website"):
            frame_ancestors = "*" if public_site_html else "'none'"
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; style-src 'self' 'unsafe-inline'; "
                f"img-src 'self' data:; frame-ancestors {frame_ancestors}"
            )
        return response


__all__ = [
    "CorrelationIdMiddleware",
    "LoggingMiddleware",
    "ErrorHandlingMiddleware",
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware",
]
