"""T103: CORS allowlist, security headers, rate limits, error envelope."""

import os
import re

import pytest

pytestmark = pytest.mark.asyncio


async def test_preflight_from_foreign_origin_gets_no_allow_origin(app_client):
    resp = await app_client.options(
        "/api/auth/login",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}


async def test_upload_preflight_allows_frontend_idempotency_key(app_client):
    """Phase 1: OPTIONS /api/documents/upload from 5173 must pass preflight.

    The frontend sends Authorization + Idempotency-Key (both non-simple),
    so the preflight requests both. Missing Idempotency-Key in allow_headers
    used to answer 400 'Disallowed CORS headers'. POST auth is untouched.
    """
    resp = await app_client.options(
        "/api/documents/upload",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,idempotency-key",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert resp.headers["access-control-allow-credentials"] == "true"


async def test_upload_preflight_never_rate_limited(monkeypatch):
    """Phase 1: OPTIONS preflight bypasses the upload rate limiter."""
    import httpx
    import api_gateway.middleware as mw
    from main import create_app

    monkeypatch.setattr(mw, "_rate_limit_settings", lambda: (1, 60, 1, 60))
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    preflight = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,idempotency-key",
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            resp = await client.options("/api/documents/upload", headers=preflight)
            assert resp.status_code == 200


async def test_no_wildcard_allow_origins_in_code():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    star = '"*"'  # built, not literal
    for base, _, files in os.walk(root):
        if "__pycache__" in base or os.path.basename(base) == "tests":
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for lineno, line in enumerate(fh, 1):
                    if "allow_origins=[" + star + "]" in line or "allow_origins=[" + star.replace('"', "'") + "]" in line:
                        offenders.append(f"{path}:{lineno}")
    assert offenders == [], f"wildcard CORS: {offenders}"


async def test_security_headers_and_correlation_id(app_client):
    resp = await app_client.get("/health")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "x-request-id" in {k.lower() for k in resp.headers}


async def test_auth_rate_limit_returns_429_with_retry_after(monkeypatch):
    import httpx
    import api_gateway.middleware as mw
    from main import create_app

    monkeypatch.setattr(mw, "_rate_limit_settings", lambda: (2, 60, 100, 60))
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/login", json={"email": "x@y.z", "password": "pw"})
        await client.post("/api/auth/login", json={"email": "x@y.z", "password": "pw"})
        limited = await client.post("/api/auth/login", json={"email": "x@y.z", "password": "pw"})
    assert limited.status_code == 429
    assert "retry-after" in {k.lower() for k in limited.headers}
    assert limited.json()["success"] is False


async def test_forced_500_returns_envelope_without_traceback():
    import httpx
    from main import create_app

    app = create_app()

    @app.get("/_test/boom")
    async def boom():
        raise RuntimeError("kaboom-secret-details")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/_test/boom")
    assert resp.status_code == 500
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "Internal server error"
    assert "kaboom" not in resp.text
    assert "traceback" not in resp.text.lower()
    assert body.get("correlation_id")
    assert "x-request-id" in {k.lower() for k in resp.headers}
