"""T102: single FastAPI app wiring (uses OpenAPI: routers are lazy in FastAPI 0.141)."""

import os
import re

import pytest

SERVICE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.asyncio


async def _get(app_client, path):
    resp = await app_client.get(path)
    return resp


async def test_single_fastapi_app_and_no_second_entrypoint():
    offenders = []
    for root, _, files in os.walk(SERVICE_DIR):
        if "__pycache__" in root or os.path.basename(root) == "tests":
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                if "FastAPI" + "(" in fh.read():
                    offenders.append(os.path.relpath(path, SERVICE_DIR))
    assert offenders == [os.path.join("main.py")], f"FastAPI() found in: {offenders}"
    assert not os.path.exists(os.path.join(SERVICE_DIR, "api_gateway", "__init__.py"))


async def test_no_doubled_route_segments():
    from main import app

    paths = list(app.openapi()["paths"].keys())
    doubled = [p for p in paths if re.search(r"/api/[^/]+/api/", p)]
    assert doubled == [], f"doubled segments: {doubled}"
    assert "/api/auth/register" in paths
    assert "/api/auth/me" in paths
    assert "/api/projects" in paths


async def test_health_info_agree(app_client):
    health = (await _get(app_client, "/health")).json()
    assert health["status"] == "ok"
    assert health["service"] == "ai-services"

    info = (await _get(app_client, "/api/info")).json()
    assert info["service"] == health["service"]
    assert info["version"] == health["version"]

    detailed = (await _get(app_client, "/health/detailed")).json()
    assert detailed["service"] == health["service"]
    assert detailed["version"] == health["version"]
