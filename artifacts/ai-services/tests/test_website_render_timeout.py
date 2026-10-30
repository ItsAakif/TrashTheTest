"""ADR-008 / T015: the synchronous website render path is bounded."""

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from services.website_gen import routes

pytestmark = pytest.mark.asyncio


def _request():
    return routes.WebsiteCreateRequest(
        project_id=str(uuid.uuid4()),
        content_type="mixed",
        title="Demo",
        structured_content={"sections": []},
    )


def _patch_access(monkeypatch):
    async def _access(project_id, user, db):
        return SimpleNamespace(id=uuid.uuid4())

    monkeypatch.setattr(routes, "require_project_access", _access)


class _SlowService:
    async def generate_website(self, gen_request, db):
        await asyncio.sleep(0.1)
        return SimpleNamespace(slug="slow")


class _FastService:
    async def generate_website(self, gen_request, db):
        return SimpleNamespace(slug="fast")


async def test_render_over_budget_raises_504(monkeypatch):
    _patch_access(monkeypatch)
    monkeypatch.setattr(routes, "website_gen_service", _SlowService())
    monkeypatch.setattr(routes, "WEBSITE_RENDER_TIMEOUT_S", 0.05)

    with pytest.raises(HTTPException) as exc:
        await routes.generate_website(
            _request(), current_user=SimpleNamespace(), db=object())

    assert exc.value.status_code == 504
    assert "timed out" in str(exc.value.detail)


async def test_fast_render_still_succeeds(monkeypatch):
    _patch_access(monkeypatch)
    monkeypatch.setattr(routes, "website_gen_service", _FastService())
    monkeypatch.setattr(routes, "WEBSITE_RENDER_TIMEOUT_S", 5.0)

    result = await routes.generate_website(
        _request(), current_user=SimpleNamespace(), db=object())

    assert result.slug == "fast"


async def test_timeout_constant_is_the_documented_ceiling():
    assert routes.WEBSITE_RENDER_TIMEOUT_S == 60.0
