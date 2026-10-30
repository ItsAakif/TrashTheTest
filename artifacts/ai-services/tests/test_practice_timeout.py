"""Practice endpoint bound: a cold model load returns a typed 504, not a hang."""

import asyncio
import uuid

import pytest
from fastapi import HTTPException

from services.transformation import routes

pytestmark = pytest.mark.asyncio


async def test_practice_timeout_returns_504_with_retry_detail(
    monkeypatch, user_a, dummy_db
):
    async def fake_project(*args, **kwargs):
        return object()

    async def fake_select_llm(*args, **kwargs):
        return None

    async def slow_generate(**kwargs):
        await asyncio.sleep(1)
        raise AssertionError("wait_for must cancel before generate returns")

    monkeypatch.setattr(routes, "require_project_access", fake_project)
    monkeypatch.setattr(routes, "_select_request_llm", fake_select_llm)
    monkeypatch.setattr(routes.transformation_engine, "generate_mcq", slow_generate)
    monkeypatch.setattr(routes, "PRACTICE_TIMEOUT_S", 0.01)

    request = routes.MCQRequest(project_id=str(uuid.uuid4()))
    with pytest.raises(HTTPException) as caught:
        await routes.generate_mcq(request, current_user=user_a, db=dummy_db)

    assert caught.value.status_code == 504
    assert caught.value.detail == (
        "Practice generation timed out while loading models; retry in a moment"
    )
