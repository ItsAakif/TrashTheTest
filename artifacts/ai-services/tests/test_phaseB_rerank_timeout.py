"""Append-only regression test for the rerank timeout-skip (Phase B find 3)."""

import asyncio
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.asyncio


async def test_maybe_rerank_skips_hung_model_within_budget():
    from services.retrieval import service as retrieval_module

    svc = retrieval_module.retrieval_service
    old_budget = getattr(svc.config, "rerank_budget_ms", 2000.0)
    try:
        svc.config.rerank_budget_ms = 200
    except Exception:
        pass

    async def hang_forever(*args, **kwargs):
        await asyncio.sleep(3600)
        return []

    svc._reranker = SimpleNamespace(
        model_name="test-hung-reranker", rerank=hang_forever)
    try:
        c1 = SimpleNamespace(id="c1", content="photosynthesis light")
        c2 = SimpleNamespace(id="c2", content="mitochondria atp")
        rows = [(c1, 0.03, "both"), (c2, 0.02, "lexical")]
        out, meta = await asyncio.wait_for(
            svc._maybe_rerank("photosynthesis", rows, True, 2), timeout=30)
        assert [str(c.id) for c, _, _ in out] == ["c1", "c2"]
        assert meta["rerank_applied"] is False
        assert "timeout-skipped" in str(meta.get("reranker"))
    finally:
        svc._reranker = None
        try:
            svc.config.rerank_budget_ms = old_budget
        except Exception:
            pass
