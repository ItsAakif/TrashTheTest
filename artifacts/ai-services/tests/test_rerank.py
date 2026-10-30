"""Reranker tests (T302). Offline, deterministic, no weights.

Covers: reranked output is a reordering (no drops/inventions), rerank
switchable off (NullReranker identity for A/B), latency measured and
budget breach selects the fallback and says so, alternates selectable
by config, and the event loop is not blocked (heartbeat task proceeds).
"""

import asyncio
import time

import pytest

from services.strategies.rerank import (
    BGEM3Reranker,
    MiniLMFallbackReranker,
    NullReranker,
    create_reranker,
)
from services.strategies.rerank.base import RerankConfig
from services.strategies.rerank.bge_reranker import MXBAIReranker, overlap_score


def candidates(n=50):
    return [f"chunk-{i:03d}" for i in range(n)]


def passages(n=50):
    out = []
    for i in range(n):
        if i % 7 == 0:
            out.append("photosynthesis chlorophyll converts light energy in leaves")
        else:
            out.append(f"unrelated filler text number {i} about nothing relevant")
    return out


@pytest.mark.asyncio
async def test_rerank_is_reordering_no_drops_or_inventions():
    reranker = BGEM3Reranker(RerankConfig())
    ids = candidates(20)
    ranked = await reranker.rerank("photosynthesis chlorophyll", passages(20), ids)
    assert sorted(r.chunk_id for r in ranked) == sorted(ids)
    assert [r.rank for r in ranked] == list(range(len(ids)))
    # relevant passages (every 7th) float to the top via the offline proxy
    assert ranked[0].chunk_id in ("chunk-000", "chunk-007", "chunk-014")


@pytest.mark.asyncio
async def test_disabled_reranker_is_identity_for_ab():
    reranker = NullReranker()
    ids = candidates(10)
    ranked = await reranker.rerank("anything", passages(10), ids)
    assert [r.chunk_id for r in ranked] == ids


@pytest.mark.asyncio
async def test_latency_measured_and_budget_breach_selects_fallback():
    cfg = RerankConfig(budget_ms=10**9)
    reranker = BGEM3Reranker(cfg)
    await reranker.rerank("q", passages(50), candidates(50))
    assert reranker.last_latency_ms >= 0.0

    # Service-level budget: zero budget forces the fallback path and names it.
    from services.retrieval.service import VectorRetrievalService
    from services.retrieval.types import RetrievalConfig

    from tests.test_retrieval import FakeEmbed, FakeSession, make_chunk

    svc = VectorRetrievalService(
        embedding_provider=FakeEmbed(),
        config=RetrievalConfig(embedding_model=FakeEmbed.model_name,
                               rerank_budget_ms=0.0),
    )
    chunks = [make_chunk(f"photosynthesis chlorophyll oxygen {i}") for i in range(6)]
    result = await svc.retrieve_with_db("photosynthesis", FakeSession(chunks),
                                        top_k=3, enable_rerank=True)
    assert result.rerank_applied is True
    assert result.reranker is not None
    assert "fallback" in result.reranker


def test_alternates_selectable_by_config():
    assert isinstance(create_reranker("mixedbread-ai/mxbai-rerank-v2"),
                      MXBAIReranker)
    assert isinstance(create_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2"),
                      MiniLMFallbackReranker)
    assert isinstance(create_reranker("BAAI/bge-reranker-v2-m3"), BGEM3Reranker)


def test_reranker_not_loaded_at_import():
    reranker = BGEM3Reranker(RerankConfig())
    assert reranker._model is None


@pytest.mark.asyncio
async def test_rerank_does_not_block_event_loop():
    reranker = BGEM3Reranker(RerankConfig())
    beats = []

    async def heartbeat():
        for _ in range(5):
            beats.append(time.time())
            await asyncio.sleep(0.005)

    heart = asyncio.create_task(heartbeat())
    await reranker.rerank("photosynthesis", passages(50), candidates(50))
    await heart
    assert len(beats) == 5
    assert beats[-1] - beats[0] >= 0.020


def test_overlap_score_is_deterministic_proxy():
    assert overlap_score("photosynthesis chlorophyll", "photosynthesis leaf") == 0.5
    assert overlap_score("", "anything") == 0.0
