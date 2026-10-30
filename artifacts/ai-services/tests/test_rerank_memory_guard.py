"""Reranker memory guard: bge-reranker must not OOM a 4GB container.

The default Qwen embedding (~1.4GB resident) and bge-reranker-v2-m3
(~2.4GB) cannot co-reside in the 4GB deployment VM. The rerank call has a
2s budget but the worker thread cannot be cancelled, so a cold bge load
keeps allocating and can kill the API mid-practice. When cgroup headroom is
clearly insufficient the service selects the ~90MB MiniLM cross-encoder
instead (reported honestly in retrieval metadata).
"""

import pytest

from services.retrieval.service import VectorRetrievalService
from services.strategies.rerank.bge_reranker import (
    BGEM3Reranker,
    MiniLMFallbackReranker,
)


def _svc():
    return VectorRetrievalService()


def test_low_headroom_uses_minilm_fallback(monkeypatch):
    monkeypatch.setattr(
        "services.strategies.rerank.base.memory_headroom_bytes",
        lambda: 900 * 1024 * 1024,
    )
    reranker = _svc()._get_reranker(True)
    assert isinstance(reranker, MiniLMFallbackReranker)


def test_ample_headroom_keeps_bge(monkeypatch):
    monkeypatch.setattr(
        "services.strategies.rerank.base.memory_headroom_bytes",
        lambda: 8 * 1024 * 1024 * 1024,
    )
    reranker = _svc()._get_reranker(True)
    assert isinstance(reranker, BGEM3Reranker)


def test_unknown_headroom_keeps_configured_choice(monkeypatch):
    monkeypatch.setattr(
        "services.strategies.rerank.base.memory_headroom_bytes",
        lambda: None,
    )
    reranker = _svc()._get_reranker(True)
    assert isinstance(reranker, BGEM3Reranker)


def test_disabled_reranker_is_none():
    assert _svc()._get_reranker(False) is None
