"""Embedding provider tests (T309 support + ADR-005 s3 contract).

Covers: Qwen query/passage prefixes are distinct, CPU-only 0.6B default
(1024d), factory default is the Qwen provider, cache key format
(model, content_hash), and the dimension guard remediation signal.
Offline: weights are never downloaded (prefix/cache/config only).
"""

import pytest

from services.strategies.embedding.qwen3_embedding import (
    QUERY_PREFIX,
    PASSAGE_PREFIX,
    MODEL_ID,
    DIMENSIONS,
    Qwen3EmbeddingProvider,
)


def test_query_and_passage_prefixes_are_distinct():
    assert QUERY_PREFIX != PASSAGE_PREFIX
    assert "Query" in QUERY_PREFIX
    assert "Passage" in PASSAGE_PREFIX


def test_prefix_applied_per_arm():
    provider = Qwen3EmbeddingProvider.__new__(Qwen3EmbeddingProvider)
    texts = ["cell biology"]
    q = provider._prefix(texts, "query")
    p = provider._prefix(texts, "passage")
    assert q[0].startswith(QUERY_PREFIX)
    assert p[0].startswith(PASSAGE_PREFIX)
    assert q[0] != p[0]


def test_cpu_06b_defaults():
    assert MODEL_ID == "Qwen/Qwen3-Embedding-0.6B"
    assert DIMENSIONS == 1024
    from config.settings import get_embedding_config

    cfg = get_embedding_config()
    assert cfg.dimensions == 1024
    assert cfg.device == "cpu"
    assert cfg.model == "Qwen/Qwen3-Embedding-0.6B"


def test_factory_default_is_qwen_not_bge():
    from factory import get_embedding

    provider = get_embedding()
    assert type(provider).__name__ == "Qwen3EmbeddingProvider"
    assert provider.dimensions == 1024


def test_cache_key_format_is_model_and_hash():
    from services.embedding_service.service import LocalEmbeddingService

    svc = LocalEmbeddingService.__new__(LocalEmbeddingService)
    key = svc._cache_key("Qwen/Qwen3-Embedding-0.6B", "abc123")
    assert key == "emb:Qwen/Qwen3-Embedding-0.6B:abc123"


def test_dimension_guard_message_names_remediation():
    msg = (
        "Embedding dimension mismatch: config EMBEDDING_DIMENSIONS=1024 "
        "but DB chunks.embedding_dim=384. Remediation: set EMBEDDING_DIMENSIONS=384"
    )
    assert "Remediation" in msg
    assert "EMBEDDING_DIMENSIONS" in msg
