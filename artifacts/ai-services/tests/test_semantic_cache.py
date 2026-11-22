"""S3 semantic cache: identity, hashing, threshold configuration."""

import uuid

from services.semantic_cache import (
    identity_key,
    query_hash,
    retrieval_config_hash,
)


def test_identity_changes_with_every_safety_dimension():
    base = dict(
        user_id="u1", project_id="p1", doc_versions=["d1:1"],
        embedding_model="Qwen3", llm_model="m1", prompt_version="v1",
        retrieval_config_hash="r1", response_format="tutor-v1",
    )
    baseline = identity_key(**base)
    for field, value in {
        "user_id": "u2",
        "project_id": "p2",
        "doc_versions": ["d1:2"],
        "embedding_model": "other",
        "llm_model": "m2",
        "prompt_version": "v2",
        "retrieval_config_hash": "r2",
        "response_format": "other",
    }.items():
        changed = dict(base)
        changed[field] = value
        assert identity_key(**changed) != baseline, field


def test_doc_version_change_invalidates_identity():
    a = identity_key(user_id="u", project_id="p", doc_versions=["d:1"])
    b = identity_key(user_id="u", project_id="p", doc_versions=["d:2"])
    assert a != b


def test_query_hash_normalises_case_and_whitespace():
    assert query_hash("  What is   ATP? ") == query_hash("what is atp?")


def test_retrieval_config_hash_is_stable_and_order_independent():
    a = retrieval_config_hash(mode="hybrid", top_k=5, rerank=True)
    b = retrieval_config_hash(rerank=True, top_k=5, mode="hybrid")
    c = retrieval_config_hash(mode="dense", top_k=5, rerank=True)
    assert a == b
    assert a != c
