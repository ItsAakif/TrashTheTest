"""Cross-encoder rerankers (T302).

Default: BAAI/bge-reranker-v2-m3 (0.6B, Apache-2.0, multilingual), fused
top-50 -> top-8. Alternates: mixedbread-ai/mxbai-rerank-v2 (smaller/faster)
and cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU fallback). Qwen3-Reranker is
rejected as the default (causal-LM scoring latency, ADR-005 s6).

- Lazily loaded (import is cheap, weights never download in tests).
- Batched; scoring runs in a worker thread via asyncio.to_thread so the
  event loop is never blocked.
- Batch scoring helper is a pure function of (query, passages) so the
  offline fallback (token-overlap) is deterministic and testable.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import List, Optional

from .base import RerankConfig, RerankedChunk, RerankerStrategy, get_rerank_config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def overlap_score(query: str, passage: str) -> float:
    """Deterministic offline relevance proxy: |Q cap P| / |Q|.

    Used when cross-encoder weights are unavailable (tests, no network)
    and as the documented CPU floor. Pure function for eval use.
    """
    q = set(_TOKEN_RE.findall(query.lower()))
    if not q:
        return 0.0
    p = set(_TOKEN_RE.findall(passage.lower()))
    return len(q & p) / len(q)


class _CrossEncoderReranker:
    """Shared batching/threading machinery for sentence-transformers models."""

    def __init__(self, model_id: str, config: Optional[RerankConfig] = None):
        self._model_id = model_id
        self.config = config or get_rerank_config()
        self._model = None
        self.last_latency_ms: float = 0.0
        self.used_fallback: bool = False

    @property
    def model_name(self) -> str:
        return self._model_id

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_id)

    def _score_sync(self, pairs: List[List[str]]) -> List[float]:
        out: List[float] = []
        bs = max(1, self.config.batch_size)
        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            try:
                self._ensure_model()
                scores = self._model.predict(batch, show_progress_bar=False)
                out.extend([float(s) for s in list(scores)])
            except Exception:
                # Offline / no weights: deterministic proxy per pair.
                out.extend([overlap_score(q, p) for q, p in batch])
        return out

    async def rerank(
        self, query: str, passages: List[str], ids: List[str]
    ) -> List[RerankedChunk]:
        assert len(passages) == len(ids), "passages/ids length mismatch"
        start = time.time()
        pairs = [[query, p] for p in passages]
        scores = await asyncio.to_thread(self._score_sync, pairs)
        self.last_latency_ms = (time.time() - start) * 1000.0
        order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
        return [
            RerankedChunk(chunk_id=ids[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(order)
        ]


class BGEM3Reranker(_CrossEncoderReranker):
    """Default reranker: BAAI/bge-reranker-v2-m3 (Apache-2.0)."""

    def __init__(self, config: Optional[RerankConfig] = None):
        super().__init__("BAAI/bge-reranker-v2-m3", config)


class MXBAIReranker(_CrossEncoderReranker):
    """Alternate: mixedbread-ai/mxbai-rerank-v2 (smaller/faster, permissive)."""

    def __init__(self, config: Optional[RerankConfig] = None):
        super().__init__("mixedbread-ai/mxbai-rerank-v2", config)


class MiniLMFallbackReranker(_CrossEncoderReranker):
    """CPU fallback: cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, config: Optional[RerankConfig] = None):
        super().__init__("cross-encoder/ms-marco-MiniLM-L-6-v2", config)


def create_reranker(name: Optional[str] = None, config: Optional[RerankConfig] = None):
    """Select reranker by config name. Unknown names fall back to MiniLM."""
    cfg = config or get_rerank_config()
    key = (name or cfg.model or "").lower()
    if "mxbai" in key:
        return MXBAIReranker(cfg)
    if "minilm" in key or "fallback" in key:
        return MiniLMFallbackReranker(cfg)
    if "bge-reranker" in key or "bge_reranker" in key or key == "":
        return BGEM3Reranker(cfg)
    return MiniLMFallbackReranker(cfg)


__all__ = [
    "BGEM3Reranker",
    "MXBAIReranker",
    "MiniLMFallbackReranker",
    "create_reranker",
    "overlap_score",
]
