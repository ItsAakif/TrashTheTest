"""Reranker strategy interface (T302).

A reranker reorders fused retrieval candidates. Contract:
- output is a reordering of the input candidate ids (no drops, no inventions)
- never loaded at import time; never blocks the event loop (callers run
  it in a thread/executor)
- switchable off by config for A/B comparison in the eval harness
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class RerankedChunk:
    chunk_id: str
    score: float
    rank: int


@dataclass
class RerankConfig:
    """Rerank knobs. All from env with CPU-safe defaults (ADR-005 s6)."""

    enabled: bool = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    fallback_model: str = os.getenv(
        "RERANK_FALLBACK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    candidates: int = int(os.getenv("RERANK_CANDIDATES", "50"))
    top_k: int = int(os.getenv("RERANK_TOP_K", "8"))
    # Latency budget in ms for one rerank call; exceeding it selects the
    # fallback and reports so in the response metadata.
    budget_ms: float = float(os.getenv("RERANK_BUDGET_MS", "2000"))
    batch_size: int = int(os.getenv("RERANK_BATCH_SIZE", "16"))


def get_rerank_config() -> RerankConfig:
    return RerankConfig()


def memory_headroom_bytes() -> Optional[int]:
    """Best-effort remaining memory for this container (``None`` if unknown).

    cgroup v2 is used inside Docker/WSL2; cgroup v1 paths are kept for older
    kernels. Docker Desktop's WSL2 VM reports ``max`` at the container
    cgroup, so fall back to ``/proc/meminfo`` (MemAvailable), which still
    reflects real pressure across containers in the VM. A missing/limitless
    environment returns ``None`` and the caller keeps its default choice.
    """
    try:
        limit_raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current_raw = Path("/sys/fs/cgroup/memory.current").read_text().strip()
        if limit_raw and limit_raw != "max":
            return int(limit_raw) - int(current_raw)
    except Exception:
        pass
    try:
        limit = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        current = int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text())
        if limit < (1 << 60):  # v1 uses a huge sentinel for "unlimited"
            return limit - current
    except Exception:
        pass
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


@runtime_checkable
class RerankerStrategy(Protocol):
    """Cross-encoder reranker over (query, passage) pairs."""

    model_name: str

    async def rerank(
        self, query: str, passages: List[str], ids: List[str]
    ) -> List[RerankedChunk]:
        """Return ids reordered by relevance (same set, new order)."""
        ...


class NullReranker:
    """Config-off reranker: identity order, for A/B comparison (T302)."""

    model_name = "rerank-disabled"

    async def rerank(
        self, query: str, passages: List[str], ids: List[str]
    ) -> List[RerankedChunk]:
        return [RerankedChunk(chunk_id=cid, score=0.0, rank=i) for i, cid in enumerate(ids)]


__all__ = [
    "RerankerStrategy",
    "RerankedChunk",
    "RerankConfig",
    "NullReranker",
    "get_rerank_config",
    "memory_headroom_bytes",
]
