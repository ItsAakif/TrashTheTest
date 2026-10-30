"""Qwen3-Embedding-0.6B provider (T206, ADR-005 section 3).

1024d multilingual default (Apache-2.0, 32k context, instruction-aware).
Query/passage prefixes are kept distinct by construction: embed_passages()
at index time, embed_queries() at retrieval time - a mismatch is a silent
quality loss, so there is exactly one method for each and no shared flag
to configure wrongly. Batches are bounded; the corpus is never held whole
in memory. Model loads lazily (CPU default) so import is cheap.

CPU/concurrency notes: model init (HF download + torch load) and
model.encode() are synchronous native work. Both run via asyncio.to_thread
so the worker event loop stays responsive while weights download or the
CPU encodes (native torch ops release the GIL). The download cache lives
under HF_HOME (persisted via the modelcache volume in docker-compose).
"""

import asyncio
import os
import time
from typing import List, Optional

from ..base import EmbeddingResult, EmbeddingStrategy
from config.settings import get_embedding_config

QUERY_PREFIX = "Instruct: Given a question, retrieve relevant passages\nQuery: "
PASSAGE_PREFIX = "Instruct: Represent this passage for retrieval\nPassage: "
MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 1024


class Qwen3EmbeddingProvider(EmbeddingStrategy):
    def __init__(self):
        config = get_embedding_config()
        self._model_name = config.model or MODEL_ID
        self._dimensions = int(config.dimensions or DIMENSIONS)
        self._device = config.device or "cpu"
        self._normalize = config.normalize_embeddings
        self._model = None

    async def _ensure_model(self):
        if self._model is None:
            def _load():
                from sentence_transformers import SentenceTransformer

                cache = (
                    os.getenv("SENTENCE_TRANSFORMERS_HOME")
                    or os.getenv("HF_HOME")
                    or None
                )
                kwargs = {"device": self._device}
                if cache:
                    kwargs["cache_folder"] = cache
                return SentenceTransformer(self._model_name, **kwargs)

            # Blocking (minutes on first run: HF weight download + torch
            # load). Off the event loop so concurrent jobs keep reporting.
            self._model = await asyncio.to_thread(_load)

    async def warm(self) -> None:
        """Load weights once (first run downloads). Reports nothing itself;
        the caller owns the honest 'downloading weights' progress message."""
        await self._ensure_model()

    def _prefix(self, texts: List[str], kind: str) -> List[str]:
        prefix = QUERY_PREFIX if kind == "query" else PASSAGE_PREFIX
        return [f"{prefix}{t}" for t in texts]

    async def _encode(self, texts: List[str], batch_size: int, kind: str) -> EmbeddingResult:
        await self._ensure_model()
        start = time.time()
        prefixed = self._prefix(texts, kind)
        out: List[List[float]] = []
        model = self._model
        for i in range(0, len(prefixed), max(1, batch_size)):
            batch = prefixed[i:i + max(1, batch_size)]
            # Sync torch encode off the event loop (CPU-bound native work).
            embs = await asyncio.to_thread(
                lambda b=batch: model.encode(b, normalize_embeddings=self._normalize,
                                             convert_to_numpy=True, show_progress_bar=False)
            )
            out.extend([list(map(float, row)) for row in embs.tolist()])
        return EmbeddingResult(embeddings=out, model=self._model_name,
                               dimensions=self._dimensions, processing_time=time.time() - start)

    async def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        # Default entry point is the PASSAGE arm (index time).
        return await self._encode(texts, batch_size, kind="passage")

    async def embed_passages(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        return await self._encode(texts, batch_size, kind="passage")

    async def embed_queries(self, texts: List[str], batch_size: int = 32) -> EmbeddingResult:
        return await self._encode(texts, batch_size, kind="query")

    async def generate_embedding(self, text: str) -> List[float]:
        return (await self._encode([text], 1, kind="passage")).embeddings[0]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value: int):
        self._dimensions = value

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        self._model_name = value


def create_embedding_provider(provider_name: Optional[str] = None):
    from .bge_small import BGESmallProvider

    try:
        from config.settings import get_embedding_config as _cfg

        provider = provider_name or _cfg().provider
    except Exception:
        provider = provider_name or "qwen3-embedding"
    if provider in ("bge-small-legacy", "bge-small", "e5-small"):
        return BGESmallProvider()
    return Qwen3EmbeddingProvider()


__all__ = ["Qwen3EmbeddingProvider", "QUERY_PREFIX", "PASSAGE_PREFIX",
           "MODEL_ID", "DIMENSIONS", "create_embedding_provider"]
