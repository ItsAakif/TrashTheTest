"""Best-effort retrieval tracing (S4). Never fails the request."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def record_trace(
    *,
    user_id: Any = None,
    project_id: Any = None,
    document_id: Any = None,
    query_text: str = "",
    retrieval_mode: str = "",
    rerank: bool = False,
    chunk_ids: Optional[list] = None,
    scores: Optional[list] = None,
    cache_layer: Optional[str] = None,
    cache_hit: bool = False,
    latency_ms: float = 0.0,
    rerank_latency_ms: float = 0.0,
    context_tokens_estimate: int = 0,
) -> None:
    try:
        from database.engine import async_session_maker
        from database.models import RetrievalTrace

        async with async_session_maker() as session:
            session.add(RetrievalTrace(
                id=uuid.uuid4(),
                request_id=None,
                user_id=user_id,
                project_id=project_id,
                document_id=document_id,
                query_text=query_text[:2000],
                retrieval_mode=retrieval_mode,
                rerank=1 if rerank else 0,
                chunk_ids=chunk_ids or [],
                scores=scores or [],
                cache_layer=cache_layer,
                cache_hit=1 if cache_hit else 0,
                latency_ms=float(latency_ms),
                rerank_latency_ms=float(rerank_latency_ms or 0.0),
                context_tokens_estimate=int(context_tokens_estimate or 0),
            ))
            await session.commit()
    except Exception as exc:
        logger.debug("retrieval trace not persisted: %s", exc)
