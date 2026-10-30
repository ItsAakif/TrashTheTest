"""
Vector Retrieval Service (Phase 3, T301-T304).

Hybrid retrieval: pgvector cosine arm + tsvector lexical arm fused by RRF
in one SQL statement (services/retrieval/hybrid.py FUSED_SQL), with an
honest offline fallback (in-Python cosine + token-overlap + same RRF math)
for SQLite/test doubles. No full-scan-into-Python on Postgres: the DB
does the vector search via the HNSW index.

- Filtered queries enable iterative HNSW scans (repositories layer).
- Rows whose embedding_model differs from the configured model are
  excluded (no mixed vector spaces mid-reindex).
- Rerank (bge-reranker-v2-m3 top50->top8) is config-switchable for A/B.
- Every chunk carries page/section/score/source + citation provenance.
- Mid-reindex scopes return indexing_in_progress with real counts.
"""

import logging
import re
import time
import uuid
import struct
from typing import Optional, List, Tuple, Any, Dict

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .types import (
    RetrievalQuery,
    RetrievedChunk,
    RetrievalResult,
    RetrievalConfig,
    RetrievalStats,
    ResearchRetrievalResult,
    CitationPayload,
)
from .hybrid import rrf_fuse, normalize_query
from .citations import build_citation_from_chunk

logger = logging.getLogger(__name__)


def _get_embedding_provider():
    """Factory-backed embedding provider (Qwen3-0.6B default, ADR-005 s3)."""
    try:
        from factory import get_embedding

        return get_embedding()
    except Exception:
        from services.strategies.embedding.bge_small import BGESmallProvider

        return BGESmallProvider()


_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


class VectorRetrievalService:
    """
    Hybrid retrieval service: SQL vector + lexical arms fused by RRF.

    Keeps the RetrievalQuery/RetrievedChunk/RetrievalResult types and
    extends them with provenance, states, and rerank metadata.
    """

    def __init__(
        self,
        embedding_provider: Any = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.provider = embedding_provider  # lazy: resolved on first use
        self.config = config or RetrievalConfig()
        self._reranker = None

    def _get_provider(self):
        if self.provider is None:
            self.provider = _get_embedding_provider()
        return self.provider

    @property
    def embedding_model_name(self) -> str:
        provider = self._get_provider()
        return getattr(provider, "model_name", None) or getattr(
            provider, "_model_name", self.config.embedding_model
        )

    # -- query preprocessing ------------------------------------------------

    def _preprocess_query(self, query: str) -> Tuple[str, bool]:
        """Normalize query; returns (query, was_truncated)."""
        cleaned, truncated = normalize_query(query or "", self.config.query_max_length)
        cleaned = cleaned.lower()
        return cleaned, truncated

    # -- embedding conversions ----------------------------------------------

    def _bytes_to_embedding(self, embedding_bytes: bytes) -> List[float]:
        num_floats = len(embedding_bytes) // 4
        return list(struct.unpack(f"{num_floats}f", embedding_bytes))

    def _embedding_to_bytes(self, embedding: List[float]) -> bytes:
        return struct.pack(f"{len(embedding)}f", *embedding)

    def _embedding_to_literal(self, embedding: List[float]) -> str:
        return "[" + ",".join(repr(float(v)) for v in embedding) + "]"

    async def _generate_query_embedding(self, query: str) -> List[float]:
        provider = self._get_provider()
        # Qwen providers expose embed_queries (query prefix); legacy BGE
        # exposes generate_embedding only. Prefer the query arm.
        if hasattr(provider, "embed_queries"):
            result = await provider.embed_queries([query])
            return list(result.embeddings[0])
        emb = await provider.generate_embedding(query)
        return list(emb)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(n))
        ma = sum(v * v for v in a[:n]) ** 0.5
        mb = sum(v * v for v in b[:n]) ** 0.5
        if ma == 0 or mb == 0:
            return 0.0
        return dot / (ma * mb)

    # -- offline fallback arms (SQLite / test doubles) -----------------------

    async def _fallback_candidates(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        query_text: str,
        project_id=None,
        document_id=None,
        per_arm: int = 50,
    ) -> Tuple[List[Tuple[Any, str]], List[Tuple[Any, str]]]:
        """In-Python vector + lexical arms for non-Postgres binds.

        Loads only the filtered scope (project/document), never the whole
        table unfiltered when a scope is given. Excludes rows whose
        embedding_model mismatches the configured model.
        """
        from database.models import Chunk, Document

        if document_id:
            q = select(Chunk).where(Chunk.document_id == document_id)
        elif project_id:
            q = (
                select(Chunk)
                .join(Document, Chunk.document_id == Document.id)
                .where(Document.project_id == project_id)
            )
        else:
            q = select(Chunk)
        try:
            result = await db.execute(q)
            chunks = list(result.scalars().all())
        except Exception:
            return [], []

        model = self.config.embedding_model
        # Never mix vector spaces mid-reindex: exclude rows tagged with a
        # different embedding model from BOTH arms (T301).
        chunks = [c for c in chunks
                  if getattr(c, "embedding_model", None) in (None, model)]
        scored = []
        for chunk in chunks:
            raw = getattr(chunk, "embedding", None)
            if raw:
                try:
                    cemb = self._bytes_to_embedding(bytes(raw))
                    sim = self._cosine_similarity(query_embedding, cemb)
                except Exception:
                    sim = 0.0
            else:
                # No legacy bytes: token overlap as the relevance proxy so
                # offline fixtures still rank meaningfully.
                qt, ct = _tokens(query_text), _tokens(getattr(chunk, "content", ""))
                sim = (len(qt & ct) / len(qt)) if qt else 0.0
            scored.append((chunk, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        vector_arm = [(c, "vector") for c, _ in scored[:per_arm]]

        qtokens = _tokens(query_text)
        lex = []
        for chunk in chunks:
            ct = _tokens(getattr(chunk, "content", ""))
            s = (len(qtokens & ct) / len(qtokens)) if qtokens else 0.0
            lex.append((chunk, s))
        lex.sort(key=lambda t: t[1], reverse=True)
        lexical_arm = [(c, "lexical") for c, s in lex[:per_arm] if s > 0]
        return vector_arm, lexical_arm

    # -- main search ----------------------------------------------------------

    async def _hybrid_search(
        self,
        db: AsyncSession,
        query_text: str,
        query_embedding: List[float],
        project_id=None,
        document_id=None,
        user_id=None,
        per_arm: int = 50,
        limit: int = 5,
        offset: int = 0,
        min_similarity: float = 0.0,
        relaxed: bool = False,
    ) -> Tuple[List[Tuple[Any, float, str]], str]:
        """Run hybrid search; returns (rows, backend).

        backend is 'pgvector' when the DB did the vector search, else
        'offline-fallback'. Raises nothing: SQL failure degrades honestly.

        ``relaxed`` asks the SQL arm for the OR-joined lexical recovery
        (same scope, same fusion). It is only ever set by the caller
        after a strict pass over the same scope returned zero rows.
        """
        backend = "offline-fallback"
        try:
            from database import repositories

            rows = await repositories.search_similar_chunks_hybrid(
                db,
                query_embedding=query_embedding,
                query_text=query_text,
                project_id=project_id,
                document_id=document_id,
                user_id=user_id,
                embedding_model=self.config.embedding_model,
                per_arm=per_arm,
                limit=limit,
                offset=offset,
                min_similarity=min_similarity,
                relaxed=relaxed,
            )
            return rows, "pgvector"
        except RuntimeError:
            pass  # non-Postgres bind: offline fallback below
        except Exception:
            # SQL failure: roll the aborted statement back so later
            # statements on this session still run, then degrade
            # honestly to the offline fallback (flagged in meta).
            try:
                await db.rollback()
            except Exception:
                pass

        vector_arm, lexical_arm = await self._fallback_candidates(
            db, query_embedding, query_text, project_id, document_id, per_arm
        )
        vec_ids = [str(c.id) for c, _ in vector_arm]
        lex_ids = [str(c.id) for c, _ in lexical_arm]
        fused = rrf_fuse([vec_ids, lex_ids], k=self.config.rrf_k,
                         top_n=limit + offset)
        by_id = {}
        for c, arm in vector_arm:
            by_id.setdefault(str(c.id), [c, arm])
        for c, arm in lexical_arm:
            if str(c.id) in by_id and by_id[str(c.id)][1] != arm:
                by_id[str(c.id)][1] = "both"
            else:
                by_id.setdefault(str(c.id), [c, arm])
        rows = []
        for cid, score in fused[offset:]:
            entry = by_id.get(cid)
            if entry is None:
                continue
            if min_similarity and score < min_similarity:
                continue
            rows.append((entry[0], score, entry[1]))
        return rows, backend

    # -- rerank ----------------------------------------------------------------

    def _get_reranker(self, enable: bool):
        if not enable:
            return None
        if self._reranker is None:
            try:
                from services.strategies.rerank import create_reranker
                from services.strategies.rerank.bge_reranker import MiniLMFallbackReranker
                from services.strategies.rerank.base import memory_headroom_bytes

                reranker = create_reranker()
                # bge-reranker-v2-m3 needs ~2.4GB resident (2.1GB weights).
                # The default Qwen embedding already holds ~1.4GB, so in a
                # 4GB container the two cannot co-reside. The 2s rerank
                # budget cannot cancel the worker thread: a cold bge load
                # keeps allocating after the timeout and can OOM the API
                # mid-practice. Use the ~90MB MiniLM cross-encoder when the
                # headroom is clearly insufficient, and say so.
                headroom = memory_headroom_bytes()
                if (
                    headroom is not None
                    and headroom < 3_000_000_000
                    and not isinstance(reranker, MiniLMFallbackReranker)
                ):
                    logger.warning(
                        "Reranker %s needs ~2.4GB but only %d MB headroom is "
                        "available; using %s",
                        getattr(reranker, "model_name", "unknown"),
                        max(0, headroom) // (1024 * 1024),
                        MiniLMFallbackReranker().model_name,
                    )
                    reranker = MiniLMFallbackReranker()
                self._reranker = reranker
            except Exception:
                return None
        return self._reranker

    async def _maybe_rerank(
        self,
        query_text: str,
        rows: List[Tuple[Any, float, str]],
        enable: Optional[bool],
        top_k: int,
    ) -> Tuple[List[Tuple[Any, float, str]], dict]:
        """Rerank fused candidates; returns (rows, rerank_meta)."""
        use_rerank = self.config.enable_rerank if enable is None else bool(enable)
        meta: dict = {"rerank_applied": False, "reranker": None,
                      "rerank_latency_ms": None}
        if not use_rerank or len(rows) <= 1:
            return rows[:top_k], meta
        reranker = self._get_reranker(True)
        if reranker is None:
            return rows[:top_k], meta
        # Rerank the fused candidate window (default top-50 -> top-K).
        window = rows[: self.config.rerank_candidates]
        passages = [getattr(c, "content", "") or "" for c, _, _ in window]
        ids = [str(getattr(c, "id")) for c, _, _ in window]
        try:
            import asyncio as _asyncio
            import time as _t

            budget_s = max(0.5, float(self.config.rerank_budget_ms) / 1000.0)
            start = _t.time()
            try:
                # Bounded wait: a cold model download must never wedge the
                # request (Phase B live find: 10+ min hang on slow net).
                # Timeout keeps RRF order and reports applied=False.
                ranked = await _asyncio.wait_for(
                    reranker.rerank(query_text, passages, ids), timeout=budget_s)
            except (_asyncio.TimeoutError, TimeoutError):
                meta.update({"reranker": getattr(
                    reranker, "model_name",
                    "bge-reranker-v2-m3") + " (timeout-skipped)"})
                return rows[:top_k], meta
            latency_ms = (_t.time() - start) * 1000.0
            active = getattr(reranker, "model_name", "bge-reranker-v2-m3")
            if latency_ms > self.config.rerank_budget_ms:
                # Budget exceeded: fall back to the CPU model and say so.
                try:
                    from services.strategies.rerank import MiniLMFallbackReranker

                    fb = MiniLMFallbackReranker()
                    start = _t.time()
                    ranked = await _asyncio.wait_for(
                        fb.rerank(query_text, passages, ids), timeout=budget_s)
                    latency_ms = (_t.time() - start) * 1000.0
                    active = fb.model_name + " (budget-fallback)"
                except Exception:
                    pass
            order = {r.chunk_id: (r.rank, r.score) for r in ranked}
            reranked = sorted(window,
                              key=lambda t: order.get(str(getattr(t[0], "id")),
                                                      (len(window), 0.0))[0])
            out = [(c, float(order.get(str(getattr(c, "id")),
                                       (0, s))[1]), "reranked")
                   for c, s, _ in reranked[:top_k]]
            meta.update({"rerank_applied": True, "reranker": active,
                         "rerank_latency_ms": round(latency_ms, 2)})
            return out, meta
        except Exception:
            return rows[:top_k], meta

    # -- row shaping ------------------------------------------------------------

    def _to_retrieved_chunk(self, chunk: Any, score: float, arm: str) -> RetrievedChunk:
        meta = dict(getattr(chunk, "chunk_metadata", None) or {})
        page = getattr(chunk, "page_start", None)
        if page is None:
            page = meta.get("page")
        section_path = getattr(chunk, "section_path", None) or meta.get("section_path") or []
        section = "/".join(section_path) if section_path else meta.get("section")
        heading = getattr(chunk, "heading", None) or meta.get("heading")
        if section is None:
            section = heading
        citation = build_citation_from_chunk(chunk)
        payload = CitationPayload(
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            doc_id=citation.document_id,
            file_name=citation.file_name,
            page_number=citation.page_number,
            page=citation.page_number,
            section_path=citation.section_path,
            section="/".join(citation.section_path) if citation.section_path else section,
            heading=citation.heading,
            method=citation.method,
            confidence=citation.confidence,
            span=citation.span,
        )
        return RetrievedChunk(
            chunk_id=str(getattr(chunk, "id")),
            content=getattr(chunk, "content", "") or "",
            similarity=float(score),
            metadata=meta,
            document_id=str(getattr(chunk, "document_id")),
            page=page,
            section=section,
            score=float(score),
            source=arm,
            citation=payload,
        )

    # -- public API ---------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        project_id: Optional[str] = None,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> RetrievalResult:
        """Preprocess-only path (no DB): validates and normalizes."""
        start_time = time.time()
        if not query or not query.strip():
            return RetrievalResult(query=query or "", total_results=0, chunks=[],
                                   processing_time=0)
        cleaned, truncated = self._preprocess_query(query)
        top_k = min(top_k or self.config.default_top_k, self.config.max_top_k)
        return RetrievalResult(
            query=cleaned, total_results=0, chunks=[],
            processing_time=time.time() - start_time,
            truncated_query=truncated,
            meta={"tokens_estimate": max(1, len(cleaned) // 4)},
        )

    async def retrieve_with_db(
        self,
        query: str,
        db: AsyncSession,
        project_id: Optional[str] = None,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        enable_rerank: Optional[bool] = None,
        offset: int = 0,
    ) -> RetrievalResult:
        """Main retrieval entry point with DB session (T301-T304)."""
        start_time = time.time()
        if not query or not query.strip():
            return RetrievalResult(query=query or "", total_results=0, chunks=[],
                                   processing_time=0)
        cleaned, truncated = self._preprocess_query(query)
        top_k = min(top_k or self.config.default_top_k, self.config.max_top_k)
        min_similarity = (self.config.min_similarity_threshold
                          if min_similarity is None else min_similarity)

        try:
            project_uuid = uuid.UUID(project_id) if project_id else None
        except Exception:
            project_uuid = None
        try:
            doc_uuid = uuid.UUID(document_id) if document_id else None
        except Exception:
            doc_uuid = None
        try:
            user_uuid = uuid.UUID(user_id) if user_id else None
        except Exception:
            user_uuid = None

        # Honest indexing state first: mid-reindex returns counts, never an
        # empty list masquerading as "no relevant content".
        state, indexed, total = "ok", 0, 0
        try:
            from database import repositories

            info = await repositories.get_indexing_state(
                db, project_id=project_uuid, document_id=doc_uuid,
                embedding_model=self.config.embedding_model,
            )
            state = info["state"]
            indexed = info["indexed_chunks"]
            total = info["total_chunks"]
        except Exception:
            # A failed counts query aborts the PG transaction: roll back
            # so retrieval/generation below still run on a fresh one.
            try:
                await db.rollback()
            except Exception:
                pass

        embed_start = time.time()
        try:
            query_embedding = await self._generate_query_embedding(cleaned)
        except Exception:
            query_embedding = []
        query_embed_time = time.time() - embed_start

        retrieval_start = time.time()
        rows: List[Tuple[Any, float, str]] = []
        backend = "offline-fallback"
        # Honest recovery flag: set only when the strict pass found nothing
        # and the OR-joined lexical pass over the SAME scope did find rows.
        recovery: Optional[str] = None
        if query_embedding:
            rows, backend = await self._hybrid_search(
                db, cleaned, query_embedding, project_uuid, doc_uuid, user_uuid,
                per_arm=self.config.per_arm_candidates, limit=top_k,
                offset=offset, min_similarity=min_similarity,
            )
            scoped = (
                project_uuid is not None
                or doc_uuid is not None
                or user_uuid is not None
            )
            # A scoped search that fused to zero rows on Postgres is retried
            # once with the OR-joined lexical arm: plainto_tsquery is an
            # AND, so a natural-language question whose tokens never
            # co-occur in one chunk reports "no evidence" even though the
            # scope holds every token. Same scope, same fusion, still a
            # real tsvector match -- reported in meta, never silent.
            if not rows and backend == "pgvector" and scoped:
                relaxed_rows, relaxed_backend = await self._hybrid_search(
                    db, cleaned, query_embedding, project_uuid, doc_uuid,
                    user_uuid,
                    per_arm=self.config.per_arm_candidates, limit=top_k,
                    offset=offset, min_similarity=min_similarity,
                    relaxed=True,
                )
                if relaxed_rows:
                    rows, backend = relaxed_rows, relaxed_backend
                    recovery = "lexical_or"
        retrieval_time = time.time() - retrieval_start

        # Rerank fused candidates (switchable off for A/B).
        cand_k = top_k
        rows, rerank_meta = await self._maybe_rerank(cleaned, rows, enable_rerank, cand_k)

        chunks = [self._to_retrieved_chunk(c, s, arm) for c, s, arm in rows]
        # The corpus state is never masked by the presence of rows: a
        # mid-reindex scope reports indexing_in_progress WITH its indexed
        # rows, never an empty list masquerading as "no relevant content".

        meta: Dict[str, Any] = {
            "backend": backend,
            "fusion": "rrf",
            "tokens_estimate": max(1, len(cleaned) // 4),
            "embedding_model": self.config.embedding_model,
            "offset": offset,
        }
        if recovery:
            # Only present when a recovery actually produced the rows.
            meta["recovery"] = recovery

        return RetrievalResult(
            query=cleaned,
            total_results=len(chunks),
            chunks=chunks,
            processing_time=time.time() - start_time,
            query_embedding_time=query_embed_time,
            retrieval_time=retrieval_time,
            state=state,  # type: ignore[arg-type]
            indexed_chunks=indexed,
            total_chunks=total,
            rerank_applied=rerank_meta["rerank_applied"],
            reranker=rerank_meta["reranker"],
            rerank_latency_ms=rerank_meta["rerank_latency_ms"],
            truncated_query=truncated,
            meta=meta,
        )

    async def get_retrieval_stats(
        self,
        db: AsyncSession,
        project_id: str,
        embedding_model: Optional[str] = None,
    ) -> RetrievalStats:
        """Get retrieval statistics for a project"""
        project_uuid = uuid.UUID(project_id)
        from database.models import Chunk, Document
        from sqlalchemy import or_, text as _text

        result = await db.execute(
            select(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.project_id == project_uuid)
        )
        total_chunks = result.scalar() or 0

        # D1: same predicate as repositories.get_indexing_state. The live
        # backfill writes embedding_v via raw SQL and tags embedding_model
        # via ORM but never fills legacy BYTEA, so a configured model
        # counts by model tag alone; without one, Postgres counts legacy
        # BYTEA OR embedding_v (raw-SQL fragment: Chunk maps no
        # embedding_v attribute; pg-guarded so SQLite never sees it).
        model = embedding_model or self.config.embedding_model
        if model:
            emb_filter: Any = (Chunk.embedding_model == model)
        else:
            try:
                _bind = db.bind if hasattr(db, "bind") else None
                _pg = (_bind.dialect.name if _bind is not None else "") == "postgresql"
            except Exception:
                _pg = False
            if _pg:
                emb_filter = or_(
                    Chunk.embedding.isnot(None),
                    _text("chunks.embedding_v IS NOT NULL"),
                )
            else:
                emb_filter = Chunk.embedding.isnot(None)

        result = await db.execute(
            select(func.count(Chunk.id))
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Document.project_id == project_uuid,
                emb_filter,
            )
        )
        chunks_with_embeddings = result.scalar() or 0

        return RetrievalStats(
            project_id=project_id,
            total_chunks=total_chunks,
            chunks_with_embeddings=chunks_with_embeddings,
            average_similarity=0.0,
            top_retrieved_topics=[],
        )

    def prepare_for_research(
        self,
        retrieval_result: RetrievalResult,
    ) -> ResearchRetrievalResult:
        """Prepare retrieval results for research augmentation."""
        doc_ids = list(set(c.document_id for c in retrieval_result.chunks if c.document_id))

        context_parts = []
        for chunk in retrieval_result.chunks[:3]:
            context_parts.append(chunk.content[:200])

        context_summary = " | ".join(context_parts)

        return ResearchRetrievalResult(
            chunks=retrieval_result.chunks,
            context_summary=context_summary,
            source_document_ids=doc_ids,
        )


# ---------------------------------------------------------------------------
# STEP 3 per-topic evidence (academic hierarchical notes, additive).
#
# Per-topic retrieval takes ALL chunks relevant to one topic (heading /
# alias / token match over the fused candidate set), never a global top-k
# cut, and returns them in SOURCE order (page, position). Vector similarity
# ordering never determines the final note order: structure orders,
# retrieval only supplies evidence. The pure select_topic_evidence works on
# RetrievedChunk objects or plain dicts so it is unit-testable offline;
# retrieve_topic_evidence reuses retrieve_with_db (pgvector hybrid + rerank
# + cites) and only narrows to the topic afterwards.
# ---------------------------------------------------------------------------

_TOPIC_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Same stopword exclusion as the academic layer: glue words must never
# create relevance between unrelated chunks.
_TOPIC_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "are", "was", "were", "have", "has", "had", "which", "their", "there",
    "into", "over", "under", "such", "than", "then", "them", "they", "its",
    "of", "in", "to", "a", "an", "on", "at", "by", "as", "is", "it", "be",
    "or", "not", "but", "you", "we", "he", "she", "him", "her", "his",
    "will", "would", "can", "could", "should", "when", "while", "where",
})


def _topic_tokens(text: str) -> set:
    return {t for t in _TOPIC_TOKEN_RE.findall((text or "").lower())
            if t not in _TOPIC_STOPWORDS}


def _evidence_sort_key(chunk: Any) -> tuple:
    meta = dict(getattr(chunk, "metadata", None) or {})
    if isinstance(chunk, dict):
        page = chunk.get("page", meta.get("page"))
        position = chunk.get("position", meta.get("position", 0))
    else:
        page = getattr(chunk, "page", None)
        if page is None:
            page = meta.get("page_start", meta.get("page"))
        position = meta.get("position", 0)
    try:
        page_key = (0, int(page)) if page is not None else (1, 0)
    except (TypeError, ValueError):
        page_key = (1, 0)
    try:
        pos_key = int(position) if position is not None else 0
    except (TypeError, ValueError):
        pos_key = 0
    return (page_key[0], page_key[1], pos_key)


def _evidence_fields(chunk: Any) -> tuple:
    """(chunk_id, heading, text) for a RetrievedChunk or a plain dict."""
    if isinstance(chunk, dict):
        cid = str(chunk.get("chunk_id") or chunk.get("id") or "")
        heading = str(chunk.get("heading") or chunk.get("section") or "")
        text = str(chunk.get("content") or chunk.get("text") or "")
    else:
        cid = str(getattr(chunk, "chunk_id", "") or "")
        heading = str(getattr(chunk, "section", "") or "")
        text = str(getattr(chunk, "content", "") or "")
    return cid, heading, text


def select_topic_evidence(
    chunks: List[Any],
    topic_title: str,
    aliases: Optional[List[str]] = None,
    extra_titles: Optional[List[str]] = None,
    min_overlap: float = 0.2,
) -> List[Any]:
    """Filter fused candidates down to one topic's evidence (pure).

    Keeps a chunk when its id is structurally claimed is handled
    caller-side; here relevance is heading/alias/token match. Returns ALL
    matches in source order -- no top-k truncation. Empty only when nothing
    matches (callers fall back to the structure's own chunk ids).
    """
    names = {(topic_title or "").strip().lower()}
    for extra in extra_titles or []:
        if (extra or "").strip():
            names.add(extra.strip().lower())
    vocab = _topic_tokens(topic_title)
    for extra in extra_titles or []:
        vocab |= _topic_tokens(extra or "")
    alias_sets = [_topic_tokens(a or "") for a in (aliases or [])]
    alias_sets = [a for a in alias_sets if a]
    hits: List[Any] = []
    for chunk in chunks or []:
        _, heading, text = _evidence_fields(chunk)
        h = heading.strip().lower()
        if h and (h in names or any(
                h in n or n in h for n in names if n)):
            hits.append(chunk)
            continue
        ctokens = _topic_tokens(text)
        if alias_sets and any(
                len(a & ctokens) / max(1, len(a)) >= 0.5 for a in alias_sets):
            hits.append(chunk)
            continue
        if vocab and ctokens and len(vocab & ctokens) / len(vocab) >= min_overlap:
            hits.append(chunk)
    hits.sort(key=_evidence_sort_key)
    return hits


async def retrieve_topic_evidence(
    self,
    topic_title: str,
    db: AsyncSession,
    project_id: Optional[str] = None,
    document_id: Optional[str] = None,
    user_id: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    extra_titles: Optional[List[str]] = None,
    min_overlap: float = 0.2,
    enable_rerank: Optional[bool] = None,
) -> RetrievalResult:
    """Per-topic evidence retrieval reusing the hybrid path (T301-T304).

    Runs retrieve_with_db over a wide candidate window (per-arm pool, no
    global top-10 cut), then narrows to the topic with
    select_topic_evidence and restores source order. Rerank/cites/meta
    behaviour is unchanged from the shared path.
    """
    query = " -- ".join([t for t in [topic_title] + list(aliases or []) if t])
    wide = await self.retrieve_with_db(
        query or topic_title,
        db,
        project_id=project_id,
        document_id=document_id,
        user_id=user_id,
        top_k=self.config.max_top_k,
        enable_rerank=enable_rerank,
    )
    evidence = select_topic_evidence(
        wide.chunks, topic_title, aliases, extra_titles, min_overlap,
    )
    return RetrievalResult(
        query=wide.query,
        total_results=len(evidence),
        chunks=evidence,
        processing_time=wide.processing_time,
        query_embedding_time=wide.query_embedding_time,
        retrieval_time=wide.retrieval_time,
        state=wide.state,
        indexed_chunks=wide.indexed_chunks,
        total_chunks=wide.total_chunks,
        rerank_applied=wide.rerank_applied,
        reranker=wide.reranker,
        rerank_latency_ms=wide.rerank_latency_ms,
        truncated_query=wide.truncated_query,
        meta={**(wide.meta or {}), "topic_scoped": True,
              "candidates_considered": wide.total_results},
    )


# Bind as a method without touching the class body above.
VectorRetrievalService.retrieve_topic_evidence = retrieve_topic_evidence  # type: ignore[attr-defined]


# Default service instance
retrieval_service = VectorRetrievalService()
