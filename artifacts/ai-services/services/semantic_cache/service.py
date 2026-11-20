"""S3 semantic cache: identity-scoped query/retrieval/answer reuse.

Design:
- Postgres is the source of truth and the semantic index (pgvector HNSW).
- Redis (when configured) is a best-effort exact-key hot path only.
- Every entry is scoped by a SHA-256 identity that includes user, project,
  document versions, embedding model, LLM model, prompt version, retrieval
  config hash and response format. A change to any of those misses.
- TTL is enforced at read time; expired rows are ignored and opportunistically
  deleted.
- Failures degrade to a cache miss; they never fail the RAG request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import delete, select, text, update

from config.settings import get_settings
from database.models import SemanticCacheEntry

logger = logging.getLogger(__name__)

PROMPT_VERSION = "tutor-v1"


def _norm(text_value: str) -> str:
    import re
    import unicodedata

    t = unicodedata.normalize("NFKC", text_value or "")
    return re.sub(r"\s+", " ", t).strip().casefold()


def query_hash(query: str) -> str:
    return hashlib.sha256(_norm(query).encode("utf-8")).hexdigest()


def identity_key(
    *,
    user_id: Any,
    project_id: Any,
    doc_versions: Optional[Iterable] = None,
    embedding_model: str = "",
    llm_model: str = "",
    prompt_version: str = PROMPT_VERSION,
    retrieval_config_hash: str = "",
    response_format: str = "tutor-v1",
) -> str:
    payload = {
        "user": str(user_id or ""),
        "project": str(project_id or ""),
        "docs": sorted(str(v) for v in (doc_versions or [])),
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "retrieval_config": retrieval_config_hash,
        "response_format": response_format,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def retrieval_config_hash(**config: Any) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:32]


async def doc_versions_for_project(db: Any, project_id: Any) -> List[str]:
    from database.models import Document

    try:
        rows = (await db.execute(
            select(Document.id, Document.version).where(Document.project_id == project_id)
        )).all()
        return sorted(f"{r[0]}:{getattr(r, 'version', None) or 1}" for r in rows)
    except Exception:
        return []


class SemanticCacheService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.enabled = bool(getattr(self.settings, "semantic_cache_enabled", True))
        self.threshold = float(getattr(self.settings, "semantic_cache_threshold", 0.92))
        self.ttl_seconds = int(getattr(self.settings, "semantic_cache_ttl_seconds", 3600))
        self.max_entries = int(getattr(self.settings, "semantic_cache_max_entries", 5000))

    # -- embeddings ---------------------------------------------------------
    async def _embed(self, query: str) -> Optional[List[float]]:
        try:
            from services.retrieval.service import retrieval_service

            return await retrieval_service._generate_query_embedding(query)
        except Exception as exc:
            logger.debug("semantic cache embedding failed: %s", exc)
            return None

    @staticmethod
    def _vec_literal(vec: List[float]) -> str:
        return "[" + ",".join(repr(float(v)) for v in vec) + "]"

    @staticmethod
    def _is_postgres(db: Any) -> bool:
        try:
            name = getattr(getattr(getattr(db, "bind", None), "dialect", None), "name", None)
            return name == "postgresql"
        except Exception:
            return False

    # -- redis exact hot path (best effort) ---------------------------------
    async def _redis(self):
        try:
            from redis import asyncio as aioredis

            return aioredis.from_url(str(self.settings.redis_url), decode_responses=True)
        except Exception:
            return None

    async def _redis_get(self, key: str) -> Optional[dict]:
        client = await self._redis()
        if client is None:
            return None
        try:
            raw = await client.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            return None
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    async def _redis_set(self, key: str, value: dict, ttl: int) -> None:
        client = await self._redis()
        if client is None:
            return
        try:
            await client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    # -- lookup -------------------------------------------------------------
    async def lookup(
        self,
        db: Any,
        *,
        layer: str,
        query: str,
        identity: str,
        embedding: Optional[List[float]] = None,
        threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        try:
            qhash = query_hash(query)
            hot_key = f"semcache:{identity}:{layer}:{qhash}"
            hot = await self._redis_get(hot_key)
            if hot is not None:
                return {**hot, "cache_hit": True, "cache_layer": layer}

            now = datetime.utcnow()
            row = (await db.execute(
                select(SemanticCacheEntry).where(
                    SemanticCacheEntry.identity_key == identity,
                    SemanticCacheEntry.layer == layer,
                    SemanticCacheEntry.query_hash == qhash,
                    SemanticCacheEntry.expires_at > now,
                ).limit(1)
            )).scalars().first()
            similarity = 1.0
            if row is None:
                vec = embedding if embedding is not None else await self._embed(query)
                if vec and self._is_postgres(db):
                    limit = threshold if threshold is not None else self.threshold
                    found = (await db.execute(
                        text(
                            "SELECT id, response, citations, "
                            "1 - (query_embedding <=> CAST(:vec AS vector(1024))) AS similarity "
                            "FROM semantic_cache_entries "
                            "WHERE identity_key = :identity AND layer = :layer "
                            "AND expires_at > now() AND query_embedding IS NOT NULL "
                            "ORDER BY query_embedding <=> CAST(:vec AS vector(1024)) "
                            "LIMIT 1"
                        ),
                        {"vec": self._vec_literal(vec), "identity": identity, "layer": layer},
                    )).first()
                    if found is not None and float(found[3] or 0.0) >= limit:
                        row = await db.get(SemanticCacheEntry, found[0])
                        similarity = float(found[3] or 0.0)
            if row is None:
                return None

            await db.execute(
                update(SemanticCacheEntry)
                .where(SemanticCacheEntry.id == row.id)
                .values(hits=SemanticCacheEntry.hits + 1, last_hit_at=now)
            )
            await db.commit()
            payload = {
                "response": row.response,
                "citations": row.citations or [],
                "document_ids": row.document_ids or [],
                "doc_versions": row.doc_versions or [],
                "similarity": round(similarity, 4),
                "cache_layer": layer,
                "cache_hit": True,
            }
            await self._redis_set(hot_key, payload, 300)
            return payload
        except Exception as exc:
            logger.warning("semantic cache lookup failed: %s", exc)
            return None

    # -- store --------------------------------------------------------------
    async def store(
        self,
        db: Any,
        *,
        layer: str,
        query: str,
        identity: str,
        response: dict,
        citations: Optional[list] = None,
        document_ids: Optional[list] = None,
        doc_versions: Optional[list] = None,
        project_id: Any = None,
        user_id: Any = None,
        embedding_model: str = "",
        llm_model: str = "",
        prompt_version: str = PROMPT_VERSION,
        retrieval_config_hash_value: str = "",
        response_format: str = "tutor-v1",
        ttl_seconds: Optional[int] = None,
    ) -> bool:
        if not self.enabled:
            return False
        try:
            qhash = query_hash(query)
            ttl = int(ttl_seconds or self.ttl_seconds)
            now = datetime.utcnow()
            expires = now + timedelta(seconds=ttl)
            vec = await self._embed(query)

            existing = (await db.execute(
                select(SemanticCacheEntry).where(
                    SemanticCacheEntry.identity_key == identity,
                    SemanticCacheEntry.layer == layer,
                    SemanticCacheEntry.query_hash == qhash,
                ).limit(1)
            )).scalars().first()
            if existing is None:
                entry = SemanticCacheEntry(
                    identity_key=identity,
                    layer=layer,
                    query_hash=qhash,
                    query_text=query[:4000],
                    response=response,
                    citations=citations or [],
                    document_ids=document_ids or [],
                    doc_versions=doc_versions or [],
                    project_id=project_id,
                    user_id=user_id,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                    prompt_version=prompt_version,
                    retrieval_config_hash=retrieval_config_hash_value,
                    response_format=response_format,
                    created_at=now,
                    expires_at=expires,
                )
                db.add(entry)
                await db.flush()
                if vec and self._is_postgres(db):
                    await db.execute(
                        text("UPDATE semantic_cache_entries "
                             "SET query_embedding = CAST(:vec AS vector(1024)) WHERE id = :id"),
                        {"vec": self._vec_literal(vec), "id": str(entry.id)},
                    )
            else:
                await db.execute(
                    update(SemanticCacheEntry)
                    .where(SemanticCacheEntry.id == existing.id)
                    .values(response=response, citations=citations or [],
                            expires_at=expires, hits=0)
                )
            await db.commit()
            await self._prune(db)
            await self._redis_set(
                f"semcache:{identity}:{layer}:{qhash}",
                {"response": response, "citations": citations or [],
                 "document_ids": document_ids or [], "doc_versions": doc_versions or [],
                 "similarity": 1.0},
                300,
            )
            return True
        except Exception as exc:
            logger.warning("semantic cache store failed: %s", exc)
            try:
                await db.rollback()
            except Exception:
                pass
            return False

    async def _prune(self, db: Any) -> None:
        try:
            count = (await db.execute(
                select(SemanticCacheEntry.id).where(SemanticCacheEntry.expires_at <= datetime.utcnow())
            )).all()
            if count:
                await db.execute(delete(SemanticCacheEntry).where(
                    SemanticCacheEntry.expires_at <= datetime.utcnow()))
                await db.commit()
        except Exception:
            pass

    # -- invalidation -------------------------------------------------------
    async def invalidate(
        self,
        db: Any,
        *,
        project_id: Any = None,
        document_id: Any = None,
        user_id: Any = None,
    ) -> int:
        try:
            total = 0
            if project_id:
                res = await db.execute(delete(SemanticCacheEntry).where(
                    SemanticCacheEntry.project_id == project_id))
                total += int(getattr(res, "rowcount", 0) or 0)
            if user_id:
                res = await db.execute(delete(SemanticCacheEntry).where(
                    SemanticCacheEntry.user_id == user_id))
                total += int(getattr(res, "rowcount", 0) or 0)
            if document_id:
                if self._is_postgres(db):
                    res = await db.execute(text(
                        "DELETE FROM semantic_cache_entries "
                        "WHERE document_ids @> CAST(:doc AS jsonb)"
                    ), {"doc": json.dumps([str(document_id)])})
                    total += int(getattr(res, "rowcount", 0) or 0)
                else:
                    rows = (await db.execute(select(SemanticCacheEntry))).scalars().all()
                    for row in rows:
                        if str(document_id) in [str(d) for d in (row.document_ids or [])]:
                            await db.delete(row)
                            total += 1
            await db.commit()
            return total
        except Exception as exc:
            logger.warning("semantic cache invalidation failed: %s", exc)
            return 0

    async def clear(self, db: Any) -> int:
        try:
            res = await db.execute(delete(SemanticCacheEntry))
            await db.commit()
            return int(getattr(res, "rowcount", 0) or 0)
        except Exception:
            return 0

    async def stats(self, db: Any) -> Dict[str, Any]:
        try:
            rows = (await db.execute(
                select(SemanticCacheEntry.layer, SemanticCacheEntry.hits)
            )).all()
            by_layer: Dict[str, Dict[str, int]] = {}
            for layer, hits in rows:
                agg = by_layer.setdefault(layer, {"entries": 0, "hits": 0})
                agg["entries"] += 1
                agg["hits"] += int(hits or 0)
            return {
                "enabled": self.enabled,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl_seconds,
                "layers": by_layer,
                "total_entries": sum(v["entries"] for v in by_layer.values()),
                "total_hits": sum(v["hits"] for v in by_layer.values()),
            }
        except Exception as exc:
            return {"enabled": self.enabled, "error": str(exc), "layers": {}}


semantic_cache = SemanticCacheService()

__all__ = [
    "SemanticCacheService",
    "semantic_cache",
    "identity_key",
    "query_hash",
    "retrieval_config_hash",
    "doc_versions_for_project",
    "PROMPT_VERSION",
]
