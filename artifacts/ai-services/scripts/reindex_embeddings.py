"""Re-embed backfill: BGE-384 BYTEA -> Qwen3-1024 pgvector (T309).

Follows ADR-005 migration notes steps 3-5 (expand already done by 004):

1. Walk chunks in batches; compute content_hash (sha256, idempotent).
2. Check the embedding cache keyed by (model, content_hash); embed only
   misses with the new model (Qwen prefixes: passages at index time).
3. Write embedding_v (vector literal) + embedding_model + embedding_dim
   (1024) via raw SQL on Postgres; legacy BYTEA bytes via ORM elsewhere
   so SQLite/offline runs stay honest.
4. Dimension guard: verify live data matches the configured dimension.
5. Create/refresh the HNSW index AFTER backfill (build_vector_index).

Resumable: rerunning re-embeds only rows whose embedding_model is not
the configured model. Progress goes through the jobs table when
--job-id is given, else stdout. Safe to run repeatedly.

Usage:
    python -m scripts.reindex_embeddings --all [--batch-size 64] [--job-id UUID]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
import uuid

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 1024


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


async def _embed_texts(texts: list, provider) -> list:
    """Embed passages with the provider's passage arm (Qwen prefixes)."""
    if hasattr(provider, "embed_passages"):
        result = await provider.embed_passages(texts)
        return [list(e) for e in result.embeddings]
    if hasattr(provider, "generate_embeddings"):
        result = await provider.generate_embeddings(texts)
        return [list(e) for e in result.embeddings]
    out = []
    for text in texts:
        out.append(list(await provider.generate_embedding(text)))
    return out


async def backfill(batch_size: int = 64, job_id=None, provider=None) -> dict:
    from sqlalchemy import select, text as _text

    from database.engine import engine, async_session_maker
    from database.models import Chunk
    from config.settings import get_embedding_config

    cfg = get_embedding_config()
    model = getattr(cfg, "model", MODEL_ID)
    dim = int(getattr(cfg, "dimensions", DIMENSIONS))

    if provider is None:
        from factory import get_embedding as _get_embedding

        provider = _get_embedding()

    try:
        from database.models import JobStatus
        from database import repositories as _repos

        has_jobs = True
    except Exception:
        has_jobs = False

    async def report(progress: int, message: str):
        if job_id and has_jobs:
            try:
                async with async_session_maker() as session:
                    await _repos.update_job_status(
                        session, uuid.UUID(str(job_id)), JobStatus.PROCESSING,
                        progress=progress, stage_message=message,
                    )
                    await session.commit()
                return
            except Exception as exc:
                logger.warning("job progress update failed: %s", exc)
        print(f"[{progress}%] {message}", flush=True)

    # Snapshot pending chunk ids (resumable: rows missing the model tag,
    # plus rows whose vector column is still NULL on Postgres — the worker
    # tags the model on the legacy BYTEA write, so tag-only rows would
    # otherwise never get embedding_v. Chunk maps no embedding_v
    # attribute, hence the pg-guarded raw fragment (cf. get_indexing_state).
    from sqlalchemy import or_ as _or, text as _text

    _pending_filter = (Chunk.embedding_model.is_(None)) | (Chunk.embedding_model != model)
    try:
        _pg = engine.dialect.name == "postgresql"
    except Exception:
        _pg = False
    if _pg:
        _pending_filter = _or(
            _pending_filter, _text("chunks.embedding_v IS NULL"))
    # Snapshot pending chunk ids (resumable; see filter above).
    async with async_session_maker() as session:
        rows = (await session.execute(
            select(Chunk.id).where(_pending_filter).order_by(Chunk.created_at.asc())
        )).scalars().all()
        pending = [str(r) for r in rows]
    total = len(pending)
    await report(0, f"backfill start: {total} chunks pending model={model}")
    if not total:
        return {"total": 0, "embedded": 0, "cache_hits": 0, "model": model}

    cache: dict = {}
    try:
        from services.cache import get_cache_service as _cache

        cache_backend = _cache()
    except Exception:
        cache_backend = None

    def cache_get(chash: str):
        key = f"emb:{model}:{chash}"
        if key in cache:
            return cache[key]
        if cache_backend is not None:
            try:
                val = cache_backend.get(key)
                if isinstance(val, (bytes, bytearray, list)):
                    cache[key] = val
                    return val
            except Exception:
                pass
        return None

    def cache_set(chash: str, value):
        cache[f"emb:{model}:{chash}"] = value

    embedded = 0
    cache_hits = 0
    is_postgres = getattr(engine, "dialect", None) is not None and engine.dialect.name == "postgresql"

    for start in range(0, total, max(1, batch_size)):
        batch_ids = pending[start:start + max(1, batch_size)]
        async with async_session_maker() as session:
            chunks = [(await session.get(Chunk, uuid.UUID(cid))) for cid in batch_ids]
            chunks = [c for c in chunks if c is not None]
            texts, targets, hashes = [], [], []
            for chunk in chunks:
                chash = getattr(chunk, "content_hash", None) or content_hash(chunk.content)
                hit = cache_get(chash)
                # target: [chunk, vector_or_None, was_hit, content_hash]
                targets.append([chunk, hit, hit is not None, chash])
                if hit is None:
                    texts.append(chunk.content or "")
                    hashes.append(chash)
            if texts:
                vectors = await _embed_texts(texts, provider)
                vec_iter = iter(vectors)
                for target in targets:
                    if target[1] is None:
                        vec = next(vec_iter)
                        cache_set(target[3], vec)
                        target[1] = vec
            for chunk, vec, was_hit, chash in targets:
                if was_hit:
                    cache_hits += 1
                    if isinstance(vec, (bytes, bytearray)):
                        import struct as _struct

                        n = len(bytes(vec)) // 4
                        vec = list(_struct.unpack(f"{n}f", bytes(vec)))
                    else:
                        vec = [float(v) for v in vec]
                else:
                    embedded += 1
                chunk.content_hash = getattr(chunk, "content_hash", None) or chash
                chunk.embedding_model = model
                chunk.embedding_dim = dim
                if is_postgres:
                    literal = "[" + ",".join(repr(float(v)) for v in vec) + "]"
                    await session.execute(
                        _text("UPDATE chunks SET embedding_v = CAST(:vec AS vector(1024)) "
                              "WHERE id = :cid"),
                        {"vec": literal, "cid": str(chunk.id)},
                    )
                else:
                    import struct as _struct

                    chunk.embedding = _struct.pack(f"{len(vec)}f", *[float(v) for v in vec])
            await session.commit()
        done = min(start + max(1, batch_size), total)
        await report(int(done / total * 90), f"embedded {done}/{total}")

    # Dimension guard against live data (ADR-005 step 4).
    async with async_session_maker() as session:
        distinct = (await session.execute(
            _text("SELECT DISTINCT embedding_dim FROM chunks WHERE embedding_dim IS NOT NULL")
            if is_postgres else select(Chunk.embedding_dim).distinct()
        )).all()
        dims = {r[0] for r in distinct} if is_postgres else {r[0] for r in distinct if r[0]}
    dims.discard(None)
    if dims and dims != {dim}:
        raise RuntimeError(
            f"Dimension guard failed: DB has {sorted(dims)} but config is {dim}. "
            "Remediation: set EMBEDDING_DIMENSIONS to match, or re-embed."
        )

    # HNSW index after bulk load (ADR-005 step 5).
    if is_postgres:
        from scripts.build_vector_index import build_index

        await build_index(None, False)
    await report(100, f"backfill complete: {embedded} new, {cache_hits} cache hits")
    return {"total": total, "embedded": embedded, "cache_hits": cache_hits, "model": model}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Backfill Qwen3-1024 embeddings (T309)")
    parser.add_argument("--all", action="store_true", help="Backfill every pending chunk")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args(argv)
    if not args.all:
        print("refusing without --all (this walks the whole corpus)", file=sys.stderr)
        return 2
    try:
        out = asyncio.run(backfill(args.batch_size, args.job_id))
    except Exception as exc:
        logger.error("Backfill failed (live services required for real run): %s", exc)
        print(f"NOT RUN against live DB: {exc}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
