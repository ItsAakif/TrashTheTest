"""Build the HNSW vector index after bulk load (T208).

Idempotent: CREATE INDEX IF NOT EXISTS. Uses CONCURRENTLY in production
mode (--prod) so ingestion is not blocked; plain CREATE INDEX in dev/test.
Before this script runs, retrieval works as an exact scan and must report
unindexed state (see services/retrieval). Run per project/document set::

    python -m scripts.build_vector_index --project <project-id> [--prod]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid

logger = logging.getLogger(__name__)

INDEX_NAME = "ix_chunks_embedding_v_hnsw"


async def build_index(project_id: str | None = None, use_concurrently: bool = False) -> dict:
    from sqlalchemy import text as _text

    from database.engine import engine

    concurrently = "CONCURRENTLY" if use_concurrently else ""
    # NOTE: pgvector HNSW limit is 2000 dims; 1024 (Qwen3-0.6B) is inside it.
    stmt = (
        f"CREATE INDEX {concurrently} IF NOT EXISTS {INDEX_NAME} "
        "ON chunks USING hnsw (embedding_v vector_cosine_ops)"
    )
    where = ""
    async with engine.connect() as conn:
        if project_id:
            # Project-scoped honesty check: refuse when nothing is indexed yet.
            missing = (await conn.execute(_text(
                "SELECT count(*) FROM chunks JOIN documents ON documents.id = chunks.document_id "
                "WHERE documents.project_id = :pid AND chunks.embedding_v IS NULL"),
                {"pid": str(project_id)}))
            n_missing = missing.scalar() or 0
            if n_missing:
                logger.warning("%s chunks still lack embeddings; index will cover completed rows only",
                               n_missing)
        await conn.execute(_text(stmt))
        await conn.commit()
    return {"index": INDEX_NAME, "concurrently": use_concurrently, "project_id": project_id}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build HNSW index after bulk load (T208)")
    parser.add_argument("--project", default=None, help="Project UUID (honesty check scope)")
    parser.add_argument("--prod", action="store_true", help="Use CREATE INDEX CONCURRENTLY")
    args = parser.parse_args(argv)
    if args.project:
        uuid.UUID(args.project)  # fail fast on malformed id
    try:
        out = asyncio.run(build_index(args.project, args.prod))
    except Exception as exc:
        logger.error("Index build failed (live Postgres required): %s", exc)
        print(f"NOT RUN against live DB: {exc}", file=sys.stderr)
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
