"""Live check: run the real FUSED_SQL on real asyncpg with NULL UUID binds.

Reproduces the reported failure mode (all scope filters None, as in an
unscoped search) and proves the $6 typing error is gone. Uses a zero
1024-dim query vector; expects success (rows or empty), never
"could not determine data type of parameter".
"""
import asyncio
import os
import sys

sys.path.insert(0, "/app" if os.path.isdir("/app") else ".")

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main() -> int:
    from services.retrieval.hybrid import FUSED_SQL

    url = os.environ["DATABASE_URL"]
    engine = create_async_engine(url)
    vec = "[" + ",".join(["0.0"] * 1024) + "]"
    params = {
        "query_embedding": vec,
        "query_text": "test query",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "project_id": None,
        "document_id": None,
        "user_id": None,
        "per_arm": 5,
        "limit": 5,
    }
    async with engine.connect() as conn:
        # Filtered-query path also sets iterative scan first (repositories).
        try:
            await conn.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        except Exception as exc:
            print(f"NOTE iterative_scan not set: {exc}")
        rows = (await conn.execute(text(FUSED_SQL), params)).all()
        print(f"FUSED_SQL OK on live asyncpg: {len(rows)} fused rows, NULL uuid binds typed.")
    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
