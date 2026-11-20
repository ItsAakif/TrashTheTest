"""S3 benchmark: retrieval cache cold vs warm.

Measures REAL local retrieval latency and cache hit/miss. Token savings are
ESTIMATES from cached context length (chars/4), not billed provider tokens.
Answer-cache generation is mocked and labeled as such.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from sqlalchemy import select

from database.engine import async_session_maker
from database.models import Project
from services.retrieval.evaluation import load_dataset, percentile
from services.retrieval.service import retrieval_service
from services.semantic_cache import semantic_cache


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/eval/dataset.json")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--out", default="tests/eval/cache_benchmark.json")
    args = parser.parse_args(argv)

    queries = load_dataset(args.dataset)
    payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    async with async_session_maker() as db:
        if args.project_id:
            project = await db.get(Project, args.project_id)
        else:
            project = (await db.execute(
                select(Project).where(Project.name == payload["project_title"]).limit(1)
            )).scalars().first()
        if project is None:
            print("project not found")
            return 2

        await semantic_cache.clear(db)
        cold_ms, warm_ms, hits, misses, tokens_avoided = [], [], 0, 0, 0
        rows = []
        for q in queries:
            t0 = time.time()
            first = await retrieval_service.retrieve_with_db(
                query=q.query, db=db, project_id=str(project.id), top_k=5
            )
            cold = (time.time() - t0) * 1000
            t0 = time.time()
            second = await retrieval_service.retrieve_with_db(
                query=q.query, db=db, project_id=str(project.id), top_k=5
            )
            warm = (time.time() - t0) * 1000
            hit = (second.meta or {}).get("cache") == "hit"
            cold_ms.append(cold)
            warm_ms.append(warm)
            hits += int(hit)
            misses += int(not hit)
            if hit:
                tokens_avoided += sum(len(c.content or "") // 4 for c in second.chunks)
            rows.append({
                "id": q.id, "query": q.query,
                "cold_ms": round(cold, 2), "warm_ms": round(warm, 2),
                "cache_hit": hit, "results": second.total_results,
            })

        stats = await semantic_cache.stats(db)

    out = {
        "provenance": "local Docker Postgres + Qwen3 embeddings; REAL retrieval timings, ESTIMATED tokens",
        "queries": len(rows),
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / max(1, len(rows)), 4),
        "cold_p50_ms": round(percentile(cold_ms, 50), 2),
        "cold_p95_ms": round(percentile(cold_ms, 95), 2),
        "warm_p50_ms": round(percentile(warm_ms, 50), 2),
        "warm_p95_ms": round(percentile(warm_ms, 95), 2),
        "median_speedup_x": round(statistics.median(cold_ms) / max(0.001, statistics.median(warm_ms)), 2),
        "estimated_tokens_avoided": tokens_avoided,
        "cache_stats": stats,
        "per_query": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_query", "cache_stats")}, indent=2))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
