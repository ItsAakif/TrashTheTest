"""S2: reproducible retrieval evaluation across strategies.

Usage (inside the api container so DB + model are available):
    python -m scripts.eval_retrieval --dataset tests/eval/dataset.json

Runs dense, sparse, hybrid and hybrid+rerank over an anchor-labelled query
set, then prints a markdown table and writes JSON results next to the
dataset. Labels are synthetic-manual (see dataset provenance); this is a
local measurement, not a production benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from database.engine import async_session_maker
from database.models import Chunk, Document, Project
from services.retrieval.evaluation import (
    ConfigResult,
    evaluate_config,
    load_dataset,
    to_dict,
)
from services.retrieval.service import retrieval_service


CONFIGS = [
    ("dense", "dense", False),
    ("sparse", "sparse", False),
    ("hybrid", "hybrid", False),
    ("hybrid+rerank", "hybrid", True),
]


async def _resolve_project(db, title: str, project_id: str | None):
    if project_id:
        return await db.get(Project, project_id)
    row = (
        await db.execute(select(Project).where(Project.name == title).limit(1))
    ).scalars().first()
    return row


async def _load_chunks(db, project_id):
    rows = await db.execute(
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.project_id == project_id)
    )
    return list(rows.scalars().all())


def _markdown(results) -> str:
    lines = [
        "| Config | Recall@K | Precision@K | MRR | nDCG@K | Hit@K | P50 ms | P95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        a = r.aggregates
        lines.append(
            f"| {r.config} | {a['recall_at_k']:.3f} | {a['precision_at_k']:.3f} | "
            f"{a['mrr']:.3f} | {a['ndcg_at_k']:.3f} | {a['hit_rate_at_k']:.3f} | "
            f"{a['latency_p50_ms']:.1f} | {a['latency_p95_ms']:.1f} |"
        )
    return "\n".join(lines)


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/eval/dataset.json")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="tests/eval/results.json")
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    queries = load_dataset(str(dataset_path))
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    async with async_session_maker() as db:
        project = await _resolve_project(
            db, payload.get("project_title", ""), args.project_id
        )
        if project is None:
            print("evaluation project not found; pass --project-id", file=sys.stderr)
            return 2
        chunks = await _load_chunks(db, project.id)

        results = []
        for label, mode, rerank in CONFIGS:
            result = await evaluate_config(
                db,
                queries,
                str(project.id),
                mode=mode,
                rerank=rerank,
                top_k=args.top_k,
                chunks=chunks,
            )
            result.config = label
            results.append(result)
            print(f"[{label}] done", file=sys.stderr)

    table = _markdown(results)
    print(table)

    out = {
        "project": str(project.id),
        "project_title": payload.get("project_title"),
        "top_k": args.top_k,
        "provenance": payload.get("provenance"),
        "corpus_chunks": len(chunks),
        "results": [to_dict(r) for r in results],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
