# Benchmark Results

All numbers below were measured locally on 2026-11-20. Simulated or estimated
values are labelled; nothing here is a billed-provider measurement.

## Environment

- Docker Desktop, Linux containers, 16 CPUs reported, 3.82 GB RAM.
- API and worker in the same compose stack; Postgres 16 + pgvector.
- Embedding: Qwen3-Embedding-0.6B, 1024d, CPU (bfloat16).
- Reranker: bge-reranker-v2-m3 unavailable due to memory headroom; MiniLM
  cross-encoder fallback used (~90 MB).
- Eval corpus: 3 documents, 6 live chunks, 30 hand-labelled queries.
- `top_k = 5`, hybrid weights 1.0/1.0, RRF k=60.

## Retrieval strategies

| Config | Recall@5 | Precision@5 | MRR | nDCG@5 | Hit@5 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|
| dense | 1.000 | 0.213 | 0.933 | 0.951 | 1.000 | 1094.5 | 1274.1 |
| sparse | 0.983 | 0.207 | 1.000 | 0.987 | 1.000 | 1038.5 | 1170.2 |
| hybrid | 1.000 | 0.213 | 0.983 | 0.988 | 1.000 | 1033.2 | 1223.3 |
| hybrid+rerank | 1.000 | 0.213 | 0.983 | 0.988 | 1.000 | 1202.4 | 1412.6 |

These are real local measurements. They are not a claim that hybrid or rerank
is universally better; on this small corpus sparse alone is strongest.

## Semantic cache: repeated-query benchmark

Method: clear cache, run each of the 30 golden queries twice through the real
retrieval path, record cold vs warm latency. Warm hits are exact-query repeats
(same normalised query), not paraphrases.

| Metric | Value |
|---|---|
| Queries | 30 |
| Warm cache hits | 30 / 30 |
| Cold P50 / P95 | 1287.65 ms / 1443.78 ms |
| Warm P50 / P95 | 517.44 ms / 626.51 ms |
| Median speed-up | 2.48x |
| Estimated tokens avoided | 20,025 (chars/4 of cached context) |

Token savings are **estimates**, not provider-billed usage. The 100% hit rate
is expected because the second call repeats the same query; it is not a
semantic hit-rate claim.

## Semantic threshold calibration

Manually authored pairs (5 equivalent, 5 near-miss), Qwen3-Embedding-0.6B:

| Threshold | Equivalent hit rate | Near-miss false-hit rate |
|---|---|---|
| 0.70 | 0.40 | 0.40 |
| 0.75 | 0.40 | 0.20 |
| 0.80 | 0.40 | 0.00 |
| 0.85 | 0.40 | 0.00 |
| 0.90 | 0.20 | 0.00 |
| 0.95 | 0.00 | 0.00 |

Chosen 0.80: zero false hits, conservative reuse. The low equivalent-pair hit
rate is a real limitation of similarity-only matching, not a bug.

## What is NOT benchmarked

- Answer-cache hit rate on paraphrases in production traffic.
- Billed token/cost savings from a real provider.
- Large corpora (>1k chunks), concurrency, or multi-tenant load.
- The 1,197 historical chunks that still lack embeddings.
