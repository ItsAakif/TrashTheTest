# Retrieval Evaluation

## What is measured

Four configurations over the same corpus and queries:

1. `dense` - pgvector only.
2. `sparse` - PostgreSQL full-text only.
3. `hybrid` - weighted RRF over both arms.
4. `hybrid+rerank` - hybrid followed by the cross-encoder (MiniLM fallback
   under memory pressure).

Metrics: Recall@K, Precision@K, MRR, nDCG@K, Hit@K, and retrieval latency
(P50/P95). K = 5.

## Dataset

- Source: the repository's existing hand-labelled golden set
  (`tests/eval/golden_queries.json`, T308): 30 queries, 36 labelled chunks
  across biology, linear algebra and French Revolution documents.
- The three documents are concatenated into `tests/eval/golden_docs/`,
  ingested through the normal pipeline into the `RAG Eval Golden` project, and
  chunked by the real chunker (6 live chunks).
- Labels are mapped from expected chunk **text** to live chunk ids via
  normalised anchor containment. No labels were invented for this run.
- Provenance: hand-labelled offline golden set; **not independently
  re-validated**, and the live mapping can mark a merged chunk relevant to
  more than one query.

## Reproduce

```bash
# inside the api container (DB + model available)
python -m scripts.setup_eval_project --title "RAG Eval Golden" --dir tests/eval/golden_docs
python -m scripts.eval_retrieval --dataset tests/eval/dataset_golden.json   --out tests/eval/results_golden.json
```

## Results (local, 2026-11-20)

| Config | Recall@5 | Precision@5 | MRR | nDCG@5 | Hit@5 | P50 ms | P95 ms |
|---|---|---|---|---|---|---|---|
| dense | 1.000 | 0.213 | 0.933 | 0.951 | 1.000 | 1094.5 | 1274.1 |
| sparse | 0.983 | 0.207 | 1.000 | 0.987 | 1.000 | 1038.5 | 1170.2 |
| hybrid | 1.000 | 0.213 | 0.983 | 0.988 | 1.000 | 1033.2 | 1223.3 |
| hybrid+rerank | 1.000 | 0.213 | 0.983 | 0.988 | 1.000 | 1202.4 | 1412.6 |

Interpretation:
- The corpus is small (6 chunks) and top-5 is generous, so recall saturates.
  Precision is bounded by the number of labelled chunks per query.
- Sparse ranking is strongest on this set (MRR 1.000), which is expected:
  many golden labels are exact-term queries.
- Reranking adds ~170 ms P50 and does not improve MRR/nDCG here; it is kept
  for harder corpora but is not justified by this measurement.
- Dense alone misses some rank-1 answers (MRR 0.933), which is exactly the
  case hybrid is meant to cover; hybrid restores MRR 0.983.

## Limitations

- 30 queries, 6 live chunks, one corpus. Not a production benchmark.
- Hand labels come from an offline fixture; mapping to live chunks is
  approximate.
- No statistical significance testing.
- Latency includes embedding (and possibly model warm-up on the first query);
  it is local Docker on a shared machine, not a controlled server.
