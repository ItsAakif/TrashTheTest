# Semantic Caching Architecture

## Why

Repeated and semantically equivalent questions currently pay the full cost of
embedding, retrieval (and, for the tutor, generation). The cache reduces that
work without returning an answer whose source, model, prompt or permissions
have changed.

## Storage choice

- **Postgres + pgvector** is the source of truth and the semantic index
  (HNSW on `query_embedding`). It already exists, is transactional, and
  supports TTL filtering plus invalidation by project/document/user.
- **Redis** is a best-effort exact-key hot path only (5-minute TTL). A Redis
  outage degrades to Postgres; it never changes correctness.
- Three logical layers share the table: `retrieval`, `answer` (and `query`
  reserved for future use).

## Identity key

`sha256(user, project, sorted(doc_id:version), embedding_model, llm_model,
prompt_version, retrieval_config_hash, response_format)`

Any change to a component produces a different identity, so a cache entry
cannot be reused across users, projects, document versions, models, prompt
versions or retrieval settings. The identity is not a bare nearest-neighbour
lookup.

## Lookup and store

1. Normalise query, compute `query_hash`.
2. Redis exact hit (best effort).
3. Postgres exact hit by `(identity, layer, query_hash)` and unexpired.
4. If no exact hit, embed the query and run a pgvector cosine search scoped to
   the identity and expiry; accept only if similarity >= threshold.
5. Increment hits and return; on miss, run the normal RAG path and store the
   result with TTL.
6. Entries expire lazily; expired rows are pruned opportunistically.

## Correctness

- A cached answer is only reused when the source document versions, embedding
  model, LLM model, prompt version and retrieval config all match.
- Document deletion and embedding regeneration explicitly invalidate cache
  entries.
- Permission changes are covered by including the user id in the identity;
  there is no cross-user reuse.
- Nothing is cached for conversational/clarification responses.

## Threshold calibration

Measured with Qwen3-Embedding-0.6B on manually authored pairs
(`tests/eval/cache_pairs.json`):

| Threshold | Equivalent-pair hit rate | Near-miss false-hit rate |
|---|---|---|
| 0.70 | 0.40 | 0.40 |
| 0.75 | 0.40 | 0.20 |
| **0.80** | **0.40** | **0.00** |
| 0.85 | 0.40 | 0.00 |
| 0.90 | 0.20 | 0.00 |
| 0.95 | 0.00 | 0.00 |

**Chosen threshold: 0.80.** It is conservative: zero false hits on the
near-miss set, but only 40% of paraphrases are reused. Lowering the threshold
raises hit rate and immediately admits wrong answers (dot vs cross product at
0.75). Correctness was prioritised over hit rate.

## TTL, size, controls

- Default TTL: 3600s; Redis hot key 300s.
- `SEMANTIC_CACHE_ENABLED`, `SEMANTIC_CACHE_THRESHOLD`,
  `SEMANTIC_CACHE_TTL_SECONDS`, `SEMANTIC_CACHE_MAX_ENTRIES`.
- `GET /api/semantic-cache/stats`, `POST /api/semantic-cache/invalidate`
  (`project_id`, `document_id`, `user_id`), `DELETE /api/semantic-cache/clear`.
- Stored fields: identity, layer, query hash/text, response, citations,
  document ids/versions, project/user, model/prompt/retrieval versions,
  created/expiry, hits, last-hit.

## Limitations

- Similarity is not entailment. The conservative threshold means many valid
  paraphrases miss.
- No LLM-as-judge verification was added (explicitly out of scope).
- The threshold was calibrated on 10 author-labelled pairs; this is not an
  independent human study.
- `answer` caching depends on the tutor generation path; other generators
  (notes, MCQs) are not cached yet.
