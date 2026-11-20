# Design Decisions

Format: decision, alternatives, rationale, consequences.

## D1 - Dual-write embeddings to BYTEA and `embedding_v`

- **Decision:** write both columns in the same chunk update; keep the legacy
  column for rollback.
- **Alternatives:** migrate fully to pgvector and drop BYTEA; write only
  `embedding_v`.
- **Rationale:** existing code and tests still read BYTEA; dual-write makes
  fresh documents searchable without a backfill and keeps rollback safe.
- **Consequence:** 4 KB duplicate per chunk until the legacy column is retired.

## D2 - Migration-managed HNSW, non-concurrent

- **Decision:** create the HNSW index in Alembic (015) with
  `CREATE INDEX IF NOT EXISTS`; keep the concurrent script for large data.
- **Rationale:** the measured relation was 8.7 MB; non-concurrent DDL is safe
  and transactional. Large deployments need the concurrent path.
- **Consequence:** on a large table this migration would block writes.

## D3 - Postgres is the semantic-cache source of truth; Redis is hot-only

- **Decision:** pgvector HNSW for semantic lookup, Postgres for TTL and
  invalidation; Redis only for exact keys.
- **Alternatives:** Redis vector search, in-memory only, separate vector DB.
- **Rationale:** no new dependency; transactional invalidation; correctness
  survives Redis loss.
- **Consequence:** semantic lookup pays a Postgres round trip.

## D4 - Cache identity is a hash of context, not a project id

- **Decision:** include user, project, document versions, embedding model, LLM
  model, prompt version, retrieval config and response format.
- **Rationale:** prevents stale/cross-tenant reuse when any input changes.
- **Consequence:** low reuse across models or prompt edits by design.

## D5 - Threshold 0.80, correctness over hit rate

- **Decision:** 0.80 cosine threshold.
- **Evidence:** 0 false hits on 5 near-miss pairs; 40% paraphrase hits.
  Lower thresholds admitted dot-product vs cross-product confusion.
- **Consequence:** many valid paraphrases miss. Improving this needs a
  verification step, which was explicitly out of scope (no LLM-as-judge).

## D6 - Retrieval modes/weights are request-visible

- **Decision:** `retrieval_mode`, `vector_weight`, `lexical_weight`,
  `source_types` are accepted per request; defaults preserve hybrid 1.0/1.0.
- **Rationale:** makes dense/sparse/hybrid independently measurable without
  redeploying.
- **Consequence:** clients can tune ranking; server clamps top_k.

## D7 - `min_similarity` is dense-only

- **Decision:** compare it to cosine similarity, never to the RRF score; keep
  lexical-only rows.
- **Rationale:** RRF scores live near 1/60..1/30; comparing them to a 0..1
  threshold silently zeroed results.
- **Consequence:** hybrid results with no dense component are not thresholded
  by this parameter.

## D8 - S0 (authorisation) deferred

- **Decision:** do not close the unauthenticated RAG-adjacent routes in this
  iteration, at the owner's request.
- **Consequence:** the app is not tenant-safe; do not expose it publicly as a
  multi-user service. This is documented in the architecture doc.

## D9 - Best-effort tracing

- **Decision:** persist retrieval traces in a separate session; failures are
  logged and ignored.
- **Rationale:** observability must never take down retrieval.
- **Consequence:** traces can be missing during DB issues.
