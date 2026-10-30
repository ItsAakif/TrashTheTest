"""Hybrid retrieval helpers (T301).

Pure, offline-testable pieces of the vector + tsvector + RRF pipeline:

- build_tsquery: plainto_tsquery-compatible token glueing for the lexical arm.
- rrf_fuse: Reciprocal Rank Fusion over per-arm rank lists. Rank-based, no
  weight tuning (ADR-005 s5). k=60 is the standard RRF constant.
- FUSED_SQL: one-statement fusion pattern (vector arm + lexical arm joined
  by FULL OUTER JOIN, fused by RRF) with iterative HNSW scan settings for
  filtered queries. Repositories bind and execute it against Postgres;
  SQLite/test doubles use rrf_fuse over in-Python arm results instead.

Live index note: EXPLAIN against pgvector/HNSW requires a live Postgres
and is recorded as NOT RUN in .planning/progress/T301-T310.md when no DB
is available. Nothing here fakes an EXPLAIN.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

RRF_K = 60

_WS_RE = re.compile(r"\s+")
_TS_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def build_tsquery(query: str) -> str:
    """Build a plainto-style '&'-joined tsquery string from raw text."""
    tokens = _TS_TOKEN_RE.findall(query.lower())
    return " & ".join(tokens)


def normalize_query(query: str, max_length: int = 1000) -> Tuple[str, bool]:
    """Trim/normalize user query. Returns (query, was_truncated)."""
    q = _WS_RE.sub(" ", (query or "").strip())
    if len(q) > max_length:
        return q[:max_length], True
    return q, False


def rrf_fuse(
    arms: Sequence[Sequence[str]],
    k: int = RRF_K,
    top_n: int | None = None,
) -> List[Tuple[str, float]]:
    """Fuse per-arm rank lists (best-first id lists) by RRF score.

    score(d) = sum_over_arms(1 / (k + rank(d))). Rank is 1-based within
    each arm. Returns [(id, score)] sorted by score desc, then id asc
    for determinism.
    """
    scores: Dict[str, float] = {}
    for arm in arms:
        for rank, cid in enumerate(arm, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n is not None:
        fused = fused[:top_n]
    return fused


# One-statement fusion: vector arm (HNSW cosine) + lexical arm (ts_rank),
# FULL OUTER JOIN, RRF score. Callers bind: query_embedding (vector
# literal string), tsquery text, model name, optional project/document/
# user UUIDs (None allowed), relaxed flag (0 strict AND / 1 OR recovery),
# per-arm limit, final limit.
#
# Filtered queries set hnsw.iterative_scan + ef_search per session before
# executing so a selective WHERE on top of HNSW still returns top_k rows.
#
# Bounded recovery (gen-grounding fix): the lexical arm is a strict AND
# (plainto_tsquery), so a multi-word question whose tokens never co-occur
# in one chunk returns 0 rows even though the corpus has every token.
# `relaxed=1` re-runs the SAME statement with the lexical arm OR-joined
# (plainto output with '&' swapped for '|'), which is still a real
# tsvector match over the same scope -- never a broadened scope, never a
# fabricated row. Callers only set it when the strict run returned zero
# rows for a scoped search, and report it in retrieval meta.
#
# NULL-UUID typing (live asyncpg find): a bare `:project_id IS NULL`
# leaves asyncpg unable to infer the bind type ("could not determine
# data type of parameter $6"). Every nullable UUID bind is therefore
# double-CAST inside the params CTE -- CAST(... AS text) first so a
# None bind arrives as typed text, then CAST(... AS uuid) so arms see
# a real uuid -- and arms reference the typed CTE columns only: no
# bare :param IS NULL anywhere in the statement. The text hop also
# tolerates empty-string binds from callers that stringify None.
FUSED_SQL = """
WITH params AS (
  SELECT
    CAST(:query_embedding AS vector(1024)) AS qvec,
    CAST(:relaxed AS integer) AS relaxed,
    -- Strict AND (default) vs OR-joined recovery (relaxed=1). The CAST
    -- keeps the plainto output as tsquery TEXT long enough to swap the
    -- operators, then to_tsquery parses it back. No bare :param::type.
    CASE WHEN CAST(:relaxed AS integer) = 1
         THEN to_tsquery('english',
              replace(CAST(plainto_tsquery('english', :query_text) AS text), '&', '|'))
         ELSE plainto_tsquery('english', :query_text) END AS qq,
    -- Never :name::type here: SQLAlchemy text() misparses a bound param
    -- followed by :: (live asyncpg 500, Phase B find). CAST() explicitly,
    -- including integer casts so LIMIT (SELECT ...) sees bigint.
    CAST(:embedding_model AS text) AS model,
    CAST(CAST(:project_id AS text) AS uuid) AS pid,
    CAST(CAST(:document_id AS text) AS uuid) AS did,
    CAST(CAST(:user_id AS text) AS uuid) AS uid,
    CAST(:per_arm AS integer) AS per_arm,
    CAST(:limit AS integer) AS lim
),
vector_arm AS (
  SELECT c.id::text AS chunk_id,
         ROW_NUMBER() OVER (ORDER BY c.embedding_v <=> (SELECT qvec FROM params)) AS rnk,
         (c.embedding_v <=> (SELECT qvec FROM params)) AS distance
  FROM chunks c
  JOIN documents d ON d.id = c.document_id
  JOIN projects p ON p.id = d.project_id
  WHERE c.embedding_v IS NOT NULL
    AND c.embedding_model = (SELECT model FROM params)
    AND ((SELECT pid FROM params) IS NULL OR d.project_id = (SELECT pid FROM params))
    AND ((SELECT did FROM params) IS NULL OR c.document_id = (SELECT did FROM params))
    AND ((SELECT uid FROM params) IS NULL OR p.user_id = (SELECT uid FROM params))
  ORDER BY c.embedding_v <=> (SELECT qvec FROM params)
  LIMIT (SELECT per_arm FROM params)
),
lexical_arm AS (
  SELECT c.id::text AS chunk_id,
         ROW_NUMBER() OVER (ORDER BY ts_rank(c.tsv, (SELECT qq FROM params)) DESC) AS rnk,
         ts_rank(c.tsv, (SELECT qq FROM params)) AS rank_score
  FROM chunks c
  JOIN documents d ON d.id = c.document_id
  JOIN projects p ON p.id = d.project_id
  WHERE c.tsv @@ (SELECT qq FROM params)
    AND ((SELECT pid FROM params) IS NULL OR d.project_id = (SELECT pid FROM params))
    AND ((SELECT did FROM params) IS NULL OR c.document_id = (SELECT did FROM params))
    AND ((SELECT uid FROM params) IS NULL OR p.user_id = (SELECT uid FROM params))
  ORDER BY rank_score DESC
  LIMIT (SELECT per_arm FROM params)
),
fused AS (
  SELECT COALESCE(v.chunk_id, l.chunk_id) AS chunk_id,
         (COALESCE(1.0 / (60 + v.rnk), 0) + COALESCE(1.0 / (60 + l.rnk), 0)) AS rrf,
         v.distance AS vector_distance,
         l.rank_score AS lexical_score,
         CASE WHEN v.chunk_id IS NOT NULL AND l.chunk_id IS NOT NULL THEN 'both'
              WHEN v.chunk_id IS NOT NULL THEN 'vector'
              ELSE 'lexical' END AS arm
  FROM vector_arm v FULL OUTER JOIN lexical_arm l ON v.chunk_id = l.chunk_id
)
SELECT chunk_id, rrf, vector_distance, lexical_score, arm
FROM fused
ORDER BY rrf DESC
LIMIT (SELECT lim FROM params)
"""

# Session settings enabling iterative HNSW scans for filtered queries
# (pgvector >= 0.8). Executed before the fused statement on Postgres.
ITERATIVE_SCAN_SQL = "SET LOCAL hnsw.iterative_scan = 'strict_order'"


__all__ = [
    "RRF_K",
    "build_tsquery",
    "normalize_query",
    "rrf_fuse",
    "FUSED_SQL",
    "ITERATIVE_SCAN_SQL",
]
