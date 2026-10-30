"""FUSED_SQL UUID typing tests (live-bug batch, bug 6). Offline.

Live PG find: a bare `:project_id IS NULL` leaves asyncpg unable to
infer the bind type ("could not determine data type of parameter $6").
Every nullable UUID bind must be CAST inside the params CTE, and the
arms may only reference the typed CTE columns.
"""

import re


def test_fused_sql_nullable_uuids_typed_in_cte():
    from services.retrieval.hybrid import FUSED_SQL

    # Double-CAST via text: None arrives typed, asyncpg infers uuid.
    assert "CAST(CAST(:project_id AS text) AS uuid) AS pid" in FUSED_SQL
    assert "CAST(CAST(:document_id AS text) AS uuid) AS did" in FUSED_SQL
    assert "CAST(CAST(:user_id AS text) AS uuid) AS uid" in FUSED_SQL
    bare = [line for line in FUSED_SQL.splitlines()
            if re.search(r"\(\s*:(project_id|document_id|user_id)\s+IS NULL", line)]
    assert bare == []
    assert "(SELECT pid FROM params) IS NULL" in FUSED_SQL
    assert "(SELECT did FROM params) IS NULL" in FUSED_SQL
    assert "(SELECT uid FROM params) IS NULL" in FUSED_SQL
