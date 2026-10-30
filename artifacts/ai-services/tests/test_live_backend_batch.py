"""Live-backend blocker batch: FUSED_SQL typing, website 008, 005 defer.

Offline tests (no live Postgres required):
- Bug 1: FUSED_SQL nullable UUID binds are double-CAST via text (asyncpg
  infers uuid even for NULL), no bare `:param IS NULL`, no `:param::type`.
- Bug 1b: repositories.search_similar_chunks_hybrid rolls back the aborted
  statement before re-raising so the service offline-fallback still runs.
- Bug 2: website reads tolerate a pre-008 store (missing title column):
  SELECT WHERE slug retries against legacy columns after rollback.
- Bug 2b: migration 008 is additive/idempotent; migration 005 defers
  (warn, no raise) so `upgrade head` from 004 still reaches 006/007/008.
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


async def test_fused_sql_double_cast_uuid_typing():
    from services.retrieval.hybrid import FUSED_SQL
    import re

    assert "CAST(CAST(:project_id AS text) AS uuid) AS pid" in FUSED_SQL
    assert "CAST(CAST(:document_id AS text) AS uuid) AS did" in FUSED_SQL
    assert "CAST(CAST(:user_id AS text) AS uuid) AS uid" in FUSED_SQL
    bare = [line for line in FUSED_SQL.splitlines()
            if re.search(r"\(\s*:(project_id|document_id|user_id)\s+IS NULL", line)]
    assert bare == []
    code = "\n".join(
        line for line in FUSED_SQL.splitlines() if not line.strip().startswith("--"))
    assert not re.search(r":[a-z_]+::", code)
    assert "(SELECT pid FROM params) IS NULL" in FUSED_SQL
    assert "(SELECT did FROM params) IS NULL" in FUSED_SQL
    assert "(SELECT uid FROM params) IS NULL" in FUSED_SQL


async def test_hybrid_repository_rolls_back_before_reraise():
    """Poisoned fused statement: txn rolled back, caller falls back."""
    from types import SimpleNamespace

    from database import repositories

    class _FailOnceDB:
        def __init__(self):
            self.rollbacks = 0
            self.bind = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        async def execute(self, *args, **kwargs):
            raise Exception('could not determine data type of parameter $6')

        async def rollback(self):
            self.rollbacks += 1

    db = _FailOnceDB()
    with pytest.raises(Exception, match="data type of parameter"):
        await repositories.search_similar_chunks_hybrid(
            db, query_embedding=[0.01] * 8, query_text="test",
            per_arm=5, limit=3)
    assert db.rollbacks == 1


async def test_website_read_tolerates_pre008_store():
    """Pre-008 SELECT (all columns incl. title) fails; legacy retry wins."""
    import uuid
    from datetime import datetime

    from database import repositories

    legacy_row = {
        "id": uuid.uuid4(), "project_id": uuid.uuid4(),
        "website_type": "mixed", "html_content": "<p>Hi</p>",
        "slug": "legacy-slug", "is_published": False,
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }

    class _Row:
        def __init__(self, mapping):
            self._mapping = mapping

    class _MappingsResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return self

        def first(self):
            return self._rows[0] if self._rows else None

        def all(self):
            return list(self._rows)

    class _Pre008DB:
        """First execute (ORM SELECT) raises missing-column; text() retry ok."""

        def __init__(self):
            self.rollbacks = 0
            self.calls = 0

        async def execute(self, stmt, params=None):
            self.calls += 1
            sql = str(getattr(stmt, "text", stmt))
            if "SELECT" in sql and "websites" in sql and self.calls == 1:
                raise Exception('column websites.title does not exist')
            return _MappingsResult([_Row(dict(legacy_row))])

        async def rollback(self):
            self.rollbacks += 1

    db = _Pre008DB()
    # slug_exists path shares the same guard; exercise the by-slug read.
    found = await repositories.get_website_by_slug(db, "legacy-slug")
    assert found is not None
    assert found.slug == "legacy-slug"
    assert db.rollbacks >= 1
    # Service metadata tolerates the legacy row (getattr guards).
    from services.website_gen.service import website_gen_service

    meta = website_gen_service._to_metadata(found)
    assert meta.slug == "legacy-slug"
    assert meta.title == "Untitled"
    assert meta.status == "draft" and meta.visibility == "private"


async def test_website_list_tolerates_pre008_store():
    import uuid
    from datetime import datetime

    from database import repositories

    class _Row:
        def __init__(self, mapping):
            self._mapping = mapping

    class _MappingsResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return self

        def first(self):
            return self._rows[0] if self._rows else None

        def all(self):
            return list(self._rows)

    row = {
        "id": uuid.uuid4(), "project_id": uuid.uuid4(),
        "website_type": "notes", "html_content": "<p>x</p>",
        "slug": "s1", "is_published": False,
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }

    class _Pre008DB:
        def __init__(self):
            self.rollbacks = 0
            self.calls = 0

        async def execute(self, stmt, params=None):
            self.calls += 1
            if self.calls == 1:
                raise Exception('column websites.title does not exist')
            return _MappingsResult([_Row(dict(row))])

        async def rollback(self):
            self.rollbacks += 1

    db = _Pre008DB()
    rows = await repositories.get_websites_by_project(db, row["project_id"])
    assert len(rows) == 1 and rows[0].slug == "s1"


async def test_migration_005_defers_instead_of_refusing():
    src = Path("migrations/versions/005_contract_drop_legacy_embedding.py").read_text()
    assert "DEFERRED" in src
    assert "raise RuntimeError" not in src
    assert "006" in src  # docstring names the chain that must still apply


async def test_migration_008_additive_idempotent():
    src = Path("migrations/versions/008_website_metadata_columns.py").read_text()
    for col in ("title", "description", "visibility", "status", "version"):
        assert col in src
    assert "ADD COLUMN IF NOT EXISTS" in src
    assert "UPDATE websites SET visibility" in src
    # Chain order intact: 008 still follows 007 (no arch rewrite).
    assert 'down_revision = "007_outputtype_phase4_values"' in src
