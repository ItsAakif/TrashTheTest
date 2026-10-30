"""Append-only: study session rows carry the live session id (B4 restart)."""

import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.asyncio


class _MemDB:
    def __init__(self):
        self.adds = []

    def add(self, obj):
        self.adds.append(obj)

    async def flush(self):
        return None

    async def refresh(self, obj):
        return obj


async def test_create_study_session_uses_live_id():
    from database import repositories

    db = _MemDB()
    sid = uuid.uuid4()
    row = await repositories.create_study_session(
        db, uuid.uuid4(), uuid.uuid4(), "tutor", session_id=sid)
    assert row.id == sid
    assert db.adds and db.adds[0].id == sid


async def test_create_study_session_defaults_to_fresh_id():
    from database import repositories

    db = _MemDB()
    row = await repositories.create_study_session(
        db, uuid.uuid4(), uuid.uuid4(), "tutor")
    assert row.id is not None


async def test_resolve_session_rebuilds_from_persisted_row(monkeypatch):
    from services.study import service as study_mod

    mgr = study_mod.session_manager
    sid = uuid.uuid4()
    owner = uuid.uuid4()
    proj = uuid.uuid4()
    row = SimpleNamespace(
        id=sid, user_id=owner, project_id=proj,
        session_type=SimpleNamespace(value="tutor"),
        completed_at=None, started_at=datetime.utcnow())
    saved = dict(mgr._sessions)
    mgr._sessions.clear()

    class _DB:
        async def get(self, model, key):
            assert str(key) == str(sid)
            return row

    try:
        rebuilt = await mgr.resolve_session(str(sid), _DB())
        assert rebuilt is not None
        assert str(rebuilt.session_id) == str(sid)
        assert str(rebuilt.user_id) == str(owner)
        assert rebuilt.items == []
    finally:
        mgr._sessions.clear()
        mgr._sessions.update(saved)


async def test_resolve_session_unknown_id_is_none():
    from services.study import service as study_mod

    mgr = study_mod.session_manager
    saved = dict(mgr._sessions)
    mgr._sessions.clear()

    class _DB:
        async def get(self, model, key):
            return None

    try:
        assert await mgr.resolve_session(str(uuid.uuid4()), _DB()) is None
    finally:
        mgr._sessions.clear()
        mgr._sessions.update(saved)
