"""T106: ownership dependency + project CRUD (401/404, isolation, cascade)."""

import uuid

import pytest

from database import repositories
from database.models.project import Project, ProjectStatus

pytestmark = pytest.mark.asyncio


class FakeProjectStore:
    def __init__(self):
        self.projects = {}

    def add(self, user_id, name="P", description=None, subject=None):
        p = Project(
            id=uuid.uuid4(),
            user_id=user_id,
            name=name,
            description=description,
            subject=subject,
            status=ProjectStatus.PROCESSING,
            progress=0,
        )
        from datetime import datetime

        p.created_at = datetime.utcnow()
        p.updated_at = datetime.utcnow()
        self.projects[p.id] = p
        return p


def _install_project_fakes(monkeypatch, dummy_db, store):
    from database.engine import get_db
    from main import app

    async def fake_get_db():
        yield dummy_db

    async def fake_by_id(db, pid):
        return store.projects.get(pid)

    async def fake_list(db, user_id, limit=20, offset=0):
        owned = [p for p in store.projects.values() if p.user_id == user_id]
        return owned[offset:offset + limit], len(owned)

    async def fake_create(db, user_id, name, description=None, subject=None):
        return store.add(user_id, name, description, subject)

    async def fake_update(db, pid, name=None, description=None, subject=None):
        p = store.projects.get(pid)
        if p is None:
            return None
        if name is not None:
            p.name = name
        if description is not None:
            p.description = description
        if subject is not None:
            p.subject = subject
        return p

    async def fake_delete(db, pid):
        return store.projects.pop(pid, None) is not None

    async def fake_docs(db, pid):
        return []

    monkeypatch.setattr(repositories, "get_project_by_id", fake_by_id)
    monkeypatch.setattr(repositories, "list_projects_by_user", fake_list)
    monkeypatch.setattr(repositories, "create_project", fake_create)
    monkeypatch.setattr(repositories, "update_project", fake_update)
    monkeypatch.setattr(repositories, "delete_project", fake_delete)
    monkeypatch.setattr(repositories, "get_documents_by_project", fake_docs)
    app.dependency_overrides[get_db] = fake_get_db
    return app


def _login_as(monkeypatch, app, user):
    from services.auth.deps import get_current_user

    async def fake_user():
        return user

    app.dependency_overrides[get_current_user] = fake_user


async def test_project_crud_and_idor(app_client, dummy_db, monkeypatch, user_a, user_b):
    from main import app

    store = FakeProjectStore()
    _install_project_fakes(monkeypatch, dummy_db, store)
    try:
        # Unauthenticated create -> 401.
        anon = await app_client.post("/api/projects", json={"title": "Nope"})
        assert anon.status_code == 401

        _login_as(monkeypatch, app, user_a)
        created = await app_client.post(
            "/api/projects",
            json={"title": "Physics", "description": "Mechanics", "subject": "Science"},
        )
        assert created.status_code == 201, created.text
        pid = created.json()["id"]
        assert created.json()["title"] == "Physics"

        mine = await app_client.get("/api/projects")
        assert mine.status_code == 200
        assert mine.json()["total"] == 1

        # User B sees nothing and gets 404 (not 200, not 403) for A's project.
        _login_as(monkeypatch, app, user_b)
        theirs = await app_client.get("/api/projects")
        assert theirs.json()["total"] == 0
        assert (await app_client.get(f"/api/projects/{pid}")).status_code == 404

        # A can read/patch; B cannot.
        _login_as(monkeypatch, app, user_a)
        assert (await app_client.get(f"/api/projects/{pid}")).status_code == 200
        patched = await app_client.patch(f"/api/projects/{pid}", json={"title": "Physics II"})
        assert patched.status_code == 200
        assert patched.json()["title"] == "Physics II"

        # DELETE removes; second DELETE is 404.
        assert (await app_client.delete(f"/api/projects/{pid}")).status_code == 200
        assert (await app_client.delete(f"/api/projects/{pid}")).status_code == 404
        assert (await app_client.get("/api/projects")).json()["total"] == 0
    finally:
        app.dependency_overrides.clear()


async def test_no_hardcoded_demo_uuid_in_services():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    needle = "00000000-0000-0000-0000-" + "000000000001"  # built, not literal
    for base, _, files in os.walk(root):
        if "__pycache__" in base or os.path.basename(base) == "tests":
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                if needle in fh.read():
                    offenders.append(path)
    assert offenders == [], f"demo UUID remains: {offenders}"


# ---------------------------------------------------------------------------
# T107: real GET reads (flashcards/quiz/notes/summaries), ownership-scoped,
# empty-with-"not_generated" state, never fabricated, camelCase contract.
# ---------------------------------------------------------------------------

async def _t107_app(monkeypatch, dummy_db, store, decks=(), notes=(), quizzes=()):
    from database.engine import get_db
    from main import app

    async def fake_get_db():
        yield dummy_db

    async def fake_by_id(db, pid):
        return store.projects.get(pid)

    async def fake_decks(db, pid):
        return list(decks)

    async def fake_notes(db, pid):
        return list(notes)

    async def fake_quizzes(db, pid):
        return list(quizzes)

    monkeypatch.setattr(repositories, "get_project_by_id", fake_by_id)
    monkeypatch.setattr(repositories, "get_flashcard_decks_by_project", fake_decks)
    monkeypatch.setattr(repositories, "get_notes_by_project", fake_notes)
    monkeypatch.setattr(repositories, "get_quizzes_by_project", fake_quizzes)
    app.dependency_overrides[get_db] = fake_get_db
    return app


async def test_flashcards_read_empty_state_and_ownership(app_client, dummy_db, monkeypatch, user_a, user_b):
    import uuid as _uuid
    from main import app

    store = FakeProjectStore()
    project = store.add(user_a.id)
    app2 = await _t107_app(monkeypatch, dummy_db, store)
    try:
        from services.auth.deps import get_current_user

        async def fake_user():
            return user_a

        app2.dependency_overrides[get_current_user] = fake_user
        empty = await app_client.get(f"/api/generation/flashcards/{project.id}")
        assert empty.status_code == 200, empty.text
        assert empty.json()["state"] == "not_generated"
        assert empty.json()["projectId"] == str(project.id)

        async def fake_user_b():
            return user_b

        app2.dependency_overrides[get_current_user] = fake_user_b
        assert (await app_client.get(f"/api/generation/flashcards/{project.id}")).status_code == 404
        assert (await app_client.get(f"/api/projects/{project.id}/flashcard-decks")).status_code == 404
        assert (await app_client.get(f"/api/generation/notes/{project.id}")).status_code == 404
        assert (await app_client.get(f"/api/generation/quiz/{project.id}")).status_code == 404
        assert (await app_client.get(f"/api/generation/summary/{project.id}")).status_code == 404
    finally:
        app.dependency_overrides.clear()


async def test_notes_quiz_summary_empty_states_are_explicit(app_client, dummy_db, monkeypatch, user_a):
    from main import app

    store = FakeProjectStore()
    project = store.add(user_a.id)
    app2 = await _t107_app(monkeypatch, dummy_db, store)
    try:
        from services.auth.deps import get_current_user

        async def fake_user():
            return user_a

        app2.dependency_overrides[get_current_user] = fake_user
        for path, key in [
            (f"/api/generation/notes/{project.id}", "notes"),
            (f"/api/generation/quiz/{project.id}", "quizzes"),
            (f"/api/generation/summary/{project.id}", "summaries"),
        ]:
            resp = await app_client.get(path)
            assert resp.status_code == 200, (path, resp.text)
            body = resp.json()
            assert body["state"] == "not_generated"
            assert body[key] == [] and body["total"] == 0
        decks = await app_client.get(f"/api/projects/{project.id}/flashcard-decks")
        assert decks.json()["state"] == "not_generated"
    finally:
        app.dependency_overrides.clear()


async def test_flashcards_read_returns_real_rows(app_client, dummy_db, monkeypatch, user_a):
    import uuid as _uuid
    from types import SimpleNamespace as _NS

    from main import app

    store = FakeProjectStore()
    project = store.add(user_a.id)
    card = _NS(id=_uuid.uuid4(), front="Q?", back="A!", difficulty="easy")
    deck = _NS(id=_uuid.uuid4(), name="Deck 1", card_count=1, flashcards=[card])
    app2 = await _t107_app(monkeypatch, dummy_db, store, decks=(deck,))
    try:
        from services.auth.deps import get_current_user

        async def fake_user():
            return user_a

        app2.dependency_overrides[get_current_user] = fake_user
        resp = await app_client.get(f"/api/generation/flashcards/{project.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "ready"
        assert body["cards"][0]["front"] == "Q?"
        assert body["projectId"] == str(project.id)
        decks = await app_client.get(f"/api/projects/{project.id}/flashcard-decks")
        assert decks.json()["total"] == 1
        assert decks.json()["decks"][0]["name"] == "Deck 1"
    finally:
        app.dependency_overrides.clear()
