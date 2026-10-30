"""Notes LIST + delete, and the truncated-JSON / oversized-topic fixes.

Two reported defects, one file:

1. "Study notes shows 2 ready but only Generate/Read (no list)" - the reads
   returned the whole set but the UI had no list, no per-note Open and no
   Delete, and the viewer could only open the newest artifact.
2. "job fails LLM output JSON was truncated" - one per-topic call asked the
   model for every chunk of a topic at once inside a shared 4096-token
   budget, so a large topic was cut off mid-JSON; the repair loop then asked
   for the same oversized answer again and the job died.

Everything here is offline: provider doubles, an in-memory session, and an
ASGI client. No network, no database, no LLM.
"""

import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from database import knowledge_repo, repositories
from database.models.project import Project, ProjectStatus
from services.academic import artifacts as academic_artifacts
from services.strategies.llm.structured import (
    MAX_TRUNCATION_RECOVERIES,
    StructuredGenerationError,
    _close_truncated_json,
    _extract_json,
    generate_structured,
)
from services.transformation.types import TopicNotesOutput

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _note_row(project_id, title, created, content=None):
    return SimpleNamespace(
        id=uuid.uuid4(),
        project_id=project_id,
        title=title,
        content=content if content is not None else {},
        created_at=created,
    )


def _notes_content(artifact_id, version, pages, mode="detailed", coverage=None):
    """A compatibility payload shaped like a hierarchical notes artifact."""
    return {
        "artifact_id": artifact_id,
        "pdf_version": version,
        "pdf_pages": pages,
        "pdf_path": f"{uuid.uuid4()}/v{version}.pdf",
        "config": {"mode": mode, "detail": "balanced"},
        "coverage": coverage
        or {"topic_coverage": 0.75, "concept_coverage": 0.5, "passes": True},
        "observability": {"topic_count": 3, "section_count": 3, "mode": mode},
        "sections": [{"heading": "H", "content": "Body."}],
    }


class _FakeStorage:
    def __init__(self):
        self.objects = {}
        self.deleted = []

    async def exists(self, key):
        return key in self.objects

    async def delete(self, key):
        if key in self.objects:
            self.deleted.append(key)
            self.objects.pop(key, None)
            return True
        return False

    async def open_stream(self, key, chunk_size=64 * 1024):
        yield self.objects.get(key, b"")


def _notes_app(
    monkeypatch,
    dummy_db,
    project,
    notes,
    user,
    storage=None,
    versions=None,
):
    """Ownership-scoped harness for the notes reads and delete (repo pattern).

    ``versions`` is the stored notes-PDF version list for the project's single
    document root; when given, the knowledge_repo reads are backed by one
    in-memory node so the artifact list/pin/delete paths are real code, not
    fakes.
    """
    from database.engine import get_db
    from main import app
    from services.auth.deps import get_current_user

    rows = {n.id: n for n in notes}

    async def fake_get_db():
        yield dummy_db

    async def fake_user():
        return user

    async def fake_project(db, _pid):
        return project

    async def fake_notes(db, _pid):
        return sorted(
            rows.values(), key=lambda n: (n.created_at, str(n.id)), reverse=True
        )

    async def fake_note_by_id(db, note_id):
        return rows.get(note_id)

    async def fake_delete_note(db, note_id):
        return rows.pop(note_id, None) is not None

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    monkeypatch.setattr(repositories, "get_notes_by_project", fake_notes)
    monkeypatch.setattr(repositories, "get_note_by_id", fake_note_by_id)
    monkeypatch.setattr(repositories, "delete_note", fake_delete_note)
    if storage is not None:
        # Both import styles exist: artifacts.py resolves it inside the
        # function, routes.py binds it at import time.
        monkeypatch.setattr("services.storage.get_default_backend", lambda: storage)
        monkeypatch.setattr(
            "services.academic.routes.get_default_backend", lambda: storage
        )
    if versions is not None:
        node = SimpleNamespace(id=uuid.uuid4(), node_metadata={})
        root = SimpleNamespace(id=node.id, title=project.name, parent_id=None)

        async def fake_nodes(_db, project_id=None, document_id=None):
            return [root]

        async def fake_list(_db, _node_id, artifact_type=None):
            if artifact_type is not None and artifact_type != "notes_pdf":
                return []
            return [dict(v) for v in versions]

        async def fake_get_node(_db, _node_id):
            return node

        async def fake_delete_version(
            _db, _node_id, artifact_type, artifact_id=None, version=None
        ):
            remaining = []
            removed_entry = None
            for entry in versions:
                if str(entry.get("id")) == str(artifact_id) and removed_entry is None:
                    removed_entry = dict(entry)
                    continue
                remaining.append(entry)
            if removed_entry is None:
                return None
            if removed_entry.get("status") == "ready" and not any(
                e.get("status") == "ready" for e in remaining
            ):
                if remaining:
                    newest = max(remaining, key=lambda v: int(v.get("version", 0) or 0))
                    newest["status"] = "ready"
                    newest.pop("superseded_by", None)
            versions[:] = remaining
            return removed_entry

        monkeypatch.setattr(knowledge_repo, "list_knowledge_nodes", fake_nodes)
        monkeypatch.setattr(knowledge_repo, "list_generated_artifacts", fake_list)
        monkeypatch.setattr(knowledge_repo, "get_knowledge_node", fake_get_node)
        monkeypatch.setattr(
            knowledge_repo, "delete_generated_artifact_version", fake_delete_version
        )
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    return app, rows


def _make_project(user, name="Notes project"):
    return Project(
        id=uuid.uuid4(),
        user_id=user.id,
        name=name,
        status=ProjectStatus.PROCESSING,
        progress=0,
    )


def _structure(topic_title="Topic A", chunks=("c1",)):
    """Minimal valid DocumentStructure with one unit/topic (fingerprint set)."""
    from services.academic.types import DocumentStructure

    return DocumentStructure.model_validate({
        "doc_id": "d",
        "title": "Doc",
        "fingerprint": "f" * 16,
        "units": [
            {"title": "Unit 1", "topics": [
                {"title": topic_title, "chunks": list(chunks)},
            ]},
        ],
    })


# ---------------------------------------------------------------------------
# 1a. read_notes returns EVERY note with list metadata
# ---------------------------------------------------------------------------


async def test_read_notes_lists_every_note_with_mode_coverage_and_pages(
    app_client, dummy_db, monkeypatch, user_a
):
    """Not just the latest: id/title/created/mode/coverage/pages per note."""
    from main import app

    project = _make_project(user_a)
    now = datetime(2026, 9, 25, 12, 0, 0)
    notes = [
        _note_row(project.id, "Older notes", now - timedelta(minutes=10),
                  _notes_content("artifact-old", 1, 4, mode="quick")),
        _note_row(project.id, "Newest notes", now,
                  _notes_content("artifact-new", 2, 9, mode="exam")),
        _note_row(project.id, "Middle notes", now - timedelta(minutes=5),
                  _notes_content("artifact-mid", 2, 6)),
    ]
    app2, _rows = _notes_app(monkeypatch, dummy_db, project, notes, user_a)
    try:
        resp = await app_client.get(f"/api/generation/notes/{project.id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "ready"
        assert body["total"] == 3
        assert len(body["notes"]) == 3
        # Newest first: the ordering the list depends on.
        assert [n["title"] for n in body["notes"]] == [
            "Newest notes", "Middle notes", "Older notes",
        ]
        newest = body["notes"][0]
        assert newest["mode"] == "exam"
        assert newest["pages"] == 9
        assert newest["version"] == 2
        assert newest["artifactId"] == "artifact-new"
        assert newest["coverage"]["topic_coverage"] == 0.75
        assert newest["createdAt"] and newest["created_at"] == newest["createdAt"]
        assert newest["id"] and newest["projectId"] == str(project.id)
    finally:
        app.dependency_overrides.clear()


async def test_read_notes_is_ownership_scoped_and_empty_state_is_honest(
    app_client, dummy_db, monkeypatch, user_a, user_b
):
    from main import app
    from services.auth.deps import get_current_user

    project = _make_project(user_a)
    note = _note_row(project.id, "Mine", datetime(2026, 9, 25), _notes_content("a", 1, 3))
    app2, _rows = _notes_app(monkeypatch, dummy_db, project, [note], user_a)
    try:
        # Another user is a 404, never another user's notes.
        async def stranger():
            return user_b

        app2.dependency_overrides[get_current_user] = stranger
        assert (await app_client.get(f"/api/generation/notes/{project.id}")).status_code == 404

        async def owner():
            return user_a

        app2.dependency_overrides[get_current_user] = owner
        assert (await app_client.get(f"/api/generation/notes/{project.id}")).status_code == 200

        # Empty project: explicit not_generated, never a fabricated note.
        empty_project = _make_project(user_a, "Empty")
        _app3, _r = _notes_app(monkeypatch, dummy_db, empty_project, [], user_a)
        body = (await app_client.get(f"/api/generation/notes/{empty_project.id}")).json()
        assert body["state"] == "not_generated"
        assert body["notes"] == [] and body["total"] == 0
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1b. DELETE is ownership-checked, removes row + PDF, then the list refetches
# ---------------------------------------------------------------------------


async def test_delete_note_removes_row_and_its_pdf(
    app_client, dummy_db, monkeypatch, user_a
):
    from main import app

    project = _make_project(user_a)
    now = datetime(2026, 9, 25, 12, 0, 0)
    keep = _note_row(project.id, "Keep me", now, _notes_content("artifact-keep", 1, 4))
    drop = _note_row(project.id, "Delete me", now, _notes_content("artifact-drop", 2, 7))
    storage = _FakeStorage()
    storage.objects[drop.content["pdf_path"]] = b"%PDF-1.4 notes"
    versions = [
        {"id": "artifact-keep", "version": 1, "status": "ready",
         "path": keep.content["pdf_path"], "title": "Keep me", "pages": 4,
         "created_at": "2026-09-25T11:00:00+00:00", "config": {"topic_count": 2}},
        {"id": "artifact-drop", "version": 2, "status": "ready",
         "path": drop.content["pdf_path"], "title": "Delete me", "pages": 7,
         "created_at": "2026-09-25T12:00:00+00:00", "config": {"topic_count": 2}},
    ]
    app2, rows = _notes_app(
        monkeypatch, dummy_db, project, [keep, drop], user_a, storage=storage,
        versions=versions,
    )
    try:
        resp = await app_client.delete(
            f"/api/generation/notes/{project.id}/{drop.id}"
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["deleted"] is True
        assert body["artifact_deleted"] is True
        assert body["bytes_deleted"] is True
        assert drop.content["pdf_path"] in storage.deleted

        # Refetch: the deleted note is gone, the other one survives.
        after = (await app_client.get(f"/api/generation/notes/{project.id}")).json()
        assert after["total"] == 1
        assert [n["title"] for n in after["notes"]] == ["Keep me"]

        # The artifact list agrees: the removed version is gone and the
        # remaining READY version is what the project read resolves to.
        meta = (await app_client.get(f"/api/academic/notes/{project.id}")).json()
        assert meta["state"] == "ready"
        assert meta["version"] == 1
        assert [a["id"] for a in meta["artifacts"]] == ["artifact-keep"]

        # Deleting again is a 404 (idempotency is visible, not silent).
        assert (
            await app_client.delete(f"/api/generation/notes/{project.id}/{drop.id}")
        ).status_code == 404
    finally:
        app.dependency_overrides.clear()


async def test_delete_note_of_another_project_or_user_is_404(
    app_client, dummy_db, monkeypatch, user_a, user_b
):
    from main import app
    from services.auth.deps import get_current_user

    project = _make_project(user_a)
    theirs = _note_row(project.id, "Theirs", datetime(2026, 9, 25),
                       _notes_content("artifact-x", 1, 3))
    mine_note = _note_row(project.id, "Mine", datetime(2026, 9, 25),
                          _notes_content("artifact-y", 1, 3))
    app2, _rows = _notes_app(
        monkeypatch, dummy_db, project, [theirs, mine_note], user_a
    )
    try:
        async def stranger():
            return user_b

        app2.dependency_overrides[get_current_user] = stranger
        assert (
            await app_client.delete(f"/api/generation/notes/{project.id}/{mine_note.id}")
        ).status_code == 404

        async def owner():
            return user_a

        app2.dependency_overrides[get_current_user] = owner
        # A note id that belongs to a different project is a 404 too.
        other_project = _make_project(user_a, "Other")

        async def other_by_id(db, pid):
            return other_project if str(pid) == str(other_project.id) else project

        monkeypatch.setattr(repositories, "get_project_by_id", other_by_id)
        assert (
            await app_client.delete(f"/api/generation/notes/{other_project.id}/{mine_note.id}")
        ).status_code == 404
        # Still readable: a refused delete changes nothing.
        assert (await app_client.get(f"/api/generation/notes/{project.id}")).json()["total"] == 2
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1c. Academic notes read: every version + a pinned version
# ---------------------------------------------------------------------------


async def test_academic_notes_read_lists_versions_and_pins_one(
    app_client, dummy_db, monkeypatch, user_a
):
    from main import app

    project = _make_project(user_a)
    now = datetime(2026, 9, 25, 12, 0, 0)
    notes = [
        _note_row(project.id, "v1 notes", now, _notes_content("artifact-1", 1, 4)),
        _note_row(project.id, "v2 notes", now, _notes_content("artifact-2", 2, 8)),
    ]
    storage = _FakeStorage()
    for row in notes:
        storage.objects[row.content["pdf_path"]] = b"%PDF-1.4"
    versions = [
        {"id": "artifact-1", "version": 1, "status": "superseded",
         "superseded_by": 2, "path": notes[0].content["pdf_path"],
         "title": "v1 notes", "pages": 4, "model": "academic-notes-hierarchical",
         "created_at": "2026-09-25T11:00:00+00:00",
         "coverage": {"topic_coverage": 0.5},
         "config": {"topic_count": 3, "section_count": 3,
                    "notes_config": {"mode": "quick", "detail": "concise"}}},
        {"id": "artifact-2", "version": 2, "status": "ready",
         "path": notes[1].content["pdf_path"], "title": "v2 notes", "pages": 8,
         "model": "academic-notes-hierarchical",
         "created_at": "2026-09-25T12:00:00+00:00",
         "coverage": {"topic_coverage": 1.0},
         "config": {"topic_count": 5, "section_count": 5,
                    "notes_config": {"mode": "exam", "detail": "balanced"}}},
    ]
    app2, _rows = _notes_app(
        monkeypatch, dummy_db, project, notes, user_a, storage=storage,
        versions=versions,
    )
    try:
        head = (await app_client.get(f"/api/academic/notes/{project.id}")).json()
        assert head["state"] == "ready" and head["version"] == 2
        assert head["total"] == 2
        assert [a["version"] for a in head["artifacts"]] == [2, 1]
        first = head["artifacts"][0]
        assert first["id"] == "artifact-2"
        assert first["title"] == "v2 notes"
        assert first["pages"] == 8
        assert first["mode"] == "exam"
        assert first["created_at"] == "2026-09-25T12:00:00+00:00"
        assert first["coverage"]["topic_coverage"] == 1.0
        # snake+camel compat on the artifact id.
        assert first["artifact_id"] == first["artifactId"] == first["id"]

        # A pinned version opens THAT note, not only the newest one.
        pinned = (await app_client.get(
            f"/api/academic/notes/{project.id}?version=1"
        )).json()
        assert pinned["state"] == "ready"
        assert pinned["version"] == 1 and pinned["artifact_id"] == "artifact-1"
        assert pinned["pages"] == 4 and pinned["mode"] == "quick"

        # Both versions stream their own bytes from the same viewer endpoint.
        for version, expected in ((1, notes[0].content["pdf_path"]),
                                  (2, notes[1].content["pdf_path"])):
            resp = await app_client.get(
                f"/api/academic/notes/{project.id}/pdf?version={version}"
            )
            assert resp.status_code == 200, resp.text
            assert resp.content == b"%PDF-1.4"
            assert "inline" in resp.headers["content-disposition"]

        # An unknown version is an honest 404, never someone else's PDF.
        assert (
            await app_client.get(f"/api/academic/notes/{project.id}/pdf?version=99")
        ).status_code == 404
        assert (
            await app_client.get(f"/api/academic/notes/{project.id}?version=99")
        ).json()["state"] == "not_generated"
    finally:
        app.dependency_overrides.clear()


async def test_academic_notes_delete_is_ownership_scoped(
    app_client, dummy_db, monkeypatch, user_a, user_b
):
    from main import app
    from services.auth.deps import get_current_user

    project = _make_project(user_a)
    note = _note_row(project.id, "n", datetime(2026, 9, 25), _notes_content("a1", 1, 3))
    storage = _FakeStorage()
    storage.objects[note.content["pdf_path"]] = b"%PDF"
    versions = [
        {"id": "a1", "version": 1, "status": "ready", "path": note.content["pdf_path"],
         "pages": 3, "created_at": "2026-09-25T12:00:00+00:00", "config": {}},
    ]
    app2, _rows = _notes_app(
        monkeypatch, dummy_db, project, [note], user_a, storage=storage,
        versions=versions,
    )
    try:
        async def stranger():
            return user_b

        app2.dependency_overrides[get_current_user] = stranger
        assert (
            await app_client.delete(f"/api/academic/notes/{project.id}/artifacts/a1")
        ).status_code == 404

        async def owner():
            return user_a

        app2.dependency_overrides[get_current_user] = owner
        assert (
            await app_client.delete(f"/api/academic/notes/{project.id}/artifacts/nope")
        ).status_code == 404
        resp = await app_client.delete(f"/api/academic/notes/{project.id}/artifacts/a1")
        assert resp.status_code == 200, resp.text
        assert resp.json()["deleted"] is True
        assert (await app_client.get(f"/api/academic/notes/{project.id}")).json()["state"] == "not_generated"
    finally:
        app.dependency_overrides.clear()


async def test_knowledge_repo_delete_promotes_remaining_ready_version():
    """Removing the only READY version must not leave a dangling read."""

    class _Session:
        def __init__(self, node):
            self.node = node

        async def flush(self):
            return None

        async def get(self, _model, node_id):
            return self.node if str(self.node.id) == str(node_id) else None

    node = SimpleNamespace(
        id=uuid.uuid4(),
        node_metadata={
            "generated_artifacts": {"notes_pdf": [
                {"id": "a1", "version": 1, "status": "superseded", "superseded_by": 2},
                {"id": "a2", "version": 2, "status": "ready"},
            ]},
        },
    )
    db = _Session(node)
    removed = await knowledge_repo.delete_generated_artifact_version(
        db, node.id, "notes_pdf", artifact_id="a2"
    )
    assert removed is not None and removed["id"] == "a2"
    versions = node.node_metadata["generated_artifacts"]["notes_pdf"]
    assert [v["id"] for v in versions] == ["a1"]
    assert versions[0]["status"] == "ready"
    assert "superseded_by" not in versions[0]
    assert await knowledge_repo.delete_generated_artifact_version(
        db, node.id, "notes_pdf", artifact_id="missing"
    ) is None


# ---------------------------------------------------------------------------
# 2a. Truncated-JSON recovery in the shared structured path
# ---------------------------------------------------------------------------


def test_close_truncated_json_keeps_written_text_and_drops_dangling_keys():
    # A cut-off paragraph is recovered, not thrown away.
    assert _extract_json('{"body": ["first paragraph is complete", "second got cut') == {
        "body": ["first paragraph is complete", "second got cut"]
    }
    # A key whose value never arrived is dropped, not half-kept.
    assert _extract_json('{"a": 1, "b": ') == {"a": 1}
    # Nested containers are closed in the right order.
    assert _extract_json('{"a": {"b": [1, 2') == {"a": {"b": [1, 2]}}
    # A cut keyword/number is not salvaged as a value.
    assert _extract_json('{"a": true, "b": nul') == {"a": True}
    # Unrecoverable text still fails loudly instead of inventing content.
    assert _close_truncated_json("definitely not json {{{") == ""
    with pytest.raises(StructuredGenerationError) as err:
        _extract_json("definitely not json {{{", allow_truncation_repair=False)
    assert err.value.truncated is True


def test_truncated_error_is_typed_and_flagged():
    with pytest.raises(StructuredGenerationError) as err:
        _extract_json('{"a": "cut', allow_truncation_repair=False)
    assert "truncated" in str(err.value)
    assert err.value.truncated is True
    assert err.value.raw == '{"a": "cut'


class _ScriptedProvider:
    def __init__(self, texts, finish_reasons=None):
        self.texts = list(texts)
        self.finish_reasons = list(finish_reasons or [])
        self.calls = 0
        self.prompts = []
        self.configs = []

    async def generate(self, prompt, config=None):
        from services.strategies.base import GenerationResult

        self.calls += 1
        self.prompts.append(prompt)
        self.configs.append(config)
        text = self.texts[min(self.calls - 1, len(self.texts) - 1)]
        reason = (
            self.finish_reasons[min(self.calls - 1, len(self.finish_reasons) - 1)]
            if self.finish_reasons
            else "stop"
        )
        return GenerationResult(
            text=text, model="stub", usage={}, finish_reason=reason,
            processing_time=0.0,
        )


def _topic_payload(**overrides):
    payload = {
        "unit_title": "Unit 1",
        "topic_title": "Topic A",
        "heading": "Topic A",
        "body_paragraphs": ["A complete paragraph of notes."],
        "key_terms": ["term"],
        "shape": "concept",
        "citations": [{"chunk_id": "c1", "span": "a complete paragraph"}],
    }
    payload.update(overrides)
    return payload


async def test_truncated_provider_output_is_recovered_without_a_second_call():
    truncated = json.dumps(_topic_payload())[:-25]
    provider = _ScriptedProvider([truncated], ["length"])
    out = await generate_structured(provider, TopicNotesOutput, "prompt")
    assert out.topic_title == "Topic A"
    assert out.body_paragraphs  # recovered prose, not a re-ask
    assert provider.calls == 1


async def test_length_stop_retries_with_a_shorter_answer_request():
    good = json.dumps(_topic_payload())
    # Unrecoverable truncation (nested openers, nothing complete to keep):
    # the only honest move is to ask again for a smaller answer.
    provider = _ScriptedProvider(
        ['{"unit_title": [[[[', good],
        ["length", "stop"],
    )
    out = await generate_structured(provider, TopicNotesOutput, "prompt")
    assert out.topic_title == "Topic A"
    assert provider.calls == 2
    # The retry asks for LESS output instead of the same oversized payload.
    assert "cut off" in provider.prompts[1]
    assert "half as many paragraphs" in provider.prompts[1]


async def test_truncation_recovery_is_bounded_and_fails_honestly():
    provider = _ScriptedProvider(['{"unit_title": "U", "body_paragraphs": ["cut'],
                                 finish_reasons=["length"])
    with pytest.raises(StructuredGenerationError) as err:
        await generate_structured(provider, TopicNotesOutput, "prompt", max_repairs=2)
    assert provider.calls == 3
    assert MAX_TRUNCATION_RECOVERIES == 2
    assert "truncated" in str(err.value)
    assert err.value.truncated is True


async def test_per_call_output_budget_is_configurable_and_bounded():
    provider = _ScriptedProvider([json.dumps(_topic_payload())])
    await generate_structured(provider, TopicNotesOutput, "prompt")
    assert provider.configs[0].max_tokens == 4096  # shared default, unchanged

    provider2 = _ScriptedProvider([json.dumps(_topic_payload())])
    await generate_structured(provider2, TopicNotesOutput, "prompt", max_tokens=8192)
    assert provider2.configs[0].max_tokens == 8192


# ---------------------------------------------------------------------------
# 2b. One topic is generated in bounded batches, then merged
# ---------------------------------------------------------------------------


def test_split_topic_evidence_cuts_on_chunk_boundaries_in_source_order():
    from services.academic.notes_hier import split_topic_evidence

    rows = [{"id": f"c{i}", "text": "x" * 400} for i in range(6)]
    batches = split_topic_evidence(rows, budget_tokens=200)
    assert len(batches) > 1
    assert [r["id"] for batch in batches for r in batch] == [f"c{i}" for i in range(6)]
    assert all(batch for batch in batches)
    # A single chunk larger than the budget is kept whole, never dropped.
    huge = [{"id": "big", "text": "y" * 5000}]
    assert split_topic_evidence(huge, budget_tokens=100) == [huge]
    assert split_topic_evidence([], budget_tokens=100) == [[]]


def test_merge_topic_payloads_keeps_order_and_deduplicates():
    from services.academic.notes_hier import merge_topic_payloads

    merged = merge_topic_payloads([
        _topic_payload(body_paragraphs=["one"], key_terms=["a"],
                       citations=[{"chunk_id": "c1", "span": "s1"}]),
        _topic_payload(body_paragraphs=["one", "two"], key_terms=["A", "b"],
                       citations=[{"chunk_id": "c1", "span": "s1"},
                                  {"chunk_id": "c2", "span": "s2"}]),
    ])
    assert merged["body_paragraphs"] == ["one", "two"]
    assert merged["key_terms"] == ["a", "b"]
    assert len(merged["citations"]) == 2
    assert merged["unit_title"] == "Unit 1"
    assert merge_topic_payloads([]) == {}
    assert merge_topic_payloads([_topic_payload()]) == _topic_payload()


async def test_topic_generation_splits_batches_and_raises_the_per_call_budget(
    monkeypatch,
):
    """A large topic is answered in bounded calls that each fit the budget."""
    from services.academic import orchestrator
    from services.academic.notes_hier import NOTES_TOPIC_MAX_OUTPUT_TOKENS
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    calls = []

    class _Engine(TransformationEngine):
        async def _generate_structured(self, mode, prompt, allowed, max_tokens=None):
            calls.append((prompt, max_tokens, tuple(sorted(allowed))))
            assert mode is GenerationMode.NOTES_TOPIC
            return _topic_payload(
                body_paragraphs=[f"paragraph for prompt {len(calls)}"],
                citations=[{"chunk_id": "c1", "span": "a complete paragraph"}],
            )

    async def fake_rows(*_args, **_kwargs):
        return [{"id": f"c{i}", "text": "x" * 400, "page": 1, "position": i}
                for i in range(6)]

    monkeypatch.setattr(orchestrator, "retrieve_topic_rows", fake_rows)
    engine = _Engine.__new__(_Engine)
    engine.llm = None

    structure = _structure("Topic A", ("c0",))
    topic = structure.units[0].topics[0]
    notes = await orchestrator.generate_topic_live_or_deterministic(
        None, "p", engine, structure, "Unit 1", topic,
        [{"id": "c0", "text": "x"}],
        use_retrieval=False,
        max_output_tokens=NOTES_TOPIC_MAX_OUTPUT_TOKENS,
        batch_tokens=200,
    )
    assert len(calls) > 1, "a 6-chunk topic must not be one oversized call"
    assert all(max_tokens == NOTES_TOPIC_MAX_OUTPUT_TOKENS for _p, max_tokens, _a in calls)
    assert notes.body_paragraphs and len(notes.body_paragraphs) == len(calls)
    assert "batch" in notes.provenance_note
    assert all(citations.chunk_id == "c1" for citations in notes.citations)


async def test_topic_failure_names_the_topic_and_the_reason(monkeypatch):
    """A still-failing topic fails the job honestly, with its name."""
    from services.academic import orchestrator

    async def boom(*_args, **_kwargs):
        raise StructuredGenerationError("LLM output JSON was truncated", truncated=True)

    monkeypatch.setattr(orchestrator, "generate_topic_live_or_deterministic", boom)
    structure = _structure("Big Topic", ("c0",))
    with pytest.raises(orchestrator.AcademicPipelineError) as err:
        await orchestrator._generate_topic_with_honest_error(
            None, "p", object(), structure, "Unit 1",
            structure.units[0].topics[0], [{"id": "c0", "text": "x"}],
        )
    assert "Big Topic" in str(err.value)
    assert "truncated" in str(err.value)


def test_notes_config_identity_drops_advisory_recommendation():
    from services.academic.orchestrator import notes_config_identity
    from services.academic.types import NotesGenerationConfig

    base = NotesGenerationConfig(mode="detailed", detail="balanced")
    same_with_recommendation = base.model_copy(
        update={"recommendation": {"recommended_mode": "exam", "reason": "changed"}}
    )
    assert notes_config_identity(base) == notes_config_identity(same_with_recommendation)
    changed = base.model_copy(update={"detail": "comprehensive"})
    assert notes_config_identity(base) != notes_config_identity(changed)
    assert "recommendation" not in notes_config_identity(same_with_recommendation)
    assert notes_config_identity(None) == {}


async def test_notes_config_exposes_topic_budgets_with_bounded_defaults():
    from services.academic.types import NotesGenerationConfig

    config = NotesGenerationConfig()
    assert config.topic_batch_tokens == 12000
    assert config.topic_output_tokens == 8192
    persisted = config.persisted_dict()
    assert persisted["topic_output_tokens"] == 8192
    with pytest.raises(Exception):
        NotesGenerationConfig(topic_output_tokens=1)  # ge=512


async def test_reused_spine_returns_root_ids_not_row_objects(monkeypatch):
    """A second notes run reuses the spine: it must get node IDS.

    Regression: the reuse branch returned the ORM rows themselves, so the
    ``root_ids`` written into the resumable job state held ``str(node)`` reprs
    and the next knowledge_repo call raised an asyncpg invalid-UUID error -
    the second generation on an unchanged project could never complete.
    """
    from services.academic import orchestrator

    node_id = uuid.uuid4()

    async def fake_structures(_db, _project_id, document_ids=None):
        return [], [SimpleNamespace(doc_id="doc-1", units=[])]

    async def fake_existing(_db, _project_id):
        return 1, SimpleNamespace(id=uuid.uuid4()), "same-fingerprint"

    async def fake_nodes(_db, project_id=None, document_id=None):
        return [
            SimpleNamespace(
                id=node_id, parent_id=None, model_version=1, node_metadata={}
            )
        ]

    async def fake_merge(structures, project_title="Academic project"):
        return SimpleNamespace(units=[])

    monkeypatch.setattr(orchestrator, "build_source_structures", fake_structures)
    monkeypatch.setattr(orchestrator, "_existing_model_version", fake_existing)
    monkeypatch.setattr(orchestrator, "merge_document_structures", fake_merge)
    monkeypatch.setattr(orchestrator.knowledge_repo, "list_knowledge_nodes", fake_nodes)
    monkeypatch.setattr(orchestrator, "_source_fingerprint", lambda _s: "same-fingerprint")

    result = await orchestrator.build_project_spine(
        None, str(uuid.uuid4()), project_title="T", force=False
    )
    assert result.reused is True
    assert list(result.root_ids) == [node_id]
    for value in result.root_ids:
        assert not isinstance(value, str), "root ids must be ids, not stringified rows"


# ---------------------------------------------------------------------------
# 2c. A headings-only artifact still cannot be marked completed
# ---------------------------------------------------------------------------


def _headings_only_artifact():
    """A headings-only artifact shaped like real model output (no prose)."""
    return {
        "title": "Notes",
        "provenance_note": "recovered from a truncated answer",
        "config": {},
        "sections": [{
            "unit_title": "Unit 1",
            "topic_title": "Topic A",
            "heading": "Topic A",
            "content": "Topic A",
            "citations": [],
        }],
    }


async def test_headings_only_artifact_is_never_accepted():
    """The truncation recovery must not weaken the hard gates."""
    from services.academic.types import NotesArtifact
    from services.academic.validation import validate_notes_artifact

    structure = _structure("Topic A", ("c1",))
    headings_only = NotesArtifact.model_validate(_headings_only_artifact())
    report = validate_notes_artifact(
        headings_only, structure, {"c1"}, {"c1": "a complete paragraph of source text"},
    )
    assert report.ok is False
    codes = {error["code"] for error in report.errors}
    assert "section_too_short" in codes or "section_missing_citations" in codes


async def test_recovered_truncated_topic_output_still_fails_headings_only_gate():
    """A recovered payload with a heading and no prose stays rejected."""
    from services.academic.types import NotesArtifact
    from services.academic.validation import validate_notes_artifact

    # Exactly what a truncated answer recovers to when the model wrote a
    # heading and nothing else: valid JSON, no body.
    recovered = json.dumps(_headings_only_artifact())
    assert _extract_json(recovered, allow_truncation_repair=False) is not None
    structure = _structure("Topic A", ("c1",))
    artifact = NotesArtifact.model_validate(_extract_json(recovered))
    report = validate_notes_artifact(
        artifact, structure, {"c1"}, {"c1": "a complete paragraph of source text"},
    )
    assert report.ok is False


# ---------------------------------------------------------------------------
# 2d. Frontend wiring mirrors (no vitest runner is wired in this package)
# ---------------------------------------------------------------------------


def _frontend_source(name):
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "..", "trash-the-test", "src", *name.split("/"))
    with open(os.path.normpath(path), encoding="utf-8") as handle:
        return handle.read()


def test_frontend_renders_the_notes_list_with_open_and_delete():
    component = _frontend_source("components/NotesArtifactList.tsx")
    assert "deleteNote" in component
    assert "Confirm delete" in component and "Delete" in component
    assert "Open" in component
    # Delete refetches both read models.
    assert 'invalidateQueries({ queryKey: ["notes", projectId] })' in component
    assert 'invalidateQueries({ queryKey: ["study-notes-pdf", projectId] })' in component
    # Open targets the specific note's version, not only the newest.
    assert "/documents/study-notes-${projectId}?version=" in component

    active = _frontend_source("pages/ActiveStudy.tsx")
    assert "NotesArtifactList" in active
    study = _frontend_source("pages/StudyMode.tsx")
    assert "NotesArtifactList" in study
    viewer = _frontend_source("pages/DocumentViewer.tsx")
    assert "studyNotesVersion" in viewer
    api = _frontend_source("lib/api-services.ts")
    assert "export const deleteNote" in api
    assert "artifacts?: StudyNotesArtifactSummary[]" in api


def test_notes_list_item_carries_the_read_metadata():
    api = _frontend_source("lib/api-services.ts")
    for field in ("mode?:", "coverage?:", "pages?:", "createdAt:", "artifactId?:"):
        assert field in api, field
