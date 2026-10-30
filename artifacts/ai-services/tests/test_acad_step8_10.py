"""STEPS 8-10 tests: notes PDF renderer -> versioned artifacts -> read paths.

No DB, no LLM, no network. The ReportLab renderer runs offline over the
synthetic corpus; artifact versioning is pure; the knowledge_repo
wrappers run against a minimal fake session (live Postgres NOT RUN);
routes run over ASGI with dependency overrides + monkeypatched repo and
storage fakes. No test depends on a live provider.
"""

import io
import os
import uuid
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from database import knowledge_repo
from services.academic import artifacts as academic_artifacts
from services.academic.coverage import attach_report, evaluate_coverage
from services.academic.extraction import extract_structure
from services.academic.metrics import (
    compute_metrics,
    metrics_artifact,
    normalize_inputs,
)
from services.academic.notes_hier import (
    render_topic_notes,
    synthesize_notes,
)
from services.academic.notes_pdf import (
    NotesPdfReport,
    notes_pdf_storage_hint,
    render_notes_pdf,
)
from services.academic.types import (
    GeneratedArtifact,
    artifact_fingerprint,
    not_generated_state,
)
from tests.eval.corpus import CORPUS

pytestmark = pytest.mark.asyncio

PINNED_EPOCH = "1704067200"


def _doc_chunks(doc_id):
    return [dict(r) for r in CORPUS if r["doc"] == doc_id]


def _chunk_texts(doc_id):
    return {r["id"]: r["text"] for r in _doc_chunks(doc_id)}


def _bio_fixture():
    chunks = _doc_chunks("doc-biology")
    structure = extract_structure("doc-biology", "Photosynthesis primer", chunks)
    notes = [
        render_topic_notes(unit.title, topic, chunks)
        for unit in structure.units
        for topic in unit.topics
    ]
    artifact = synthesize_notes(
        "Photosynthesis primer", structure, notes, chunk_texts=_chunk_texts("doc-biology")
    )
    report = evaluate_coverage(structure, notes)
    return structure, attach_report(artifact, report)


def _meta(**over):
    base = {
        "title": "Photosynthesis primer",
        "version": 1,
        "model": "test-model",
        "created_at": "2026-09-23",
        "epoch": int(PINNED_EPOCH),
    }
    base.update(over)
    return base


def _flat_outline(reader):
    out = []

    def _walk(items):
        for item in items:
            if isinstance(item, list):
                _walk(item)
            else:
                out.append(item.title)

    _walk(list(reader.outline))
    return out


# ---------------------------------------------------------------------------
# STEP 8: renderer structure
# ---------------------------------------------------------------------------


async def test_pdf_renders_valid_extractable_pdf_with_metadata(monkeypatch):
    """AC-J3 (part): non-empty PDF, extractable text, title/author metadata."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    structure, artifact = _bio_fixture()
    report = render_notes_pdf(artifact, structure, _meta())
    assert isinstance(report, NotesPdfReport)
    assert report.page_count > 0
    assert len(report.pdf_bytes) > 0
    assert len(report.sha256) == 64
    reader = PdfReader(io.BytesIO(report.pdf_bytes))
    assert len(reader.pages) == report.page_count
    assert reader.metadata.title == "Photosynthesis primer"
    assert reader.metadata.author == "TrashTheTest"
    body = "\n".join((page.extract_text() or "") for page in reader.pages)
    assert "TRASHTHETEST" in body
    assert "Contents" in body


async def test_pdf_body_contains_every_topic_heading(monkeypatch):
    """AC-J3: every spine topic heading lands in the body text."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    structure, artifact = _bio_fixture()
    report = render_notes_pdf(artifact, structure, _meta())
    body = _norm_ws("\n".join(
        (page.extract_text() or "") for page in PdfReader(io.BytesIO(report.pdf_bytes)).pages
    ))
    topics = [topic.title for unit in structure.units for topic in unit.topics]
    assert topics, "fixture must carry topics"
    for title in topics:
        assert title in body, f"topic heading missing from PDF body: {title!r}"
    assert report.topic_count == len(topics)


async def test_pdf_outline_has_unit_and_topic_entries(monkeypatch):
    """AC-J3: bookmarks hold one entry per unit and per topic."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    structure, artifact = _bio_fixture()
    report = render_notes_pdf(artifact, structure, _meta())
    titles = _flat_outline(PdfReader(io.BytesIO(report.pdf_bytes)))
    units = [unit.title for unit in structure.units]
    topics = [topic.title for unit in structure.units for topic in unit.topics]
    for unit in units:
        assert any(unit in entry for entry in titles), f"unit missing from outline: {unit!r}"
    for topic in topics:
        assert any(topic in entry for entry in titles), f"topic missing from outline: {topic!r}"
    assert report.bookmark_count == len(report.toc_entries) >= len(units) + len(topics)


async def test_pdf_toc_page_numbers_match_heading_pages(monkeypatch):
    """AC-J3: TOC page numbers match the pages the headings land on."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    structure, artifact = _bio_fixture()
    report = render_notes_pdf(artifact, structure, _meta())
    reader = PdfReader(io.BytesIO(report.pdf_bytes))
    pages_text = [_norm_ws(page.extract_text() or "") for page in reader.pages]
    topic_entries = [e for e in report.toc_entries if e["level"] == 1]
    assert topic_entries, "TOC must list topics"
    for entry in topic_entries:
        assert 1 <= entry["page"] <= len(pages_text), entry
        tail = entry["title"].split("  ")[-1][:24]
        assert tail in pages_text[entry["page"] - 1], (
            f"TOC entry {entry['title']!r} points at page {entry['page']} "
            "where the heading does not land"
        )


def _norm_ws(text: str) -> str:
    """Collapse all whitespace (pypdf inserts line breaks mid-cell)."""
    return " ".join((text or "").split())


async def test_pdf_tables_and_cites_survive(monkeypatch):
    """AC-J3: key-term tables + explicit tables keep contiguous cell values; cites listed."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    structure, artifact = _bio_fixture()
    first_unit = structure.units[0]
    first_topic = first_unit.topics[0]
    extras = {
        (first_unit.title, first_topic.title): {
            "key_terms": ["chlorophyll", "thylakoid", "stroma", "atp synthase"],
            "table": [
                ["Pigment", "Absorbs", "Role"],
                ["Chlorophyll a", "red and blue", "reaction centre"],
                ["Chlorophyll b", "blue and orange", "antenna"],
            ],
        }
    }
    report = render_notes_pdf(artifact, structure, _meta(), extras)
    body = _norm_ws("\n".join(
        (page.extract_text() or "") for page in PdfReader(io.BytesIO(report.pdf_bytes)).pages
    ))
    for term in ("chlorophyll", "thylakoid", "stroma", "atp synthase"):
        assert term in body, f"key term lost from table: {term!r}"
    for cell in ("Chlorophyll a", "reaction centre", "antenna", "Absorbs"):
        assert cell in body, f"table cell lost: {cell!r}"
    assert "Sources:" in body


async def test_pdf_determinism_pinned_epoch(monkeypatch):
    """AC-J4: same artifact twice with pinned SOURCE_DATE_EPOCH -> identical sha256."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    structure, artifact = _bio_fixture()
    first = render_notes_pdf(artifact, structure, _meta())
    second = render_notes_pdf(artifact, structure, _meta())
    assert first.sha256 == second.sha256
    assert first.pdf_bytes == second.pdf_bytes


async def test_pdf_no_clip_long_title_wide_table_and_cjk(monkeypatch):
    """Edge cases: 300-char title, 6-col wide table, non-Latin title never clips/crashes."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    from pypdf import PdfReader

    structure, artifact = _bio_fixture()
    first_unit = structure.units[0]
    first_topic = first_unit.topics[0]
    wide = [["H1", "H2", "H3", "H4", "H5", "H6"]]
    wide += [[f"cell row {r} col {c} with a long wrapped value" for c in range(6)] for r in range(4)]
    extras = {(first_unit.title, first_topic.title): {"table": wide}}
    long_title = "L" * 300
    report = render_notes_pdf(artifact, structure, _meta(title=long_title), extras)
    assert report.page_count > 0
    body = _norm_ws("\n".join(
        (page.extract_text() or "") for page in PdfReader(io.BytesIO(report.pdf_bytes)).pages
    ))
    assert "L" * 20 in body
    assert "cell row 3 col 5 with a long wrapped value" in body
    cjk = render_notes_pdf(artifact, structure, _meta(title="光合作用 primer"), extras)
    assert cjk.page_count > 0
    cjk_body = _norm_ws("\n".join(
        (page.extract_text() or "") for page in PdfReader(io.BytesIO(cjk.pdf_bytes)).pages
    ))
    assert len(cjk_body.strip()) > 0


async def test_pdf_empty_sections_rejected():
    """No silent empty PDF: rendering without sections raises honestly."""
    with pytest.raises(ValueError):
        render_notes_pdf({"title": "Empty", "sections": []}, None, _meta())


async def test_pdf_storage_hint_is_safe_filename():
    hint = notes_pdf_storage_hint("proj-123_abc", 2)
    assert hint.endswith(".pdf") and "v2" in hint
    assert ".." not in hint and "/" not in hint


# ---------------------------------------------------------------------------
# STEP 9: GeneratedArtifact model + versioning
# ---------------------------------------------------------------------------


def _artifact_dict(version=1, status="ready", fingerprint=None, **over):
    record = {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "sources": ["doc-1"],
        "type": "notes_pdf",
        "version": version,
        "status": status,
        "path": f"notes-pdf/v{version}.pdf",
        "pages": 10,
        "hash": "a" * 64,
        "model": "test-model",
        "config": {"topic_count": 3},
        "coverage": {"topic_coverage": 1.0},
        "fingerprint": fingerprint or "f" * 64,
        "created_at": "2026-09-23T00:00:00+00:00",
        "updated_at": "2026-09-23T00:00:00+00:00",
    }
    record.update(over)
    return record


async def test_generated_artifact_schema_and_aliases():
    """STEP 9 minimum fields; wire aliases sources/type/hash round-trip."""
    record = GeneratedArtifact.model_validate(_artifact_dict())
    assert record.version == 1 and record.pages == 10
    dumped = record.model_dump(by_alias=True)
    assert dumped["sources"] == ["doc-1"] and dumped["type"] == "notes_pdf"
    assert dumped["hash"] == "a" * 64
    with pytest.raises(ValidationError):
        GeneratedArtifact.model_validate({**_artifact_dict(), "bogus": "nope"})
    with pytest.raises(ValidationError):
        GeneratedArtifact.model_validate({**_artifact_dict(), "hash": "short"})


async def test_artifact_fingerprint_traceable_and_order_invariant():
    first = artifact_fingerprint(["b", "a"], "m", {"k": 1}, {"topic_coverage": 1.0})
    assert first == artifact_fingerprint(["a", "b"], "m", {"k": 1}, {"topic_coverage": 1.0})
    assert first != artifact_fingerprint(["a", "b"], "other", {"k": 1}, {"topic_coverage": 1.0})
    assert first != artifact_fingerprint(["a", "b"], "m", {"k": 2}, {"topic_coverage": 1.0})
    assert first != artifact_fingerprint(["a", "b"], "m", {"k": 1}, {"topic_coverage": 0.5})
    assert len(first) == 64


async def test_assign_artifact_version_short_circuits_unchanged():
    existing = [_artifact_dict(version=1, fingerprint="same")]
    version, match = knowledge_repo.assign_artifact_version(existing, "same")
    assert (version, match["version"]) == (1, 1)
    version, match = knowledge_repo.assign_artifact_version(existing, "changed")
    assert (version, match) == (2, None)
    assert knowledge_repo.assign_artifact_version([], "new") == (1, None)


async def test_build_notes_pdf_artifact_versions_and_reuses(monkeypatch):
    """AC-J1/J2: first build v1; unchanged rebuild reuses; changed rebuild v2."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", PINNED_EPOCH)
    structure, artifact = _bio_fixture()
    record, report, reused = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m1", config={"project_id": "p1"})
    assert record["version"] == 1 and reused is False
    assert record["pages"] == report.page_count == report.page_count
    assert record["path"].endswith(".pdf")
    same, _, reused_same = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m1", config={"project_id": "p1"},
        existing_versions=[record])
    assert reused_same is True and same["version"] == 1
    assert same["id"] == record["id"], "short-circuit returns the existing artifact"
    nxt, _, reused_nxt = academic_artifacts.build_notes_pdf_artifact(
        artifact, structure, model="m2", config={"project_id": "p1"},
        existing_versions=[record])
    assert reused_nxt is False and nxt["version"] == 2


class _FakeSession:
    """Minimal stand-in: node lookup by id + flush (no real database)."""

    def __init__(self, nodes):
        self._nodes = {n.id: n for n in nodes}

    async def get(self, _model, node_id):
        return self._nodes.get(node_id)

    async def flush(self):
        return None


def _doc_node(node_id, metadata=None):
    return SimpleNamespace(id=node_id, node_metadata=dict(metadata or {}))


async def test_generated_artifact_store_supersede_and_reread():
    """AC-J1: finalize supersedes the previous ready row; old versions stay readable."""
    node_id = uuid.uuid4()
    db = _FakeSession([_doc_node(node_id)])
    first = _artifact_dict(version=1, fingerprint="1" * 64)
    await knowledge_repo.save_generated_artifact(db, node_id, first)
    second = _artifact_dict(version=2, fingerprint="2" * 64)
    await knowledge_repo.save_generated_artifact(db, node_id, second)
    latest = await knowledge_repo.get_generated_artifact(db, node_id, "notes_pdf")
    assert latest["version"] == 2 and latest["status"] == "ready"
    old = await knowledge_repo.get_generated_artifact(db, node_id, "notes_pdf", version=1)
    assert old["version"] == 1 and old["superseded_by"] == 2
    listed = await knowledge_repo.list_generated_artifacts(db, node_id, "notes_pdf")
    assert [v["version"] for v in listed] == [1, 2]


async def test_generated_artifact_store_missing_node_and_type():
    db = _FakeSession([])
    assert await knowledge_repo.get_generated_artifact(db, uuid.uuid4(), "notes_pdf") is None
    assert await knowledge_repo.list_generated_artifacts(db, uuid.uuid4()) == []
    with pytest.raises(ValueError):
        await knowledge_repo.save_generated_artifact(db, uuid.uuid4(), _artifact_dict())


async def test_notes_pdf_status_ready_and_not_generated(monkeypatch):
    """STEP 10 read model: metadata topics/pages/coverage/date/model, or honest empty."""
    project_id = uuid.uuid4()
    root = _doc_node(uuid.uuid4())
    stored = _artifact_dict(version=3, fingerprint="3" * 64)
    stored["config"] = {"topic_count": 11, "section_count": 12}

    async def fake_roots(db, project_id=None):
        return [root]

    async def fake_list(db, node_id, artifact_type=None):
        assert artifact_type == "notes_pdf"
        return [dict(stored)]

    monkeypatch.setattr(knowledge_repo, "list_knowledge_nodes", fake_roots)
    monkeypatch.setattr(knowledge_repo, "list_generated_artifacts", fake_list)
    state = await academic_artifacts.get_notes_pdf_status(SimpleNamespace(), project_id)
    assert state["state"] == "ready"
    assert state["version"] == 3 and state["pages"] == 10
    assert state["topics"] == 11 and state["model"] == "test-model"
    assert state["coverage"]["topic_coverage"] == 1.0
    assert state["created_at"].startswith("2026-09-23")
    assert state["sha256"] == "a" * 64
    # The read model lists EVERY stored version, not just the latest one.
    assert state["total"] == 1
    assert state["artifacts"][0]["version"] == 3
    assert state["artifacts"][0]["pages"] == 10

    # A pinned version resolves that version; an unknown version is honest.
    pinned = await academic_artifacts.get_notes_pdf_status(
        SimpleNamespace(), project_id, version=3
    )
    assert pinned["state"] == "ready" and pinned["version"] == 3
    missing = await academic_artifacts.get_notes_pdf_status(
        SimpleNamespace(), project_id, version=99
    )
    assert missing["state"] == "not_generated"

    async def fake_empty(db, node_id, artifact_type=None):
        return []

    monkeypatch.setattr(knowledge_repo, "list_generated_artifacts", fake_empty)
    empty = await academic_artifacts.get_notes_pdf_status(SimpleNamespace(), project_id)
    assert empty["state"] == "not_generated"
    assert academic_artifacts.NOTES_GENERATION_CALL in empty["generation_call"]
    assert empty["artifacts"] == [] and empty["total"] == 0


async def test_insights_status_serves_tables_and_cards(monkeypatch):
    """STEP 10 insights: backend-first tables/cards payload, or honest empty."""
    project_id = uuid.uuid4()
    root = _doc_node(uuid.uuid4())
    chunks = _doc_chunks("doc-biology")
    structure = extract_structure("doc-biology", "Photosynthesis primer", chunks)
    normalized = normalize_inputs(structure, [{"id": r["id"], "text": r["text"]} for r in chunks])
    payload = metrics_artifact(compute_metrics(normalized))

    async def fake_roots(db, project_id=None):
        return [root]

    async def fake_get(db, node_id, key):
        return dict(payload)

    monkeypatch.setattr(knowledge_repo, "list_knowledge_nodes", fake_roots)
    monkeypatch.setattr(knowledge_repo, "get_academic_artifact", fake_get)
    state = await academic_artifacts.get_insights_status(SimpleNamespace(), project_id)
    assert state["state"] == "ready"
    assert len(state["table"]) == len(payload["tables"]) > 0
    assert state["cards"] and state["totals"]

    async def fake_empty(db, node_id, key):
        return None

    monkeypatch.setattr(knowledge_repo, "get_academic_artifact", fake_empty)
    empty = await academic_artifacts.get_insights_status(SimpleNamespace(), project_id)
    assert empty["state"] == "not_generated"
    assert academic_artifacts.INSIGHTS_GENERATION_CALL in empty["generation_call"]


async def test_not_generated_state_names_generation_call():
    state = not_generated_state("study-notes-pdf", "POST /api/generation/notes")
    assert state["state"] == "not_generated"
    assert "POST /api/generation/notes" in state["message"]


# ---------------------------------------------------------------------------
# STEP 10: routes (ASGI, ownership enforced, bytes match the artifact row)
# ---------------------------------------------------------------------------


def _route_harness(monkeypatch, owner_id, project_id, status=None, pdf_bytes=b"",
                   stored_path="notes-pdf/v1.pdf", backend_exists=True):
    import httpx
    from main import app
    from database import repositories
    from database.engine import get_db
    from services.academic import routes as acad_routes
    from services.auth.deps import get_current_user

    project = SimpleNamespace(id=project_id, user_id=owner_id)

    async def fake_get_db():
        yield SimpleNamespace()

    async def fake_user():
        return SimpleNamespace(id=owner_id)

    async def fake_project(db, _pid):
        return project

    async def fake_status(db, _pid, version=None):
        state = dict(status or {})
        if version is not None and state.get("version") != int(version):
            return {"state": "not_generated"}
        return state

    class _FakeBackend:
        async def exists(self, _key):
            return backend_exists

        async def open_stream(self, _key, chunk_size=64 * 1024):
            yield pdf_bytes

    async def fake_roots(db, _pid):
        return [SimpleNamespace(id=uuid.uuid4())]

    async def fake_stored(db, _node_id, _atype=None):
        return [{"id": "artifact-1", "path": stored_path}]

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    monkeypatch.setattr(academic_artifacts, "get_notes_pdf_status", fake_status)
    monkeypatch.setattr(academic_artifacts, "_document_roots", fake_roots)
    monkeypatch.setattr(knowledge_repo, "list_generated_artifacts", fake_stored)
    monkeypatch.setattr(acad_routes, "get_default_backend", lambda: _FakeBackend())
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test")
    return app, client


async def test_routes_notes_meta_ready_and_empty(monkeypatch):
    owner, project_id = uuid.uuid4(), uuid.uuid4()
    ready = {"state": "ready", "version": 2, "topics": 11, "pages": 9,
             "model": "m", "sha256": "b" * 64}
    app, client = _route_harness(monkeypatch, owner, project_id, status=ready)
    try:
        async with client:
            resp = await client.get(f"/api/academic/notes/{project_id}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["version"] == 2
    finally:
        app.dependency_overrides.clear()
    app2, client2 = _route_harness(
        monkeypatch, owner, project_id,
        status={"state": "not_generated", "generation_call": "POST /api/generation/notes"})
    try:
        async with client2:
            resp = await client2.get(f"/api/academic/notes/{project_id}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["state"] == "not_generated"
    finally:
        app2.dependency_overrides.clear()


async def test_routes_notes_pdf_bytes_match_and_missing_key(monkeypatch):
    import hashlib

    owner, project_id = uuid.uuid4(), uuid.uuid4()
    payload = b"%PDF-1.4 fake-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    ready = {"state": "ready", "artifact_id": "artifact-1",
             "title": "Bio notes", "sha256": digest}
    app, client = _route_harness(monkeypatch, owner, project_id,
                                 status=ready, pdf_bytes=payload)
    try:
        async with client:
            resp = await client.get(f"/api/academic/notes/{project_id}/pdf")
            assert resp.status_code == 200, resp.text
            assert resp.headers["content-type"] == "application/pdf"
            assert "inline" in resp.headers["content-disposition"]
            assert hashlib.sha256(resp.content).hexdigest() == digest
    finally:
        app.dependency_overrides.clear()
    app2, client2 = _route_harness(monkeypatch, owner, project_id,
                                   status=ready, pdf_bytes=payload,
                                   backend_exists=False)
    try:
        async with client2:
            resp = await client2.get(f"/api/academic/notes/{project_id}/pdf")
            assert resp.status_code == 404, resp.text
            assert "unavailable" in resp.json()["detail"]
    finally:
        app2.dependency_overrides.clear()


async def test_routes_another_user_gets_404(monkeypatch):
    import httpx
    from main import app
    from database import repositories
    from database.engine import get_db
    from services.auth.deps import get_current_user

    owner, stranger, project_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=owner)

    async def fake_get_db():
        yield SimpleNamespace()

    async def fake_stranger():
        return SimpleNamespace(id=stranger)

    async def fake_project(db, _pid):
        return project

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_stranger
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            for path in (f"/api/academic/notes/{project_id}",
                         f"/api/academic/notes/{project_id}/pdf",
                         f"/api/academic/insights/{project_id}"):
                resp = await client.get(path)
                assert resp.status_code == 404, (path, resp.text)
    finally:
        app.dependency_overrides.clear()


async def test_routes_insights_ready_and_empty(monkeypatch):
    import httpx
    from main import app
    from database import repositories
    from database.engine import get_db
    from services.auth.deps import get_current_user

    owner, project_id = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_id, user_id=owner)
    ready = {"state": "ready", "cards": [{"card": "coverage"}],
             "table": [{"topic": "T1"}], "totals": {}}

    async def fake_get_db():
        yield SimpleNamespace()

    async def fake_user():
        return SimpleNamespace(id=owner)

    async def fake_project(db, _pid):
        return project

    async def fake_status(db, _pid):
        return dict(ready)

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    monkeypatch.setattr(academic_artifacts, "get_insights_status", fake_status)
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            resp = await client.get(f"/api/academic/insights/{project_id}")
            assert resp.status_code == 200, resp.text
            assert resp.json()["table"] == [{"topic": "T1"}]
    finally:
        app.dependency_overrides.clear()


async def test_academic_router_registered_on_app():
    from main import app

    paths = set()

    def _walk(routes):
        for route in routes:
            inner = getattr(route, "original_router", None)
            if inner is not None:
                _walk(getattr(inner, "routes", []))
            elif hasattr(route, "routes"):
                _walk(route.routes)
            else:
                path = getattr(route, "path", "")
                if path:
                    paths.add(path)

    _walk(app.routes)
    assert "/api/academic/notes/{project_id}" in paths
    assert "/api/academic/notes/{project_id}/pdf" in paths
    assert "/api/academic/insights/{project_id}" in paths
