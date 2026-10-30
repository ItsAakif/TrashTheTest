"""Frontend-integration + study-website pipeline tests (20).

Covers: auth/ownership on website manage routes, slug uniqueness + traversal
guard, private/draft rejection on public resolve, published public resolve,
preview/published parity, XSS/URL sanitising with citation preservation,
foreign-content blocking, update/delete ownership, notes/MCQ/flash/answer
rendering, missing-content safety, marks-aware answer route + word band,
MCQ route, and the Puter route (default-OFF explicit degraded).
"""

import uuid
from datetime import datetime

import pytest

from database import repositories
from database.models.project import Project, ProjectStatus
from database.models.website import Website, WebsiteStatus, WebsiteType, WebsiteVisibility

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------

class FakeWebsiteStore:
    def __init__(self):
        self.projects = {}
        self.websites = {}

    def add_project(self, user_id, name="P"):
        p = Project(id=uuid.uuid4(), user_id=user_id, name=name,
                    status=ProjectStatus.PROCESSING, progress=0)
        p.created_at = datetime.utcnow()
        p.updated_at = datetime.utcnow()
        self.projects[p.id] = p
        return p

    def add_website(self, project, slug, published=False, html="<p>Study</p>",
                    wtype="mixed", title="T"):
        w = Website(id=uuid.uuid4(), project_id=project.id, slug=slug,
                    website_type=WebsiteType(wtype), html_content=html)
        w.title = title
        w.is_published = published
        try:
            w.status = WebsiteStatus.PUBLISHED if published else WebsiteStatus.DRAFT
            w.visibility = WebsiteVisibility.PUBLIC if published else WebsiteVisibility.PRIVATE
        except Exception:
            pass
        w.created_at = datetime.utcnow()
        w.updated_at = datetime.utcnow()
        self.websites[slug] = w
        return w


def _install(monkeypatch, dummy_db, store, user):
    from database.engine import get_db
    from main import app
    from services.auth.deps import get_current_user, get_optional_user

    async def fake_get_db():
        yield dummy_db

    async def fake_project(db, pid):
        return store.projects.get(pid)

    async def fake_website_by_slug(db, slug):
        return store.websites.get(slug)

    async def fake_websites_by_project(db, pid):
        return [w for w in store.websites.values() if w.project_id == pid]

    async def fake_slug_exists(db, slug):
        return slug in store.websites

    async def fake_create(db, project_id, slug, website_type, html_content=None,
                          title=None, description=None):
        proj = next((p for p in store.projects.values() if p.id == project_id), None)
        w = store.add_website(proj or list(store.projects.values())[0], slug,
                              html=html_content or "", wtype=website_type,
                              title=title or "T")
        return w

    async def fake_delete(db, website):
        store.websites.pop(website.slug, None)

    async def fake_user():
        return user

    monkeypatch.setattr(repositories, "get_project_by_id", fake_project)
    monkeypatch.setattr(repositories, "get_website_by_slug", fake_website_by_slug)
    monkeypatch.setattr(repositories, "get_websites_by_project", fake_websites_by_project)
    monkeypatch.setattr(repositories, "slug_exists", fake_slug_exists)
    monkeypatch.setattr(repositories, "create_website", fake_create)
    monkeypatch.setattr(repositories, "delete_website", fake_delete)
    app.dependency_overrides[get_db] = fake_get_db
    app.dependency_overrides[get_current_user] = fake_user
    app.dependency_overrides[get_optional_user] = fake_user
    return app


@pytest.fixture
def site(monkeypatch, dummy_db, user_a):
    store = FakeWebsiteStore()
    project = store.add_project(user_a.id)
    from main import app
    _install(monkeypatch, dummy_db, store, user_a)
    try:
        yield store, project
    finally:
        app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# 1-4: answer route (marks-aware) + MCQ route
# --------------------------------------------------------------------------

async def _stub_answer(monkeypatch, text_words=160):
    from services.transformation import engine as enginemod
    from services.transformation.types import GenerationMode, GenerationResult

    async def fake_generate_answer(self, project_id, db, query, marks=5,
                                   style="detailed", enable_research=False):
        from services.transformation.types import constraints_for_marks
        band = constraints_for_marks(int(marks))
        return GenerationResult(
            project_id=project_id, mode=GenerationMode.ANSWER,
            content={"question": query,
                     "answer_text": " ".join(["word"] * (band["min_words"] + 1)),
                     "marks": int(marks)},
            cached=False, processing_time=0.01)

    monkeypatch.setattr(enginemod.TransformationEngine, "generate_answer",
                        fake_generate_answer)


async def test_01_answer_requires_query(app_client, site, monkeypatch):
    _store, project = site
    resp = await app_client.post("/api/generation/answer",
                                 json={"project_id": str(project.id),
                                       "query": "  ", "marks": 5})
    assert resp.status_code == 400
    assert "Query is required" in resp.text


async def test_02_answer_rejects_bad_marks(app_client, site, monkeypatch):
    _store, project = site
    for bad in (0, -3, 101):
        resp = await app_client.post(
            "/api/generation/answer",
            json={"project_id": str(project.id), "query": "What is X?", "marks": bad})
        assert resp.status_code == 400, (bad, resp.text)


async def test_03_answer_returns_word_band(app_client, site, monkeypatch):
    _store, project = site
    await _stub_answer(monkeypatch)
    resp = await app_client.post("/api/generation/answer",
                                 json={"project_id": str(project.id),
                                       "query": "What is X?", "marks": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["marks"] == 5
    assert body["word_band"] == {"min_words": 150, "max_words": 350}
    assert body["mode"] == "answer"


async def test_04_answer_foreign_project_rejected(app_client, site, monkeypatch,
                                                  user_b):
    from main import app
    from services.auth.deps import get_current_user

    async def fake_b():
        return user_b

    app.dependency_overrides[get_current_user] = fake_b
    _store, project = site  # owned by user_a
    resp = await app_client.post("/api/generation/answer",
                                 json={"project_id": str(project.id),
                                       "query": "What is X?", "marks": 2})
    assert resp.status_code == 404


async def test_05_mcq_route_ok(app_client, site, monkeypatch):
    from services.transformation import engine as enginemod
    from services.transformation.types import GenerationMode, GenerationResult

    async def fake_mcq(self, project_id, db, query=None, enable_research=False):
        return GenerationResult(project_id=project_id, mode=GenerationMode.MCQ,
                                content={"title": "T", "questions": []},
                                cached=False, processing_time=0.01)

    monkeypatch.setattr(enginemod.TransformationEngine, "generate_mcq", fake_mcq)
    _store, project = site
    resp = await app_client.post("/api/generation/mcq",
                                 json={"project_id": str(project.id)})
    assert resp.status_code == 200, resp.text
    assert resp.json()["mode"] == "mcq"


# --------------------------------------------------------------------------
# 6-8: auth + ownership + slug uniqueness
# --------------------------------------------------------------------------

async def test_06_website_generate_requires_auth(app_client, dummy_db, monkeypatch):
    from database.engine import get_db
    from main import app

    async def fake_get_db():
        yield dummy_db

    app.dependency_overrides[get_db] = fake_get_db
    try:
        resp = await app_client.post("/api/website/generate",
                                     json={"project_id": str(uuid.uuid4()),
                                           "content_type": "mixed", "content": {}})
        assert resp.status_code in (401, 403), resp.status_code
    finally:
        app.dependency_overrides.clear()


async def test_07_website_generate_foreign_project_404(app_client, site, monkeypatch,
                                                       user_b):
    from main import app
    from services.auth.deps import get_current_user

    async def fake_b():
        return user_b

    app.dependency_overrides[get_current_user] = fake_b
    _store, project = site
    resp = await app_client.post("/api/website/generate",
                                 json={"project_id": str(project.id),
                                       "content_type": "mixed", "content": {}})
    assert resp.status_code == 404


async def test_08_slug_unique_per_generate(app_client, site):
    store, project = site
    first = await app_client.post("/api/website/generate",
                                  json={"project_id": str(project.id),
                                        "content_type": "mixed",
                                        "title": "Biology",
                                        "content": {"sections": []}})
    assert first.status_code == 200, first.text
    second = await app_client.post("/api/website/generate",
                                   json={"project_id": str(project.id),
                                         "content_type": "mixed",
                                         "title": "Biology",
                                         "content": {"sections": []}})
    assert second.status_code == 200, second.text
    assert first.json()["slug"] != second.json()["slug"]


async def test_09_slug_traversal_rejected(app_client, site):
    _store, _project = site
    for bad in ("..", "a/b", "BAD SLUG!", "../x"):
        resp = await app_client.get(f"/api/website/public/{bad}")
        assert resp.status_code in (400, 404, 422), (bad, resp.status_code)
    # At least one must be an explicit 400 (guard, not accidental 404).
    resp = await app_client.get("/api/website/public/BAD SLUG!")
    assert resp.status_code in (400, 422)


# --------------------------------------------------------------------------
# 10-12: public resolve only published / draft hidden / published ok
# --------------------------------------------------------------------------

async def test_10_private_rejected_on_public_resolve(app_client, site):
    store, project = site
    store.add_website(project, "private-site-aaa", published=False)
    assert (await app_client.get("/api/website/public/private-site-aaa")).status_code == 404
    assert (await app_client.get("/api/website/public/private-site-aaa/html")).status_code == 404


async def test_11_draft_hidden_not_public(app_client, site):
    store, project = site
    w = store.add_website(project, "draft-site-bbb", published=False,
                          html="<p>Draft body</p>")
    # Owner preview works (manage), public does not.
    assert (await app_client.get("/api/website/preview/draft-site-bbb")).status_code == 200
    assert (await app_client.get("/api/website/public/draft-site-bbb")).status_code == 404


async def test_12_published_public_resolve(app_client, site):
    store, project = site
    store.add_website(project, "public-site-ccc", published=True,
                      html="<p>Public body</p>")
    meta = await app_client.get("/api/website/public/public-site-ccc")
    assert meta.status_code == 200, meta.text
    assert meta.json()["slug"] == "public-site-ccc"
    html = await app_client.get("/api/website/public/public-site-ccc/html")
    assert html.status_code == 200
    assert "Public body" in html.text


# --------------------------------------------------------------------------
# 13-15: sanitise XSS / URLs / citations
# --------------------------------------------------------------------------

async def test_13_sanitise_strips_xss():
    from services.website_gen.sanitise import sanitise_html
    dirty = ('<p>Hello</p><script>alert(1)</script>'
             '<a href="javascript:alert(2)">click</a>'
             '<div onclick="evil()">x</div>')
    clean = sanitise_html(dirty)
    assert "<script" not in clean
    assert "javascript:" not in clean
    assert "onclick" not in clean
    assert "Hello" in clean and "click" in clean


async def test_14_sanitise_ssrf_urls_dropped():
    from services.website_gen.sanitise import sanitise_html
    dirty = ('<a href="http://localhost:8000/admin">local</a>'
             '<a href="http://127.0.0.1/x">loop</a>'
             '<a href="data:text/html,<script>alert(1)</script>">data</a>'
             '<a href="https://example.com/ok">ok</a>')
    clean = sanitise_html(dirty)
    assert "localhost" not in clean and "127.0.0.1" not in clean
    assert "data:text/html" not in clean
    assert "https://example.com/ok" in clean


async def test_15_citations_preserved_through_sanitise():
    from services.website_gen.sanitise import sanitise_html
    dirty = ('<p>Chlorophyll absorbs light '
             '<span class="citation" data-chunk-id="abc123" '
             'data-page="4" data-method="ppocr">[chunk abc123, p.4, ppocr]</span></p>')
    clean = sanitise_html(dirty)
    assert "data-chunk-id" in clean and "abc123" in clean
    assert "data-page" in clean and "data-method" in clean


# --------------------------------------------------------------------------
# 16-17: foreign content blocked + update/delete ownership
# --------------------------------------------------------------------------

async def test_16_foreign_website_blocked(app_client, site, monkeypatch, user_b):
    from main import app
    from services.auth.deps import get_current_user

    store, project = site
    store.add_website(project, "owner-site-ddd", published=False)

    async def fake_b():
        return user_b

    app.dependency_overrides[get_current_user] = fake_b
    assert (await app_client.get("/api/website/owner-site-ddd")).status_code == 404
    assert (await app_client.get("/api/website/preview/owner-site-ddd")).status_code == 404
    assert (await app_client.delete("/api/website/manage/owner-site-ddd")).status_code == 404


async def test_17_update_delete_ownership(app_client, site):
    store, project = site
    store.add_website(project, "edit-site-eee", published=False)
    patched = await app_client.patch("/api/website/manage/edit-site-eee",
                                     json={"title": "New title"})
    assert patched.status_code == 200, patched.text
    assert patched.json()["title"] == "New title"
    assert (await app_client.delete("/api/website/manage/edit-site-eee")).status_code == 200
    assert (await app_client.get("/api/website/edit-site-eee")).status_code == 404


# --------------------------------------------------------------------------
# 18-20: preview parity, renders, missing content, puter OFF, unpublish
# --------------------------------------------------------------------------

async def test_18_preview_published_parity_and_unpublish(app_client, site):
    store, project = site
    created = await app_client.post(
        "/api/website/generate",
        json={"project_id": str(project.id), "content_type": "notes",
              "title": "Parity", "content": {"sections": [{"heading": "H"}]}})
    assert created.status_code == 200, created.text
    slug = created.json()["slug"]
    preview = await app_client.get(f"/api/website/preview/{slug}")
    assert preview.status_code == 200
    # Draft: public hidden.
    assert (await app_client.get(f"/api/website/public/{slug}")).status_code == 404
    # Publish: public bytes equal preview bytes.
    pub = await app_client.post("/api/website/publish", json={"slug": slug})
    assert pub.status_code == 200
    public = await app_client.get(f"/api/website/public/{slug}/html")
    assert public.status_code == 200
    assert public.text == preview.text
    # Visibility toggle off: unpublish hides it again.
    unpub = await app_client.post("/api/website/unpublish", json={"slug": slug})
    assert unpub.status_code == 200
    assert (await app_client.get(f"/api/website/public/{slug}")).status_code == 404
    vis = await app_client.post("/api/website/visibility",
                                json={"slug": slug, "visibility": "public"})
    assert vis.status_code == 200
    assert (await app_client.get(f"/api/website/public/{slug}")).status_code == 200


async def test_19_content_types_render_and_missing_safe(app_client, site):
    _store, project = site
    for ctype in ("notes", "flashcards", "quiz", "mixed"):
        resp = await app_client.post(
            "/api/website/generate",
            json={"project_id": str(project.id), "content_type": ctype,
                  "title": f"{ctype} guide",
                  "content": {"sections": [{"heading": "Intro", "content": "Body"}],
                              "flashcards": [{"front": "Q?", "back": "A"}],
                              "questions": [{"question": "Q?",
                                             "options": ["A", "B"]}]}})
        assert resp.status_code == 200, (ctype, resp.text)
        preview = await app_client.get(f"/api/website/preview/{resp.json()['slug']}")
        assert preview.status_code == 200
        assert f"{ctype} guide" in preview.text
    # Missing/empty content is safe (never 500).
    empty = await app_client.post("/api/website/generate",
                                  json={"project_id": str(project.id),
                                        "content_type": "mixed",
                                        "title": "Empty", "content": {}})
    assert empty.status_code == 200, empty.text


async def test_20_puter_route_default_off_explicit(app_client, site):
    resp = await app_client.post("/api/ocr/puter", content=b"fake-image-bytes",
                                 headers={"X-Puter-Page": "1",
                                          "X-Puter-Token": "user-token-xyz"})
    # Default OFF -> explicit 503 (never silent); documents the degraded path.
    assert resp.status_code == 503
    assert "disabled" in resp.text.lower()
    health = await app_client.get("/api/ocr/puter/health")
    assert health.status_code == 200
    assert health.json()["available"] is False
