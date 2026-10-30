"""Phase B: domain persist (reads serve domain rows, not just cache) +
007 enum migration. Offline fakes; no network, no Postgres."""

import uuid

import pytest

pytestmark = pytest.mark.asyncio


class _FakeDB:
    def __init__(self):
        self.adds = []
        self.commits = 0

    def add(self, obj):
        self.adds.append(obj)

    async def flush(self):
        return None

    async def refresh(self, obj):
        return obj

    async def commit(self):
        self.commits += 1


def _deck_id():
    return uuid.uuid4()


async def test_persist_notes_writes_note_row(monkeypatch):
    from types import SimpleNamespace as _NS

    from database import repositories
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    seen = {}

    async def fake_create_note(db, project_id, title, content=None):
        seen.update(project_id=project_id, title=title, content=content)
        return _NS(id=uuid.uuid4())

    monkeypatch.setattr(repositories, "create_note", fake_create_note)
    engine = TransformationEngine.__new__(TransformationEngine)
    pid = uuid.uuid4()
    content = {"title": "Bio Notes",
               "sections": [{"heading": "H", "content": "C"}]}
    await engine._persist_domain_rows(None, pid, GenerationMode.NOTES, content)
    assert seen["project_id"] == pid
    assert seen["title"] == "Bio Notes"
    assert seen["content"]["sections"][0]["heading"] == "H"


async def test_persist_flashcards_writes_deck_and_cards(monkeypatch):
    from database import repositories
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    calls = {"decks": [], "cards": []}

    async def fake_deck(db, project_id, name):
        calls["decks"].append((project_id, name))
        from types import SimpleNamespace as _NS
        return _NS(id=_deck_id(), name=name)

    async def fake_card(db, deck_id, front, back, difficulty="medium"):
        calls["cards"].append((deck_id, front, back, difficulty))

    monkeypatch.setattr(repositories, "create_flashcard_deck", fake_deck)
    monkeypatch.setattr(repositories, "create_flashcard", fake_card)
    engine = TransformationEngine.__new__(TransformationEngine)
    pid = uuid.uuid4()
    content = {"title": "Bio Deck",
               "flashcards": [{"front": "Q?", "back": "A.",
                               "difficulty": "easy"}]}
    await engine._persist_domain_rows(None, pid, GenerationMode.FLASHCARDS, content)
    assert calls["decks"] == [(pid, "Bio Deck")]
    assert len(calls["cards"]) == 1
    assert calls["cards"][0][1:] == ("Q?", "A.", "easy")


async def test_persist_quiz_and_mcq_write_quiz_rows(monkeypatch):
    from database import repositories
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    calls = []

    async def fake_quiz(db, project_id, title, questions=None):
        calls.append((project_id, title, questions))

    monkeypatch.setattr(repositories, "create_quiz", fake_quiz)
    engine = TransformationEngine.__new__(TransformationEngine)
    pid = uuid.uuid4()
    questions = [{"question": "Q?", "options": ["a", "b", "c", "d"],
                  "correct_answer": 0, "explanation": "E"}]
    await engine._persist_domain_rows(None, pid, GenerationMode.MCQ,
                                      {"title": "T", "questions": questions})
    await engine._persist_domain_rows(None, pid, GenerationMode.QUIZ,
                                      {"title": "T2", "questions": questions})
    assert len(calls) == 2
    assert calls[0][1] == "T" and len(calls[0][2]) == 1


async def test_persist_other_modes_write_nothing(monkeypatch):
    from services.transformation.engine import TransformationEngine
    from services.transformation.types import GenerationMode

    engine = TransformationEngine.__new__(TransformationEngine)
    # Must not raise and must not touch the DB session at all.
    await engine._persist_domain_rows(None, uuid.uuid4(),
                                      GenerationMode.SUMMARY, {"title": "S"})
    await engine._persist_domain_rows(None, uuid.uuid4(),
                                      GenerationMode.ANSWER, {"answer_text": "x"})


async def test_migration_007_adds_six_labels():
    from pathlib import Path

    src = Path("migrations/versions/007_outputtype_phase4_values.py").read_text()
    for label in ("mcq", "answer", "syllabus", "pyq", "mapping", "analysis"):
        assert label in src
    assert "def upgrade()" in src and "def downgrade()" in src
    assert 'down_revision = "006_user_provider_credentials"' in src
