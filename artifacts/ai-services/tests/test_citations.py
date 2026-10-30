"""Citation tests (T303). Offline, deterministic.

Covers: every returned citation carries provenance and resolves to real
stored text, page_number correctness on a multi-page fixture, stale /
foreign references rejected (never rendered), method + confidence
present for low-confidence (OCR-derived) passages.
"""

import uuid

import pytest

from services.retrieval.citations import (
    build_citation_from_chunk,
    resolve_citation,
    validate_citation_ids,
)


class FakeDoc:
    def __init__(self, file_name="paper.pdf"):
        self.file_name = file_name


class FakeChunk:
    def __init__(self, **kw):
        self.id = kw.get("id", uuid.uuid4())
        self.document_id = kw.get("document_id", uuid.uuid4())
        self.content = kw.get("content", "real stored text about photosynthesis")
        self.page_start = kw.get("page_start", 4)
        self.page_end = kw.get("page_end", 5)
        self.section_path = kw.get("section_path", ["Ch 2", "Calvin cycle"])
        self.heading = kw.get("heading", "Calvin cycle")
        self.chunk_metadata = kw.get("chunk_metadata", {"method": "vision:ppocr-tiny-ort",
                                                        "confidence": 0.42})
        self.document = None


def test_citation_carries_full_provenance():
    chunk = FakeChunk()
    citation = build_citation_from_chunk(chunk, FakeDoc("paper.pdf"))
    d = citation.to_dict()
    assert d["chunk_id"] == str(chunk.id)
    assert d["document_id"] == str(chunk.document_id)
    assert d["file_name"] == "paper.pdf"
    assert d["page_number"] == 4
    assert d["page"] == 4
    assert d["section_path"] == ["Ch 2", "Calvin cycle"]
    assert d["heading"] == "Calvin cycle"
    assert d["method"] == "vision:ppocr-tiny-ort"
    assert d["confidence"] == 0.42
    assert d["span"] in chunk.content
    assert d["valid"] is True


def test_page_number_correct_for_multipage_passage():
    chunk = FakeChunk(page_start=5, page_end=7)
    citation = build_citation_from_chunk(chunk)
    assert citation.page_number == 5
    assert citation.page_end == 7


class FakeDB:
    def __init__(self, chunks):
        self._by_id = {str(c.id): c for c in chunks}

    async def get(self, model, pk):
        name = getattr(model, "__name__", "")
        if name == "Chunk":
            return self._by_id.get(str(pk))
        if name == "Document":
            return FakeDoc()
        return None


@pytest.mark.asyncio
async def test_resolve_rejects_stale_reference():
    db = FakeDB([])
    citation = await resolve_citation(db, str(uuid.uuid4()))
    assert citation.valid is False
    assert "not found" in (citation.reason or "")


@pytest.mark.asyncio
async def test_resolve_rejects_foreign_chunk():
    chunk = FakeChunk()
    db = FakeDB([chunk])
    citation = await resolve_citation(db, str(chunk.id),
                                      allowed_chunk_ids={"something-else"})
    assert citation.valid is False
    assert "not in the retrieval set" in (citation.reason or "")


@pytest.mark.asyncio
async def test_resolve_accepts_real_stored_text():
    chunk = FakeChunk()
    db = FakeDB([chunk])
    citation = await resolve_citation(db, str(chunk.id),
                                      allowed_chunk_ids={str(chunk.id)})
    assert citation.valid is True
    assert citation.span


def test_validate_splits_valid_and_rejected():
    valid, rejected = validate_citation_ids(["a", "b", "c"], {"a", "c"})
    assert valid == ["a", "c"]
    assert rejected == ["b"]
