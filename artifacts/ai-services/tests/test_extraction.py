"""T203/T204/T205: tiered extraction, provenance, vision degradation (offline).

Real PDFs are built with PyMuPDF; the vision tier is a counting fake OCR
provider injected into the extractor (no network, no VLM). Asserts:
- module imports (D1 SyntaxError regression)
- born-digital pages stay native with zero OCR calls
- garbage-text-layer escalates on quality (not char count)
- image-only pages take the OCR path and record confidence
- unreachable vision -> per-page ocr_unavailable, never silent empty text
- no hardcoded /tmp page files
"""

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.asyncio


def _make_pdf(path: Path, page_texts: list[str], garbage_page: int = -1) -> Path:
    import fitz

    doc = fitz.open()
    for i, text in enumerate(page_texts):
        page = doc.new_page()
        body = text
        if i == garbage_page:
            # Long but garbage: passes char-count, fails alnum-ratio gate.
            body = ("\u00ac \u25a1 \ufffd " * 60) + " " + text
        page.insert_text((72, 72), body)
    doc.save(str(path))
    doc.close()
    return path


def _make_blank_pdf(path: Path, pages: int = 2) -> Path:
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()  # no text layer: image-dominant/empty
    doc.save(str(path))
    doc.close()
    return path


class FakeOCR:
    """Counting vision provider double."""

    def __init__(self, text="scanned transcription", confidence=0.82):
        self.calls = 0
        self.text = text
        self.confidence = confidence

    async def extract_text(self, file_path, pages=None):
        from services.strategies.base import ExtractionResult, OCRResult

        self.calls += 1
        return ExtractionResult(
            pages=[OCRResult(text=self.text, confidence=self.confidence,
                             page_number=1, method="ppocr-tiny-ort")],
            total_pages=1, method="ppocr-tiny-ort", processing_time=0.01)


class UnreachableOCR:
    async def extract_text(self, file_path, pages=None):
        from services.strategies.ocr.ppocr_onnx import OCRAvailabilityError

        raise OCRAvailabilityError("ocr_unavailable: no vision provider reachable")


@pytest.fixture
def fake_db_pages():
    store = {"pages": [], "doc": None}

    async def upsert_page(db, document_id, page_number, text="", method="native",
                          confidence=0.0, parser_version=None, image_stored=False,
                          bbox_summary=None, failure_reason=None):
        store["pages"].append({"document_id": document_id, "page_number": page_number,
                               "text": text, "method": method, "confidence": confidence,
                               "failure_reason": failure_reason})
        return SimpleNamespace(**store["pages"][-1])

    async def get_document_by_id(db, document_id):
        return store["doc"]

    async def get_pages(db, document_id):
        return [SimpleNamespace(**p) for p in store["pages"]
                if p["document_id"] == document_id]

    async def breakdown(db, document_id):
        out = {}
        for p in store["pages"]:
            if p["document_id"] == document_id:
                out[p["method"]] = out.get(p["method"], 0) + 1
        return out

    return SimpleNamespace(store=store, upsert_page=upsert_page,
                           get_document_by_id=get_document_by_id,
                           get_pages=get_pages, breakdown=breakdown)


def _install(monkeypatch, fake, doc_id, tmp_path, pdf_path):
    import services.extraction.extractor as ext_mod
    from database import repositories
    from services.ingestion import storage as storage_mod

    doc = SimpleNamespace(id=doc_id, project_id=uuid.uuid4(), file_name="f.pdf",
                          file_path=str(pdf_path), page_count=None,
                          extracted_text=None, indexing_state=None,
                          extraction_method_summary=None, content_hash=None,
                          page_count_confirmed=None)
    fake.store["doc"] = doc
    monkeypatch.setattr(repositories, "upsert_page", fake.upsert_page)
    monkeypatch.setattr(repositories, "get_document_by_id", fake.get_document_by_id)
    monkeypatch.setattr(repositories, "get_pages", fake.get_pages)
    monkeypatch.setattr(repositories, "page_method_breakdown", fake.breakdown)
    monkeypatch.setattr(storage_mod.storage_manager, "get_absolute_path",
                        lambda p: Path(p), raising=False)
    return doc


async def test_import_succeeds_syntax_regression():
    import services.extraction.extractor  # noqa: F401 (D1 gate)


async def test_native_pages_use_tier0_no_ocr(monkeypatch, tmp_path, fake_db_pages):
    from services.extraction.extractor import DocumentExtractor

    pdf = _make_pdf(tmp_path / "native.pdf",
                    ["Introduction to physics. Definitions and concepts " * 4,
                     "Chapter two covers theory and worked examples " * 4])
    doc_id = uuid.uuid4()
    _install(monkeypatch, fake_db_pages, doc_id, tmp_path, pdf)
    ext = DocumentExtractor()
    fake_ocr = FakeOCR()
    ext._ocr = fake_ocr
    result = await ext.extract(doc_id, str(pdf), "pdf", object())
    assert fake_ocr.calls == 0
    assert all(p.method == "native" for p in result.pages)
    assert all(isinstance(p.confidence, float) for p in result.pages)
    persisted = [p for p in fake_db_pages.store["pages"]]
    assert len(persisted) == 2 and all(p["method"] == "native" for p in persisted)


async def test_garbage_text_layer_escalates_on_quality(monkeypatch, tmp_path, fake_db_pages):
    from services.extraction.extractor import DocumentExtractor

    pdf = _make_pdf(tmp_path / "garbage.pdf",
                    ["Real academic content about biology " * 8], garbage_page=0)
    doc_id = uuid.uuid4()
    _install(monkeypatch, fake_db_pages, doc_id, tmp_path, pdf)
    ext = DocumentExtractor()
    fake_ocr = FakeOCR()
    ext._ocr = fake_ocr
    result = await ext.extract(doc_id, str(pdf), "pdf", object())
    assert fake_ocr.calls == 1  # escalated despite long char count
    assert result.pages[0].method == "ppocr-tiny-ort"


async def test_scanned_pages_take_ocr_path(monkeypatch, tmp_path, fake_db_pages):
    from services.extraction.extractor import DocumentExtractor

    pdf = _make_blank_pdf(tmp_path / "scanned.pdf", pages=2)
    doc_id = uuid.uuid4()
    _install(monkeypatch, fake_db_pages, doc_id, tmp_path, pdf)
    ext = DocumentExtractor()
    fake_ocr = FakeOCR(confidence=0.77)
    ext._ocr = fake_ocr
    result = await ext.extract(doc_id, str(pdf), "pdf", object())
    assert fake_ocr.calls == 2
    assert all("ocr" in p.method or "vision" in p.method for p in result.pages)
    assert all(p.confidence == 0.77 for p in result.pages)
    assert all(p.text.strip() for p in result.pages)


async def test_mixed_pdf_splits_native_and_vision(monkeypatch, tmp_path, fake_db_pages):
    from services.extraction.extractor import DocumentExtractor

    import fitz

    pdf = tmp_path / "mixed.pdf"
    doc = fitz.open()
    p0 = doc.new_page()
    p0.insert_text((72, 72), "Digital page with real academic text " * 6)
    doc.new_page()  # image-only
    doc.save(str(pdf))
    doc.close()
    doc_id = uuid.uuid4()
    _install(monkeypatch, fake_db_pages, doc_id, tmp_path, pdf)
    ext = DocumentExtractor()
    fake_ocr = FakeOCR()
    ext._ocr = fake_ocr
    result = await ext.extract(doc_id, str(pdf), "pdf", object())
    assert fake_ocr.calls == 1
    assert result.pages[0].method == "native"
    assert result.pages[1].method == "ppocr-tiny-ort"


async def test_unreachable_vision_marks_pages_ocr_unavailable(monkeypatch, tmp_path, fake_db_pages):
    from services.extraction.extractor import DocumentExtractor

    import fitz

    # Mixed doc: page 1 native (completes), page 2 blank (fails per-page).
    pdf = tmp_path / "novlm.pdf"
    doc = fitz.open()
    p0 = doc.new_page()
    p0.insert_text((72, 72), "Digital page with real academic text " * 6)
    doc.new_page()  # image-only, VLM unreachable
    doc.save(str(pdf))
    doc.close()
    doc_id = uuid.uuid4()
    _install(monkeypatch, fake_db_pages, doc_id, tmp_path, pdf)
    ext = DocumentExtractor()
    ext._ocr = UnreachableOCR()
    result = await ext.extract(doc_id, str(pdf), "pdf", object())
    assert result.pages[0].method == "native"
    assert result.pages[1].method == "ocr_unavailable"
    persisted = [p for p in fake_db_pages.store["pages"] if p["page_number"] == 2][0]
    assert persisted["failure_reason"] and "ocr_unavailable" in persisted["failure_reason"]
    # Partial, never completed-with-empty-text: page 1 text survives.
    assert result.pages[0].text.strip()


async def test_no_hardcoded_tmp_page_files():
    src = Path("services/extraction/extractor.py").read_text()
    assert "/tmp/page_" not in src


async def test_quality_gate_unit():
    from services.extraction.extractor import _needs_vision
    from services.extraction.extractor import ExtractorConfig

    cfg = ExtractorConfig()
    need, reason = _needs_vision("Introduction to physics definitions " * 4, False, cfg)
    assert need is False
    need, reason = _needs_vision("\u00ac \u25a1 \ufffd " * 60, False, cfg)
    assert need is True and reason == "low_alnum_ratio"
    need, reason = _needs_vision("   ", True, cfg)
    assert need is True and reason == "no_text_layer"
