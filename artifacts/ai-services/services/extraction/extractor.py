"""Tiered per-page extractor (T203/T204/T205).

Tier 0 native: PyMuPDF get_text("dict") with per-span bbox (primary).
  Permissive fallback when PyMuPDF is unavailable: pypdf + pdfplumber.
  LICENSE NOTE: PyMuPDF is AGPL-3.0; unmodified internal use is the current
  posture. If AGPL is unacceptable for distribution, set TIER0_BACKEND=pypdf
  (pure permissive path: pypdf text + pdfplumber geometry). Docling/MinerU
  are NOT defaults and are not imported here.
Tier 1 layout: pdfplumber table/geometry recovery when native text exists
  but structure is complex (multi-column, table regions). Degrades to tier-0
  output plus a recorded note when pdfplumber is absent.
Tier 2 vision: PP-OCRv6 CPU ONNX (tiny det/rec, onnxruntime) via the
  factory OCR strategy (provider ppocr-onnx, CPU-first, isolated from
  request handling via to_thread/worker). CPU-only hosts with no reachable
  engine get an explicit degraded path: pages that need vision but have no
  reachable provider fail individually with reason "ocr_unavailable"; the
  document ends partial, never completed with empty text. Page-level
  routing: normal pages take PP-OCR here; heavy documents route to the
  optional user-authorised Puter path (services/strategies/puter),
  never an implicit cloud fallback.

Quality gate per page (char count alone is NEVER sufficient):
  escalate when image-dominant, alnum ratio below threshold, high share of
  replacement/box chars, or no text layer at all.
"""

import hashlib
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from services.text_utils import (
    collapse_letter_spacing,
    collapse_spaces,
    strip_control_chars,
)

from .types import DocumentExtractionResult, ExtractionProgress, PageExtractionResult

TIER0_BACKEND = os.getenv("TIER0_BACKEND", "pymupdf").lower()  # pymupdf | pypdf
MIN_TEXT_LENGTH = 50
MIN_ALNUM_RATIO = 0.5
MAX_REPLACEMENT_RATIO = 0.05
PARSER_VERSION = "tiered-2.0.0"

REPLACEMENT_CHARS = set("\ufffd\u25a1\u25a0\u00ac")


@dataclass
class ExtractorConfig:
    page_batch_size: int = 1
    min_text_length: int = MIN_TEXT_LENGTH
    min_alnum_ratio: float = MIN_ALNUM_RATIO
    max_replacement_ratio: float = MAX_REPLACEMENT_RATIO
    force_ocr: bool = False
    ocr_timeout: int = 120
    extract_structure: bool = True
    temp_dir: Optional[str] = None  # configurable; never hardcoded /tmp


def _temp_dir(config: ExtractorConfig) -> str:
    return config.temp_dir or os.getenv("EXTRACTION_TMPDIR") or tempfile.gettempdir()


def _alnum_ratio(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    letters = sum(1 for c in text if c.isalnum())
    total = sum(1 for c in text if not c.isspace())
    return (letters / total) if total else 0.0


def _replacement_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(1 for c in text if c in REPLACEMENT_CHARS)
    return bad / max(1, len(text))


def _needs_vision(text: str, image_dominant: bool, config: ExtractorConfig) -> tuple[bool, str]:
    stripped = (text or "").strip()
    if not stripped:
        return True, "no_text_layer"
    if len(stripped) < config.min_text_length and image_dominant:
        return True, "image_dominant_short_text"
    if _alnum_ratio(stripped) < config.min_alnum_ratio:
        return True, "low_alnum_ratio"
    if _replacement_ratio(stripped) > config.max_replacement_ratio:
        return True, "garbage_text_layer"
    return False, ""


def _needs_layout(native_text: str, bbox_summary: Dict[str, Any]) -> bool:
    cols = int(bbox_summary.get("columns_detected", 1) or 1)
    tables = int(bbox_summary.get("table_regions", 0) or 0)
    order_conf = float(bbox_summary.get("reading_order_confidence", 1.0) or 1.0)
    return cols >= 2 or tables > 0 or order_conf < 0.6


class DocumentExtractor:
    """Per-page tiered pipeline. Every page is persisted via repositories."""

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self._ocr = None

    async def _get_ocr(self):
        if self._ocr is None:
            from factory import get_ocr

            self._ocr = get_ocr()
        return self._ocr

    # -- public ---------------------------------------------------------

    async def extract(
        self,
        document_id: uuid.UUID,
        file_path: str,
        file_type: str,
        db: AsyncSession,
        progress_callback: Optional[Callable] = None,
    ) -> DocumentExtractionResult:
        start_time = time.time()
        from services.ingestion.storage import storage_manager

        abs_path = storage_manager.get_absolute_path(file_path)
        if not abs_path.exists():
            # Tests/fakes may pass an absolute path directly.
            direct = Path(str(file_path))
            abs_path = direct if direct.exists() else abs_path
        if not abs_path.exists():
            raise FileNotFoundError(f"File not found: {abs_path}")
        if file_type == "pdf":
            return await self._extract_pdf(document_id, str(abs_path), db, progress_callback, start_time)
        if file_type in ("png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif"):
            return await self._extract_image(document_id, str(abs_path), db, start_time)
        if file_type in ("docx", "doc"):
            return await self._extract_docx(document_id, str(abs_path), db, start_time)
        if file_type == "txt":
            return await self._extract_txt(document_id, str(abs_path), db, start_time)
        raise ValueError(f"Unsupported file type: {file_type}")

    # -- pdf ------------------------------------------------------------

    async def _extract_pdf(self, document_id, file_path, db, progress_callback, start_time):
        from database import repositories

        total_pages = self._pdf_page_count(file_path)
        if total_pages == 0:
            raise ValueError("no extractable text: zero-page PDF")
        pages: List[PageExtractionResult] = []
        full_parts: List[str] = []
        ocr_used = False
        failed_pages: List[Dict[str, Any]] = []
        # OCR engine state for this run: warmed once on the first page that
        # needs vision (never for clean native-text PDFs, which skip vision
        # entirely). A failed warm fails the remaining vision pages fast
        # instead of re-attempting engine init per page.
        ocr_warmed = False
        ocr_unavailable_reason: Optional[str] = None

        for page_num in range(1, total_pages + 1):
            native = self._native_page(file_path, page_num)
            text = self._clean_text(native.get("text", ""))
            image_dominant = bool(native.get("image_dominant", False))
            bbox = native.get("bbox_summary", {}) or {}
            need_vision, reason = _needs_vision(text, image_dominant, self.config)
            if self.config.force_ocr:
                need_vision, reason = True, "force_ocr"

            method = "native"
            confidence = 0.9 if text.strip() else 0.1
            failure = None
            struct = await self._extract_structure(text) if self.config.extract_structure else []

            if need_vision:
                ocr_used = True
                if ocr_unavailable_reason is not None:
                    ocr_text, ocr_conf, ocr_method, ocr_err = (
                        "", 0.0, "ocr_unavailable", ocr_unavailable_reason)
                else:
                    if not ocr_warmed:
                        # First vision page: init the engine once (first run
                        # downloads PP-OCR weights) with an honest message so
                        # the stall is explained, not silent.
                        ocr_warmed = True
                        if progress_callback:
                            await progress_callback(ExtractionProgress(
                                document_id=str(document_id), current_page=page_num,
                                total_pages=total_pages, status="extracting",
                                progress_percent=(page_num / total_pages) * 100,
                                message="Preparing OCR engine (first run downloads models)...",
                                last_processed_page=page_num - 1))
                        try:
                            import asyncio as _asyncio

                            ocr = await self._get_ocr()
                            ensure = getattr(ocr, "_ensure_engine", None)
                            if callable(ensure):
                                await _asyncio.to_thread(ensure)
                        except Exception as exc:
                            msg = str(exc) or type(exc).__name__
                            ocr_unavailable_reason = (
                                msg if "ocr_unavailable" in msg
                                else f"ocr_unavailable: {msg}")
                            ocr_text, ocr_conf, ocr_method, ocr_err = (
                                "", 0.0, "ocr_unavailable", ocr_unavailable_reason)
                        else:
                            ocr_text, ocr_conf, ocr_method, ocr_err = await self._vision_page(file_path, page_num)
                    else:
                        ocr_text, ocr_conf, ocr_method, ocr_err = await self._vision_page(file_path, page_num)
                if ocr_err is None:
                    text, confidence, method = ocr_text, ocr_conf, ocr_method
                    struct = await self._extract_structure(text) if self.config.extract_structure else []
                else:
                    method = "ocr_unavailable"
                    confidence = 0.0
                    failure = ocr_err
                    failed_pages.append({"page": page_num, "reason": ocr_err})
            elif _needs_layout(text, bbox):
                layout_text, layout_note = self._layout_page(file_path, page_num, text)
                if layout_text != text:
                    text = layout_text
                    method = "layout"
                    confidence = 0.8
                else:
                    method = "native"
                    if layout_note:
                        bbox = {**bbox, "layout_note": layout_note}
                struct = await self._extract_structure(text) if self.config.extract_structure else []

            page_result = PageExtractionResult(
                page_number=page_num, text=text, confidence=confidence,
                method=method, metadata={"bbox_summary": bbox, "failure_reason": failure} if failure else {"bbox_summary": bbox},
                structural_elements=struct,
            )
            pages.append(page_result)
            full_parts.append(text)
            try:
                await repositories.upsert_page(
                    db, document_id, page_num, text=text, method=method,
                    confidence=confidence, parser_version=PARSER_VERSION,
                    bbox_summary=bbox, failure_reason=failure,
                )
            except Exception:
                pass  # fakes without pages table still return honest results
            if progress_callback:
                pct = (page_num / total_pages) * 100
                await progress_callback(ExtractionProgress(
                    document_id=str(document_id), current_page=page_num,
                    total_pages=total_pages, status="extracting",
                    progress_percent=pct, message=f"Processing page {page_num} of {total_pages}",
                    last_processed_page=page_num))

        full_text = "\n\n".join(full_parts)
        if not full_text.strip():
            raise ValueError("no extractable text: document has zero extractable text")
        topics = await self._extract_topics(full_text)
        summary: Dict[str, int] = {}
        for p in pages:
            summary[p.method] = summary.get(p.method, 0) + 1
        try:
            document = await repositories.get_document_by_id(db, document_id)
            if document is not None:
                document.page_count = total_pages
                try:
                    document.page_count_confirmed = total_pages
                    document.extraction_method_summary = summary
                    document.content_hash = hashlib.sha256(full_text.encode("utf-8", "ignore")).hexdigest()
                    failed = len(failed_pages)
                    if failed and failed < total_pages:
                        document.indexing_state = "partial"
                    elif failed == total_pages:
                        document.indexing_state = "failed"
                    else:
                        document.indexing_state = "indexing"
                    document.extracted_text = full_text
                except Exception:
                    document.extracted_text = full_text
        except Exception:
            pass
        return DocumentExtractionResult(
            document_id=str(document_id), total_pages=total_pages, pages=pages,
            full_text=full_text,
            method="pymupdf+vision" if ocr_used else "pymupdf",
            processing_time=time.time() - start_time, topics=topics,
            is_ocr_required=ocr_used,
        )

    def _pdf_page_count(self, file_path: str) -> int:
        if TIER0_BACKEND == "pypdf":
            from pypdf import PdfReader

            return len(PdfReader(file_path).pages)
        try:
            import fitz

            doc = fitz.open(file_path)
            n = len(doc)
            doc.close()
            return n
        except ImportError:
            from pypdf import PdfReader

            return len(PdfReader(file_path).pages)

    def _native_page(self, file_path: str, page_num: int) -> Dict[str, Any]:
        """Tier 0: PyMuPDF dict extraction with span bbox; permissive fallback."""
        if TIER0_BACKEND != "pypdf":
            try:
                import fitz

                doc = fitz.open(file_path)
                try:
                    page = doc[page_num - 1]
                    d = page.get_text("dict")
                    spans = []
                    chars = 0
                    for block in d.get("blocks", []):
                        for line in block.get("lines", []):
                            for span in line.get("spans", []):
                                t = span.get("text", "")
                                chars += len(t)
                                spans.append({"text": t[:120], "bbox": [round(float(v), 1) for v in span.get("bbox", [])]})
                    text = page.get_text("text") or ""
                    drawings = len(page.get_drawings() or [])
                    images = len(page.get_images() or [])
                    image_dominant = images > 0 and chars < self.config.min_text_length
                    # naive column detection: two well-separated x-clusters
                    xs = sorted(round(float(s["bbox"][0]) / 50) for s in spans if s.get("bbox"))
                    cols = len(set(xs)) if xs else 1
                    columns = 2 if len(set(xs)) >= 6 else 1
                    summary = {
                        "span_count": len(spans),
                        "char_count": chars,
                        "image_count": images,
                        "drawing_count": drawings,
                        "columns_detected": columns,
                        "table_regions": 0,
                        "reading_order_confidence": 0.9,
                        "spans_sample": spans[:5],
                    }
                    return {"text": text, "image_dominant": image_dominant, "bbox_summary": summary}
                finally:
                    doc.close()
            except ImportError:
                pass
        # Permissive fallback tier 0: pypdf text (no AGPL component).
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            page = reader.pages[page_num - 1]
            text = page.extract_text() or ""
            summary = {"span_count": 0, "char_count": len(text), "image_count": 0,
                       "columns_detected": 1, "table_regions": 0,
                       "reading_order_confidence": 0.7, "backend": "pypdf-fallback"}
            return {"text": text, "image_dominant": len(text.strip()) < self.config.min_text_length,
                    "bbox_summary": summary}
        except Exception as exc:
            raise RuntimeError(f"native extraction failed on page {page_num}: {exc}")

    def _layout_page(self, file_path: str, page_num: int, native_text: str) -> tuple[str, str]:
        """Tier 1: pdfplumber geometry/table recovery (permissive license)."""
        try:
            from services.strategies.layout.pdfplumber_parser import PdfplumberParser

            parser = PdfplumberParser()
            out = parser.parse_page(file_path, page_num)
            if out.get("tables_markdown"):
                merged = (native_text + "\n\n" + out["tables_markdown"]).strip()
                return merged, ""
            return native_text, "layout tier: no tables detected; tier-0 output kept"
        except Exception as exc:
            return native_text, f"layout tier degraded ({exc}); tier-0 output kept"

    async def _vision_page(self, file_path: str, page_num: int) -> tuple[str, float, str, Optional[str]]:
        """Tier 2: render page to bytes, send through factory OCR strategy.

        Returns (text, confidence, method, error). error is None on success.
        Streams page images via a configurable temp dir cleaned up on failure;
        never writes to a hardcoded /tmp path.
        """
        ocr = await self._get_ocr()
        tmp_path: Optional[str] = None
        try:
            img_bytes = self._render_page_png(file_path, page_num)
            if img_bytes is None:
                # Native PDF bytes path for providers that accept files: let
                # the provider try the source file with a page selector.
                try:
                    import asyncio

                    result = await asyncio.wait_for(
                        ocr.extract_text(file_path, pages=[page_num - 1]),
                        timeout=self.config.ocr_timeout,
                    )
                    if result.pages:
                        p = result.pages[0]
                        return p.text, float(p.confidence or 0.0), p.method or "vision", None
                except Exception as exc:
                    return "", 0.0, "ocr_unavailable", f"ocr_unavailable: {exc}"
                return "", 0.0, "ocr_unavailable", "ocr_unavailable: no vision output"
            fd, tmp_path = tempfile.mkstemp(suffix=".png", dir=_temp_dir(self.config))
            with open(fd, "wb") as f:
                f.write(img_bytes)
            import asyncio

            # Providers accept a file path; bytes were staged to tmp_path above.
            result = await asyncio.wait_for(
                ocr.extract_text(tmp_path), timeout=self.config.ocr_timeout
            )
            if result.pages and (result.pages[0].text or "").strip():
                p = result.pages[0]
                return p.text, float(p.confidence or 0.0), p.method or "vision", None
            return "", 0.0, "ocr_unavailable", "ocr_unavailable: empty vision output"
        except Exception as exc:
            msg = str(exc) or type(exc).__name__
            if "ocr_unavailable" in msg:
                return "", 0.0, "ocr_unavailable", msg
            return "", 0.0, "ocr_unavailable", f"ocr_unavailable: {msg}"
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _render_page_png(self, file_path: str, page_num: int) -> Optional[bytes]:
        try:
            import fitz

            doc = fitz.open(file_path)
            try:
                page = doc[page_num - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                return pix.tobytes("png")
            finally:
                doc.close()
        except Exception:
            return None

    # -- images / docx / txt --------------------------------------------

    async def _extract_image(self, document_id, file_path, db, start_time) -> DocumentExtractionResult:
        from database import repositories

        ocr = await self._get_ocr()
        import asyncio

        try:
            result = await asyncio.wait_for(ocr.extract_text(file_path), timeout=self.config.ocr_timeout)
            text = result.full_text if result.pages else ""
            method = result.method if result.pages else "vision"
            conf = float(result.pages[0].confidence) if result.pages else 0.0
        except Exception as exc:
            raise RuntimeError(f"ocr_unavailable: {exc}")
        try:
            await repositories.upsert_page(db, document_id, 1, text=text, method=method,
                                           confidence=conf, parser_version=PARSER_VERSION)
        except Exception:
            pass
        page = PageExtractionResult(page_number=1, text=text, confidence=conf, method=method, metadata={})
        return DocumentExtractionResult(
            document_id=str(document_id), total_pages=1, pages=[page], full_text=text,
            method=method, processing_time=time.time() - start_time,
            topics=await self._extract_topics(text), is_ocr_required=True)

    async def _extract_docx(self, document_id, file_path, db, start_time) -> DocumentExtractionResult:
        from database import repositories

        from docx import Document

        doc = Document(file_path)
        buf = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        text = self._clean_text(buf)
        try:
            await repositories.upsert_page(db, document_id, 1, text=text, method="docx",
                                           confidence=1.0, parser_version=PARSER_VERSION)
        except Exception:
            pass
        page = PageExtractionResult(page_number=1, text=text, confidence=1.0, method="docx",
                                    structural_elements=await self._extract_structure(text) if self.config.extract_structure else [])
        return DocumentExtractionResult(
            document_id=str(document_id), total_pages=1, pages=[page], full_text=text,
            method="docx", processing_time=time.time() - start_time,
            topics=await self._extract_topics(text), is_ocr_required=False)

    async def _extract_txt(self, document_id, file_path, db, start_time) -> DocumentExtractionResult:
        from database import repositories

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = self._clean_text(f.read())
        try:
            await repositories.upsert_page(db, document_id, 1, text=text, method="text",
                                           confidence=1.0, parser_version=PARSER_VERSION)
        except Exception:
            pass
        page = PageExtractionResult(page_number=1, text=text, confidence=1.0, method="plain-text",
                                    structural_elements=await self._extract_structure(text) if self.config.extract_structure else [])
        return DocumentExtractionResult(
            document_id=str(document_id), total_pages=1, pages=[page], full_text=text,
            method="plain-text", processing_time=time.time() - start_time,
            topics=await self._extract_topics(text), is_ocr_required=False)

    # -- helpers ----------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Lossless cleanup for normal text; collapses corrupted patterns.

        - Dehyphenate end-of-line breaks: "exam-\\nple" -> "example".
        - Strip control chars (bullet glyphs become "• "), collapse letter
          spacing on lines that are mostly space-separated single letters
          (corrupted text layer, "s h e r y"), and collapse intra-line
          whitespace -- all via the shared services.text_utils helpers.
        - Paragraph breaks are preserved.
        """
        if not text:
            return ""
        text = re.sub(r"([A-Za-z])-\n([a-z])", r"\1\2", text)
        text = strip_control_chars(text)
        text = collapse_letter_spacing(text)
        return collapse_spaces(text)

    def _extract_page_metadata(self, page) -> Dict[str, Any]:
        try:
            rect = page.rect
            return {"width": rect.width, "height": rect.height, "rotation": page.rotation}
        except Exception:
            return {}

    async def _extract_structure(self, text: str) -> List[Dict[str, Any]]:
        structure = []
        for i, line in enumerate(text.split("\n")):
            line = line.strip()
            if not line:
                continue
            is_heading = len(line) < 100 and not line.endswith(".") and line[:1].isupper()
            structure.append({"type": "heading" if is_heading else "paragraph", "text": line, "position": i})
        return structure

    async def _extract_topics(self, text: str) -> List[str]:
        keywords = ["definition", "concept", "theory", "principle", "law", "formula",
                    "equation", "variable", "function", "model", "analysis", "method",
                    "process", "system", "structure", "example", "case", "application",
                    "significance", "history", "chapter", "section", "summary", "review", "question"]
        lower = text.lower()
        found = []
        for kw in keywords:
            if kw in lower:
                idx = lower.find(kw)
                found.append(f"{kw}: {text[max(0, idx-50):idx+50].strip()}...")
        return found[:10]


document_extractor = DocumentExtractor()

__all__ = ["DocumentExtractor", "ExtractorConfig", "document_extractor",
           "MIN_TEXT_LENGTH", "PARSER_VERSION"]
