"""PP-OCRv6 CPU ONNX provider (tiny det/rec, onnxruntime).

Single local OCR path: paddleocr 3.7.x engine backed by onnxruntime,
CPU-first, lightweight. Inference is isolated from request handling via
asyncio.to_thread (never blocks the event loop).

Page-level routing contract:
- normal pages -> PP-OCR here
- heavy pages (e.g. ~100+ page documents) -> Puter user-authorised path
  (services/strategies/puter), never an implicit fallback.

process_page(image, timeout) -> OCRResult: never returns silent empty
text. Unavailable engine / empty output raises OCRAvailabilityError with
an explicit reason so callers mark the page ocr_unavailable / document
partial.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional, Union

from ..base import OCRStrategy, OCRResult, ExtractionResult
from config.settings import get_ocr_config

PROVIDER_NAME = "ppocr-onnx"
METHOD_NAME = "ppocr-tiny-ort"


class OCRAvailabilityError(RuntimeError):
    """Typed error: OCR engine unreachable (per-page failure, partial doc)."""


class OCRTimeoutError(RuntimeError):
    """Typed error: OCR exceeded the per-page timeout."""


class PPOCROnxProvider(OCRStrategy):
    """PaddleOCR PP-OCRv6 tiny ONNX provider on CPU via onnxruntime."""

    def __init__(self, timeout_seconds: float = 120.0):
        try:
            config = get_ocr_config()
        except Exception:
            config = None
        self._det_model = getattr(config, "det_model", "PP-OCRv6_mobile_det_tiny")
        self._rec_model = getattr(config, "rec_model", "PP-OCRv6_mobile_rec_tiny")
        self._engine = getattr(config, "engine", "onnxruntime")
        self._device = getattr(config, "device", "cpu")
        self._lang = getattr(config, "lang", "en")
        self._timeout = float(timeout_seconds or 120.0)
        self._ocr = None  # lazy PaddleOCR instance

    def _ensure_engine(self):
        """Lazily build the PaddleOCR engine (CPU, ONNX inference)."""
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR  # paddleocr>=3.7
        except ImportError as exc:
            raise OCRAvailabilityError(
                "ocr_unavailable: paddleocr>=3.7 not installed; "
                "pip install paddleocr>=3.7 onnxruntime paddlepaddle-cpu "
                "(CPU-only, no GPU wheel)."
            ) from exc
        try:
            import onnxruntime  # noqa: F401 - engine must be present
        except ImportError as exc:
            raise OCRAvailabilityError(
                "ocr_unavailable: onnxruntime not installed; "
                "pip install onnxruntime (CPU execution provider)."
            ) from exc
        try:
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang=self._lang,
                det_model_name=self._det_model,
                rec_model_name=self._rec_model,
                use_onnx=True,
                device=self._device,
                show_log=False,
            )
        except TypeError:
            # Older paddleocr kwarg surface: fall back to minimal kwargs.
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang=self._lang,
                use_onnx=True,
                show_log=False,
            )
        except Exception as exc:
            raise OCRAvailabilityError(
                f"ocr_unavailable: PP-OCR engine init failed: {exc}"
            ) from exc
        return self._ocr

    async def health(self) -> dict:
        """Health probe: engine importable (weights may download on first use)."""
        try:
            self._ensure_engine()
        except OCRAvailabilityError as exc:
            return {"available": False, "reason": str(exc), "provider": PROVIDER_NAME}
        except Exception as exc:
            return {"available": False, "reason": str(exc), "provider": PROVIDER_NAME}
        return {
            "available": True,
            "provider": PROVIDER_NAME,
            "method": METHOD_NAME,
            "engine": self._engine,
            "device": self._device,
            "det_model": self._det_model,
            "rec_model": self._rec_model,
        }

    async def process_page(
        self,
        image: Union[str, bytes, Path],
        timeout: Optional[float] = None,
        page_number: int = 1,
    ) -> OCRResult:
        """OCR one page image.

        Args:
            image: file path (str/Path) or raw image bytes (PNG/JPG).
            timeout: per-page seconds (defaults to constructor timeout).
            page_number: 1-indexed page number for provenance.

        Returns:
            OCRResult with text/confidence/provider/method/page/timing.
            Never silent empty: empty output raises OCRAvailabilityError.

        Raises:
            OCRTimeoutError: inference exceeded timeout.
            OCRAvailabilityError: engine missing/failed or empty output.
        """
        limit = float(timeout) if timeout else self._timeout
        started = time.time()
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_inference, image), timeout=limit
            )
        except asyncio.TimeoutError as exc:
            raise OCRTimeoutError(
                f"ocr timeout: PP-OCR exceeded {limit}s on page {page_number}"
            ) from exc
        elapsed = time.time() - started
        text = (result.get("text") or "").strip()
        if not text:
            raise OCRAvailabilityError(
                f"ocr_unavailable: PP-OCR empty output on page {page_number} "
                "(no silent empty; mark ocr_unavailable/partial)"
            )
        return OCRResult(
            text=text,
            confidence=float(result.get("confidence", 0.0)),
            page_number=page_number,
            method=METHOD_NAME,
            metadata={
                "provider": PROVIDER_NAME,
                "method": METHOD_NAME,
                "page": page_number,
                "timing_seconds": round(elapsed, 3),
                "engine": self._engine,
                "device": self._device,
                "det_model": self._det_model,
                "rec_model": self._rec_model,
            },
        )

    def _run_inference(self, image: Union[str, bytes, Path]) -> dict:
        ocr = self._ensure_engine()
        target = image
        tmp_path: Optional[str] = None
        if isinstance(image, (bytes, bytearray)):
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            with open(fd, "wb") as fh:
                fh.write(bytes(image))
            target = tmp_path
        else:
            target = str(image)
        try:
            raw = ocr.ocr(target)
        finally:
            if tmp_path:
                try:
                    import os

                    os.remove(tmp_path)
                except Exception:
                    pass
        # paddleocr 3.x returns list[page] of list[line] dicts/tuples.
        lines: list = []
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            lines = raw[0]
        elif isinstance(raw, list):
            lines = raw
        texts: list[str] = []
        scores: list[float] = []
        for line in lines:
            try:
                if isinstance(line, dict):
                    t = str(line.get("text", "") or "").strip()
                    s = float(line.get("score", 0.0) or 0.0)
                elif isinstance(line, (list, tuple)) and len(line) >= 2:
                    payload = line[1]
                    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
                        t, s = str(payload[0] or "").strip(), float(payload[1] or 0.0)
                    else:
                        t, s = str(payload or "").strip(), 0.0
                else:
                    t, s = str(line or "").strip(), 0.0
            except Exception:
                continue
            if t:
                texts.append(t)
                scores.append(max(0.0, min(1.0, s)))
        text = "\n".join(texts).strip()
        conf = (sum(scores) / len(scores)) if scores else 0.0
        return {"text": text, "confidence": conf}

    async def extract_text(
        self, file_path: str, pages: Optional[list[int]] = None
    ) -> ExtractionResult:
        """Extract text from a file (image direct; PDF first page selector)."""
        start = time.time()
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                raise OCRAvailabilityError(
                    "ocr_unavailable: empty text file (no silent empty)"
                )
            return ExtractionResult(
                pages=[OCRResult(text=text, confidence=1.0,
                                 page_number=1, method="plain-text")],
                total_pages=1, method="plain-text",
                processing_time=time.time() - start,
            )
        if suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".gif", ".webp"):
            page = await self.process_page(str(path), page_number=1)
            return ExtractionResult(
                pages=[page], total_pages=1, method=METHOD_NAME,
                processing_time=time.time() - start,
            )
        if suffix == ".pdf":
            # Single-file entry: OCR the requested selector pages by
            # rendering via the extractor path is preferred; here handle
            # the direct case (first page) explicitly, never silent.
            raise OCRAvailabilityError(
                "ocr_unavailable: PDF input needs per-page rendering via "
                "DocumentExtractor._vision_page (page-level routing); "
                "direct PDF bytes are not OCRed silently."
            )
        raise ValueError(f"Unsupported file type: {suffix}")

    async def extract_page(self, file_path: str, page_number: int) -> OCRResult:
        """Extract text from a single rendered page image file."""
        return await self.process_page(file_path, page_number=page_number)

    async def close(self) -> None:
        self._ocr = None


def create_ocr_provider(timeout_seconds: float = 120.0) -> PPOCROnxProvider:
    """Create and return a PP-OCRv6 ONNX provider instance."""
    return PPOCROnxProvider(timeout_seconds=timeout_seconds)


__all__ = [
    "PPOCROnxProvider",
    "OCRAvailabilityError",
    "OCRTimeoutError",
    "PROVIDER_NAME",
    "METHOD_NAME",
    "create_ocr_provider",
]
