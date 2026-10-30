"""Puter optional OCR path for heavy documents.

Contract:
- Default OFF (PUTER_ENABLED=false). Nothing routes here unless the user
  gives an explicit gesture (per-request opt-in flag).
- Explicit auth: the caller supplies a short-lived user token per call;
  no credentials are stored server-side (never persisted, never logged).
- Failure -> explicit degraded: any transport/auth/parse failure raises
  PuterDegradedError with a reason; callers mark ocr_unavailable/partial,
  never silent empty text.
- Provenance preserved: success returns method "puter-ocr" with
  provider/method/page/timing in metadata, same shape as PP-OCR pages.

Page-level routing: normal pages take PP-OCR (services/strategies/ocr);
only heavy documents (page_count >= heavy_page_threshold, default ~100,
configurable via PUTER_HEAVY_PAGE_THRESHOLD) may OFFER this path, and the
frontend surfaces a warning rather than a hard invariant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from services.strategies.base import OCRResult, ExtractionResult


class PuterDisabledError(RuntimeError):
    """Puter path is disabled or was not explicitly authorised."""


class PuterDegradedError(RuntimeError):
    """Puter call failed explicitly: per-page ocr_unavailable, never silent."""


@dataclass
class PuterConfig:
    enabled: bool = False
    heavy_page_threshold: int = 100
    endpoint: str = "https://api.puter.com/v1/ocr"
    timeout_seconds: float = 120.0


def _get_config() -> PuterConfig:
    try:
        from config.settings import get_puter_config

        cfg = get_puter_config()
        return PuterConfig(
            enabled=bool(getattr(cfg, "enabled", False)),
            heavy_page_threshold=int(getattr(cfg, "heavy_page_threshold", 100)),
        )
    except Exception:
        return PuterConfig()


def is_heavy_document(page_count: int, threshold: Optional[int] = None) -> bool:
    """Configurable heavy-flow gate (default ~100 pages, not a hard invariant)."""
    limit = int(threshold if threshold is not None else _get_config().heavy_page_threshold)
    try:
        return int(page_count or 0) >= limit
    except Exception:
        return False


class PuterOCRProvider:
    """User-authorised heavy-document OCR via Puter (explicit gesture only)."""

    METHOD = "puter-ocr"
    PROVIDER = "puter"

    def __init__(self, config: Optional[PuterConfig] = None):
        self._config = config or _get_config()

    async def health(self) -> dict:
        return {
            "available": bool(self._config.enabled),
            "provider": self.PROVIDER,
            "method": self.METHOD,
            "reason": None if self._config.enabled else "puter disabled (default OFF)",
        }

    async def process_page(
        self,
        image_path: str,
        user_token: Optional[str] = None,
        page_number: int = 1,
        timeout: Optional[float] = None,
    ) -> OCRResult:
        """OCR one page via Puter with an explicit per-call user token.

        Raises:
            PuterDisabledError: path disabled or no explicit authorisation.
            PuterDegradedError: transport/auth/parse failure or empty output.
        """
        if not self._config.enabled:
            raise PuterDisabledError(
                "puter disabled: set PUTER_ENABLED=true and pass an explicit "
                "user gesture + token per call (default OFF)"
            )
        if not user_token:
            raise PuterDisabledError(
                "puter requires an explicit user gesture: no token supplied, "
                "no credentials stored server-side"
            )
        started = time.time()
        try:
            text, confidence = await self._call_puter(
                image_path, user_token, timeout or self._config.timeout_seconds
            )
        except (PuterDisabledError, PuterDegradedError):
            raise
        except Exception as exc:
            raise PuterDegradedError(
                f"puter degraded on page {page_number}: {exc}"
            ) from exc
        if not (text or "").strip():
            raise PuterDegradedError(
                f"puter degraded on page {page_number}: empty output "
                "(explicit degraded, never silent)"
            )
        elapsed = time.time() - started
        return OCRResult(
            text=text.strip(),
            confidence=float(confidence),
            page_number=page_number,
            method=self.METHOD,
            metadata={
                "provider": self.PROVIDER,
                "method": self.METHOD,
                "page": page_number,
                "timing_seconds": round(elapsed, 3),
                "heavy_path": True,
            },
        )

    async def _call_puter(
        self, image_path: str, user_token: str, timeout: float
    ) -> tuple[str, float]:
        """Transport layer. No credentials stored; token lives in this call only."""
        import aiohttp

        # Token is header-scoped to this request; never persisted/logged.
        redacted_headers = {"Authorization": "Bearer <redacted>"}
        _ = redacted_headers  # only shape kept for diagnostics
        real_headers = {"Authorization": "Bearer " + str(user_token)}
        try:
            with open(image_path, "rb") as fh:
                payload = fh.read()
        except Exception as exc:
            raise PuterDegradedError(f"puter cannot read page image: {exc}") from exc
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._config.endpoint,
                    headers=real_headers,
                    data=payload,
                    timeout=aiohttp.ClientTimeout(total=float(timeout)),
                ) as resp:
                    if resp.status == 401:
                        raise PuterDegradedError("puter auth rejected the user token")
                    if resp.status != 200:
                        raise PuterDegradedError(
                            f"puter HTTP {resp.status} (explicit degraded)"
                        )
                    data = await resp.json()
        except (PuterDisabledError, PuterDegradedError):
            raise
        except Exception as exc:
            raise PuterDegradedError(f"puter transport failed: {exc}") from exc
        try:
            text = str(data.get("text", "") or "")
            conf = float(data.get("confidence", 0.0) or 0.0)
        except Exception as exc:
            raise PuterDegradedError(f"puter bad response shape: {exc}") from exc
        return text, max(0.0, min(1.0, conf))

    async def extract_text(
        self, file_path: str, pages: Optional[list[int]] = None,
        user_token: Optional[str] = None,
    ) -> ExtractionResult:
        """File-level entry (explicit token required); delegates per page."""
        start = time.time()
        page_no = (pages[0] + 1) if pages else 1
        page = await self.process_page(file_path, user_token=user_token,
                                       page_number=page_no)
        return ExtractionResult(
            pages=[page], total_pages=1, method=self.METHOD,
            processing_time=time.time() - start,
        )

    async def extract_page(
        self, file_path: str, page_number: int, user_token: Optional[str] = None
    ) -> OCRResult:
        return await self.process_page(file_path, user_token=user_token,
                                       page_number=page_number)


def create_puter_provider(config: Optional[PuterConfig] = None) -> PuterOCRProvider:
    """Create a Puter provider (still default OFF unless config enables it)."""
    return PuterOCRProvider(config=config)


__all__ = [
    "PuterConfig",
    "PuterDegradedError",
    "PuterDisabledError",
    "PuterOCRProvider",
    "is_heavy_document",
    "create_puter_provider",
]
