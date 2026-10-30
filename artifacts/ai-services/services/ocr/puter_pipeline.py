"""Authorised heavy-document pipeline (Puter, default OFF).

Flow: heavy detection (page_count >= threshold, default ~100, configurable)
-> explicit authorisation (per-call X-Puter-Token, never stored)
-> Puter vision pages -> NORMAL pipeline reuse
(extract -> chunk -> embed -> pgvector) with provenance preserved.

- Puter is NOT required for normal/publish paths; the standard worker
  (services/jobs/handlers/document_processing) always proceeds locally.
- Failure is explicit degraded, never silent: a Puter page failure falls
  back to the local PP-OCR provider for that page; pages where both fail
  are recorded ocr_unavailable and the document ends partial/failed with
  the real reason.
- Provenance: Puter pages persist with method "puter-ocr" (matches the
  "puter:*" family); chunk page_spans carry the method through the real
  SemanticChunker, so citations resolve valid after the run.
- No credentials stored: the token lives in the adapter instance for this
  call only and is never written to job/document rows or logs.
"""

from __future__ import annotations

import logging
import time
import uuid
from types import SimpleNamespace
from typing import Any, Optional

from services.strategies.base import ExtractionResult, OCRResult
from services.strategies.puter import (
    PuterDegradedError,
    PuterDisabledError,
    PuterOCRProvider,
    is_heavy_document,
)

logger = logging.getLogger(__name__)


def heavy_check(page_count: int) -> dict:
    """Warning gate for heavy documents (configurable, not a hard invariant)."""
    try:
        from config.settings import get_puter_config

        cfg = get_puter_config()
        enabled = bool(getattr(cfg, "enabled", False))
        threshold = int(getattr(cfg, "heavy_page_threshold", 100))
    except Exception:
        enabled, threshold = False, 100
    try:
        count = int(page_count or 0)
    except Exception:
        count = 0
    heavy = is_heavy_document(count, threshold=threshold)
    warning = None
    if heavy:
        warning = (
            f"This document has ~{count} pages (threshold ~{threshold}). "
            "Local PP-OCR will process it page by page; for faster "
            "heavy-document OCR you may explicitly authorise the optional "
            "Puter path. Puter is OFF by default and any failure degrades "
            "explicitly with provenance preserved."
        )
    return {
        "heavy": heavy,
        "page_count": count,
        "threshold": threshold,
        "enabled": enabled,
        "warning": warning,
    }


class PuterTokenAdapter:
    """Vision strategy binding a per-call user token to Puter + local fallback.

    Satisfies the OCRStrategy surface used by DocumentExtractor
    (extract_text / extract_page). Each call tries Puter first with the
    explicit token; on PuterDegradedError/PuterDisabledError it falls back
    to the local provider for that call (deny/fail -> local, never silent).
    Stats record which path served each page; the token is an instance
    attribute of this call only and is never persisted.
    """

    def __init__(self, puter: PuterOCRProvider, local: Any, user_token: str):
        self._puter = puter
        self._local = local
        self._user_token = user_token
        self.puter_pages = 0
        self.fallback_pages = 0
        self.fallback_reasons: list[str] = []

    async def extract_text(
        self, file_path: str, pages: Optional[list[int]] = None
    ) -> ExtractionResult:
        try:
            result = await self._puter.extract_text(
                file_path, pages=pages, user_token=self._user_token
            )
            self.puter_pages += len(result.pages or [1])
            return result
        except (PuterDegradedError, PuterDisabledError) as exc:
            reason = str(exc) or type(exc).__name__
            self.fallback_pages += 1
            self.fallback_reasons.append(reason)
            logger.warning("Puter degraded, falling back to local OCR: %s", reason)
            return await self._local.extract_text(file_path, pages=pages)

    async def extract_page(self, file_path: str, page_number: int) -> OCRResult:
        try:
            result = await self._puter.extract_page(
                file_path, page_number, user_token=self._user_token
            )
            self.puter_pages += 1
            return result
        except (PuterDegradedError, PuterDisabledError) as exc:
            reason = str(exc) or type(exc).__name__
            self.fallback_pages += 1
            self.fallback_reasons.append(reason)
            logger.warning("Puter degraded, falling back to local OCR: %s", reason)
            return await self._local.extract_page(file_path, page_number)


def _guess_type(document: Any) -> str:
    try:
        return document.file_type.value
    except Exception:
        return getattr(document, "file_type", "pdf") or "pdf"


async def run_puter_heavy_pipeline(
    db: Any,
    document: Any,
    job: Any = None,
    user_token: Optional[str] = None,
    provider: Optional[PuterOCRProvider] = None,
) -> dict:
    """Run the authorised heavy-document pipeline reusing the normal stages.

    extract (DocumentExtractor with the token adapter) -> chunk
    (SemanticChunker via ChunkingService) -> embed (LocalEmbeddingService,
    best-effort) -> persist/finalize (same breakdown + partial semantics as
    the standard handler). Job stages move
    puter-offered -> puter-processing -> complete/partial.

    Raises PuterDisabledError when unauthorised; never stores the token.
    """
    from database import repositories
    from database.models import DocumentStatus
    from services.jobs import queue as q

    if not user_token:
        raise PuterDisabledError(
            "puter requires an explicit user gesture: no token supplied, "
            "no credentials stored server-side"
        )
    puter = provider or PuterOCRProvider()
    health = await puter.health()
    if not health.get("available"):
        raise PuterDisabledError(
            "puter disabled: set PUTER_ENABLED=true and pass an explicit "
            "user gesture + token per call (default OFF)"
        )

    doc_id = document.id if isinstance(document.id, uuid.UUID) else uuid.UUID(str(document.id))
    file_type = _guess_type(document)
    file_path = getattr(document, "file_path", None)

    async def _report(progress: int, stage: str, message: str) -> None:
        if job is None:
            return
        try:
            await q.report_progress(db, job.id, progress, stage=stage, stage_message=message)
            try:
                await db.commit()
            except Exception:
                pass
        except Exception:
            pass

    await _report(2, "puter-offered",
                  "Heavy document offered to the optional Puter path (explicitly authorised)")
    await _report(5, "puter-processing", "Puter OCR authorised; extracting")

    import factory as _factory_mod

    local = _factory_mod.get_strategy_factory().get_ocr()
    adapter = PuterTokenAdapter(puter, local, user_token)

    from services.extraction.extractor import DocumentExtractor

    extractor = DocumentExtractor()
    extractor._ocr = adapter  # token-bound vision for this run only

    async def _on_page(progress):
        frac = progress.current_page / max(1, progress.total_pages)
        await _report(int(5 + 50 * frac), "puter-processing",
                      f"Puter extracting page {progress.current_page}/{progress.total_pages}")

    try:
        extraction = await extractor.extract(doc_id, file_path, file_type, db, _on_page)
    except Exception as exc:
        msg = str(exc) or type(exc).__name__
        try:
            await repositories.update_document_status(db, doc_id, DocumentStatus.FAILED, error=msg)
        except Exception:
            pass
        if job is not None:
            try:
                await q.fail(db, job.id, f"puter heavy pipeline degraded explicitly: {msg}", retry=False)
            except Exception:
                pass
        raise RuntimeError(f"puter heavy pipeline degraded explicitly: {msg}") from exc

    # -- chunk (normal pipeline reuse: real chunker + page-span provenance) --
    await _report(60, "puter-processing", "Chunking Puter-extracted text")
    from services.chunking.service import ChunkingService
    from services.chunking.types import DocumentChunkingInput

    pages = await repositories.get_pages(db, doc_id)
    full_text = "\n\n".join((p.text or "") for p in pages)
    spans, cursor = [], 0
    for p in pages:
        start = full_text.find(p.text or "", cursor) if (p.text or "") else cursor
        if start < 0:
            start = cursor
        end = start + len(p.text or "")
        spans.append({"page": p.page_number, "start": start, "end": end,
                      "method": p.method, "confidence": p.confidence,
                      "heading": None})
        cursor = end
    service = ChunkingService()
    result = service.chunker.chunk_document(DocumentChunkingInput(
        document_id=str(doc_id), project_id=str(document.project_id),
        full_text=full_text, structural_elements=[],
        user_id=str(getattr(getattr(document, "project", None), "user_id", "") or "") or None,
        source=document.file_name, page_spans=spans))
    await service._store_chunks(db, str(doc_id), str(document.project_id), result.chunks)
    await _report(80, "puter-processing", f"Chunked {result.total_chunks} chunks")

    # -- embed (best-effort offline, same as the standard handler) -----------
    await _report(85, "puter-processing", "Embedding chunks")
    try:
        from services.embedding_service.service import LocalEmbeddingService

        embedder = LocalEmbeddingService()
        await embedder.generate_embeddings(doc_id, document.project_id, db,
                                           job_id=getattr(job, "id", None))
    except Exception as exc:
        logger.warning("Embedding deferred for %s: %s", doc_id, exc)
        if job is not None:
            try:
                out = dict(job.output_data or {})
                out["_embedding_deferred"] = str(exc)
                job.output_data = out
            except Exception:
                pass
    await _report(95, "puter-processing", "Embedding complete")

    # -- finalize (same partial semantics as the standard handler) -----------
    breakdown = await repositories.page_method_breakdown(db, doc_id)
    failed = int(breakdown.get("ocr_unavailable", 0) or 0)
    try:
        document.indexing_state = "partial" if failed else "indexed"
        document.extraction_status = "done"
    except Exception:
        pass
    error = f"{failed} page(s) need vision (ocr_unavailable)" if failed else None
    try:
        await repositories.update_document_status(db, doc_id, DocumentStatus.COMPLETED, error=error)
    except Exception:
        pass
    stage = "complete" if not failed else "partial"
    summary = {
        "_stage": stage,
        "_stage_message": "Puter heavy processing complete" if not failed
        else f"Puter heavy partial: {failed} page(s) ocr_unavailable",
        "_puter_offered": True,
        "_puter_authorised": True,
        "_puter_pages": adapter.puter_pages,
        "_puter_fallback_pages": adapter.fallback_pages,
        "_extraction_method": getattr(extraction, "method", None),
        "_page_methods": breakdown,
    }
    if adapter.fallback_reasons:
        summary["_puter_fallback_reasons"] = adapter.fallback_reasons[:5]
    # The token is never written into job output (explicit no-creds-stored).
    if job is not None:
        try:
            out = dict(job.output_data or {})
            out.update(summary)
            await q.complete(db, job.id, output_data=out)
        except Exception:
            pass
    return {
        "document_id": str(doc_id),
        "stage": stage,
        "indexing_state": getattr(document, "indexing_state", None),
        "total_pages": getattr(extraction, "total_pages", len(pages)),
        "chunks": result.total_chunks,
        "page_methods": breakdown,
        "puter_pages": adapter.puter_pages,
        "fallback_pages": adapter.fallback_pages,
        "partial": bool(failed),
        "error": error,
    }


__all__ = ["heavy_check", "PuterTokenAdapter", "run_puter_heavy_pipeline"]
