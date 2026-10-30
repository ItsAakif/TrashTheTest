"""Document processing stage machine (T209).

validate -> extract (per page) -> chunk -> embed -> index -> finalize.

Progress derives from real counts (pages processed / pages total, chunks
embedded / chunks total); STAGE_PROGRESS_RANGES is the static budget within
which each stage reports. Completed stages are skipped on retry (resume
from job.stage); cancellation is checked at stage boundaries; permanent
failure marks the document failed with the real error message.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

STAGES = ["validate", "extract", "structure", "validate2", "chunk", "embed", "complete"]

# Static budget per stage (kept); within a stage the percentage comes from
# real counts mapped into the stage's slice.
STAGE_PROGRESS_RANGES = {
    "validate": (0, 5),
    "extract": (5, 55),
    "structure": (55, 60),
    "validate2": (60, 65),
    "chunk": (65, 80),
    "embed": (80, 95),
    "complete": (95, 100),
}


def _slice(stage: str, frac: float) -> int:
    lo, hi = STAGE_PROGRESS_RANGES[stage]
    return int(lo + (hi - lo) * max(0.0, min(1.0, frac)))


async def _is_cancelled(db: Any, job: Any) -> bool:
    try:
        fresh = await db.get(type(job), job.id)
        return fresh is not None and fresh.status.value == "cancelled"
    except Exception:
        return False


async def process_job(ctx: Any, job: Any) -> None:
    """Run the resumable stage machine for one document_processing job."""
    from database import repositories
    from database.models import DocumentStatus
    from services.jobs import queue as q

    db = ctx.db
    data = dict(job.input_data or {})
    doc_id_raw = data.get("document_id")
    if not doc_id_raw:
        await q.fail(db, job.id, "Job has no document_id", retry=False)
        return
    try:
        doc_id = doc_id_raw if isinstance(doc_id_raw, uuid.UUID) else uuid.UUID(str(doc_id_raw))
    except (ValueError, TypeError):
        await q.fail(db, job.id, f"Invalid document_id: {doc_id_raw}", retry=False)
        return

    document = await repositories.get_document_by_id_with_project(db, doc_id)
    if document is None:
        document = await repositories.get_document_by_id(db, doc_id)
    if document is None:
        await q.fail(db, job.id, "Document was deleted before processing", retry=False)
        return

    # T401: prefer ids from job.input_data so no relationship access is needed
    # at all; the eager-loaded .project above is a fallback only.
    project_id = data.get("project_id") or getattr(document, "project_id", None)
    user_id = data.get("user_id")
    if user_id is None:
        try:
            project = getattr(document, "project", None)
            user_id = getattr(project, "user_id", None) if project is not None else None
        except Exception:
            user_id = None

    done = set((job.output_data or {}).get("_stages_done", []))
    if job.stage in STAGES:
        # Resume: everything at or before the recorded stage is done.
        done.update(STAGES[:STAGES.index(job.stage) + 1])

    # T409: single monotonic progress writer. Every percentage derives from a
    # real count mapped into the stage slice; never decreases within a run.
    # Progress is COMMITTED here (not just flushed): the worker holds one
    # long transaction per job, so without this commit the API pollers keep
    # reading progress=0 until the job finishes (the "stuck 0%" symptom).
    # expire_on_commit=False (database/engine.py) keeps ORM objects usable.
    _last_progress = [int(job.progress or 0)]

    async def _report(progress: int, stage: str, message: str) -> None:
        pct = max(0, min(100, int(progress)))
        if pct < _last_progress[0]:
            pct = _last_progress[0]
        _last_progress[0] = pct
        await q.report_progress(db, job.id, pct, stage=stage, stage_message=message)
        try:
            await db.commit()
        except Exception as exc:
            logger.warning("Progress commit failed for job %s: %s", job.id, exc)
            try:
                await db.rollback()
            except Exception:
                pass

    def _mark_done(stage: str) -> None:
        done.add(stage)
        out = dict(job.output_data or {})
        out["_stages_done"] = sorted(done)
        out["_stage"] = stage
        job.output_data = out

    file_type = data.get("file_type") or _guess_type(document)
    file_path = data.get("file_path") or document.file_path
    extraction = None

    try:
        # -- validate -------------------------------------------------
        if "validate" not in done:
            if await _is_cancelled(db, job):
                return
            await _report(_slice("validate", 0.5),
                          stage="validate", message="Validating document record")
            if document is None:
                raise RuntimeError("Document was deleted before processing")
            await repositories.update_document_status(db, doc_id, DocumentStatus.PROCESSING)
            _mark_done("validate")

        # -- puter offer (heavy-document warning gate, never a blocker) --
        # Heavy docs (>= threshold, default ~100 pages, configurable) are
        # OFFERED the optional Puter path: the stage is recorded so the UI
        # can show the warning + explicit Authorise button, but the local
        # pipeline below always runs (Puter is never required).
        try:
            from services.strategies.puter import is_heavy_document as _is_heavy

            _hint = data.get("page_count")
            if _hint is None:
                _hint = getattr(document, "page_count", None)
            try:
                _threshold = None
                from config.settings import get_puter_config as _get_pc

                _threshold = int(_get_pc().heavy_page_threshold)
            except Exception:
                _threshold = None
            if _hint is not None and _is_heavy(int(_hint), threshold=_threshold) \
                    and not data.get("puter_authorised"):
                await _report(
                    _slice("extract", 0.0), stage="puter-offered",
                    message=(
                        f"Heavy document (~{_hint} pages, threshold "
                        f"~{_threshold or 100}): local OCR continues; the optional "
                        "Puter path is available on explicit authorisation"))
                _out = dict(job.output_data or {})
                _out["_puter_offered"] = True
                job.output_data = _out
        except Exception:
            pass  # the offer is advisory; it must never break local processing

        # -- extract (per page, real counts) ---------------------------
        if "extract" not in done:
            from services.extraction.extractor import DocumentExtractor

            extractor = DocumentExtractor()

            async def _on_page(progress):
                if await _is_cancelled(db, job):
                    raise asyncio.CancelledError()
                frac = (progress.current_page / max(1, progress.total_pages))
                await _report(_slice("extract", frac),
                              stage="extract",
                              message=f"Extracting page {progress.current_page}/{progress.total_pages}")

            extraction = await extractor.extract(doc_id, file_path, file_type, db, _on_page)
            if await _is_cancelled(db, job):
                return
            _mark_done("extract")
        else:
            extraction = await _reload_extraction(db, doc_id)

        # -- structure -------------------------------------------------
        if "structure" not in done:
            if await _is_cancelled(db, job):
                return
            pages = await repositories.get_pages(db, doc_id)
            await _report(_slice("structure", 1.0),
                          stage="structure",
                          message=f"Structure recorded for {len(pages)} pages")
            _mark_done("structure")

        # -- validate2 (post-extract gate, T402 honest degraded state) -------
        # (a) zero pages / extractor produced nothing -> real failure,
        #     retry=False, typed message. (b) pages exist but none yielded
        #     text (vision OCR unavailable) -> degraded terminal state
        #     indexing_state="no_text" with a recorded degraded_reason; the
        #     job completes once and is never retried.
        if "validate2" not in done:
            if await _is_cancelled(db, job):
                return
            pages = await repositories.get_pages(db, doc_id)
            with_text = [p for p in pages if (p.text or "").strip()]
            if not pages:
                no_pages_reason = "no extractable text: extractor produced zero pages"
                await repositories.set_document_indexing_state(
                    db, doc_id, "failed",
                    degraded_reason=no_pages_reason,
                    extraction_status="failed")
                await repositories.update_document_status(
                    db, doc_id, DocumentStatus.FAILED, error=no_pages_reason)
                _mark_done("validate2")
                await db.flush()
                await q.fail(db, job.id, no_pages_reason, retry=False)
                return
            if not with_text:
                reason = (
                    f"no extractable text: {len(pages)} page(s) present but none "
                    "yielded text (vision OCR unavailable)")
                await repositories.set_document_indexing_state(
                    db, doc_id, "no_text", degraded_reason=reason,
                    extraction_status="done")
                # Terminal honest outcome: the job completes, so the document
                # must read completed (with the no_text reason), never failed.
                await repositories.update_document_status(
                    db, doc_id, DocumentStatus.COMPLETED, error=reason)
                try:
                    document.extraction_status = "done"
                except Exception:
                    pass
                _mark_done("validate2")
                await db.flush()
                out = dict(job.output_data or {})
                out["_stage"] = "validate2"
                out["_stage_message"] = reason
                out["degraded_reason"] = reason
                out["indexing_state"] = "no_text"
                await q.complete(db, job.id, output_data=out)
                return
            await _report(_slice("validate2", 1.0),
                          stage="validate2",
                          message=f"Validated {len(with_text)}/{len(pages)} pages with text")
            _mark_done("validate2")

        # -- chunk ------------------------------------------------------
        if "chunk" not in done:
            from services.chunking.service import ChunkingService

            if await _is_cancelled(db, job):
                return
            pages = await repositories.get_pages(db, doc_id)
            full_text = "\n\n".join((p.text or "") for p in pages)
            # page spans map char offsets -> page provenance (never batch index).
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

            async def _on_chunk(progress):
                # T409: derive from the real reported count (progress_percent
                # 0-100 over batches); the previous frac computation was dead.
                frac = max(0.0, min(1.0, (progress.progress_percent or 0.0) / 100.0))
                made = getattr(progress, "chunks_created", "?")
                await _report(_slice("chunk", frac),
                              stage="chunk",
                              message=progress.message or f"Chunking ({made} chunks)")

            from services.chunking.types import DocumentChunkingInput
            from services.text_utils import find_boilerplate_lines

            result = service.chunker.chunk_document(DocumentChunkingInput(
                document_id=str(doc_id), project_id=str(project_id),
                full_text=full_text, structural_elements=[],
                user_id=str(user_id) if user_id is not None else None,
                source=document.file_name, page_spans=spans,
                boilerplate_lines=find_boilerplate_lines(
                    [(p.text or "") for p in pages])))
            await service._store_chunks(db, str(doc_id), str(project_id), result.chunks)
            await _report(_slice("chunk", 1.0),
                          stage="chunk", message=f"Chunked {result.total_chunks} chunks")
            _mark_done("chunk")
            await db.flush()  # T405: stage boundary owned by the worker

        # -- embed -------------------------------------------------------
        if "embed" not in done:
            if await _is_cancelled(db, job):
                return
            chunks = await repositories.get_chunks_by_document(db, doc_id)
            total = max(1, len(chunks))
            await _report(_slice("embed", 0.0),
                          stage="embed", message=f"Embedding 0/{len(chunks)} chunks")
            embedder = None
            try:
                from services.embedding_service.service import LocalEmbeddingService

                embedder = LocalEmbeddingService()
                # Warm the embedding weights ONCE before the batch loop. The
                # first run downloads ~1GB from HF into the persisted cache
                # (see modelcache volume); without this message the job sits
                # at "Embedding 0/N" for minutes with no explanation.
                await _report(_slice("embed", 0.0),
                              stage="embed",
                              message=("Loading embedding model "
                                       "(first run downloads weights)..."))
                try:
                    import asyncio as _asyncio

                    warm = getattr(embedder.provider, "warm", None)
                    if callable(warm):
                        res = warm()
                        if _asyncio.iscoroutine(res):
                            await res
                    else:
                        ensure = getattr(embedder.provider, "_ensure_model", None)
                        if callable(ensure):
                            res = ensure()
                            if _asyncio.iscoroutine(res):
                                await res
                except Exception as exc:
                    logger.warning("Embedding warmup for %s: %s", doc_id, exc)
                await _report(_slice("embed", 0.0),
                              stage="embed", message=f"Embedding 0/{len(chunks)} chunks")

                async def _on_embed(progress):
                    # Real counts: chunks_processed/total mapped into the slice.
                    done_n = getattr(progress, "chunks_processed", 0) or 0
                    total_n = getattr(progress, "total_chunks", 0) or 0
                    frac = (done_n / total_n) if total_n else 0.0
                    await _report(_slice("embed", frac),
                                  stage="embed",
                                  message=f"Embedding {done_n}/{total_n} chunks")

                result = await embedder.generate_embeddings(
                    doc_id, project_id, db, job_id=job.id,
                    progress_callback=_on_embed)
                embedded = getattr(result, "embedded_chunks", len(chunks)) or 0
                if len(chunks) and embedded < len(chunks):
                    reason = (f"partial embeddings: {embedded}/{len(chunks)} "
                              "chunks embedded")
                    await repositories.set_document_indexing_state(
                        db, doc_id, "partial", degraded_reason=reason)
                    out = dict(job.output_data or {})
                    out["_embedding_partial"] = reason
                    job.output_data = out
            except Exception as exc:
                # T404: embedding failure is recorded honestly, never
                # swallowed into a false "indexed" claim.
                logger.warning("Embedding deferred for %s: %s", doc_id, exc)
                reason = f"unembedded: embedding provider failed ({exc})"
                await repositories.set_document_indexing_state(
                    db, doc_id, "unembedded", degraded_reason=reason)
                out = dict(job.output_data or {})
                out["_embedding_deferred"] = str(exc)
                out["degraded_reason"] = reason
                out["indexing_state"] = "unembedded"
                job.output_data = out
            await _report(_slice("embed", 1.0),
                          stage="embed", message="Embedding complete")
            _mark_done("embed")
            await db.flush()  # T405: stage boundary owned by the worker

        # -- complete ----------------------------------------------------
        if await _is_cancelled(db, job):
            return
        breakdown = await repositories.page_method_breakdown(db, doc_id)
        failed = breakdown.get("ocr_unavailable", 0)
        # T404: never claim "indexed" when embeddings are missing/deferred.
        out_state = dict(job.output_data or {})
        embed_honest = out_state.get("indexing_state")
        degraded = out_state.get("degraded_reason") or out_state.get("_embedding_deferred")
        if failed:
            indexing_state = "partial"
            msg = f"Partial: {failed} page(s) ocr_unavailable"
        elif embed_honest in ("unembedded", "partial") or "_embedding_deferred" in out_state \
                or "_embedding_partial" in out_state:
            indexing_state = embed_honest if embed_honest in ("unembedded", "partial") else "partial"
            msg = degraded or "Partial: embeddings incomplete"
        else:
            indexing_state = "indexed"
            msg = "Processing complete"
        try:
            document.indexing_state = indexing_state
            document.extraction_status = "done"
        except Exception:
            pass
        await db.flush()  # attribute writes land before any commit expires them
        await repositories.update_document_status(
            db, doc_id,
            DocumentStatus.COMPLETED,
            error=(f"{failed} page(s) need vision (ocr_unavailable)" if failed
                   else (degraded if isinstance(degraded, str) else None)))
        _mark_done("complete")
        out = dict(job.output_data or {})
        out["_stage"] = "complete"
        out["_stage_message"] = msg
        out["indexing_state"] = indexing_state
        if indexing_state != "indexed" and "degraded_reason" not in out:
            out["degraded_reason"] = msg
        await q.complete(db, job.id, output_data=out)
    except asyncio.CancelledError:
        logger.info("Job %s cancelled; stopping at page/stage boundary", job.id)
        return
    except Exception as exc:
        msg = str(exc) or type(exc).__name__
        try:
            await repositories.update_document_status(db, doc_id,
                                                      __import__("database.models", fromlist=["DocumentStatus"]).DocumentStatus.FAILED,
                                                      error=msg)
        except Exception:
            pass
        await q.fail(db, job.id, msg, retry=True)


def _guess_type(document: Any) -> str:
    try:
        return document.file_type.value
    except Exception:
        return "pdf"


async def _reload_extraction(db: Any, doc_id: uuid.UUID):
    from database import repositories

    pages = await repositories.get_pages(db, doc_id)
    full = "\n\n".join((p.text or "") for p in pages)
    return {"pages": pages, "full_text": full}


__all__ = ["process_job", "STAGES", "STAGE_PROGRESS_RANGES"]
