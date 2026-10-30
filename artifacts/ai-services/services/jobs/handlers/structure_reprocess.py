"""STRUCTURE_REPROCESS job: rebuild chunks + structure for one doc.

Re-runs the chunker on the document's already-extracted text (NOT the
file); (document_id, content_hash) upsert is the dedupe key so the
job is idempotent.  Old chunks whose content_hash is not in the new
set are deleted in the same transaction.  Document row is updated
with structure_method / structure_confidence / structure_degraded.

ADR-008.  Additive: no existing column, contract, or handler changes.
"""

import logging
import uuid
from typing import Any

from services.chunking.service import ChunkingService, chunking_service
from database import repositories

logger = logging.getLogger(__name__)

# Verdict defaults: used when the stamp path raises so q.complete() below
# can still report a well-formed output payload (spec initializes these
# before the try for exactly that reason).
_DEFAULT_METHOD = "page_bounded"


async def process_job(ctx: Any, job: Any) -> None:
    from services.jobs import queue as q

    db = ctx.db
    data = dict(getattr(job, "input_data", None) or {})
    doc_id_raw = data.get("document_id")
    if not doc_id_raw:
        await q.fail(db, job.id, "STRUCTURE_REPROCESS: no document_id",
                     retry=False)
        return
    try:
        doc_id = doc_id_raw if isinstance(doc_id_raw, uuid.UUID) \
            else uuid.UUID(str(doc_id_raw))
    except (ValueError, TypeError):
        await q.fail(db, job.id,
                     f"STRUCTURE_REPROCESS: invalid document_id "
                     f"{doc_id_raw!r}", retry=False)
        return

    document = await repositories.get_document_by_id_with_project(db, doc_id)
    if document is None:
        document = await repositories.get_document_by_id(db, doc_id)
    if document is None:
        await q.fail(db, job.id,
                     "STRUCTURE_REPROCESS: document was deleted",
                     retry=False)
        return

    full_text = getattr(document, "extracted_text", "") or ""
    if not full_text.strip():
        await q.fail(db, job.id,
                     "STRUCTURE_REPROCESS: no extracted_text to "
                     "reprocess", retry=False)
        return

    project_id = getattr(document, "project_id", None)
    user_id = data.get("user_id")

    svc: ChunkingService = chunking_service
    # repositories.get_pages orders by page_number ascending; we rebuild
    # char offsets because document_pages stores no start/end offsets.
    pages = await repositories.get_pages(db, doc_id)
    spans = []
    cursor = 0
    for p in pages:
        t = p.text or ""
        start = full_text.find(t, cursor) if t else cursor
        if start < 0:
            start = cursor
        end = start + len(t)
        spans.append({"page": p.page_number, "start": start,
                      "end": end,
                      "method": getattr(p, "method", None),
                      "confidence": getattr(p, "confidence", None),
                      "heading": None})
        cursor = end

    from services.chunking.types import DocumentChunkingInput
    result = svc.chunker.chunk_document(
        DocumentChunkingInput(
            document_id=str(doc_id),
            project_id=str(project_id) if project_id else str(doc_id),
            full_text=full_text,
            structural_elements=[],
            user_id=str(user_id) if user_id else None,
            source=getattr(document, "file_name", None),
            page_spans=spans,
        )
    )

    # Compute the set of new content_hashes.
    new_hashes = {c.content_hash for c in result.chunks}

    # Delete old chunks whose hash is not in the new set. Idempotent:
    # on a second run with the same input the deletion is a no-op.
    await repositories.delete_chunks_not_in_hashes(db, doc_id, new_hashes)

    await svc._store_chunks(db, str(doc_id),
                            str(project_id) if project_id else str(doc_id),
                            result.chunks)
    await db.commit()

    # Stamp the document with the structure verdict.  All columns
    # are additive JSONB / nullable; old rows stay valid.  The three
    # values are read by q.complete() after the except block, so they
    # are bound before the try (a stamp failure must not NameError).
    dominant_method = _DEFAULT_METHOD
    avg_conf = 0.0
    degraded = True
    try:
        methods = []
        confidences = []
        for c in result.chunks:
            meta = getattr(c, "metadata", None)
            if meta is None:
                continue
            method = getattr(meta, "structure_method", None)
            if method:
                methods.append(method)
            raw_conf = getattr(meta, "structure_confidence", None)
            if raw_conf is not None:
                confidences.append(float(raw_conf))
        dominant_method = (max(set(methods), key=methods.count)
                           if methods else _DEFAULT_METHOD)
        avg_conf = (sum(confidences) / len(confidences)
                    if confidences else 0.0)
        degraded = (
            dominant_method in {"page_bounded", "inferred"}
            or avg_conf < 0.6
        )
        await repositories.set_document_structure_verdict(
            db, doc_id, method=dominant_method,
            confidence=avg_conf, degraded=degraded)
        await db.commit()
    except Exception as exc:
        logger.warning(
            "STRUCTURE_REPROCESS verdict stamp failed for %s: %s",
            doc_id, exc)
        await db.rollback()

    await q.complete(db, job.id, output_data={
        "document_id": str(doc_id),
        "chunks_total": result.total_chunks,
        "method": dominant_method,
        "confidence": avg_conf,
        "degraded": degraded,
    })
