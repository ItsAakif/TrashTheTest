"""POST /api/ocr/puter: minimal user-authorised heavy-document OCR.

Contract (mirrors the provider + frontend wrapper):
- Default OFF: 503 with an explicit reason when PUTER_ENABLED=false.
- Explicit auth: per-call user token via X-Puter-Token header (never
  persisted, never logged). Missing token -> 401 explicit.
- Heavy threshold (~100 pages, configurable via PUTER_HEAVY_PAGE_THRESHOLD)
  is a warning gate: callers below the threshold get an explicit note but
  are still served when authorised (no hard invariant).
- Failure -> explicit degraded (502 with reason), never silent empty text.
- Success preserves provenance: text/confidence/method/page metadata.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Depends, Header, Request, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db

from database.models.user import User
from services.auth.deps import get_current_user
from services.strategies.puter import (
    PuterDegradedError,
    PuterDisabledError,
    PuterOCRProvider,
    is_heavy_document,
)
from .puter_pipeline import heavy_check, run_puter_heavy_pipeline


router = APIRouter(prefix="/api/ocr", tags=["ocr"])


class PuterOCRResponse(BaseModel):
    text: str
    confidence: float
    method: str
    page: int
    heavy_path: bool = True
    below_threshold_note: Optional[str] = None


class HeavyCheckResponse(BaseModel):
    heavy: bool
    page_count: int = Field(default=0, alias="pageCount")
    threshold: int
    enabled: bool
    warning: Optional[str] = None

    model_config = {"populate_by_name": True}


class PuterProcessResponse(BaseModel):
    document_id: str = Field(alias="documentId")
    job_id: Optional[str] = Field(default=None, alias="jobId")
    stage: str
    indexing_state: Optional[str] = Field(default=None, alias="indexingState")
    total_pages: int = Field(default=0, alias="totalPages")
    chunks: int = 0
    page_methods: dict = Field(default_factory=dict, alias="pageMethods")
    puter_pages: int = Field(default=0, alias="puterPages")
    fallback_pages: int = Field(default=0, alias="fallbackPages")
    partial: bool = False
    error: Optional[str] = None
    message: str = ""

    model_config = {"populate_by_name": True}


@router.get("/puter/heavy-check", response_model=HeavyCheckResponse)
async def puter_heavy_check(
    page_count: int = 0,
    current_user: User = Depends(get_current_user),
):
    """Warning gate: is this document heavy (~100 pages, configurable)?"""
    return HeavyCheckResponse(**heavy_check(page_count))


@router.post("/puter/document/{document_id}", response_model=PuterProcessResponse)
async def puter_process_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    puter_token: Optional[str] = Header(default=None, alias="X-Puter-Token"),
):
    """Authorised heavy-document run: Puter vision -> normal pipeline.

    Explicit per-call auth via X-Puter-Token (never persisted, never
    logged). Default OFF -> 503; missing token -> 401; Puter failure ->
    explicit degraded with local fallback per page (never silent).
    Ownership is enforced (another user's document -> 404).
    """
    from api_gateway.deps import require_document_access
    from services.jobs import queue as job_queue

    document, _project = await require_document_access(document_id, current_user, db)
    provider = PuterOCRProvider()
    health = await provider.health()
    if not health.get("available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Puter path is disabled by default (PUTER_ENABLED=false). "
                   "Set PUTER_ENABLED=true and pass an explicit per-call token.",
        )
    if not puter_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Puter requires an explicit per-call user token "
                   "(X-Puter-Token); nothing is stored server-side.",
        )
    try:
        job = await job_queue.latest_for_document(db, document.id)
    except Exception:
        job = None
    try:
        summary = await run_puter_heavy_pipeline(
            db, document, job, user_token=puter_token, provider=provider)
    except PuterDisabledError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    try:
        await db.commit()
    except Exception:
        pass
    return PuterProcessResponse(
        documentId=summary["document_id"],
        jobId=str(job.id) if job is not None else None,
        stage=summary["stage"],
        indexingState=summary["indexing_state"],
        totalPages=summary["total_pages"],
        chunks=summary["chunks"],
        pageMethods=summary["page_methods"],
        puterPages=summary["puter_pages"],
        fallbackPages=summary["fallback_pages"],
        partial=summary["partial"],
        error=summary["error"],
        message=("Puter heavy processing complete; normal pipeline reused "
                 "(extract->chunk->embed) with provenance preserved."
                 if not summary["partial"] else
                 "Puter heavy partial: some pages ocr_unavailable (explicit, not silent)."),
    )


@router.post("/puter", response_model=PuterOCRResponse)
async def puter_ocr(
    request: Request,
    current_user: User = Depends(get_current_user),
    puter_page: int = Header(default=1, alias="X-Puter-Page"),
    puter_token: Optional[str] = Header(default=None, alias="X-Puter-Token"),
    puter_pages: Optional[int] = Header(default=None, alias="X-Puter-Pages"),
):
    provider = PuterOCRProvider()
    health = await provider.health()
    if not health.get("available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Puter path is disabled by default (PUTER_ENABLED=false). "
                   "Set PUTER_ENABLED=true and pass an explicit per-call token.",
        )
    if not puter_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Puter requires an explicit per-call user token "
                   "(X-Puter-Token); nothing is stored server-side.",
        )

    body = await request.body()
    if not body:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Empty page image")

    note = None
    try:
        total_pages = int(puter_pages) if puter_pages is not None else None
    except (TypeError, ValueError):
        total_pages = None
    if total_pages is not None and not is_heavy_document(total_pages):
        note = (
            f"Document has ~{total_pages} pages (below the heavy threshold): "
            "served on explicit authorisation; local PP-OCR remains the default."
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".img") as tmp:
            tmp.write(body)
            tmp_path = tmp.name
        try:
            page_no = int(puter_page)
        except (TypeError, ValueError):
            page_no = 1
        try:
            result = await provider.process_page(
                tmp_path, user_token=puter_token, page_number=page_no)
        except PuterDisabledError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail=str(exc))
        except PuterDegradedError as exc:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=f"Puter OCR degraded explicitly: {exc}")
        return PuterOCRResponse(
            text=result.text,
            confidence=float(result.confidence),
            method=result.method,
            page=int(getattr(result, "page_number", page_no) or page_no),
            heavy_path=True,
            below_threshold_note=note,
        )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@router.get("/puter/health")
async def puter_health():
    provider = PuterOCRProvider()
    return await provider.health()


__all__ = ["router"]
