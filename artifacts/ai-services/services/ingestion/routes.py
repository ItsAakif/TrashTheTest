"""Document ingestion API (T110).

Identity from get_current_user, ownership via require_project_access /
require_document_access, storage via StorageBackend, camelCase response
shape via pydantic aliases (DB fields keep snake_case). Progress is derived
from the document + latest job row - never a hardcoded status->50 mapping.
Supports Idempotency-Key header and checksum dedupe per project.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile, status
from fastapi.exceptions import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import require_document_access, require_project_access
from database.engine import get_db
from database.models import DocumentStatus, DocumentSourceType
from database.models.project import Project
from database.models.user import User
from database import repositories
from services.auth.deps import get_current_user
from services.jobs import queue as job_queue
from services.storage import get_default_backend
from .validation import (
    EXTENSION_TO_MIME,
    MAX_FILE_SIZE,
    STREAM_CHUNK_SIZE,
    ValidationError,
    read_upload_in_chunks,
    requires_ocr,
    validate_file_type,
    validate_magic_bytes,
)


router = APIRouter(prefix="/api/documents", tags=["Documents"])


def _camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _source_type(value: Optional[str]) -> str:
    raw = (value or "study_material").strip().lower()
    allowed = {item.value for item in DocumentSourceType}
    if raw not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source_type must be one of: " + ", ".join(sorted(allowed)),
        )
    return raw


def _stored_source_type(document: Any, fallback: str = "study_material") -> str:
    value = getattr(document, "source_type", None)
    return str(getattr(value, "value", value) or fallback)


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=_camel)

    document_id: str = Field(alias="documentId")
    file_name: str = Field(alias="fileName")
    file_type: str = Field(alias="fileType")
    source_type: str = Field(default="study_material", alias="sourceType")
    status: str
    message: str
    job_id: Optional[str] = Field(default=None, alias="jobId")


class DocumentDetailResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=_camel)

    id: str
    project_id: str = Field(alias="projectId")
    file_name: str = Field(alias="fileName")
    file_type: str = Field(alias="fileType")
    source_type: str = Field(default="study_material", alias="sourceType")
    file_path: str = Field(alias="filePath")
    page_count: Optional[int] = Field(default=None, alias="pageCount")
    processing_status: str = Field(alias="processingStatus")
    processing_error: Optional[str] = Field(default=None, alias="processingError")
    is_ocr_required: bool = Field(alias="isOcrRequired")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    # File metadata (T110; None for pre-migration rows)
    mime_type: Optional[str] = Field(default=None, alias="mimeType")
    file_size: Optional[int] = Field(default=None, alias="fileSize")
    checksum: Optional[str] = None
    # Phase 2 provenance (T210): visible partial state, never silent.
    indexing_state: Optional[str] = Field(default=None, alias="indexingState")
    page_methods: Optional[dict] = Field(default=None, alias="pageMethods")
    page_count_confirmed: Optional[int] = Field(default=None, alias="pageCountConfirmed")


class DocumentProgressResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=_camel)

    document_id: str = Field(alias="documentId")
    status: str
    progress: int
    stage: Optional[str] = None
    message: Optional[str] = None


class DocumentListResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    documents: list[DocumentDetailResponse]
    total: int


class DeleteResponse(BaseModel):
    success: bool
    message: str


async def _to_detail(document, db: AsyncSession | None = None) -> DocumentDetailResponse:
    methods = None
    if db is not None:
        try:
            methods = await repositories.page_method_breakdown(db, document.id)
        except Exception:
            methods = None
    return DocumentDetailResponse(
        id=str(document.id),
        projectId=str(document.project_id),
        fileName=document.file_name,
        fileType=document.file_type.value,
        sourceType=_stored_source_type(document),
        filePath=document.file_path,
        pageCount=document.page_count,
        processingStatus=document.processing_status.value,
        processingError=document.processing_error,
        isOcrRequired=document.is_ocr_required,
        createdAt=document.created_at,
        updatedAt=document.updated_at,
        mimeType=getattr(document, "mime_type", None),
        fileSize=getattr(document, "file_size", None),
        checksum=getattr(document, "checksum", None),
        indexingState=getattr(document, "indexing_state", None),
        pageMethods=methods,
        pageCountConfirmed=getattr(document, "page_count_confirmed", None),
    )


async def _latest_job_progress(db: AsyncSession, document) -> tuple[str, int, Optional[str], Optional[str]]:
    """Real progress from the latest job row; 'not_queued' when none."""
    try:
        job = await job_queue.latest_for_document(db, document.id)
    except Exception:
        job = None
    if job is None:
        status_value = document.processing_status.value
        if status_value == "pending":
            return "not_queued", 0, "not_queued", "Waiting to be queued"
        if status_value == "completed":
            return status_value, 100, "finalization", "Processing complete"
        if status_value == "failed":
            return status_value, 100, "failed", document.processing_error
        return status_value, 0, "not_queued", "Waiting to be queued"
    stage = job.stage or (job.output_data or {}).get("_stage")
    message = job.stage_message or (job.output_data or {}).get("_stage_message") or job.error_message
    return job.status.value, int(job.progress or 0), stage, message


async def _enqueue_processing_job(
    db: AsyncSession,
    document,
    project: Project,
    file_type: str,
    is_ocr: bool,
    idempotency_key: Optional[str],
    source_type: str = "study_material",
):
    from database.models.job import JobType

    job = await job_queue.enqueue(
        db,
        JobType.DOCUMENT_PROCESSING,
        input_data={
            "document_id": str(document.id),
            "project_id": str(project.id),
            "file_path": document.file_path,
            "file_type": file_type,
            "source_type": source_type,
            "is_ocr_required": is_ocr,
        },
        project_id=project.id,
        document_id=document.id,
        idempotency_key=idempotency_key,
        # Uploads outrank academic LLM work: never starve a PDF ingest behind
        # a multi-minute notes job.
        priority=job_queue.PRIORITY_DOCUMENT_PROCESSING,
    )
    await db.commit()
    await db.refresh(job)
    return job


async def _dedupe_response_for_existing(
    db: AsyncSession,
    existing_doc,
    project: Project,
    file_type: str,
    is_ocr: bool,
    source_type: str = "study_material",
) -> DocumentUploadResponse:
    """Checksum-dedupe decision for an already-stored document.

    A live (pending/processing) latest job is reused honestly. A dead
    terminal state (job failed/cancelled, missing job, or document failed)
    is REQUEUED with a fresh job so a re-upload retries instead of handing
    back a dead job; completed documents keep returning the existing row.
    Ownership is unchanged: the caller already passed require_project_access
    and the new job is scoped to the same project/document.
    """
    from database.models.job import JobStatus

    try:
        existing_job = await job_queue.latest_for_document(db, existing_doc.id)
    except Exception:
        existing_job = None
    active = (
        existing_job is not None
        and existing_job.status in (JobStatus.PENDING, JobStatus.PROCESSING)
    )
    doc_status = getattr(existing_doc, "processing_status", None)
    doc_failed = doc_status == DocumentStatus.FAILED or getattr(doc_status, "value", doc_status) == "failed"
    terminal_dead = (
        existing_job is None
        or existing_job.status in (JobStatus.FAILED, JobStatus.CANCELLED)
        or doc_failed
    )
    if active or not terminal_dead:
        return DocumentUploadResponse(
            documentId=str(existing_doc.id),
            fileName=existing_doc.file_name,
            fileType=existing_doc.file_type.value,
            sourceType=_stored_source_type(existing_doc),
            status=existing_doc.processing_status.value,
            message="Duplicate content: returning existing document",
            jobId=str(existing_job.id) if existing_job else None,
        )
    # Dead state -> honest retry: reset the document and queue a fresh job.
    prev = existing_job.status.value if existing_job is not None else "missing job"
    try:
        await repositories.update_document_status(db, existing_doc.id, DocumentStatus.PENDING)
    except Exception:
        pass
    for attr, value in (
        ("processing_error", None),
        ("extraction_status", "pending"),
        ("ocr_status", "pending" if is_ocr else "not_required"),
        ("embedding_status", "pending"),
        ("indexing_state", "pending"),
    ):
        try:
            setattr(existing_doc, attr, value)
        except Exception:
            pass
    job = await _enqueue_processing_job(
        db, existing_doc, project, file_type, is_ocr, None,
        source_type=_stored_source_type(existing_doc, source_type),
    )
    return DocumentUploadResponse(
        documentId=str(existing_doc.id),
        fileName=existing_doc.file_name,
        fileType=existing_doc.file_type.value,
        sourceType=_stored_source_type(existing_doc, source_type),
        status=DocumentStatus.PENDING.value,
        message=f"Duplicate content: previous attempt {prev}; requeued for processing",
        jobId=str(job.id),
    )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    project_id: str = Form(...),
    source_type: str = Form("study_material"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Standard multipart upload (up to MAX_STANDARD_UPLOAD_SIZE).

    Streams the body in bounded chunks with an incremental byte counter -
    never buffers the whole file beyond one chunk.
    """
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project_id format")
    project = await require_project_access(project_uuid, current_user, db)
    source_type = _source_type(source_type)

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")
    try:
        validate_file_type(file.filename)
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    # Idempotency: same key + project + type returns the existing queued job.
    if idempotency_key:
        from database.models.job import JobType as _JT

        existing = await job_queue.find_by_idempotency_key(db, idempotency_key, project.id, _JT.DOCUMENT_PROCESSING)
        if existing is not None and existing.document_id is not None:
            doc = await repositories.get_document_by_id(db, existing.document_id)
            if doc is not None:
                return DocumentUploadResponse(
                    documentId=str(doc.id),
                    fileName=doc.file_name,
                    fileType=doc.file_type.value,
                    sourceType=_stored_source_type(doc),
                    status=doc.processing_status.value,
                    message="Duplicate request: returning existing document",
                    jobId=str(existing.id),
                )

    # Single bounded pass: read chunks, sniff the head, enforce the limit
    # inline, accumulate (bounded by MAX_STANDARD_UPLOAD_SIZE, 10MB default).
    backend = get_default_backend()
    scope_hint = f"{current_user.id}/{project.id}/{file.filename}"
    buf = bytearray()
    sniffed_ext: Optional[str] = None
    try:
        async for chunk in read_upload_in_chunks(file, allow_large=False):
            if not buf:
                try:
                    sniffed_ext = validate_magic_bytes(chunk[:8192], file.filename)
                except ValidationError as e:
                    raise HTTPException(status_code=e.status_code, detail=e.message)
            buf.extend(chunk)
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    total = len(buf)
    if total == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
    body = bytes(buf)

    checksum = hashlib.sha256(body).hexdigest()
    file_ext = sniffed_ext or validate_file_type(file.filename)
    is_ocr = requires_ocr(file_ext)
    # Checksum dedupe: same content + same project returns the existing doc,
    # or requeues it when the previous attempt died (never a dead job id).
    try:
        existing_docs = await repositories.get_documents_by_project(db, project.id)
        for existing_doc in existing_docs:
            if getattr(existing_doc, "checksum", None) == checksum:
                return await _dedupe_response_for_existing(
                    db, existing_doc, project, file_ext[1:], is_ocr,
                    source_type=source_type)
    except HTTPException:
        raise
    except Exception:
        pass  # dedupe is best-effort on stores without the new column

    storage_key = await backend.save(scope_hint, body)
    mime = EXTENSION_TO_MIME.get(file_ext, "application/octet-stream")

    try:
        document = await repositories.create_document(
            db=db,
            project_id=project.id,
            file_name=file.filename,
            file_type=file_ext[1:],
            file_path=storage_key,
            is_ocr_required=is_ocr,
            source_type=source_type,
        )
        try:
            document.storage_key = storage_key
            document.mime_type = mime
            document.file_size = total
            document.checksum = checksum
            document.extraction_status = "pending"
            document.ocr_status = "pending" if is_ocr else "not_required"
            document.embedding_status = "pending"
        except Exception:
            pass  # pre-migration stores without the new columns
    except Exception as e:
        await backend.delete(storage_key)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to create document record: {e}")

    try:
        job = await _enqueue_processing_job(
            db, document, project, file_ext[1:], is_ocr, idempotency_key,
            source_type=source_type,
        )
        job_id = str(job.id)
    except Exception:
        await db.rollback()
        # Document row survives without a job (progress reports not_queued).
        docs = await repositories.get_documents_by_project(db, project.id)
        document = next((d for d in docs if getattr(d, "checksum", None) == checksum), document)
        await db.commit()
        job_id = None

    return DocumentUploadResponse(
        documentId=str(document.id),
        fileName=file.filename,
        fileType=file_ext[1:],
        sourceType=source_type,
        status=DocumentStatus.PENDING.value,
        message="Document uploaded successfully. Processing will begin shortly.",
        jobId=job_id,
    )


@router.post("/upload/stream", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_stream(
    request: Request,
    project_id: str = Header(..., alias="X-Project-Id"),
    file_name: str = Header(..., alias="X-File-Name"),
    source_type: str = Header("study_material", alias="X-Source-Type"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Streaming upload for large files (up to MAX_FILE_SIZE).

    Writes the request body to storage incrementally in STREAM_CHUNK_SIZE
    chunks with an inline byte counter; over-limit streams abort and remove
    the partial file. Peak extra memory is ~2 chunks, documented in
    validation.STREAM_CHUNK_SIZE.
    """
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project_id format")
    project = await require_project_access(project_uuid, current_user, db)
    source_type = _source_type(source_type)

    if not file_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="X-File-Name is required")
    try:
        validate_file_type(file_name)
    except ValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    backend = get_default_backend()
    scope_hint = f"{current_user.id}/{project.id}/{file_name}"
    digest = hashlib.sha256()
    total = 0
    sniff_head = b""
    sniffed_ext: Optional[str] = None
    storage_key: Optional[str] = None
    dest_tmp: Optional[str] = None

    # Stream body -> temp backend object, chunk by chunk.
    import tempfile

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    try:
        async for chunk in request.stream():
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                tmp.close()
                try:
                    import os as _os

                    _os.unlink(tmp_path)
                except Exception:
                    pass
                raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                                    detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024 * 1024):.1f}MB")
            if len(sniff_head) < 8192:
                sniff_head += chunk[: 8192 - len(sniff_head)]
                if sniffed_ext is None and len(sniff_head) >= 4:
                    try:
                        sniffed_ext = validate_magic_bytes(sniff_head, file_name)
                    except ValidationError:
                        pass  # need more bytes (zip head) or real mismatch; decided below
            digest.update(chunk)
            tmp.write(chunk)
        tmp.close()

        if total == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")
        try:
            sniffed_ext = validate_magic_bytes(sniff_head, file_name)
        except ValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.message)

        checksum = digest.hexdigest()
        file_ext = sniffed_ext or validate_file_type(file_name)
        is_ocr = requires_ocr(file_ext)
        try:
            existing_docs = await repositories.get_documents_by_project(db, project.id)
            for existing_doc in existing_docs:
                if getattr(existing_doc, "checksum", None) == checksum:
                    return await _dedupe_response_for_existing(
                        db, existing_doc, project, file_ext[1:], is_ocr,
                    source_type=source_type)
        except HTTPException:
            raise
        except Exception:
            pass

        # Move bytes into storage without ever holding the whole file in
        # memory: re-stream the temp file through the backend in chunks.
        with open(tmp_path, "rb") as f:
            storage_key, stored_total, _ = await backend.save_stream(
                scope_hint, f, chunk_size=STREAM_CHUNK_SIZE, max_bytes=MAX_FILE_SIZE
            )
        total = stored_total
        mime = EXTENSION_TO_MIME.get(file_ext, "application/octet-stream")
        try:
            document = await repositories.create_document(
                db=db,
                project_id=project.id,
                file_name=file_name,
                file_type=file_ext[1:],
                file_path=storage_key,
                is_ocr_required=is_ocr,
                source_type=source_type,
            )
            try:
                document.storage_key = storage_key
                document.mime_type = mime
                document.file_size = total
                document.checksum = checksum
                document.extraction_status = "pending"
                document.ocr_status = "pending" if is_ocr else "not_required"
                document.embedding_status = "pending"
            except Exception:
                pass
        except Exception as e:
            await backend.delete(storage_key)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Failed to create document record: {e}")
        try:
            job = await _enqueue_processing_job(
                db, document, project, file_ext[1:], is_ocr, idempotency_key,
                source_type=source_type,
            )
            job_id = str(job.id)
        except Exception:
            await db.rollback()
            job_id = None
    finally:
        try:
            import os as _os

            _os.unlink(tmp_path)
        except Exception:
            pass

    return DocumentUploadResponse(
        documentId=str(document.id),
        fileName=file_name,
        fileType=file_ext[1:],
        sourceType=source_type,
        status=DocumentStatus.PENDING.value,
        message="Document uploaded successfully. Processing will begin shortly.",
        jobId=job_id,
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: str,
    access: tuple = Depends(require_document_access),
    db: AsyncSession = Depends(get_db),
):
    document, _project = access
    return await _to_detail(document, db)


@router.get("/{document_id}/progress", response_model=DocumentProgressResponse)
async def get_document_progress(
    document_id: str,
    access: tuple = Depends(require_document_access),
    db: AsyncSession = Depends(get_db),
):
    document, _project = access
    status_value, progress, stage, message = await _latest_job_progress(db, document)
    return DocumentProgressResponse(
        documentId=str(document.id),
        status=status_value,
        progress=progress,
        stage=stage,
        message=message,
    )


@router.get("/{document_id}/file")
async def get_document_file(
    document_id: str,
    access: tuple = Depends(require_document_access),
):
    """Stream the stored source bytes inline (Phase B PDF viewer).

    Ownership is enforced by require_document_access (404 for another
    user's document, never content). Missing storage bytes yield a typed
    404, never a 500. Content-Disposition is inline so the browser/PDF
    viewer renders rather than downloads.
    """
    from fastapi.responses import StreamingResponse

    document, _project = access
    file_type = getattr(getattr(document, "file_type", None), "value", None) or "pdf"
    mime = getattr(document, "mime_type", None) or EXTENSION_TO_MIME.get(
        f".{file_type}", "application/octet-stream"
    )
    backend = get_default_backend()
    # exists() is the eager check: open_stream is a lazy async iterator,
    # so a missing key would otherwise surface mid-stream as a 500.
    if not await backend.exists(document.file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Source file unavailable for this document")
    filename = getattr(document, "file_name", "document") or "document"
    safe_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".", " ")).strip() or "document"
    return StreamingResponse(
        backend.open_stream(document.file_path),
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{safe_name}"'},
    )


@router.get("/project/{project_id}", response_model=DocumentListResponse)
async def list_project_documents(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project_id format")
    project = await require_project_access(project_uuid, current_user, db)
    documents = await repositories.get_documents_by_project(db, project.id)
    return DocumentListResponse(documents=[await _to_detail(d, db) for d in documents], total=len(documents))


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    access: tuple = Depends(require_document_access),
    db: AsyncSession = Depends(get_db),
):
    """Cascade delete (T210): cancel active job, remove pages/chunks/jobs/file."""
    from database.models.job import JobStatus as _JS

    document, _project = access
    # Resolve active job deterministically: cancel it, never orphan it.
    try:
        job = await job_queue.latest_for_document(db, document.id)
        if job is not None and job.status in (_JS.PENDING, _JS.PROCESSING):
            try:
                await job_queue.cancel(db, job.id)
            except Exception:
                pass
    except Exception:
        pass
    try:
        await repositories.delete_pages_for_document(db, document.id)
    except Exception:
        pass
    try:
        await repositories.delete_chunks_for_document(db, document.id)
    except Exception:
        pass
    await get_default_backend().delete(document.file_path)
    await db.delete(document)
    await db.commit()
    return DeleteResponse(success=True, message="Document deleted successfully")


@router.post("/upload/multi", status_code=status.HTTP_201_CREATED)
async def upload_documents_multi(
    files: list[UploadFile] = File(...),
    project_id: str = Form(...),
    source_type: str = Form("study_material"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Multi-file upload (T-phase2): each file gets an independent job.

    One file failing validation never affects the others; the response is
    [{document_id, job_id}] per file with per-file status/failure.
    """
    from fastapi.responses import JSONResponse

    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid project_id format")
    project = await require_project_access(project_uuid, current_user, db)
    source_type = _source_type(source_type)

    results: list[dict] = []
    for upload in files:
        try:
            if not upload.filename:
                results.append({"fileName": None, "documentId": None, "jobId": None,
                                "status": "failed", "error": "No filename provided"})
                continue
            try:
                validate_file_type(upload.filename)
            except ValidationError as e:
                results.append({"fileName": upload.filename, "documentId": None, "jobId": None,
                                "status": "failed", "error": e.message})
                continue
            backend = get_default_backend()
            scope_hint = f"{current_user.id}/{project.id}/{upload.filename}"
            buf = bytearray()
            sniffed_ext: Optional[str] = None
            try:
                async for chunk in read_upload_in_chunks(upload, allow_large=False):
                    if not buf:
                        try:
                            sniffed_ext = validate_magic_bytes(chunk[:8192], upload.filename)
                        except ValidationError as e:
                            raise e
                    buf.extend(chunk)
            except ValidationError as e:
                results.append({"fileName": upload.filename, "documentId": None, "jobId": None,
                                "status": "failed", "error": e.message})
                continue
            if not buf:
                results.append({"fileName": upload.filename, "documentId": None, "jobId": None,
                                "status": "failed", "error": "Empty file"})
                continue
            body = bytes(buf)
            checksum = hashlib.sha256(body).hexdigest()
            file_ext = sniffed_ext or validate_file_type(upload.filename)
            storage_key = await backend.save(scope_hint, body)
            is_ocr = requires_ocr(file_ext)
            try:
                document = await repositories.create_document(
                    db=db, project_id=project.id, file_name=upload.filename,
                    file_type=file_ext[1:], file_path=storage_key, is_ocr_required=is_ocr,
                    source_type=source_type)
                try:
                    document.mime_type = EXTENSION_TO_MIME.get(file_ext, "application/octet-stream")
                    document.file_size = len(body)
                    document.checksum = checksum
                except Exception:
                    pass
            except Exception as e:
                await backend.delete(storage_key)
                results.append({"fileName": upload.filename, "documentId": None, "jobId": None,
                                "status": "failed", "error": f"Failed to create document record: {e}"})
                continue
            try:
                job = await _enqueue_processing_job(
                    db, document, project, file_ext[1:], is_ocr, None,
                    source_type=source_type,
                )
                job_id = str(job.id)
            except Exception:
                await db.rollback()
                job_id = None
            results.append({"fileName": upload.filename, "documentId": str(document.id),
                            "sourceType": source_type, "jobId": job_id,
                            "status": DocumentStatus.PENDING.value, "error": None})
        except Exception as e:
            results.append({"fileName": getattr(upload, "filename", None), "documentId": None,
                            "jobId": None, "status": "failed", "error": str(e)})
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=results)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "document-ingestion"}


__all__ = ["router"]
