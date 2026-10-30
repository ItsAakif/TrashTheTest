"""
Text Extraction API Routes

FastAPI endpoints for document extraction.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from database import repositories
from database.models.user import User
from services.auth.deps import get_current_user

from .service import extraction_service
from .types import DocumentExtractionResult


# Create router
router = APIRouter(prefix="/api/extraction", tags=["Extraction"])


# ==================== Request/Response Models ====================

class ExtractionRequest(BaseModel):
    """Request to trigger extraction"""
    document_id: str


class ExtractionResponse(BaseModel):
    """Response for extraction trigger"""
    success: bool
    message: str
    document_id: str
    job_id: Optional[str] = None


class ExtractionStatusResponse(BaseModel):
    """Response for extraction status"""
    document_id: str
    status: str
    extracted_text: Optional[str] = None
    total_pages: Optional[int] = None
    topics: Optional[list[str]] = None
    is_ocr_required: bool = False
    method: Optional[str] = None
    processing_time: Optional[float] = None


# ==================== Endpoints ====================

@router.post("/extract", response_model=ExtractionResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_extraction(
    request: ExtractionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger document extraction.
    
    This starts an async job to extract text from the document.
    """
    user_id = current_user.id
    
    try:
        doc_uuid = uuid.UUID(request.document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document_id format"
        )
    
    # Get document
    document = await repositories.get_document_by_id(db, doc_uuid)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check ownership (404 for another user's resource)
    project = await repositories.get_project_by_id(db, document.project_id)
    if not project or project.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check if already processing
    if document.processing_status == document.processing_status.__class__.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is already being processed"
        )
    
    # Create a job for tracking
    job = await repositories.create_job(
        db=db,
        job_type="text_extraction",
        input_data={
            "document_id": str(document.id),
            "project_id": str(document.project_id),
            "file_path": document.file_path,
            "file_type": document.file_type.value
        }
    )
    await db.commit()
    
    # Start extraction (in background, this would be handled by a worker)
    # For now, we'll run it directly
    try:
        result = await extraction_service.process_document(
            document_id=document.id,
            project_id=document.project_id,
            file_path=document.file_path,
            file_type=document.file_type.value,
            db=db,
            job_id=job.id
        )
        
        return ExtractionResponse(
            success=True,
            message=f"Extraction completed. {result.total_pages} pages processed.",
            document_id=request.document_id,
            job_id=str(job.id)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )


@router.get("/status/{document_id}", response_model=ExtractionStatusResponse)
async def get_extraction_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get extraction status and results"""
    user_id = current_user.id
    
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document_id format"
        )
    
    document = await repositories.get_document_by_id(db, doc_uuid)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check ownership (404 for another user's resource)
    project = await repositories.get_project_by_id(db, document.project_id)
    if not project or project.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Get job for additional info
    # (In production, store extraction metadata in a separate table)
    
    return ExtractionStatusResponse(
        document_id=document_id,
        status=document.processing_status.value,
        extracted_text=document.extracted_text[:500] if document.extracted_text else None,
        total_pages=document.page_count,
        is_ocr_required=document.is_ocr_required
    )


@router.get("/topics/{document_id}")
async def get_extraction_topics(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get extracted topics for research preparation"""
    user_id = current_user.id
    
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document_id format"
        )
    
    document = await repositories.get_document_by_id(db, doc_uuid)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Check ownership (404 for another user's resource)
    project = await repositories.get_project_by_id(db, document.project_id)
    if not project or project.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Topics would be stored in metadata or a separate table
    # For now, extract from extracted text
    if not document.extracted_text:
        return {"topics": [], "message": "No text extracted yet"}
    
    # Simple topic extraction
    keywords = [
        "definition", "concept", "theory", "principle", "law",
        "formula", "equation", "function", "analysis", "method",
        "process", "system", "structure", "example", "application"
    ]
    
    text_lower = document.extracted_text.lower()
    found_topics = []
    
    for keyword in keywords:
        if keyword in text_lower:
            idx = text_lower.find(keyword)
            start = max(0, idx - 30)
            end = min(len(document.extracted_text), idx + 30)
            context = document.extracted_text[start:end].replace("\n", " ")
            found_topics.append(f"...{context}...")
    
    return {
        "topics": found_topics[:10],
        "count": len(found_topics[:10])
    }


# Health check
@router.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "service": "extraction"}


__all__ = ["router"]