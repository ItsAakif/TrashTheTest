"""
Job Trigger Service

Triggers background jobs for document processing after ingestion.
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Job, JobType, JobStatus
from database.repositories import create_job, update_job_status


async def trigger_document_processing(
    db: AsyncSession,
    document_id: uuid.UUID,
    project_id: uuid.UUID,
    file_path: str,
    file_type: str,
    is_ocr_required: bool,
    extracted_text: Optional[str] = None
) -> Job:
    """
    Trigger a document processing job.
    
    This creates a background job that will:
    1. Extract text from the document (if not already done)
    2. Create semantic chunks
    3. Generate embeddings
    
    Args:
        db: Database session
        document_id: Document ID to process
        project_id: Project ID
        file_path: Storage path to file
        file_type: File type (pdf, docx, txt, image)
        is_ocr_required: Whether OCR is needed
        extracted_text: Pre-extracted text (if available)
        
    Returns:
        Created Job instance
    """
    # Create input data for the job
    input_data = {
        "document_id": str(document_id),
        "project_id": str(project_id),
        "file_path": file_path,
        "file_type": file_type,
        "is_ocr_required": is_ocr_required,
        "extracted_text": extracted_text,
    }
    
    # Create the job
    job = await create_job(
        db=db,
        job_type=JobType.DOCUMENT_PROCESSING.value,
        input_data=input_data
    )
    
    return job


async def trigger_text_extraction(
    db: AsyncSession,
    document_id: uuid.UUID,
    project_id: uuid.UUID,
    file_path: str,
    file_type: str
) -> Job:
    """
    Trigger a text extraction job (OCR).
    
    Args:
        db: Database session
        document_id: Document ID
        project_id: Project ID
        file_path: Storage path to file
        file_type: File type
        
    Returns:
        Created Job instance
    """
    input_data = {
        "document_id": str(document_id),
        "project_id": str(project_id),
        "file_path": file_path,
        "file_type": file_type,
    }
    
    job = await create_job(
        db=db,
        job_type=JobType.TEXT_EXTRACTION.value,
        input_data=input_data
    )
    
    return job


async def trigger_embedding_generation(
    db: AsyncSession,
    project_id: uuid.UUID,
    document_id: Optional[uuid.UUID] = None
) -> Job:
    """
    Trigger embedding generation for chunks.
    
    Args:
        db: Database session
        project_id: Project ID
        document_id: Optional document ID (for single document)
        
    Returns:
        Created Job instance
    """
    input_data = {
        "project_id": str(project_id),
    }
    
    if document_id:
        input_data["document_id"] = str(document_id)
    
    job = await create_job(
        db=db,
        job_type=JobType.EMBEDDING.value,
        input_data=input_data
    )
    
    return job


async def update_document_progress(
    db: AsyncSession,
    document_id: uuid.UUID,
    status: str,
    progress: int,
    error: Optional[str] = None
) -> None:
    """
    Update document processing progress.
    
    This is called by the job processor to update the document status.
    
    Args:
        db: Database session
        document_id: Document ID
        status: Processing status
        progress: Progress percentage (0-100)
        error: Error message if failed
    """
    from database.models import DocumentStatus
    from database.repositories import update_document_status
    
    # Map status to DocumentStatus enum
    status_map = {
        "pending": DocumentStatus.PENDING,
        "processing": DocumentStatus.PROCESSING,
        "completed": DocumentStatus.COMPLETED,
        "failed": DocumentStatus.FAILED,
    }
    
    doc_status = status_map.get(status.lower(), DocumentStatus.PENDING)
    
    await update_document_status(
        db=db,
        document_id=document_id,
        status=doc_status,
        error=error
    )


__all__ = [
    "trigger_document_processing",
    "trigger_text_extraction",
    "trigger_embedding_generation",
    "update_document_progress",
]