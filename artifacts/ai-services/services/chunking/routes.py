"""
Semantic Chunking API Routes

REST API endpoints for document chunking operations.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from database import repositories
from database.models import JobStatus

from .service import ChunkingService, chunking_service
from .types import ChunkingProgress, ChunkingResult


router = APIRouter(prefix="/api/chunking", tags=["chunking"])


class ChunkingRequest(BaseModel):
    """Request to start chunking for a document"""
    document_id: str
    project_id: str
    batch_size: Optional[int] = 50


class ChunkingResponse(BaseModel):
    """Response after starting chunking"""
    job_id: str
    status: str
    message: str


class ChunkStatusResponse(BaseModel):
    """Response for chunking status"""
    job_id: str
    status: str
    progress: int
    chunks_created: int
    message: Optional[str] = None


class ChunkInfoResponse(BaseModel):
    """Response for chunk information"""
    document_id: str
    total_chunks: int
    total_tokens: int
    average_tokens_per_chunk: float
    chunks: list


@router.post("/chunk", response_model=ChunkingResponse)
async def start_chunking(
    request: ChunkingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start chunking process for a document.
    
    This endpoint triggers the chunking job and returns immediately.
    The actual processing happens asynchronously.
    """
    try:
        # Validate document exists
        document_id = uuid.UUID(request.document_id)
        project_id = uuid.UUID(request.project_id)
        
        document = await repositories.get_document_by_id(db, document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {request.document_id} not found"
            )
        
        # Check document has extracted text
        if not document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no extracted text. Run extraction first."
            )
        
        # Create job for chunking
        job = await repositories.create_job(
            db=db,
            job_type="chunking",
            input_data={
                "document_id": request.document_id,
                "project_id": request.project_id,
                "batch_size": request.batch_size
            }
        )
        
        # Start chunking in background (fire and forget for now)
        # In production, use a proper task queue
        import asyncio
        
        async def run_chunking():
            try:
                # Get document text
                full_text = document.extracted_text
                
                # Run chunking service
                await chunking_service.process_document(
                    document_id=document_id,
                    project_id=project_id,
                    full_text=full_text,
                    structural_elements=[],  # Could pass from document metadata
                    db=db,
                    job_id=job.id
                )
                
            except Exception as e:
                # Update job on failure
                await repositories.update_job_status(
                    db=db,
                    job_id=job.id,
                    status=JobStatus.FAILED,
                    error_message=str(e)
                )
        
        # Run async
        asyncio.create_task(run_chunking())
        
        return ChunkingResponse(
            job_id=str(job.id),
            status="started",
            message="Chunking job started"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start chunking: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=ChunkStatusResponse)
async def get_chunking_status(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get the status of a chunking job"""
    try:
        job_uuid = uuid.UUID(job_id)
        job = await repositories.get_job_by_id(db, job_uuid)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Get chunks count from output data
        chunks_created = 0
        message = None
        if job.output_data:
            chunks_created = job.output_data.get("chunks_created", 0)
            message = job.output_data.get("message")
        
        return ChunkStatusResponse(
            job_id=job_id,
            status=job.status.value,
            progress=job.progress,
            chunks_created=chunks_created,
            message=message
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format"
        )


@router.get("/chunks/{document_id}", response_model=ChunkInfoResponse)
async def get_document_chunks(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get all chunks for a document"""
    try:
        doc_uuid = uuid.UUID(document_id)
        chunks = await repositories.get_chunks_by_document(db, doc_uuid)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No chunks found for document {document_id}"
            )
        
        total_tokens = sum(c.token_count or 0 for c in chunks)
        
        return ChunkInfoResponse(
            document_id=document_id,
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            average_tokens_per_chunk=total_tokens / len(chunks) if chunks else 0,
            chunks=[
                {
                    "id": str(c.id),
                    "content": c.content[:100] + "..." if len(c.content) > 100 else c.content,
                    "token_count": c.token_count,
                    "metadata": c.chunk_metadata
                }
                for c in chunks[:50]  # Limit to first 50 for response size
            ]
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )


@router.post("/resume/{document_id}")
async def resume_chunking(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Resume chunking from last processed position"""
    try:
        doc_uuid = uuid.UUID(document_id)
        
        document = await repositories.get_document_by_id(db, doc_uuid)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        if not document.extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no extracted text"
            )
        
        # Get project ID
        project = await repositories.get_project_by_id(db, document.project_id)
        
        # Resume chunking
        result = await chunking_service.resume_chunking(
            document_id=doc_uuid,
            project_id=document.project_id,
            full_text=document.extracted_text,
            db=db
        )
        
        return {
            "document_id": document_id,
            "chunks_added": result.total_chunks,
            "total_chunks": result.total_chunks,
            "total_tokens": result.total_tokens
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )


__all__ = ["router"]