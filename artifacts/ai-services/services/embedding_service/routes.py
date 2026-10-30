"""
Embedding Generation API Routes

REST API endpoints for embedding generation operations.
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from database.engine import get_db
from database import repositories
from database.models import Chunk, JobStatus

from .service import LocalEmbeddingService, embedding_service
from .types import EmbeddingProgress, EmbeddingResult, EmbeddingStats


router = APIRouter(prefix="/api/embedding", tags=["embedding"])


class EmbeddingRequest(BaseModel):
    """Request to start embedding generation for a document"""
    document_id: str
    project_id: str
    batch_size: Optional[int] = 32
    force_regenerate: bool = False


class EmbeddingResponse(BaseModel):
    """Response after starting embedding generation"""
    job_id: str
    status: str
    message: str


class EmbeddingStatusResponse(BaseModel):
    """Response for embedding status"""
    job_id: str
    status: str
    progress: int
    chunks_processed: int
    total_chunks: int
    cached_count: int
    message: Optional[str] = None


class EmbeddingStatsResponse(BaseModel):
    """Response for embedding statistics"""
    document_id: str
    total_chunks: int
    chunks_with_embeddings: int
    chunks_without_embeddings: int
    average_tokens: float


@router.post("/generate", response_model=EmbeddingResponse)
async def start_embedding_generation(
    request: EmbeddingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Start embedding generation for a document.
    
    This endpoint triggers the embedding job and returns immediately.
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
        
        # Check document has chunks
        chunks = await repositories.get_chunks_by_document(db, document_id)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has no chunks. Run chunking first."
            )
        
        # Create job for embedding
        job = await repositories.create_job(
            db=db,
            job_type="embedding",
            input_data={
                "document_id": request.document_id,
                "project_id": request.project_id,
                "batch_size": request.batch_size,
                "force_regenerate": request.force_regenerate
            }
        )
        
        # Start embedding in background (fire and forget for now)
        # In production, use a proper task queue
        import asyncio
        
        async def run_embedding():
            try:
                # Run embedding service
                await embedding_service.generate_embeddings(
                    document_id=document_id,
                    project_id=project_id,
                    db=db,
                    job_id=job.id,
                    force_regenerate=request.force_regenerate
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
        asyncio.create_task(run_embedding())
        
        return EmbeddingResponse(
            job_id=str(job.id),
            status="started",
            message="Embedding job started"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start embedding: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=EmbeddingStatusResponse)
async def get_embedding_status(
    job_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get the status of an embedding job"""
    try:
        job_uuid = uuid.UUID(job_id)
        job = await repositories.get_job_by_id(db, job_uuid)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Get progress info from output data
        chunks_processed = 0
        cached_count = 0
        message = None
        if job.output_data:
            chunks_processed = job.output_data.get("chunks_processed", 0)
            cached_count = job.output_data.get("cached_count", 0)
            message = job.output_data.get("message")
        
        return EmbeddingStatusResponse(
            job_id=job_id,
            status=job.status.value,
            progress=job.progress,
            chunks_processed=chunks_processed,
            total_chunks=0,  # Would need to fetch from document
            cached_count=cached_count,
            message=message
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format"
        )


@router.get("/stats/{document_id}", response_model=EmbeddingStatsResponse)
async def get_embedding_stats(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get embedding statistics for a document"""
    try:
        doc_uuid = uuid.UUID(document_id)
        
        # Get total chunks
        result = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == doc_uuid)
        )
        total_chunks = result.scalar() or 0
        
        # Get chunks with embeddings
        result = await db.execute(
            select(func.count(Chunk.id)).where(
                Chunk.document_id == doc_uuid,
                Chunk.embedding.isnot(None)
            )
        )
        chunks_with_embeddings = result.scalar() or 0
        
        # Get average tokens
        result = await db.execute(
            select(func.avg(Chunk.token_count)).where(Chunk.document_id == doc_uuid)
        )
        avg_tokens = result.scalar() or 0
        
        return EmbeddingStatsResponse(
            document_id=document_id,
            total_chunks=total_chunks,
            chunks_with_embeddings=chunks_with_embeddings,
            chunks_without_embeddings=total_chunks - chunks_with_embeddings,
            average_tokens=float(avg_tokens)
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )


@router.post("/generate/incremental/{document_id}")
async def generate_incremental_embeddings(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Generate embeddings only for chunks that don't have them.
    
    More efficient for adding new chunks to existing documents.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
        
        # Check document exists
        document = await repositories.get_document_by_id(db, doc_uuid)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Run incremental embedding
        result = await embedding_service.generate_embeddings_incremental(
            document_id=doc_uuid,
            project_id=document.project_id,
            db=db
        )
        
        return {
            "document_id": document_id,
            "chunks_embedded": result.embedded_chunks,
            "processing_time": result.processing_time
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )


@router.post("/regenerate/{document_id}")
async def regenerate_embeddings(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Force regeneration of all embeddings for a document.
    
    Clears existing embeddings and regenerates them.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
        
        # Check document exists
        document = await repositories.get_document_by_id(db, doc_uuid)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Clear existing embeddings
        from sqlalchemy import update
        await db.execute(
            update(Chunk)
            .where(Chunk.document_id == doc_uuid)
            .values(embedding=None)
        )
        await db.commit()
        
        # Generate new embeddings
        result = await embedding_service.generate_embeddings(
            document_id=doc_uuid,
            project_id=document.project_id,
            db=db,
            force_regenerate=True
        )
        
        return {
            "document_id": document_id,
            "chunks_embedded": result.embedded_chunks,
            "processing_time": result.processing_time
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )


__all__ = ["router"]