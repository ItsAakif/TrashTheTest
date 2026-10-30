"""
Text Extraction Service

Handles document extraction as an async job, with progress tracking
and integration with the job system.
"""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from .extractor import DocumentExtractor, ExtractorConfig, document_extractor
from .types import DocumentExtractionResult, ExtractionProgress
from database.models import DocumentStatus, JobStatus
from database import repositories


class ExtractionService:
    """
    Service for running document extraction as a background job.
    
    Handles:
    - Job status updates
    - Progress tracking
    - Result storage
    - Error handling
    """
    
    def __init__(self, extractor: Optional[DocumentExtractor] = None):
        self.extractor = extractor or document_extractor
    
    async def process_document(
        self,
        document_id: uuid.UUID,
        project_id: uuid.UUID,
        file_path: str,
        file_type: str,
        db: AsyncSession,
        job_id: Optional[uuid.UUID] = None,
        progress_callback: Optional[callable] = None
    ) -> DocumentExtractionResult:
        """
        Process a document for extraction.
        
        Args:
            document_id: Document ID
            project_id: Project ID
            file_path: Storage path to file
            file_type: File type
            db: Database session
            job_id: Optional job ID for progress updates
            progress_callback: Optional callback for progress
            
        Returns:
            DocumentExtractionResult with extracted content
        """
        try:
            # Update document status to processing
            await repositories.update_document_status(
                db=db,
                document_id=document_id,
                status=DocumentStatus.PROCESSING
            )
            
            # Update job progress if job exists
            if job_id:
                await self._update_job_progress(
                    db, job_id, 10, "Starting extraction..."
                )
            
            # Define progress callback for the extractor
            async def extraction_progress(progress: ExtractionProgress):
                # Update document with progress
                if job_id:
                    await self._update_job_progress(
                        db, job_id, 
                        int(progress.progress_percent),
                        progress.message or f"Processing page {progress.current_page}"
                    )
                
                if progress_callback:
                    await progress_callback(progress)
            
            # Run extraction
            result = await self.extractor.extract(
                document_id=document_id,
                file_path=file_path,
                file_type=file_type,
                db=db,
                progress_callback=extraction_progress
            )
            
            # Update job progress for completion
            if job_id:
                await self._update_job_progress(
                    db, job_id, 90, "Saving results..."
                )
            
            # Update document with extracted text
            await repositories.update_document_status(
                db=db,
                document_id=document_id,
                status=DocumentStatus.COMPLETED,
                extracted_text=result.full_text
            )
            
            # Update document page count
            document = await repositories.get_document_by_id(db, document_id)
            if document:
                document.page_count = result.total_pages
                await db.flush()
            
            # Final job update
            if job_id:
                await self._update_job_progress(
                    db, job_id, 100, "Extraction complete",
                    output_data={
                        "total_pages": result.total_pages,
                        "topics": result.topics,
                        "is_ocr_required": result.is_ocr_required,
                        "method": result.method
                    }
                )
            
            await db.commit()
            
            return result
            
        except Exception as e:
            # Update document status to failed
            await repositories.update_document_status(
                db=db,
                document_id=document_id,
                status=DocumentStatus.FAILED,
                error=str(e)
            )
            
            # Update job to failed
            if job_id:
                await repositories.update_job_status(
                    db=db,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error_message=str(e)
                )
            
            await db.commit()
            
            raise
    
    async def _update_job_progress(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        progress: int,
        message: str,
        output_data: Optional[dict] = None
    ):
        """Update job progress"""
        await repositories.update_job_status(
            db=db,
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=progress,
            output_data=output_data
        )


# Global extraction service
extraction_service = ExtractionService()


__all__ = [
    "ExtractionService",
    "extraction_service",
]