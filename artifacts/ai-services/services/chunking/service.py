"""
Semantic Chunking Service

Service layer for running document chunking as a background job.
Handles integration with the job system, progress tracking, and incremental storage.
"""

import uuid
import time
from datetime import datetime
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession

from .chunker import SemanticChunker, semantic_chunker
from .types import (
    DocumentChunkingInput,
    ChunkingProgress,
    ChunkingResult,
    ChunkingJobInput,
    ChunkingConfig,
)
from database.models import DocumentStatus, JobStatus
from database import repositories


class ChunkingService:
    """
    Service for running document chunking as a background job.
    
    Handles:
    - Job status updates
    - Progress tracking with callbacks
    - Incremental chunk storage
    - Resume capability for interrupted processing
    - Error handling
    """
    
    def __init__(self, chunker: Optional[SemanticChunker] = None):
        self.chunker = chunker or semantic_chunker
    
    async def _update_job_progress(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        progress: int,
        message: str
    ):
        """Update job progress and status (merge-safe: _stages_done survives)."""
        await repositories.update_job_status(
            db=db,
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=progress,
            output_data={"message": message}
        )
    
    async def process_document(
        self,
        document_id: uuid.UUID,
        project_id: uuid.UUID,
        full_text: str,
        structural_elements: List[dict],
        db: AsyncSession,
        job_id: Optional[uuid.UUID] = None,
        config: Optional[ChunkingConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> ChunkingResult:
        """
        Process a document for chunking.
        
        Args:
            document_id: Document ID
            project_id: Project ID
            full_text: Extracted text from Module 3
            structural_elements: Structural metadata from extraction
            db: Database session
            job_id: Optional job ID for progress updates
            config: Chunking configuration
            progress_callback: Optional callback for progress
            
        Returns:
            ChunkingResult with all chunks
        """
        # Update chunker config if provided
        if config:
            self.chunker = SemanticChunker(config)
        
        try:
            # Update job progress
            if job_id:
                await self._update_job_progress(db, job_id, 10, "Starting chunking...")
            
            # Prepare input
            input_data = DocumentChunkingInput(
                document_id=str(document_id),
                project_id=str(project_id),
                full_text=full_text,
                structural_elements=structural_elements
            )
            
            # Determine if we should use streaming based on text size
            # Use streaming for documents > 500KB
            use_streaming = len(full_text) > 500 * 1024
            
            if use_streaming:
                # Use streaming approach for large documents
                result = await self._process_streaming(
                    input_data=input_data,
                    db=db,
                    job_id=job_id,
                    progress_callback=progress_callback
                )
            else:
                # Use synchronous approach for smaller documents
                result = await self._process_sync(
                    input_data=input_data,
                    db=db,
                    job_id=job_id,
                    progress_callback=progress_callback
                )
            
            # Update job progress for completion
            if job_id:
                await self._update_job_progress(
                    db, job_id, 100, f"Completed: {result.total_chunks} chunks created"
                )
            
            return result
            
        except Exception as e:
            # Handle error
            if job_id:
                await repositories.update_job_status(
                    db=db,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error_message=str(e)
                )
            raise
    
    async def _process_sync(
        self,
        input_data: DocumentChunkingInput,
        db: AsyncSession,
        job_id: Optional[uuid.UUID],
        progress_callback: Optional[callable]
    ) -> ChunkingResult:
        """Synchronous processing for smaller documents"""
        
        # Update progress
        if job_id:
            await self._update_job_progress(db, job_id, 20, "Chunking document...")
        
        # Run chunking
        result = self.chunker.chunk_document(input_data)
        
        # Update progress
        if job_id:
            await self._update_job_progress(db, job_id, 50, "Storing chunks...")
        
        # Store chunks in database
        await self._store_chunks(
            db=db,
            document_id=input_data.document_id,
            project_id=input_data.project_id,
            chunks=result.chunks
        )
        
        # Update progress
        if job_id:
            await self._update_job_progress(db, job_id, 90, "Finalizing...")
        
        return result
    
    async def _process_streaming(
        self,
        input_data: DocumentChunkingInput,
        db: AsyncSession,
        job_id: Optional[uuid.UUID],
        progress_callback: Optional[callable]
    ) -> ChunkingResult:
        """Streaming processing for large documents with incremental storage"""
        
        all_chunks = []
        total_tokens = 0
        start_time = time.time()
        
        # Define progress callback for the chunker
        async def chunking_progress(progress: ChunkingProgress):
            # Update job progress
            if job_id:
                await self._update_job_progress(
                    db, job_id,
                    int(progress.progress_percent * 0.7),  # 70% for chunking
                    progress.message or f"Processing batch {progress.current_batch}"
                )
            
            if progress_callback:
                await progress_callback(progress)
        
        # Stream chunks in batches
        batch_count = 0
        async for batch_chunks, progress in self.chunker.chunk_document_streaming(
            input_data,
            progress_callback=chunking_progress
        ):
            batch_count += 1
            
            # Store chunks incrementally
            await self._store_chunks(
                db=db,
                document_id=input_data.document_id,
                project_id=input_data.project_id,
                chunks=batch_chunks
            )
            
            # Accumulate results
            all_chunks.extend(batch_chunks)
            total_tokens += sum(c.token_count for c in batch_chunks)
            
            # Update job progress after each batch
            if job_id:
                await self._update_job_progress(
                    db, job_id,
                    70 + int((batch_count / progress.total_batches) * 20),  # 70-90%
                    f"Stored {len(all_chunks)} chunks..."
                )
        
        processing_time = time.time() - start_time
        
        return ChunkingResult(
            document_id=input_data.document_id,
            project_id=input_data.project_id,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            total_tokens=total_tokens,
            processing_time=processing_time,
            is_complete=True
        )
    
    async def _store_chunks(
        self,
        db: AsyncSession,
        document_id: str,
        project_id: str,
        chunks: List
    ):
        """Store chunks idempotently: upsert on (document_id, content_hash)."""

        # Prepare chunk data for idempotent upsert
        chunks_data = []
        for chunk in chunks:
            meta = chunk.metadata.model_dump() if hasattr(chunk.metadata, "model_dump") else dict(chunk.metadata or {})
            chunks_data.append({
                "document_id": uuid.UUID(document_id),
                "content": chunk.content,
                "metadata": meta,
                "token_count": chunk.token_count,
                "content_hash": chunk.content_hash,
                "page_start": meta.get("page_start") or meta.get("page_number"),
                "page_end": meta.get("page_end") or meta.get("page_number"),
                "section_path": meta.get("section_path") or ([meta["section"]] if meta.get("section") else []),
                "heading": meta.get("heading"),
                "position": meta.get("position"),
            })

        # Insert batch (idempotent)
        try:
            await repositories.upsert_chunks_batch(db, chunks_data)
        except Exception:
            await repositories.create_chunks_batch(db, chunks_data)

        # T405: flush only. The worker owns the commit at explicit stage
        # boundaries; committing here would expire the shared Document
        # instance mid-pipeline.
        await db.flush()
    
    async def resume_chunking(
        self,
        document_id: uuid.UUID,
        project_id: uuid.UUID,
        full_text: str,
        db: AsyncSession,
        job_id: Optional[uuid.UUID] = None,
        progress_callback: Optional[callable] = None
    ) -> ChunkingResult:
        """
        Resume chunking from last processed position.
        
        Uses existing chunks to determine where to continue.
        """
        
        # Get existing chunks for this document
        existing_chunks = await repositories.get_chunks_by_document(db, document_id)
        
        if not existing_chunks:
            # No existing chunks, start fresh
            return await self.process_document(
                document_id=document_id,
                project_id=project_id,
                full_text=full_text,
                structural_elements=[],
                db=db,
                job_id=job_id,
                progress_callback=progress_callback
            )
        
        # Find last position
        last_position = max(
            (c.chunk_metadata.get("position", 0) if c.chunk_metadata else 0)
            for c in existing_chunks
        )
        
        # Prepare input
        input_data = DocumentChunkingInput(
            document_id=str(document_id),
            project_id=str(project_id),
            full_text=full_text,
            structural_elements=[]
        )
        
        # Continue from last position
        result = self.chunker.chunk_document(input_data, start_position=last_position + 1)
        
        # Store new chunks
        await self._store_chunks(
            db=db,
            document_id=str(document_id),
            project_id=str(project_id),
            chunks=result.chunks
        )
        
        return result


# Default service instance
chunking_service = ChunkingService()