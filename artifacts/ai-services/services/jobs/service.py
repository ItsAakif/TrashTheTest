"""
Job Manager Service

Manages job lifecycle, progress tracking, and job operations.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Awaitable
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from database.models import Job, Job as JobModel, JobType, JobStatus
from database import repositories
from .types import (
    JobStage,
    JobStatus as JobStatusType,
    JobType as JobTypeType,
    STAGE_PROGRESS_RANGES,
    DEFAULT_MAX_RETRIES,
    PipelineState,
)

logger = logging.getLogger(__name__)


class JobProgressCallback:
    """Callback for progress updates"""
    
    def __init__(self, db: AsyncSession, job_id: uuid.UUID):
        self.db = db
        self.job_id = job_id
    
    async def __call__(
        self, 
        progress: int, 
        stage: JobStage, 
        message: str,
        output_data: Optional[Dict[str, Any]] = None
    ):
        """Update job progress"""
        await repositories.update_job_status(
            db=self.db,
            job_id=self.job_id,
            status=JobStatus.PROCESSING,
            progress=progress,
            output_data=output_data
        )
        await self.db.commit()
        logger.info(f"Job {self.job_id}: Stage {stage.value} - {message} ({progress}%)")


class JobManager:
    """
    Manages job lifecycle and operations.
    
    Responsibilities:
    - Create jobs with proper input data
    - Update job status and progress
    - Track pipeline stages
    - Handle retries on failure
    - Resume from last successful stage
    """
    
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        self.max_retries = max_retries
    
    async def create_job(
        self,
        db: AsyncSession,
        job_type: JobType,
        input_data: Optional[Dict[str, Any]] = None,
        project_id: Optional[uuid.UUID] = None,
        document_id: Optional[uuid.UUID] = None,
        check_duplicate: bool = True
    ) -> Job:
        """
        Create a new job with proper input data.
        
        Args:
            db: Database session
            job_type: Type of job to create
            input_data: Job input data
            project_id: Associated project ID
            document_id: Associated document ID
            check_duplicate: Whether to check for duplicate pending jobs
            
        Returns:
            Created Job instance
            
        Raises:
            ValueError: If duplicate job exists
        """
        # Check for duplicate pending jobs
        if check_duplicate and document_id:
            existing = await self._find_duplicate_job(db, job_type, document_id)
            if existing:
                raise ValueError(
                    f"Duplicate job exists: {existing.id} "
                    f"(status: {existing.status.value})"
                )
        
        # Enrich input data with metadata
        enriched_input = input_data or {}
        if project_id:
            enriched_input["project_id"] = str(project_id)
        if document_id:
            enriched_input["document_id"] = str(document_id)
        
        # Create the job (real FK columns + input_data mirror for compat).
        job = await repositories.create_job(
            db=db,
            job_type=job_type.value,
            input_data=enriched_input,
            project_id=project_id,
            document_id=document_id,
        )
        await db.commit()
        
        logger.info(f"Created job {job.id} of type {job_type.value}")
        return job
    
    async def update_progress(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        progress: int,
        stage: JobStage,
        message: str,
        output_data: Optional[Dict[str, Any]] = None
    ) -> Job:
        """
        Update job progress with stage information.
        
        Args:
            db: Database session
            job_id: Job ID to update
            progress: Progress percentage (0-100)
            stage: Current pipeline stage
            message: Status message
            output_data: Optional output data
            
        Returns:
            Updated Job instance
        """
        # Enrich output data with stage info
        enriched_output = output_data or {}
        enriched_output["_stage"] = stage.value
        enriched_output["_stage_message"] = message
        
        job = await repositories.update_job_status(
            db=db,
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=progress,
            output_data=enriched_output
        )
        await db.commit()
        
        return job
    
    async def start_processing(
        self,
        db: AsyncSession,
        job_id: uuid.UUID
    ) -> Job:
        """
        Mark job as processing.
        
        Args:
            db: Database session
            job_id: Job ID
            
        Returns:
            Updated Job instance
        """
        job = await repositories.update_job_status(
            db=db,
            job_id=job_id,
            status=JobStatus.PROCESSING,
            progress=0
        )
        await db.commit()
        
        logger.info(f"Job {job_id} started processing")
        return job
    
    async def complete_job(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        output_data: Optional[Dict[str, Any]] = None
    ) -> Job:
        """
        Mark job as completed.
        
        Args:
            db: Database session
            job_id: Job ID
            output_data: Final job output
            
        Returns:
            Completed Job instance
        """
        job = await repositories.update_job_status(
            db=db,
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            output_data=output_data
        )
        await db.commit()
        
        logger.info(f"Job {job_id} completed successfully")
        return job
    
    async def fail_job(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        error: str,
        retry: bool = True
    ) -> Optional[Job]:
        """
        Mark job as failed, optionally retrying.
        
        Args:
            db: Database session
            job_id: Job ID
            error: Error message
            retry: Whether to attempt retry
            
        Returns:
            Job instance if not retrying, None if retry queued
        """
        job = await repositories.get_job_by_id(db, job_id)
        
        if not job:
            logger.error(f"Job {job_id} not found")
            return None
        
        # Check retry count
        retry_count = job.input_data.get("_retry_count", 0) if job.input_data else 0
        
        if retry and retry_count < self.max_retries:
            # Queue for retry
            new_retry_count = retry_count + 1
            input_data = job.input_data or {}
            input_data["_retry_count"] = new_retry_count
            input_data["_last_error"] = error
            
            await repositories.update_job_status(
                db=db,
                job_id=job_id,
                status=JobStatus.PENDING,
                progress=0,
                input_data=input_data,
                error_message=f"Retry {new_retry_count}/{self.max_retries}: {error}"
            )
            await db.commit()
            
            logger.warning(
                f"Job {job_id} failed, retry {new_retry_count}/{self.max_retries}"
            )
            return None
        else:
            # Permanent failure
            job = await repositories.update_job_status(
                db=db,
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message=error
            )
            await db.commit()
            
            logger.error(f"Job {job_id} permanently failed: {error}")
            return job
    
    async def cancel_job(
        self,
        db: AsyncSession,
        job_id: uuid.UUID
    ) -> Job:
        """
        Cancel a pending or processing job.
        
        Args:
            db: Database session
            job_id: Job ID
            
        Returns:
            Updated Job instance
        """
        job = await repositories.get_job_by_id(db, job_id)
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if job.status == JobStatus.COMPLETED:
            raise ValueError("Cannot cancel completed job")
        
        if job.status == JobStatus.FAILED:
            raise ValueError("Cannot cancel failed job")

        if job.status == JobStatus.CANCELLED:
            raise ValueError("Job is already cancelled")
        
        job = await repositories.update_job_status(
            db=db,
            job_id=job_id,
            status=JobStatus.CANCELLED,
            error_message="Cancelled by user"
        )
        await db.commit()
        
        logger.info(f"Job {job_id} cancelled")
        return job
    
    async def get_job(
        self,
        db: AsyncSession,
        job_id: uuid.UUID
    ) -> Optional[Job]:
        """Get job by ID"""
        return await repositories.get_job_by_id(db, job_id)
    
    async def get_jobs_for_project(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Job]:
        """
        Get jobs for a project.
        
        Args:
            db: Database session
            project_id: Project ID
            status: Optional status filter
            limit: Max results
            offset: Result offset
            
        Returns:
            List of jobs
        """
        from sqlalchemy import select, or_
        from database.models import Job

        # Real column first; input_data mirror as OR-fallback for pre-T111 rows.
        query = select(Job).where(
            or_(
                Job.project_id == project_id,
                Job.input_data["project_id"].astext == str(project_id),
            )
        )
        
        if status:
            query = query.where(Job.status == status)
        
        query = query.order_by(Job.created_at.desc()).limit(limit).offset(offset)
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def get_pending_jobs(
        self,
        db: AsyncSession,
        limit: int = 100
    ) -> List[Job]:
        """Get pending jobs for processing"""
        return await repositories.get_pending_jobs(db, limit)
    
    async def _find_duplicate_job(
        self,
        db: AsyncSession,
        job_type: JobType,
        document_id: uuid.UUID
    ) -> Optional[Job]:
        """Find duplicate pending job for same document"""
        from sqlalchemy import select, and_, or_
        from database.models import Job

        result = await db.execute(
            select(Job).where(
                and_(
                    Job.job_type == job_type,
                    or_(
                        Job.document_id == document_id,
                        Job.input_data["document_id"].astext == str(document_id),
                    ),
                    Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
                )
            )
        )
        return result.scalars().first()
    
    def calculate_stage_progress(
        self,
        stage: JobStage,
        stage_progress: float
    ) -> int:
        """
        Calculate overall progress based on stage and stage-specific progress.
        
        Args:
            stage: Current pipeline stage
            stage_progress: Progress within the stage (0.0 - 1.0)
            
        Returns:
            Overall progress percentage (0-100)
        """
        start, end = STAGE_PROGRESS_RANGES.get(stage, (0, 100))
        return int(start + (end - start) * stage_progress)


# Global job manager instance
job_manager = JobManager()


__all__ = [
    "JobManager",
    "job_manager",
    "JobProgressCallback",
]