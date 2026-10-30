"""
Job System API Routes

API endpoints for job management:
- GET /api/jobs/:id - Job status
- GET /api/jobs/:id/progress - Detailed progress
- GET /api/projects/:id/jobs - List jobs for project
- POST /api/jobs/:id/cancel - Cancel job
- GET /api/jobs/:id/stream - SSE for real-time progress
"""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import json

from database import get_db, repositories
from database.models import Job, JobStatus
from database.models.project import Project
from database.models.user import User
from api_gateway.deps import require_project_access
from services.auth.deps import get_current_user
from .service import job_manager
from . import queue as job_queue
from .types import (
    JobResponse,
    JobProgress,
    JobListResponse,
    JobStage,
    JobEvent,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


async def _require_job_access(
    db: AsyncSession, job: Job, current_user: User
) -> Job:
    """Ownership for jobs: resolve the linked project, 404 for another user's."""
    # Prefer the real project_id column (T111); fall back to the input_data
    # mirror for pre-T111 rows that were created before the column existed.
    project_id = job.project_id
    if project_id is None and job.input_data:
        project_id = (job.input_data or {}).get("project_id")
    if not project_id:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        pid = project_id if isinstance(project_id, uuid.UUID) else uuid.UUID(str(project_id))
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="Job not found")
    project = await repositories.get_project_by_id(db, pid)
    if project is None or project.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get job status and details.

    Returns:
        Job with current status, progress, and output data
    """
    job = await job_manager.get_job(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await _require_job_access(db, job, current_user)

    # Stage info lives on real columns (output_data kept as compat mirror).
    stage = job.stage or (job.output_data or {}).get("_stage") if job.output_data else job.stage
    stage_message = job.stage_message or (job.output_data or {}).get("_stage_message") if job.output_data else job.stage_message

    return JobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        stage=stage,
        stage_message=stage_message,
        input_data=job.input_data,
        output_data=job.output_data,
        error_message=job.error_message,
        retry_count=job.attempts,
        max_retries=job.max_attempts,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("/{job_id}/progress", response_model=JobProgress)
async def get_job_progress(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed job progress with stage information.

    Returns:
        Progress info including current stage and message
    """
    job = await job_manager.get_job(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await _require_job_access(db, job, current_user)

    # Extract stage info
    stage = None
    stage_message = job.stage_message or (job.output_data or {}).get("_stage_message")

    stage_str = job.stage or (job.output_data or {}).get("_stage")
    if stage_str:
        try:
            stage = JobStage(stage_str)
        except ValueError:
            # Academic jobs intentionally use their own stage ledger
            # (model, outline, ..., pdf). Preserve that label rather than
            # dropping it or failing the shared progress response schema.
            stage = str(stage_str)
    
    return JobProgress(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        stage=stage,
        stage_message=stage_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("/{job_id}/stream")
async def stream_job_progress(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Stream job progress updates via Server-Sent Events.

    Real-time progress updates while job is processing.
    """
    job = await job_manager.get_job(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await _require_job_access(db, job, current_user)
    
    async def event_generator():
        """Generate SSE events for job progress"""
        last_progress = -1
        last_stage = None

        while True:
            # Get current job state
            current_job = await job_manager.get_job(db, job_id)

            if not current_job:
                break

            # Check if job changed
            stage_now = current_job.stage or (current_job.output_data or {}).get("_stage")
            if current_job.progress != last_progress or stage_now != last_stage:
                last_progress = current_job.progress
                last_stage = stage_now

                # Extract stage info
                stage = stage_now
                message = current_job.stage_message or (current_job.output_data or {}).get("_stage_message")

                event = JobEvent(
                    job_id=current_job.id,
                    status=current_job.status,
                    progress=current_job.progress,
                    stage=stage,
                    message=message
                )

                yield f"data: {event.model_dump_json()}\n\n"

            # Check if job is in terminal state
            if current_job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                break

            # Wait before next check
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a pending or processing job.

    Returns:
        Updated job with cancelled status
    """
    existing = await job_manager.get_job(db, job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Job not found")
    await _require_job_access(db, existing, current_user)
    try:
        job = await job_manager.cancel_job(db, job_id)
        return {"success": True, "job_id": str(job.id), "status": job.status.value}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/project/{project_id}", response_model=JobListResponse)
async def list_project_jobs(
    project_id: uuid.UUID,
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db)
):
    """
    List jobs for a project.
    
    Args:
        project_id: Project ID
        status: Optional status filter (pending, processing, completed, failed)
        page: Page number
        page_size: Results per page
        
    Returns:
        List of jobs with pagination info
    """
    # Convert status string to enum
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Must be one of: pending, processing, completed, failed"
            )
    
    # Calculate offset
    offset = (page - 1) * page_size
    
    # Get jobs
    jobs = await job_manager.get_jobs_for_project(
        db=db,
        project_id=project_id,
        status=status_filter,
        limit=page_size,
        offset=offset
    )
    
    # Convert to response format
    job_responses = []
    for job in jobs:
        stage = job.stage or (job.output_data or {}).get("_stage") if job.output_data else job.stage
        stage_message = job.stage_message or (job.output_data or {}).get("_stage_message") if job.output_data else job.stage_message

        job_responses.append(JobResponse(
            id=job.id,
            job_type=job.job_type,
            status=job.status,
            progress=job.progress,
            stage=stage,
            stage_message=stage_message,
            input_data=job.input_data,
            output_data=job.output_data,
            error_message=job.error_message,
            retry_count=job.attempts,
            max_retries=job.max_attempts,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        ))
    
    # Get total count (simplified - could be optimized)
    total = len(job_responses)
    
    return JobListResponse(
        jobs=job_responses,
        total=total,
        page=page,
        page_size=page_size,
    )


# Manual trigger endpoint: runs the job inline via the real queue claim path
# (same handler the worker process uses). For local/dev use; prod uses the
# separate worker process (python -m services.jobs.worker).
@router.post("/trigger/{job_id}")
async def trigger_job_processing(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Manually trigger job processing inline (dev/test path).
    """
    job = await job_manager.get_job(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    await _require_job_access(db, job, current_user)

    if job.status not in (JobStatus.PENDING,):
        raise HTTPException(
            status_code=400,
            detail=f"Job is not in pending state (current: {job.status.value})"
        )

    # Claim for this request, then run the registered worker handler inline.
    from services.jobs import worker as _worker

    claimed = await job_queue.claim(db, worker_id=f"trigger-{current_user.id}", limit=100)
    await db.commit()
    target = next((j for j in claimed if j.id == job_id), None)
    if target is None:
        raise HTTPException(status_code=409, detail="Job was claimed by another worker")

    handler = _worker._HANDLERS.get(
        target.job_type.value if hasattr(target.job_type, "value") else str(target.job_type)
    )
    if handler is None:
        raise HTTPException(status_code=501, detail=f"No handler for job type {target.job_type}")

    class _InlineCtx:
        def __init__(self, session):
            self.db = session

    try:
        await handler(_InlineCtx(db), target)
        await db.commit()
    except Exception as e:
        await db.rollback()
        await job_queue.fail(db, job_id, str(e), retry=True)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))

    result = await job_manager.get_job(db, job_id)
    return {
        "success": True,
        "job_id": str(result.id),
        "status": result.status.value,
        "progress": result.progress
    }


__all__ = ["router"]