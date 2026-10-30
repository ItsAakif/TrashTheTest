"""
Job System Types

Type definitions for the job management system.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Any, Dict, Union
from enum import Enum
from pydantic import BaseModel, Field


class JobType(str, Enum):
    """Type of background job"""
    DOCUMENT_PROCESSING = "document_processing"
    GENERATION = "generation"
    OCR = "ocr"
    EMBEDDING = "embedding"
    TEXT_EXTRACTION = "text_extraction"
    CHUNKING = "chunking"
    WEBSITE_GENERATION = "website_generation"
    KNOWLEDGE_MODEL = "knowledge_model"
    NOTES_HIERARCHICAL = "notes_hierarchical"
    ACADEMIC_GENERATION = "academic_generation"
    # Additive structure-integrity re-chunk job (ADR-008).
    STRUCTURE_REPROCESS = "structure_reprocess"


class JobStatus(str, Enum):
    """Job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStage(str, Enum):
    """Document-processing stage labels (legacy vocabulary)."""
    INIT = "init"
    EXTRACTION = "extraction"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    FINALIZATION = "finalization"


# Academic jobs use a separate stage ledger (model -> ... -> pdf) so they do
# not consume or rewrite the document-processing budget. Job API models keep
# accepting those labels as strings; otherwise a valid academic progress
# update would make the shared JobResponse validation fail with a 500.
JobStageValue = Union[JobStage, str]


# Progress ranges for each stage (as per requirements)
STAGE_PROGRESS_RANGES = {
    JobStage.INIT: (0, 5),
    JobStage.EXTRACTION: (5, 30),
    JobStage.CHUNKING: (30, 60),
    JobStage.EMBEDDING: (60, 90),
    JobStage.FINALIZATION: (90, 100),
}


class JobCreate(BaseModel):
    """Request model for creating a job"""
    job_type: JobType
    input_data: Optional[Dict[str, Any]] = None
    project_id: Optional[uuid.UUID] = None
    document_id: Optional[uuid.UUID] = None


class JobUpdate(BaseModel):
    """Request model for updating a job"""
    status: Optional[JobStatus] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class JobProgress(BaseModel):
    """Detailed job progress information"""
    job_id: uuid.UUID
    status: JobStatus
    progress: int
    stage: Optional[JobStageValue] = None
    stage_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    """Full job response"""
    id: uuid.UUID
    job_type: JobType
    status: JobStatus
    progress: int
    stage: Optional[str] = None
    stage_message: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Response for listing jobs"""
    jobs: List[JobResponse]
    total: int
    page: int = 1
    page_size: int = 20


class PipelineState(BaseModel):
    """State tracking for pipeline execution"""
    current_stage: JobStage = JobStage.INIT
    extraction_completed: bool = False
    chunking_completed: bool = False
    embedding_completed: bool = False
    last_successful_stage: Optional[JobStage] = None
    stage_outputs: Dict[str, Any] = {}


class JobEvent(BaseModel):
    """Server-Sent Events job update"""
    event: str = "job_update"
    job_id: uuid.UUID
    status: JobStatus
    progress: int
    stage: Optional[str] = None
    message: Optional[str] = None


# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 5