"""Job Model

Represents async background jobs for document processing and content generation.
Queue columns (T111): document_id/project_id links, stage, attempts, priority,
run_after, locked_by/locked_at for SKIP LOCKED claiming, idempotency key.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
import enum

from ..engine import Base


class JobType(enum.Enum):
    """Type of background job"""
    DOCUMENT_PROCESSING = "document_processing"
    GENERATION = "generation"
    OCR = "ocr"
    EMBEDDING = "embedding"
    TEXT_EXTRACTION = "text_extraction"
    CHUNKING = "chunking"
    WEBSITE_GENERATION = "website_generation"
    # Academic production jobs (expand-only labels).
    KNOWLEDGE_MODEL = "knowledge_model"
    NOTES_HIERARCHICAL = "notes_hierarchical"
    ACADEMIC_GENERATION = "academic_generation"
    # Additive structure-integrity re-chunk job (ADR-008).
    STRUCTURE_REPROCESS = "structure_reprocess"


class JobStatus(enum.Enum):
    """Job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(Base):
    """
    Job entity - tracks async background tasks.
    """
    __tablename__ = "jobs"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Job type (values_callable persists lowercase VALUES to match the
    # native PG enums created by migrations; NAME binding fails on Postgres).
    job_type = Column(
        SQLEnum(JobType, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        index=True
    )

    # Links (nullable for pre-T111 rows / non-document jobs)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True, index=True)

    # Status and progress
    status = Column(
        SQLEnum(JobStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=JobStatus.PENDING,
        nullable=False,
        index=True
    )
    progress = Column(Integer, default=0, nullable=False)  # 0-100

    # Pipeline stage written by the worker that performs it (T111)
    stage = Column(String(64), nullable=True)
    stage_message = Column(Text, nullable=True)

    # Attempts / queue ordering
    attempts = Column(Integer, default=0, nullable=False)
    max_attempts = Column(Integer, default=3, nullable=False)
    priority = Column(Integer, default=0, nullable=False)
    run_after = Column(DateTime, nullable=True, index=True)

    # SKIP LOCKED claim bookkeeping
    locked_by = Column(String(255), nullable=True)
    locked_at = Column(DateTime, nullable=True, index=True)

    # Idempotency: client-supplied key scoped per project+type
    idempotency_key = Column(String(255), nullable=True, index=True)

    # Input/output data (JSONB for flexibility)
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Job(id={self.id}, type={self.job_type.value}, status={self.status.value}, progress={self.progress})>"

    @property
    def is_terminal(self) -> bool:
        """Check if job is in terminal state"""
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def is_pending(self) -> bool:
        """Check if job is pending"""
        return self.status == JobStatus.PENDING

    @property
    def is_processing(self) -> bool:
        """Check if job is processing"""
        return self.status == JobStatus.PROCESSING
