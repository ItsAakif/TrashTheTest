"""
Document Model

Represents an uploaded document within a project.
"""

import uuid
from datetime import datetime
from sqlalchemy import BigInteger, Column, String, Integer, Text, Boolean, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum

from ..engine import Base


class DocumentStatus(enum.Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(enum.Enum):
    """Supported file types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"  # png, jpg, jpeg


class DocumentSourceType(str, enum.Enum):
    """Academic role of an uploaded source document.

    This is intentionally separate from FileType: the same PDF format can be
    a textbook, syllabus, question paper, question bank, or reference.
    """

    STUDY_MATERIAL = "study_material"
    SYLLABUS = "syllabus"
    QUESTION_PAPER = "question_paper"
    QUESTION_BANK = "question_bank"
    REFERENCE = "reference"


class Document(Base):
    """
    Document entity - represents an uploaded file within a project.
    """
    __tablename__ = "documents"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to project
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False, index=True)
    
    # File information
    file_name = Column(String(500), nullable=False)
    # values_callable persists lowercase VALUES to match the native PG enums
    # created by migrations (NAME binding fails on Postgres).
    file_type = Column(
        SQLEnum(FileType, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        nullable=False
    )
    # Academic source role is independent of the file extension.  Nullable at
    # the storage level during rolling upgrades; application reads normalize
    # legacy NULL values to study_material.
    source_type = Column(
        SQLEnum(
            DocumentSourceType,
            native_enum=False,
            values_callable=lambda e: [m.value for m in e],
        ),
        nullable=True,
        default=DocumentSourceType.STUDY_MATERIAL,
        index=True,
    )
    file_path = Column(String(1000), nullable=False)  # Storage reference (legacy name)
    storage_key = Column(String(1000), nullable=True)  # Opaque backend key (== file_path going forward)

    # File metadata (T110/T111): original name is a DB string only; the
    # stored name is server-generated inside the storage backend.
    mime_type = Column(String(255), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    checksum = Column(String(64), nullable=True, index=True)  # sha256 hex
    
    # Document metadata
    page_count = Column(Integer, nullable=True)
    
    # Extracted text - for large docs, use incremental storage
    # For very large documents, consider separate table or chunk-based storage
    extracted_text = Column(Text, nullable=True)  # Can be large, but manageable
    
    # Processing status
    processing_status = Column(
        SQLEnum(DocumentStatus, native_enum=False, values_callable=lambda e: [m.value for m in e]),
        default=DocumentStatus.PENDING,
        nullable=False
    )
    processing_error = Column(Text, nullable=True)
    is_ocr_required = Column(Boolean, default=False, nullable=False)

    # Pipeline provenance (Phase 2 fills these; Phase 1 writes initial values).
    extraction_status = Column(String(32), nullable=True)  # pending|running|done|failed
    extraction_method = Column(String(64), nullable=True)  # native|layout|vision|text
    ocr_status = Column(String(32), nullable=True)  # not_required|pending|done|failed|unavailable
    embedding_status = Column(String(32), nullable=True)  # pending|running|done|failed

    # Phase 2 provenance (T201 expand step; nullable so pre-migration rows boot).
    page_count_confirmed = Column(Integer, nullable=True)
    extraction_method_summary = Column(JSONB, nullable=True)  # {"native": 8, "vision:ppocr-tiny-ort": 2}
    content_hash = Column(String(64), nullable=True, index=True)  # sha256 of joined page text
    indexing_state = Column(String(32), nullable=True, default="none")  # none|indexing|indexed|partial|failed
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    pages = relationship("DocumentPage", back_populates="document", cascade="all, delete-orphan")
    # Academic spine (STEP 1, additive): document-scoped knowledge nodes.
    knowledge_nodes = relationship("KnowledgeNode", backref="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, file_name={self.file_name}, status={self.processing_status.value})>"