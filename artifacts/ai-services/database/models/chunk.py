"""
Chunk Model

Represents a semantic chunk from a document with embedding for vector search.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Float, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, BYTEA
from sqlalchemy.orm import relationship

from ..engine import Base


class Chunk(Base):
    """
    Chunk entity - semantic text segment with embedding for RAG.
    
    The embedding field uses pgvector for similarity search.
    Dimension is 1024 to match Qwen3-Embedding-0.6B (default provider
    qwen3-embedding; BGE-small 384d is legacy only).
    """
    __tablename__ = "chunks"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Foreign key to document
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    
    # Content
    content = Column(Text, nullable=False)
    
    # Embedding - using BYTEA for pgvector compatible storage
    # Alternative: use pgvector.sqlalchemy.Vector(1024) with pgvector extension
    # (default model Qwen/Qwen3-Embedding-0.6B, 1024d; BGE-small 384d legacy)
    # We'll use a custom type or BYTEA for compatibility
    embedding = Column(BYTEA, nullable=True)  # 1024-dimensional vector (Qwen3 default)

    # Phase 2 expand columns (T201): legacy BYTEA above is untouched and kept
    # reversible; new vector/tsv columns are created by migration 004 on
    # Postgres. SQLite/test doubles ignore them (getattr-guarded writes).
    embedding_model = Column(String(128), nullable=True)
    embedding_dim = Column(Integer, nullable=True)
    content_hash = Column(String(64), nullable=True, index=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    section_path = Column(JSONB, nullable=True)
    heading = Column(Text, nullable=True)
    position = Column(Integer, nullable=True)
    
    # Metadata for chunk (page number, section, heading, position).
    # NOTE: mapped as `chunk_metadata` onto the `metadata` column because
    # `metadata` is a reserved attribute name on SQLAlchemy 2.0 declarative
    # classes (import of this module fails otherwise). DB column unchanged.
    chunk_metadata = Column("metadata", JSONB, default={}, nullable=True)

    # Token count for reference
    token_count = Column(Integer, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, document_id={self.document_id}, tokens={self.token_count})>"


# For pgvector with SQLAlchemy, use this alternative:
# Note: Requires pgvector extension to be enabled
"""
from pgvector.sqlalchemy import Vector

class ChunkWithVector(Base):
    __tablename__ = "chunks_vector"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)  # pgvector type (Qwen3 1024d default)
    metadata = Column(JSONB, default={}, nullable=True)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
"""