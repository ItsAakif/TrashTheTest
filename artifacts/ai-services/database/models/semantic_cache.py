"""Semantic cache entries (S3).

Postgres is the source of truth and the semantic index (pgvector). Redis is
an optional exact-key hot path. Entries are scoped by an identity hash so a
cached answer can never cross user/project/document-version/model/prompt
boundaries.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB

from ..engine import Base


class SemanticCacheEntry(Base):
    __tablename__ = "semantic_cache_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identity_key = Column(String(64), nullable=False, index=True)
    layer = Column(String(16), nullable=False, index=True)  # retrieval|answer
    query_hash = Column(String(64), nullable=False)
    query_text = Column(Text, nullable=True)
    response = Column(JSONB, nullable=False)
    citations = Column(JSONB, nullable=True)
    document_ids = Column(JSONB, nullable=True)
    doc_versions = Column(JSONB, nullable=True)
    project_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    embedding_model = Column(String(128), nullable=True)
    llm_model = Column(String(128), nullable=True)
    prompt_version = Column(String(64), nullable=True)
    retrieval_config_hash = Column(String(64), nullable=True)
    response_format = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)
    hits = Column(Integer, default=0, nullable=False)
    last_hit_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("identity_key", "layer", "query_hash",
                         name="uq_semantic_cache_identity_layer_query"),
    )
