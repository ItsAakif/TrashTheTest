"""Retrieval traces (S4): persisted, redacted retrieval observability."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB

from ..engine import Base


class RetrievalTrace(Base):
    __tablename__ = "retrieval_traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(64), nullable=True, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    project_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    document_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    query_text = Column(Text, nullable=True)
    retrieval_mode = Column(String(16), nullable=True)
    rerank = Column(Integer, nullable=True)
    chunk_ids = Column(JSONB, nullable=True)
    scores = Column(JSONB, nullable=True)
    cache_layer = Column(String(16), nullable=True)
    cache_hit = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    rerank_latency_ms = Column(Float, nullable=True)
    context_tokens_estimate = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
