"""
Academic Knowledge Spine models (STEP 1, additive).

Single-table hierarchy Document -> Unit -> Topic -> Subtopic -> Concept
plus a shared canonical-topic registry for alias resolution. No graph
database: parent_id self-reference + order_index carry the tree, and
source document order is authoritative (dense order_index per parent).

Tables are created by migration 009_knowledge_spine.py; this module only
declares the ORM mapping. Existing tables are untouched.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..engine import Base


class CanonicalTopic(Base):
    """Shared registry row: one canonical topic identity across projects.

    Writes are append-only: a second build may append unseen aliases but
    must never overwrite `name`. Alias collisions are reported, never
    resolved by stealing an alias from another canonical row.
    """

    __tablename__ = "canonical_topics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_key = Column(String(300), unique=True, nullable=False, index=True)
    name = Column(String(300), nullable=False)
    aliases = Column(JSONB, default=list, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    nodes = relationship("KnowledgeNode", back_populates="canonical_topic")

    def __repr__(self):
        return f"<CanonicalTopic(key={self.canonical_key})>"


class KnowledgeModel(Base):
    """Version header for one persisted academic spine.

    The node tree remains the queryable source of truth.  This small header
    supplies atomic project/version identity without duplicating node data;
    ``source_fingerprint`` makes an unchanged rebuild a no-op and status is
    append-only (``building`` -> ``ready``/``failed``/``superseded``).
    """

    __tablename__ = "knowledge_models"
    __table_args__ = (
        UniqueConstraint("project_id", "version", name="uq_knowledge_models_project_version"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    version = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, default="building")
    source_fingerprint = Column(String(64), nullable=False, index=True)
    node_count = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def __repr__(self):
        return (
            f"<KnowledgeModel(project_id={self.project_id}, "
            f"version={self.version}, status={self.status})>"
        )


class KnowledgeNode(Base):
    """One spine node: document root, unit, topic, subtopic, or concept.

    `node_metadata` maps onto the `metadata` JSONB column (`metadata` is a
    reserved attribute name on SQLAlchemy 2.0 declarative classes).
    `chunks` holds the cited chunk-id list backing this node; `pages`
    holds the source page numbers. Both are plain lists, never invented.
    """

    __tablename__ = "knowledge_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    project_id = Column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    document_id = Column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Version is nullable for rows written by the pre-versioning spine; the
    # production builder always supplies it for new model versions.
    model_version = Column(Integer, nullable=True, index=True)
    parent_id = Column(
        UUID(as_uuid=True), ForeignKey("knowledge_nodes.id", ondelete="CASCADE"),
        nullable=True, index=True,
    )

    # Node type as plain VARCHAR (document|unit|topic|subtopic|concept);
    # validated at the Pydantic layer so no PG enum migration is needed.
    type = Column(String(16), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    order_index = Column(Integer, nullable=False)

    pages = Column(JSONB, default=list, nullable=False)
    chunks = Column(JSONB, default=list, nullable=False)
    aliases = Column(JSONB, default=list, nullable=False)
    confidence = Column(Float, nullable=True)
    node_metadata = Column("metadata", JSONB, default=dict, nullable=False)

    canonical_topic_id = Column(
        UUID(as_uuid=True), ForeignKey("canonical_topics.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    canonical_topic = relationship("CanonicalTopic", back_populates="nodes")

    def __repr__(self):
        return (
            f"<KnowledgeNode(type={self.type}, title={self.title!r}, "
            f"order_index={self.order_index})>"
        )
