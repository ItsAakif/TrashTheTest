"""Document page model (T201/T202).

Source of truth for per-page provenance: method + confidence recorded by
the tiered extractor. documents.extracted_text is kept for backward
compatibility only.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Text, Float, Boolean, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from ..engine import Base


class DocumentPage(Base):
    """One extracted page: text + how it was obtained."""

    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_document_pages_doc_page"),)

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=True, default="")
    char_count = Column(Integer, nullable=True, default=0)
    # native | layout | vision:<provider> | text | docx | ocr_unavailable | failed
    method = Column(String(64), nullable=False, default="native")
    confidence = Column(Float, nullable=True, default=0.0)
    parser_version = Column(String(64), nullable=True)
    image_stored = Column(Boolean, nullable=False, default=False)
    bbox_summary = Column(JSONB, nullable=True)
    failure_reason = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="pages")

    def __repr__(self):
        return f"<DocumentPage(doc={self.document_id} page={self.page_number} method={self.method})>"


__all__ = ["DocumentPage"]
