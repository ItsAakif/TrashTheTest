"""Citation payload with resolvable provenance (T303).

Every retrieved chunk carries a citation: document_id, file_name,
page_number (from page_start/page_end), section_path/heading, method,
confidence, and the exact text span. The resolver turns a chunk id into
a citation by looking up the chunk and its document page rows - a stale
or deleted reference is detected rather than rendered.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    file_name: Optional[str] = None
    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: list = field(default_factory=list)
    heading: Optional[str] = None
    method: Optional[str] = None
    confidence: Optional[float] = None
    span: str = ""
    valid: bool = True
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "doc_id": self.document_id,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "page": self.page_number,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": self.section_path,
            "section": "/".join(self.section_path) if self.section_path else None,
            "heading": self.heading,
            "method": self.method,
            "confidence": self.confidence,
            "span": self.span,
            "valid": self.valid,
            "reason": self.reason,
        }


def build_citation_from_chunk(chunk: Any, document: Any = None) -> Citation:
    """Build a citation from an ORM chunk (+ optional document) offline-safe."""
    meta = getattr(chunk, "chunk_metadata", None) or {}
    page_start = getattr(chunk, "page_start", None)
    page_end = getattr(chunk, "page_end", None)
    page_number = page_start if page_start is not None else meta.get("page")
    section_path = getattr(chunk, "section_path", None) or meta.get("section_path") or []
    heading = getattr(chunk, "heading", None) or meta.get("heading")
    method = meta.get("method")
    confidence = meta.get("confidence")
    if method is None and document is not None:
        method = getattr(document, "extraction_method_summary", None)
        if isinstance(method, dict):
            method = ",".join(sorted(method.keys())) or None
    file_name = getattr(document, "file_name", None) if document is not None else meta.get("file_name")
    if file_name is None:
        # Never lazy-load chunk.document here: in async context that is a
        # MissingGreenlet. Read the already-loaded relationship only.
        doc = getattr(chunk, "__dict__", {}).get("document")
        file_name = getattr(doc, "file_name", None) if doc is not None else None
    return Citation(
        chunk_id=str(getattr(chunk, "id")),
        document_id=str(getattr(chunk, "document_id")),
        file_name=file_name,
        page_number=page_number,
        page_start=page_start,
        page_end=page_end,
        section_path=list(section_path) if section_path else [],
        heading=heading,
        method=method,
        confidence=confidence,
        span=(getattr(chunk, "content", "") or "")[:500],
        valid=True,
    )


async def resolve_citation(
    db: AsyncSession, chunk_id: str, allowed_chunk_ids: Optional[set] = None
) -> Citation:
    """Resolve a chunk id to a citation; rejects stale/foreign references.

    - unknown chunk id -> valid=False (never rendered as real)
    - allowed_chunk_ids given and id not in it -> valid=False (T307: model
      may cite only chunks present in the prompt)
    - resolved span must equal stored chunk text prefix (existence check)
    """
    from database.models import Chunk, Document

    try:
        cid = uuid.UUID(str(chunk_id))
    except Exception:
        return Citation(
            chunk_id=str(chunk_id), document_id="", valid=False,
            reason="malformed chunk id",
        )
    chunk = await db.get(Chunk, cid)
    if chunk is None:
        return Citation(
            chunk_id=str(chunk_id), document_id="", valid=False,
            reason="chunk not found (stale or deleted)",
        )
    if allowed_chunk_ids is not None and str(chunk.id) not in allowed_chunk_ids:
        return Citation(
            chunk_id=str(chunk.id), document_id=str(chunk.document_id),
            valid=False, reason="chunk was not in the retrieval set",
        )
    document = await db.get(Document, chunk.document_id)
    citation = build_citation_from_chunk(chunk, document)
    if not (chunk.content or "").strip():
        citation.valid = False
        citation.reason = "stored chunk text is empty"
    return citation


def validate_citation_ids(
    cited_ids: list, allowed_ids: set
) -> tuple[list, list]:
    """Split model-output citation ids into (valid, rejected). Pure helper."""
    valid = [c for c in cited_ids if str(c) in allowed_ids]
    rejected = [c for c in cited_ids if str(c) not in allowed_ids]
    return valid, rejected


__all__ = [
    "Citation",
    "build_citation_from_chunk",
    "resolve_citation",
    "validate_citation_ids",
]
