"""
Vector Retrieval API Routes (Phase 3, T304).

POST /api/retrieval/search: authenticated, ownership-scoped by project,
caps top_k and candidate count, returns results with citations or an
explicit state (indexing_in_progress with counts, unindexed,
empty_corpus). Never an empty list indistinguishable from "no relevant
content", never a 500 for a normal state.

Request naming accepts the frontend aliases (searchContent in
api-services.ts sends query/projectId/documentId/topK/minSimilarity).
"""

import uuid
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from database import repositories
from services.auth.deps import get_optional_user

from .service import retrieval_service
from .types import RetrievalStats


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


class SearchRequest(BaseModel):
    """Request for vector search (snake_case + frontend camelCase aliases)."""

    query: str = ""
    project_id: Optional[str] = Field(default=None, alias="projectId")
    document_id: Optional[str] = Field(default=None, alias="documentId")
    top_k: Optional[int] = Field(default=5, alias="topK")
    min_similarity: Optional[float] = Field(default=0.0, alias="minSimilarity")
    enable_rerank: Optional[bool] = None
    offset: int = 0

    model_config = {"populate_by_name": True}

    def resolved(self) -> dict:
        data = self.model_dump()
        # Pydantic v2 with aliases: model_dump keys are field names.
        return {
            "query": data.get("query", ""),
            "project_id": data.get("project_id"),
            "document_id": data.get("document_id"),
            "top_k": data.get("top_k", 5),
            "min_similarity": data.get("min_similarity", 0.0),
            "enable_rerank": data.get("enable_rerank"),
            "offset": data.get("offset", 0),
        }


class ChunkPayload(BaseModel):
    chunk_id: str
    document_id: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None
    similarity: float = 0.0
    content: str = ""
    source: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    citation: Optional[dict] = None


class SearchResponse(BaseModel):
    """Response for vector search"""
    query: str
    total_results: int
    chunks: list
    processing_time: float
    query_embedding_time: float = 0.0
    retrieval_time: float = 0.0
    # Phase 3 additions (all optional-safe for old clients)
    state: str = "ok"
    indexed_chunks: int = 0
    total_chunks: int = 0
    citations: list = Field(default_factory=list)
    rerank_applied: bool = False
    reranker: Optional[str] = None
    rerank_latency_ms: Optional[float] = None
    truncated_query: bool = False
    meta: dict = Field(default_factory=dict)


class ProjectSearchResponse(BaseModel):
    """Response for project-wide search"""
    project_id: str
    query: str
    results: SearchResponse


class DocumentSearchResponse(BaseModel):
    """Response for document-specific search"""
    document_id: str
    query: str
    results: SearchResponse


async def _check_scope(
    db: AsyncSession,
    project_id: Optional[str],
    document_id: Optional[str],
    current_user: Any,
) -> tuple:
    """Validate scope + ownership. Unknown/other-user scope -> 404.

    Returns (project_uuid_or_None, doc_uuid_or_None, user_id_or_None).
    """
    user_id = str(getattr(current_user, "id")) if current_user is not None else None
    project_uuid = None
    doc_uuid = None

    if project_id:
        try:
            project_uuid = uuid.UUID(project_id)
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid project ID format")
        project = await repositories.get_project_by_id(db, project_uuid)
        if project is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Project {project_id} not found")
        owner = str(getattr(project, "user_id", ""))
        if user_id is not None and owner and owner != user_id:
            # Indistinguishable from missing: user cannot probe others' ids.
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Project {project_id} not found")

    if document_id:
        try:
            doc_uuid = uuid.UUID(document_id)
        except Exception:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid document ID format")
        document = await repositories.get_document_by_id(db, doc_uuid)
        if document is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Document {document_id} not found")
        if project_uuid is None:
            # Ownership via the document's project when no project given.
            parent = await repositories.get_project_by_id(db, document.project_id)
            if parent is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Document {document_id} not found")
            owner = str(getattr(parent, "user_id", ""))
            if user_id is not None and owner and owner != user_id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                    detail=f"Document {document_id} not found")
            project_uuid = document.project_id
        elif document.project_id != project_uuid:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"Document {document_id} not found")
    return project_uuid, doc_uuid, user_id


def _to_response(result) -> SearchResponse:
    chunks = []
    for c in result.chunks:
        citation = c.citation.model_dump() if c.citation is not None else None
        chunks.append({
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "page": c.page,
            "section": c.section,
            "score": c.score if c.score is not None else c.similarity,
            "similarity": c.similarity,
            "content": c.content,
            "source": c.source,
            "metadata": c.metadata,
            "citation": citation,
        })
    return SearchResponse(
        query=result.query,
        total_results=result.total_results,
        chunks=chunks,
        processing_time=result.processing_time,
        query_embedding_time=result.query_embedding_time,
        retrieval_time=result.retrieval_time,
        state=result.state,
        indexed_chunks=result.indexed_chunks,
        total_chunks=result.total_chunks,
        citations=result.citations(),
        rerank_applied=result.rerank_applied,
        reranker=result.reranker,
        rerank_latency_ms=result.rerank_latency_ms,
        truncated_query=result.truncated_query,
        meta=result.meta,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_optional_user),
):
    """
    Search for relevant chunks using hybrid retrieval.

    - blank query -> 422; over-long query truncated + flagged
    - mid-index project -> indexing_in_progress with counts
    - other-user scope -> 404; top_k clamped to the configured maximum
    """
    params = request.resolved()
    raw_query = params["query"] or ""
    if not raw_query.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            detail="query must not be blank")

    project_uuid, doc_uuid, user_id = await _check_scope(
        db, params["project_id"], params["document_id"], current_user
    )
    if project_uuid is None and doc_uuid is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Either project_id or document_id must be provided")

    try:
        result = await retrieval_service.retrieve_with_db(
            query=raw_query,
            db=db,
            project_id=str(project_uuid) if project_uuid else None,
            document_id=str(doc_uuid) if doc_uuid else None,
            user_id=user_id,
            top_k=params["top_k"],
            min_similarity=params["min_similarity"],
            enable_rerank=params["enable_rerank"],
            offset=params["offset"] or 0,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )
    return _to_response(result)


@router.post("/search/project/{project_id}", response_model=ProjectSearchResponse)
async def search_project(
    project_id: str,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_optional_user),
):
    """Search across all documents in a project."""
    params = request.resolved()
    if not (params["query"] or "").strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            detail="query must not be blank")
    project_uuid, _, user_id = await _check_scope(db, project_id, None, current_user)
    result = await retrieval_service.retrieve_with_db(
        query=params["query"],
        db=db,
        project_id=str(project_uuid),
        top_k=params["top_k"] or 5,
        min_similarity=params["min_similarity"] or 0.0,
        enable_rerank=params["enable_rerank"],
        offset=params["offset"] or 0,
    )
    return ProjectSearchResponse(
        project_id=project_id,
        query=result.query,
        results=_to_response(result),
    )


@router.post("/search/document/{document_id}", response_model=DocumentSearchResponse)
async def search_document(
    document_id: str,
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Any = Depends(get_optional_user),
):
    """Search within a specific document."""
    params = request.resolved()
    if not (params["query"] or "").strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                            detail="query must not be blank")
    _, doc_uuid, user_id = await _check_scope(db, None, document_id, current_user)
    result = await retrieval_service.retrieve_with_db(
        query=params["query"],
        db=db,
        document_id=str(doc_uuid),
        top_k=params["top_k"] or 5,
        min_similarity=params["min_similarity"] or 0.0,
        enable_rerank=params["enable_rerank"],
        offset=params["offset"] or 0,
    )
    return DocumentSearchResponse(
        document_id=document_id,
        query=result.query,
        results=_to_response(result),
    )


@router.get("/stats/{project_id}", response_model=RetrievalStats)
async def get_retrieval_stats(
    project_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get retrieval statistics for a project"""
    try:
        project_uuid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid project ID format",
        )
    project = await repositories.get_project_by_id(db, project_uuid)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    stats = await retrieval_service.get_retrieval_stats(db, project_id)
    return stats


@router.get("/health")
async def health_check():
    """Health check for retrieval service"""
    return {
        "status": "healthy",
        "service": "retrieval",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": 1024,
        "fusion": "rrf",
        "reranker": "BAAI/bge-reranker-v2-m3",
    }


__all__ = ["router"]
