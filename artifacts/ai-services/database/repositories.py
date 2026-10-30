"""
Repository Functions for Database Access

Simple, reusable helper functions for common database operations.
"""

import uuid
from datetime import datetime
from typing import Optional, List, Any, Tuple
from sqlalchemy import select, update, delete, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import (
    User, Project, Document, DocumentPage, Chunk,
    FlashcardDeck, Flashcard, Note, Quiz,
    StudySession, GeneratedContentCache,
    Job, Website,
    ProjectStatus, DocumentStatus, JobStatus
)


# ==================== USER REPOSITORY ====================

async def create_user(
    db: AsyncSession,
    email: str,
    password_hash: str,
    name: Optional[str] = None,
    settings: Optional[dict] = None
) -> User:
    """Create a new user"""
    user = User(
        id=uuid.uuid4(),
        email=email,
        password_hash=password_hash,
        name=name,
        settings=settings or {}
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    """Get user by ID"""
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email"""
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()


async def get_user_by_email_ci(db: AsyncSession, email: str) -> Optional[User]:
    """Get user by email, case-insensitively (pre-lower()-index rows)."""
    from sqlalchemy import func

    result = await db.execute(
        select(User).where(func.lower(User.email) == email.strip().lower())
    )
    return result.scalar_one_or_none()


# ==================== PROJECT REPOSITORY ====================

async def create_project(
    db: AsyncSession,
    user_id: uuid.UUID,
    name: str,
    description: Optional[str] = None,
    subject: Optional[str] = None,
) -> Project:
    """Create a new project"""
    project = Project(
        id=uuid.uuid4(),
        user_id=user_id,
        name=name,
        description=description,
        subject=subject,
        status=ProjectStatus.PROCESSING,
        progress=0
    )
    db.add(project)
    await db.flush()
    await db.refresh(project)
    return project


async def get_project_by_id(db: AsyncSession, project_id: uuid.UUID) -> Optional[Project]:
    """Get project by ID"""
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    return result.scalar_one_or_none()


async def get_projects_by_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 100
) -> List[Project]:
    """Get all projects for a user"""
    result = await db.execute(
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(Project.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def list_projects_by_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0,
) -> Tuple[List[Project], int]:
    """Paginated owned-projects listing with total count."""
    from sqlalchemy import func

    total = (
        await db.execute(
            select(func.count()).select_from(Project).where(Project.user_id == user_id)
        )
    ).scalar() or 0
    result = await db.execute(
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(Project.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all()), int(total)


async def update_project(
    db: AsyncSession,
    project_id: uuid.UUID,
    name: Optional[str] = None,
    description: Optional[str] = None,
    subject: Optional[str] = None,
) -> Optional[Project]:
    """Update mutable project fields."""
    project = await get_project_by_id(db, project_id)
    if project is None:
        return None
    if name is not None:
        project.name = name
    if description is not None:
        project.description = description
    if subject is not None:
        project.subject = subject
    project.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(project)
    return project


async def delete_project(db: AsyncSession, project_id: uuid.UUID) -> bool:
    """Delete a project (documents/chunks cascade). Returns False when missing."""
    project = await get_project_by_id(db, project_id)
    if project is None:
        return False
    await db.delete(project)
    await db.flush()
    return True


async def update_project_status(
    db: AsyncSession,
    project_id: uuid.UUID,
    status: ProjectStatus,
    progress: Optional[int] = None
) -> Optional[Project]:
    """Update project status and optionally progress"""
    project = await get_project_by_id(db, project_id)
    if project:
        project.status = status
        if progress is not None:
            project.progress = progress
        project.updated_at = datetime.utcnow()
    return project


# ==================== DOCUMENT REPOSITORY ====================

async def create_document(
    db: AsyncSession,
    project_id: uuid.UUID,
    file_name: str,
    file_type: str,
    file_path: str,
    page_count: Optional[int] = None,
    is_ocr_required: bool = False,
    source_type: Optional[str] = None,
) -> Document:
    """Create a new document record"""
    from database.models import FileType
    from database.models.document import DocumentSourceType

    normalized_source = source_type or "study_material"
    try:
        source_enum = DocumentSourceType(normalized_source)
    except (ValueError, TypeError):
        raise ValueError(
            "source_type must be one of: " + ", ".join(item.value for item in DocumentSourceType)
        )
    document = Document(
        id=uuid.uuid4(),
        project_id=project_id,
        file_name=file_name,
        file_type=FileType(file_type),
        file_path=file_path,
        page_count=page_count,
        processing_status=DocumentStatus.PENDING,
        is_ocr_required=is_ocr_required,
        source_type=source_enum,
    )
    db.add(document)
    await db.flush()
    await db.refresh(document)
    return document


async def get_document_by_id(db: AsyncSession, document_id: uuid.UUID) -> Optional[Document]:
    """Get document by ID"""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    return result.scalar_one_or_none()


async def get_document_by_id_with_project(
    db: AsyncSession, document_id: uuid.UUID
) -> Optional[Document]:
    """Get document with its project eager-loaded (T401/ADR-006 A1).

    Async worker code must never lazy-load relationships (MissingGreenlet).
    Callers must still prefer job.input_data project_id/user_id and treat
    .project as a fallback only.
    """
    result = await db.execute(
        select(Document)
        .options(selectinload(Document.project))
        .where(Document.id == document_id)
    )
    return result.scalar_one_or_none()


async def set_document_indexing_state(
    db: AsyncSession,
    document_id: uuid.UUID,
    indexing_state: str,
    degraded_reason: Optional[str] = None,
    extraction_status: Optional[str] = None,
) -> Optional[Document]:
    """Record an honest indexing/degraded state (T402/T404).

    Writes indexing_state (+ degraded info on best-effort attrs) and flushes.
    The caller owns the commit.
    """
    document = await get_document_by_id(db, document_id)
    if document is None:
        return None
    try:
        document.indexing_state = indexing_state
    except Exception:
        pass
    if degraded_reason is not None:
        try:
            document.processing_error = degraded_reason
        except Exception:
            pass
    if extraction_status is not None:
        try:
            document.extraction_status = extraction_status
        except Exception:
            pass
    try:
        await db.flush()
    except Exception:
        pass
    return document


async def get_documents_by_project(
    db: AsyncSession,
    project_id: uuid.UUID
) -> List[Document]:
    """Get all documents for a project"""
    result = await db.execute(
        select(Document)
        .where(Document.project_id == project_id)
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars().all())


async def update_document_status(
    db: AsyncSession,
    document_id: uuid.UUID,
    status: DocumentStatus,
    extracted_text: Optional[str] = None,
    error: Optional[str] = None
) -> Optional[Document]:
    """Update document processing status"""
    document = await get_document_by_id(db, document_id)
    if document:
        document.processing_status = status
        if extracted_text is not None:
            document.extracted_text = extracted_text
        if error is not None:
            document.processing_error = error
        document.updated_at = datetime.utcnow()
    return document


# ==================== DOCUMENT PAGE REPOSITORY (T202) ====================

async def upsert_page(
    db: AsyncSession,
    document_id: uuid.UUID,
    page_number: int,
    text: str = "",
    method: str = "native",
    confidence: float = 0.0,
    parser_version: Optional[str] = None,
    image_stored: bool = False,
    bbox_summary: Optional[dict] = None,
    failure_reason: Optional[str] = None,
) -> Any:
    """Idempotent upsert on (document_id, page_number). Never duplicates."""
    existing = await get_page(db, document_id, page_number)
    if existing is not None:
        existing.text = text
        existing.char_count = len(text or "")
        existing.method = method
        existing.confidence = confidence
        if parser_version is not None:
            existing.parser_version = parser_version
        existing.image_stored = image_stored
        if bbox_summary is not None:
            existing.bbox_summary = bbox_summary
        existing.failure_reason = failure_reason
        await db.flush()
        return existing
    page = DocumentPage(
        id=uuid.uuid4(),
        document_id=document_id,
        page_number=page_number,
        text=text or "",
        char_count=len(text or ""),
        method=method,
        confidence=confidence,
        parser_version=parser_version,
        image_stored=image_stored,
        bbox_summary=bbox_summary,
        failure_reason=failure_reason,
    )
    db.add(page)
    await db.flush()
    return page


async def get_page(
    db: AsyncSession, document_id: uuid.UUID, page_number: int
) -> Optional[Any]:
    result = await db.execute(
        select(DocumentPage).where(
            DocumentPage.document_id == document_id,
            DocumentPage.page_number == page_number,
        )
    )
    return result.scalar_one_or_none()


async def get_pages(
    db: AsyncSession, document_id: uuid.UUID
) -> List[Any]:
    result = await db.execute(
        select(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number.asc())
    )
    return list(result.scalars().all())


async def count_pages_with_text(db: AsyncSession, document_id: uuid.UUID) -> int:
    pages = await get_pages(db, document_id)
    return sum(1 for p in pages if (p.text or "").strip())


async def page_method_breakdown(db: AsyncSession, document_id: uuid.UUID) -> dict:
    pages = await get_pages(db, document_id)
    breakdown: dict = {}
    for p in pages:
        breakdown[p.method] = breakdown.get(p.method, 0) + 1
    return breakdown


async def delete_pages_for_document(db: AsyncSession, document_id: uuid.UUID) -> int:
    pages = await get_pages(db, document_id)
    for p in pages:
        await db.delete(p)
    await db.flush()
    return len(pages)


# ==================== CHUNK IDEMPOTENCY (T207) ====================

async def upsert_chunks_batch(
    db: AsyncSession, chunks_data: List[dict]
) -> List[Chunk]:
    """Upsert on (document_id, content_hash); reprocessing adds zero rows."""
    out: List[Chunk] = []
    for data in chunks_data:
        doc_id = data["document_id"]
        chash = data.get("content_hash")
        existing = None
        if chash:
            result = await db.execute(
                select(Chunk).where(Chunk.document_id == doc_id, Chunk.content_hash == chash)
            )
            existing = result.scalar_one_or_none()
        if existing is not None:
            # Refresh mutable provenance fields.
            for key in ("content", "page_start", "page_end", "section_path",
                        "heading", "position", "embedding_model", "embedding_dim"):
                if key in data and data[key] is not None:
                    try:
                        setattr(existing, key, data[key])
                    except Exception:
                        pass
            if data.get("metadata") is not None:
                try:
                    existing.chunk_metadata = data["metadata"]
                except Exception:
                    pass
            await db.flush()
            out.append(existing)
            continue
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            content=data["content"],
            embedding=data.get("embedding"),
            chunk_metadata=data.get("metadata"),
            token_count=data.get("token_count"),
        )
        for key in ("embedding_model", "embedding_dim", "content_hash",
                    "page_start", "page_end", "section_path", "heading", "position"):
            try:
                if data.get(key) is not None:
                    setattr(chunk, key, data[key])
            except Exception:
                pass
        db.add(chunk)
        await db.flush()
        out.append(chunk)
    return out


async def delete_chunks_for_document(db: AsyncSession, document_id: uuid.UUID) -> int:
    chunks = await get_chunks_by_document(db, document_id)
    for c in chunks:
        await db.delete(c)
    await db.flush()
    return len(chunks)


# ==================== CHUNK REPOSITORY ====================

async def create_chunk(
    db: AsyncSession,
    document_id: uuid.UUID,
    content: str,
    embedding: Optional[bytes] = None,
    metadata: Optional[dict] = None,
    token_count: Optional[int] = None
) -> Chunk:
    """Create a new chunk"""
    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document_id,
        content=content,
        embedding=embedding,
        chunk_metadata=metadata,
        token_count=token_count
    )
    db.add(chunk)
    await db.flush()
    await db.refresh(chunk)
    return chunk


async def create_chunks_batch(
    db: AsyncSession,
    chunks_data: List[dict]
) -> List[Chunk]:
    """Create multiple chunks at once"""
    chunks = [
        Chunk(
            id=uuid.uuid4(),
            document_id=data["document_id"],
            content=data["content"],
            embedding=data.get("embedding"),
            chunk_metadata=data.get("metadata"),
            token_count=data.get("token_count")
        )
        for data in chunks_data
    ]
    db.add_all(chunks)
    await db.flush()
    return chunks


async def get_chunks_by_document(
    db: AsyncSession,
    document_id: uuid.UUID
) -> List[Chunk]:
    """Get all chunks for a document"""
    result = await db.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.created_at.asc())
    )
    return list(result.scalars().all())


async def get_chunks_by_project(
    db: AsyncSession,
    project_id: uuid.UUID,
    limit: int = 1000
) -> List[Chunk]:
    """Get all chunks for a project"""
    result = await db.execute(
        select(Chunk)
        .join(Document)
        .where(Document.project_id == project_id)
        .limit(limit)
    )
    return list(result.scalars().all())


async def search_similar_chunks(
    db: AsyncSession,
    project_id: uuid.UUID,
    embedding: bytes,
    limit: int = 5
) -> List[Chunk]:
    """
    Deprecated wrapper kept for backward compatibility.
    Prefer search_similar_chunks_hybrid, which runs the vector search in
    SQL (pgvector cosine) with project/document/user filtering instead of
    ignoring the embedding. This wrapper scopes to the project only.
    """
    rows = await search_similar_chunks_hybrid(
        db,
        query_embedding=embedding,
        project_id=project_id,
        limit=limit,
    )
    return [row[0] for row in rows]


async def _is_postgres(db: AsyncSession) -> bool:
    try:
        bind = db.bind if hasattr(db, "bind") else None
        return (bind.dialect.name if bind is not None else "") == "postgresql"
    except Exception:
        return False


async def get_indexing_state(
    db: AsyncSession,
    project_id: Optional[uuid.UUID] = None,
    document_id: Optional[uuid.UUID] = None,
    embedding_model: Optional[str] = None,
) -> dict:
    """Honest indexing counts for a scope (T301).

    total = chunks in scope; indexed = chunks with a current-model
    embedding. A scope with total > 0 and indexed < total is mid-reindex
    and retrieval must report indexing_in_progress, never an empty list.
    """
    from sqlalchemy import func, or_, text as _text

    from database.models import Document as _Document

    pg = await _is_postgres(db)
    base = select(func.count(Chunk.id))
    if embedding_model:
        # D1: live backfill writes embedding_v via raw SQL and tags
        # embedding_model via ORM but never fills legacy BYTEA, so the
        # model tag alone is the indexed predicate. Requiring legacy
        # embedding NOT NULL here reported 0 on fully backfilled scopes.
        idx = select(func.count(Chunk.id)).where(
            Chunk.embedding_model == embedding_model
        )
    elif pg:
        # No model filter on Postgres: legacy BYTEA OR new vector column.
        # Chunk ORM maps no embedding_v attribute, so the vector arm is a
        # raw-SQL fragment (SQLite/test doubles never see it: pg-guarded).
        idx = select(func.count(Chunk.id)).where(
            or_(
                Chunk.embedding.isnot(None),
                _text("chunks.embedding_v IS NOT NULL"),
            )
        )
    else:
        idx = select(func.count(Chunk.id)).where(Chunk.embedding.isnot(None))
    if document_id is not None:
        base = base.where(Chunk.document_id == document_id)
        idx = idx.where(Chunk.document_id == document_id)
    elif project_id is not None:
        base = base.join(_Document, Chunk.document_id == _Document.id).where(
            _Document.project_id == project_id
        )
        idx = idx.join(_Document, Chunk.document_id == _Document.id).where(
            _Document.project_id == project_id
        )
    # Legacy BYTEA embeddings count as indexed only when no model filter
    # is configured; a configured model requires the model tag so a
    # partially re-embedded project cannot mix vector spaces.
    total = (await db.execute(base)).scalar() or 0
    indexed = (await db.execute(idx)).scalar() or 0
    if total == 0:
        state = "empty_corpus"
    elif indexed == 0:
        state = "unindexed"
    elif indexed < total:
        state = "indexing_in_progress"
    else:
        state = "ok"
    return {
        "state": state,
        "total_chunks": int(total),
        "indexed_chunks": int(indexed),
    }


async def search_similar_chunks_hybrid(
    db: AsyncSession,
    query_embedding: Any,
    query_text: str = "",
    project_id: Optional[uuid.UUID] = None,
    document_id: Optional[uuid.UUID] = None,
    user_id: Optional[uuid.UUID] = None,
    embedding_model: Optional[str] = None,
    per_arm: int = 50,
    limit: int = 5,
    offset: int = 0,
    min_similarity: float = 0.0,
    relaxed: bool = False,
) -> List[Tuple[Chunk, float, str]]:
    """Hybrid retrieval executed in SQL on Postgres (T301).

    - Vector arm: ORDER BY embedding_v <=> :q LIMIT :per_arm (HNSW index;
      iterative scan enabled for filtered queries so top_k rows are
      returned, not silently fewer).
    - Lexical arm: ts_rank over the tsv generated column (GIN index).
    - Fusion: RRF in one statement (FULL OUTER JOIN pattern, ADR-005 s5).
    - Rows whose embedding_model differs from the configured model are
      excluded, so a partially re-embedded project cannot mix spaces.
    - User isolation: when user_id is given, only projects owned by that
      user are searched.

    ``relaxed`` re-runs the lexical arm OR-joined instead of AND-joined.
    It never widens the scope: it is only meant for a caller whose strict
    pass over THIS scope returned zero rows (a question whose tokens do
    not co-occur in one chunk), so the recovery stays a real lexical
    match over the same project/document/user filter.

    Returns [(Chunk, fused_score, arm)] with arm in
    vector|lexical|both. On non-Postgres binds (SQLite tests) raises
    RuntimeError so the caller can use the offline RRF fallback honestly.
    """
    from sqlalchemy import text as _text

    from database.models import Document as _Document
    from database.models import Project as _Project
    from services.retrieval.hybrid import FUSED_SQL, ITERATIVE_SCAN_SQL

    if not await _is_postgres(db):
        raise RuntimeError("hybrid SQL requires Postgres+pgvector; use offline fallback")

    if isinstance(query_embedding, (bytes, bytearray)):
        import struct as _struct

        n = len(bytes(query_embedding)) // 4
        vec = list(_struct.unpack(f"{n}f", bytes(query_embedding)))
    else:
        vec = [float(v) for v in query_embedding]
    vec_literal = "[" + ",".join(repr(v) for v in vec) + "]"

    filtered = project_id is not None or document_id is not None or user_id is not None
    # Per-session iterative scan for filtered HNSW queries (pgvector>=0.8).
    if filtered:
        try:
            await db.execute(_text(ITERATIVE_SCAN_SQL))
        except Exception:
            pass
    params = {
        "query_embedding": vec_literal,
        "query_text": query_text or "",
        "embedding_model": embedding_model or "Qwen/Qwen3-Embedding-0.6B",
        "per_arm": int(per_arm),
        "limit": int(limit) + int(offset),
        # 1 = OR-joined lexical arm (bounded recovery for a strict zero
        # result over the same scope), 0 = strict plainto AND (default).
        "relaxed": 1 if relaxed else 0,
        # Typed UUID binds (live asyncpg fix): always bound (None =
        # unscoped), FUSED_SQL double-CASTs via text so asyncpg infers
        # uuid even for NULL. Empty strings are normalised to None so
        # NULLIF-style surprises never reach the CTE.
        "project_id": str(project_id) if project_id else None,
        "document_id": str(document_id) if document_id else None,
        # Ownership gate lives inside FUSED_SQL (typed uid CTE column):
        # None disables it. Always bound so asyncpg sees one typed $n.
        "user_id": str(user_id) if user_id else None,
    }
    sql = FUSED_SQL
    try:
        result = await db.execute(_text(sql), params)
    except Exception:
        # A failed fused statement aborts the PG transaction: roll back
        # here (service also rolls back before its offline fallback) so
        # later statements on this session still run. Re-raise so the
        # caller degrades honestly instead of returning partial rows.
        try:
            await db.rollback()
        except Exception:
            pass
        raise
    id_rows = result.all()
    out: List[Tuple[Chunk, float, str]] = []
    for row in id_rows[offset:]:
        mapping = row._mapping  # type: ignore[attr-defined]
        cid = uuid.UUID(str(mapping["chunk_id"]))
        # Eager-load chunk.document: any lazy relationship touch later
        # (citation building) would MissingGreenlet in async context
        # (T401 pattern; Phase B live find).
        from sqlalchemy.orm import selectinload as _selectinload

        chunk = (
            await db.execute(
                select(Chunk)
                .options(_selectinload(Chunk.document))
                .where(Chunk.id == cid)
            )
        ).scalars().first()
        if chunk is None:
            continue
        score = float(mapping["rrf"] or 0.0)
        if min_similarity and score < min_similarity:
            continue
        out.append((chunk, score, str(mapping.get("arm") or "both")))
    return out


# ==================== FLASHCARD REPOSITORY ====================

async def create_flashcard_deck(
    db: AsyncSession,
    project_id: uuid.UUID,
    name: str
) -> FlashcardDeck:
    """Create a new flashcard deck"""
    deck = FlashcardDeck(
        id=uuid.uuid4(),
        project_id=project_id,
        name=name,
        card_count=0
    )
    db.add(deck)
    await db.flush()
    await db.refresh(deck)
    return deck


async def create_flashcard(
    db: AsyncSession,
    deck_id: uuid.UUID,
    front: str,
    back: str,
    difficulty: str = "medium"
) -> Flashcard:
    """Create a new flashcard"""
    from database.models import Difficulty
    
    flashcard = Flashcard(
        id=uuid.uuid4(),
        deck_id=deck_id,
        front=front,
        back=back,
        difficulty=Difficulty(difficulty)
    )
    db.add(flashcard)
    
    # Update deck card count
    deck = await db.get(FlashcardDeck, deck_id)
    if deck:
        deck.card_count += 1
    
    await db.flush()
    await db.refresh(flashcard)
    return flashcard


async def get_flashcard_decks_by_project(
    db: AsyncSession,
    project_id: uuid.UUID
) -> List[FlashcardDeck]:
    """Get all flashcard decks for a project.

    Eager-loads deck.flashcards: any lazy collection touch later
    (read route, study sessions) would MissingGreenlet in async
    context (T401 pattern; Phase B live find).
    """
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(FlashcardDeck)
        .options(selectinload(FlashcardDeck.flashcards))
        .where(FlashcardDeck.project_id == project_id)
    )
    return list(result.scalars().all())


# ==================== NOTE REPOSITORY ====================

async def create_note(
    db: AsyncSession,
    project_id: uuid.UUID,
    title: str,
    content: Optional[dict] = None
) -> Note:
    """Create a new note"""
    note = Note(
        id=uuid.uuid4(),
        project_id=project_id,
        title=title,
        content=content
    )
    db.add(note)
    await db.flush()
    await db.refresh(note)
    return note


async def get_notes_by_project(
    db: AsyncSession,
    project_id: uuid.UUID
) -> List[Note]:
    """Get all notes for a project, newest first (stable order).

    Ordering is explicit (created_at desc, then id) so the notes LIST is
    deterministic instead of whatever order the planner returns. Callers get
    every note artifact for the project, not only the latest.
    """
    result = await db.execute(
        select(Note)
        .where(Note.project_id == project_id)
        .order_by(Note.created_at.desc(), Note.id.desc())
    )
    return list(result.scalars().all())


async def get_note_by_id(
    db: AsyncSession,
    note_id: uuid.UUID,
) -> Optional[Note]:
    """Get one note by id (or None)."""
    result = await db.execute(select(Note).where(Note.id == note_id))
    return result.scalars().first()


async def delete_note(
    db: AsyncSession,
    note_id: uuid.UUID,
) -> bool:
    """Delete one note row. True when a row was removed.

    Ownership is the caller's check (the note must already be scoped to the
    project the caller may access); this only removes the row.
    """
    note = await get_note_by_id(db, note_id)
    if note is None:
        return False
    await db.delete(note)
    await db.flush()
    return True


# ==================== QUIZ REPOSITORY ====================

async def create_quiz(
    db: AsyncSession,
    project_id: uuid.UUID,
    title: str,
    questions: Optional[List[dict]] = None
) -> Quiz:
    """Create a new quiz"""
    quiz = Quiz(
        id=uuid.uuid4(),
        project_id=project_id,
        title=title,
        questions=questions
    )
    db.add(quiz)
    await db.flush()
    await db.refresh(quiz)
    return quiz


async def get_quizzes_by_project(
    db: AsyncSession,
    project_id: uuid.UUID
) -> List[Quiz]:
    """Get all quizzes for a project"""
    result = await db.execute(
        select(Quiz)
        .where(Quiz.project_id == project_id)
    )
    return list(result.scalars().all())


# ==================== JOB REPOSITORY ====================

async def create_job(
    db: AsyncSession,
    job_type: str,
    input_data: Optional[dict] = None,
    project_id: Optional[uuid.UUID] = None,
    document_id: Optional[uuid.UUID] = None,
    priority: int = 0,
    max_attempts: int = 3,
    idempotency_key: Optional[str] = None,
) -> Job:
    """Create a new job (queue columns default; commit is the caller's job)."""
    from database.models import JobType

    job = Job(
        id=uuid.uuid4(),
        job_type=JobType(job_type),
        status=JobStatus.PENDING,
        progress=0,
        document_id=document_id,
        project_id=project_id,
        attempts=0,
        max_attempts=max_attempts,
        priority=priority,
        input_data=input_data,
        idempotency_key=idempotency_key,
    )
    db.add(job)
    await db.flush()
    await db.refresh(job)
    return job


async def update_job_status(
    db: AsyncSession,
    job_id: uuid.UUID,
    status: JobStatus,
    progress: Optional[int] = None,
    output_data: Optional[dict] = None,
    error_message: Optional[str] = None,
    input_data: Optional[dict] = None,
    stage: Optional[str] = None,
    stage_message: Optional[str] = None,
) -> Optional[Job]:
    """Update job status.

    output_data keys are MERGED into the existing mapping (T404/T409 single
    writer rule): a progress write must never wipe _stages_done/_stage and
    break resume. Pass output_data=None to leave it untouched.
    """
    job = await db.get(Job, job_id)
    if job:
        job.status = status
        if progress is not None:
            job.progress = progress
        if output_data is not None:
            merged = dict(job.output_data or {})
            merged.update(output_data)
            job.output_data = merged
        if input_data is not None:
            job.input_data = input_data
        if error_message is not None:
            job.error_message = error_message
        if stage is not None:
            job.stage = stage
        if stage_message is not None:
            job.stage_message = stage_message
        if status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = datetime.utcnow()
        if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            job.completed_at = datetime.utcnow()
        job.updated_at = datetime.utcnow()
    return job


async def get_job_by_id(db: AsyncSession, job_id: uuid.UUID) -> Optional[Job]:
    """Get job by ID"""
    return await db.get(Job, job_id)


async def get_pending_jobs(
    db: AsyncSession,
    limit: int = 100
) -> List[Job]:
    """Get pending jobs"""
    result = await db.execute(
        select(Job)
        .where(Job.status == JobStatus.PENDING)
        .order_by(Job.created_at.asc())
        .limit(limit)
    )
    return list(result.scalars().all())


# ==================== STUDY SESSION REPOSITORY ====================

async def create_study_session(
    db: AsyncSession,
    user_id: uuid.UUID,
    project_id: uuid.UUID,
    session_type: str,
    started_at: Optional[datetime] = None,
    session_id: Optional[uuid.UUID] = None,
) -> StudySession:
    """Create a new study session.

    session_id defaults to a fresh UUID, but start_session passes the
    live in-memory id so the row is restart-resolvable: after a process
    restart resolve_session rebuilds state from this row (T408/B4).
    """
    from database.models import SessionType

    session = StudySession(
        id=session_id or uuid.uuid4(),
        user_id=user_id,
        project_id=project_id,
        session_type=SessionType(session_type),
        started_at=started_at or datetime.utcnow()
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


async def complete_study_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    score: float,
    duration_ms: int
) -> Optional[StudySession]:
    """Complete a study session"""
    session = await db.get(StudySession, session_id)
    if session:
        session.completed_at = datetime.utcnow()
        session.score = score
        session.duration_ms = duration_ms
    return session


# ==================== GENERATED CONTENT CACHE ====================

async def get_cached_content(
    db: AsyncSession,
    project_id: uuid.UUID,
    content_hash: str,
    output_type: str
) -> Optional[GeneratedContentCache]:
    """Get cached generated content"""
    from database.models import OutputType
    
    result = await db.execute(
        select(GeneratedContentCache)
        .where(
            GeneratedContentCache.project_id == project_id,
            GeneratedContentCache.content_hash == content_hash,
            GeneratedContentCache.output_type == OutputType(output_type)
        )
    )
    return result.scalar_one_or_none()


async def create_cached_content(
    db: AsyncSession,
    project_id: uuid.UUID,
    content_hash: str,
    output_type: str,
    output_content: dict
) -> GeneratedContentCache:
    """Create cached generated content"""
    from database.models import OutputType
    
    cache = GeneratedContentCache(
        id=uuid.uuid4(),
        project_id=project_id,
        content_hash=content_hash,
        output_type=OutputType(output_type),
        output_content=output_content
    )
    db.add(cache)
    await db.flush()
    await db.refresh(cache)
    return cache


# ==================== WEBSITE REPOSITORY ====================
#
# Live-bug batch: databases stamped before migration 008 have NO
# title/description/visibility/status/version columns, so a plain
# select(Website) 500s with `column websites.title does not exist`.
# Every read below catches that missing-column error, rolls the
# aborted statement back, and retries against the legacy column set
# only; writes retry without the additive columns. No data loss:
# legacy columns are always selected, additive ones are best-effort.

_WEBSITE_LEGACY_COLS = (
    "id", "project_id", "website_type", "html_content",
    "slug", "is_published", "created_at", "updated_at",
)

_WEBSITE_ADDITIVE_COLS = ("title", "description", "visibility", "status", "version")


def _is_missing_website_column(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "does not exist" not in msg:
        return False
    return any(
        f"websites.{col}" in msg or f'"{col}"' in msg or f" {col} " in msg
        for col in ("title", "description", "visibility", "status", "version")
    )


async def _rollback_quiet(db: AsyncSession) -> None:
    try:
        await db.rollback()
    except Exception:
        pass


def _legacy_website_from_row(row: Any) -> "Website":
    """Build a Website ORM object from a legacy-column row mapping."""
    from database.models import WebsiteType as _WT

    m = row._mapping  # type: ignore[attr-defined]
    w = Website(
        id=m["id"],
        project_id=m["project_id"],
        slug=m["slug"],
        website_type=_WT(m["website_type"]) if not isinstance(m["website_type"], _WT) else m["website_type"],
        html_content=m.get("html_content"),
    )
    w.is_published = bool(m.get("is_published", False))
    w.created_at = m.get("created_at")
    w.updated_at = m.get("updated_at")
    # Additive attrs stay at ORM defaults; service reads them via getattr.
    return w


async def _legacy_websites_by_slug(db: AsyncSession, slug: str) -> Optional["Website"]:
    from sqlalchemy import text as _text

    cols = ", ".join(_WEBSITE_LEGACY_COLS)
    result = await db.execute(
        _text(f"SELECT {cols} FROM websites WHERE slug = :slug"), {"slug": slug}
    )
    row = result.mappings().first()
    return _legacy_website_from_row(row) if row is not None else None


async def _legacy_websites_by_project(db: AsyncSession, project_id: uuid.UUID) -> List["Website"]:
    from sqlalchemy import text as _text

    cols = ", ".join(_WEBSITE_LEGACY_COLS)
    result = await db.execute(
        _text(f"SELECT {cols} FROM websites WHERE project_id = :pid"),
        {"pid": str(project_id)},
    )
    return [_legacy_website_from_row(r) for r in result.mappings().all()]


async def create_website(
    db: AsyncSession,
    project_id: uuid.UUID,
    slug: str,
    website_type: str,
    html_content: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> Website:
    """Create a new website (content refs + rendered HTML, draft/private)."""
    from database.models import WebsiteType

    website = Website(
        id=uuid.uuid4(),
        project_id=project_id,
        slug=slug,
        website_type=WebsiteType(website_type),
        html_content=html_content,
    )
    # Additive columns may be missing on pre-migration stores; set best-effort.
    try:
        if title is not None:
            website.title = title
        if description is not None:
            website.description = description
    except Exception:
        pass
    db.add(website)
    try:
        await db.flush()
    except Exception as exc:
        if not _is_missing_website_column(exc):
            raise
        # Pre-008 store: retry the INSERT without additive columns.
        await _rollback_quiet(db)
        from database.models import Website as _W
        from database.models import WebsiteType as _WT2

        legacy = _W(
            id=website.id,
            project_id=project_id,
            slug=slug,
            website_type=_WT2(website_type),
            html_content=html_content,
        )
        db.add(legacy)
        await db.flush()
        website = legacy
    try:
        await db.refresh(website)
    except Exception as exc:
        if not _is_missing_website_column(exc):
            raise
        # Refresh SELECT hit the missing column; the row is persisted,
        # expunge + return the flushed instance (service getattr-guards).
        await _rollback_quiet(db)
    return website


async def slug_exists(db: AsyncSession, slug: str) -> bool:
    """True when a website already claims this slug (uniqueness guard)."""
    try:
        result = await db.execute(select(Website).where(Website.slug == slug))
        return result.scalar_one_or_none() is not None
    except Exception as exc:
        if not _is_missing_website_column(exc):
            raise
        await _rollback_quiet(db)
        return await _legacy_websites_by_slug(db, slug) is not None


async def delete_website(db: AsyncSession, website: Website) -> None:
    """Delete a website row (ownership checked by the caller)."""
    await db.delete(website)
    await db.commit()


async def get_website_by_slug(db: AsyncSession, slug: str) -> Optional[Website]:
    """Get website by slug (pre-008 tolerant: legacy-column retry)."""
    try:
        result = await db.execute(
            select(Website).where(Website.slug == slug)
        )
        return result.scalar_one_or_none()
    except Exception as exc:
        if not _is_missing_website_column(exc):
            raise
        await _rollback_quiet(db)
        return await _legacy_websites_by_slug(db, slug)


async def get_websites_by_project(
    db: AsyncSession,
    project_id: uuid.UUID
) -> List[Website]:
    """Get all websites for a project (pre-008 tolerant)."""
    try:
        result = await db.execute(
            select(Website)
            .where(Website.project_id == project_id)
        )
        return list(result.scalars().all())
    except Exception as exc:
        if not _is_missing_website_column(exc):
            raise
        await _rollback_quiet(db)
        return await _legacy_websites_by_project(db, project_id)


# ==================== STRUCTURE REPROCESS (ADR-008 / T009) ====================


def _existing_columns_sync(bind: Any, table: str) -> Optional[set]:
    """Live column names of `table` for a SYNC bind, or None when absent.

    Must be reached through `AsyncSession.run_sync` (see
    `_existing_columns`).  Calling `sqlalchemy.inspect()` directly on an
    AsyncSession bind performs sync IO on an async connection; the
    resulting MissingGreenlet was swallowed by the caller's guard and
    degraded every schema check to a silent no-op.
    """
    try:
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(bind)
        if table not in inspector.get_table_names():
            return None
        return {str(row["name"]) for row in inspector.get_columns(table)}
    except Exception:
        return None


async def _existing_columns(db: AsyncSession, table: str) -> Optional[set]:
    """Live column names of `table`, or None when unknown/absent.

    Migration 014 adds the structure verdict columns additively; a database
    that has not run it (or a test double with no real bind) must degrade to
    a no-op instead of raising.
    """
    run_sync = getattr(db, "run_sync", None)
    if run_sync is not None:
        try:
            return await run_sync(
                lambda session: _existing_columns_sync(
                    session.get_bind(), table))
        except Exception:
            pass
    # Test doubles / synchronous binds: inspect directly.
    try:
        return _existing_columns_sync(db.get_bind(), table)
    except Exception:
        return None


async def delete_chunks_not_in_hashes(
    db: AsyncSession, document_id: uuid.UUID, hashes: Any
) -> int:
    """Delete this document's chunks whose content_hash is not in `hashes`.

    STRUCTURE_REPROCESS idempotency: content the new pass still produces is
    upserted in place by (document_id, content_hash); anything it no longer
    produces is removed in the same transaction.  Returns the deleted count.
    """
    columns = await _existing_columns(db, "chunks")
    if columns is None:
        return 0
    if "document_id" not in columns or "content_hash" not in columns:
        return 0
    wanted = {str(h) for h in (hashes or ()) if h}
    if not wanted:
        # An empty new set means the chunker produced nothing; wiping every
        # chunk would be data loss, so treat it as a failed/empty pass.
        return 0
    result = await db.execute(
        delete(Chunk).where(
            Chunk.document_id == document_id,
            Chunk.content_hash.notin_(wanted),
        )
    )
    return int(getattr(result, "rowcount", 0) or 0)


async def set_document_structure_verdict(
    db: AsyncSession,
    document_id: uuid.UUID,
    *,
    method: Optional[str],
    confidence: float,
    degraded: bool,
) -> None:
    """Stamp documents.structure_method/structure_confidence/structure_degraded.

    Raw SQL on purpose: those columns are additive (migration 014) and are
    deliberately not declared on the Document ORM mapping, so pre-014 rows
    and older models keep working.
    """
    columns = await _existing_columns(db, "documents")
    if columns is None:
        return
    required = ("structure_method", "structure_confidence",
                "structure_degraded")
    if any(name not in columns for name in required):
        return
    from sqlalchemy import text as _text

    await db.execute(
        _text(
            "UPDATE documents SET structure_method = :method, "
            "structure_confidence = :confidence, "
            "structure_degraded = :degraded "
            "WHERE id = :id"
        ),
        {
            "method": method,
            "confidence": confidence,
            "degraded": degraded,
            "id": document_id,
        },
    )


def _project_filter(project_id: Any) -> Optional[uuid.UUID]:
    """Normalize --project to a UUID, or None when unset/unparseable."""
    if not project_id:
        return None
    if isinstance(project_id, uuid.UUID):
        return project_id
    try:
        return uuid.UUID(str(project_id))
    except (ValueError, TypeError, AttributeError):
        return None


async def list_documents_for_reprocess(
    db: AsyncSession,
    *,
    project_id: Any = None,
    limit: Optional[int] = None,
    degraded_only: bool = True,
) -> List[Any]:
    """SELECT (never a write) the documents a structure re-chunk should cover.

    Predicate (ADR-008 backfill): ``structure_method IS NULL`` (never
    stamped), one of the legacy Conclusion/Introduction section labels, or
    an unfinished/failed indexing pass.  ``degraded_only=False`` relaxes it
    to every document in scope.  Pre-014 databases have no
    ``structure_method`` column and return ``[]`` instead of raising.

    Rows expose ``.id`` and ``.project_id`` (only those two columns are
    selected, so large ``extracted_text`` payloads are never fetched).
    """
    columns = await _existing_columns(db, "documents")
    if columns is None or "structure_method" not in columns:
        return []
    pid = _project_filter(project_id)
    if project_id and pid is None:
        return []

    from sqlalchemy import column as _column, or_ as _or

    conditions = []
    if degraded_only:
        structure = _column("structure_method")
        conditions.append(
            _or(
                structure.is_(None),
                structure.in_(("legacy_conclusion", "legacy_introduction")),
                Document.indexing_state.in_(("indexing", "partial", "failed")),
            )
        )
    if pid is not None:
        conditions.append(Document.project_id == pid)

    stmt = select(Document.id, Document.project_id).order_by(
        Document.created_at.asc()
    )
    if conditions:
        stmt = stmt.where(*conditions)
    if limit is not None:
        stmt = stmt.limit(int(limit))
    result = await db.execute(stmt)
    return list(result.all())


async def list_all_document_ids(
    db: AsyncSession,
    *,
    project_id: Any = None,
    limit: Optional[int] = None,
) -> List[uuid.UUID]:
    """Every document id in scope (``--all``), oldest first. Read-only."""
    return [
        row.id
        for row in await list_documents_for_reprocess(
            db, project_id=project_id, limit=limit, degraded_only=False
        )
    ]