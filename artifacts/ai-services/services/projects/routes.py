"""Project CRUD: POST/GET/GET{id}/PATCH/DELETE /api/projects.

All routes resolve identity from the JWT dependency and enforce ownership
through require_project_access (404 for another user's project).
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import require_project_access
from database import repositories
from database.engine import get_db
from database.models.project import Project
from database.models.user import User
from services.auth.deps import get_current_user

router = APIRouter(prefix="/api/projects", tags=["Projects"])


# ---------------------------------------------------------------------------
# Schemas (frontend contract: title/description/subject in, id out)
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    subject: Optional[str] = Field(default=None, max_length=255)

    def resolved_name(self) -> str:
        name = (self.title or self.name or "").strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="title (or name) is required",
            )
        return name


class ProjectUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    subject: Optional[str] = Field(default=None, max_length=255)


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    title: str = Field(alias="title")
    description: Optional[str] = None
    subject: Optional[str] = None
    status: str = "processing"
    progress: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]
    total: int
    page: int = 1
    page_size: int = 20


def _to_response(project: Project) -> ProjectResponse:
    raw_status = getattr(project, "status", "processing")
    status_value = getattr(raw_status, "value", raw_status)
    return ProjectResponse(
        id=project.id,
        title=project.name,
        description=project.description,
        subject=getattr(project, "subject", None),
        status=str(status_value),
        progress=int(project.progress or 0),
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def create_project(
    body: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await repositories.create_project(
        db=db,
        user_id=current_user.id,
        name=body.resolved_name(),
        description=body.description,
        subject=body.subject,
    )
    await db.commit()
    await db.refresh(project)
    return _to_response(project)


@router.get("", response_model=ProjectListResponse)
@router.get("/", response_model=ProjectListResponse, include_in_schema=False)
async def list_projects(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    projects, total = await repositories.list_projects_by_user(
        db, current_user.id, limit=page_size, offset=(page - 1) * page_size
    )
    return ProjectListResponse(
        projects=[_to_response(p) for p in projects],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project: Project = Depends(require_project_access)):
    return _to_response(project)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID,
    body: ProjectUpdate,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    name = (body.title or body.name)
    updated = await repositories.update_project(
        db,
        project.id,
        name=name.strip() if name and name.strip() else None,
        description=body.description,
        subject=body.subject,
    )
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    await db.commit()
    return _to_response(updated)


@router.delete("/{project_id}", status_code=status.HTTP_200_OK)
async def delete_project(
    project_id: uuid.UUID,
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    # Best-effort storage cleanup; DB delete is authoritative.
    storage_root: Optional[str] = None
    try:
        from config.settings import get_storage_config

        storage_root = get_storage_config().path
    except Exception:
        storage_root = None

    ok = await repositories.delete_project(db, project.id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    await db.commit()

    if storage_root:
        import os
        import shutil

        for candidate in (
            os.path.join(storage_root, str(project.user_id), str(project.id)),
            os.path.join(storage_root, str(project.id)),
        ):
            try:
                resolved = os.path.realpath(candidate)
                root = os.path.realpath(storage_root)
                if os.path.commonpath([resolved, root]) == root and os.path.isdir(resolved):
                    shutil.rmtree(resolved, ignore_errors=True)
            except Exception:
                pass

    return {"success": True, "message": "Project deleted successfully"}


@router.get("/{project_id}/documents")
async def list_project_documents(
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Documents of an owned project (summary shape; full detail lives in ingestion)."""
    documents = await repositories.get_documents_by_project(db, project.id)
    items = []
    for doc in documents:
        items.append(
            {
                "id": str(doc.id),
                "documentId": str(doc.id),
                "projectId": str(doc.project_id),
                "project_id": str(doc.project_id),
                "fileName": doc.file_name,
                "file_name": doc.file_name,
                "fileType": doc.file_type.value,
                "file_type": doc.file_type.value,
                "processingStatus": doc.processing_status.value,
                "processing_status": doc.processing_status.value,
                "pageCount": doc.page_count,
                "page_count": doc.page_count,
                "createdAt": doc.created_at.isoformat() if doc.created_at else None,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
            }
        )
    return {"documents": items, "total": len(items)}


@router.get("/{project_id}/flashcard-decks")
async def list_project_flashcard_decks(
    project: Project = Depends(require_project_access),
    db: AsyncSession = Depends(get_db),
):
    """Flashcard decks of an owned project; explicit empty state when none."""
    decks = await repositories.get_flashcard_decks_by_project(db, project.id)
    items = []
    for deck in decks:
        cards = getattr(deck, "flashcards", []) or []
        items.append(
            {
                "id": str(deck.id),
                "projectId": str(project.id),
                "name": deck.name,
                "cards": [
                    {
                        "id": str(c.id),
                        "front": c.front,
                        "back": c.back,
                        "difficulty": c.difficulty.value if hasattr(c.difficulty, "value") else c.difficulty,
                    }
                    for c in cards
                ],
                "cardCount": deck.card_count,
            }
        )
    if not items:
        return {"projectId": str(project.id), "state": "not_generated",
                "message": "No flashcards generated yet for this project.",
                "decks": [], "total": 0}
    return {"projectId": str(project.id), "state": "ready", "decks": items, "total": len(items)}


__all__ = ["router"]
