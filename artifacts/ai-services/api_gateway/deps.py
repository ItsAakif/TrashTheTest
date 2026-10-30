"""Ownership dependencies: single reusable enforcement point.

Every project-scoped route must use require_project_access (or
require_document_access built on top of it) instead of inline owner
comparisons. Another user's resource returns 404, never 200 (and never a
403 that confirms existence to an attacker).
"""

import uuid
from typing import Tuple

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import repositories
from database.engine import get_db
from database.models.document import Document
from database.models.project import Project
from database.models.user import User
from services.auth.deps import get_current_user


def _not_found(detail: str = "Project not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _parse_uuid(value: str, label: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {label} format",
        )


async def require_project_access(
    project_id: uuid.UUID | str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Project:
    """Return the Project when owned by the caller, else 404."""
    pid = project_id if isinstance(project_id, uuid.UUID) else _parse_uuid(project_id, "project_id")
    project = await repositories.get_project_by_id(db, pid)
    if project is None or project.user_id != current_user.id:
        raise _not_found()
    return project


async def require_document_access(
    document_id: uuid.UUID | str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Tuple[Document, Project]:
    """Return (Document, Project) when the document's project is owned, else 404."""
    did = document_id if isinstance(document_id, uuid.UUID) else _parse_uuid(document_id, "document_id")
    document = await repositories.get_document_by_id(db, did)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    project = await repositories.get_project_by_id(db, document.project_id)
    if project is None or project.user_id != current_user.id:
        raise _not_found("Document not found")
    return document, project


__all__ = ["require_project_access", "require_document_access"]
