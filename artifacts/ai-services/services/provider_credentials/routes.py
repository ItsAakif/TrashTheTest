"""Settings > AI Provider endpoints (user credential store).

- GET    /api/settings/providers/status        masked per-provider status
- GET    /api/settings/providers/models        dropdown catalog (+search)
- PUT    /api/settings/providers               save/replace {provider, api_key, model}
- POST   /api/settings/providers/test          live test (saved or supplied key)
- DELETE /api/settings/providers/{provider}    remove
- PUT    /api/settings/providers/project/{project_id}    set project override
- DELETE /api/settings/providers/project/{project_id}    clear project override

Raw keys are accepted inbound in POST/PUT bodies only and are never
returned, logged, cached, or placed in URLs.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api_gateway.deps import require_project_access
from database.engine import get_db
from database.models.user import User
from services.auth.deps import get_current_user

from . import service as creds
from .catalog import DEFAULT_LIMIT, get_catalog, get_live_catalog

router = APIRouter(prefix="/api/settings/providers", tags=["provider-settings"])


class SaveProviderRequest(BaseModel):
    provider: str
    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1)


class TestProviderRequest(BaseModel):
    provider: str
    # Either test the saved credential (use_saved=True) or a supplied
    # key/model pair (never persisted by this endpoint).
    use_saved: bool = False
    api_key: Optional[str] = None
    model: Optional[str] = None


class ProjectOverrideRequest(BaseModel):
    provider: str
    model: str = Field(min_length=1)


@router.get("/status")
async def provider_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await creds.get_status(db, current_user)


@router.get("/models")
async def provider_models(
    provider: Optional[str] = None,
    search: Optional[str] = None,
    q: Optional[str] = None,
    live: bool = False,
    refresh: bool = False,
    limit: int = DEFAULT_LIMIT,
    current_user: User = Depends(get_current_user),
):
    """Model catalog: curated static by default, live provider lists on opt-in.

    - `provider`: groq | openrouter (default: both).
    - `search` (alias `q`): case-insensitive filter across id + name.
    - `live=true`: fetch live /models server-side (1h cache, static
      fallback). Auth is by the caller's JWT; the Groq fetch uses the
      server app key and user keys are never used here.
    - `limit`: max entries per provider (default 100, max 500).
    - `refresh=true`: bypass the live cache for this call.
    Never returns keys.
    """
    if provider is not None:
        try:
            from .catalog import validate_provider

            validate_provider(provider)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail=str(exc))
    needle = search if search is not None else q
    if live:
        return await get_live_catalog(
            provider=provider, search=needle, limit=limit, refresh=refresh)
    return get_catalog(provider=provider, search=needle)


@router.put("")
async def save_provider(
    request: SaveProviderRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        saved = await creds.save_credential(
            db, current_user, request.provider, request.api_key, request.model
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    try:
        await db.commit()
    except AttributeError:
        pass
    # Masked only (never the raw key).
    return {"saved": True, **saved}


@router.post("/test")
async def test_provider(
    request: TestProviderRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from .catalog import validate_model_id, validate_provider

    try:
        provider = validate_provider(request.provider)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if request.use_saved:
        row = await creds._find_credential(db, current_user.id, provider)
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail=f"No saved {provider} key. Save one first.")
        from .crypto import decrypt_key

        try:
            api_key = decrypt_key(row.key_encrypted)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        model = row.model
    else:
        if not (request.api_key or "").strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="API key is required (or use_saved=true).")
        api_key = request.api_key or ""
        try:
            model = validate_model_id(request.model or "")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        return await creds.test_connection(provider, api_key, model)
    except creds.ProviderTestError as exc:
        raise HTTPException(status_code=exc.status,
                            detail={"code": exc.code, "message": exc.message})


@router.delete("/{provider}")
async def remove_provider(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        removed = await creds.remove_credential(db, current_user, provider)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"No saved {provider} key.")
    try:
        await db.commit()
    except AttributeError:
        pass
    return {"removed": True, "provider": provider}


@router.put("/project/{project_id}")
async def set_project_provider_override(
    project_id: str,
    request: ProjectOverrideRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await require_project_access(project_id, current_user, db)
    try:
        override = await creds.set_project_override(db, project, request.provider, request.model)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    try:
        await db.commit()
    except AttributeError:
        pass
    return {"project_id": str(project.id), "override": override}


@router.delete("/project/{project_id}")
async def clear_project_provider_override(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await require_project_access(project_id, current_user, db)
    await creds.clear_project_override(db, project)
    try:
        await db.commit()
    except AttributeError:
        pass
    return {"project_id": str(project.id), "override": None}


__all__ = ["router"]
