"""Auth routes: register, login, refresh, logout, me, change-password."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from database.models.user import User

from .deps import get_current_user
from .service import (
    AuthError,
    authenticate_user,
    change_password,
    issue_token_pair,
    refresh_access_token,
    register_user,
    revoke_all_tokens,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: Optional[str] = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    name: Optional[str] = None
    created_at: Optional[datetime] = None


class TokenPairResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    user: UserResponse


def _user_response(user: User) -> UserResponse:
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        created_at=user.created_at,
    )


@router.post("/register", response_model=TokenPairResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    try:
        user = await register_user(db, body.email, body.password, body.name)
    except AuthError as exc:
        if str(exc) == "exists":
            # 409 without revealing more than necessary.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists",
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Registration failed"
        )
    pair = issue_token_pair(user.id, int(getattr(user, "token_version", 0) or 0))
    return TokenPairResponse(
        access_token=pair["access_token"],
        refresh_token=pair["refresh_token"],
        user=_user_response(user),
    )


@router.post("/login", response_model=TokenPairResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    try:
        user, pair = await authenticate_user(db, body.email, body.password)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        )
    return TokenPairResponse(
        access_token=pair["access_token"],
        refresh_token=pair["refresh_token"],
        user=_user_response(user),
    )


@router.post("/refresh")
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await refresh_access_token(db, body.refresh_token)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        )


@router.post("/logout")
async def logout(
    current: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    # Bump token_version: all previously issued tokens stop verifying.
    await revoke_all_tokens(db, current)
    return {"success": True, "message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def me(current: User = Depends(get_current_user)):
    return _user_response(current)


@router.post("/change-password")
async def change_password_route(
    body: ChangePasswordRequest,
    current: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await change_password(db, current, body.current_password, body.new_password)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        )
    return {"success": True, "message": "Password changed successfully"}


__all__ = ["router"]
