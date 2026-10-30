"""FastAPI auth dependencies: real JWT, works over ASGI (no request.state)."""

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_db
from database.models.user import User

from .service import AuthError, user_from_access_token

_bearer = HTTPBearer(auto_error=False)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Resolve identity from the JWT dependency. 401 when absent/invalid."""
    if credentials is None or not credentials.credentials:
        raise _unauthorized()
    try:
        return await user_from_access_token(db, credentials.credentials)
    except AuthError:
        raise _unauthorized()


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """Return the user when a valid token is present, else None.

    Reads the Authorization header directly (never request.state) so it
    also works in tests via ASGI transport.
    """
    if credentials is None or not credentials.credentials:
        return None
    try:
        return await user_from_access_token(db, credentials.credentials)
    except AuthError:
        return None


async def require_user(current: User = Depends(get_current_user)) -> User:
    """Route-level dependency alias for protected routes."""
    return current


__all__ = ["get_current_user", "get_optional_user", "require_user"]
