"""Auth package: JWT issue/verify, password hashing, FastAPI dependencies."""

from .service import (
    hash_password,
    verify_password,
    needs_rehash,
    create_access_token,
    create_refresh_token,
    decode_token,
    authenticate_user,
    register_user,
    refresh_access_token,
    revoke_all_tokens,
    change_password,
    AuthError,
)
from .deps import get_current_user, get_optional_user, require_user

__all__ = [
    "hash_password",
    "verify_password",
    "needs_rehash",
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "authenticate_user",
    "register_user",
    "refresh_access_token",
    "revoke_all_tokens",
    "change_password",
    "AuthError",
    "get_current_user",
    "get_optional_user",
    "require_user",
]
