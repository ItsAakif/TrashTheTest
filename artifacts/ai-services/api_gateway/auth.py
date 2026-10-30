"""Authentication entry point (compatibility shim).

The legacy demo path that accepted any bearer token and the header-based
demo bypass are REMOVED, along with the hardcoded demo user. The canonical
implementation lives in services.auth; this module re-exports the real
dependencies so older imports keep working.
"""

from services.auth.deps import get_current_user, get_optional_user, require_user
from services.auth.routes import router
from services.auth.service import AuthError

# Backwards-compatible aliases used across the codebase.
optional_user = get_optional_user
require_user_dep = require_user

__all__ = [
    "AuthError",
    "get_current_user",
    "get_optional_user",
    "optional_user",
    "require_user",
    "require_user_dep",
    "router",
]
