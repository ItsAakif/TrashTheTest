"""Pytest bootstrap: deterministic env BEFORE any app import.

config.settings fail-fasts at import when JWT_SECRET/DATABASE_URL are
missing, so tests set throwaway values here (never real secrets).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JWT_SECRET", "test-secret-at-least-32-chars-long-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")
