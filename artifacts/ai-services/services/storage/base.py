"""StorageBackend interface + LocalStorage + S3-compatible backend (T108).

Traversal defense lives HERE (resolve + assert-inside-root on read AND
write) so no caller can forget it. Stored filenames are server-generated;
the original filename is stored only as a DB string, never as a path
component. Raw filesystem paths are never exposed to callers - only opaque
storage keys (relative posix paths under the storage root).
"""

from __future__ import annotations

import abc
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Optional


class StorageError(Exception):
    """Raised for any storage failure (including traversal attempts)."""


def _check_hint(key_hint: str) -> None:
    """Fail fast on traversal attempts in caller-supplied hints."""
    if not key_hint or key_hint.startswith(("/", "\\")) or ".." in Path(key_hint).parts:
        raise StorageError(f"Unsafe storage key: {key_hint!r}")


def _assert_inside_root(root: Path, candidate: Path) -> Path:
    """Resolve `candidate` and assert it stays inside `root`.

    Missing trailing parts are appended to the nearest existing ancestor so
    the full relative structure is preserved (never collapsed).
    """
    resolved_root = root.resolve()
    cand = candidate
    missing: list[str] = []
    while not cand.exists():
        missing.append(cand.name)
        parent = cand.parent
        if parent == cand:  # pragma: no cover - filesystem root guard
            break
        cand = parent
    resolved = cand.resolve()
    for part in reversed(missing):
        resolved = resolved / part
    # Robust containment check (py3.8 compatible).
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise StorageError(f"Path escapes storage root: {candidate}")
    # Extra guard for symlinked edge cases.
    if os.path.commonpath([str(resolved), str(resolved_root)]) != str(resolved_root):
        raise StorageError(f"Path escapes storage root: {candidate}")
    return resolved


def generate_stored_name(original_filename: str) -> str:
    """Server-generated stored name; original name contributes only a safe stem."""
    suffix = Path(original_filename or "file").suffix.lower()[:16]
    # Keep only alphanumerics from the stem so nothing user-controlled
    # reaches the filesystem as a special name.
    stem = "".join(c for c in Path(original_filename or "file").stem if c.isalnum() or c in ("-", "_"))[:48] or "file"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}_{stem}{suffix}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class StorageBackend(abc.ABC):
    """Abstract storage backend. Keys are opaque, never raw paths."""

    @abc.abstractmethod
    async def save(self, key_hint: str, data: bytes) -> str:
        """Save bytes under a server-generated key; return the opaque key."""
        raise NotImplementedError

    @abc.abstractmethod
    async def save_stream(
        self,
        key_hint: str,
        stream: BinaryIO,
        chunk_size: int = 64 * 1024,
        max_bytes: Optional[int] = None,
    ) -> tuple[str, int, str]:
        """Stream bytes to storage. Returns (key, total_bytes, sha256)."""
        raise NotImplementedError

    @abc.abstractmethod
    async def load(self, key: str) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    async def open_stream(self, key: str, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
        """Async generator yielding file bytes in bounded chunks."""
        raise NotImplementedError  # pragma: no cover - interface
        yield b""  # type: ignore[unreachable]

    @abc.abstractmethod
    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def size(self, key: str) -> int:
        raise NotImplementedError
