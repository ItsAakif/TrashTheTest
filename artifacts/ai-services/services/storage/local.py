"""LocalStorage backend (default; no extra service needed) (T108).

Layout: {root}/{user_id}/{project_id}/{server_generated_name}
Traversal defense inherited via _assert_inside_root on reads AND writes.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Optional

import aiofiles
import aiofiles.os

from .base import (
    StorageBackend,
    StorageError,
    _assert_inside_root,
    _check_hint,
    generate_stored_name,
)


class LocalStorage(StorageBackend):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- internal helpers -------------------------------------------------
    def _scoped_dir(self, user_id: uuid.UUID | str, project_id: uuid.UUID | str) -> Path:
        candidate = self.root / str(user_id) / str(project_id)
        resolved = _assert_inside_root(self.root, candidate)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _resolve_key(self, key: str) -> Path:
        if not key or key.startswith(("/", "\\")) or ".." in Path(key).parts:
            # Fast reject before touching the filesystem; the resolve check
            # below is the authoritative defense.
            if key.startswith(("/", "\\")) or ".." in Path(key).parts:
                raise StorageError(f"Unsafe storage key: {key!r}")
        candidate = self.root / key
        return _assert_inside_root(self.root, candidate)

    # -- interface ----------------------------------------------------------
    async def save(self, key_hint: str, data: bytes) -> str:
        _check_hint(key_hint)
        rel = Path(key_hint)
        if len(rel.parts) >= 3:
            user_id, project_id = rel.parts[0], rel.parts[1]
        else:  # pragma: no cover - defensive; callers pass user/project scope
            raise StorageError(f"key_hint must scope user/project: {key_hint!r}")
        try:
            uuid.UUID(str(user_id))
            uuid.UUID(str(project_id))
        except ValueError:
            raise StorageError(f"key_hint scope must be UUIDs: {key_hint!r}")
        directory = self._scoped_dir(user_id, project_id)
        name = generate_stored_name(Path(key_hint).name)
        dest = _assert_inside_root(self.root, directory / name)
        async with aiofiles.open(dest, "wb") as f:
            await f.write(data)
        return str(dest.relative_to(self.root).as_posix())

    async def save_stream(
        self,
        key_hint: str,
        stream: BinaryIO,
        chunk_size: int = 64 * 1024,
        max_bytes: Optional[int] = None,
    ) -> tuple[str, int, str]:
        _check_hint(key_hint)
        rel = Path(key_hint)
        if len(rel.parts) < 3:
            raise StorageError(f"key_hint must scope user/project: {key_hint!r}")
        directory = self._scoped_dir(rel.parts[0], rel.parts[1])
        name = generate_stored_name(Path(key_hint).name)
        dest = _assert_inside_root(self.root, directory / name)
        digest = hashlib.sha256()
        total = 0
        try:
            async with aiofiles.open(dest, "wb") as f:
                while True:
                    chunk = stream.read(chunk_size)
                    if isinstance(chunk, bytes) and not chunk:
                        break
                    if chunk is None:
                        break
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise StorageError("Stream yielded non-bytes chunk")
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise StorageError(f"Upload exceeds limit ({max_bytes} bytes)")
                    digest.update(chunk)
                    await f.write(chunk)
        except Exception:
            try:
                if dest.exists():
                    await aiofiles.os.remove(dest)
            except Exception:
                pass
            raise
        return str(dest.relative_to(self.root).as_posix()), total, digest.hexdigest()

    async def load(self, key: str) -> bytes:
        dest = self._resolve_key(key)
        try:
            async with aiofiles.open(dest, "rb") as f:
                return await f.read()
        except FileNotFoundError:
            raise StorageError(f"Key not found: {key!r}")

    async def open_stream(self, key: str, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
        dest = self._resolve_key(key)
        try:
            async with aiofiles.open(dest, "rb") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except FileNotFoundError:
            raise StorageError(f"Key not found: {key!r}")

    async def delete(self, key: str) -> bool:
        try:
            dest = self._resolve_key(key)
        except StorageError:
            return False
        try:
            if dest.exists():
                await aiofiles.os.remove(dest)
                return True
        except Exception:
            pass
        return False

    async def exists(self, key: str) -> bool:
        try:
            dest = self._resolve_key(key)
        except StorageError:
            return False
        return dest.exists()

    async def size(self, key: str) -> int:
        dest = self._resolve_key(key)
        if dest.exists():
            return (await aiofiles.os.stat(dest)).st_size
        return 0

    async def cleanup_project(self, user_id: uuid.UUID | str, project_id: uuid.UUID | str) -> int:
        """Remove the storage directory for a project. Returns files deleted."""
        candidate = self.root / str(user_id) / str(project_id)
        try:
            resolved = _assert_inside_root(self.root, candidate)
        except StorageError:
            return 0
        count = 0
        if resolved.exists() and resolved.is_dir():
            for child in resolved.iterdir():
                if child.is_file():
                    try:
                        await aiofiles.os.remove(child)
                        count += 1
                    except Exception:
                        pass
            try:
                resolved.rmdir()
            except Exception:
                pass
        return count


__all__ = ["LocalStorage"]
