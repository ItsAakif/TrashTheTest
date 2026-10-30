"""S3-compatible storage backend (T108).

Used when STORAGE_BACKEND=s3. The client is injected (aioboto3-compatible)
so tests exercise this against a mocked client and prod passes a real one.
Traversal defense applies to keys on reads AND writes, same as LocalStorage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Optional

from .base import StorageBackend, StorageError, _check_hint, generate_stored_name


def _check_key(key: str) -> str:
    if not key or key.startswith(("/", "\\")) or ".." in Path(key).parts:
        raise StorageError(f"Unsafe storage key: {key!r}")
    return key.replace("\\", "/")


class S3Storage(StorageBackend):
    def __init__(self, client: Any, bucket: str, prefix: str = ""):
        if not bucket:
            raise StorageError("S3 bucket is required for S3Storage")
        self._client = client
        self._bucket = bucket
        self._prefix = prefix.strip("/")

    def _full_key(self, key: str) -> str:
        safe = _check_key(key)
        return f"{self._prefix}/{safe}" if self._prefix else safe

    def _new_key(self, key_hint: str) -> str:
        _check_hint(key_hint)
        rel = Path(key_hint)
        if len(rel.parts) < 3:
            raise StorageError(f"key_hint must scope user/project: {key_hint!r}")
        scope = f"{rel.parts[0]}/{rel.parts[1]}"
        _check_key(scope)  # traversal defense on the scope
        return f"{scope}/{generate_stored_name(rel.name)}"

    async def save(self, key_hint: str, data: bytes) -> str:
        key = self._new_key(key_hint)
        await self._client.put_object(Bucket=self._bucket, Key=self._full_key(key), Body=data)
        return key

    async def save_stream(
        self,
        key_hint: str,
        stream: BinaryIO,
        chunk_size: int = 64 * 1024,
        max_bytes: Optional[int] = None,
    ) -> tuple[str, int, str]:
        import hashlib

        key = self._new_key(key_hint)
        digest = hashlib.sha256()
        total = 0
        parts: list[bytes] = []
        # Memory bound: accumulate only up to the caller's max (uploads are
        # size-limited by validation before this path); S3 multipart resume
        # is a documented follow-up, not claimed here.
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise StorageError(f"Upload exceeds limit ({max_bytes} bytes)")
            digest.update(chunk)
            parts.append(bytes(chunk))
        body = b"".join(parts)
        await self._client.put_object(Bucket=self._bucket, Key=self._full_key(key), Body=body)
        return key, total, digest.hexdigest()

    async def load(self, key: str) -> bytes:
        resp = await self._client.get_object(Bucket=self._bucket, Key=self._full_key(key))
        body = resp.get("Body", resp)
        if hasattr(body, "read"):
            data = await body.read() if callable(getattr(body.read, "__await__", None)) else body.read()
            return data if isinstance(data, bytes) else bytes(data)
        return bytes(body)

    async def open_stream(self, key: str, chunk_size: int = 64 * 1024) -> AsyncIterator[bytes]:
        data = await self.load(key)
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    async def delete(self, key: str) -> bool:
        try:
            full = self._full_key(key)
        except StorageError:
            return False
        try:
            await self._client.delete_object(Bucket=self._bucket, Key=full)
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        try:
            _check_key(key)
        except StorageError:
            return False
        try:
            await self._client.head_object(Bucket=self._bucket, Key=self._full_key(key))
            return True
        except Exception:
            return False

    async def size(self, key: str) -> int:
        resp = await self._client.head_object(Bucket=self._bucket, Key=self._full_key(key))
        return int(resp.get("ContentLength", 0))


__all__ = ["S3Storage"]
