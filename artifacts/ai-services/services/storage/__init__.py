"""Storage package: backend factory + legacy StorageManager shim (T108)."""

from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Optional

from .base import StorageBackend, StorageError, generate_stored_name, sha256_bytes
from .local import LocalStorage
from .s3 import S3Storage


def get_storage_backend() -> StorageBackend:
    """Build the configured backend. Local needs no credentials."""
    from config.settings import get_storage_config

    cfg = get_storage_config()
    if (cfg.backend or "local").lower() == "s3":
        try:
            import aioboto3  # type: ignore[import-not-found]
        except ImportError as exc:
            raise StorageError(
                "STORAGE_BACKEND=s3 requires aioboto3 (pip install aioboto3). "
                "One-line justification (C5): S3-compatible object storage for prod."
            ) from exc
        session = aioboto3.Session()
        client = session.client(
            "s3",
            endpoint_url=cfg.s3_endpoint,
            region_name=cfg.s3_region,
            aws_access_key_id=cfg.s3_access_key,
            aws_secret_access_key=cfg.s3_secret_key,
        )
        if not cfg.s3_bucket:
            raise StorageError("S3_BUCKET is required when STORAGE_BACKEND=s3")
        return S3Storage(client, cfg.s3_bucket)
    return LocalStorage(cfg.path)


@lru_cache(maxsize=1)
def _default_backend() -> StorageBackend:
    from config.settings import get_storage_config

    return LocalStorage(get_storage_config().path)


def get_default_backend() -> StorageBackend:
    """Process-wide default backend (local). Tests inject their own roots."""
    return _default_backend()


class StorageManager:
    """Backward-compatible facade over StorageBackend (preserved per NON-GOALS).

    Existing callers (ingestion routes pre-T110, project delete) keep working;
    new code should use StorageBackend directly.
    """

    def __init__(self, base_path: Optional[str] = None):
        import os as _os

        if base_path is None:
            try:
                from config.settings import get_storage_config

                root = get_storage_config().path
            except Exception:
                # Import-time safe: storage must not hard-require JWT_SECRET.
                root = _os.getenv("STORAGE_PATH", "./storage")
        else:
            root = base_path
        self.base_path = Path(root)
        self._backend = LocalStorage(self.base_path)

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    async def save_file(self, user_id: uuid.UUID, project_id: uuid.UUID, file_data: bytes, original_filename: str) -> str:
        return await self._backend.save(f"{user_id}/{project_id}/{original_filename}", file_data)

    async def save_stream(self, user_id: uuid.UUID, project_id: uuid.UUID, original_filename: str, file_stream: BinaryIO, chunk_size: int = 64 * 1024) -> tuple[str, int]:
        key, total, _ = await self._backend.save_stream(
            f"{user_id}/{project_id}/{original_filename}", file_stream, chunk_size=chunk_size
        )
        return key, total

    async def delete_file(self, relative_path: str) -> bool:
        return await self._backend.delete(relative_path)

    def get_absolute_path(self, relative_path: str) -> Path:
        # Never expose raw paths to API callers; internal use only.
        candidate = self.base_path / relative_path
        from .base import _assert_inside_root

        return _assert_inside_root(self.base_path, candidate)

    async def file_exists(self, relative_path: str) -> bool:
        return await self._backend.exists(relative_path)

    async def get_file_size(self, relative_path: str) -> int:
        return await self._backend.size(relative_path)

    async def cleanup_project_files(self, user_id: uuid.UUID, project_id: uuid.UUID) -> int:
        backend = self._backend
        if isinstance(backend, LocalStorage):
            return await backend.cleanup_project(user_id, project_id)
        return 0


storage_manager = StorageManager()


__all__ = [
    "StorageBackend",
    "StorageError",
    "LocalStorage",
    "S3Storage",
    "generate_stored_name",
    "sha256_bytes",
    "get_storage_backend",
    "get_default_backend",
    "StorageManager",
    "storage_manager",
]
