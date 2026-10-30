"""T108: StorageBackend - LocalStorage default + S3 (mocked), traversal defense."""

import io
import uuid

import pytest

from services.storage import (
    LocalStorage,
    S3Storage,
    StorageError,
    StorageManager,
    generate_stored_name,
)

pytestmark = pytest.mark.asyncio


def _uid():
    return uuid.uuid4()


async def test_save_load_delete_round_trip_preserves_bytes(tmp_path):
    store = LocalStorage(tmp_path / "root")
    key = await store.save(f"{_uid()}/{_uid()}/report.pdf", b"%PDF-1.4 hello")
    assert await store.exists(key)
    assert await store.size(key) == len(b"%PDF-1.4 hello")
    assert await store.load(key) == b"%PDF-1.4 hello"
    chunks = [c async for c in store.open_stream(key, chunk_size=4)]
    assert b"".join(chunks) == b"%PDF-1.4 hello"
    assert await store.delete(key) is True
    assert await store.exists(key) is False


async def test_stored_names_are_server_generated(tmp_path):
    store = LocalStorage(tmp_path / "root")
    user, project = _uid(), _uid()
    first = await store.save(f"{user}/{project}/report.pdf", b"a")
    second = await store.save(f"{user}/{project}/report.pdf", b"b")
    assert first != second  # same original name -> distinct stored keys
    from pathlib import Path as _Path

    assert _Path(first).name != "report.pdf"  # stored name is server-generated
    assert first.startswith(f"{user}/{project}/")


async def test_traversal_rejected_on_write_and_read(tmp_path):
    store = LocalStorage(tmp_path / "root")
    user, project = _uid(), _uid()
    with pytest.raises(StorageError):
        await store.save(f"{user}/{project}/../../evil.pdf", b"x")
    with pytest.raises(StorageError):
        await store.save("/absolute/evil.pdf", b"x")
    for bad in ("../evil.pdf", "../../etc/passwd", "/absolute/key", "..\\evil"):
        with pytest.raises(StorageError):
            await store.load(bad)
        assert await store.exists(bad) is False
        assert await store.delete(bad) is False


async def test_save_stream_reports_checksum_and_enforces_limit(tmp_path):
    store = LocalStorage(tmp_path / "root")
    stream = io.BytesIO(b"B" * 100 + b"C" * 100)
    key, total, checksum = await store.save_stream(
        f"{_uid()}/{_uid()}/big.pdf", stream, chunk_size=32
    )
    assert total == 200
    import hashlib

    assert checksum == hashlib.sha256(b"B" * 100 + b"C" * 100).hexdigest()
    assert await store.load(key) == b"B" * 100 + b"C" * 100


async def test_save_stream_over_limit_leaves_no_partial_file(tmp_path):
    store = LocalStorage(tmp_path / "root")
    stream = io.BytesIO(b"Z" * 1024)
    with pytest.raises(StorageError):
        await store.save_stream(f"{_uid()}/{_uid()}/big.pdf", stream, chunk_size=64, max_bytes=100)
    remaining = list((tmp_path / "root").rglob("*"))
    assert [p for p in remaining if p.is_file()] == []


class _FakeS3Client:
    """Minimal aioboto3-compatible in-memory mock."""

    def __init__(self):
        self.objects = {}

    async def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    async def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    async def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    async def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)
        return {}


async def test_s3_backend_round_trip_against_mocked_client():
    client = _FakeS3Client()
    store = S3Storage(client, "test-bucket")
    key = await store.save(f"{_uid()}/{_uid()}/doc.pdf", b"%PDF data")
    assert await store.exists(key)
    assert await store.size(key) == len(b"%PDF data")
    assert await store.load(key) == b"%PDF data"
    assert [c async for c in store.open_stream(key)] == [b"%PDF data"]
    assert await store.delete(key) is True
    assert await store.exists(key) is False
    with pytest.raises(StorageError):
        await store.save(f"{_uid()}/{_uid()}/../../evil.pdf", b"x")
    with pytest.raises(StorageError):
        await store.load("../evil.pdf")


async def test_storage_manager_shim_preserves_behavior(tmp_path):
    manager = StorageManager(base_path=str(tmp_path / "legacy"))
    user, project = _uid(), _uid()
    key = await manager.save_file(user, project, b"bytes", "notes.pdf")
    assert await manager.file_exists(key)
    assert await manager.get_file_size(key) == 5
    assert await manager.delete_file(key) is True
    # get_absolute_path stays inside the root.
    assert str(manager.get_absolute_path(key)).startswith(str(tmp_path))


async def test_generate_stored_name_never_uses_raw_filename():
    name = generate_stored_name("../../etc/passwd.pdf")
    assert ".." not in name and "/" not in name
    assert name.endswith(".pdf")
