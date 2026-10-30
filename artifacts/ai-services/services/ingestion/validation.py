"""Document validation: magic-byte sniffing + incremental size limits (T109).

Content signature is authoritative; the extension allowlist is a second
check. SVG and unlisted archive types are rejected. Size is enforced WHILE
reading (incremental byte counter), never after buffering the whole file.
"""

import os
import uuid
from typing import AsyncIterator, Optional

from fastapi import UploadFile, status


# Allowed file extensions and their MIME types
ALLOWED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

# Allowed file types (for easy lookup)
ALLOWED_FILE_TYPES = {
    "pdf", "docx", "doc", "txt",
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif",
}

# Explicitly rejected even if sniffable: vector XML with script surface.
REJECTED_EXTENSIONS = {".svg", ".svgz", ".html", ".htm", ".js", ".exe", ".sh", ".bat", ".zip", ".rar", ".7z"}

EXTENSION_TO_MIME = dict(ALLOWED_EXTENSIONS)

# Maximum file size (100MB by default)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))

# Maximum file size for standard upload (10MB - larger files use streaming)
MAX_STANDARD_UPLOAD_SIZE = int(os.getenv("MAX_STANDARD_UPLOAD_SIZE", 10 * 1024 * 1024))

# Chunk size for streaming uploads (64KB).
# Memory bound: at most ~2 * STREAM_CHUNK_SIZE bytes are ever held for
# sniffing/counting before bytes are handed to storage (<150 MB above
# baseline by orders of magnitude; asserted in tests).
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", 64 * 1024))


class ValidationError(Exception):
    """Custom validation error"""
    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ---------------------------------------------------------------------------
# Magic-byte sniffing
# ---------------------------------------------------------------------------

def sniff_mime(head: bytes) -> Optional[str]:
    """Return the detected MIME type from leading bytes, or None if unknown."""
    if head.startswith(b"%PDF"):
        return "application/pdf"
    if head.startswith(b"\x50\x4B\x03\x04") or head.startswith(b"\x50\x4B\x05\x06") or head.startswith(b"\x50\x4B\x07\x08"):
        return "application/zip"  # docx is a zip container; refined below
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head.startswith(b"GIF87a") or head.startswith(b"GIF89a"):
        return "image/gif"
    if head.startswith(b"BM"):
        return "image/bmp"
    if head.startswith(b"II*\x00") or head.startswith(b"MM\x00*"):
        return "image/tiff"
    # Plain text heuristic: no NUL bytes and mostly printable in the head.
    if head and b"\x00" not in head[:8192]:
        sample = head[:8192]
        try:
            text = sample.decode("utf-8")
        except UnicodeDecodeError:
            return None
        printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
        if len(text) > 0 and printable / len(text) >= 0.85:
            return "text/plain"
    return None


def sniff_extension(head: bytes, filename: str) -> Optional[str]:
    """Map sniffed bytes to a canonical extension (None when unrecognized)."""
    mime = sniff_mime(head)
    if mime == "application/pdf":
        return ".pdf"
    if mime == "application/zip":
        # Only docx containers are accepted; anything else zipped is rejected.
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".docx":
            return ".docx"
        return None
    if mime == "image/png":
        return ".png"
    if mime == "image/jpeg":
        ext = os.path.splitext(filename)[1].lower()
        return ".jpg" if ext in (".jpg", ".jpeg") else ".jpeg"
    if mime == "image/gif":
        return ".gif"
    if mime == "image/bmp":
        return ".bmp"
    if mime == "image/tiff":
        ext = os.path.splitext(filename)[1].lower()
        return ".tif" if ext == ".tif" else ".tiff"
    if mime == "text/plain":
        return ".txt"
    # Legacy .doc (OLE compound document) signature.
    if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return ".doc"
    return None


def validate_file_type(filename: str) -> str:
    """Validate file extension is allowed (second check; content is first)."""
    ext = os.path.splitext(filename or "")[1].lower()
    if not ext:
        raise ValidationError("Filename must include an extension", status.HTTP_400_BAD_REQUEST)
    if ext in REJECTED_EXTENSIONS:
        raise ValidationError(
            f"File type '{ext}' is not accepted",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"File type '{ext}' not allowed. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS.keys()))}",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )
    # Reject path components smuggled in the filename (stored name is
    # server-generated anyway; this fails fast with a 400).
    if ".." in (filename or "") or "/" in (filename or "") or "\\" in (filename or ""):
        raise ValidationError("Filename must not contain path components", status.HTTP_400_BAD_REQUEST)
    return ext


def validate_magic_bytes(head: bytes, filename: str) -> str:
    """Validate sniffed content agrees with the declared extension.

    Returns the canonical extension. An executable renamed to .pdf (MZ
    header) or any mismatch raises 415.
    """
    declared = os.path.splitext(filename or "")[1].lower()
    sniffed = sniff_extension(head, filename)
    if sniffed is None:
        raise ValidationError(
            "File content not recognized (magic-byte mismatch)",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )
    # Normalize jpg/jpeg equivalence.
    pair = {declared, sniffed}
    if pair <= {".jpg", ".jpeg"} or pair <= {".tif", ".tiff"}:
        return declared if declared in ALLOWED_EXTENSIONS else sniffed
    if declared != sniffed:
        # A valid PDF renamed .txt (or vice versa) is rejected either way.
        raise ValidationError(
            f"File content ({sniffed}) does not match extension ({declared or 'none'})",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )
    return sniffed


def validate_file_size(size: int, allow_large: bool = False) -> None:
    max_size = MAX_FILE_SIZE if allow_large else MAX_STANDARD_UPLOAD_SIZE
    if size > max_size:
        max_mb = max_size / (1024 * 1024)
        raise ValidationError(
            f"File too large. Maximum size: {max_mb:.1f}MB",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )


async def read_upload_in_chunks(
    file: UploadFile,
    allow_large: bool = False,
    chunk_size: int = STREAM_CHUNK_SIZE,
) -> AsyncIterator[bytes]:
    """Yield upload bytes in bounded chunks, enforcing the size limit inline.

    Memory bound: never holds more than one chunk (~64KB) plus the small
    sniff head; callers stream each chunk to storage immediately.
    """
    max_size = MAX_FILE_SIZE if allow_large else MAX_STANDARD_UPLOAD_SIZE
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise ValidationError(
                f"File too large. Maximum size: {max_size / (1024 * 1024):.1f}MB",
                status.HTTP_413_CONTENT_TOO_LARGE,
            )
        yield chunk


async def validate_upload_file(file: UploadFile, allow_large: bool = False) -> tuple[str, int]:
    """Pre-validate header-declared upload (extension + declared size).

    Content sniffing happens on the first real bytes at the call site via
    validate_magic_bytes; this keeps the header fast-path honest without
    trusting it.
    """
    if not file.filename:
        raise ValidationError("No filename provided", status.HTTP_400_BAD_REQUEST)
    ext = validate_file_type(file.filename)
    content_length = (file.headers or {}).get("content-length")
    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            size = 0
        else:
            validate_file_size(size, allow_large)
            return ext, size
    return ext, 0


def get_file_type_category(file_extension: str) -> str:
    ext = file_extension.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext == ".pdf":
        return "pdf"
    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"):
        return "image"
    elif ext in (".docx", ".doc"):
        return "document"
    elif ext == ".txt":
        return "text"
    else:
        return "unknown"


def requires_ocr(file_extension: str) -> bool:
    category = get_file_type_category(file_extension)
    return category in ("image", "pdf")


async def validate_project_ownership(project_id: uuid.UUID, user_id: uuid.UUID, db_session) -> bool:
    """Legacy helper (kept for compat); T110 routes use require_*_access."""
    from database.repositories import get_project_by_id

    project = await get_project_by_id(db_session, project_id)
    if not project:
        raise ValidationError("Project not found", status.HTTP_404_NOT_FOUND)
    if project.user_id != user_id:
        raise ValidationError("Project not found", status.HTTP_404_NOT_FOUND)
    return True


__all__ = [
    "ValidationError",
    "ALLOWED_EXTENSIONS",
    "ALLOWED_FILE_TYPES",
    "REJECTED_EXTENSIONS",
    "EXTENSION_TO_MIME",
    "MAX_FILE_SIZE",
    "MAX_STANDARD_UPLOAD_SIZE",
    "STREAM_CHUNK_SIZE",
    "sniff_mime",
    "sniff_extension",
    "validate_file_type",
    "validate_magic_bytes",
    "validate_file_size",
    "read_upload_in_chunks",
    "validate_upload_file",
    "get_file_type_category",
    "requires_ocr",
    "validate_project_ownership",
]
