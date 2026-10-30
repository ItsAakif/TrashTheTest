"""Puter strategy package (optional heavy-document path, default OFF)."""

from .provider import (
    PuterConfig,
    PuterDegradedError,
    PuterDisabledError,
    PuterOCRProvider,
    is_heavy_document,
    create_puter_provider,
)

__all__ = [
    "PuterConfig",
    "PuterDegradedError",
    "PuterDisabledError",
    "PuterOCRProvider",
    "is_heavy_document",
    "create_puter_provider",
]
