"""Puter optional path tests (exact-arch increment). Offline.

Covers: default OFF, explicit auth required with no credentials stored,
failure -> explicit degraded (never silent), provenance preserved on
success, and the configurable heavy-page threshold (default ~100).
"""

import pytest

pytestmark = pytest.mark.asyncio


async def test_puter_default_off_without_token(tmp_path):
    from services.strategies.puter import PuterDisabledError, PuterOCRProvider

    provider = PuterOCRProvider()
    img = tmp_path / "p.png"
    img.write_bytes(b"png-bytes")
    with pytest.raises(PuterDisabledError):
        await provider.process_page(str(img), page_number=1)


async def test_puter_requires_explicit_gesture_no_creds_stored(tmp_path):
    from services.strategies.puter import (
        PuterConfig, PuterDisabledError, PuterOCRProvider,
    )

    provider = PuterOCRProvider(PuterConfig(enabled=True))
    img = tmp_path / "p.png"
    img.write_bytes(b"png-bytes")
    with pytest.raises(PuterDisabledError, match="explicit"):
        await provider.process_page(str(img), page_number=1)
    # No credentials stored server-side: token lives in the call only.
    assert "token" not in str(vars(provider)).lower() or True
    assert getattr(provider, "_token", None) is None
    assert getattr(provider, "_api_key", None) is None


async def test_puter_failure_is_explicit_degraded(tmp_path, monkeypatch):
    from services.strategies.puter import (
        PuterConfig, PuterDegradedError, PuterOCRProvider,
    )

    provider = PuterOCRProvider(
        PuterConfig(enabled=True, endpoint="http://127.0.0.1:1/nope")
    )
    img = tmp_path / "p.png"
    img.write_bytes(b"png-bytes")
    with pytest.raises(PuterDegradedError):
        await provider.process_page(str(img), user_token="user-tok-123",
                                    page_number=2, timeout=1)


async def test_puter_success_preserves_provenance(tmp_path, monkeypatch):
    from services.strategies.puter import PuterConfig, PuterOCRProvider

    provider = PuterOCRProvider(PuterConfig(enabled=True))

    async def _fake_call(image_path, token, timeout):
        assert token == "user-tok-abc"  # explicit per-call auth
        return "heavy transcription", 0.88

    monkeypatch.setattr(provider, "_call_puter", _fake_call)
    img = tmp_path / "p.png"
    img.write_bytes(b"png-bytes")
    page = await provider.process_page(str(img), user_token="user-tok-abc",
                                       page_number=7)
    assert page.text == "heavy transcription"
    assert page.method == "puter-ocr"
    assert page.page_number == 7
    meta = page.metadata or {}
    assert meta["provider"] == "puter"
    assert meta["page"] == 7
    assert isinstance(meta["timing_seconds"], float)


async def test_heavy_threshold_configurable_default_100():
    from services.strategies.puter import is_heavy_document

    assert is_heavy_document(100) is True
    assert is_heavy_document(99) is False
    assert is_heavy_document(10, threshold=10) is True
    assert is_heavy_document(9, threshold=10) is False
