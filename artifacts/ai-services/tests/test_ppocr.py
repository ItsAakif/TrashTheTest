"""PP-OCRv6 CPU ONNX provider tests (exact-arch increment). Offline.

Covers: factory default is the single PP path, process_page contract
(text/confidence/provider/method/page/timing, never silent empty),
missing-engine explicit degraded, per-page timeout, quality-gate signals,
page-level routing (normal->PPOCR vs heavy->Puter offered), to_thread
isolation, and settings defaults (tiny det/rec, onnxruntime, cpu).
"""

import asyncio

import pytest

pytestmark = pytest.mark.asyncio


async def test_factory_default_is_single_ppocr_path():
    import factory as factory_mod

    factory_mod.get_strategy_factory().reset()
    try:
        provider = factory_mod.get_strategy_factory().get_ocr()
        assert type(provider).__name__ == "PPOCROnxProvider"
    finally:
        factory_mod.get_strategy_factory().reset()


async def test_process_page_contract_with_fake_engine(monkeypatch):
    from services.strategies.ocr.ppocr_onnx import PPOCROnxProvider

    provider = PPOCROnxProvider()
    monkeypatch.setattr(
        provider, "_run_inference",
        lambda image: {"text": "hello world", "confidence": 0.91},
    )
    page = await provider.process_page(b"fake-png-bytes", page_number=3)
    assert page.text == "hello world"
    assert page.confidence == pytest.approx(0.91)
    assert page.method == "ppocr-tiny-ort"
    assert page.page_number == 3
    meta = page.metadata or {}
    assert meta["provider"] == "ppocr-onnx"
    assert meta["method"] == "ppocr-tiny-ort"
    assert meta["page"] == 3
    assert isinstance(meta["timing_seconds"], float)


async def test_process_page_never_silent_empty(monkeypatch):
    from services.strategies.ocr.ppocr_onnx import (
        OCRAvailabilityError, PPOCROnxProvider,
    )

    provider = PPOCROnxProvider()
    monkeypatch.setattr(
        provider, "_run_inference", lambda image: {"text": "   ", "confidence": 0.0}
    )
    with pytest.raises(OCRAvailabilityError, match="empty output"):
        await provider.process_page(b"bytes", page_number=1)


async def test_missing_engine_is_explicit_degraded():
    from services.strategies.ocr.ppocr_onnx import PPOCROnxProvider

    provider = PPOCROnxProvider()
    provider._ocr = None
    health = await provider.health()
    # paddleocr/onnxruntime are not installed in CI: explicit, never silent.
    assert health["available"] is False
    assert health["provider"] == "ppocr-onnx"
    assert "reason" in health and health["reason"]


async def test_per_page_timeout_is_typed(monkeypatch):
    import time

    from services.strategies.ocr.ppocr_onnx import OCRTimeoutError, PPOCROnxProvider

    provider = PPOCROnxProvider(timeout_seconds=60)

    def _slow(image):
        time.sleep(0.2)
        return {"text": "late", "confidence": 1.0}

    monkeypatch.setattr(provider, "_run_inference", _slow)
    with pytest.raises(OCRTimeoutError, match="exceeded"):
        await provider.process_page(b"bytes", timeout=0.01, page_number=1)


async def test_quality_gate_signals_cover_all_dimensions():
    from services.extraction.extractor import ExtractorConfig, _needs_vision

    cfg = ExtractorConfig()
    # char count / density: healthy text passes
    need, _ = _needs_vision("Introduction to physics definitions " * 4, False, cfg)
    assert need is False
    # suspicious ratio: garbage layer escalates despite length
    need, reason = _needs_vision("\u00ac \u25a1 \ufffd " * 60, False, cfg)
    assert need is True and reason == "low_alnum_ratio"
    # whitespace-only: no text layer
    need, reason = _needs_vision("   ", True, cfg)
    assert need is True and reason == "no_text_layer"
    # completeness: image-dominant short text escalates
    need, reason = _needs_vision("ab", True, cfg)
    assert need is True and reason == "image_dominant_short_text"


async def test_page_routing_normal_takes_ppocr():
    from services.graph.nodes import ocr_routing_node

    state = {"page_count": 4, "puter_authorised": False,
             "gate": {"signals": [{"page": 1, "needs_vision": True}]}}
    assert ocr_routing_node(state)["ocr_route"] == "ppocr"
    state2 = {"page_count": 4, "gate": {"signals": [{"page": 1, "needs_vision": False}]}}
    assert ocr_routing_node(state2)["ocr_route"] == "none"


async def test_page_routing_heavy_offers_puter_not_hard_invariant():
    from services.graph.nodes import ocr_routing_node
    from services.strategies.puter import is_heavy_document

    assert is_heavy_document(100) is True
    assert is_heavy_document(99) is False
    assert is_heavy_document(50, threshold=50) is True
    state = {"page_count": 250, "puter_authorised": False,
             "gate": {"signals": [{"page": 1, "needs_vision": True}]}}
    assert ocr_routing_node(state)["ocr_route"] == "puter-offered"
    state["puter_authorised"] = True
    assert ocr_routing_node(state)["ocr_route"] == "ppocr"


async def test_ocr_isolated_from_request_handling():
    import inspect

    from services.strategies.ocr import ppocr_onnx

    src = inspect.getsource(ppocr_onnx.PPOCROnxProvider.process_page)
    assert "to_thread" in src  # CPU inference never blocks the event loop


async def test_settings_defaults_are_ppocr_tiny_cpu():
    from config.settings import get_settings

    ocr = get_settings().ocr
    assert ocr.provider == "ppocr-onnx"
    assert "tiny" in ocr.det_model and "tiny" in ocr.rec_model
    assert ocr.engine == "onnxruntime"
    assert ocr.device == "cpu"
