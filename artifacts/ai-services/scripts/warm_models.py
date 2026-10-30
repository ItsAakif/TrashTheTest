"""Warm ML weights once (embedding + OCR) so the first real job is fast.

Downloads the Qwen3-Embedding-0.6B weights (~1GB) and the PP-OCRv6 tiny
det/rec models into the persisted cache (HF_HOME / modelcache volume).
Run once after a fresh build, or after wiping the modelcache volume:

    docker compose run --rm worker python -m scripts.warm_models
    # or locally:
    python -m scripts.warm_models

Honest output only: prints what was warmed and where it is cached.
No GPU, CPU-only, same providers the worker uses (via the factory).
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


async def main_async() -> dict:
    from factory import get_embedding, get_ocr

    report: dict = {
        "hf_home": os.getenv("HF_HOME", "(default ~/.cache/huggingface)"),
        "embedding_model": None,
        "embedding_ok": False,
        "ocr_ok": False,
        "ocr_reason": None,
    }

    provider = get_embedding()
    await provider.warm()
    report["embedding_model"] = getattr(provider, "model_name", "unknown")
    # One real encode so lazy backends (tokenizer, pooling) init now.
    res = await provider.embed_passages(["warmup"], batch_size=1)
    report["embedding_ok"] = bool(res.embeddings) and len(res.embeddings[0]) > 0
    report["embedding_dim"] = len(res.embeddings[0]) if res.embeddings else 0

    try:
        ocr = get_ocr()
        ensure = getattr(ocr, "_ensure_engine", None)
        if callable(ensure):
            await asyncio.to_thread(ensure)
        health = await ocr.health()
        report["ocr_ok"] = bool(health.get("available"))
        report["ocr_reason"] = health.get("reason")
        report["ocr_provider"] = health.get("provider")
    except Exception as exc:
        report["ocr_ok"] = False
        report["ocr_reason"] = str(exc)

    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = asyncio.run(main_async())
    print(f"embedding: model={report['embedding_model']} "
          f"ok={report['embedding_ok']} dim={report.get('embedding_dim')}")
    print(f"ocr: ok={report['ocr_ok']} reason={report.get('ocr_reason')}")
    print(f"cache: {report['hf_home']}")
    if not report["embedding_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
