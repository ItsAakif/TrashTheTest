"""Academic baseline report writer (STEP 0).

Runs the CURRENT single-shot Notes pipeline deterministically over the
synthetic corpus (tests/eval/corpus.py: 3 documents, 36 chunks -- the
only deterministic fixture available) and writes the baseline artifact
plus its measurements. No LLM, no DB, no network.

Usage (from artifacts/ai-services):
    python -m scripts.academic_baseline --out ../../.planning/evidence/academic-baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Same throwaway bootstrap as tests/__init__.py: config.settings
# fail-fasts at import without these; the baseline needs no DB.
os.environ.setdefault("JWT_SECRET", "test-secret-at-least-32-chars-long-0123456789")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost:5432/test")

from services.academic.baseline import run_baseline
from tests.eval.corpus import CORPUS, DOCUMENTS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="STEP 0 academic baseline")
    parser.add_argument(
        "--out", default="../../.planning/evidence/academic-baseline.json"
    )
    args = parser.parse_args(argv)

    artifact = run_baseline(CORPUS)
    payload = {
        "fixture_documents": DOCUMENTS,
        "corpus_chunks": len(CORPUS),
        "baseline": artifact,
    }
    out = args.out
    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    metrics = artifact["metrics"]
    print(
        "baseline "
        f"source_topics={metrics['source_topics']} "
        f"generated_topics={metrics['generated_topics']} "
        f"order_accuracy={metrics['order_accuracy']} "
        f"concept_coverage={metrics['concept_coverage']} "
        f"citation_coverage={metrics['citation_coverage']} "
        f"dupes={metrics['dupes']} "
        f"length_tokens_est={metrics['length_tokens_est']} "
        f"missing_count={metrics['missing_count']} "
        f"unsupported={metrics['grounding']['unsupported']} "
        f"-> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
