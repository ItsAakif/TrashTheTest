"""STEP 7 narrative prompt: the LLM explains metric results ONLY.

The prompt embeds the deterministic MetricsReport JSON and instructs the
model to use ONLY numbers present in it. validate_narrative() enforces
that instruction (numbers compared in normalized form); a narrative that
invents a percentage is rejected, repaired once, then withheld.
"""

import json
from typing import Any, Dict

from services.academic.types import MetricsReport


def narrative_prompt(report: MetricsReport) -> str:
    """Build the explanation-only prompt from a computed MetricsReport."""
    payload = json.dumps(report.model_dump(mode="json"), sort_keys=True)
    return (
        "Explain the exam-intelligence metrics below for a student. "
        "Use ONLY numbers that appear verbatim in the metrics JSON: "
        "do not invent percentages, counts, years or ranks. "
        "Name hotspot topics and never-asked topics exactly as listed.\n\n"
        f"METRICS:\n{payload}\n\n"
        "Return one short paragraph of plain prose (no JSON)."
    )


async def generate_metrics_narrative(
    engine: Any,
    report: MetricsReport,
    allowed_chunk_ids: set,
) -> Dict[str, Any]:
    """Live LLM narrative call through the shared schema-enforced path."""
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import GenerationMode

    prompt = PromptTemplates.get_prompt(
        GenerationMode.METRICS_NARRATIVE, narrative_prompt(report),
        "exam intelligence narrative",
    )
    return await engine._generate_structured(
        GenerationMode.METRICS_NARRATIVE, prompt, allowed_chunk_ids
    )


__all__ = ["narrative_prompt", "generate_metrics_narrative"]
