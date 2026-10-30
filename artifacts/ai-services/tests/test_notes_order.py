"""Notes document-order tests (live-bug batch, bug 5). Offline.

Notes context blocks must follow source document order
(page_start/position), not relevance rank; the notes prompt must carry
the document-order rule.
"""

import uuid

import pytest

from services.transformation.engine import TransformationEngine, _document_order_key
from services.transformation.prompts import PromptTemplates
from services.retrieval.types import RetrievedChunk, RetrievalResult


pytestmark = pytest.mark.asyncio


def _chunk(cid, content, similarity, page=None, section=None, position=None):
    meta = {}
    if position is not None:
        meta["position"] = position
    return RetrievedChunk(
        chunk_id=cid,
        content=content,
        similarity=similarity,
        metadata=meta,
        document_id=str(uuid.uuid4()),
        page=page,
        section=section,
    )


def _engine():
    class StubLLM:
        async def generate(self, prompt, config=None):
            from services.strategies.base import GenerationResult

            return GenerationResult(text="{}", model="stub", usage={},
                                    finish_reason="stop", processing_time=0.0)

    return TransformationEngine(llm_provider=StubLLM())


async def test_notes_context_follows_document_order_not_relevance():
    engine = _engine()
    late = _chunk("late-id", "late chapter content here", 0.99, page=9, section="Ch9")
    early = _chunk("early-id", "early chapter content here", 0.10, page=1, section="Ch1")
    unpaged = _chunk("unpaged-id", "unpaged content here", 0.50)
    result = RetrievalResult(query="q", total_results=3,
                             chunks=[late, unpaged, early], processing_time=0)
    ordered = engine._build_context(result, document_order=True)
    assert ordered.index("chunk_id=early-id") < ordered.index("chunk_id=late-id")
    assert "page=1" in ordered and "section=Ch1" in ordered
    # Unpaginated chunks sort after paginated ones.
    assert ordered.index("chunk_id=late-id") < ordered.index("chunk_id=unpaged-id")
    # Default path keeps relevance order (no behavior change elsewhere).
    relevance = engine._build_context(result)
    assert relevance.index("chunk_id=late-id") < relevance.index("chunk_id=unpaged-id")


async def test_document_order_key_position_breaks_page_ties():
    engine = _engine()
    first = _chunk("a", "first on page two content", 0.9, page=2, position=0)
    second = _chunk("b", "second on page two content", 0.9, page=2, position=3)
    result = RetrievalResult(query="q", total_results=2,
                             chunks=[second, first], processing_time=0)
    ordered = engine._build_context(result, document_order=True)
    assert ordered.index("chunk_id=a") < ordered.index("chunk_id=b")
    assert _document_order_key(first) < _document_order_key(second)


async def test_notes_prompt_carries_document_order_rule():
    prompt = PromptTemplates.get_notes_prompt("ctx")
    assert "DOCUMENT ORDER RULE" in prompt
