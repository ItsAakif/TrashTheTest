"""Graph topology/run tests (exact-arch increment). Offline.

Proves the graphs are genuine: named nodes, entry points, conditional
edges, bounded retries/refine caps, and end-to-end invoke over the
deterministic pipeline nodes (no network, no keys).
"""

import pytest

pytestmark = pytest.mark.asyncio


async def test_ingestion_graph_topology_and_run():
    from services.graph import build_ingestion_graph, INGESTION_NODES

    g = build_ingestion_graph()
    topo = g.topology
    assert sorted(topo["nodes"]) == sorted(INGESTION_NODES)
    assert topo["entry"] == "inspect"
    assert "ocr_routing" in topo["conditional_edges"]
    out = await _maybe_await(g.invoke({
        "document_id": "d1", "file_type": "pdf", "puter_authorised": False,
        "pages": [
            {"page": 1, "text": "Introduction to physics definitions " * 6,
             "image_dominant": False},
            {"page": 2, "text": "Chapter two theory and examples " * 6,
             "image_dominant": False},
        ],
    }))
    assert out["classification"] == "born-digital"
    assert out["ocr_route"] == "none"
    assert out["validation"]["ok"] is True
    assert out["persist_ok"] is True


async def test_ingestion_heavy_awaits_explicit_auth():
    from services.graph import build_ingestion_graph

    g = build_ingestion_graph()
    out = await _maybe_await(g.invoke({
        "document_id": "d2", "file_type": "pdf", "puter_authorised": False,
        # inspect derives the real page count from pages: 120 blank pages
        # crosses the heavy threshold (~100) and needs vision throughout.
        "pages": [{"page": i, "text": "", "image_dominant": True}
                  for i in range(1, 121)],
    }))
    assert out["page_count"] == 120
    assert out["ocr_route"] == "puter-offered"


async def test_tutor_graph_refine_loop_bounded():
    from services.graph import build_tutor_graph
    from services.graph.tutor_graph import MAX_REFINE_ROUNDS

    assert MAX_REFINE_ROUNDS == 2
    g = build_tutor_graph()
    assert "validate" in g.topology["conditional_edges"]
    out = await _maybe_await(g.invoke({
        "query": "Explain photosynthesis",
        "candidates": [{"id": "c1", "arm": "vector"}, {"id": "c2", "arm": "lexical"}],
        "citations": [],
    }))
    assert out["intent"] in ("ask", "explain", "study")
    assert out["fused"]
    assert out.get("refine_rounds", 0) <= MAX_REFINE_ROUNDS


async def test_research_graph_dedupes_and_cites():
    from services.graph import build_research_graph

    g = build_research_graph()
    out = await _maybe_await(g.invoke({
        "question": "What is RRF?",
        "search_results": [],
    }))
    assert "plan" in out and "citations" in out


async def test_study_graph_routes_difficulty():
    from services.graph import build_study_graph

    g = build_study_graph()
    out = await _maybe_await(g.invoke({"content": "Photosynthesis converts light."}))
    assert out["difficulty"] == "easy"
    assert out["next_step"] == "advance"
    empty = await _maybe_await(g.invoke({"content": "   "}))
    assert empty.get("error")  # explicit, never fabricated


async def test_graph_nodes_have_bounded_retries():
    from services.graph import build_ingestion_graph, build_study_graph

    assert build_ingestion_graph().topology["max_retries"] == 2
    assert build_study_graph().topology["max_retries"] == 2


async def test_langchain_adapters_present():
    from services.graph import langchain_adapters as adapters

    for name in ("build_prompt", "make_llm", "make_retriever",
                 "make_structured", "make_tool"):
        assert hasattr(adapters, name)


async def _maybe_await(value):
    import inspect

    if inspect.isawaitable(value):
        return await value
    return value
