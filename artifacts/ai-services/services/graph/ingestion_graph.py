"""Ingestion graph: inspect -> classify -> extract -> gate -> ocr-routing
-> clean -> structure -> chunk -> embed -> persist -> validate.

Conditional edges: ocr-routing branches on gate signals + heavy-doc offer;
validate branches to END ok / END needs-attention; every node has bounded
retries (max_retries=2) via the runner / langgraph RetryPolicy.
"""

from __future__ import annotations

from . import nodes as N
from .runner import SimpleGraph, try_state_graph
from .state import IngestionState

INGESTION_NODES = ["inspect", "classify", "extract", "gate", "ocr_routing",
                   "clean", "structure", "chunk", "embed", "persist", "validate"]

_CHAIN = [("inspect", N.inspect_node), ("classify", N.classify_node),
          ("extract", N.extract_node), ("gate", N.gate_node),
          ("ocr_routing", N.ocr_routing_node), ("clean", N.clean_node),
          ("structure", N.structure_node), ("chunk", N.chunk_node),
          ("embed", N.embed_node), ("persist", N.persist_node),
          ("validate", N.validate_node)]


def _route_after_ocr(state: dict) -> str:
    route = (state.get("ocr_route") or "ppocr")
    if route == "puter-offered" and not state.get("puter_authorised"):
        return "await_auth"  # explicit gesture required; stop before clean
    return "clean"


def _route_after_validate(state: dict) -> str:
    if state.get("error"):
        return "needs_attention"
    return "ok" if (state.get("validation") or {}).get("ok") else "needs_attention"


class _IngestionCompiled:
    """Runner-backed compiled graph (offline-safe, same topology)."""

    def __init__(self):
        g = SimpleGraph(state_keys=tuple(IngestionState.__annotations__))
        for name, fn in _CHAIN:
            g.add_node(name, fn)
        g.set_entry("inspect")
        for a, b in zip(INGESTION_NODES, INGESTION_NODES[1:]):
            if a == "ocr_routing":
                continue
            if a == "validate":
                continue
            g.add_edge(a, b)
        g.add_conditional_edges("ocr_routing", _route_after_ocr,
                                {"clean": "clean", "await_auth": None})
        # validate is terminal: conditional records ok/needs_attention.
        g.conditional["validate"] = (_route_after_validate,
                                     {"ok": None, "needs_attention": None})
        self._graph = g
        self.topology = {**g.topology, "terminal": ["validate"]}

    def invoke(self, state: dict) -> dict:
        return self._graph.invoke(state)


def build_ingestion_graph():
    """Compile the ingestion graph (real StateGraph when langgraph present)."""
    sg = try_state_graph(IngestionState)
    if sg is not None:  # pragma: no cover - needs langgraph installed
        try:
            from langgraph.graph import END
            for name, fn in _CHAIN:
                sg.add_node(name, fn)
            sg.set_entry_point("inspect")
            for a, b in zip(INGESTION_NODES, INGESTION_NODES[1:]):
                if a == "ocr_routing":
                    continue
                sg.add_edge(a, b)
            sg.add_conditional_edges("ocr_routing", _route_after_ocr,
                                     {"clean": "clean", "await_auth": END})
            sg.add_edge("validate", END)
            compiled = sg.compile()
            compiled.topology = {"nodes": INGESTION_NODES, "entry": "inspect",
                                 "conditional_edges": ["ocr_routing"],
                                 "terminal": ["validate"], "max_retries": 2,
                                 "backend": "langgraph"}
            return compiled
        except Exception:
            pass
    return _IngestionCompiled()


__all__ = ["build_ingestion_graph", "INGESTION_NODES"]
