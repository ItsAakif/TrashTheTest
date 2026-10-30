"""Tutor graph: intent -> query-transform -> hybrid -> RRF -> rerank ->
generate -> validate -> refine loop.

Conditional edges: validate branches to END (valid) or refine (invalid);
refine loops back to query-transform with a bounded round cap (2), then END.
"""

from __future__ import annotations

from . import nodes as N
from .runner import SimpleGraph, try_state_graph
from .state import TutorState

TUTOR_NODES = ["intent", "query_transform", "hybrid", "rrf", "rerank",
               "generate", "validate", "refine"]

MAX_REFINE_ROUNDS = 2


def _route_after_validate(state: dict) -> str:
    if state.get("valid"):
        return "done"
    if int(state.get("refine_rounds", 0) or 0) >= MAX_REFINE_ROUNDS:
        return "done"  # bounded: stop refining, report invalid honestly
    return "refine"


class _TutorCompiled:
    def __init__(self):
        g = SimpleGraph(state_keys=tuple(TutorState.__annotations__))
        mapping = {"intent": N.intent_node, "query_transform": N.query_transform_node,
                   "hybrid": N.hybrid_node, "rrf": N.rrf_node,
                   "rerank": N.rerank_node, "generate": N.generate_node,
                   "validate": N.tutor_validate_node, "refine": N.refine_node}
        for name in TUTOR_NODES:
            g.add_node(name, mapping[name])
        g.set_entry("intent")
        g.add_edge("intent", "query_transform")
        g.add_edge("query_transform", "hybrid")
        g.add_edge("hybrid", "rrf")
        g.add_edge("rrf", "rerank")
        g.add_edge("rerank", "generate")
        g.add_edge("generate", "validate")
        g.add_conditional_edges("validate", _route_after_validate,
                                {"done": None, "refine": "refine"})
        g.add_edge("refine", "query_transform")
        self._graph = g
        self.topology = {**g.topology, "terminal": ["validate"],
                         "max_refine_rounds": MAX_REFINE_ROUNDS}

    def invoke(self, state: dict) -> dict:
        return self._graph.invoke(state)


def build_tutor_graph():
    sg = try_state_graph(TutorState)
    if sg is not None:  # pragma: no cover - needs langgraph installed
        try:
            from langgraph.graph import END
            mapping = {"intent": N.intent_node, "query_transform": N.query_transform_node,
                       "hybrid": N.hybrid_node, "rrf": N.rrf_node,
                       "rerank": N.rerank_node, "generate": N.generate_node,
                       "validate": N.tutor_validate_node, "refine": N.refine_node}
            for name in TUTOR_NODES:
                sg.add_node(name, mapping[name])
            sg.set_entry_point("intent")
            sg.add_edge("intent", "query_transform")
            sg.add_edge("query_transform", "hybrid")
            sg.add_edge("hybrid", "rrf")
            sg.add_edge("rrf", "rerank")
            sg.add_edge("rerank", "generate")
            sg.add_edge("generate", "validate")
            sg.add_conditional_edges("validate", _route_after_validate,
                                     {"done": END, "refine": "refine"})
            sg.add_edge("refine", "query_transform")
            compiled = sg.compile()
            compiled.topology = {"nodes": TUTOR_NODES, "entry": "intent",
                                 "conditional_edges": ["validate"],
                                 "terminal": ["validate"],
                                 "max_refine_rounds": MAX_REFINE_ROUNDS,
                                 "max_retries": 2, "backend": "langgraph"}
            return compiled
        except Exception:
            pass
    return _TutorCompiled()


__all__ = ["build_tutor_graph", "TUTOR_NODES", "MAX_REFINE_ROUNDS"]
