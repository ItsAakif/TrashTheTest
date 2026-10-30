"""Research graph: plan -> search -> dedupe -> eval -> synth -> cites.

Conditional edge: eval branches to synth (evidence ok) or re-plan (empty),
with a bounded re-plan cap (1) so the graph always terminates.
"""

from __future__ import annotations

from . import nodes as N
from .runner import SimpleGraph, try_state_graph
from .state import ResearchState

RESEARCH_NODES = ["plan", "search", "dedupe", "eval", "synth", "cites"]

MAX_REPLANS = 1


def _route_after_eval(state: dict) -> str:
    scores = state.get("eval_scores") or {}
    if scores.get("ok"):
        return "synth"
    if int(state.get("replans", 0) or 0) >= MAX_REPLANS:
        return "synth"  # bounded: synthesize honestly over empty evidence
    return "replan"


class _ResearchCompiled:
    def __init__(self):
        g = SimpleGraph(state_keys=tuple(ResearchState.__annotations__))
        mapping = {"plan": N.plan_node, "search": N.search_node,
                   "dedupe": N.dedupe_node, "eval": N.research_eval_node,
                   "synth": N.synth_node, "cites": N.cites_node}
        for name in RESEARCH_NODES:
            g.add_node(name, mapping[name])
        g.set_entry("plan")
        g.add_edge("plan", "search")
        g.add_edge("search", "dedupe")
        g.add_edge("dedupe", "eval")
        g.add_conditional_edges("eval", _route_after_eval,
                                {"synth": "synth", "replan": "plan"})
        g.add_edge("synth", "cites")
        self._graph = g
        self.topology = {**g.topology, "terminal": ["cites"],
                         "max_replans": MAX_REPLANS}

    def invoke(self, state: dict) -> dict:
        return self._graph.invoke(state)


def build_research_graph():
    sg = try_state_graph(ResearchState)
    if sg is not None:  # pragma: no cover - needs langgraph installed
        try:
            from langgraph.graph import END
            mapping = {"plan": N.plan_node, "search": N.search_node,
                       "dedupe": N.dedupe_node, "eval": N.research_eval_node,
                       "synth": N.synth_node, "cites": N.cites_node}
            for name in RESEARCH_NODES:
                sg.add_node(name, mapping[name])
            sg.set_entry_point("plan")
            sg.add_edge("plan", "search")
            sg.add_edge("search", "dedupe")
            sg.add_edge("dedupe", "eval")
            sg.add_conditional_edges("eval", _route_after_eval,
                                     {"synth": "synth", "replan": "plan"})
            sg.add_edge("synth", "cites")
            sg.add_edge("cites", END)
            compiled = sg.compile()
            compiled.topology = {"nodes": RESEARCH_NODES, "entry": "plan",
                                 "conditional_edges": ["eval"],
                                 "terminal": ["cites"],
                                 "max_replans": MAX_REPLANS,
                                 "max_retries": 2, "backend": "langgraph"}
            return compiled
        except Exception:
            pass
    return _ResearchCompiled()


__all__ = ["build_research_graph", "RESEARCH_NODES", "MAX_REPLANS"]
