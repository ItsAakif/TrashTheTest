"""Study graph: content -> check -> eval -> difficulty -> next.

Conditional edge: eval branches to difficulty (content ok) or to END with
an explicit empty-content error (never a fabricated study plan).
"""

from __future__ import annotations

from . import nodes as N
from .runner import SimpleGraph, try_state_graph
from .state import StudyState

STUDY_NODES = ["content", "check", "eval", "difficulty", "next"]


def _route_after_eval(state: dict) -> str:
    if float(state.get("eval_score", 0.0) or 0.0) > 0:
        return "difficulty"
    state["error"] = "no study content: empty input (explicit, not fabricated)"
    return "stop"


class _StudyCompiled:
    def __init__(self):
        g = SimpleGraph(state_keys=tuple(StudyState.__annotations__))
        mapping = {"content": N.study_content_node, "check": N.study_check_node,
                   "eval": N.study_eval_node, "difficulty": N.difficulty_node,
                   "next": N.next_node}
        for name in STUDY_NODES:
            g.add_node(name, mapping[name])
        g.set_entry("content")
        g.add_edge("content", "check")
        g.add_edge("check", "eval")
        g.add_conditional_edges("eval", _route_after_eval,
                                {"difficulty": "difficulty", "stop": None})
        g.add_edge("difficulty", "next")
        self._graph = g
        self.topology = {**g.topology, "terminal": ["next"]}

    def invoke(self, state: dict) -> dict:
        return self._graph.invoke(state)


def build_study_graph():
    sg = try_state_graph(StudyState)
    if sg is not None:  # pragma: no cover - needs langgraph installed
        try:
            from langgraph.graph import END
            mapping = {"content": N.study_content_node, "check": N.study_check_node,
                       "eval": N.study_eval_node, "difficulty": N.difficulty_node,
                       "next": N.next_node}
            for name in STUDY_NODES:
                sg.add_node(name, mapping[name])
            sg.set_entry_point("content")
            sg.add_edge("content", "check")
            sg.add_edge("check", "eval")
            sg.add_conditional_edges("eval", _route_after_eval,
                                     {"difficulty": "difficulty", "stop": END})
            sg.add_edge("difficulty", "next")
            sg.add_edge("next", END)
            compiled = sg.compile()
            compiled.topology = {"nodes": STUDY_NODES, "entry": "content",
                                 "conditional_edges": ["eval"],
                                 "terminal": ["next"],
                                 "max_retries": 2, "backend": "langgraph"}
            return compiled
        except Exception:
            pass
    return _StudyCompiled()


__all__ = ["build_study_graph", "STUDY_NODES"]
