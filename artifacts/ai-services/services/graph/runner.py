"""Minimal graph runner: StateGraph semantics without requiring langgraph.

Used when langgraph is not installed. Mirrors the topology API the tests
assert: named nodes, edges, conditional edges, bounded per-node retries,
and invoke(state) -> state. When langgraph IS installed, builders prefer
the real StateGraph (see each *_graph.py).
"""

from __future__ import annotations

from typing import Any, Callable


class SimpleGraph:
    def __init__(self, state_keys: tuple = ()):
        self.state_keys = tuple(state_keys)
        self.nodes: dict[str, Callable] = {}
        self.edges: dict[str, list[str]] = {}
        self.conditional: dict[str, tuple[Callable, dict]] = {}
        self.entry: str | None = None
        self.max_retries: int = 2

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def set_entry(self, name: str) -> None:
        self.entry = name

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.setdefault(src, []).append(dst)

    def add_conditional_edges(self, src: str, fn: Callable, mapping: dict) -> None:
        self.conditional[src] = (fn, dict(mapping))

    @property
    def topology(self) -> dict:
        return {
            "nodes": sorted(self.nodes),
            "edges": {k: list(v) for k, v in self.edges.items()},
            "conditional_edges": sorted(self.conditional),
            "entry": self.entry,
            "max_retries": self.max_retries,
        }

    def invoke(self, state: dict) -> dict:
        if not self.entry:
            raise ValueError("graph has no entry point")
        current = dict(state)
        attempts: dict[str, int] = dict(current.get("attempts") or {})
        current["attempts"] = attempts
        node = self.entry
        steps = 0
        while node and steps < 64:
            steps += 1
            fn = self.nodes.get(node)
            if fn is None:
                current["error"] = f"unknown node: {node}"
                break
            try:
                update = fn(current) or {}
            except Exception as exc:
                n = int(attempts.get(node, 0) or 0) + 1
                attempts[node] = n
                if n <= self.max_retries:
                    continue  # bounded retry of the same node
                current["error"] = f"{node} failed after {n} attempts: {exc}"
                break
            if isinstance(update, dict):
                current.update(update)
            if node in self.conditional:
                fn_c, mapping = self.conditional[node]
                try:
                    key = fn_c(current)
                except Exception as exc:
                    current["error"] = f"conditional {node} failed: {exc}"
                    break
                node = mapping.get(key)
                continue
            nxt = (self.edges.get(node) or [None])[0]
            node = nxt
        return current


def try_state_graph(state_cls):
    """Return langgraph StateGraph when importable, else None."""
    try:
        from langgraph.graph import StateGraph

        return StateGraph(state_cls)
    except Exception:
        return None


__all__ = ["SimpleGraph", "try_state_graph"]
