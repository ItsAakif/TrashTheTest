"""Academic spine pure helpers (STEP 1, deterministic, no DB, no LLM).

Shared registry alias resolution lives here as a pure function
(resolve_alias over an in-memory key map); the DB-backed wrapper sits in
database/repositories.py. Source document order is authoritative:
assign_order_index produces dense order_index values per parent.
"""

import re
from typing import Dict, List, Optional, Tuple

from services.academic.types import NODE_DEPTH, NodeType

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def normalize_key(text: str) -> str:
    """Deterministic normalizer for titles and aliases.

    Lowercase, strip punctuation, collapse whitespace, singularize a
    trailing 's' (simple plural fold, no stemming library). Two surface
    forms such as "Sampling" and "sampling methods" fold predictably.
    """
    folded = _PUNCT_RE.sub(" ", (text or "").lower())
    folded = " ".join(folded.split())
    if len(folded) > 3 and folded.endswith("s") and not folded.endswith("ss"):
        folded = folded[:-1]
    return folded


def canonical_key_for(title: str, aliases: Optional[List[str]] = None) -> str:
    """Canonical registry key for a node title (normalized title)."""
    key = normalize_key(title)
    if not key:
        raise ValueError("canonical_key_for requires a non-empty title")
    return key


def merge_alias_lists(*lists: Optional[List[str]]) -> List[str]:
    """Union of alias lists, order-preserving, case-insensitive deduped."""
    seen: set = set()
    out: List[str] = []
    for lst in lists:
        for alias in lst or []:
            key = (alias or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(alias)
    return out


def resolve_alias(
    title: str,
    aliases: Optional[List[str]],
    registry: Dict[str, str],
    alias_index: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[str], str, Optional[str]]:
    """Resolve a node against the shared registry (pure, no DB).

    registry maps canonical_key -> canonical_key (known identities);
    alias_index maps alias -> canonical_key (keys are normalized here, so
    callers may pass raw surface forms). Returns
    (canonical_key | None, via, collision | None) where via is one of
    direct_key | alias | created | unresolved. When a normalized alias
    belongs to a DIFFERENT canonical key than the direct key, the alias
    is not stolen: via stays on the direct path and the collision is
    reported.
    """
    direct = normalize_key(title)
    collision: Optional[str] = None
    norm_aliases = {normalize_key(a): a for a in (aliases or []) if normalize_key(a)}
    norm_index = {
        normalize_key(k): v for k, v in (alias_index or {}).items() if normalize_key(k)
    }
    if direct and direct in registry:
        if norm_index:
            for norm_alias, raw_alias in norm_aliases.items():
                owner = norm_index.get(norm_alias)
                if owner is not None and owner != direct:
                    collision = (
                        f"alias {raw_alias!r} belongs to canonical "
                        f"{owner!r}, not {direct!r}"
                    )
                    break
        return direct, "direct_key", collision
    if norm_index:
        for norm_alias in norm_aliases:
            owner = norm_index.get(norm_alias)
            if owner is not None:
                return owner, "alias", None
    if direct:
        return direct, "created", None
    return None, "unresolved", None


def assign_order_index(
    items: List[Dict],
    parent_key: str = "parent_ref",
    order_key: str = "order_index",
) -> List[Dict]:
    """Assign dense order_index per parent in input (document) order.

    Items keep their relative order; the first child of a parent gets 0,
    the next 1, and so on. Returns new dicts, inputs untouched.
    """
    counters: Dict[int, int] = {}
    out: List[Dict] = []
    for item in items:
        row = dict(item)
        parent = row.get(parent_key)
        nxt = counters.get(parent, 0)
        row[order_key] = nxt
        counters[parent] = nxt + 1
        out.append(row)
    return out


def validate_spine_structure(nodes: List[Dict]) -> Dict:
    """Validate a plain-dict spine (DB rows or payloads), mirroring the
    Pydantic KnowledgeSpineSpec rules. Returns stats; raises ValueError
    naming the offending node on any violation.
    """
    if not nodes:
        raise ValueError("spine must hold at least one node")
    roots = [i for i, n in enumerate(nodes) if n.get("parent_ref") is None]
    if len(roots) != 1:
        raise ValueError(f"spine must have exactly one root, found {len(roots)}")
    if str(nodes[roots[0]].get("node_type")) != NodeType.DOCUMENT.value:
        raise ValueError(
            f"node {roots[0]} ({nodes[roots[0]].get('title')!r}): root must "
            "be node_type 'document'"
        )
    depths: Dict[int, int] = {}

    def _depth(idx: int, trail: tuple) -> int:
        if idx in depths:
            return depths[idx]
        if idx in trail:
            raise ValueError(f"node {idx}: cycle in parent chain")
        ref = nodes[idx].get("parent_ref")
        if ref is None:
            depths[idx] = 0
        else:
            if not isinstance(ref, int) or not (0 <= ref < len(nodes)):
                raise ValueError(f"node {idx}: parent_ref {ref!r} out of range")
            if ref >= idx:
                raise ValueError(
                    f"node {idx}: parent_ref must point to an earlier node"
                )
            depths[idx] = _depth(ref, trail + (idx,)) + 1
        expected = NODE_DEPTH[NodeType(str(nodes[idx].get("node_type")))]
        if depths[idx] != expected:
            raise ValueError(
                f"node {idx} ({nodes[idx].get('title')!r}): depth "
                f"{depths[idx]} contradicts node_type "
                f"{nodes[idx].get('node_type')!r}"
            )
        return depths[idx]

    for i in range(len(nodes)):
        _depth(i, ())
    by_parent: Dict[Optional[int], List[int]] = {}
    for i, node in enumerate(nodes):
        by_parent.setdefault(node.get("parent_ref"), []).append(i)
    for parent, members in by_parent.items():
        ords = sorted(nodes[m].get("order_index") for m in members)
        if len(set(ords)) != len(ords):
            raise ValueError(f"duplicate order_index under parent {parent}: {ords}")
        if ords != list(range(len(members))):
            raise ValueError(f"sparse order_index under parent {parent}: {ords}")
    counts: Dict[str, int] = {}
    for node in nodes:
        key = str(node.get("node_type"))
        counts[key] = counts.get(key, 0) + 1
    return {"node_count": counts, "total": len(nodes)}


__all__ = [
    "normalize_key",
    "canonical_key_for",
    "merge_alias_lists",
    "resolve_alias",
    "assign_order_index",
    "validate_spine_structure",
]
