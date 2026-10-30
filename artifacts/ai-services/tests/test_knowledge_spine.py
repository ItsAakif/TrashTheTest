"""STEP 1 spine tests: typed hierarchy + shared registry (offline).

No DB, no LLM, no network. Live-Postgres checks (migration apply,
cascade, uniqueness enforcement) need a real database and are marked
NOT RUN when it is absent -- the migration contract is asserted on the
migration source plus the ORM table metadata instead.
"""

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from services.academic.spine import (
    assign_order_index,
    canonical_key_for,
    merge_alias_lists,
    normalize_key,
    resolve_alias,
    validate_spine_structure,
)
from services.academic.types import (
    KnowledgeNodeSpec,
    KnowledgeSpineBuildRequest,
    KnowledgeSpineSpec,
    NodeType,
)

pytestmark = pytest.mark.asyncio


def _node(node_type, title, parent_ref=None, order_index=0, **kw):
    payload = {
        "node_type": node_type,
        "title": title,
        "parent_ref": parent_ref,
        "order_index": order_index,
    }
    payload.update(kw)
    return payload


def _full_spine():
    return {
        "title": "Physics primer",
        "nodes": [
            _node("document", "Physics primer"),
            _node("unit", "Mechanics", parent_ref=0),
            _node("topic", "Newton laws", parent_ref=1),
            _node("subtopic", "Friction", parent_ref=2),
            _node("concept", "Static friction", parent_ref=3),
        ],
    }


# -- schema -----------------------------------------------------------------


async def test_spine_models_reject_extra_keys():
    with pytest.raises(ValidationError):
        KnowledgeNodeSpec(
            node_type="topic", title="T", order_index=0, bogus_key="nope"
        )
    with pytest.raises(ValidationError):
        KnowledgeSpineSpec(title="T", nodes=[_node("document", "D")], bogus=1)


async def test_valid_spine_round_trips():
    spine = KnowledgeSpineSpec.model_validate(_full_spine())
    assert [n.node_type for n in spine.nodes] == [
        NodeType.DOCUMENT, NodeType.UNIT, NodeType.TOPIC,
        NodeType.SUBTOPIC, NodeType.CONCEPT,
    ]
    again = KnowledgeSpineSpec.model_validate(spine.model_dump())
    assert again.model_dump() == spine.model_dump()


async def test_spine_rejects_second_root():
    bad = _full_spine()
    bad["nodes"].append(_node("document", "Second root", order_index=1))
    with pytest.raises(ValidationError, match="exactly one root"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_spine_rejects_non_document_root():
    bad = {"title": "T", "nodes": [_node("unit", "Lonely unit")]}
    with pytest.raises(ValidationError, match="must be node_type 'document'"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_spine_rejects_forward_parent_ref():
    bad = _full_spine()
    bad["nodes"][1]["parent_ref"] = 2  # unit points forward at topic: no cycles
    with pytest.raises(ValidationError, match="earlier node"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_spine_rejects_parent_out_of_range():
    bad = _full_spine()
    bad["nodes"][2]["parent_ref"] = 9
    with pytest.raises(ValidationError, match="out of range"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_spine_rejects_depth_contradicting_type():
    bad = _full_spine()
    bad["nodes"][1] = _node("topic", "Topic straight under doc", parent_ref=0)
    with pytest.raises(ValidationError, match="contradicts node_type"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_spine_rejects_duplicate_ord():
    bad = _full_spine()
    bad["nodes"].append(
        _node("unit", "Thermo", parent_ref=0, order_index=0)  # clashes unit ord 0
    )
    with pytest.raises(ValidationError, match="duplicate order_index"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_spine_rejects_sparse_ord():
    bad = _full_spine()
    bad["nodes"][1]["order_index"] = 3  # gap: source order must be dense
    with pytest.raises(ValidationError, match="sparse order_index"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_spine_rejects_duplicate_sibling_canonical_key():
    bad = _full_spine()
    bad["nodes"][1]["canonical_key"] = "mechanics"
    bad["nodes"].append(
        _node("unit", "Mechanics repeat", parent_ref=0, order_index=1,
              canonical_key="mechanics")
    )
    with pytest.raises(ValidationError, match="duplicate canonical_key"):
        KnowledgeSpineSpec.model_validate(bad)


async def test_node_confidence_bounds_and_alias_dedupe():
    with pytest.raises(ValidationError):
        KnowledgeNodeSpec(node_type="topic", title="T", order_index=0, confidence=2.0)
    node = KnowledgeNodeSpec(
        node_type="topic", title="Sampling", order_index=0,
        aliases=["Sampling", " sampling ", "SAMPLING", "Samples"],
    )
    assert node.aliases == ["Sampling", "Samples"]


async def test_build_request_accepts_snake_and_camel():
    snake = KnowledgeSpineBuildRequest(project_id="p1")
    assert snake.project_id == "p1"
    camel = KnowledgeSpineBuildRequest(projectId="p2")
    assert camel.project_id == "p2"
    with pytest.raises(ValidationError):
        KnowledgeSpineBuildRequest(project_id="p1", bogus="x")


# -- pure helpers ------------------------------------------------------------


async def test_normalize_key_folds_surface_forms():
    assert normalize_key("Sampling") == "sampling"
    assert normalize_key("  sampling  ") == "sampling"
    assert normalize_key("Samplings") == "sampling"
    assert normalize_key("Newton's Laws!") == "newton s law"
    assert normalize_key("") == ""


async def test_canonical_key_requires_title():
    assert canonical_key_for("Friction") == "friction"
    with pytest.raises(ValueError):
        canonical_key_for("   ")


async def test_merge_alias_lists_union_ordered():
    assert merge_alias_lists(["A", "b"], ["B", "c"], None, ["a"]) == ["A", "b", "c"]


async def test_resolve_alias_paths():
    registry = {"sampling": "sampling", "friction": "friction"}
    alias_index = {"samples": "sampling"}
    key, via, collision = resolve_alias("Sampling", ["samples"], registry, alias_index)
    assert (key, via, collision) == ("sampling", "direct_key", None)
    key, via, _ = resolve_alias("Samples", ["samples"], registry, alias_index)
    assert (key, via) == ("sampling", "alias")
    key, via, _ = resolve_alias("Entropy", [], registry, alias_index)
    assert (key, via) == ("entropy", "created")
    key, via, _ = resolve_alias("   ", [], {}, None)
    assert (key, via) == (None, "unresolved")


async def test_resolve_alias_collision_never_steals():
    registry = {"sampling": "sampling", "specimen": "specimen"}
    alias_index = {"samples": "specimen"}  # alias owned by another canonical row
    key, via, collision = resolve_alias("Sampling", ["samples"], registry, alias_index)
    assert key == "sampling" and via == "direct_key"
    assert collision is not None and "specimen" in collision


async def test_assign_order_index_dense_per_parent():
    items = [
        {"parent_ref": None}, {"parent_ref": 0}, {"parent_ref": 0},
        {"parent_ref": 1}, {"parent_ref": None},
    ]
    out = assign_order_index(items)
    assert [r["order_index"] for r in out] == [0, 0, 1, 0, 1]
    assert all("order_index" not in r for r in items)  # inputs untouched


async def test_validate_spine_structure_mirrors_pydantic():
    nodes = [
        {"node_type": "document", "title": "D", "parent_ref": None, "order_index": 0},
        {"node_type": "unit", "title": "U", "parent_ref": 0, "order_index": 0},
        {"node_type": "topic", "title": "T", "parent_ref": 1, "order_index": 0},
    ]
    stats = validate_spine_structure(nodes)
    assert stats == {
        "node_count": {"document": 1, "unit": 1, "topic": 1}, "total": 3,
    }
    dup = [dict(n) for n in nodes] + [
        {"node_type": "unit", "title": "U2", "parent_ref": 0, "order_index": 0}
    ]
    with pytest.raises(ValueError, match="duplicate order_index"):
        validate_spine_structure(dup)
    with pytest.raises(ValueError, match="at least one node"):
        validate_spine_structure([])


# -- ORM + migration contract -------------------------------------------------


async def test_knowledge_models_carry_required_columns():
    from database.models.knowledge import CanonicalTopic, KnowledgeNode

    node_cols = {c.name for c in KnowledgeNode.__table__.columns}
    for col in ("id", "document_id", "parent_id", "type", "title",
                "order_index", "pages", "chunks", "aliases", "confidence",
                "metadata", "created_at", "updated_at", "canonical_topic_id",
                "project_id"):
        assert col in node_cols, f"knowledge_nodes missing {col}"
    # `metadata` column maps off the reserved attribute name (same pattern
    # as Chunk.chunk_metadata -> "metadata").
    assert "metadata" in {c.name for c in KnowledgeNode.__table__.columns}
    assert "node_metadata" in KnowledgeNode.__mapper__.attrs.keys()
    reg_cols = {c.name for c in CanonicalTopic.__table__.columns}
    for col in ("id", "canonical_key", "name", "aliases", "created_at"):
        assert col in reg_cols, f"canonical_topics missing {col}"
    assert CanonicalTopic.__table__.c["canonical_key"].unique is True


async def test_migration_009_additive_and_reversible():
    src = Path("migrations/versions/009_knowledge_spine.py").read_text()
    assert "knowledge_nodes" in src and "canonical_topics" in src
    assert "uq_knowledge_nodes_doc_parent_ord" in src
    assert "ix_knowledge_nodes_doc_type" in src
    assert "ix_knowledge_nodes_parent_ord" in src
    assert "GIN" in src  # alias-containment indexes (Postgres path)
    assert re.search(r"def downgrade\(\)", src)
    assert "drop_table" in src
    for legacy in ("documents", "projects", "chunks", "notes", "quizzes"):
        assert f"alter_column({legacy!r}" not in src.lower().replace('"', "'")
        assert f"drop_column({legacy!r}" not in src.lower().replace('"', "'")
    assert "down_revision" in src and "008_website_metadata_columns" in src


async def test_project_and_document_expose_spine_relationships():
    from database.models.document import Document
    from database.models.project import Project

    assert "knowledge_nodes" in Project.__mapper__.relationships
    assert "knowledge_nodes" in Document.__mapper__.relationships
