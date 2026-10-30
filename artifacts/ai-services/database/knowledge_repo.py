
# ==================== KNOWLEDGE SPINE REPOSITORY (STEP 1) ====================
#
# Additive helpers for the academic knowledge spine (migration 009).
# Helpers flush; the caller owns the commit. Registry writes are
# append-only: get_or_create_canonical_topic never overwrites `name`.

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def create_knowledge_model(
    db: AsyncSession,
    project_id: uuid.UUID,
    version: int,
    source_fingerprint: str,
    status: str = "building",
    node_count: int = 0,
    error: Optional[str] = None,
):
    """Create one version header; caller owns the transaction commit."""
    from database.models.knowledge import KnowledgeModel

    model = KnowledgeModel(
        id=uuid.uuid4(),
        project_id=project_id,
        version=int(version),
        status=status,
        source_fingerprint=source_fingerprint,
        node_count=int(node_count),
        error=error,
    )
    db.add(model)
    await db.flush()
    return model


async def get_knowledge_model(
    db: AsyncSession,
    project_id: uuid.UUID,
    version: Optional[int] = None,
):
    """Read one model header, or the latest version when version is None."""
    from database.models.knowledge import KnowledgeModel

    stmt = select(KnowledgeModel).where(KnowledgeModel.project_id == project_id)
    if version is not None:
        stmt = stmt.where(KnowledgeModel.version == int(version))
    else:
        stmt = stmt.order_by(KnowledgeModel.version.desc())
    result = await db.execute(stmt)
    if version is not None:
        return result.scalars().first()
    return result.scalars().first()


async def list_knowledge_models(db: AsyncSession, project_id: uuid.UUID):
    """List version headers oldest-first for deterministic status reads."""
    from database.models.knowledge import KnowledgeModel

    result = await db.execute(
        select(KnowledgeModel)
        .where(KnowledgeModel.project_id == project_id)
        .order_by(KnowledgeModel.version.asc())
    )
    return list(result.scalars().all())


async def set_knowledge_model_status(
    db: AsyncSession,
    model_id: uuid.UUID,
    status: str,
    node_count: Optional[int] = None,
    error: Optional[str] = None,
):
    """Update a version header without committing; caller owns commit."""
    from database.models.knowledge import KnowledgeModel

    model = await db.get(KnowledgeModel, model_id)
    if model is None:
        return None
    model.status = status
    if node_count is not None:
        model.node_count = int(node_count)
    if error is not None:
        model.error = error
    await db.flush()
    return model


async def supersede_knowledge_models(
    db: AsyncSession,
    project_id: uuid.UUID,
    before_version: int,
):
    """Mark older headers superseded while keeping them readable."""
    rows = await list_knowledge_models(db, project_id)
    changed = []
    for row in rows:
        if int(row.version) < int(before_version) and row.status != "superseded":
            row.status = "superseded"
            changed.append(row)
    if changed:
        await db.flush()
    return changed

async def create_knowledge_node(
    db: AsyncSession,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    node_type: str,
    title: str,
    order_index: int,
    parent_id: Optional[uuid.UUID] = None,
    model_version: Optional[int] = None,
    pages: Optional[list] = None,
    chunks: Optional[list] = None,
    aliases: Optional[list] = None,
    confidence: Optional[float] = None,
    metadata: Optional[dict] = None,
):
    """Create one spine node (source order authoritative via order_index)."""
    from database.models.knowledge import KnowledgeNode

    node = KnowledgeNode(
        id=uuid.uuid4(),
        project_id=project_id,
        document_id=document_id,
        parent_id=parent_id,
        model_version=model_version,
        type=node_type,
        title=title,
        order_index=order_index,
        pages=list(pages or []),
        chunks=list(chunks or []),
        aliases=list(aliases or []),
        confidence=confidence,
        node_metadata=dict(metadata or {}),
    )
    db.add(node)
    await db.flush()
    return node


async def get_knowledge_node(db: AsyncSession, node_id: uuid.UUID):
    """Get one spine node by id (or None)."""
    from database.models.knowledge import KnowledgeNode

    return await db.get(KnowledgeNode, node_id)


async def list_knowledge_nodes(
    db: AsyncSession,
    document_id: Optional[uuid.UUID] = None,
    project_id: Optional[uuid.UUID] = None,
):
    """List spine nodes in document order (parent first, then order_index).

    Null-parent roots sort first; siblings follow dense order_index. The
    tree itself is assembled caller-side from parent_id links.
    """
    from sqlalchemy import case

    from database.models.knowledge import KnowledgeNode

    stmt = select(KnowledgeNode)
    if document_id is not None:
        stmt = stmt.where(KnowledgeNode.document_id == document_id)
    if project_id is not None:
        stmt = stmt.where(KnowledgeNode.project_id == project_id)
    stmt = stmt.order_by(
        case((KnowledgeNode.parent_id.is_(None), 0), else_=1),
        KnowledgeNode.order_index.asc(),
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_or_create_canonical_topic(
    db: AsyncSession,
    canonical_key: str,
    name: str,
    aliases: Optional[list] = None,
):
    """Fetch the registry row for canonical_key or insert it.

    Append-only: unseen aliases are appended to an existing row; `name`
    is never overwritten by a second build.
    """
    from database.models.knowledge import CanonicalTopic
    from services.academic.spine import merge_alias_lists

    result = await db.execute(
        select(CanonicalTopic).where(CanonicalTopic.canonical_key == canonical_key)
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = CanonicalTopic(
            id=uuid.uuid4(),
            canonical_key=canonical_key,
            name=name,
            aliases=list(aliases or []),
        )
        db.add(row)
        await db.flush()
        return row
    merged = merge_alias_lists(row.aliases, aliases)
    if merged != list(row.aliases or []):
        row.aliases = merged
        await db.flush()
    return row


async def find_canonical_by_alias(db: AsyncSession, normalized_alias: str):
    """Find the registry row claiming a normalized alias (or None).

    Portable Python-side scan (no PG-only JSONB operator): the shared
    registry is small and this keeps SQLite/test doubles working. The
    GIN indexes from migration 009 accelerate the PG path once queries
    use containment operators.
    """
    from database.models.knowledge import CanonicalTopic
    from services.academic.spine import normalize_key

    wanted = normalize_key(normalized_alias)
    if not wanted:
        return None
    result = await db.execute(select(CanonicalTopic))
    for row in result.scalars().all():
        if normalize_key(row.canonical_key) == wanted:
            return row
        for alias in row.aliases or []:
            if normalize_key(alias) == wanted:
                return row
    return None


async def set_node_canonical(
    db: AsyncSession,
    node_id: uuid.UUID,
    canonical_id: Optional[uuid.UUID],
):
    """Link (or unlink) a node to its canonical registry row."""
    node = await get_knowledge_node(db, node_id)
    if node is None:
        return None
    node.canonical_topic_id = canonical_id
    await db.flush()
    return node


# ==================== STEPS 2-4 PERSISTENCE (additive, no migration) =======
#
# Spine rows land in knowledge_nodes (migration 009, reused as-is); the
# hierarchical notes artifact for a document is stored on its document
# root's node_metadata["notes_artifact"] (JSONB, no schema change, so no
# new migration and the PG outputtype enum stays untouched). Helpers flush;
# the caller owns the commit. Pure converters (spine_spec_to_rows,
# rows_to_tree) are DB-free and unit-tested offline.


def spine_spec_to_rows(
    spec: Any,
    project_id: Any,
    document_id: Any,
    model_version: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Flatten a validated KnowledgeSpineSpec to insert-ready row dicts.

    Rows keep spine order (parents before children: parent_ref always
    points backwards), so persist_spine can insert sequentially and map
    parent_ref -> generated ids. Pure: no DB, no LLM.
    """
    nodes = spec.nodes if hasattr(spec, "nodes") else spec.get("nodes", [])
    rows: List[Dict[str, Any]] = []
    for node in nodes:
        if hasattr(node, "model_dump"):
            payload = node.model_dump()
            node_type = str(payload.get("node_type"))
            if hasattr(node.node_type, "value"):
                node_type = node.node_type.value
            rows.append({
                "project_id": project_id,
                "document_id": document_id,
                "model_version": model_version,
                "parent_ref": payload.get("parent_ref"),
                "type": node_type,
                "title": payload.get("title"),
                "order_index": payload.get("order_index"),
                "pages": list(payload.get("pages") or []),
                "chunks": list(payload.get("chunks") or []),
                "aliases": list(payload.get("aliases") or []),
                "confidence": payload.get("confidence"),
                "node_metadata": dict(payload.get("node_metadata") or {}),
                "canonical_key": payload.get("canonical_key"),
            })
        else:
            rows.append({
                "project_id": project_id,
                "document_id": document_id,
                "model_version": model_version,
                "parent_ref": node.get("parent_ref"),
                "type": str(node.get("node_type")),
                "title": node.get("title"),
                "order_index": node.get("order_index"),
                "pages": list(node.get("pages") or []),
                "chunks": list(node.get("chunks") or []),
                "aliases": list(node.get("aliases") or []),
                "confidence": node.get("confidence"),
                "node_metadata": dict(node.get("node_metadata") or {}),
                "canonical_key": node.get("canonical_key"),
            })
    return rows


def rows_to_tree(rows: List[Any]) -> Optional[Dict[str, Any]]:
    """Nest flat spine rows into a document-ordered tree (pure).

    Accepts ORM nodes (attribute access) or plain dicts. Children follow
    dense order_index; the root is the single parent-less row.
    """
    def _get(row: Any, name: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(name, default)
        return getattr(row, name, default)

    def _id(value: Any) -> Any:
        return str(value) if value is not None else None

    entries: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for row in rows or []:
        rid = _id(_get(row, "id"))
        entries[rid] = {
            "id": rid,
            "parent_id": _id(_get(row, "parent_id")),
            "type": _get(row, "type"),
            "title": _get(row, "title"),
            "order_index": _get(row, "order_index", 0),
            "pages": list(_get(row, "pages") or []),
            "chunks": list(_get(row, "chunks") or []),
            "children": [],
        }
        order.append(rid)
    roots = [e for e in entries.values() if not e["parent_id"]]
    if len(roots) != 1:
        return None
    for entry in entries.values():
        parent = entry["parent_id"]
        if parent and parent in entries:
            entries[parent]["children"].append(entry)
    for entry in entries.values():
        entry["children"].sort(key=lambda c: (c["order_index"], c["title"] or ""))
    return roots[0]


async def persist_spine(
    db: AsyncSession,
    project_id: uuid.UUID,
    document_id: uuid.UUID,
    spec: Any,
    model_version: Optional[int] = None,
) -> List[Any]:
    """Insert a validated spine spec as knowledge_nodes (source order).

    Idempotency is caller-side via DocumentStructure.fingerprint: rebuilding
    the same sources with the same fingerprint must skip this call. Topic
    and unit nodes are linked to append-only canonical registry rows.
    Helpers flush; the caller owns the commit.
    """
    from services.academic.spine import canonical_key_for

    rows = spine_spec_to_rows(spec, project_id, document_id, model_version)
    created: List[Any] = []
    for row in rows:
        canonical_key = row.pop("canonical_key", None)
        node = await create_knowledge_node(
            db,
            project_id=row["project_id"],
            document_id=row["document_id"],
            node_type=row["type"],
            title=row["title"],
            order_index=row["order_index"],
            model_version=row.get("model_version"),
            parent_id=(
                created[row["parent_ref"]].id
                if row.get("parent_ref") is not None else None
            ),
            pages=row["pages"],
            chunks=row["chunks"],
            aliases=row["aliases"],
            confidence=row["confidence"],
            metadata=row["node_metadata"],
        )
        if row["type"] in ("unit", "topic") and node.title:
            try:
                key = canonical_key or canonical_key_for(node.title)
                registry = await get_or_create_canonical_topic(
                    db, key, node.title, list(node.aliases or []),
                )
                node.canonical_topic_id = registry.id
                await db.flush()
            except Exception:
                pass
        created.append(node)
    return created


async def save_notes_artifact(
    db: AsyncSession,
    document_node_id: uuid.UUID,
    artifact: Dict[str, Any],
) -> Any:
    """Store the hierarchical notes artifact on the document root's metadata.

    Additive JSONB write (node_metadata["notes_artifact"]); no new table,
    no migration. Overwrites only the artifact key, preserving the rest of
    the metadata. Helpers flush; the caller owns the commit.
    """
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        raise ValueError(f"spine document node {document_node_id} not found")
    metadata = dict(getattr(node, "node_metadata", None) or {})
    metadata["notes_artifact"] = dict(artifact)
    node.node_metadata = metadata
    await db.flush()
    return node


async def get_notes_artifact(
    db: AsyncSession,
    document_node_id: uuid.UUID,
) -> Optional[Dict[str, Any]]:
    """Read the stored notes artifact (or None when not yet generated)."""
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        return None
    metadata = dict(getattr(node, "node_metadata", None) or {})
    artifact = metadata.get("notes_artifact")
    return dict(artifact) if isinstance(artifact, dict) else None


# ==================== STEPS 8-10 GENERATED ARTIFACTS =====================
#
# Versioned deliverable records (study-notes PDF, insights payload) ride
# the same document-root node_metadata pattern as Steps 2-7:
# node_metadata["generated_artifacts"][artifact_type] is a version list of
# GeneratedArtifact dicts (see services/academic/types.py). No new table,
# no migration, no redundant generated_artifacts concept: the PDF bytes
# live in the storage backend under the artifact's `path` (an opaque
# storage key, the same vocabulary as Document.file_path), never in the
# DB. Helpers flush; the caller owns the commit.

GENERATED_ARTIFACTS_KEY = "generated_artifacts"

NOTES_PDF_ARTIFACT_TYPE = "notes_pdf"
INSIGHTS_ARTIFACT_TYPE = "insights"


def assign_artifact_version(
    existing: List[Dict[str, Any]],
    fingerprint: str,
) -> tuple:
    """Next version for one artifact type, with fingerprint short-circuit.

    Pure (no DB): when an existing READY version carries this fingerprint,
    return (its version, the matching dict) so the rebuild is a no-op (no
    new version, no new PDF, no provider calls). Otherwise return
    (max version + 1, None). A requested old version stays readable: the
    list is append-only, never rewritten.
    """
    ordered = [dict(v) for v in (existing or [])]
    for entry in ordered:
        if entry.get("status") == "ready" and entry.get("fingerprint") == fingerprint:
            return entry.get("version", 1), entry
    highest = 0
    for entry in ordered:
        try:
            highest = max(highest, int(entry.get("version", 0) or 0))
        except (TypeError, ValueError):
            continue
    return highest + 1, None


async def save_generated_artifact(
    db: AsyncSession,
    document_node_id: uuid.UUID,
    artifact: Dict[str, Any],
) -> Any:
    """Append (or refresh) one GeneratedArtifact version on the doc root.

    Overwrites only the generated_artifacts key, preserving the rest of
    the metadata. A superseded previous READY version of the same type is
    marked with superseded_by pointing at the new version (both stay
    readable). Helpers flush; the caller owns the commit.
    """
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        raise ValueError(f"spine document node {document_node_id} not found")
    metadata = dict(getattr(node, "node_metadata", None) or {})
    store = dict(metadata.get(GENERATED_ARTIFACTS_KEY) or {})
    artifact_type = str(artifact.get("artifact_type") or artifact.get("type") or "")
    if not artifact_type:
        raise ValueError("generated artifact requires an artifact_type")
    versions = [dict(v) for v in (store.get(artifact_type) or [])]
    incoming = dict(artifact)
    replaced = False
    for pos, entry in enumerate(versions):
        if entry.get("version") == incoming.get("version"):
            versions[pos] = incoming
            replaced = True
    if not replaced:
        for entry in versions:
            if entry.get("status") == "ready" and incoming.get("status") == "ready":
                entry["status"] = "superseded"
                entry["superseded_by"] = incoming.get("version")
        versions.append(incoming)
    versions.sort(key=lambda v: int(v.get("version", 0) or 0))
    store[artifact_type] = versions
    metadata[GENERATED_ARTIFACTS_KEY] = store
    node.node_metadata = metadata
    await db.flush()
    return node


async def get_generated_artifact(
    db: AsyncSession,
    document_node_id: uuid.UUID,
    artifact_type: str,
    version: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Read one generated artifact version (latest READY by default).

    version=None returns the latest READY version (falling back to the
    latest version of any status); an explicit version returns that
    version even after supersession. None when nothing is stored.
    """
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        return None
    metadata = dict(getattr(node, "node_metadata", None) or {})
    store = dict(metadata.get(GENERATED_ARTIFACTS_KEY) or {})
    versions = [dict(v) for v in (store.get(artifact_type) or [])]
    if not versions:
        return None
    if version is not None:
        for entry in versions:
            if int(entry.get("version", 0) or 0) == int(version):
                return entry
        return None
    ready = [v for v in versions if v.get("status") == "ready"]
    pool = ready or versions
    return max(pool, key=lambda v: int(v.get("version", 0) or 0))


async def list_generated_artifacts(
    db: AsyncSession,
    document_node_id: uuid.UUID,
    artifact_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List stored artifact versions (version order, oldest first)."""
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        return []
    metadata = dict(getattr(node, "node_metadata", None) or {})
    store = dict(metadata.get(GENERATED_ARTIFACTS_KEY) or {})
    out: List[Dict[str, Any]] = []
    for stored_type, versions in store.items():
        if artifact_type is not None and stored_type != artifact_type:
            continue
        for entry in versions or []:
            row = dict(entry)
            row.setdefault("artifact_type", stored_type)
            out.append(row)
    out.sort(key=lambda v: (str(v.get("artifact_type") or ""), int(v.get("version", 0) or 0)))
    return out


async def delete_generated_artifact_version(
    db: AsyncSession,
    document_node_id: uuid.UUID,
    artifact_type: str,
    artifact_id: Optional[str] = None,
    version: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Remove one stored artifact version; return the removed record or None.

    Read/ownership scoping is the caller's job (the route resolves the
    artifact first). When the removed version was the only READY one, the
    newest remaining version is promoted to READY so project-level reads
    never resolve to a superseded (or missing) artifact.
    """
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        return None
    metadata = dict(getattr(node, "node_metadata", None) or {})
    store = dict(metadata.get(GENERATED_ARTIFACTS_KEY) or {})
    versions = [dict(v) for v in (store.get(artifact_type) or [])]
    if not versions:
        return None
    wanted = str(artifact_id) if artifact_id is not None else None
    removed: Optional[Dict[str, Any]] = None
    remaining: List[Dict[str, Any]] = []
    for entry in versions:
        matches = wanted is not None and str(entry.get("id")) == wanted
        if not matches and version is not None:
            try:
                matches = int(entry.get("version", 0) or 0) == int(version)
            except (TypeError, ValueError):
                matches = False
        if matches and removed is None:
            removed = dict(entry)
            continue
        remaining.append(entry)
    if removed is None:
        return None
    if removed.get("status") == "ready" and not any(
        entry.get("status") == "ready" for entry in remaining
    ):
        if remaining:
            newest = max(remaining, key=lambda v: int(v.get("version", 0) or 0))
            newest["status"] = "ready"
            newest.pop("superseded_by", None)
    remaining.sort(key=lambda v: int(v.get("version", 0) or 0))
    store[artifact_type] = remaining
    metadata[GENERATED_ARTIFACTS_KEY] = store
    node.node_metadata = metadata
    await db.flush()
    return removed


# ==================== STEPS 5-7 ARTIFACTS (additive, no migration) =========
#
# Flashcards, MCQ and metrics artifacts ride the same document-root
# node_metadata pattern as notes (keys below), so no new table, no new
# migration, and the PG outputtype enum stays untouched. Rich typed sets
# live here; the Flashcards UI and Practice UI keep consuming the
# existing deck/quiz projections (to_deck_cards / project_quiz_questions).
# Helpers flush; the caller owns the commit.

FLASHCARDS_ARTIFACT_KEY = "flashcards_artifact"
MCQ_ARTIFACT_KEY = "mcq_artifact"
METRICS_ARTIFACT_KEY = "metrics_artifact"


async def save_academic_artifact(
    db: AsyncSession,
    document_node_id: uuid.UUID,
    key: str,
    artifact: Dict[str, Any],
) -> Any:
    """Store a Step 5-7 artifact on the document root's metadata.

    Overwrites only the artifact key, preserving the rest of the
    metadata. Helpers flush; the caller owns the commit.
    """
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        raise ValueError(f"spine document node {document_node_id} not found")
    metadata = dict(getattr(node, "node_metadata", None) or {})
    metadata[key] = dict(artifact)
    node.node_metadata = metadata
    await db.flush()
    return node


async def get_academic_artifact(
    db: AsyncSession,
    document_node_id: uuid.UUID,
    key: str,
) -> Optional[Dict[str, Any]]:
    """Read a Step 5-7 artifact (or None when not yet generated)."""
    node = await get_knowledge_node(db, document_node_id)
    if node is None:
        return None
    metadata = dict(getattr(node, "node_metadata", None) or {})
    artifact = metadata.get(key)
    return dict(artifact) if isinstance(artifact, dict) else None
