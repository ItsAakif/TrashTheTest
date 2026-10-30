"""Academic spine Pydantic schemas (STEP 1, additive).

Snake+camel compatibility: request models accept both `project_id` and
`projectId` (populate_by_name). Every new model is extra="forbid".
Source document order is authoritative: order_index is dense per parent.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class NodeType(str, Enum):
    """Spine hierarchy levels, root to leaf."""

    DOCUMENT = "document"
    TOPIC = "topic"
    UNIT = "unit"
    SUBTOPIC = "subtopic"
    CONCEPT = "concept"


# Depth contract: a child's depth is exactly parent depth + 1, and the
# node type must match its depth. Documented here so validators share it.
NODE_DEPTH = {
    NodeType.DOCUMENT: 0,
    NodeType.UNIT: 1,
    NodeType.TOPIC: 2,
    NodeType.SUBTOPIC: 3,
    NodeType.CONCEPT: 4,
}

# Ordered from root to leaf; index equals depth.
TYPE_ORDER = [
    NodeType.DOCUMENT,
    NodeType.UNIT,
    NodeType.TOPIC,
    NodeType.SUBTOPIC,
    NodeType.CONCEPT,
]


class KnowledgeNodeSpec(BaseModel):
    """One validated spine node (typed flesh over the JSONB columns)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    node_type: NodeType
    title: str = Field(min_length=1, max_length=500)
    aliases: List[str] = Field(default_factory=list)
    order_index: int = Field(ge=0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    pages: List[int] = Field(default_factory=list)
    chunks: List[str] = Field(default_factory=list)
    node_metadata: Dict[str, Any] = Field(default_factory=dict)
    # Index of the parent node inside KnowledgeSpineSpec.nodes (None = root).
    parent_ref: Optional[int] = Field(default=None, ge=0)
    canonical_key: Optional[str] = Field(default=None, min_length=1)

    @field_validator("aliases")
    @classmethod
    def _aliases_deduped(cls, v: List[str]) -> List[str]:
        seen: set = set()
        out: List[str] = []
        for alias in v:
            key = (alias or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(alias)
        return out

    @field_validator("pages")
    @classmethod
    def _pages_non_negative(cls, v: List[int]) -> List[int]:
        if any(p < 0 for p in v):
            raise ValueError("pages must hold non-negative page numbers")
        return v


class KnowledgeSpineSpec(BaseModel):
    """A validated node list forming one document spine.

    Hierarchy validation rejects (with the offending node named): a second
    root, a parent_ref out of range, a cycle, a depth contradicting the
    node type, a duplicate (parent_ref, order_index), and a duplicate
    canonical_key among siblings. Dense order_index per parent is required.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(min_length=1)
    nodes: List[KnowledgeNodeSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _hierarchy_valid(self) -> "KnowledgeSpineSpec":
        nodes = self.nodes
        roots = [i for i, n in enumerate(nodes) if n.parent_ref is None]
        if len(roots) != 1:
            raise ValueError(
                f"spine must have exactly one root, found {len(roots)}"
            )
        root = roots[0]
        if nodes[root].node_type != NodeType.DOCUMENT:
            raise ValueError(
                f"node {root} ({nodes[root].title!r}): root must be "
                "node_type 'document'"
            )
        # Parent refs must point backwards (document order) and in range;
        # backward-only refs make cycles structurally impossible, and the
        # chain walk below still verifies depth consistency.
        for i, node in enumerate(nodes):
            if node.parent_ref is None:
                continue
            if node.parent_ref >= len(nodes):
                raise ValueError(
                    f"node {i} ({node.title!r}): parent_ref "
                    f"{node.parent_ref} out of range"
                )
            if node.parent_ref >= i:
                raise ValueError(
                    f"node {i} ({node.title!r}): parent_ref must point to "
                    "an earlier node (document order, no cycles)"
                )
        # Depth must match node type along each parent chain.
        depths: Dict[int, int] = {}

        def _depth(idx: int, trail: tuple) -> int:
            if idx in depths:
                return depths[idx]
            if idx in trail:
                raise ValueError(
                    f"node {idx} ({nodes[idx].title!r}): cycle detected "
                    f"in parent chain {list(trail) + [idx]}"
                )
            node = nodes[idx]
            if node.parent_ref is None:
                depths[idx] = 0
            else:
                depths[idx] = _depth(node.parent_ref, trail + (idx,)) + 1
            expected = NODE_DEPTH[node.node_type]
            if depths[idx] != expected:
                raise ValueError(
                    f"node {idx} ({node.title!r}): depth {depths[idx]} "
                    f"contradicts node_type {node.node_type.value!r} "
                    f"(expected depth {expected})"
                )
            return depths[idx]

        for i in range(len(nodes)):
            _depth(i, ())
        # Duplicate (parent_ref, order_index) and dense ord per parent.
        by_parent: Dict[Optional[int], List[int]] = {}
        for i, node in enumerate(nodes):
            key = node.parent_ref
            by_parent.setdefault(key, []).append(i)
        for parent, members in by_parent.items():
            ords = sorted(nodes[m].order_index for m in members)
            if len(set(ords)) != len(ords):
                raise ValueError(
                    f"duplicate order_index under parent {parent}: {ords}"
                )
            if ords != list(range(len(members))):
                raise ValueError(
                    f"sparse order_index under parent {parent}: {ords} "
                    "(source order is authoritative, dense from 0)"
                )
        # Duplicate canonical_key among siblings.
        for parent, members in by_parent.items():
            seen: Dict[str, int] = {}
            for m in members:
                key = nodes[m].canonical_key
                if key is None:
                    continue
                if key in seen:
                    raise ValueError(
                        f"node {m} ({nodes[m].title!r}): duplicate "
                        f"canonical_key {key!r} among siblings of parent "
                        f"{parent} (first at node {seen[key]})"
                    )
                seen[key] = m
        return self


class KnowledgeSpineBuildRequest(BaseModel):
    """Request to build a spine (snake+camel compatible)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    project_id: str = Field(min_length=1, alias="projectId")
    document_ids: List[str] = Field(default_factory=list)


class CanonicalResolveResult(BaseModel):
    """Outcome of resolving one node title against the shared registry."""

    model_config = ConfigDict(extra="forbid")

    canonical_key: Optional[str] = None
    resolved: bool = False
    via: str = "unresolved"  # direct_key | alias | created | unresolved
    collision: Optional[str] = None


class SpineStats(BaseModel):
    """Counts per node type plus unresolved registry nodes."""

    model_config = ConfigDict(extra="forbid")

    node_count: Dict[str, int] = Field(default_factory=dict)
    unresolved_count: int = Field(ge=0)
    canonical_collisions: int = Field(ge=0)


# ---------------------------------------------------------------------------
# STEP 2 + STEP 3 + STEP 4: extraction structure, hierarchical notes,
# coverage. Every model is extra="forbid". Snake+camel compatible where a
# request crosses the API boundary (populate_by_name + aliases).
# ---------------------------------------------------------------------------


class StructureMethod(str, Enum):
    """How one structure node was detected (recorded in provenance)."""

    TOC_HEADING = "toc_heading"
    HEADING_GROUP = "heading_group"
    INFERRED_SPLIT = "inferred_split"
    SINGLE_CHUNK = "single_chunk"


class ConceptProvenance(BaseModel):
    """Provenance for one extracted concept (every concept carries one)."""

    model_config = ConfigDict(extra="forbid")

    method: StructureMethod
    source_heading: Optional[str] = None
    chunk_ids: List[str] = Field(min_length=1)
    note: str = Field(min_length=1)


class ExtractedConcept(BaseModel):
    """One leaf concept: exactly one source chunk, full provenance."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    pages: List[int] = Field(default_factory=list)
    chunks: List[str] = Field(min_length=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    provenance: ConceptProvenance


class ExtractedSubtopic(BaseModel):
    """Conservative subtopic: one per topic unless headings say otherwise."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    pages: List[int] = Field(default_factory=list)
    chunks: List[str] = Field(default_factory=list)
    concepts: List[ExtractedConcept] = Field(default_factory=list)
    method: StructureMethod = StructureMethod.HEADING_GROUP


class ExtractedTopic(BaseModel):
    """One topic in source order with its chunks and subtopics."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    aliases: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    chunks: List[str] = Field(default_factory=list)
    subtopics: List[ExtractedSubtopic] = Field(default_factory=list)
    concepts: List[ExtractedConcept] = Field(default_factory=list)
    method: StructureMethod = StructureMethod.HEADING_GROUP


class StructuredUnit(BaseModel):
    """One unit in source order (single unit when no unit markers exist)."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)
    aliases: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    topics: List[ExtractedTopic] = Field(min_length=1)
    method: StructureMethod = StructureMethod.HEADING_GROUP


class DocumentStructure(BaseModel):
    """STEP 2 output: ordered units/topics(/subtopics)/concepts for one doc.

    Source order is authoritative throughout: units, topics, subtopics and
    concepts all follow first-seen source order. No notes prose here --
    extraction runs BEFORE generation.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    units: List[StructuredUnit] = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    source_type: str = "study_material"
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_spine_spec(self) -> "KnowledgeSpineSpec":
        """Convert to validated spine rows (document/unit/topic/subtopic/concept)."""
        nodes: List[Dict[str, Any]] = [
            {
                "node_type": NodeType.DOCUMENT,
                "title": self.title,
                "parent_ref": None,
                "order_index": 0,
            }
        ]
        for unit_pos, unit in enumerate(self.units):
            unit_ref = len(nodes)
            nodes.append({
                "node_type": NodeType.UNIT,
                "title": unit.title,
                "aliases": list(unit.aliases),
                "parent_ref": 0,
                "order_index": unit_pos,
                "pages": list(unit.pages),
                "chunks": _union_in_order(
                    [t.chunks for t in unit.topics]
                ),
                "node_metadata": {"method": unit.method.value},
            })
            # Dense per-parent ordinals are positional: topics count from 0
            # under their unit, subtopics from 0 under their topic, and so on.
            for topic_pos, topic in enumerate(unit.topics):
                topic_ref = len(nodes)
                nodes.append({
                    "node_type": NodeType.TOPIC,
                    "title": topic.title,
                    "aliases": list(topic.aliases),
                    "parent_ref": unit_ref,
                    "order_index": topic_pos,
                    "pages": list(topic.pages),
                    "chunks": list(topic.chunks),
                    "node_metadata": {"method": topic.method.value},
                })
                subtopics = topic.subtopics or [
                    ExtractedSubtopic(
                        title=topic.title,
                        pages=list(topic.pages),
                        chunks=list(topic.chunks),
                        concepts=list(topic.concepts),
                        method=topic.method,
                    )
                ]
                if topic.concepts and topic.subtopics:
                    # Topic-level concepts with explicit subtopics: keep them
                    # under the first subtopic so no concept is ever dropped.
                    first = subtopics[0].model_copy(
                        update={"concepts": list(subtopics[0].concepts) + list(topic.concepts)}
                    )
                    subtopics = [first] + subtopics[1:]
                for sub_pos, sub in enumerate(subtopics):
                    sub_ref = len(nodes)
                    nodes.append({
                        "node_type": NodeType.SUBTOPIC,
                        "title": sub.title,
                        "parent_ref": topic_ref,
                        "order_index": sub_pos,
                        "pages": list(sub.pages),
                        "chunks": list(sub.chunks),
                        "node_metadata": {"method": sub.method.value},
                    })
                    concepts = sub.concepts
                    for concept_pos, concept in enumerate(concepts):
                        nodes.append({
                            "node_type": NodeType.CONCEPT,
                            "title": concept.title,
                            "parent_ref": sub_ref,
                            "order_index": concept_pos,
                            "pages": list(concept.pages),
                            "chunks": list(concept.chunks),
                            "confidence": concept.confidence,
                            "node_metadata": {
                                "method": concept.provenance.method.value,
                                "provenance_note": concept.provenance.note,
                            },
                        })
        return KnowledgeSpineSpec(title=self.title, nodes=nodes)  # type: ignore[arg-type]


def _union_in_order(lists: List[List[str]]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for lst in lists:
        for item in lst or []:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


class TopicShape(str, Enum):
    """Adaptive note shape per topic kind (STEP 3 content contract)."""

    CONCEPT = "concept"
    ALGORITHM = "algorithm"
    FORMULA = "formula"
    COMPARISON = "comparison"


class NotesGenerationMode(str, Enum):
    """User-facing notes modes. Values are wire-safe lowercase strings."""

    QUICK = "quick"
    DETAILED = "detailed"
    EXAM = "exam"
    MEMORY = "memory"
    COMPLETE = "complete"


class NotesDetailLevel(str, Enum):
    CONCISE = "concise"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"


class NotesScope(str, Enum):
    ENTIRE = "entire"
    SELECTED = "selected"


class NotesGenerationConfig(BaseModel):
    """Explicit notes controls persisted with the job and artifact.

    Recommendations are advisory metadata and never mutate this object.  Web
    research is opt-in and uploaded source material remains primary.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    mode: NotesGenerationMode = NotesGenerationMode.DETAILED
    detail: NotesDetailLevel = NotesDetailLevel.BALANCED
    web: bool = Field(
        default=False,
        validation_alias=AliasChoices("web", "use_web", "useWeb", "web_opt_in", "webOptIn"),
    )
    exam_focus: bool = Field(
        default=False,
        validation_alias=AliasChoices("exam_focus", "examFocus"),
    )
    memory_aids: bool = Field(
        default=False,
        validation_alias=AliasChoices("memory_aids", "memoryAids"),
    )
    scope: NotesScope = NotesScope.ENTIRE
    selected_topic_ids: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("selected_topic_ids", "selectedTopicIds"),
    )
    topic_budget_tokens: int = Field(default=20000, ge=256, le=100000)
    # Per-topic generation budgets. ``topic_batch_tokens`` bounds the evidence
    # handed to ONE topic call (larger topics are split on chunk boundaries);
    # ``topic_output_tokens`` is that call's output budget. Both are explicit
    # so a topic is never asked for more than the provider can return.
    topic_batch_tokens: int = Field(default=12000, ge=256, le=100000)
    topic_output_tokens: int = Field(default=8192, ge=512, le=100000)
    # Optional per-topic execution bounds.  None keeps the module defaults
    # (env-overridable): a per-topic timeout so a slow provider cannot hang
    # the job, and a small generation pool so topics are not paid for one at
    # a time.  Additive: existing callers that omit them keep working.
    topic_timeout_s: Optional[float] = None
    topic_concurrency: Optional[int] = None
    recommendation: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("mode", mode="before")
    @classmethod
    def _normalise_mode(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @field_validator("detail", mode="before")
    @classmethod
    def _normalise_detail(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @field_validator("scope", mode="before")
    @classmethod
    def _normalise_scope(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @field_validator("selected_topic_ids")
    @classmethod
    def _dedupe_topic_ids(cls, values: List[str]) -> List[str]:
        seen = set()
        out = []
        for value in values:
            key = str(value or "").strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def persisted_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode="json", by_alias=False)


def recommend_notes_config(
    topic_count: int,
    has_question_papers: bool = False,
    has_syllabus: bool = True,
) -> NotesGenerationConfig:
    """Deterministic advisory recommendation; callers retain user choice."""

    if topic_count <= 3:
        mode, detail = NotesGenerationMode.QUICK, NotesDetailLevel.BALANCED
    elif has_question_papers:
        mode, detail = NotesGenerationMode.EXAM, NotesDetailLevel.COMPREHENSIVE
    elif has_syllabus and topic_count >= 12:
        mode, detail = NotesGenerationMode.COMPLETE, NotesDetailLevel.COMPREHENSIVE
    else:
        mode, detail = NotesGenerationMode.DETAILED, NotesDetailLevel.BALANCED
    return NotesGenerationConfig(
        mode=mode,
        detail=detail,
        exam_focus=has_question_papers,
        memory_aids=False,
        web=False,
        recommendation={
            "recommended_mode": mode.value,
            "recommended_detail": detail.value,
            "reason": "based on the number of detected topics and source types",
            "advisory": True,
        },
    )


class TopicNotes(BaseModel):
    """STEP 3 typed per-topic notes: new prose + cites, never bare copying."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    unit_title: str = Field(min_length=1)
    topic_title: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    body_paragraphs: List[str] = Field(default_factory=list)
    key_terms: List[str] = Field(default_factory=list)
    shape: TopicShape = TopicShape.CONCEPT
    shape_extras: Dict[str, Any] = Field(default_factory=dict)
    citations: List["TopicCitation"] = Field(default_factory=list)
    no_source: bool = False
    sufficiency: Optional[str] = None
    source_gap_reason: Optional[str] = None
    provenance_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _cited_or_honest(self) -> "TopicNotes":
        if not self.citations and not self.no_source:
            raise ValueError(
                f"topic {self.topic_title!r}: empty citations require "
                "no_source=True (never silently uncited)"
            )
        return self


class TopicCitation(BaseModel):
    """One chunk backing a TopicNotes section (span occurs in the chunk)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    span: str = Field(min_length=1)


class NotesSection(BaseModel):
    """One ordered synthesis section (Unit -> Topic -> Subtopic)."""

    model_config = ConfigDict(extra="forbid")

    unit_title: str = Field(min_length=1)
    topic_title: str = Field(min_length=1)
    subtopic_title: Optional[str] = None
    heading: str = Field(min_length=1)
    content: str = Field(min_length=1)
    citations: List[TopicCitation] = Field(default_factory=list)
    # ADR-008 T006: forwarded structure verdict of the cited chunks
    # (structure_method / structure_confidence).  Absent for healthy,
    # heading-derived evidence, which keeps the topic_mismatch gate strict.
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PerTopicScore(BaseModel):
    """STEP 4 coverage score for one topic (source concepts -> represented)."""

    model_config = ConfigDict(extra="forbid")

    unit_title: str = Field(min_length=1)
    topic_title: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    represented_concepts: int = Field(ge=0)
    total_concepts: int = Field(ge=0)
    missing: List[str] = Field(default_factory=list)
    citations_ok: bool = True


class CoverageReport(BaseModel):
    """STEP 4 deterministic coverage verdict (no page-count metric)."""

    model_config = ConfigDict(extra="forbid")

    topic_coverage: float = Field(ge=0.0, le=1.0)
    concept_coverage: float = Field(ge=0.0, le=1.0)
    per_topic: List[PerTopicScore] = Field(default_factory=list)
    uncovered: List[str] = Field(default_factory=list)
    passes: bool = False
    threshold: float = Field(ge=0.0, le=1.0)
    nli: str = "unavailable"


class RepairRecord(BaseModel):
    """One bounded-repair pass decision (observable, revertible)."""

    model_config = ConfigDict(extra="forbid")

    pass_no: int = Field(ge=1)
    regenerated_topics: List[str] = Field(default_factory=list)
    coverage_before: float = Field(ge=0.0, le=1.0)
    coverage_after: float = Field(ge=0.0, le=1.0)
    kept: bool = True
    note: str = Field(min_length=1)


class NotesArtifact(BaseModel):
    """STEP 3 final artifact: ordered synthesis + coverage (no page limit)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(min_length=1)
    sections: List[NotesSection] = Field(min_length=1)
    cross_topic_notes: List[str] = Field(default_factory=list)
    coverage: Optional[CoverageReport] = None
    uncovered: List[str] = Field(default_factory=list)
    source_gaps: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    observability: Dict[str, Any] = Field(default_factory=dict)
    repair_log: List[RepairRecord] = Field(default_factory=list)
    provenance_note: str = Field(min_length=1)


class HierarchicalNotesRequest(BaseModel):
    """Request to run hierarchical notes (snake+camel compatible)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    project_id: str = Field(min_length=1, alias="projectId")
    document_ids: List[str] = Field(default_factory=list)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


TopicNotes.model_rebuild()
NotesSection.model_rebuild()
NotesArtifact.model_rebuild()


# ---------------------------------------------------------------------------
# STEP 5 + STEP 6 + STEP 7: typed flashcards (spine-bound, Bloom-allocated,
# FSRS-ready), MCQ matrix (Topic x Bloom, error-tagged distractors), and
# deterministic intelligence metrics + grounded narrative. Every model is
# extra="forbid". Flashcards are built FROM THE SPINE (DocumentStructure
# concepts), never from raw PDF text: every card carries a concept_id that
# resolves into the structure's concept index. FSRS fields are stored with
# documented defaults only; no scheduling logic lives here (deferred work).
# ---------------------------------------------------------------------------


class BloomLevel(str, Enum):
    """Bloom cognitive levels (six-level ADR-007 D4 default)."""

    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


# Default Bloom allocation for flashcard decks. Sums to 1.0 (asserted at
# import; OQ-1 resolution per ADR-007 D4). Configurable per call.
BLOOM_MIX: Dict[str, float] = {
    BloomLevel.REMEMBER.value: 0.30,
    BloomLevel.UNDERSTAND.value: 0.30,
    BloomLevel.APPLY.value: 0.20,
    BloomLevel.ANALYZE.value: 0.10,
    BloomLevel.EVALUATE.value: 0.05,
    BloomLevel.CREATE.value: 0.05,
}

_BLOOM_TOTAL = sum(BLOOM_MIX.values())
assert abs(_BLOOM_TOTAL - 1.0) < 1e-9, (
    f"BLOOM_MIX must sum to 1.0, got {_BLOOM_TOTAL}"
)
del _BLOOM_TOTAL


class CardType(str, Enum):
    """Typed flashcard kinds (STEP 5 contract)."""

    DEFINITION = "definition"
    CONCEPTUAL = "conceptual"
    PROCESS = "process"
    FORMULA = "formula"
    COMPARISON = "comparison"
    CAUSE_EFFECT = "cause-effect"
    APPLICATION = "application"
    MISCONCEPTION = "misconception"


class CardDifficulty(str, Enum):
    """Card difficulty (matches the Flashcards UI contract)."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# FSRS-ready defaults written on every generated card. The optimiser is
# deferred: these fields are stored so a later scheduler can run; nothing
# here transitions review state.
FSRS_NEW_REVIEW: Dict[str, Any] = {
    "fsrs_state": "new",
    "fsrs_stability": 0.0,
    "fsrs_difficulty": 0.0,
    "due_at": None,
    "last_review_at": None,
    "reps": 0,
    "lapses": 0,
    "elapsed_days": 0,
    "scheduled_days": 0,
}

# Front length ceiling: a front over 30 words is rejected as un-flashcard.
FLASHCARD_FRONT_MAX_WORDS = 30


class FlashcardSpec(BaseModel):
    """STEP 5 typed flashcard: spine-bound, Bloom-allocated, FSRS-ready.

    question/answer are the display strings (the Flashcards UI reads them
    as front/back via the deck projection). ``type`` and ``cognitive``
    are accepted as aliases for card_type and bloom_level so the wire
    shape matches the task contract verbatim; ``cites`` aliases
    citations for the same reason.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    concept_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    card_type: CardType = Field(alias="type")
    difficulty: CardDifficulty = CardDifficulty.MEDIUM
    bloom_level: BloomLevel = Field(alias="cognitive")
    citations: List[TopicCitation] = Field(default_factory=list, alias="cites")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    review: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _nontrivial_and_sized(self) -> "FlashcardSpec":
        import re as _re

        front_words = len((self.question or "").split())
        if front_words > FLASHCARD_FRONT_MAX_WORDS:
            raise ValueError(
                f"card {self.id!r}: front has {front_words} words "
                f"(ceiling is {FLASHCARD_FRONT_MAX_WORDS})"
            )

        def _norm(text: str) -> str:
            folded = _re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
            return " ".join(folded.split())

        if _norm(self.question) == _norm(self.answer):
            raise ValueError(
                f"card {self.id!r}: back must differ from front (trivial card)"
            )
        if not _norm(self.answer):
            raise ValueError(f"card {self.id!r}: back must be non-empty")
        return self


class FlashcardDeckReport(BaseModel):
    """Validation verdict for one generated deck (Bloom guard included)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool = False
    total: int = Field(ge=0)
    bloom_realized: Dict[str, float] = Field(default_factory=dict)
    bloom_ok: bool = False
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    shortfall: str = ""


class FlashcardSetSpec(BaseModel):
    """STEP 5 validated deck: typed cards + realized Bloom mix + provenance."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(min_length=1)
    flashcards: List[FlashcardSpec] = Field(min_length=1)
    bloom_realized: Dict[str, float] = Field(default_factory=dict)
    provenance_note: str = Field(min_length=1)


class ErrorTag(str, Enum):
    """Closed distractor-error vocabulary (STEP 6 contract)."""

    MISCONCEPTION = "misconception"
    PARTIAL_TRUTH = "partial_truth"
    UNIT_CONFUSION = "unit_confusion"
    REVERSED_RELATION = "reversed_relation"
    OVERGENERALIZATION = "overgeneralization"
    IRRELEVANT = "irrelevant"


class MCQOption(BaseModel):
    """One rich option: correctness + error tag + one-sentence rationale."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    is_correct: bool = False
    error_tag: Optional[ErrorTag] = None
    rationale: str = ""

    @model_validator(mode="after")
    def _distractor_tagged(self) -> "MCQOption":
        if not self.is_correct:
            if self.error_tag is None:
                raise ValueError(
                    f"distractor {self.text!r}: error_tag is required "
                    "(closed ErrorTag vocabulary)"
                )
            if not (self.rationale or "").strip():
                raise ValueError(
                    f"distractor {self.text!r}: a one-sentence rationale "
                    "is required"
                )
        return self


class MCQSpec(BaseModel):
    """STEP 6 rich MCQ: exactly 4 options, exactly 1 key, tagged distractors.

    ``level`` aliases bloom_level; ``cites`` aliases citations so the wire
    shape matches the task contract (question/options/correct/explanation/
    topic/difficulty/level/cites + distractor metadata on each option).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: List[MCQOption] = Field(min_length=4, max_length=4)
    correct_answer: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    difficulty: CardDifficulty = CardDifficulty.MEDIUM
    bloom_level: BloomLevel = Field(alias="level")
    citations: List[TopicCitation] = Field(default_factory=list, alias="cites")

    @model_validator(mode="after")
    def _single_key(self) -> "MCQSpec":
        keys = [i for i, o in enumerate(self.options) if o.is_correct]
        if len(keys) != 1:
            raise ValueError(
                f"question {self.id!r}: must have exactly 1 correct option, "
                f"found {len(keys)}"
            )
        if self.correct_answer != keys[0]:
            raise ValueError(
                f"question {self.id!r}: correct_answer index "
                f"{self.correct_answer} does not point at the is_correct "
                f"option ({keys[0]})"
            )
        import re as _re

        def _norm(text: str) -> str:
            folded = _re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
            return " ".join(folded.split())

        seen: set = set()
        for option in self.options:
            key = _norm(option.text)
            if key in seen:
                raise ValueError(
                    f"question {self.id!r}: duplicate option "
                    f"{option.text!r}"
                )
            seen.add(key)
        return self


class MCQCell(BaseModel):
    """One Topic x Bloom matrix cell with target and filled counts."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(min_length=1)
    topic_title: str = Field(min_length=1)
    bloom_level: BloomLevel
    target: int = Field(ge=0)
    filled: int = Field(ge=0)


class UnfilledCell(BaseModel):
    """A matrix cell with no source material (reported, never invented)."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(min_length=1)
    topic_title: str = Field(min_length=1)
    bloom_level: BloomLevel
    reason: str = Field(min_length=1)


class MCQPlan(BaseModel):
    """STEP 6 matrix plan: cells plus the honest unfilled list."""

    model_config = ConfigDict(extra="forbid")

    cells: List[MCQCell] = Field(default_factory=list)
    unfilled: List[UnfilledCell] = Field(default_factory=list)
    total_target: int = Field(ge=0)

    @model_validator(mode="after")
    def _targets_sum(self) -> "MCQPlan":
        planned = sum(c.target for c in self.cells)
        if planned != self.total_target:
            raise ValueError(
                f"plan targets sum to {planned}, expected total_target "
                f"{self.total_target} (largest-remainder must match exactly)"
            )
        return self


class MCQSetSpec(BaseModel):
    """STEP 6 validated set: rich questions + plan + unfilled."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    title: str = Field(min_length=1)
    questions: List[MCQSpec] = Field(default_factory=list)
    plan: MCQPlan
    provenance_note: str = Field(min_length=1)


class TopicMetricRow(BaseModel):
    """STEP 7 one normalized topic row (every claim carries evidence)."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(min_length=1)
    topic_title: str = Field(min_length=1)
    unit_title: str = Field(min_length=1)
    ord: int = Field(ge=0)
    frequency: int = Field(ge=0)
    recurrence: int = Field(ge=0)
    marks_total: int = Field(ge=0)
    marks_share: float = Field(ge=0.0, le=100.0)
    recency_score: float = Field(ge=0.0)
    hotspot_score: float = Field(ge=0.0)
    hotspot_rank: int = Field(ge=0)
    in_syllabus: bool = False
    in_documents: bool = False
    in_pyq: bool = False
    in_questions: bool = False
    evidence: Dict[str, List[str]] = Field(default_factory=dict)


class UnitMetricRow(BaseModel):
    """STEP 7 unit -> topic distribution row."""

    model_config = ConfigDict(extra="forbid")

    unit_title: str = Field(min_length=1)
    topic_count: int = Field(ge=0)
    frequency: int = Field(ge=0)
    marks_total: int = Field(ge=0)
    topics: List[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    """STEP 7 deterministic intelligence verdict (no LLM numbers inside)."""

    model_config = ConfigDict(extra="forbid")

    topics: List[TopicMetricRow] = Field(default_factory=list)
    units: List[UnitMetricRow] = Field(default_factory=list)
    never_asked: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)
    hotspots: List[str] = Field(default_factory=list)
    totals: Dict[str, Any] = Field(default_factory=dict)
    half_life: float = Field(gt=0.0)
    current_year: int = Field(ge=1950, le=2100)
    provenance_note: str = Field(min_length=1)


class MetricsResponse(BaseModel):
    """STEP 7 response: deterministic metrics + optionally grounded narrative.

    The LLM explains the metrics only: a narrative containing a number the
    metric payload does not contain is rejected (repaired once, then
    withheld with narrative=None and narrative_status="withheld").
    """

    model_config = ConfigDict(extra="forbid")

    metrics: MetricsReport
    narrative: Optional[str] = None
    narrative_status: str = "ok"  # ok | repaired | withheld | not_requested


def not_generated_state(kind: str, generation_call: str) -> Dict[str, Any]:
    """Honest empty read: names the generation call, never 500/empty-success."""
    return {
        "state": "not_generated",
        "kind": kind,
        "generation_call": generation_call,
        "message": (
            f"No {kind} generated yet for this project. "
            f"Run {generation_call} first."
        ),
    }


# ---------------------------------------------------------------------------
# STEP 8 + STEP 9: versioned generated artifacts (backend-first, no new table).
#
# A GeneratedArtifact is the versioned record for one derived deliverable
# (study-notes PDF, insights payload). It reuses the document-root
# node_metadata pattern from Steps 2-7 (no new table, no migration, no
# redundant generated_artifacts concept): the row-equivalent dict lives in
# node_metadata["generated_artifacts"][artifact_type] as a version list, and
# the PDF bytes live in the storage backend under `path` (an opaque storage
# key, the same vocabulary as Document.file_path), never in the DB.
#
# Versioning is traceable: artifact_fingerprint() hashes the ordered source
# ids + model + config + coverage summary, so an unchanged rebuild is a
# short-circuit (no new version) and every version names what produced it.
# ---------------------------------------------------------------------------


class ArtifactStatus(str, Enum):
    """Lifecycle of one generated artifact version."""

    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    # Older versions remain readable after a newer version is finalized.
    SUPERSEDED = "superseded"


class GeneratedArtifact(BaseModel):
    """STEP 9 versioned artifact record (minimum field set).

    Wire aliases match the task contract verbatim: `sources`, `type`,
    `hash`. Every model is extra="forbid".
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    source_ids: List[str] = Field(default_factory=list, alias="sources")
    artifact_type: str = Field(min_length=1, alias="type")
    version: int = Field(ge=1)
    status: ArtifactStatus = ArtifactStatus.BUILDING
    path: str = Field(min_length=1)
    pages: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64, alias="hash")
    model: str = Field(min_length=1)
    config: Dict[str, Any] = Field(default_factory=dict)
    coverage: Dict[str, Any] = Field(default_factory=dict)
    # Optional model/artifact linkage.  The JSONB node-metadata store keeps
    # these fields additive, so older rows without them remain readable.
    model_version: Optional[int] = Field(default=None, ge=1)
    payload: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
    superseded_by: Optional[int] = Field(default=None, ge=1)
    # Traceability link: sha256 over sources/model/config/coverage (see
    # artifact_fingerprint). Optional so legacy rows without it still read.
    fingerprint: Optional[str] = Field(default=None, min_length=64, max_length=64)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)


def artifact_fingerprint(
    source_ids: List[str],
    model: str,
    config: Dict[str, Any],
    coverage: Dict[str, Any],
) -> str:
    """Traceable version identity: sha256 over ordered sources + model + config + coverage.

    Pure and deterministic: source ids are sorted, JSON is canonical
    (sort_keys, compact separators), so the same inputs always yield the
    same fingerprint regardless of call order.
    """
    import hashlib
    import json as _json

    canonical = _json.dumps(
        {
            "sources": sorted(source_ids or []),
            "model": model or "",
            "config": config or {},
            "coverage": coverage or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


FlashcardSpec.model_rebuild()
MCQSpec.model_rebuild()
MCQPlan.model_rebuild()
MetricsReport.model_rebuild()
MetricsResponse.model_rebuild()


__all__ = [
    "NodeType",
    "NODE_DEPTH",
    "TYPE_ORDER",
    "KnowledgeNodeSpec",
    "KnowledgeSpineSpec",
    "KnowledgeSpineBuildRequest",
    "CanonicalResolveResult",
    "SpineStats",
    "StructureMethod",
    "ConceptProvenance",
    "ExtractedConcept",
    "ExtractedSubtopic",
    "ExtractedTopic",
    "StructuredUnit",
    "DocumentStructure",
    "TopicShape",
    "TopicNotes",
    "TopicCitation",
    "NotesGenerationMode",
    "NotesDetailLevel",
    "NotesScope",
    "NotesGenerationConfig",
    "recommend_notes_config",
    "NotesSection",
    "PerTopicScore",
    "CoverageReport",
    "RepairRecord",
    "NotesArtifact",
    "HierarchicalNotesRequest",
    "BloomLevel",
    "BLOOM_MIX",
    "CardType",
    "CardDifficulty",
    "FSRS_NEW_REVIEW",
    "FLASHCARD_FRONT_MAX_WORDS",
    "FlashcardSpec",
    "FlashcardDeckReport",
    "FlashcardSetSpec",
    "ErrorTag",
    "MCQOption",
    "MCQSpec",
    "MCQCell",
    "UnfilledCell",
    "MCQPlan",
    "MCQSetSpec",
    "TopicMetricRow",
    "UnitMetricRow",
    "MetricsReport",
    "MetricsResponse",
    "not_generated_state",
    "ArtifactStatus",
    "GeneratedArtifact",
    "artifact_fingerprint",
]
