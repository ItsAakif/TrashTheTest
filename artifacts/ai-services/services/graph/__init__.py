"""LangGraph workflows over the deterministic pipeline (exact-arch increment).

Graphs orchestrate; they never replace the deterministic pipeline:
- Ingestion: inspect -> classify -> extract -> gate -> ocr-routing ->
  clean -> structure -> chunk -> embed -> persist -> validate.
- Tutor: intent -> query-transform -> hybrid -> RRF -> rerank ->
  generate -> validate -> refine loop.
- Research: plan -> search -> dedupe -> eval -> synth -> cites.
- Study: content -> check -> eval -> difficulty -> next.

Each graph carries typed state, conditional edges, and bounded retries.
LangChain is used only for LLM/retriever/structured/tool/prompt
abstractions where valuable (see langchain_adapters.py); FastAPI and
SQLAlchemy stay direct. When langgraph is installed the graphs compile
to real StateGraphs; otherwise a minimal local runner preserves the same
node/edge/retry semantics so offline tests still prove the topology.
"""

from .ingestion_graph import build_ingestion_graph, INGESTION_NODES
from .tutor_graph import build_tutor_graph, TUTOR_NODES
from .research_graph import build_research_graph, RESEARCH_NODES
from .study_graph import build_study_graph, STUDY_NODES

__all__ = [
    "build_ingestion_graph",
    "INGESTION_NODES",
    "build_tutor_graph",
    "TUTOR_NODES",
    "build_research_graph",
    "RESEARCH_NODES",
    "build_study_graph",
    "STUDY_NODES",
]
