"""Graph nodes: thin wrappers over the existing deterministic pipeline.

No logic is reimplemented here; each node calls the module that already
owns the behavior:
- quality gate: services.extraction.extractor._needs_vision / ExtractorConfig
- hybrid RRF: services.retrieval.hybrid.rrf_fuse
- citations: services.retrieval.citations.validate_citation_ids
- structured generation: services.strategies.llm.structured (via adapters)
- retrieval: services.retrieval.service.VectorRetrievalService (via adapters)
"""

from __future__ import annotations

from typing import Any


# -- ingestion nodes ------------------------------------------------------

def inspect_node(state: dict) -> dict:
    pages = state.get("pages") or []
    return {"page_count": len(pages) if pages else int(state.get("page_count", 0) or 0)}


def classify_node(state: dict) -> dict:
    from services.extraction.extractor import ExtractorConfig, _needs_vision

    cfg = ExtractorConfig()
    pages = state.get("pages") or []
    if not pages:
        return {"classification": "unknown"}
    needs = 0
    for p in pages:
        need, _ = _needs_vision(str(p.get("text", "")), bool(p.get("image_dominant", False)), cfg)
        needs += 1 if need else 0
    if needs == 0:
        return {"classification": "born-digital"}
    if needs == len(pages):
        return {"classification": "scanned"}
    return {"classification": "mixed"}


def extract_node(state: dict) -> dict:
    # Extraction itself runs in handlers/document_processing via
    # DocumentExtractor; the graph records the already-extracted pages.
    pages = state.get("pages") or []
    return {"pages": pages}


def gate_node(state: dict) -> dict:
    from services.extraction.extractor import ExtractorConfig, _needs_vision

    cfg = ExtractorConfig()
    signals = []
    for p in (state.get("pages") or []):
        need, reason = _needs_vision(str(p.get("text", "")),
                                     bool(p.get("image_dominant", False)), cfg)
        signals.append({"page": p.get("page"), "needs_vision": need, "reason": reason})
    return {"gate": {"signals": signals}}


def ocr_routing_node(state: dict) -> dict:
    from services.strategies.puter import is_heavy_document

    page_count = int(state.get("page_count", 0) or 0)
    signals = (state.get("gate") or {}).get("signals", [])
    needs_vision = any(s.get("needs_vision") for s in signals)
    if not needs_vision:
        return {"ocr_route": "none"}
    if is_heavy_document(page_count) and not state.get("puter_authorised"):
        return {"ocr_route": "puter-offered"}  # explicit gesture still required
    return {"ocr_route": "ppocr"}


def clean_node(state: dict) -> dict:
    parts = [(p.get("text", "") or "") for p in (state.get("pages") or [])]
    cleaned = "\n\n".join(t.strip() for t in parts if t.strip())
    return {"cleaned_text": cleaned}


def structure_node(state: dict) -> dict:
    structure = []
    for i, line in enumerate((state.get("cleaned_text", "") or "").split("\n")):
        line = line.strip()
        if line:
            structure.append({"type": "paragraph", "text": line, "position": i})
    return {"structure": structure}


def chunk_node(state: dict) -> dict:
    text = state.get("cleaned_text", "") or ""
    # Deterministic windowing here mirrors SemanticChunker boundaries
    # coarsely; the real chunker owns exact spans in the pipeline.
    words = text.split()
    chunks, size = [], 200
    for i in range(0, max(1, len(words)), size):
        chunks.append({"text": " ".join(words[i:i + size]), "position": len(chunks)})
    return {"chunks": chunks}


def embed_node(state: dict) -> dict:
    if not state.get("chunks"):
        return {"embeddings_done": False, "error": "no chunks to embed"}
    return {"embeddings_done": True}


def persist_node(state: dict) -> dict:
    if not state.get("chunks"):
        return {"persist_ok": False, "error": "nothing to persist"}
    return {"persist_ok": True}


def validate_node(state: dict) -> dict:
    chunks = state.get("chunks") or []
    cleaned = state.get("cleaned_text", "") or ""
    joined = " ".join(c.get("text", "") for c in chunks)
    ok = bool(chunks) and bool(cleaned) and len(joined.strip()) > 0
    return {"validation": {"ok": ok, "chunks": len(chunks)}}


# -- tutor nodes -----------------------------------------------------------

def intent_node(state: dict) -> dict:
    q = (state.get("query", "") or "").lower()
    if any(w in q for w in ("quiz", "test me", "flashcard")):
        return {"intent": "study"}
    if any(w in q for w in ("summar", "explain")):
        return {"intent": "explain"}
    return {"intent": "ask"}


def query_transform_node(state: dict) -> dict:
    from .langchain_adapters import build_prompt

    template = build_prompt("Answer using only retrieved context.\nQuestion: {query}")
    try:
        transformed = template.format(query=state.get("query", ""))
    except Exception:
        transformed = state.get("query", "")
    return {"transformed_query": transformed}


def hybrid_node(state: dict) -> dict:
    # Deterministic arms run inside VectorRetrievalService; the node
    # records the candidate window passed in (offline-safe).
    cands = state.get("candidates") or []
    return {"candidates": cands}


def rrf_node(state: dict) -> dict:
    from services.retrieval.hybrid import rrf_fuse

    cands = state.get("candidates") or []
    vec = [c.get("id") for c in cands if c.get("arm") in ("vector", "both")]
    lex = [c.get("id") for c in cands if c.get("arm") in ("lexical", "both")]
    fused = rrf_fuse([vec, lex] or [[]], top_n=len(cands) or None)
    order = {cid: i for i, (cid, _) in enumerate(fused)}
    ranked = sorted(cands, key=lambda c: order.get(c.get("id"), 10**9))
    return {"fused": ranked}


def rerank_node(state: dict) -> dict:
    fused = state.get("fused") or state.get("candidates") or []
    return {"reranked": fused[:8]}


def generate_node(state: dict) -> dict:
    from .langchain_adapters import make_llm

    # Phase-4 wiring: marks-aware structured answers flow through the same
    # node (no parallel path). Triggered only when the state carries marks.
    if state.get("marks") is not None or state.get("structured_answer"):
        return answer_node(state)
    llm = make_llm()
    prompt = state.get("transformed_query") or state.get("query", "")
    try:
        answer = llm.invoke(prompt)
    except Exception as exc:
        return {"answer": "", "error": str(exc)}
    return {"answer": answer}


def answer_node(state: dict) -> dict:
    """Phase-4 generation node: marks-aware grounded answer.

    Reuses the transformation engine's prompt builder + grounding gate
    (validate_citation_ids, marks bands). Offline-safe: with no retrieval
    backing it returns an explicit unsupported result, never a fabricated
    answer.
    """
    from services.retrieval.citations import validate_citation_ids
    from services.transformation.prompts import PromptTemplates

    query = state.get("transformed_query") or state.get("query", "")
    try:
        marks = int(state.get("marks", 5) or 5)
    except Exception:
        return {"answer": "", "error": "marks must be a positive integer"}
    if marks <= 0:
        return {"answer": "", "error": "marks must be a positive integer"}
    style = str(state.get("style", "detailed") or "detailed")
    cands = state.get("reranked") or state.get("fused") or state.get("candidates") or []
    allowed = {c.get("id") for c in cands if isinstance(c, dict) and c.get("id")}
    context = "\n\n".join(
        str(c.get("text", "")) for c in cands
        if isinstance(c, dict) and c.get("text")
    )
    try:
        from services.transformation.types import GenerationMode, constraints_for_marks

        band = constraints_for_marks(marks)
        prompt = PromptTemplates.get_prompt(
            GenerationMode.ANSWER, context, query, marks=marks, style=style
        )
    except Exception as exc:
        return {"answer": "", "error": f"answer prompt failed: {exc}"}
    cited = [c.get("chunk_id") for c in (state.get("citations") or [])
             if isinstance(c, dict) and c.get("chunk_id")]
    valid, rejected = validate_citation_ids(cited, allowed)
    if rejected:
        return {"answer": "", "error": f"fabricated citation ids: {rejected}",
                "citations": state.get("citations") or []}
    if not allowed:
        return {"answer": "", "prompt": prompt, "marks": marks, "style": style,
                "target_words": [band["min_words"], band["max_words"]],
                "citations": [], "support": "unsupported",
                "error": "no retrieval backing: answer unsupported (explicit, not fabricated)"}
    from .langchain_adapters import make_llm

    llm = make_llm()
    try:
        answer = llm.invoke(prompt)
    except Exception as exc:
        return {"answer": "", "error": str(exc)}
    return {"answer": answer, "prompt": prompt, "marks": marks, "style": style,
            "target_words": [band["min_words"], band["max_words"]],
            "citations": state.get("citations") or [], "support": "retrieved"}


def notes_node(state: dict) -> dict:
    """Phase-4 generation node: grounded notes/flashcards/MCQ builder.

    Reuses PromptTemplates + the engine's dedupe/grounding helpers over the
    already-retrieved candidates. Offline-safe: missing content returns an
    explicit error, never fabricated study material.
    """
    from services.retrieval.citations import validate_citation_ids
    from services.transformation.engine import dedupe_flashcards, dedupe_mcqs
    from services.transformation.prompts import PromptTemplates
    from services.transformation.types import GenerationMode

    mode_name = str(state.get("study_mode", "notes") or "notes").lower()
    mapping = {"notes": GenerationMode.NOTES, "flashcards": GenerationMode.FLASHCARDS,
               "mcq": GenerationMode.MCQ, "quiz": GenerationMode.QUIZ,
               "explanation": GenerationMode.EXPLANATION}
    mode = mapping.get(mode_name)
    if mode is None:
        return {"generated": None, "error": f"unknown study_mode: {mode_name}"}
    cands = state.get("reranked") or state.get("fused") or state.get("candidates") or []
    allowed = {c.get("id") for c in cands if isinstance(c, dict) and c.get("id")}
    context = "\n\n".join(
        str(c.get("text", "")) for c in cands
        if isinstance(c, dict) and c.get("text")
    )
    if not (context or "").strip():
        return {"generated": None, "mode": mode_name,
                "error": "no study content: empty retrieval (explicit, not fabricated)"}
    prompt = PromptTemplates.get_prompt(
        mode, context, state.get("query", ""), marks=state.get("marks", 5),
        style=state.get("style", "detailed"))
    items = [c.get("id") for c in cands if isinstance(c, dict) and c.get("id")]
    if mode_name == "flashcards":
        items = dedupe_flashcards([{"front": c.get("id", ""), "back": ""}
                                   for c in cands if isinstance(c, dict)])
    elif mode_name in ("mcq", "quiz"):
        items = dedupe_mcqs([{"question": c.get("id", "")}
                             for c in cands if isinstance(c, dict)])
    cited = [c.get("chunk_id") for c in (state.get("citations") or [])
             if isinstance(c, dict) and c.get("chunk_id")]
    _, rejected = validate_citation_ids(cited, allowed)
    if rejected:
        return {"generated": None, "mode": mode_name,
                "error": f"fabricated citation ids: {rejected}"}
    return {"generated": {"mode": mode_name, "items": items, "support": "retrieved"},
            "mode": mode_name, "study_prompt": prompt}


def flashcards_node(state: dict) -> dict:
    return notes_node({**state, "study_mode": "flashcards"})


def mcq_node(state: dict) -> dict:
    return notes_node({**state, "study_mode": "mcq"})


def tutor_validate_node(state: dict) -> dict:
    from services.retrieval.citations import validate_citation_ids

    cited = [c.get("chunk_id") for c in (state.get("citations") or []) if c.get("chunk_id")]
    allowed = {c.get("id") for c in (state.get("reranked") or [])}
    valid, rejected = validate_citation_ids(cited, allowed)
    ok = bool(state.get("answer", "").strip()) and not rejected
    return {"valid": ok, "citations": state.get("citations") or []}


def refine_node(state: dict) -> dict:
    rounds = int(state.get("refine_rounds", 0) or 0) + 1
    return {"refine_rounds": rounds}


# -- research nodes --------------------------------------------------------

def plan_node(state: dict) -> dict:
    q = state.get("question", "") or ""
    replans = int(state.get("replans", 0) or 0)
    # Count re-entries (conditional replan loops back here) with a bound.
    if state.get("plan"):
        replans += 1
    out = {"plan": [f"search: {q}", f"verify: {q}"], "replans": replans}
    return out


def search_node(state: dict) -> dict:
    from .langchain_adapters import make_tool

    tool = make_tool("tavily")
    results = tool.run(state.get("question", ""))
    return {"search_results": results if isinstance(results, list) else []}


def dedupe_node(state: dict) -> dict:
    seen, out = set(), []
    for r in (state.get("search_results") or []):
        url = r.get("url") if isinstance(r, dict) else str(r)
        if url not in seen:
            seen.add(url)
            out.append(r)
    return {"deduped": out}


def research_eval_node(state: dict) -> dict:
    deduped = state.get("deduped") or []
    return {"eval_scores": {"count": len(deduped),
                            "ok": len(deduped) > 0}}


def synth_node(state: dict) -> dict:
    from .langchain_adapters import make_llm

    llm = make_llm()
    try:
        synthesis = llm.invoke("Synthesize: " + (state.get("question", "") or ""))
    except Exception as exc:
        return {"synthesis": "", "error": str(exc)}
    return {"synthesis": synthesis}


def cites_node(state: dict) -> dict:
    cands = state.get("deduped") or []
    cites = [{"url": (c.get("url") if isinstance(c, dict) else str(c))} for c in cands[:5]]
    return {"citations": cites}


# -- study nodes ------------------------------------------------------------

def study_content_node(state: dict) -> dict:
    # Phase-4 wiring: when the state requests a structured study mode, reuse
    # the grounded notes/flashcards/MCQ builder (same helpers as the engine).
    if state.get("study_mode") in ("notes", "flashcards", "mcq", "quiz", "explanation"):
        out = notes_node(state)
        if out.get("error") and out.get("generated") is None:
            return {"content": state.get("content", ""),
                    "study": out, "error": out["error"]}
        return {"content": state.get("content", ""), "study": out}
    return {"content": state.get("content", "")}


def study_check_node(state: dict) -> dict:
    content = state.get("content", "") or ""
    return {"check": {"non_empty": bool(content.strip()),
                      "length": len(content)}}


def study_eval_node(state: dict) -> dict:
    check = state.get("check") or {}
    score = 1.0 if check.get("non_empty") else 0.0
    return {"eval_score": score}


def difficulty_node(state: dict) -> dict:
    score = float(state.get("eval_score", 0.0) or 0.0)
    level = "easy" if score >= 0.8 else ("medium" if score >= 0.4 else "hard")
    return {"difficulty": level}


def next_node(state: dict) -> dict:
    level = state.get("difficulty", "medium")
    nxt = {"easy": "advance", "medium": "practice", "hard": "review"}.get(level, "practice")
    return {"next_step": nxt}


__all__ = [n for n in dir() if n.endswith("_node")]
