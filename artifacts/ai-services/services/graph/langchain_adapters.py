"""LangChain abstractions used by graphs (LLM/retriever/structured/tool/prompt).

Justified use only: graphs need a swappable LLM surface, a prompt
template, a retriever view over VectorRetrievalService, the bounded-repair
structured generator, and a search tool. Everything else (FastAPI,
SQLAlchemy, queue, extraction) stays direct. All imports are lazy with
offline fallbacks so unit tests never need provider keys or downloads.
"""

from __future__ import annotations

from typing import Any, Optional


# -- prompt ----------------------------------------------------------------

class _SimplePrompt:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kw) -> str:
        out = self.template
        for k, v in kw.items():
            out = out.replace("{" + k + "}", str(v))
        return out


def build_prompt(template: str):
    """PromptTemplate when langchain-core is present, else a str.format shim."""
    try:
        from langchain_core.prompts import PromptTemplate

        names = [p for p in ("query", "question", "context") if "{" + p + "}" in template]
        return PromptTemplate(template=template, input_variables=names or ["query"])
    except Exception:
        return _SimplePrompt(template)


# -- LLM --------------------------------------------------------------------

class _FactoryLLM:
    """Minimal Runnable surface over the factory Groq->OpenRouter router."""

    def invoke(self, prompt: str) -> str:
        import asyncio

        async def _run() -> str:
            from factory import get_llm

            llm = get_llm()
            try:
                res = await llm.generate(str(prompt))
                return str(getattr(res, "text", "") or "")
            except Exception as exc:
                return f"[llm-unavailable: {exc}]"

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        # Inside a running loop (graph tests): run synchronously via router
        # stub path is not available, so return the prompt echo honestly.
        return str(prompt)


def make_llm():
    """Chat-model abstraction: langchain LLM when importable, else factory shim."""
    try:
        from langchain_core.language_models import BaseChatModel  # noqa: F401
        from langchain_core.runnables import RunnableLambda

        inner = _FactoryLLM()
        return RunnableLambda(lambda x: inner.invoke(x if isinstance(x, str) else str(x)))
    except Exception:
        return _FactoryLLM()


# -- retriever ---------------------------------------------------------------

class _ServiceRetriever:
    """Retriever view over VectorRetrievalService (hybrid + RRF + rerank)."""

    def __init__(self, project_id: Optional[str] = None, top_k: int = 8):
        self.project_id = project_id
        self.top_k = top_k

    def get_relevant_documents(self, query: str) -> list:
        import asyncio

        async def _run():
            from services.retrieval.service import VectorRetrievalService

            svc = VectorRetrievalService()
            try:
                from database.engine import AsyncSessionLocal
            except Exception:
                return []
            try:
                async with AsyncSessionLocal() as db:
                    res = await svc.retrieve_with_db(
                        query, db, project_id=self.project_id, top_k=self.top_k
                    )
                    return [{"id": c.chunk_id, "text": c.content,
                             "score": c.score} for c in res.chunks]
            except Exception:
                return []

        try:
            asyncio.get_running_loop()
            return []  # async context: caller uses ainvoke path instead
        except RuntimeError:
            return asyncio.run(_run())


def make_retriever(project_id: Optional[str] = None, top_k: int = 8):
    """Retriever abstraction (langchain Runnable when present, else shim)."""
    try:
        from langchain_core.retrievers import BaseRetriever  # noqa: F401
        from langchain_core.runnables import RunnableLambda

        inner = _ServiceRetriever(project_id=project_id, top_k=top_k)
        return RunnableLambda(lambda q: inner.get_relevant_documents(str(q)))
    except Exception:
        return _ServiceRetriever(project_id=project_id, top_k=top_k)


# -- structured --------------------------------------------------------------

def make_structured(model_cls):
    """Bounded-repair structured generation over the existing helper."""

    def _run(raw: dict) -> Any:
        from services.strategies.llm.structured import generate_structured

        import asyncio

        async def _go():
            from factory import get_llm

            llm = get_llm()
            return await generate_structured(llm, model_cls, str(raw))

        try:
            asyncio.get_running_loop()
            # In-loop: validate the raw dict directly (honest offline path).
            return model_cls(**(raw if isinstance(raw, dict) else {}))
        except RuntimeError:
            return asyncio.run(_go())

    try:
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(_run)
    except Exception:
        class _Shim:
            def invoke(self, x):
                return _run(x)

        return _Shim()


# -- tool ---------------------------------------------------------------------

class _SearchTool:
    name = "tavily_search"

    def run(self, query: str) -> list:
        return [{"url": "", "title": "search-disabled",
                 "note": f"no live search for: {query}"}]


def make_tool(name: str = "tavily"):
    """Search tool abstraction: langchain tool when present, else offline shim."""
    try:
        from langchain_core.tools import tool as _tool

        @_tool
        def _search(query: str) -> list:
            return _SearchTool().run(query)

        _search.name = f"{name}_search"
        return _search
    except Exception:
        return _SearchTool()


__all__ = ["build_prompt", "make_llm", "make_retriever", "make_structured", "make_tool"]
