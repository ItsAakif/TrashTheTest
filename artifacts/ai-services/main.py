"""
TrashTheTest AI Services - Main FastAPI Application (single entry point).

This module is the ONLY place a FastAPI app is constructed and the ONLY
place routers are included (each router carries its own prefix).
Middleware/auth/response helpers are salvaged from api_gateway/
(the second app there was deleted per ADR-005 section 14).
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings

# Import routes from submodules (each router carries its own prefix)
from services.ingestion import router as ingestion_router
from services.extraction import router as extraction_router
from services.chunking import router as chunking_router
from services.embedding_service import router as embedding_router
from services.retrieval import router as retrieval_router
from services.transformation import router as transformation_router
from services.research import router as research_router
from services.website_gen import router as website_gen_router
from services.study import router as study_router
from services.jobs import router as jobs_router
from services.cache.routes import router as cache_router
from services.tts.routes import router as tts_router
from services.auth.routes import router as auth_router
from services.academic.routes import (
    router as academic_router,
    generation_router as academic_generation_router,
    knowledge_router as academic_knowledge_router,
)
from services.projects.routes import router as projects_router
from services.provider_credentials.routes import router as provider_settings_router
from services.ocr.routes import router as ocr_router

from api_gateway.middleware import (
    CorrelationIdMiddleware,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
)

logger = logging.getLogger(__name__)

SERVICE_NAME = "ai-services"
SERVICE_VERSION = "1.0.0"


async def _warm_retrieval_models() -> None:
    """Load embedding + reranker weights once, before the first request.

    Practice (``POST /api/generation/mcq``) cold-loaded the Qwen embedding
    model and bge-reranker inside the request and blew the client deadline.
    Warm the exact singletons retrieval uses (factory embedding provider and
    the module-level retrieval service's cached reranker). Failures are
    logged as warnings only: a missing download/flaky FS degrades retrieval
    honestly, it must not prevent the API from starting.
    """
    # uvicorn configures only its own loggers; root stays at WARNING with no
    # handler, so module INFO milestones would be invisible. Attach a
    # handler to this logger once so the warm-up duration is always logged.
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    started = time.perf_counter()
    try:
        from factory import get_embedding

        provider = get_embedding()
        warm = getattr(provider, "warm", None)
        if callable(warm):
            await warm()
        else:  # providers without an explicit warm hook
            ensure = getattr(provider, "_ensure_model", None)
            if callable(ensure):
                await asyncio.to_thread(ensure)
        # One real encode initialises the lazy tokenizer/pooling layers.
        encode = getattr(provider, "embed_passages", None) or getattr(
            provider, "generate_embeddings", None
        )
        if callable(encode):
            await encode(["warmup"], batch_size=1)
        logger.info("Embedding model warm (%.1fs)", time.perf_counter() - started)
    except Exception as exc:
        logger.warning("Embedding model warm-up failed: %s", exc)

    rerank_started = time.perf_counter()
    try:
        from services.retrieval.service import retrieval_service

        reranker = retrieval_service._get_reranker(True)
        if reranker is not None:
            ensure = getattr(reranker, "_ensure_model", None)
            if callable(ensure):
                # CrossEncoder construction/weight load is blocking CPU/IO.
                await asyncio.to_thread(ensure)
            logger.info(
                "Reranker warm (%s, %.1fs)",
                getattr(reranker, "model_name", "unknown"),
                time.perf_counter() - rerank_started,
            )
    except Exception as exc:
        logger.warning("Reranker warm-up failed: %s", exc)

    logger.info(
        "Retrieval model warm-up finished in %.1fs",
        time.perf_counter() - started,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: cache init (best-effort) + config boot checks."""
    try:
        from services.cache import initialize_cache

        await initialize_cache()
    except Exception as exc:
        logger.warning("Cache initialization failed: %s", exc)

    # Warm retrieval models AFTER the cache so a cache-backed embedding
    # provider is fully configured. Best-effort by design (see helper).
    await _warm_retrieval_models()

    from config.settings import run_boot_checks

    await run_boot_checks()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="TrashTheTest AI Services",
        description="Model Strategy Layer - Interchangeable AI/ML components",
        version=SERVICE_VERSION,
        lifespan=lifespan,
    )

    # Custom middleware (order matters - last added executes first).
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # CORS is deliberately the OUTERMOST middleware.  If an inner error
    # handler returns a 4xx/5xx, the browser must still receive
    # Access-Control-Allow-Origin; otherwise a real backend 500 is rendered
    # by Chromium as the misleading net::ERR_FAILED CORS symptom.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Request-ID", "X-Project-Id", "X-File-Name",
                       "X-Puter-Token", "X-Puter-Page", "X-Puter-Pages", "X-Source-Type"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    # Academic failures are expected input/validation outcomes for a direct
    # synchronous generation request.  Keep them typed 4xx responses instead
    # of letting the generic middleware turn an empty or invalid source scope
    # into an opaque 500.  The job worker records the same exception as a
    # terminal job failure and never fabricates a ready artifact.
    from services.academic.orchestrator import AcademicPipelineError, NoExtractableTextError

    @app.exception_handler(NoExtractableTextError)
    async def _academic_no_extractable_text(_request, exc: NoExtractableTextError):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": str(exc),
                "error_code": "NO_EXTRACTABLE_TEXT",
            },
        )

    @app.exception_handler(AcademicPipelineError)
    async def _academic_pipeline_error(_request, exc: AcademicPipelineError):
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": str(exc),
                "error_code": "ACADEMIC_PIPELINE_ERROR",
            },
        )

    # Routers (no extra prefixes - each router defines its own).
    app.include_router(auth_router)
    app.include_router(academic_router)
    app.include_router(academic_generation_router)
    app.include_router(academic_knowledge_router)
    app.include_router(projects_router)
    app.include_router(provider_settings_router)
    app.include_router(ocr_router)
    app.include_router(ingestion_router)
    app.include_router(extraction_router)
    app.include_router(chunking_router)
    app.include_router(embedding_router)
    app.include_router(retrieval_router)
    app.include_router(transformation_router)
    app.include_router(research_router)
    app.include_router(website_gen_router)
    app.include_router(study_router)
    app.include_router(jobs_router)
    app.include_router(cache_router)
    app.include_router(tts_router)

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}

    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with service status"""
        from sqlalchemy import text as _text

        health: dict = {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION, "services": {}}
        try:
            from database.engine import engine

            async with engine.connect() as conn:
                await conn.execute(_text("SELECT 1"))
            health["services"]["database"] = "healthy"
        except Exception as exc:
            health["services"]["database"] = f"unhealthy: {exc}"
            health["status"] = "degraded"
        try:
            from services.cache import get_cache_service

            cache = get_cache_service()
            cache_health = await cache.get_health()
            health["services"]["cache"] = cache_health.status
        except Exception as exc:
            health["services"]["cache"] = f"unhealthy: {exc}"
        return health

    @app.get("/api/info")
    async def api_info():
        """Get API information"""
        return {
            "name": "TrashTheTest AI Services",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "providers": {
                "ocr": settings.ocr.provider,
                "embedding": settings.embedding.provider,
                "llm": settings.llm.provider,
                "tts": settings.tts.provider,
                "search": settings.search.provider,
            },
        }

    return app


# Single application instance.
app = create_app()


# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn

    config = get_settings()
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
    )
