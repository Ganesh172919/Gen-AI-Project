"""FaithForge — FastAPI application entry point.

A self-verifying, multi-agent RAG system with a fine-tuned faithfulness verifier.
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.evaluate import router as evaluate_router
from app.api.query import router as query_router
from app.core.config import settings
from app.core.logging import get_logger, request_filter, setup_logging
from app.services.llm_adapter import close_llm, get_llm
from app.services.queue import get_queue
from app.services.tracing import setup_tracing

logger = get_logger("faithforge")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks.

    Startup:
    - Initialize logging
    - Set up OpenTelemetry tracing
    - Connect to Redis
    - Pre-initialize LLM adapter singleton

    Shutdown:
    - Close LLM adapter
    - Disconnect from Redis
    """
    # ── Startup ──────────────────────────────────────────────────────────
    setup_logging()
    setup_tracing()
    logger.info("FaithForge starting up (version=0.2.0, debug=%s)", settings.debug)
    logger.info("Configuration: provider=%s, vector_store=%s, log_format=%s",
                settings.llm_provider.value, settings.vector_store_type.value,
                getattr(settings, "log_format", "text"))

    # Connect to Redis (optional for local dev)
    queue = get_queue()
    try:
        await queue.connect()
    except Exception as e:
        logger.warning("Redis not available (%s) — queue features disabled", str(e))

    # Pre-initialize LLM adapter
    try:
        get_llm()
        logger.info("LLM adapter initialized (provider=%s)", settings.llm_provider.value)
    except Exception as e:
        logger.warning("LLM adapter init failed (%s) — will retry on first call", str(e))

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("FaithForge shutting down")
    await close_llm()
    await queue.disconnect()


app = FastAPI(
    title=settings.app_name,
    description=(
        "A self-verifying, multi-agent RAG system with a fine-tuned "
        "faithfulness verifier. Most RAG systems hope the LLM tells the "
        "truth — FaithForge trains a small model whose only job is to check."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# ── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add request ID, timing, and structured error handling to every request.

    Features:
    - Unique X-Request-ID per request (propagated through all logs)
    - X-Response-Time header with millisecond precision
    - Request/response body logging for POST endpoints (debug mode)
    - Structured error responses with request ID
    """
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    # Set request ID in the logging context filter
    request_filter.set_request_id(request_id)

    start = time.time()

    # Log request body for POST/PUT endpoints (in debug mode)
    if settings.debug and request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.body()
            if body:
                body_text = body.decode("utf-8", errors="replace")[:500]
                logger.debug("Request body [%s]: %s", request_id, body_text)
        except Exception:
            pass

    try:
        response = await call_next(request)
        elapsed_ms = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"

        logger.debug(
            "%s %s → %d (%.1fms) [%s]",
            request.method, request.url.path, response.status_code, elapsed_ms, request_id,
        )

        # Log slow requests as warnings
        if elapsed_ms > 5000:
            logger.warning(
                "Slow request: %s %s → %d (%.1fms) [%s]",
                request.method, request.url.path, response.status_code, elapsed_ms, request_id,
            )

        return response

    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        logger.error(
            "%s %s → 500 (%.1fms) [%s] %s: %s",
            request.method, request.url.path, elapsed_ms, request_id, type(e).__name__, str(e),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e) if settings.debug else "An unexpected error occurred",
                "request_id": request_id,
            },
        )
    finally:
        # Clear request ID from context
        request_filter.clear_request_id()


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

app.include_router(query_router)
app.include_router(evaluate_router)


@app.get("/health")
async def health():
    """Health check endpoint with dependency status."""
    queue = get_queue()
    queue_stats = await queue.get_queue_stats() if queue._redis else {"status": "disconnected"}

    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "0.2.0",
        "llm_provider": settings.llm_provider.value,
        "vector_store": settings.vector_store_type.value,
        "queue": queue_stats,
        "log_format": getattr(settings, "log_format", "text"),
    }


@app.get("/")
async def root():
    """Root endpoint — service info."""
    return {
        "name": settings.app_name,
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query",
            "query_stream": "GET /query/stream?q=...",
            "evaluate": "POST /evaluate",
            "evaluate_status": "GET /evaluate/status/{job_id}",
            "evaluate_results": "GET /evaluate/results/{job_id}",
        },
        "features": {
            "fine_tuned_verifier": "QLoRA fine-tuned faithfulness checker",
            "adaptive_routing": "Query complexity classification",
            "targeted_correction": "Claim-level correction (not full regen)",
            "hybrid_retrieval": "Dense + Sparse + RRF fusion + Reranking",
            "sse_streaming": "Real-time pipeline progress events",
        },
    }
