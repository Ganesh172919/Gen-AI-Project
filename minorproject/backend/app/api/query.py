"""Query endpoint for FaithForge.

Exposes:
- POST /query — run a single query, return full result
- GET /query/stream — SSE stream with stage-by-stage progress events
"""

import json
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.core.logging import get_logger
from app.models.schemas import (
    Claim,
    ClaimVerification,
    CorrectionRecord,
    PipelineTrace,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    RetrievalStrategy,
)
from app.services.tracing import trace_agent_hop

logger = get_logger("faithforge.api.query")

router = APIRouter(prefix="/query", tags=["query"])


# ── SSE Streaming ────────────────────────────────────────────────────────────

async def _run_pipeline_streaming(query: str, max_iterations: int | None = None):
    """Generator that yields SSE events as each pipeline stage completes.

    Event format:
        event: stage
        data: {"stage": "planner", "status": "running"}

        event: stage
        data: {"stage": "planner", "status": "complete", "data": {...}}

        event: done
        data: {"trace": {...}, "final_answer": "...", "latency_ms": ...}
    """
    start_time = time.time()

    # Helper to emit SSE events
    def event(stage: str, status: str, data: dict | None = None):
        payload = {"stage": stage, "status": status}
        if data:
            payload["data"] = data
        return {"event": "stage", "data": json.dumps(payload)}

    # ── Stage 1: Planner ─────────────────────────────────────────────────
    yield event("planner", "running")
    try:
        from app.agents.planner import get_planner
        planner = get_planner()
        strategy, sub_queries = await planner.classify(query)
        yield event("planner", "complete", {
            "strategy": strategy.value,
            "sub_queries": sub_queries,
        })
    except NotImplementedError:
        # Planner not yet implemented — use defaults
        strategy = RetrievalStrategy.SINGLE_HOP
        sub_queries = [query]
        yield event("planner", "complete", {
            "strategy": strategy.value,
            "sub_queries": sub_queries,
            "note": "using default (planner not implemented)",
        })

    # ── Stage 2: Retriever ───────────────────────────────────────────────
    yield event("retriever", "running")
    chunks: list[RetrievedChunk] = []
    try:
        from app.services.retriever import HybridRetriever
        retriever = HybridRetriever()
        await retriever.initialize()
        for sq in sub_queries:
            sq_chunks = await retriever.retrieve(sq)
            chunks.extend(sq_chunks)
        await retriever.close()
        yield event("retriever", "complete", {
            "chunks_found": len(chunks),
            "chunks": [c.model_dump() for c in chunks[:5]],  # preview
        })
    except NotImplementedError:
        yield event("retriever", "complete", {
            "chunks_found": 0,
            "note": "retriever not implemented yet",
        })

    # ── Stage 3: Generator ───────────────────────────────────────────────
    yield event("generator", "running")
    answer = ""
    claims: list[Claim] = []
    try:
        from app.services.generator import get_generator
        generator = get_generator()
        answer, claims = await generator.generate(query, chunks)
        yield event("generator", "complete", {
            "answer_length": len(answer),
            "claims_count": len(claims),
            "claims": [c.model_dump() for c in claims],
        })
    except Exception as e:
        answer = f"[Generator error: {e}]"
        yield event("generator", "complete", {"error": str(e)})

    # ── Stage 4: Verifier ────────────────────────────────────────────────
    yield event("verifier", "running")
    verifications: list[ClaimVerification] = []
    try:
        from app.agents.verifier import FaithfulnessVerifier
        verifier = FaithfulnessVerifier()
        await verifier.load()
        chunk_map: dict[str, list[RetrievedChunk]] = {}
        for chunk in chunks:
            chunk_map.setdefault(chunk.chunk_id, []).append(chunk)
        verifications = await verifier.verify_batch(claims, chunk_map)
        await verifier.unload()
        passed = sum(1 for v in verifications if v.status.value == "verified")
        yield event("verifier", "complete", {
            "total": len(verifications),
            "passed": passed,
            "failed": len(verifications) - passed,
            "verifications": [v.model_dump() for v in verifications],
        })
    except NotImplementedError:
        yield event("verifier", "complete", {
            "total": 0,
            "note": "verifier not implemented yet",
        })

    # ── Stage 5: Corrector (conditional) ─────────────────────────────────
    corrections: list[CorrectionRecord] = []
    failed = [v for v in verifications if v.status.value == "failed"]
    if failed:
        yield event("corrector", "running")
        try:
            from app.agents.corrector import CorrectiveAgent
            from app.services.retriever import HybridRetriever
            retriever = HybridRetriever()
            await retriever.initialize()
            corrector = CorrectiveAgent(retriever)
            correction_records, _ = await corrector.correct_claims(failed, claims, 1)
            corrections = correction_records
            await retriever.close()
            yield event("corrector", "complete", {
                "corrections_count": len(corrections),
                "corrections": [c.model_dump() for c in corrections],
            })
        except NotImplementedError:
            yield event("corrector", "complete", {
                "corrections_count": 0,
                "note": "corrector not implemented yet",
            })
    else:
        yield event("corrector", "complete", {"corrections_count": 0, "note": "no failed claims"})

    # ── Final result ─────────────────────────────────────────────────────
    latency_ms = (time.time() - start_time) * 1000
    all_faithful = all(v.status.value == "verified" for v in verifications) if verifications else False

    trace = PipelineTrace(
        query=query,
        strategy=strategy,
        sub_queries=sub_queries,
        retrieved_chunks=chunks,
        generated_answer=answer,
        claims=claims,
        verifications=verifications,
        corrections=corrections,
        total_iterations=1,
        all_claims_faithful=all_faithful,
    )

    result = QueryResponse(
        trace=trace,
        final_answer=answer,
        latency_ms=latency_ms,
    )

    yield {"event": "done", "data": result.model_dump_json()}


@router.get("/stream")
async def query_stream(
    q: str = Query(..., min_length=1, max_length=2000, description="The user's question"),
    max_iterations: int | None = Query(None, ge=1, le=5),
):
    """Stream a query through the FaithForge pipeline via Server-Sent Events.

    Each pipeline stage emits progress events as it runs, allowing the
    frontend to animate the pipeline in real time.

    Events:
    - `stage` with data: {"stage": "...", "status": "running|complete", "data": {...}}
    - `done` with data: full QueryResponse JSON
    """
    logger.info("SSE stream started: '%s...'", q[:100])
    return EventSourceResponse(
        _run_pipeline_streaming(q, max_iterations),
        media_type="text/event-stream",
    )


# ── Standard POST endpoint ───────────────────────────────────────────────────

@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a query through the FaithForge pipeline.

    Returns the final answer with a full pipeline trace showing
    retrieved chunks, claims, verification scores, and correction history.
    """
    start_time = time.time()

    logger.info("Received query: '%s...'", request.query[:100])

    try:
        # Collect all events from the streaming generator
        trace_data = {}
        final_result = None

        async for event in _run_pipeline_streaming(request.query, request.max_iterations):
            if event.get("event") == "done":
                final_result = QueryResponse.model_validate_json(event["data"])

        if final_result is None:
            # Pipeline didn't complete — return stub
            trace = PipelineTrace(
                query=request.query,
                strategy=RetrievalStrategy.SINGLE_HOP,
                sub_queries=[request.query],
                retrieved_chunks=[],
                generated_answer="[STUB] Pipeline not yet fully implemented.",
                claims=[],
                verifications=[],
                corrections=[],
                total_iterations=0,
                all_claims_faithful=False,
            )
            return QueryResponse(
                trace=trace,
                final_answer="[STUB] Fill in the TODO modules to enable the full pipeline.",
                latency_ms=(time.time() - start_time) * 1000,
            )

        return final_result

    except Exception as e:
        logger.error("Query failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
