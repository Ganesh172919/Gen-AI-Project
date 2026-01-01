"""LangGraph orchestration graph for FaithForge.

This is the central orchestration module — a state machine that wires together:
1. Planner → complexity classification
2. Retriever → hybrid retrieval
3. Generator → grounded answer generation
4. Verifier → claim-level faithfulness checking
5. Corrector → targeted claim correction (conditional loop)

Graph topology:
    START → planner → retriever → generator → verifier
                                              ↓
                                    [all claims pass] → END
                                    [claims fail] → corrector → verifier (loop, max N)
"""

import time
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    Claim,
    ClaimStatus,
    ClaimVerification,
    CorrectionRecord,
    PipelineTrace,
    RetrievedChunk,
    RetrievalStrategy,
)
from app.agents.corrector import CorrectiveAgent
from app.agents.planner import get_planner
from app.agents.verifier import FaithfulnessVerifier
from app.services.generator import get_generator
from app.services.retriever import HybridRetriever
from app.services.tracing import trace_agent_hop

logger = get_logger("faithforge.graph")


# ── Graph State ──────────────────────────────────────────────────────────────

class ForgeState(dict):
    """State dict flowing through the FaithForge graph.

    Keys:
        query: str — the original user query
        strategy: RetrievalStrategy — complexity classification
        sub_queries: list[str] — decomposed queries
        retrieved_chunks: list[RetrievedChunk] — evidence
        answer: str — generated answer text
        claims: list[Claim] — extracted claims
        verifications: list[ClaimVerification] — verification results
        corrections: list[CorrectionRecord] — correction history
        iteration: int — current verify→correct iteration
        max_iterations: int — loop cap
        trace: PipelineTrace — accumulated trace for the response
        final_answer: str — answer after any corrections
        timings: dict — per-stage timing data
    """
    pass


# ── Node Functions ───────────────────────────────────────────────────────────

async def planner_node(state: ForgeState) -> dict:
    """Classify query complexity and determine retrieval strategy.

    Input: state["query"]
    Output: state["strategy"], state["sub_queries"]
    """
    start = time.time()
    logger.info("━" * 60)
    logger.info("STAGE 1: PLANNER — classifying query complexity")

    planner = get_planner()
    query = state["query"]

    async with trace_agent_hop("planner.classify", query=query[:200]):
        strategy, sub_queries = await planner.classify(query)

    elapsed = (time.time() - start) * 1000
    logger.info(
        "Planner complete: strategy=%s, sub_queries=%d, elapsed=%.1fms",
        strategy.value, len(sub_queries), elapsed,
    )

    timings = state.get("timings", {})
    timings["planner_ms"] = elapsed

    return {
        "strategy": strategy,
        "sub_queries": sub_queries,
        "timings": timings,
    }


async def retriever_node(state: ForgeState) -> dict:
    """Retrieve evidence chunks for all sub-queries.

    Input: state["sub_queries"]
    Output: state["retrieved_chunks"]
    """
    start = time.time()
    logger.info("━" * 60)
    logger.info("STAGE 2: RETRIEVER — hybrid retrieval for %d sub-queries", len(state["sub_queries"]))

    retriever = HybridRetriever()
    await retriever.initialize()

    all_chunks: list[RetrievedChunk] = []
    seen_ids: set[str] = set()

    for i, sq in enumerate(state["sub_queries"]):
        logger.info("  Sub-query %d/%d: '%s...'", i + 1, len(state["sub_queries"]), sq[:80])
        async with trace_agent_hop("retriever.retrieve", query=sq[:200]):
            chunks = await retriever.retrieve(sq)
        for c in chunks:
            if c.chunk_id not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c.chunk_id)

    await retriever.close()

    elapsed = (time.time() - start) * 1000
    logger.info("Retriever complete: %d unique chunks, elapsed=%.1fms", len(all_chunks), elapsed)

    timings = state.get("timings", {})
    timings["retriever_ms"] = elapsed

    return {"retrieved_chunks": all_chunks, "timings": timings}


async def generator_node(state: ForgeState) -> dict:
    """Generate a grounded answer with claim tagging.

    Input: state["query"], state["retrieved_chunks"]
    Output: state["answer"], state["claims"]
    """
    start = time.time()
    logger.info("━" * 60)
    logger.info("STAGE 3: GENERATOR — generating grounded answer with %d chunks", len(state["retrieved_chunks"]))

    generator = get_generator()

    async with trace_agent_hop("generator.generate", query=state["query"][:200]):
        answer, claims = await generator.generate(
            state["query"],
            state["retrieved_chunks"],
        )

    elapsed = (time.time() - start) * 1000
    logger.info(
        "Generator complete: answer_len=%d, claims=%d, elapsed=%.1fms",
        len(answer), len(claims), elapsed,
    )

    timings = state.get("timings", {})
    timings["generator_ms"] = elapsed

    return {"answer": answer, "claims": claims, "timings": timings}


async def verifier_node(state: ForgeState) -> dict:
    """Verify all claims against their cited evidence.

    Input: state["claims"], state["retrieved_chunks"]
    Output: state["verifications"]
    """
    start = time.time()
    iteration = state.get("iteration", 1)
    logger.info("━" * 60)
    logger.info("STAGE 4: VERIFIER — verifying %d claims (iteration %d)", len(state["claims"]), iteration)

    verifier = FaithfulnessVerifier()
    await verifier.load()

    # Build chunk_id → chunk lookup
    chunk_map: dict[str, list[RetrievedChunk]] = {}
    for chunk in state["retrieved_chunks"]:
        chunk_map.setdefault(chunk.chunk_id, []).append(chunk)

    async with trace_agent_hop("verifier.verify_batch", num_claims=len(state["claims"])):
        verifications = await verifier.verify_batch(state["claims"], chunk_map)

    await verifier.unload()

    elapsed = (time.time() - start) * 1000
    verified = sum(1 for v in verifications if v.status == ClaimStatus.VERIFIED)
    failed = sum(1 for v in verifications if v.status == ClaimStatus.FAILED)
    logger.info(
        "Verifier complete: verified=%d, failed=%d, elapsed=%.1fms",
        verified, failed, elapsed,
    )

    timings = state.get("timings", {})
    timings[f"verifier_ms_iter{iteration}"] = elapsed

    return {"verifications": verifications, "timings": timings}


async def corrector_node(state: ForgeState) -> dict:
    """Correct failed claims via targeted re-retrieval and rewriting.

    Input: state["verifications"], state["claims"], state["iteration"]
    Output: state["corrections"], state["claims"] (updated), state["iteration"] (incremented)
    """
    start = time.time()
    iteration = state.get("iteration", 1)
    logger.info("━" * 60)
    logger.info("STAGE 5: CORRECTOR — correcting failed claims (iteration %d)", iteration)

    retriever = HybridRetriever()
    await retriever.initialize()
    corrector = CorrectiveAgent(retriever)

    failed = [v for v in state["verifications"] if v.status == ClaimStatus.FAILED]
    logger.info("  %d claims need correction", len(failed))

    async with trace_agent_hop("corrector.correct_claims", num_failed=len(failed)):
        correction_records, corrected_claims = await corrector.correct_claims(
            failed, state["claims"], iteration
        )

    # Update claims: keep passed claims, replace corrected ones
    passed_ids = {v.claim_id for v in state["verifications"] if v.status == ClaimStatus.VERIFIED}
    corrected_map = {c.claim_id: c for c in corrected_claims}

    updated_claims = []
    for claim in state["claims"]:
        if claim.claim_id in passed_ids:
            updated_claims.append(claim)
        elif claim.claim_id in corrected_map:
            updated_claims.append(corrected_map[claim.claim_id])
        # Claims with no correction are dropped

    await retriever.close()

    elapsed = (time.time() - start) * 1000
    logger.info(
        "Corrector complete: %d corrections, %d updated claims, elapsed=%.1fms",
        len(correction_records), len(updated_claims), elapsed,
    )

    timings = state.get("timings", {})
    timings[f"corrector_ms_iter{iteration}"] = elapsed

    return {
        "corrections": state.get("corrections", []) + correction_records,
        "claims": updated_claims,
        "iteration": iteration + 1,
        "timings": timings,
    }


# ── Conditional Edge Functions ───────────────────────────────────────────────

def should_continue(state: ForgeState) -> str:
    """Decide whether to loop back to the verifier or end.

    Returns:
        "corrector" if any claims failed and iteration < max_iterations
        "end" if all claims pass or max iterations reached
    """
    verifications = state.get("verifications", [])
    failed = [v for v in verifications if v.status == ClaimStatus.FAILED]
    iteration = state.get("iteration", 1)
    max_iterations = state.get("max_iterations", settings.verifier_max_iterations)

    if failed and iteration < max_iterations:
        logger.info(
            "Continuing correction loop: %d failed claims, iteration %d/%d",
            len(failed), iteration, max_iterations,
        )
        return "corrector"

    if failed:
        logger.info(
            "Max iterations reached (%d) with %d failed claims — ending",
            max_iterations, len(failed),
        )
    else:
        logger.info("All %d claims verified — ending", len(verifications))

    return "end"


def build_graph() -> Any:
    """Build and compile the FaithForge LangGraph state machine.

    Graph topology:
        START → planner → retriever → generator → verifier
                                                  ↓
                                        should_continue()
                                       /                \
                                  corrector              END
                                      ↓
                                   verifier (loop)

    Returns:
        Compiled LangGraph graph, ready to invoke.
    """
    logger.info("Building FaithForge LangGraph...")

    graph = StateGraph(ForgeState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("corrector", corrector_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Sequential edges: planner → retriever → generator → verifier
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "verifier")

    # Conditional edge from verifier
    graph.add_conditional_edges(
        "verifier",
        should_continue,
        {
            "corrector": "corrector",
            "end": END,
        },
    )

    # Loop: corrector → verifier
    graph.add_edge("corrector", "verifier")

    compiled = graph.compile()
    logger.info("LangGraph compiled successfully")
    return compiled


async def run_pipeline(
    query: str,
    max_iterations: Optional[int] = None,
) -> dict:
    """Run the full FaithForge pipeline on a query.

    This is the top-level entry point called by the /query endpoint.

    Args:
        query: The user's question.
        max_iterations: Override the default verify→correct loop cap.

    Returns:
        Final state dict with answer, claims, verifications, corrections, and trace.
    """
    start = time.time()
    logger.info("=" * 60)
    logger.info("FAITHFORGE PIPELINE START: '%s...'", query[:100])
    logger.info("=" * 60)

    graph = build_graph()

    max_iter = max_iterations or settings.verifier_max_iterations

    initial_state = ForgeState({
        "query": query,
        "strategy": None,
        "sub_queries": [],
        "retrieved_chunks": [],
        "answer": "",
        "claims": [],
        "verifications": [],
        "corrections": [],
        "iteration": 1,
        "max_iterations": max_iter,
        "trace": None,
        "final_answer": "",
        "timings": {},
    })

    # Invoke the graph
    final_state = await graph.ainvoke(initial_state)

    # Build the pipeline trace
    all_faithful = all(
        v.status == ClaimStatus.VERIFIED
        for v in final_state.get("verifications", [])
    ) if final_state.get("verifications") else False

    trace = PipelineTrace(
        query=query,
        strategy=final_state.get("strategy", RetrievalStrategy.SINGLE_HOP),
        sub_queries=final_state.get("sub_queries", []),
        retrieved_chunks=final_state.get("retrieved_chunks", []),
        generated_answer=final_state.get("answer", ""),
        claims=final_state.get("claims", []),
        verifications=final_state.get("verifications", []),
        corrections=final_state.get("corrections", []),
        total_iterations=final_state.get("iteration", 1) - 1,
        all_claims_faithful=all_faithful,
    )

    final_state["trace"] = trace
    final_state["final_answer"] = final_state.get("answer", "")

    total_elapsed = (time.time() - start) * 1000
    timings = final_state.get("timings", {})
    timings["total_ms"] = total_elapsed

    logger.info("=" * 60)
    logger.info(
        "FAITHFORGE PIPELINE COMPLETE: claims=%d, verified=%d, corrections=%d, "
        "all_faithful=%s, total=%.1fms",
        len(trace.claims),
        sum(1 for v in trace.verifications if v.status == ClaimStatus.VERIFIED),
        len(trace.corrections),
        all_faithful,
        total_elapsed,
    )
    logger.info("  Timings: %s", {k: f"{v:.1f}ms" for k, v in timings.items()})
    logger.info("=" * 60)

    return final_state
