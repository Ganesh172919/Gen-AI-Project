"""Evaluate endpoint for FaithForge.

Exposes:
- POST /evaluate — submit a batch evaluation job
- GET /evaluate/status/{job_id} — poll job status
- GET /evaluate/results/{job_id} — get completed results

Evaluation runs are queued via Redis and processed asynchronously,
so the API doesn't block on long-running eval jobs.
"""

import time
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.models.schemas import (
    AblationResult,
    EvaluationMetrics,
    EvaluationRequest,
    EvaluationResponse,
)
from app.services.queue import JobStatus, get_queue

logger = get_logger("faithforge.api.evaluate")

router = APIRouter(prefix="/evaluate", tags=["evaluate"])


@router.post("", response_model=dict)
async def submit_evaluation(request: EvaluationRequest):
    """Submit a batch evaluation job.

    Queues the evaluation for async processing. Poll
    GET /evaluate/status/{job_id} for progress.

    Args:
        request: Evaluation parameters (dataset, sample size, ablations flag).

    Returns:
        Dict with job_id for polling.
    """
    logger.info(
        "Submitting evaluation: dataset=%s, sample_size=%s, ablations=%s",
        request.dataset_name,
        request.sample_size,
        request.run_ablations,
    )

    queue = get_queue()
    job_id = await queue.enqueue(
        job_type="evaluate",
        payload={
            "dataset_name": request.dataset_name,
            "sample_size": request.sample_size,
            "run_ablations": request.run_ablations,
        },
    )

    logger.info("Evaluation job queued: %s", job_id)
    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}", response_model=dict)
async def get_evaluation_status(job_id: str):
    """Poll the status of an evaluation job.

    Args:
        job_id: The job ID returned by POST /evaluate.

    Returns:
        Job status and progress info.
    """
    queue = get_queue()
    job = await queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error": job.get("error"),
    }


@router.get("/results/{job_id}", response_model=EvaluationResponse)
async def get_evaluation_results(job_id: str):
    """Get the results of a completed evaluation job.

    Args:
        job_id: The job ID.

    Returns:
        Full evaluation results with metrics and ablations.

    Raises:
        404 if job not found, 400 if job not completed.
    """
    queue = get_queue()
    job = await queue.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is {job['status']}, not completed",
        )

    result = job["result"]

    # TODO: Parse the result dict into EvaluationResponse
    # For now, return a stub
    return EvaluationResponse(
        metrics=EvaluationMetrics(
            total_queries=result.get("total_queries", 0),
            faithfulness_accuracy=result.get("faithfulness_accuracy", 0.0),
            claim_level_precision=result.get("claim_level_precision", 0.0),
            claim_level_recall=result.get("claim_level_recall", 0.0),
            avg_iterations=result.get("avg_iterations", 0.0),
            avg_latency_ms=result.get("avg_latency_ms", 0.0),
        ),
        ablations=[],
        run_id=job_id,
        completed_at=datetime.fromisoformat(job["completed_at"]) if job.get("completed_at") else datetime.utcnow(),
    )
