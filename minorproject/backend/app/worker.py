"""Background worker for FaithForge evaluation jobs.

Polls the Redis queue for pending evaluation jobs and processes them.
Run as a separate process: python -m app.worker

The worker:
1. Dequeues pending jobs from Redis
2. Runs the evaluation pipeline (TODO: wire to evaluation/ablations.py)
3. Updates job status and stores results
"""

import asyncio
import signal
import sys
from datetime import datetime

from app.core.logging import get_logger, setup_logging
from app.services.queue import JobStatus, get_queue

logger = get_logger("faithforge.worker")

# Poll interval in seconds
POLL_INTERVAL = 2.0

# Graceful shutdown flag
_shutdown = False


def _handle_shutdown(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    global _shutdown
    logger.info("Received signal %d, shutting down...", signum)
    _shutdown = True


async def process_evaluation_job(job: dict) -> dict:
    """Process a single evaluation job.

    This is where you'd wire in the actual evaluation pipeline
    from evaluation/ablations.py.

    Args:
        job: Job record from Redis.

    Returns:
        Result dict to store in the job record.
    """
    payload = job.get("payload", {})
    dataset_name = payload.get("dataset_name", "unknown")
    sample_size = payload.get("sample_size")
    run_ablations = payload.get("run_ablations", True)

    logger.info(
        "Processing evaluation: dataset=%s, sample_size=%s, ablations=%s",
        dataset_name, sample_size, run_ablations,
    )

    # TODO: Wire in actual evaluation pipeline
    # from evaluation.ablations import run_all_ablations
    # results = run_all_ablations(eval_data, verifier_model_path, output_dir)

    # Simulate work for now
    await asyncio.sleep(1)

    result = {
        "total_queries": sample_size or 0,
        "faithfulness_accuracy": 0.0,
        "claim_level_precision": 0.0,
        "claim_level_recall": 0.0,
        "avg_iterations": 0.0,
        "avg_latency_ms": 0.0,
        "ablations": [],
        "note": "Wire in evaluation/ablations.py for real results",
    }

    logger.info("Evaluation completed: dataset=%s", dataset_name)
    return result


async def worker_loop():
    """Main worker loop — poll for jobs and process them."""
    queue = get_queue()
    await queue.connect()

    logger.info("Worker started, polling every %.1fs", POLL_INTERVAL)

    while not _shutdown:
        try:
            job_id = await queue.dequeue_pending()

            if job_id is None:
                await asyncio.sleep(POLL_INTERVAL)
                continue

            logger.info("Dequeued job %s", job_id)
            job = await queue.get_job(job_id)

            if not job:
                logger.warning("Job %s not found in Redis", job_id)
                continue

            # Mark as running
            await queue.update_job(job_id, JobStatus.RUNNING)

            try:
                # Process the job
                result = await process_evaluation_job(job)
                await queue.update_job(job_id, JobStatus.COMPLETED, result=result)
                logger.info("Job %s completed", job_id)
            except Exception as e:
                logger.error("Job %s failed: %s", job_id, str(e), exc_info=True)
                await queue.update_job(job_id, JobStatus.FAILED, error=str(e))

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Worker loop error: %s", str(e), exc_info=True)
            await asyncio.sleep(POLL_INTERVAL)

    # Cleanup
    await queue.disconnect()
    logger.info("Worker stopped")


def main():
    """Entry point for the worker process."""
    setup_logging()

    # Register signal handlers
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    logger.info("FaithForge worker starting (pid=%d)", __import__("os").getpid())

    try:
        asyncio.run(worker_loop())
    except KeyboardInterrupt:
        logger.info("Worker interrupted")


if __name__ == "__main__":
    main()
