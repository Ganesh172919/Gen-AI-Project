"""Redis + BullMQ queue integration for FaithForge.

Provides async job queuing for batch evaluation runs.
The frontend can submit evaluation jobs that run in the background,
and poll for results via the /evaluate/status endpoint.
"""

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import redis.asyncio as redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("faithforge.queue")


class JobStatus(str, Enum):
    """Job status in the queue."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord:
    """Record of a queued evaluation job."""

    def __init__(self, job_id: str, job_type: str, payload: dict):
        self.job_id = job_id
        self.job_type = job_type
        self.payload = payload
        self.status = JobStatus.PENDING
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow().isoformat()
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class QueueManager:
    """Manages async job queues via Redis.

    This is a simplified queue implementation using Redis directly.
    For production, you'd use BullMQ (Node.js) or python-rq, but
    this keeps the dependency footprint small while demonstrating
    the pattern.

    Redis key structure:
    - faithforge:jobs:{job_id} → JobRecord JSON
    - faithforge:queue:pending → list of pending job IDs
    - faithforge:queue:running → list of running job IDs
    """

    def __init__(self):
        self._redis: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        logger.info("Connecting to Redis: %s", settings.redis_url)
        self._redis = redis.from_url(settings.redis_url, decode_responses=True)
        # Test connection
        await self._redis.ping()
        logger.info("Redis connected")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            logger.info("Redis disconnected")

    async def enqueue(
        self,
        job_type: str,
        payload: dict,
    ) -> str:
        """Add a job to the queue.

        Args:
            job_type: Type of job (e.g., "evaluate", "batch_query").
            payload: Job parameters.

        Returns:
            The job ID.
        """
        job_id = str(uuid.uuid4())
        job = JobRecord(job_id, job_type, payload)

        await self._redis.set(
            f"faithforge:jobs:{job_id}",
            json.dumps(job.to_dict()),
        )
        await self._redis.rpush("faithforge:queue:pending", job_id)

        logger.info("Enqueued job %s (type=%s)", job_id, job_type)
        return job_id

    async def get_job(self, job_id: str) -> Optional[dict]:
        """Get the status and result of a job.

        Args:
            job_id: The job ID.

        Returns:
            Job record dict, or None if not found.
        """
        data = await self._redis.get(f"faithforge:jobs:{job_id}")
        if data:
            return json.loads(data)
        return None

    async def update_job(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update a job's status.

        Args:
            job_id: The job ID.
            status: New status.
            result: Result data (for completed jobs).
            error: Error message (for failed jobs).
        """
        data = await self._redis.get(f"faithforge:jobs:{job_id}")
        if not data:
            logger.warning("Job %s not found", job_id)
            return

        job = json.loads(data)
        job["status"] = status.value
        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error
        if status == JobStatus.RUNNING:
            job["started_at"] = datetime.utcnow().isoformat()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job["completed_at"] = datetime.utcnow().isoformat()

        await self._redis.set(f"faithforge:jobs:{job_id}", json.dumps(job))
        logger.info("Job %s updated: %s", job_id, status.value)

    async def dequeue_pending(self) -> Optional[str]:
        """Pop the next pending job ID from the queue.

        Returns:
            Job ID, or None if queue is empty.
        """
        job_id = await self._redis.lpop("faithforge:queue:pending")
        if job_id:
            await self._redis.rpush("faithforge:queue:running", job_id)
        return job_id

    async def get_queue_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Dict with pending and running counts.
        """
        pending = await self._redis.llen("faithforge:queue:pending")
        running = await self._redis.llen("faithforge:queue:running")
        return {"pending": pending, "running": running}


# ── Module-level singleton ───────────────────────────────────────────────────

_queue: Optional[QueueManager] = None


def get_queue() -> QueueManager:
    """Get or create the module-level QueueManager singleton."""
    global _queue
    if _queue is None:
        _queue = QueueManager()
    return _queue
