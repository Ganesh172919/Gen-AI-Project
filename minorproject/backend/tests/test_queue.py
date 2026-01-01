"""Tests for Redis queue integration.

These tests require a running Redis instance.
Skip with: pytest -m "not redis"
"""

import pytest

from app.services.queue import JobStatus, QueueManager


@pytest.fixture
async def queue():
    """Create a test queue manager."""
    q = QueueManager()
    try:
        await q.connect()
        yield q
        await q.disconnect()
    except Exception:
        pytest.skip("Redis not available")


@pytest.mark.asyncio
async def test_enqueue_and_get_job(queue):
    """Test basic enqueue and retrieve."""
    job_id = await queue.enqueue("test", {"key": "value"})
    assert job_id is not None

    job = await queue.get_job(job_id)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["status"] == JobStatus.PENDING.value


@pytest.mark.asyncio
async def test_update_job_status(queue):
    """Test job status transitions."""
    job_id = await queue.enqueue("test", {})

    await queue.update_job(job_id, JobStatus.RUNNING)
    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.RUNNING.value
    assert job["started_at"] is not None

    await queue.update_job(job_id, JobStatus.COMPLETED, result={"answer": 42})
    job = await queue.get_job(job_id)
    assert job["status"] == JobStatus.COMPLETED.value
    assert job["result"]["answer"] == 42
    assert job["completed_at"] is not None


@pytest.mark.asyncio
async def test_dequeue_pending(queue):
    """Test FIFO dequeue behavior."""
    id1 = await queue.enqueue("test", {"n": 1})
    id2 = await queue.enqueue("test", {"n": 2})

    dequeued1 = await queue.dequeue_pending()
    dequeued2 = await queue.dequeue_pending()

    assert dequeued1 == id1
    assert dequeued2 == id2


@pytest.mark.asyncio
async def test_get_nonexistent_job(queue):
    """Test getting a job that doesn't exist."""
    job = await queue.get_job("nonexistent-id")
    assert job is None


@pytest.mark.asyncio
async def test_queue_stats(queue):
    """Test queue statistics."""
    # Clear any existing jobs
    while await queue.dequeue_pending():
        pass

    await queue.enqueue("test", {})
    await queue.enqueue("test", {})

    stats = await queue.get_queue_stats()
    assert stats["pending"] >= 2
