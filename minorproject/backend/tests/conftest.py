"""Shared test fixtures for FaithForge."""

import pytest


@pytest.fixture
def sample_claim():
    """A sample claim for testing."""
    return {
        "claim_id": "c1",
        "text": "Python was created by Guido van Rossum in 1991.",
        "source_chunk_ids": ["chunk_001"],
    }


@pytest.fixture
def sample_chunks():
    """Sample retrieved chunks for testing."""
    return [
        {
            "chunk_id": "chunk_001",
            "source": "wikipedia_python",
            "text": "Python is a high-level programming language. It was created by Guido van Rossum and first released in 1991.",
            "score": 0.95,
            "retrieval_method": "dense",
        },
    ]
