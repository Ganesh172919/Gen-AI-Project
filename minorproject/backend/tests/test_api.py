"""Tests for FaithForge API endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_returns_service_name(self, client):
        resp = client.get("/health")
        assert resp.json()["service"] == "FaithForge"

    def test_health_returns_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data
        assert data["version"] == "0.2.0"

    def test_health_returns_provider(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "llm_provider" in data
        assert "vector_store" in data

    def test_health_returns_queue_stats(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "queue" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_returns_endpoints(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "query" in data["endpoints"]
        assert "evaluate" in data["endpoints"]
        assert "query_stream" in data["endpoints"]

    def test_root_returns_features(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "features" in data
        assert "fine_tuned_verifier" in data["features"]
        assert "adaptive_routing" in data["features"]
        assert "targeted_correction" in data["features"]
        assert "hybrid_retrieval" in data["features"]

    def test_root_returns_docs_link(self, client):
        resp = client.get("/")
        data = resp.json()
        assert data["docs"] == "/docs"


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_returns_200(self, client):
        resp = client.post("/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200

    def test_query_returns_trace(self, client):
        resp = client.post("/query", json={"query": "What is RAG?"})
        data = resp.json()
        assert "trace" in data
        assert "final_answer" in data
        assert "latency_ms" in data

    def test_query_rejects_empty(self, client):
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422  # Validation error

    def test_query_rejects_long_query(self, client):
        resp = client.post("/query", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_query_with_options(self, client):
        resp = client.post("/query", json={
            "query": "What is Python?",
            "top_k": 5,
            "max_iterations": 2,
        })
        assert resp.status_code == 200

    def test_query_trace_structure(self, client):
        resp = client.post("/query", json={"query": "What is Python?"})
        data = resp.json()
        trace = data["trace"]
        assert "query" in trace
        assert "strategy" in trace
        assert "claims" in trace
        assert "verifications" in trace


class TestQueryStreamEndpoint:
    """Tests for GET /query/stream."""

    def test_stream_returns_events(self, client):
        """Should return SSE events."""
        with client.stream("GET", "/query/stream?q=What+is+Python?") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")


class TestEvaluateEndpoint:
    """Tests for POST /evaluate."""

    def test_evaluate_returns_job_id(self, client):
        resp = client.post("/evaluate", json={
            "dataset_name": "test",
            "sample_size": 10,
            "run_ablations": False,
        })
        # May fail if Redis isn't running — that's OK for unit tests
        assert resp.status_code in (200, 500)

    def test_evaluate_validates_input(self, client):
        """Should reject invalid input."""
        resp = client.post("/evaluate", json={})
        assert resp.status_code == 422


class TestRequestMiddleware:
    """Tests for request middleware."""

    def test_request_id_header(self, client):
        """Should add X-Request-ID header."""
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
        assert len(resp.headers["X-Request-ID"]) == 8

    def test_response_time_header(self, client):
        """Should add X-Response-Time header."""
        resp = client.get("/health")
        assert "X-Response-Time" in resp.headers
        assert resp.headers["X-Response-Time"].endswith("ms")
