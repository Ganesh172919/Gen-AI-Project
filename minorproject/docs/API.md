# FaithForge API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production, add API key authentication.

## Endpoints

### Health Check

```
GET /health
```

Check the health status of the service and its dependencies.

**Response:**
```json
{
    "status": "healthy",
    "service": "FaithForge",
    "version": "0.2.0",
    "llm_provider": "groq",
    "vector_store": "chromadb",
    "queue": {
        "pending": 0,
        "running": 0
    },
    "log_format": "text"
}
```

### Service Info

```
GET /
```

Get service information and available endpoints.

**Response:**
```json
{
    "name": "FaithForge",
    "version": "0.2.0",
    "docs": "/docs",
    "health": "/health",
    "endpoints": {
        "query": "POST /query",
        "query_stream": "GET /query/stream?q=...",
        "evaluate": "POST /evaluate",
        "evaluate_status": "GET /evaluate/status/{job_id}",
        "evaluate_results": "GET /evaluate/results/{job_id}"
    },
    "features": {
        "fine_tuned_verifier": "QLoRA fine-tuned faithfulness checker",
        "adaptive_routing": "Query complexity classification",
        "targeted_correction": "Claim-level correction (not full regen)",
        "hybrid_retrieval": "Dense + Sparse + RRF fusion + Reranking",
        "sse_streaming": "Real-time pipeline progress events"
    }
}
```

---

### Query (Synchronous)

```
POST /query
```

Run a single query through the FaithForge pipeline and return the complete result.

**Request Body:**
```json
{
    "query": "What is Python?",
    "top_k": 10,
    "max_iterations": 3
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | The user's question (1-2000 chars) |
| `top_k` | integer | No | Override retrieval top-k (1-50) |
| `max_iterations` | integer | No | Override verify→correct loop cap (1-5) |

**Response:**
```json
{
    "trace": {
        "query": "What is Python?",
        "strategy": "single_hop",
        "sub_queries": ["What is Python?"],
        "retrieved_chunks": [
            {
                "chunk_id": "doc_42",
                "source": "wiki:Python_(programming_language)",
                "text": "Python is a high-level, general-purpose programming language...",
                "score": 0.92,
                "retrieval_method": "fused"
            }
        ],
        "generated_answer": "Python is a high-level programming language created by Guido van Rossum.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "Python is a high-level programming language",
                "source_chunk_ids": ["doc_42"]
            },
            {
                "claim_id": "c2",
                "text": "Python was created by Guido van Rossum",
                "source_chunk_ids": ["doc_42"]
            }
        ],
        "verifications": [
            {
                "claim_id": "c1",
                "claim_text": "Python is a high-level programming language",
                "evidence_text": "Python is a high-level, general-purpose programming language...",
                "entailment_label": "entailment",
                "faithfulness_score": 0.95,
                "status": "verified",
                "iteration": 1
            }
        ],
        "corrections": [],
        "total_iterations": 1,
        "all_claims_faithful": true
    },
    "final_answer": "Python is a high-level programming language created by Guido van Rossum.",
    "latency_ms": 1234.5
}
```

---

### Query (Streaming)

```
GET /query/stream?q=What+is+Python?&max_iterations=3
```

Stream a query through the pipeline via Server-Sent Events (SSE).

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | Yes | The user's question (1-2000 chars) |
| `max_iterations` | integer | No | Override verify→correct loop cap (1-5) |

**SSE Events:**

Each event has an `event` type and `data` payload.

#### Stage Events

```sse
event: stage
data: {"stage": "planner", "status": "running"}

event: stage
data: {"stage": "planner", "status": "complete", "data": {"strategy": "single_hop", "sub_queries": ["What is Python?"]}}

event: stage
data: {"stage": "retriever", "status": "running"}

event: stage
data: {"stage": "retriever", "status": "complete", "data": {"chunks_found": 10, "chunks": [...]}}

event: stage
data: {"stage": "generator", "status": "running"}

event: stage
data: {"stage": "generator", "status": "complete", "data": {"answer_length": 150, "claims_count": 3, "claims": [...]}}

event: stage
data: {"stage": "verifier", "status": "running"}

event: stage
data: {"stage": "verifier", "status": "complete", "data": {"total": 3, "passed": 2, "failed": 1, "verifications": [...]}}

event: stage
data: {"stage": "corrector", "status": "running"}

event: stage
data: {"stage": "corrector", "status": "complete", "data": {"corrections_count": 1, "corrections": [...]}}
```

#### Done Event

```sse
event: done
data: {"trace": {...}, "final_answer": "...", "latency_ms": 1234.5}
```

**Stage Values:**
- `planner` — Query complexity classification
- `retriever` — Evidence retrieval
- `generator` — Answer generation
- `verifier` — Claim verification
- `corrector` — Claim correction (only if claims failed)

**Status Values:**
- `running` — Stage is executing
- `complete` — Stage finished (check `data` for results)

---

### Submit Evaluation

```
POST /evaluate
```

Submit a batch evaluation job for async processing.

**Request Body:**
```json
{
    "dataset_name": "ragtruth",
    "sample_size": 100,
    "run_ablations": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_name` | string | Yes | Dataset name (e.g., "ragtruth", "hotpotqa", "custom") |
| `sample_size` | integer | No | Number of samples (1-1000) |
| `run_ablations` | boolean | No | Run ablation studies (default: true) |

**Response:**
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued"
}
```

---

### Evaluation Status

```
GET /evaluate/status/{job_id}
```

Poll the status of an evaluation job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID from POST /evaluate |

**Response:**
```json
{
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "running",
    "created_at": "2024-01-15T10:30:00Z",
    "started_at": "2024-01-15T10:30:05Z",
    "completed_at": null,
    "error": null
}
```

**Status Values:**
- `pending` — Job is queued
- `running` — Job is executing
- `completed` — Job finished successfully
- `failed` — Job failed (check `error`)

---

### Evaluation Results

```
GET /evaluate/results/{job_id}
```

Get the results of a completed evaluation job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID from POST /evaluate |

**Response:**
```json
{
    "metrics": {
        "total_queries": 100,
        "faithfulness_accuracy": 0.85,
        "claim_level_precision": 0.90,
        "claim_level_recall": 0.80,
        "avg_iterations": 1.5,
        "avg_latency_ms": 250.0,
        "ragas_faithfulness": 0.82,
        "ragas_answer_relevancy": 0.88,
        "ragas_context_precision": 0.75,
        "ragas_context_recall": 0.70
    },
    "ablations": [
        {
            "ablation_name": "verifier_vs_self_critique",
            "description": "Fine-tuned verifier vs. same-LLM self-critique",
            "baseline_metric": 0.70,
            "ablated_metric": 0.85,
            "improvement_pct": 21.4
        }
    ],
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "completed_at": "2024-01-15T10:35:00Z"
}
```

---

## Error Responses

### 422 Validation Error

```json
{
    "detail": [
        {
            "loc": ["body", "query"],
            "msg": "ensure this value has at least 1 characters",
            "type": "value_error"
        }
    ]
}
```

### 404 Not Found

```json
{
    "detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

### 500 Internal Server Error

```json
{
    "error": "Internal server error",
    "detail": "...",
    "request_id": "abc12345"
}
```

---

## Headers

### Request Headers

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/json` for POST requests |
| `Accept` | `text/event-stream` for SSE streaming |

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier (8 chars) |
| `X-Response-Time` | Response time in milliseconds |
| `Content-Type` | `application/json` or `text/event-stream` |

---

## Rate Limiting

Currently, there are no rate limits. For production, implement rate limiting based on your needs.

## CORS

Allowed origins are configured via `CORS_ORIGINS` environment variable. Default: `["http://localhost:3000"]`.
