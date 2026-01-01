# FaithForge Architecture Guide

## Overview

FaithForge is a self-verifying, multi-agent RAG system that addresses the hallucination problem in retrieval-augmented generation. Unlike traditional RAG systems that trust the generator LLM's output, FaithForge introduces a fine-tuned faithfulness verifier that independently checks each claim against the retrieved evidence.

## System Components

### 1. Planner Agent

**Purpose:** Classify query complexity and determine retrieval strategy.

**Classification:**
- `none` — Trivial queries that don't need retrieval (e.g., "What is 2+2?")
- `single_hop` — Queries requiring one retrieval step (e.g., "When was Python created?")
- `multi_hop` — Queries requiring multiple retrieval steps (e.g., "Compare GDP of India and China")

**Implementation:**
- Baseline: LLM-based few-shot classification
- Innovation: Fine-tuned DistilBERT for lower latency

**Output:** Retrieval strategy + list of sub-queries

### 2. Hybrid Retriever

**Purpose:** Retrieve relevant evidence chunks using multiple retrieval methods.

**Pipeline:**
```
Query → [Dense Search] + [Sparse Search] → RRF Fusion → Reranker → Top-K Results
```

**Components:**

#### Dense Retrieval
- **Backend:** ChromaDB or pgvector
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Method:** Cosine similarity search over dense embeddings

#### Sparse Retrieval
- **Algorithm:** BM25Okapi
- **Index:** Pre-built pickle file
- **Method:** Term frequency-inverse document frequency matching

#### Reciprocal Rank Fusion (RRF)
- **Formula:** `score(d) = Σ 1/(k + rank_i(d))`
- **Purpose:** Merge rankings without score normalization
- **Constant:** k=60 (standard from literature)

#### Cross-Encoder Reranker
- **Model:** cross-encoder/ms-marco-MiniLM-L-6-v2
- **Method:** Score (query, passage) pairs directly
- **Innovation:** Fine-tuned on domain-specific data

### 3. Grounded Generator

**Purpose:** Generate answers with claim-level source attribution.

**Prompt Strategy:**
```
Given a user question and retrieved evidence, produce:
1. "answer": Clear, concise answer
2. "claims": List of atomic claims, each with:
   - "claim_id": Short identifier
   - "text": Claim text
   - "source_chunk_ids": Evidence chunk IDs supporting this claim
```

**Key Features:**
- Structured JSON output
- Atomic claim extraction
- Explicit source attribution
- Evidence-grounded generation

### 4. Faithfulness Verifier

**Purpose:** Classify each claim against its evidence as entailment/contradiction/neutral.

**Model Architecture:**
- **Base Model:** Qwen2.5-1.5B-Instruct
- **Fine-tuning:** QLoRA (4-bit quantization + LoRA adapters)
- **Task:** NLI-style classification

**Input Format:**
```
Claim: {claim_text}
Evidence: {evidence_text}
```

**Output Format:**
```
{entailment|contradiction|neutral}
Score: {0.0-1.0}
```

**Verification Logic:**
- `ENTAILMENT` + score ≥ threshold → PASS
- `CONTRADICTION` → FAIL
- `NEUTRAL` → FAIL (soft fail, needs correction)

**Training Data:**
- Synthetic generation via LLM
- Entailed claims from real passages
- Corrupted claims (negation, entity swap, unsupported)

### 5. Corrective Agent

**Purpose:** Fix failed claims through targeted re-retrieval and rewriting.

**Process:**
1. Identify failed claims from verification
2. Re-retrieve evidence specifically for each failed claim
3. Ask LLM to rewrite claim based on new evidence
4. Return correction records for logging

**Key Innovation:** Only corrects flagged claims, not the entire answer. This:
- Preserves correct claims
- Reduces token usage
- Cuts latency vs. full regeneration

## Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Planner Agent  │
│  (classify)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hybrid Retriever│
│ (dense+sparse)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generator     │
│ (claims+answer) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Verifier      │
│ (check claims)  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌─────────┐
│  END  │ │Corrector│
│       │ │ (fix)   │
└───────┘ └────┬────┘
               │
               └──► back to Verifier
```

## State Machine

The LangGraph orchestration implements a state machine:

```python
class ForgeState:
    query: str
    strategy: RetrievalStrategy
    sub_queries: list[str]
    retrieved_chunks: list[RetrievedChunk]
    answer: str
    claims: list[Claim]
    verifications: list[ClaimVerification]
    corrections: list[CorrectionRecord]
    iteration: int
    max_iterations: int
```

**State Transitions:**
1. `START → planner` — Classify query
2. `planner → retriever` — Retrieve evidence
3. `retriever → generator` — Generate answer
4. `generator → verifier` — Verify claims
5. `verifier → END` — All claims pass
6. `verifier → corrector` — Some claims fail
7. `corrector → verifier` — Re-verify after correction

## Configuration

All settings are environment-variable based:

```python
class Settings:
    # App
    app_name: str = "FaithForge"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "text"  # "text" or "json"
    
    # LLM
    llm_provider: LLMProvider  # groq, cerebras, openrouter
    generator_model: str
    generator_temperature: float
    
    # Vector Store
    vector_store_type: VectorStoreType  # chromadb, pgvector
    embedding_model: str
    
    # Retriever
    retriever_top_k: int = 10
    reranker_top_k: int = 5
    reranker_model: str
    
    # Verifier
    verifier_model_path: str
    verifier_base_model: str
    verifier_max_iterations: int = 3
    verifier_confidence_threshold: float = 0.7
```

## Observability

### Logging
- Structured JSON logging for production
- Human-readable colored logs for development
- File logging with rotation
- Request correlation IDs

### Tracing
- OpenTelemetry integration
- Span per agent hop
- Automatic exception recording
- Console exporter (dev) / OTLP exporter (prod)

### Metrics
- Request latency (X-Response-Time)
- Pipeline stage timing
- LLM call statistics
- Queue job statistics

## Error Handling

### Graceful Degradation
- Missing ChromaDB → Skip dense retrieval
- Missing BM25 index → Skip sparse retrieval
- Missing reranker → Use fusion results directly
- Missing verifier → Use rule-based fallback
- Redis unavailable → Disable queue features

### Error Responses
```json
{
    "error": "Internal server error",
    "detail": "...",
    "request_id": "abc12345"
}
```

## Performance Considerations

### Latency
- Planner: ~500ms (LLM call)
- Retriever: ~200ms (dense + sparse + fusion)
- Generator: ~1000ms (LLM call)
- Verifier: ~100ms (local model) or ~500ms (LLM fallback)
- Corrector: ~500ms per claim

### Token Usage
- Generator: ~500 tokens per query
- Corrector: ~100 tokens per claim correction
- Targeted correction saves ~60% tokens vs. full regeneration

### Scaling
- Stateless API servers (horizontal scaling)
- Redis for job queue (distributed processing)
- Vector store for shared index (ChromaDB/PostgreSQL)
