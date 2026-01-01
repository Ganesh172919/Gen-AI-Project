# ⚒️ FaithForge

**A Self-Verifying, Multi-Agent Retrieval-Augmented Generation System with a Fine-Tuned Faithfulness Verifier**

> Most RAG systems hope the LLM tells the truth. FaithForge trains a small model whose only job is to check.

---

## Architecture

```
User Query
   │
   ▼
[1] Planner Agent ──(complexity classifier)──> retrieval strategy
   │                                             (none / single-hop / multi-hop)
   ▼
[2] Hybrid Retriever
   ├─ Dense (ChromaDB/pgvector embeddings)
   ├─ Sparse (BM25)
   ├─ Reciprocal Rank Fusion
   └─ Fine-tuned cross-encoder reranker
   │
   ▼
[3] Grounded Generator (Groq/Cerebras/OpenRouter LLM)
   → answer with inline claim-to-source tags
   │
   ▼
[4] Fine-Tuned Faithfulness Verifier (LoRA/QLoRA model)
   → per-claim entailment/contradiction/neutral + faithfulness score
   │
   ├─ all claims pass ──────────────────► Final Answer + Evidence Trace
   │
   └─ claims fail ──► [5] Corrective Agent
                         ├─ re-retrieve for flagged claims only
                         └─ targeted claim rewrite (not full regen)
                         │
                         └──► back to [4], max N iterations
```

## Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| Vector store | ChromaDB or pgvector (configurable) |
| Sparse retrieval | rank_bm25 |
| Reranker fine-tuning | sentence-transformers CrossEncoder + PEFT/LoRA |
| Verifier fine-tuning | HuggingFace Transformers + PEFT (QLoRA, 4-bit) |
| Generator LLM | Groq / Cerebras / OpenRouter free-tier models |
| Queue | Redis |
| Backend | FastAPI + SSE streaming |
| Frontend | Next.js + Tailwind CSS |
| Tracing | OpenTelemetry |
| Evaluation | RAGAS + custom scripts |

## Key Innovations

### 1. Fine-Tuned Faithfulness Verifier
Unlike RAG systems that rely on the same LLM to self-check, FaithForge trains a dedicated small model (Qwen2.5-1.5B) using QLoRA for claim-level verification. This independent verifier catches hallucinations that the generator's self-critique misses.

### 2. Adaptive Query Complexity Routing
The planner agent classifies queries as none/single-hop/multi-hop and decomposes complex queries into sub-queries. This reduces unnecessary retrieval calls for simple questions while ensuring thorough evidence gathering for complex ones.

### 3. Targeted Claim Correction
When verification fails, only the flagged claims are re-retrieved and rewritten—not the entire answer. This preserves correct claims, reduces token usage, and cuts latency compared to full regeneration.

### 4. Hybrid Retrieval with RRF Fusion
Combines dense (embedding) and sparse (BM25) retrieval using Reciprocal Rank Fusion, followed by cross-encoder reranking. This captures both semantic similarity and lexical matching.

### 5. Real-Time Pipeline Streaming
The `/query/stream` endpoint uses Server-Sent Events to stream stage-by-stage progress, enabling the frontend to animate the pipeline in real time.

## Project Structure

```
faithforge/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI routes
│   │   │   ├── query.py      # POST /query + GET /query/stream (SSE)
│   │   │   └── evaluate.py   # POST /evaluate + status/results
│   │   ├── agents/           # Agent implementations
│   │   │   ├── planner.py    # Query complexity classifier
│   │   │   ├── verifier.py   # Faithfulness verifier (QLoRA)
│   │   │   ├── corrector.py  # Corrective agent
│   │   │   └── graph.py      # LangGraph orchestration
│   │   ├── core/             # Config, logging
│   │   ├── models/           # Pydantic schemas
│   │   ├── services/         # LLM adapter, retriever, generator, queue, tracing
│   │   ├── worker.py         # Background job processor
│   │   └── main.py           # FastAPI entry point
│   ├── scripts/
│   │   └── ingest_corpus.py  # Data ingestion CLI
│   ├── verifier/
│   │   ├── train.py          # QLoRA fine-tuning
│   │   └── data_synthesis.py # Synthetic data generation
│   ├── retrieval/
│   │   └── fusion.py         # RRF fusion + reranker training
│   ├── evaluation/
│   │   └── ablations.py      # 4 ablation studies
│   ├── tests/                # Comprehensive test suite
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # Dashboard (streaming pipeline)
│   │   │   ├── evaluate/page.tsx # Evaluation dashboard
│   │   │   └── layout.tsx        # Root layout with nav
│   │   ├── components/
│   │   │   ├── QueryInput.tsx
│   │   │   ├── PipelineVisualization.tsx
│   │   │   ├── ClaimsDisplay.tsx
│   │   │   ├── CorrectionHistory.tsx
│   │   │   └── SkeletonLoader.tsx
│   │   ├── lib/api.ts        # API client (streaming + REST)
│   │   └── types/index.ts    # TypeScript types
│   ├── package.json
│   └── Dockerfile
├── docs/                     # Comprehensive documentation
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── DEPLOYMENT.md
├── docker-compose.yml
├── CONTRIBUTING.md
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis (optional, for queue features)
- PostgreSQL with pgvector (optional, if using pgvector backend)

### 1. Clone & Configure

```bash
cd minorproject
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys
```

### 2. Start with Docker Compose (recommended)

```bash
docker compose up --build
```

This starts:
- **Backend** at http://localhost:8000 (API docs at /docs)
- **Worker** (background job processor)
- **Frontend** at http://localhost:3000
- **Redis** at localhost:6379
- **PostgreSQL** at localhost:5432

### 3. Start Manually (development)

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env  # fill in your API keys
uvicorn app.main:app --reload --port 8000
```

**Worker (separate terminal):**
```bash
cd backend
python -m app.worker
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### 4. Ingest a Corpus

```bash
cd backend
python scripts/ingest_corpus.py --source hotpotqa --max-docs 1000
```

### 5. Run Tests

```bash
cd backend
pytest -v
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/query/stream?q=...` | SSE stream with stage-by-stage progress |
| `POST` | `/query` | Run a single query (returns full result) |
| `POST` | `/evaluate` | Submit a batch evaluation job |
| `GET` | `/evaluate/status/{job_id}` | Poll job status |
| `GET` | `/evaluate/results/{job_id}` | Get evaluation results |
| `GET` | `/health` | Health check with dependency status |
| `GET` | `/docs` | Swagger UI |

## Evaluation Plan

| Ablation | What's Compared |
|---|---|
| 1 | Fine-tuned verifier vs. same-LLM self-critique |
| 2 | Adaptive retrieval routing vs. fixed routing |
| 3 | Targeted claim correction vs. full regeneration |
| 4 | Verifier score vs. RAGAS faithfulness score |

## Logging & Observability

FaithForge provides comprehensive logging:

- **Structured JSON logging** for production (`LOG_FORMAT=json`)
- **Human-readable colored logs** for development
- **File logging with rotation** (`LOG_FILE_PATH=./logs/faithforge.log`)
- **Request correlation IDs** (`X-Request-ID` header)
- **Response timing** (`X-Response-Time` header)
- **OpenTelemetry tracing** through all agent hops
- **Pipeline stage logging** with timing for each stage

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) — Detailed system architecture
- [API Reference](docs/API.md) — Full API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) — Production deployment
- [Contributing Guide](CONTRIBUTING.md) — Development workflow

## License

MIT
