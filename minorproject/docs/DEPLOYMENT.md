# FaithForge Deployment Guide

## Overview

This guide covers deploying FaithForge in various environments, from local development to production.

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+
- Redis (optional)
- PostgreSQL with pgvector (optional)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd minorproject

# Backend setup
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env
# Edit .env with your API keys

# Start backend
uvicorn app.main:app --reload --port 8000

# In another terminal, start worker
python -m app.worker

# In another terminal, start frontend
cd frontend
npm install
npm run dev
```

### Environment Variables

Create `backend/.env` from `.env.example`:

```bash
# Required: At least one LLM API key
GROQ_API_KEY=your_groq_key
# or
CEREBRAS_API_KEY=your_cerebras_key
# or
OPENROUTER_API_KEY=your_openrouter_key

# Optional: Redis for job queue
REDIS_URL=redis://localhost:6379/0

# Optional: Logging
LOG_LEVEL=INFO
LOG_FORMAT=text
# LOG_FILE_PATH=./logs/faithforge.log
```

---

## Docker Compose

### Production Setup

```bash
# Build and start all services
docker compose up --build -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| backend | 8000 | FastAPI API server |
| worker | - | Background job processor |
| frontend | 3000 | Next.js web app |
| redis | 6379 | Job queue |
| postgres | 5432 | Vector store (pgvector) |

### Custom Configuration

Edit `docker-compose.yml` to customize:

```yaml
services:
  backend:
    environment:
      - LOG_LEVEL=DEBUG
      - LOG_FORMAT=json
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

---

## Production Deployment

### Backend

#### Using Gunicorn

```bash
pip install gunicorn

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

#### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Frontend

#### Build for Production

```bash
cd frontend
npm run build
npm start
```

#### Using Docker

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json .
RUN npm ci --production
CMD ["npm", "start"]
```

---

## Environment Variables Reference

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | FaithForge | Application name |
| `DEBUG` | false | Enable debug mode |
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | text | Log format (text or json) |
| `LOG_FILE_PATH` | - | Path to log file (optional) |
| `LOG_MAX_BYTES` | 10000000 | Max log file size before rotation |
| `LOG_BACKUP_COUNT` | 7 | Number of rotated log backups |

### LLM Providers

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | groq | LLM provider (groq, cerebras, openrouter) |
| `GROQ_API_KEY` | - | Groq API key |
| `CEREBRAS_API_KEY` | - | Cerebras API key |
| `OPENROUTER_API_KEY` | - | OpenRouter API key |
| `GENERATOR_MODEL` | llama-3.3-70b-versatile | Model for answer generation |
| `GENERATOR_TEMPERATURE` | 0.3 | Generation temperature |
| `GENERATOR_MAX_TOKENS` | 2048 | Max tokens per generation |

### Vector Store

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_TYPE` | chromadb | Backend (chromadb or pgvector) |
| `CHROMADB_PATH` | ./data/chromadb | ChromaDB storage path |
| `PGVECTOR_URL` | postgresql://... | PostgreSQL connection URL |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |

### Retriever

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRIEVAL_TOP_K` | 10 | Number of results to retrieve |
| `RERANKER_TOP_K` | 5 | Number of results after reranking |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker model |
| `BM25_INDEX_PATH` | ./data/bm25_index.pkl | BM25 index file path |

### Verifier

| Variable | Default | Description |
|----------|---------|-------------|
| `VERIFIER_MODEL_PATH` | ./models/verifier | Fine-tuned verifier path |
| `VERIFIER_BASE_MODEL` | Qwen/Qwen2.5-1.5B-Instruct | Base model for verifier |
| `VERIFIER_MAX_ITERATIONS` | 3 | Max verify→correct loops |
| `VERIFIER_CONFIDENCE_THRESHOLD` | 0.7 | Threshold for verification |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection URL |
| `OTEL_EXPORTER_ENDPOINT` | http://localhost:4317 | OpenTelemetry endpoint |
| `OTEL_SERVICE_NAME` | faithforge-backend | Service name for tracing |
| `CORS_ORIGINS` | ["http://localhost:3000"] | Allowed CORS origins |

---

## Data Ingestion

### HotpotQA

```bash
cd backend
python scripts/ingest_corpus.py --source hotpotqa --max-docs 1000
```

### Custom Data

```bash
python scripts/ingest_corpus.py --source custom --file data/my_corpus.jsonl
```

**JSONL Format:**
```json
{"text": "Document text...", "source": "source_name"}
```

### RAGTruth

```bash
# Download RAGTruth data first
# https://github.com/IAAR-Shanghai/RAGTruth
python scripts/ingest_corpus.py --source ragtruth
```

---

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Logs

```bash
# Docker logs
docker compose logs -f backend

# File logs (if configured)
tail -f logs/faithforge.log
```

### OpenTelemetry

Configure OTLP exporter for production:

```bash
OTEL_EXPORTER_ENDPOINT=http://your-otel-collector:4317
```

---

## Scaling

### Horizontal Scaling

- **Backend:** Add more backend containers behind a load balancer
- **Worker:** Add more worker containers for parallel job processing
- **Frontend:** Static files can be served from CDN

### Vertical Scaling

- Increase memory for vector store
- Use GPU for model inference
- Increase Redis memory for job queue

---

## Troubleshooting

### Common Issues

**Redis connection failed:**
```bash
# Check Redis is running
redis-cli ping

# Or disable Redis features
# Queue features will be unavailable
```

**ChromaDB not found:**
```bash
# Run data ingestion first
python scripts/ingest_corpus.py --source hotpotqa --max-docs 100
```

**LLM API errors:**
```bash
# Check API key is set
echo $GROQ_API_KEY

# Check provider status
curl https://api.groq.com/openai/v1/models
```

**Model loading fails:**
```bash
# Check model path
ls -la ./models/verifier

# Or use fallback mode (rule-based verification)
```

---

## Security

### API Keys

- Never commit API keys to version control
- Use environment variables or secrets management
- Rotate keys regularly

### Network

- Use HTTPS in production
- Restrict CORS origins
- Implement rate limiting

### Data

- Encrypt sensitive data at rest
- Use secure Redis connections
- Implement access controls
