# FaithForge Backend — Dockerfile
# Multi-stage build for smaller final image

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ─────────────────────────────────────────────────────────────

FROM base AS deps

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# ── Application ──────────────────────────────────────────────────────────────

FROM deps AS app

COPY . .

# Create data directories
RUN mkdir -p /app/data/chromadb /app/data/bm25_index /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
