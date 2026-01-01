"""Reciprocal Rank Fusion (RRF) and reranker fine-tuning for FaithForge.

RRF merges rankings from dense (embedding) and sparse (BM25) retrievers
into a single fused ranking without requiring score normalization.

The reranker is a fine-tuned cross-encoder that rescores (query, passage) pairs
for final ranking.

References:
    - Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
      individual Rank Learning Methods", SIGIR 2009.
"""

import json
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger

logger = get_logger("faithforge.fusion")


# ── RRF Fusion ───────────────────────────────────────────────────────────────

def rrf_score(rank: int, k: int = 60) -> float:
    """Compute the Reciprocal Rank Fusion score for a single result.

    RRF score = 1 / (k + rank)

    Args:
        rank: 1-based rank position in a result list (rank=1 is first).
        k: Smoothing constant (default 60, per the RRF paper).

    Returns:
        The RRF contribution score.

    Example:
        >>> rrf_score(1, k=60)
        0.01639344262295082
        >>> rrf_score(10, k=60)
        0.014285714285714285
    """
    if rank < 1:
        raise ValueError(f"Rank must be >= 1, got {rank}")
    return 1.0 / (k + rank)


def rrf_merge(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge dense and sparse retrieval results using Reciprocal Rank Fusion.

    Each chunk's RRF score is the sum of its contributions from every ranking
    list it appears in. A chunk appearing in both dense and sparse results
    gets a higher fused score than one appearing in only one list.

    Args:
        dense_results: List of (chunk_id, embedding_score), sorted by score desc.
        sparse_results: List of (chunk_id, bm25_score), sorted by score desc.
        k: RRF smoothing constant.

    Returns:
        Merged list of (chunk_id, rrf_score), sorted by rrf_score desc.
        Each chunk_id appears once; its RRF score is the sum of its
        contributions from each ranking list it appears in.

    Example:
        >>> dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        >>> sparse = [("b", 5.0), ("d", 4.0), ("a", 3.0)]
        >>> rrf_merge(dense, sparse, k=60)
        [("b", ...), ("a", ...), ("d", ...), ("c", ...)]
    """
    score_dict: dict[str, float] = {}

    # Accumulate RRF scores from dense results
    for rank, (chunk_id, _score) in enumerate(dense_results, start=1):
        score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + rrf_score(rank, k)

    # Accumulate RRF scores from sparse results
    for rank, (chunk_id, _score) in enumerate(sparse_results, start=1):
        score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + rrf_score(rank, k)

    # Sort by accumulated RRF score descending
    merged = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    logger.debug(
        "RRF merge: dense=%d, sparse=%d, fused=%d unique chunks",
        len(dense_results), len(sparse_results), len(merged),
    )
    return merged


# ── Reranker Fine-Tuning ────────────────────────────────────────────────────

def prepare_reranker_training_data(
    queries: list[str],
    passages: list[str],
    relevance_labels: list[int],
    output_path: Path,
) -> None:
    """Prepare training data for the cross-encoder reranker.

    Writes a JSONL file with (query, passage, label) triples suitable
    for sentence-transformers CrossEncoder training.

    Args:
        queries: List of query strings.
        passages: List of corresponding passage strings.
        relevance_labels: Binary relevance labels (1 = relevant, 0 = not relevant).
        output_path: Where to write the prepared training data (JSONL).
    """
    if not (len(queries) == len(passages) == len(relevance_labels)):
        raise ValueError(
            f"Length mismatch: queries={len(queries)}, passages={len(passages)}, "
            f"labels={len(relevance_labels)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for query, passage, label in zip(queries, passages, relevance_labels):
            row = {
                "query": query,
                "passage": passage,
                "label": int(label),
            }
            f.write(json.dumps(row) + "\n")

    logger.info(
        "Prepared %d reranker training samples → %s",
        len(queries), output_path,
    )


def train_reranker(
    train_data_path: Path,
    val_data_path: Path,
    output_dir: Path,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> dict:
    """Fine-tune a cross-encoder reranker.

    Uses sentence-transformers CrossEncoder with a classification head.
    Training data should be JSONL with (query, passage, label) fields.

    Args:
        train_data_path: Path to training data JSONL.
        val_data_path: Path to validation data JSONL.
        output_dir: Where to save the fine-tuned model.
        base_model: HuggingFace model ID for the base cross-encoder.
        epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.

    Returns:
        Dict with training metrics.
    """
    from sentence_transformers import CrossEncoder

    logger.info(
        "Training reranker: base=%s, epochs=%d, batch=%d, lr=%s",
        base_model, epochs, batch_size, learning_rate,
    )

    # Load training data
    train_samples = _load_reranker_data(train_data_path)
    val_samples = _load_reranker_data(val_data_path)

    logger.info("Train samples: %d, Val samples: %d", len(train_samples), len(val_samples))

    # Initialize model
    model = CrossEncoder(base_model, num_labels=1)

    # Train
    model.fit(
        train_dataloader=train_samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_path=str(output_dir),
        evaluation_steps=100,
        warmup_steps=100,
    )

    logger.info("Reranker saved to %s", output_dir)

    return {
        "base_model": base_model,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "epochs": epochs,
        "output_path": str(output_dir),
    }


def _load_reranker_data(data_path: Path) -> list[tuple[str, str, int]]:
    """Load reranker training data from JSONL.

    Returns:
        List of (query, passage, label) tuples.
    """
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            samples.append((row["query"], row["passage"], row["label"]))
    return samples
