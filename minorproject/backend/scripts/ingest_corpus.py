"""Data ingestion script for FaithForge.

Loads a corpus of passages into the vector store and builds a BM25 index.

Usage:
    python scripts/ingest_corpus.py --source hotpotqa --max-docs 1000
    python scripts/ingest_corpus.py --source custom --file data/my_corpus.jsonl
    python scripts/ingest_corpus.py --source ragtruth

Sources:
    - hotpotqa: Load from HuggingFace datasets (hotpotqa)
    - ragtruth: Load RAGTruth benchmark data
    - custom: Load from a JSONL file with {"text": "...", "source": "..."} per line
"""

import argparse
import asyncio
import json
import pickle
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import get_logger, setup_logging

logger = get_logger("faithforge.ingest")


async def ingest_hotpotqa(max_docs: int = 1000) -> list[dict]:
    """Load passages from HotpotQA via HuggingFace datasets.

    Args:
        max_docs: Maximum number of documents to load.

    Returns:
        List of dicts with 'text' and 'source' keys.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        return []

    logger.info("Loading HotpotQA (max_docs=%d)...", max_docs)

    dataset = load_dataset("hotpotqa", "fullwiki", split="train", streaming=True)

    documents = []
    seen_texts = set()

    for i, row in enumerate(dataset):
        if len(documents) >= max_docs:
            break

        # HotpotQA has context as list of (title, sentences) tuples
        context = row.get("context", [])
        for title, sentences in context:
            text = " ".join(sentences)
            if text not in seen_texts and len(text) > 50:
                seen_texts.add(text)
                documents.append({
                    "text": text,
                    "source": f"hotpotqa:{title}",
                    "metadata": {"dataset": "hotpotqa", "title": title},
                })

        if (i + 1) % 100 == 0:
            logger.info("  processed %d rows, collected %d docs", i + 1, len(documents))

    logger.info("Loaded %d unique passages from HotpotQA", len(documents))
    return documents


async def ingest_ragtruth() -> list[dict]:
    """Load RAGTruth benchmark data.

    RAGTruth is a human-annotated hallucination benchmark for RAG outputs.
    This loads the source passages (not the annotations — those are used
    for evaluation in evaluation/ablations.py).

    Returns:
        List of dicts with 'text' and 'source' keys.
    """
    ragtruth_path = Path("./data/ragtruth")

    if not ragtruth_path.exists():
        logger.warning(
            "RAGTruth data not found at %s. "
            "Download from https://github.com/IAAR-Shanghai/RAGTruth "
            "and place in ./data/ragtruth/",
            ragtruth_path,
        )
        return []

    documents = []
    for json_file in ragtruth_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if "context" in item:
                        documents.append({
                            "text": item["context"],
                            "source": f"ragtruth:{json_file.stem}",
                            "metadata": {"dataset": "ragtruth", "file": json_file.name},
                        })

    logger.info("Loaded %d passages from RAGTruth", len(documents))
    return documents


async def ingest_custom(file_path: str) -> list[dict]:
    """Load documents from a custom JSONL file.

    Each line should be: {"text": "...", "source": "..."}

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dicts with 'text' and 'source' keys.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        return []

    documents = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "text" in row:
                    documents.append({
                        "text": row["text"],
                        "source": row.get("source", f"custom:line_{line_num}"),
                        "metadata": row.get("metadata", {}),
                    })
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at line %d", line_num)

    logger.info("Loaded %d documents from %s", len(documents), path)
    return documents


async def build_chromadb_index(documents: list[dict]) -> None:
    """Build a ChromaDB collection from documents.

    Args:
        documents: List of dicts with 'text', 'source', and optional 'metadata'.
    """
    try:
        import chromadb
    except ImportError:
        logger.error("chromadb not installed. Run: pip install chromadb")
        return

    logger.info("Building ChromaDB index at %s...", settings.chromadb_path)

    client = chromadb.PersistentClient(path=settings.chromadb_path)

    # Delete existing collection if it exists
    try:
        client.delete_collection("faithforge")
    except Exception:
        pass

    collection = client.create_collection(
        name="faithforge",
        metadata={"hnsw:space": "cosine"},
    )

    # Add documents in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        ids = [f"doc_{i + j}" for j in range(len(batch))]
        texts = [d["text"] for d in batch]
        metadatas = [{"source": d["source"], **d.get("metadata", {})} for d in batch]

        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        logger.info("  indexed %d/%d docs", min(i + batch_size, len(documents)), len(documents))

    logger.info("ChromaDB index built: %d documents", collection.count())


async def build_bm25_index(documents: list[dict]) -> None:
    """Build a BM25 index from documents and save to disk.

    Args:
        documents: List of dicts with 'text' and 'source'.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.error("rank-bm25 not installed. Run: pip install rank-bm25")
        return

    logger.info("Building BM25 index...")

    # Tokenize documents
    tokenized_docs = [doc["text"].lower().split() for doc in documents]

    bm25 = BM25Okapi(tokenized_docs)

    # Save index + metadata
    index_path = Path(settings.bm25_index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    index_data = {
        "bm25": bm25,
        "documents": documents,
    }

    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info("BM25 index saved to %s (%d documents)", index_path, len(documents))


async def main():
    parser = argparse.ArgumentParser(description="Ingest a corpus into FaithForge")
    parser.add_argument(
        "--source",
        choices=["hotpotqa", "ragtruth", "custom"],
        required=True,
        help="Data source to ingest",
    )
    parser.add_argument("--file", type=str, help="Path to custom JSONL file (for --source custom)")
    parser.add_argument("--max-docs", type=int, default=1000, help="Max documents to load")
    parser.add_argument(
        "--skip-chromadb", action="store_true", help="Skip ChromaDB index building"
    )
    parser.add_argument(
        "--skip-bm25", action="store_true", help="Skip BM25 index building"
    )

    args = parser.parse_args()
    setup_logging()

    # Load documents
    if args.source == "hotpotqa":
        documents = await ingest_hotpotqa(args.max_docs)
    elif args.source == "ragtruth":
        documents = await ingest_ragtruth()
    elif args.source == "custom":
        if not args.file:
            logger.error("--file required for --source custom")
            return
        documents = await ingest_custom(args.file)
    else:
        logger.error("Unknown source: %s", args.source)
        return

    if not documents:
        logger.warning("No documents loaded — nothing to index")
        return

    # Truncate to max_docs
    if len(documents) > args.max_docs:
        documents = documents[: args.max_docs]
        logger.info("Truncated to %d documents", args.max_docs)

    # Build indexes
    if not args.skip_chromadb:
        await build_chromadb_index(documents)

    if not args.skip_bm25:
        await build_bm25_index(documents)

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    asyncio.run(main())
