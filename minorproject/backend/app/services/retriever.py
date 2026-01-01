"""Hybrid retriever service for FaithForge.

Combines dense (embedding) retrieval, sparse (BM25) retrieval, and
Reciprocal Rank Fusion (RRF) to produce a unified ranked list of
evidence chunks, followed by cross-encoder reranking.

The dense backend is swappable between ChromaDB and pgvector via config.
"""

import pickle
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import RetrievedChunk

logger = get_logger("faithforge.retriever")


class HybridRetriever:
    """Hybrid retriever combining dense + sparse + RRF fusion + reranking.

    Pipeline:
        1. Dense retrieval → top-K embedding matches (ChromaDB or pgvector)
        2. Sparse retrieval → top-K BM25 matches
        3. RRF fusion → merged ranking (no score normalization needed)
        4. Reranker → cross-encoder rescores (query, passage) pairs → final top-K
    """

    def __init__(self):
        self._dense_backend = None  # ChromaDB collection
        self._bm25_index = None  # BM25Okapi instance
        self._bm25_documents = None  # Documents backing the BM25 index
        self._reranker = None  # CrossEncoder model
        self._embedding_model = None  # SentenceTransformer for dense search
        self._initialized = False

    async def initialize(self) -> None:
        """Load indexes and models.

        Called once per request (or at startup for singleton usage).
        Loads:
        - Dense: ChromaDB collection or pgvector connection
        - Sparse: BM25 index from disk
        - Reranker: cross-encoder model
        """
        logger.info(
            "Initializing retriever (backend=%s, embedding=%s)",
            settings.vector_store_type.value,
            settings.embedding_model,
        )

        # Load dense backend
        await self._init_dense_backend()

        # Load BM25 index
        self._init_bm25_index()

        # Load reranker
        self._init_reranker()

        self._initialized = True
        logger.info("Retriever initialized successfully")

    async def _init_dense_backend(self) -> None:
        """Initialize the dense vector store backend."""
        if settings.vector_store_type.value == "chromadb":
            try:
                import chromadb
                client = chromadb.PersistentClient(path=settings.chromadb_path)
                self._dense_backend = client.get_or_create_collection(
                    name="faithforge",
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB loaded: %d documents",
                    self._dense_backend.count(),
                )
            except Exception as e:
                logger.warning("ChromaDB init failed: %s", e)
                self._dense_backend = None
        else:
            logger.info("pgvector backend selected (not yet implemented)")
            self._dense_backend = None

    def _init_bm25_index(self) -> None:
        """Load the BM25 index from disk."""
        index_path = Path(settings.bm25_index_path)
        if not index_path.exists():
            logger.warning("BM25 index not found at %s", index_path)
            return

        try:
            with open(index_path, "rb") as f:
                index_data = pickle.load(f)
            self._bm25_index = index_data["bm25"]
            self._bm25_documents = index_data["documents"]
            logger.info("BM25 index loaded: %d documents", len(self._bm25_documents))
        except Exception as e:
            logger.warning("BM25 index load failed: %s", e)

    def _init_reranker(self) -> None:
        """Load the cross-encoder reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(settings.reranker_model)
            logger.info("Reranker loaded: %s", settings.reranker_model)
        except Exception as e:
            logger.warning("Reranker load failed: %s", e)

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve evidence chunks for a query.

        Pipeline:
        1. Dense retrieval → top-K embedding matches
        2. Sparse retrieval → top-K BM25 matches
        3. RRF fusion → merged ranking
        4. Reranker → final top-K

        Args:
            query: The search query.
            top_k: Override the default number of results.

        Returns:
            Ranked list of RetrievedChunk objects.
        """
        if not self._initialized:
            raise RuntimeError("Retriever not initialized — call initialize() first")

        k = top_k or settings.retriever_top_k
        logger.info("Retrieving: query_len=%d, top_k=%d", len(query), k)

        # Step 1: Dense retrieval
        dense_results = await self._dense_search(query, k * 2)  # Over-fetch for fusion

        # Step 2: Sparse retrieval
        sparse_results = await self._sparse_search(query, k * 2)

        # Step 3: RRF fusion
        fused = await self._rrf_fusion(dense_results, sparse_results)

        # Step 4: Rerank
        reranked = await self._rerank(query, fused, k)

        logger.info("Retrieved %d chunks (dense=%d, sparse=%d, fused=%d)",
                     len(reranked), len(dense_results), len(sparse_results), len(fused))

        return reranked

    async def retrieve_for_claim(
        self,
        claim: str,
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """Targeted retrieval for a single claim (used in correction loop).

        Same pipeline as retrieve() but optimized for shorter,
        claim-specific queries with fewer results.
        """
        return await self.retrieve(claim, top_k=top_k)

    async def _dense_search(self, query: str, k: int) -> list[tuple[str, float, str]]:
        """Dense embedding search via ChromaDB.

        Returns:
            List of (chunk_id, score, text) tuples sorted by score desc.
        """
        if self._dense_backend is None:
            logger.debug("Dense backend not available, returning empty results")
            return []

        try:
            results = self._dense_backend.query(
                query_texts=[query],
                n_results=min(k, self._dense_backend.count()),
            )

            chunks = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    text = results["documents"][0][i] if results["documents"] else ""
                    # ChromaDB returns distances; convert to similarity score
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    score = 1.0 - distance  # cosine distance → similarity
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    source = metadata.get("source", "unknown")
                    chunks.append((doc_id, score, text))

            return chunks

        except Exception as e:
            logger.error("Dense search failed: %s", e)
            return []

    async def _sparse_search(self, query: str, k: int) -> list[tuple[str, float, str]]:
        """Sparse BM25 search over the corpus.

        Returns:
            List of (chunk_id, score, text) tuples sorted by score desc.
        """
        if self._bm25_index is None:
            logger.debug("BM25 index not available, returning empty results")
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self._bm25_index.get_scores(tokenized_query)

            # Get top-K indices
            import numpy as np
            top_indices = np.argsort(scores)[::-1][:k]

            chunks = []
            for idx in top_indices:
                score = float(scores[idx])
                if score <= 0:
                    break
                doc = self._bm25_documents[idx]
                chunk_id = f"bm25_{idx}"
                chunks.append((chunk_id, score, doc["text"]))

            return chunks

        except Exception as e:
            logger.error("BM25 search failed: %s", e)
            return []

    async def _rrf_fusion(
        self,
        dense: list[tuple[str, float, str]],
        sparse: list[tuple[str, float, str]],
    ) -> list[tuple[str, float, str]]:
        """Reciprocal Rank Fusion combining dense + sparse results.

        Args:
            dense: List of (chunk_id, score, text) from dense search.
            sparse: List of (chunk_id, score, text) from sparse search.

        Returns:
            Merged list of (chunk_id, rrf_score, text) sorted by RRF score desc.
        """
        from retrieval.fusion import rrf_merge

        # Extract (chunk_id, score) pairs for RRF
        dense_pairs = [(cid, score) for cid, score, _ in dense]
        sparse_pairs = [(cid, score) for cid, score, _ in sparse]

        # Build text lookup
        text_lookup: dict[str, str] = {}
        for cid, _, text in dense:
            text_lookup[cid] = text
        for cid, _, text in sparse:
            text_lookup[cid] = text

        # Merge with RRF
        merged_pairs = rrf_merge(dense_pairs, sparse_pairs)

        # Reconstruct with text
        result = []
        for cid, rrf_score in merged_pairs:
            text = text_lookup.get(cid, "")
            result.append((cid, rrf_score, text))

        return result

    async def _rerank(
        self,
        query: str,
        chunks: list[tuple[str, float, str]],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using the cross-encoder reranker.

        Args:
            query: The search query.
            chunks: List of (chunk_id, rrf_score, text) from fusion.
            top_k: Number of final results to return.

        Returns:
            Reranked list of RetrievedChunk objects.
        """
        if not chunks:
            return []

        if self._reranker is None:
            # No reranker available — return top-K from fusion directly
            logger.debug("Reranker not available, returning fusion results directly")
            return [
                RetrievedChunk(
                    chunk_id=cid,
                    source="fused",
                    text=text,
                    score=score,
                    retrieval_method="fused",
                )
                for cid, score, text in chunks[:top_k]
            ]

        try:
            # Prepare (query, passage) pairs for the cross-encoder
            pairs = [(query, text) for _, _, text in chunks]

            # Score with cross-encoder
            scores = self._reranker.predict(pairs)

            # Sort by cross-encoder score
            scored_chunks = list(zip(chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Build final results
            results = []
            for (cid, _rrf_score, text), rerank_score in scored_chunks[:top_k]:
                results.append(RetrievedChunk(
                    chunk_id=cid,
                    source="reranked",
                    text=text,
                    score=float(rerank_score),
                    retrieval_method="fused",
                ))

            return results

        except Exception as e:
            logger.error("Reranking failed: %s", e)
            # Fallback to fusion results
            return [
                RetrievedChunk(
                    chunk_id=cid,
                    source="fused",
                    text=text,
                    score=score,
                    retrieval_method="fused",
                )
                for cid, score, text in chunks[:top_k]
            ]

    async def close(self) -> None:
        """Release resources."""
        self._dense_backend = None
        self._bm25_index = None
        self._bm25_documents = None
        self._reranker = None
        self._initialized = False
        logger.info("Retriever closed")
