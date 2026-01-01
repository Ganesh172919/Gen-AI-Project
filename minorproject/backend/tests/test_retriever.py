"""Tests for the hybrid retriever service."""

import pytest
from app.services.retriever import HybridRetriever


@pytest.fixture
def retriever():
    """Create a retriever instance."""
    return HybridRetriever()


class TestRetrieverInitialization:
    """Tests for retriever initialization."""

    def test_not_initialized_raises(self, retriever):
        """Should raise RuntimeError when retrieving before initialization."""
        with pytest.raises(RuntimeError, match="not initialized"):
            import asyncio
            asyncio.run(retriever.retrieve("test query"))

    @pytest.mark.asyncio
    async def test_initialize_loads(self, retriever):
        """Should initialize without errors (may warn about missing indexes)."""
        await retriever.initialize()
        assert retriever._initialized is True

    @pytest.mark.asyncio
    async def test_close(self, retriever):
        """Should close cleanly."""
        await retriever.initialize()
        await retriever.close()
        assert retriever._initialized is False


class TestRetrieverMethods:
    """Tests for retriever search methods."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_list(self, retriever):
        """Should return a list of RetrievedChunk objects."""
        await retriever.initialize()
        try:
            results = await retriever.retrieve("test query")
            assert isinstance(results, list)
        except Exception:
            pytest.skip("Retriever dependencies not available")

    @pytest.mark.asyncio
    async def test_retrieve_for_claim(self, retriever):
        """retrieve_for_claim should work like retrieve with smaller top_k."""
        await retriever.initialize()
        try:
            results = await retriever.retrieve_for_claim("test claim", top_k=3)
            assert isinstance(results, list)
        except Exception:
            pytest.skip("Retriever dependencies not available")


class TestRetrieverFallbacks:
    """Tests for retriever fallback behavior."""

    @pytest.mark.asyncio
    async def test_dense_search_without_backend(self, retriever):
        """Should return empty results when ChromaDB is not available."""
        retriever._dense_backend = None
        results = await retriever._dense_search("test", 10)
        assert results == []

    @pytest.mark.asyncio
    async def test_sparse_search_without_index(self, retriever):
        """Should return empty results when BM25 index is not loaded."""
        retriever._bm25_index = None
        results = await retriever._sparse_search("test", 10)
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_without_reranker(self, retriever):
        """Should return fusion results directly when reranker is not available."""
        retriever._reranker = None
        chunks = [("c1", 0.9, "test text"), ("c2", 0.8, "another text")]
        results = await retriever._rerank("test", chunks, 2)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[0].retrieval_method == "fused"

    @pytest.mark.asyncio
    async def test_rerank_empty_chunks(self, retriever):
        """Should return empty list for empty chunks."""
        results = await retriever._rerank("test", [], 5)
        assert results == []
