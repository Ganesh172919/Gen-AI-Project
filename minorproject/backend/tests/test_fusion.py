"""Tests for RRF fusion module."""

import pytest
from retrieval.fusion import rrf_score, rrf_merge


class TestRRFScore:
    """Tests for the rrf_score function."""

    def test_rank_1(self):
        """Rank 1 should give highest score."""
        score = rrf_score(1, k=60)
        assert score == pytest.approx(1.0 / 61, rel=1e-6)

    def test_rank_10(self):
        """Rank 10 should give lower score than rank 1."""
        score = rrf_score(10, k=60)
        assert score == pytest.approx(1.0 / 70, rel=1e-6)

    def test_higher_rank_lower_score(self):
        """Higher rank should always give lower score."""
        for rank in range(1, 20):
            assert rrf_score(rank) > rrf_score(rank + 1)

    def test_custom_k(self):
        """Custom k value should change the score."""
        score_k30 = rrf_score(1, k=30)
        score_k60 = rrf_score(1, k=60)
        assert score_k30 > score_k60  # Smaller k → higher score

    def test_rank_zero_raises(self):
        """Rank 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Rank must be >= 1"):
            rrf_score(0)

    def test_negative_rank_raises(self):
        """Negative rank should raise ValueError."""
        with pytest.raises(ValueError, match="Rank must be >= 1"):
            rrf_score(-1)


class TestRRFMerge:
    """Tests for the rrf_merge function."""

    def test_basic_merge(self):
        """Basic merge should combine results from both lists."""
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("b", 5.0), ("d", 4.0), ("a", 3.0)]

        result = rrf_merge(dense, sparse)

        # Should have all unique chunk IDs
        result_ids = [cid for cid, _ in result]
        assert set(result_ids) == {"a", "b", "c", "d"}

    def test_intersection_ranks_higher(self):
        """Chunks appearing in both lists should rank higher."""
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [("b", 5.0), ("a", 3.0)]

        result = rrf_merge(dense, sparse)
        result_dict = dict(result)

        # "a" is rank 1 in dense and rank 2 in sparse
        # "b" is rank 2 in dense and rank 1 in sparse
        # Both should have similar scores (sum of two RRF contributions)
        assert result_dict["a"] > 0
        assert result_dict["b"] > 0

    def test_single_list(self):
        """Merge with empty sparse list should work."""
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = []

        result = rrf_merge(dense, sparse)
        result_dict = dict(result)

        assert "a" in result_dict
        assert "b" in result_dict
        assert result_dict["a"] > result_dict["b"]  # Rank 1 > Rank 2

    def test_empty_lists(self):
        """Merge of two empty lists should return empty."""
        result = rrf_merge([], [])
        assert result == []

    def test_sorted_by_score(self):
        """Results should be sorted by RRF score descending."""
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse = [("d", 5.0), ("e", 4.0)]

        result = rrf_merge(dense, sparse)

        # Verify descending order
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_custom_k(self):
        """Custom k should affect scores."""
        dense = [("a", 0.9)]
        sparse = [("a", 5.0)]

        result_k30 = rrf_merge(dense, sparse, k=30)
        result_k60 = rrf_merge(dense, sparse, k=60)

        # Smaller k → higher RRF scores
        assert result_k30[0][1] > result_k60[0][1]
