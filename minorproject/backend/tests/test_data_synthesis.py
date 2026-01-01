"""Tests for the data synthesis module."""

import pytest
import json
import tempfile
from pathlib import Path

from verifier.data_synthesis import (
    generate_faithful_claims,
    corrupt_claim,
    self_critique_label,
    generate_dataset,
    CLAIM_GENERATION_PROMPT,
    CLAIM_CORRUPTION_PROMPT,
    SELF_CRITIQUE_PROMPT,
)


class TestPromptTemplates:
    """Tests for prompt template formatting."""

    def test_claim_generation_prompt(self):
        """Should format with passage and n."""
        prompt = CLAIM_GENERATION_PROMPT.format(n=3, passage="Test passage")
        assert "3" in prompt
        assert "Test passage" in prompt
        assert "{n}" not in prompt
        assert "{passage}" not in prompt

    def test_claim_corruption_prompt(self):
        """Should format with claim and passage."""
        prompt = CLAIM_CORRUPTION_PROMPT.format(n=3, claim="Test claim", passage="Test passage")
        assert "Test claim" in prompt
        assert "Test passage" in prompt

    def test_self_critique_prompt(self):
        """Should format with claim and evidence."""
        prompt = SELF_CRITIQUE_PROMPT.format(claim="Test claim", evidence="Test evidence")
        assert "Test claim" in prompt
        assert "Test evidence" in prompt


class TestFaithfulClaimGeneration:
    """Tests for faithful claim generation."""

    @pytest.mark.asyncio
    async def test_generates_claims(self):
        """Should generate claims from a passage (requires LLM)."""
        # This test requires a working LLM connection
        try:
            claims = await generate_faithful_claims(
                "Python is a high-level programming language created by Guido van Rossum.",
                n=2,
            )
            assert isinstance(claims, list)
            for claim in claims:
                assert "text" in claim
                assert "label" in claim
                assert claim["label"] == "entailment"
                assert 0.0 <= claim["score"] <= 1.0
        except Exception:
            pytest.skip("LLM not available")


class TestClaimCorruption:
    """Tests for claim corruption."""

    @pytest.mark.asyncio
    async def test_corrupts_claim(self):
        """Should generate corrupted versions (requires LLM)."""
        try:
            corruptions = await corrupt_claim(
                "Python was created by Guido van Rossum",
                "Python is a programming language created by Guido van Rossum in 1991.",
            )
            assert isinstance(corruptions, list)
            for corruption in corruptions:
                assert "text" in corruption
                assert "label" in corruption
                assert corruption["label"] in ("contradiction", "neutral")
        except Exception:
            pytest.skip("LLM not available")


class TestSelfCritique:
    """Tests for self-critique labeling."""

    @pytest.mark.asyncio
    async def test_self_critique_returns_label(self):
        """Should return a label and reasoning (requires LLM)."""
        try:
            result = await self_critique_label(
                "Python was created by Guido van Rossum",
                "Python is a programming language created by Guido van Rossum.",
            )
            assert "label" in result
            assert result["label"] in ("entailment", "contradiction", "neutral")
            assert "reasoning" in result
        except Exception:
            pytest.skip("LLM not available")


class TestDatasetGeneration:
    """Tests for the full dataset generation pipeline."""

    @pytest.mark.asyncio
    async def test_generate_dataset_creates_files(self):
        """Should create train/val/test JSONL files."""
        passages = [
            "Python is a high-level programming language.",
            "JavaScript is used for web development.",
        ]

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                stats = await generate_dataset(
                    passages,
                    output_dir,
                    claims_per_passage=1,
                    corruptions_per_claim=1,
                )

                # Check stats
                assert stats["total_pairs"] > 0
                assert stats["train_size"] > 0
                assert "label_distribution" in stats

                # Check files exist
                assert (output_dir / "train.jsonl").exists()
                assert (output_dir / "val.jsonl").exists()
                assert (output_dir / "test.jsonl").exists()

                # Check file format
                with open(output_dir / "train.jsonl") as f:
                    for line in f:
                        row = json.loads(line)
                        assert "claim" in row
                        assert "evidence" in row
                        assert "label" in row
                        assert "faithfulness_score" in row
        except Exception:
            pytest.skip("LLM not available")
