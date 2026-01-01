"""Tests for the faithfulness verifier agent."""

import pytest
from app.agents.verifier import FaithfulnessVerifier
from app.models.schemas import (
    Claim,
    ClaimStatus,
    EntailmentLabel,
    RetrievedChunk,
)


@pytest.fixture
def verifier():
    """Create a verifier instance."""
    return FaithfulnessVerifier()


@pytest.fixture
def sample_claim():
    """Create a sample claim."""
    return Claim(
        claim_id="c1",
        text="Python was created by Guido van Rossum",
        source_chunk_ids=["chunk_1"],
    )


@pytest.fixture
def sample_evidence():
    """Create sample evidence chunks."""
    return [
        RetrievedChunk(
            chunk_id="chunk_1",
            source="wiki",
            text="Python was created by Guido van Rossum and first released in 1991.",
            score=0.95,
            retrieval_method="dense",
        )
    ]


class TestVerifierFallback:
    """Tests for the verifier's fallback (rule-based) mode."""

    @pytest.mark.asyncio
    async def test_load_without_model(self, verifier):
        """Verifier should load successfully even without a fine-tuned model."""
        await verifier.load()
        assert verifier._loaded is True

    @pytest.mark.asyncio
    async def test_verify_claim_basic(self, verifier, sample_claim, sample_evidence):
        """Should verify a claim against evidence."""
        await verifier.load()
        result = await verifier.verify_claim(sample_claim, sample_evidence)

        assert result.claim_id == "c1"
        assert result.entailment_label in EntailmentLabel
        assert 0.0 <= result.faithfulness_score <= 1.0
        assert result.status in ClaimStatus

    @pytest.mark.asyncio
    async def test_verify_claim_no_evidence(self, verifier, sample_claim):
        """Should handle claims with no evidence."""
        await verifier.load()
        result = await verifier.verify_claim(sample_claim, [])

        # No evidence should result in neutral/failed
        assert result.status == ClaimStatus.FAILED

    @pytest.mark.asyncio
    async def test_verify_batch(self, verifier):
        """Should verify multiple claims."""
        await verifier.load()

        claims = [
            Claim(claim_id="c1", text="Python is a programming language", source_chunk_ids=["chunk_1"]),
            Claim(claim_id="c2", text="Python was created in 1991", source_chunk_ids=["chunk_1"]),
        ]

        chunk_map = {
            "chunk_1": [
                RetrievedChunk(
                    chunk_id="chunk_1",
                    source="wiki",
                    text="Python is a high-level programming language created by Guido van Rossum in 1991.",
                    score=0.9,
                    retrieval_method="dense",
                )
            ]
        }

        results = await verifier.verify_batch(claims, chunk_map)

        assert len(results) == 2
        for result in results:
            assert result.claim_id in ("c1", "c2")
            assert result.status in ClaimStatus

    @pytest.mark.asyncio
    async def test_verify_batch_missing_chunk(self, verifier):
        """Should handle claims with missing chunk IDs."""
        await verifier.load()

        claims = [
            Claim(claim_id="c1", text="Test claim", source_chunk_ids=["nonexistent"]),
        ]

        results = await verifier.verify_batch(claims, {})

        assert len(results) == 1
        assert results[0].status == ClaimStatus.FAILED


class TestFallbackHeuristic:
    """Tests for the rule-based fallback verification."""

    @pytest.mark.asyncio
    async def test_high_overlap_entailment(self, verifier):
        """High word overlap should suggest entailment."""
        await verifier.load()

        claim = Claim(claim_id="c1", text="Python is a programming language", source_chunk_ids=["c1"])
        evidence = [
            RetrievedChunk(
                chunk_id="c1", source="wiki",
                text="Python is a popular programming language used worldwide",
                score=0.9, retrieval_method="dense",
            )
        ]

        result = await verifier.verify_claim(claim, evidence)
        # Should have reasonable overlap
        assert result.faithfulness_score > 0.0

    @pytest.mark.asyncio
    async def test_unload(self, verifier):
        """Unload should free resources."""
        await verifier.load()
        await verifier.unload()
        assert verifier._loaded is False


class TestVerifierOutputParsing:
    """Tests for parsing verifier model output."""

    def test_parse_entailment(self, verifier):
        """Should parse entailment label and score."""
        label, score = verifier._parse_verifier_output("entailment\nScore: 0.95")
        assert label == EntailmentLabel.ENTAILMENT
        assert score == pytest.approx(0.95)

    def test_parse_contradiction(self, verifier):
        """Should parse contradiction label."""
        label, score = verifier._parse_verifier_output("contradiction\nScore: 0.10")
        assert label == EntailmentLabel.CONTRADICTION
        assert score == pytest.approx(0.10)

    def test_parse_neutral(self, verifier):
        """Should parse neutral label."""
        label, score = verifier._parse_verifier_output("neutral\nScore: 0.50")
        assert label == EntailmentLabel.NEUTRAL
        assert score == pytest.approx(0.50)

    def test_parse_without_score(self, verifier):
        """Should use default score when not provided."""
        label, score = verifier._parse_verifier_output("entailment")
        assert label == EntailmentLabel.ENTAILMENT
        assert score == pytest.approx(0.85)  # Default for entailment

    def test_parse_case_insensitive(self, verifier):
        """Should handle mixed case."""
        label, score = verifier._parse_verifier_output("ENTAILMENT\nScore: 0.90")
        assert label == EntailmentLabel.ENTAILMENT
