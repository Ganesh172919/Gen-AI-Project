"""Tests for the LangGraph orchestration graph."""

import pytest
from app.agents.graph import (
    ForgeState,
    should_continue,
    planner_node,
    retriever_node,
    generator_node,
    verifier_node,
    corrector_node,
)
from app.models.schemas import (
    Claim,
    ClaimStatus,
    ClaimVerification,
    EntailmentLabel,
    RetrievalStrategy,
)


class TestShouldContinue:
    """Tests for the should_continue conditional edge function."""

    def test_all_verified_ends(self):
        """Should return 'end' when all claims are verified."""
        state = ForgeState({
            "verifications": [
                ClaimVerification(
                    claim_id="c1", claim_text="test", evidence_text="evidence",
                    entailment_label=EntailmentLabel.ENTAILMENT,
                    faithfulness_score=0.95, status=ClaimStatus.VERIFIED,
                ),
            ],
            "iteration": 1,
            "max_iterations": 3,
        })
        assert should_continue(state) == "end"

    def test_failed_claims_continues(self):
        """Should return 'corrector' when claims failed and under max iterations."""
        state = ForgeState({
            "verifications": [
                ClaimVerification(
                    claim_id="c1", claim_text="test", evidence_text="evidence",
                    entailment_label=EntailmentLabel.CONTRADICTION,
                    faithfulness_score=0.1, status=ClaimStatus.FAILED,
                ),
            ],
            "iteration": 1,
            "max_iterations": 3,
        })
        assert should_continue(state) == "corrector"

    def test_max_iterations_ends(self):
        """Should return 'end' when max iterations reached."""
        state = ForgeState({
            "verifications": [
                ClaimVerification(
                    claim_id="c1", claim_text="test", evidence_text="evidence",
                    entailment_label=EntailmentLabel.CONTRADICTION,
                    faithfulness_score=0.1, status=ClaimStatus.FAILED,
                ),
            ],
            "iteration": 3,
            "max_iterations": 3,
        })
        assert should_continue(state) == "end"

    def test_empty_verifications_ends(self):
        """Should return 'end' when no verifications exist."""
        state = ForgeState({
            "verifications": [],
            "iteration": 1,
            "max_iterations": 3,
        })
        assert should_continue(state) == "end"

    def test_mixed_results_continues(self):
        """Should continue when mix of verified and failed claims."""
        state = ForgeState({
            "verifications": [
                ClaimVerification(
                    claim_id="c1", claim_text="test1", evidence_text="evidence",
                    entailment_label=EntailmentLabel.ENTAILMENT,
                    faithfulness_score=0.95, status=ClaimStatus.VERIFIED,
                ),
                ClaimVerification(
                    claim_id="c2", claim_text="test2", evidence_text="evidence",
                    entailment_label=EntailmentLabel.CONTRADICTION,
                    faithfulness_score=0.1, status=ClaimStatus.FAILED,
                ),
            ],
            "iteration": 1,
            "max_iterations": 3,
        })
        assert should_continue(state) == "corrector"


class TestForgeState:
    """Tests for the ForgeState dict."""

    def test_initial_state(self):
        """Should create state with all required keys."""
        state = ForgeState({
            "query": "test query",
            "strategy": None,
            "sub_queries": [],
            "retrieved_chunks": [],
            "answer": "",
            "claims": [],
            "verifications": [],
            "corrections": [],
            "iteration": 1,
            "max_iterations": 3,
        })
        assert state["query"] == "test query"
        assert state["iteration"] == 1
        assert state["max_iterations"] == 3

    def test_state_update(self):
        """Should allow updating state values."""
        state = ForgeState({"iteration": 1})
        state["iteration"] = 2
        assert state["iteration"] == 2
