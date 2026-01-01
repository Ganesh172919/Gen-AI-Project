"""Tests for Pydantic schema models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.schemas import (
    AblationResult,
    Claim,
    ClaimStatus,
    ClaimVerification,
    CorrectionRecord,
    EntailmentLabel,
    EvaluationMetrics,
    EvaluationRequest,
    EvaluationResponse,
    PipelineTrace,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    RetrievalStrategy,
)


class TestQueryRequest:
    def test_valid_request(self):
        req = QueryRequest(query="What is RAG?")
        assert req.query == "What is RAG?"
        assert req.top_k is None
        assert req.max_iterations is None

    def test_with_options(self):
        req = QueryRequest(query="test", top_k=5, max_iterations=2)
        assert req.top_k == 5
        assert req.max_iterations == 2

    def test_rejects_empty_query(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_rejects_long_query(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="x" * 2001)

    def test_rejects_invalid_top_k(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=51)

    def test_rejects_invalid_max_iterations(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", max_iterations=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", max_iterations=6)


class TestEvaluationRequest:
    def test_valid_request(self):
        req = EvaluationRequest(dataset_name="ragtruth")
        assert req.dataset_name == "ragtruth"
        assert req.sample_size is None
        assert req.run_ablations is True

    def test_with_options(self):
        req = EvaluationRequest(
            dataset_name="custom",
            sample_size=100,
            run_ablations=False,
        )
        assert req.sample_size == 100
        assert req.run_ablations is False

    def test_rejects_invalid_sample_size(self):
        with pytest.raises(ValidationError):
            EvaluationRequest(dataset_name="test", sample_size=0)
        with pytest.raises(ValidationError):
            EvaluationRequest(dataset_name="test", sample_size=1001)


class TestRetrievedChunk:
    def test_valid_chunk(self):
        chunk = RetrievedChunk(
            chunk_id="c1",
            source="wiki",
            text="Some text",
            score=0.95,
            retrieval_method="dense",
        )
        assert chunk.chunk_id == "c1"
        assert chunk.score == 0.95

    def test_model_dump(self):
        chunk = RetrievedChunk(
            chunk_id="c1", source="wiki", text="text",
            score=0.9, retrieval_method="dense",
        )
        d = chunk.model_dump()
        assert d["chunk_id"] == "c1"
        assert d["score"] == 0.9


class TestClaim:
    def test_valid_claim(self):
        claim = Claim(
            claim_id="c1",
            text="Python was created in 1991",
            source_chunk_ids=["chunk_001"],
        )
        assert claim.claim_id == "c1"
        assert len(claim.source_chunk_ids) == 1

    def test_claim_without_sources(self):
        claim = Claim(claim_id="c1", text="test")
        assert claim.source_chunk_ids == []

    def test_claim_with_multiple_sources(self):
        claim = Claim(
            claim_id="c1", text="test",
            source_chunk_ids=["c1", "c2", "c3"],
        )
        assert len(claim.source_chunk_ids) == 3


class TestClaimVerification:
    def test_verified_claim(self):
        v = ClaimVerification(
            claim_id="c1",
            claim_text="test",
            evidence_text="evidence",
            entailment_label=EntailmentLabel.ENTAILMENT,
            faithfulness_score=0.95,
            status=ClaimStatus.VERIFIED,
        )
        assert v.status == ClaimStatus.VERIFIED

    def test_failed_claim(self):
        v = ClaimVerification(
            claim_id="c1",
            claim_text="test",
            evidence_text="evidence",
            entailment_label=EntailmentLabel.CONTRADICTION,
            faithfulness_score=0.1,
            status=ClaimStatus.FAILED,
        )
        assert v.status == ClaimStatus.FAILED

    def test_default_iteration(self):
        v = ClaimVerification(
            claim_id="c1", claim_text="test", evidence_text="evidence",
            entailment_label=EntailmentLabel.ENTAILMENT,
            faithfulness_score=0.9, status=ClaimStatus.VERIFIED,
        )
        assert v.iteration == 1

    def test_custom_iteration(self):
        v = ClaimVerification(
            claim_id="c1", claim_text="test", evidence_text="evidence",
            entailment_label=EntailmentLabel.ENTAILMENT,
            faithfulness_score=0.9, status=ClaimStatus.VERIFIED,
            iteration=3,
        )
        assert v.iteration == 3


class TestCorrectionRecord:
    def test_valid_record(self):
        r = CorrectionRecord(
            claim_id="c1",
            original_claim="wrong claim",
            corrected_claim="correct claim",
            iteration=1,
        )
        assert r.claim_id == "c1"
        assert r.corrected_claim == "correct claim"

    def test_insufficient_evidence(self):
        r = CorrectionRecord(
            claim_id="c1",
            original_claim="wrong claim",
            corrected_claim=None,
            iteration=1,
        )
        assert r.corrected_claim is None


class TestEnums:
    def test_retrieval_strategy_values(self):
        assert RetrievalStrategy.NONE.value == "none"
        assert RetrievalStrategy.SINGLE_HOP.value == "single_hop"
        assert RetrievalStrategy.MULTI_HOP.value == "multi_hop"

    def test_entailment_labels(self):
        assert EntailmentLabel.ENTAILMENT.value == "entailment"
        assert EntailmentLabel.CONTRADICTION.value == "contradiction"
        assert EntailmentLabel.NEUTRAL.value == "neutral"

    def test_claim_statuses(self):
        assert ClaimStatus.VERIFIED.value == "verified"
        assert ClaimStatus.FAILED.value == "failed"
        assert ClaimStatus.CORRECTED.value == "corrected"


class TestPipelineTrace:
    def test_minimal_trace(self):
        trace = PipelineTrace(
            query="test",
            strategy=RetrievalStrategy.SINGLE_HOP,
        )
        assert trace.query == "test"
        assert trace.all_claims_faithful is False
        assert trace.total_iterations == 1

    def test_full_trace(self):
        trace = PipelineTrace(
            query="What is Python?",
            strategy=RetrievalStrategy.SINGLE_HOP,
            sub_queries=["What is Python?"],
            retrieved_chunks=[
                RetrievedChunk(
                    chunk_id="c1", source="wiki", text="Python is a language",
                    score=0.9, retrieval_method="dense",
                )
            ],
            generated_answer="Python is a programming language.",
            claims=[
                Claim(claim_id="c1", text="Python is a programming language", source_chunk_ids=["c1"])
            ],
            verifications=[
                ClaimVerification(
                    claim_id="c1", claim_text="Python is a programming language",
                    evidence_text="Python is a language",
                    entailment_label=EntailmentLabel.ENTAILMENT,
                    faithfulness_score=0.9, status=ClaimStatus.VERIFIED,
                )
            ],
            all_claims_faithful=True,
        )
        assert len(trace.claims) == 1
        assert trace.all_claims_faithful is True

    def test_trace_with_corrections(self):
        trace = PipelineTrace(
            query="test",
            strategy=RetrievalStrategy.SINGLE_HOP,
            corrections=[
                CorrectionRecord(
                    claim_id="c1",
                    original_claim="wrong",
                    corrected_claim="right",
                    iteration=1,
                )
            ],
            total_iterations=2,
        )
        assert len(trace.corrections) == 1
        assert trace.total_iterations == 2


class TestEvaluationModels:
    def test_evaluation_metrics(self):
        metrics = EvaluationMetrics(
            total_queries=100,
            faithfulness_accuracy=0.85,
            claim_level_precision=0.9,
            claim_level_recall=0.8,
            avg_iterations=1.5,
            avg_latency_ms=250.0,
        )
        assert metrics.total_queries == 100
        assert metrics.ragas_faithfulness is None

    def test_evaluation_metrics_with_ragas(self):
        metrics = EvaluationMetrics(
            total_queries=100,
            faithfulness_accuracy=0.85,
            claim_level_precision=0.9,
            claim_level_recall=0.8,
            avg_iterations=1.5,
            avg_latency_ms=250.0,
            ragas_faithfulness=0.82,
            ragas_answer_relevancy=0.88,
        )
        assert metrics.ragas_faithfulness == 0.82

    def test_ablation_result(self):
        result = AblationResult(
            ablation_name="verifier_vs_self_critique",
            description="Test ablation",
            baseline_metric=0.7,
            ablated_metric=0.85,
            improvement_pct=21.4,
        )
        assert result.improvement_pct == pytest.approx(21.4)

    def test_evaluation_response(self):
        resp = EvaluationResponse(
            metrics=EvaluationMetrics(
                total_queries=10,
                faithfulness_accuracy=0.9,
                claim_level_precision=0.95,
                claim_level_recall=0.85,
                avg_iterations=1.2,
                avg_latency_ms=200.0,
            ),
            ablations=[],
            run_id="test-123",
            completed_at=datetime.utcnow(),
        )
        assert resp.run_id == "test-123"
        assert len(resp.ablations) == 0
