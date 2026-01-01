/** FaithForge TypeScript type definitions. */

// ── Enums ────────────────────────────────────────────────────────────────────

export type RetrievalStrategy = "none" | "single_hop" | "multi_hop";
export type EntailmentLabel = "entailment" | "contradiction" | "neutral";
export type ClaimStatus = "verified" | "failed" | "corrected";

// ── Pipeline Stage Models ────────────────────────────────────────────────────

export interface RetrievedChunk {
  chunk_id: string;
  source: string;
  text: string;
  score: number;
  retrieval_method: string;
}

export interface Claim {
  claim_id: string;
  text: string;
  source_chunk_ids: string[];
}

export interface ClaimVerification {
  claim_id: string;
  claim_text: string;
  evidence_text: string;
  entailment_label: EntailmentLabel;
  faithfulness_score: number;
  status: ClaimStatus;
  iteration: number;
}

export interface CorrectionRecord {
  claim_id: string;
  original_claim: string;
  corrected_claim: string | null;
  new_evidence: RetrievedChunk[];
  iteration: number;
}

export interface PipelineTrace {
  query: string;
  strategy: RetrievalStrategy;
  sub_queries: string[];
  retrieved_chunks: RetrievedChunk[];
  generated_answer: string;
  claims: Claim[];
  verifications: ClaimVerification[];
  corrections: CorrectionRecord[];
  total_iterations: number;
  all_claims_faithful: boolean;
}

// ── API Response Models ──────────────────────────────────────────────────────

export interface QueryResponse {
  trace: PipelineTrace;
  final_answer: string;
  latency_ms: number;
}

export interface EvaluationMetrics {
  total_queries: number;
  faithfulness_accuracy: number;
  claim_level_precision: number;
  claim_level_recall: number;
  avg_iterations: number;
  avg_latency_ms: number;
}

export interface AblationResult {
  ablation_name: string;
  description: string;
  baseline_metric: number;
  ablated_metric: number;
  improvement_pct: number;
}
