"""Corrective agent for FaithForge.

When the verifier flags a claim as unfaithful, this agent:
1. Re-retrieves evidence targeted at only the flagged claim
2. Rewrites only that claim (not the full answer)

This is the targeted-correction approach — preserving good claims while
fixing only the bad ones. Bound by a configurable max iteration count.
"""

from app.core.logging import get_logger
from app.models.schemas import (
    Claim,
    ClaimVerification,
    CorrectionRecord,
    EntailmentLabel,
    RetrievedChunk,
)
from app.services.llm_adapter import get_llm
from app.services.retriever import HybridRetriever

logger = get_logger("faithforge.corrector")

CORRECTION_SYSTEM_PROMPT = """You are a claim correction specialist.

Given a claim that failed faithfulness verification, and new evidence retrieved
specifically for that claim, rewrite the claim to be faithful to the evidence.

Rules:
- Only change what's needed to make the claim faithful to the new evidence.
- If the new evidence still doesn't support any version of the claim, output
  "INSUFFICIENT_EVIDENCE" as the corrected claim.
- Keep the claim atomic and verifiable.
- Output JSON: {"corrected_claim": "...", "reasoning": "..."}"""


class CorrectiveAgent:
    """Handles targeted claim correction when verification fails.

    For each failed claim:
    1. Re-retrieves evidence specifically for that claim
    2. Asks the LLM to rewrite the claim based on new evidence
    3. Returns the correction record for logging
    """

    def __init__(self, retriever: HybridRetriever):
        self._retriever = retriever

    async def correct_claims(
        self,
        failed_verifications: list[ClaimVerification],
        original_claims: list[Claim],
        iteration: int,
    ) -> tuple[list[CorrectionRecord], list[Claim]]:
        """Correct a batch of failed claims.

        Args:
            failed_verifications: Verifications that returned FAILED status.
            original_claims: The original claims (to look up claim text).
            iteration: Current iteration number (for logging).

        Returns:
            Tuple of (correction_records, corrected_claims).
            corrected_claims includes both passed claims and newly corrected ones.
        """
        llm = get_llm()
        correction_records = []
        corrected_claims = []

        # Build a lookup from claim_id to original claim
        claim_map = {c.claim_id: c for c in original_claims}

        for verification in failed_verifications:
            claim = claim_map.get(verification.claim_id)
            if not claim:
                logger.warning("Claim %s not found in original claims", verification.claim_id)
                continue

            logger.info("Correcting claim %s: '%s...'", claim.claim_id, claim.text[:80])

            # Step 1: Re-retrieve evidence for this specific claim
            new_chunks = await self._retriever.retrieve_for_claim(claim.text, top_k=5)

            # Step 2: Ask LLM to rewrite the claim
            evidence_block = "\n\n".join(
                f"[{c.chunk_id}] {c.text}" for c in new_chunks
            )

            user_msg = f"""## Failed Claim
{claim.text}

## Verification Result
Label: {verification.entailment_label.value}
Score: {verification.faithfulness_score:.3f}

## New Evidence (retrieved specifically for this claim)
{evidence_block}

Rewrite the claim to be faithful to the new evidence. Output JSON."""

            data = await llm.chat_json(CORRECTION_SYSTEM_PROMPT, user_msg)
            corrected_text = data.get("corrected_claim", "")

            # Build the correction record
            record = CorrectionRecord(
                claim_id=claim.claim_id,
                original_claim=claim.text,
                corrected_claim=corrected_text if corrected_text != "INSUFFICIENT_EVIDENCE" else None,
                new_evidence=new_chunks,
                iteration=iteration,
            )
            correction_records.append(record)

            if corrected_text and corrected_text != "INSUFFICIENT_EVIDENCE":
                corrected_claims.append(Claim(
                    claim_id=claim.claim_id,
                    text=corrected_text,
                    source_chunk_ids=[c.chunk_id for c in new_chunks],
                ))
                logger.info("Corrected claim %s: '%s...'", claim.claim_id, corrected_text[:80])
            else:
                logger.warning("Claim %s: insufficient evidence for correction", claim.claim_id)

        return correction_records, corrected_claims
