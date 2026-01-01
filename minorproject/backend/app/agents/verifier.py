"""Faithfulness verifier agent for FaithForge.

This is the core research component — a fine-tuned model that classifies
claim-vs-evidence entailment and produces a faithfulness score.

The fine-tuning code lives in verifier/train.py.
This module loads the trained model and runs inference.

Model: QLoRA fine-tuned Qwen2.5-1.5B-Instruct (or similar 1-3B model)
Task: NLI-style classification — (claim, evidence) → (label, score)
"""

import re
from typing import Optional

import torch

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    Claim,
    ClaimStatus,
    ClaimVerification,
    EntailmentLabel,
    RetrievedChunk,
)

logger = get_logger("faithforge.verifier")

# Input format for the verifier
INPUT_TEMPLATE = "Claim: {claim}\nEvidence: {evidence}"


class FaithfulnessVerifier:
    """Loads the fine-tuned verifier model and runs claim-level verification.

    For each (claim, evidence) pair, the verifier outputs:
    - entailment_label: ENTAILMENT | CONTRADICTION | NEUTRAL
    - faithfulness_score: 0.0 (unfaithful) to 1.0 (fully faithful)

    A claim passes verification if:
    - label == ENTAILMENT and score >= verifier_confidence_threshold
    - label == NEUTRAL is treated as a soft fail (needs correction)
    - label == CONTRADICTION is a hard fail
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._loaded = False

    async def load(self) -> None:
        """Load the fine-tuned verifier model from disk.

        Uses QLoRA/PEFT — loads the base model in 4-bit, then applies
        the LoRA adapter weights.

        Model path is configured via settings.verifier_model_path.
        Base model is configured via settings.verifier_base_model.
        """
        from pathlib import Path

        model_path = Path(settings.verifier_model_path)

        logger.info(
            "Loading verifier model: base=%s, adapter=%s",
            settings.verifier_base_model,
            model_path,
        )

        # Check if fine-tuned adapter exists
        if not model_path.exists():
            logger.warning(
                "Fine-tuned adapter not found at %s — using fallback mode. "
                "Run verifier training first or use self-critique mode.",
                model_path,
            )
            self._loaded = True
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load base model in 4-bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            base_model = AutoModelForCausalLM.from_pretrained(
                settings.verifier_base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Apply LoRA adapter
            self._model = PeftModel.from_pretrained(base_model, str(model_path))
            self._model.eval()

            # Determine device
            self._device = next(self._model.parameters()).device
            self._loaded = True

            logger.info("Verifier model loaded on device: %s", self._device)

        except Exception as e:
            logger.error("Failed to load verifier model: %s", e)
            logger.info("Falling back to rule-based verification")
            self._loaded = True  # Allow fallback mode

    async def verify_claim(
        self,
        claim: Claim,
        evidence_chunks: list[RetrievedChunk],
    ) -> ClaimVerification:
        """Verify a single claim against its evidence.

        If the fine-tuned model is available, uses it for inference.
        Otherwise, falls back to a rule-based heuristic.

        Args:
            claim: The claim to verify.
            evidence_chunks: The evidence chunks the claim cites.

        Returns:
            ClaimVerification with entailment label, score, and status.
        """
        if not self._loaded:
            raise RuntimeError("Verifier not loaded — call load() first")

        # Concatenate evidence for the verifier input
        evidence_text = "\n\n".join(chunk.text for chunk in evidence_chunks)

        logger.debug("Verifying claim %s: '%s...'", claim.claim_id, claim.text[:80])

        # Try fine-tuned model first
        if self._model is not None and self._tokenizer is not None:
            label, score = await self._model_predict(claim.text, evidence_text)
        else:
            # Fallback: rule-based heuristic
            label, score = self._fallback_verify(claim.text, evidence_text)

        # Determine pass/fail status
        if label == EntailmentLabel.ENTAILMENT and score >= settings.verifier_confidence_threshold:
            status = ClaimStatus.VERIFIED
        elif label == EntailmentLabel.CONTRADICTION:
            status = ClaimStatus.FAILED
        else:  # NEUTRAL or low-confidence entailment
            status = ClaimStatus.FAILED

        result = ClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            evidence_text=evidence_text[:500],  # Truncate for storage
            entailment_label=label,
            faithfulness_score=score,
            status=status,
            iteration=1,
        )

        logger.info(
            "Claim %s: label=%s, score=%.3f, status=%s",
            claim.claim_id, label.value, score, status.value,
        )
        return result

    async def _model_predict(self, claim: str, evidence: str) -> tuple[EntailmentLabel, float]:
        """Run inference through the fine-tuned model.

        Args:
            claim: The claim text.
            evidence: The evidence text.

        Returns:
            Tuple of (entailment_label, faithfulness_score).
        """
        input_text = INPUT_TEMPLATE.format(
            claim=claim,
            evidence=evidence[:1000],
        )

        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only the new tokens
        generated = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Parse the output
        return self._parse_verifier_output(generated)

    def _parse_verifier_output(self, output: str) -> tuple[EntailmentLabel, float]:
        """Parse the verifier's output into label and score.

        Expected format: "{label}\nScore: {score}"

        Args:
            output: The raw model output string.

        Returns:
            Tuple of (entailment_label, faithfulness_score).
        """
        output_lower = output.lower()

        # Parse label
        if "entailment" in output_lower:
            label = EntailmentLabel.ENTAILMENT
        elif "contradiction" in output_lower:
            label = EntailmentLabel.CONTRADICTION
        else:
            label = EntailmentLabel.NEUTRAL

        # Parse score
        score_match = re.search(r"score:\s*([\d.]+)", output_lower)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        else:
            # Default scores based on label
            score_defaults = {
                EntailmentLabel.ENTAILMENT: 0.85,
                EntailmentLabel.CONTRADICTION: 0.15,
                EntailmentLabel.NEUTRAL: 0.5,
            }
            score = score_defaults[label]

        return label, score

    def _fallback_verify(self, claim: str, evidence: str) -> tuple[EntailmentLabel, float]:
        """Rule-based fallback verification when no model is available.

        Uses simple heuristics:
        - Word overlap ratio between claim and evidence
        - Negation detection
        - Length ratio

        Args:
            claim: The claim text.
            evidence: The evidence text.

        Returns:
            Tuple of (entailment_label, faithfulness_score).
        """
        claim_words = set(claim.lower().split())
        evidence_words = set(evidence.lower().split())

        # Word overlap
        if not claim_words:
            return EntailmentLabel.NEUTRAL, 0.5

        overlap = len(claim_words & evidence_words) / len(claim_words)

        # Negation detection
        negation_words = {"not", "no", "never", "neither", "nobody", "nothing", "nowhere", "nor", "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't"}
        claim_has_negation = bool(claim_words & negation_words)
        evidence_has_negation = bool(evidence_words & negation_words)

        # If one has negation and the other doesn't, likely contradiction
        if claim_has_negation != evidence_has_negation:
            return EntailmentLabel.CONTRADICTION, 0.2

        # High overlap → entailment
        if overlap > 0.6:
            return EntailmentLabel.ENTAILMENT, min(0.95, 0.5 + overlap * 0.5)

        # Medium overlap → neutral
        if overlap > 0.3:
            return EntailmentLabel.NEUTRAL, 0.4 + overlap * 0.3

        # Low overlap → neutral (could be unrelated)
        return EntailmentLabel.NEUTRAL, 0.3

    async def verify_batch(
        self,
        claims: list[Claim],
        chunk_map: dict[str, list[RetrievedChunk]],
    ) -> list[ClaimVerification]:
        """Verify multiple claims.

        Args:
            claims: List of claims to verify.
            chunk_map: Mapping from chunk_id → RetrievedChunk for lookup.

        Returns:
            List of ClaimVerification results.
        """
        results = []
        for claim in claims:
            # Look up the evidence chunks this claim cites
            evidence = []
            for cid in claim.source_chunk_ids:
                if cid in chunk_map:
                    evidence.extend(chunk_map[cid])

            if not evidence:
                # No evidence found — mark as failed
                logger.warning("Claim %s: no evidence found for chunk IDs %s", claim.claim_id, claim.source_chunk_ids)
                results.append(ClaimVerification(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    evidence_text="",
                    entailment_label=EntailmentLabel.NEUTRAL,
                    faithfulness_score=0.0,
                    status=ClaimStatus.FAILED,
                    iteration=1,
                ))
            else:
                results.append(await self.verify_claim(claim, evidence))

        verified = sum(1 for r in results if r.status == ClaimStatus.VERIFIED)
        logger.info("Batch verification: %d/%d claims verified", verified, len(results))

        return results

    async def unload(self) -> None:
        """Free model memory."""
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Verifier model unloaded")
