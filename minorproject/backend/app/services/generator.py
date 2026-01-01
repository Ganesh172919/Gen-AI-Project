"""Grounded answer generator for FaithForge.

Prompts the LLM to produce structured JSON output: a list of claims,
each tagged with the source-document IDs it relies on. This structure
is what makes claim-level verification tractable downstream.
"""

import json
from typing import Optional

from app.core.logging import get_logger
from app.models.schemas import Claim, RetrievedChunk
from app.services.llm_adapter import get_llm

logger = get_logger("faithforge.generator")

SYSTEM_PROMPT = """You are a precise, evidence-grounded answer generator.

Given a user question and a set of retrieved evidence chunks, produce a JSON response with:
1. "answer": A clear, concise answer to the question.
2. "claims": A list of atomic claims in the answer. Each claim must be a single,
   verifiable statement. For each claim, include:
   - "claim_id": A short identifier (e.g., "c1", "c2")
   - "text": The claim text
   - "source_chunk_ids": List of chunk IDs from the evidence that support this claim

RULES:
- Every claim MUST be traceable to at least one evidence chunk.
- Do NOT include information not present in the evidence chunks.
- If the evidence doesn't contain enough information to answer, say so explicitly.
- Keep claims atomic — each should be independently verifiable.
- Output valid JSON only."""


class GroundedGenerator:
    """Generates evidence-grounded answers with claim tagging.

    Uses the unified LLM adapter so the generator model is swappable
    across Groq/Cerebras/OpenRouter without code changes.
    """

    async def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        model: Optional[str] = None,
    ) -> tuple[str, list[Claim]]:
        """Generate an answer and extract claims.

        Args:
            query: The user's question.
            chunks: Retrieved evidence chunks (ranked).
            model: Optional model override.

        Returns:
            Tuple of (answer_text, list_of_claims).

        Raises:
            ValueError: If the LLM response can't be parsed.
        """
        llm = get_llm()

        # Build evidence block
        evidence_lines = []
        for i, chunk in enumerate(chunks):
            evidence_lines.append(
                f"[{chunk.chunk_id}] (source: {chunk.source}, score: {chunk.score:.3f})\n{chunk.text}"
            )
        evidence_block = "\n\n---\n\n".join(evidence_lines)

        user_message = f"""## Question
{query}

## Retrieved Evidence
{evidence_block}

Produce your JSON response now."""

        logger.info("Generating answer: query_len=%d, chunks=%d", len(query), len(chunks))

        raw = await llm.chat(
            SYSTEM_PROMPT,
            user_message,
            model=model,
        )

        # Parse the structured response
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract JSON from markdown code block
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
                data = json.loads(json_str)
            else:
                raise ValueError(f"Could not parse LLM response as JSON: {raw[:200]}")

        answer = data.get("answer", "")
        raw_claims = data.get("claims", [])

        claims = [
            Claim(
                claim_id=c.get("claim_id", f"c{i+1}"),
                text=c.get("text", ""),
                source_chunk_ids=c.get("source_chunk_ids", []),
            )
            for i, c in enumerate(raw_claims)
        ]

        logger.info("Generated answer with %d claims", len(claims))
        return answer, claims


# ── Module-level singleton ───────────────────────────────────────────────────

_generator: Optional[GroundedGenerator] = None


def get_generator() -> GroundedGenerator:
    """Get or create the module-level GroundedGenerator singleton."""
    global _generator
    if _generator is None:
        _generator = GroundedGenerator()
    return _generator
