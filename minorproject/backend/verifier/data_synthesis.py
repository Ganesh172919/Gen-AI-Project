"""Synthetic claim/evidence pair generation for verifier training.

Generates labeled (claim, evidence, label, score) training pairs for the
faithfulness verifier. Uses a strong LLM to:
1. Generate entailed claims from real passages (positives)
2. Corrupt claims via negation, entity swapping, or injected unsupported facts (negatives)

This is a callback to synthetic-data-generator project experience.

Output format (JSONL):
    {
        "claim": "string — the claim text",
        "evidence": "string — the evidence passage",
        "label": "entailment" | "contradiction" | "neutral",
        "faithfulness_score": float (0.0 to 1.0)
    }
"""

import json
import random
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.services.llm_adapter import get_llm

logger = get_logger("faithforge.data_synthesis")


# ── Prompt Templates ─────────────────────────────────────────────────────────

CLAIM_GENERATION_PROMPT = """Given the following passage, generate {n} atomic, verifiable claims
that are directly supported by the passage. Each claim should be a single sentence.

Passage:
{passage}

Output JSON: {{"claims": ["claim1", "claim2", ...]}}"""


CLAIM_CORRUPTION_PROMPT = """Given the following faithful claim and its source passage,
create {n} corrupted versions using these strategies:
1. "negation" — negate a key fact in the claim
2. "entity_swap" — replace a named entity with a different one
3. "unsupported" — add a detail not present in the passage

Faithful claim: {claim}
Source passage: {passage}

Output JSON: {{
  "corruptions": [
    {{"text": "corrupted claim", "strategy": "negation|entity_swap|unsupported"}}
  ]
}}"""


SELF_CRITIQUE_PROMPT = """You are a fact-checker. Given a claim and evidence, determine whether
the claim is supported by the evidence.

Claim: {claim}
Evidence: {evidence}

Classify the claim as:
- "entailment" if the evidence fully supports the claim
- "contradiction" if the evidence contradicts the claim
- "neutral" if the evidence neither supports nor contradicts the claim

Output JSON: {{"label": "entailment|contradiction|neutral", "reasoning": "brief explanation"}}"""


# ── Faithful Claim Generation ────────────────────────────────────────────────

async def generate_faithful_claims(passage: str, n: int = 3) -> list[dict]:
    """Generate n faithful claims from a passage using the LLM.

    Each claim is an atomic, verifiable statement directly supported
    by the passage text. These become the positive (entailment) training examples.

    Args:
        passage: Source text to generate claims from.
        n: Number of claims to generate.

    Returns:
        List of dicts: [{"text": "claim text", "label": "entailment", "score": 0.9}, ...]
    """
    llm = get_llm()

    prompt = CLAIM_GENERATION_PROMPT.format(n=n, passage=passage[:2000])

    try:
        data = await llm.chat_json(
            "You are a precise claim extraction system. Generate atomic, verifiable claims.",
            prompt,
        )
        raw_claims = data.get("claims", [])
    except Exception as e:
        logger.error("Claim generation failed: %s", e)
        return []

    results = []
    for claim_text in raw_claims:
        if not claim_text or len(claim_text.strip()) < 10:
            continue
        results.append({
            "text": claim_text.strip(),
            "label": "entailment",
            "score": round(random.uniform(0.8, 1.0), 2),
        })

    logger.debug("Generated %d faithful claims from passage", len(results))
    return results


# ── Claim Corruption ─────────────────────────────────────────────────────────

async def corrupt_claim(claim: str, passage: str) -> list[dict]:
    """Generate corrupted versions of a faithful claim.

    Creates three types of corruptions:
    - Negation: negate a key fact (contradiction)
    - Entity swap: replace a named entity (contradiction)
    - Unsupported: add a detail not in the passage (neutral)

    Args:
        claim: The original faithful claim.
        passage: The source passage (for context).

    Returns:
        List of dicts: [
            {"text": "corrupted claim", "label": "contradiction", "score": 0.1, "strategy": "negation"},
            ...
        ]
    """
    llm = get_llm()

    prompt = CLAIM_CORRUPTION_PROMPT.format(n=3, claim=claim, passage=passage[:1500])

    try:
        data = await llm.chat_json(
            "You are a claim corruption system. Create modified versions of claims.",
            prompt,
        )
        corruptions = data.get("corruptions", [])
    except Exception as e:
        logger.error("Claim corruption failed: %s", e)
        return []

    # Score ranges by strategy
    score_ranges = {
        "negation": (0.0, 0.2),
        "entity_swap": (0.0, 0.15),
        "unsupported": (0.3, 0.6),
    }

    results = []
    for corruption in corruptions:
        text = corruption.get("text", "").strip()
        strategy = corruption.get("strategy", "unsupported")

        if not text or len(text) < 10:
            continue

        # Determine label and score based on strategy
        if strategy in ("negation", "entity_swap"):
            label = "contradiction"
        else:
            label = "neutral"

        score_range = score_ranges.get(strategy, (0.3, 0.6))
        score = round(random.uniform(*score_range), 2)

        results.append({
            "text": text,
            "label": label,
            "score": score,
            "strategy": strategy,
        })

    logger.debug("Generated %d corruptions for claim", len(results))
    return results


# ── Self-Critique Baseline ───────────────────────────────────────────────────

async def self_critique_label(claim: str, evidence: str) -> dict:
    """Generate a self-critique label using the same LLM that generated the answer.

    This is the baseline for ablation 1: does the generator LLM's self-assessment
    match the fine-tuned verifier's judgment?

    Args:
        claim: The claim to evaluate.
        evidence: The evidence to check against.

    Returns:
        Dict: {"label": "entailment|contradiction|neutral", "reasoning": "..."}
    """
    llm = get_llm()

    prompt = SELF_CRITIQUE_PROMPT.format(claim=claim, evidence=evidence[:1500])

    try:
        data = await llm.chat_json(
            "You are a fact-checker. Be precise and conservative.",
            prompt,
        )
        return {
            "label": data.get("label", "neutral"),
            "reasoning": data.get("reasoning", ""),
        }
    except Exception as e:
        logger.error("Self-critique failed: %s", e)
        return {"label": "neutral", "reasoning": f"Error: {e}"}


# ── Full Dataset Generation ──────────────────────────────────────────────────

async def generate_dataset(
    passages: list[str],
    output_dir: Path,
    claims_per_passage: int = 3,
    corruptions_per_claim: int = 3,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """Generate the full training dataset.

    Pipeline:
    1. For each passage, generate faithful claims (entailment)
    2. For each faithful claim, generate corruptions (contradiction + neutral)
    3. Shuffle all pairs
    4. Split into train/val/test
    5. Write JSONL files

    Args:
        passages: List of source passages.
        output_dir: Directory to write train.jsonl, val.jsonl, test.jsonl.
        claims_per_passage: How many faithful claims per passage.
        corruptions_per_claim: How many corruptions per faithful claim.
        val_ratio: Fraction for validation split.
        test_ratio: Fraction for test split.

    Returns:
        Dict with dataset stats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    label_counts = {"entailment": 0, "contradiction": 0, "neutral": 0}

    logger.info(
        "Generating dataset: %d passages, %d claims/passage, %d corruptions/claim",
        len(passages), claims_per_passage, corruptions_per_claim,
    )

    for i, passage in enumerate(passages):
        if (i + 1) % 10 == 0:
            logger.info("  Processing passage %d/%d", i + 1, len(passages))

        # Generate faithful claims
        faithful_claims = await generate_faithful_claims(passage, n=claims_per_passage)

        for claim_data in faithful_claims:
            # Add the faithful claim
            all_pairs.append({
                "claim": claim_data["text"],
                "evidence": passage,
                "label": claim_data["label"],
                "faithfulness_score": claim_data["score"],
            })
            label_counts[claim_data["label"]] += 1

            # Generate corruptions
            corruptions = await corrupt_claim(claim_data["text"], passage)
            for corruption in corruptions[:corruptions_per_claim]:
                all_pairs.append({
                    "claim": corruption["text"],
                    "evidence": passage,
                    "label": corruption["label"],
                    "faithfulness_score": corruption["score"],
                })
                label_counts[corruption["label"]] += 1

    # Shuffle
    random.shuffle(all_pairs)

    # Split
    total = len(all_pairs)
    test_size = int(total * test_ratio)
    val_size = int(total * val_ratio)
    train_size = total - test_size - val_size

    train_data = all_pairs[:train_size]
    val_data = all_pairs[train_size:train_size + val_size]
    test_data = all_pairs[train_size + val_size:]

    # Write JSONL files
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for pair in split_data:
                f.write(json.dumps(pair) + "\n")
        logger.info("  Wrote %d pairs to %s", len(split_data), path)

    stats = {
        "total_pairs": total,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "label_distribution": label_counts,
    }

    logger.info("Dataset generation complete: %s", stats)
    return stats
