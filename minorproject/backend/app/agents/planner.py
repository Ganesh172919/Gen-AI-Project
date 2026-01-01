"""Planner / complexity router agent for FaithForge.

Classifies incoming queries by complexity and determines the retrieval strategy:
- NONE: trivial, can be answered without retrieval
- SINGLE_HOP: answer requires one retrieval step
- MULTI_HOP: answer requires multiple retrieval steps / sub-queries

Baseline: few-shot LLM classification.
Your contribution: fine-tune a small classifier (DistilBERT) on labeled data.
"""

from app.core.logging import get_logger
from app.models.schemas import RetrievalStrategy
from app.services.llm_adapter import get_llm

logger = get_logger("faithforge.planner")

PLANNER_SYSTEM_PROMPT = """You are a query complexity classifier for a RAG system.

Classify the user's query into one of these strategies:
- "none": The query is trivial and can be answered without retrieval (e.g., "What is 2+2?")
- "single_hop": The answer requires retrieving information from one source (e.g., "When was Python created?")
- "multi_hop": The answer requires combining information from multiple sources or reasoning steps (e.g., "Compare the GDP of India and China in 2023")

For multi_hop queries, also break the query into sub-queries that each target a single retrieval step.

Respond in JSON:
{
  "strategy": "none" | "single_hop" | "multi_hop",
  "sub_queries": ["sub-query 1", "sub-query 2", ...],
  "reasoning": "brief explanation of classification"
}"""


class PlannerAgent:
    """Classifies query complexity and determines retrieval strategy.

    Uses LLM-based classification as the baseline. You can swap in a
    fine-tuned DistilBERT classifier for lower latency (see the TODO
    in the classify method).
    """

    async def classify(self, query: str) -> tuple[RetrievalStrategy, list[str]]:
        """Classify a query and determine retrieval strategy.

        Args:
            query: The user's raw query.

        Returns:
            Tuple of (strategy, sub_queries).
            For none/single_hop, sub_queries will contain just the original query.
            For multi_hop, sub_queries will contain the decomposed queries.
        """
        logger.info("Classifying query: '%s...'", query[:100])

        llm = get_llm()

        data = await llm.chat_json(
            PLANNER_SYSTEM_PROMPT,
            f"Classify this query:\n\n{query}",
        )

        strategy_str = data.get("strategy", "single_hop")
        strategy = RetrievalStrategy(strategy_str)
        sub_queries = data.get("sub_queries", [query])

        # Ensure sub_queries is never empty
        if not sub_queries:
            sub_queries = [query]

        # TODO: Replace the LLM call above with a fine-tuned DistilBERT classifier
        # for lower latency. The training script is in the complexity classifier
        # training notebook. Interface should be:
        #
        #   strategy, sub_queries = self._classifier_predict(query)

        logger.info("Strategy: %s, sub_queries: %d", strategy.value, len(sub_queries))
        return strategy, sub_queries


# ── Module-level singleton ───────────────────────────────────────────────────

_planner: PlannerAgent | None = None


def get_planner() -> PlannerAgent:
    """Get or create the module-level PlannerAgent singleton."""
    global _planner
    if _planner is None:
        _planner = PlannerAgent()
    return _planner
