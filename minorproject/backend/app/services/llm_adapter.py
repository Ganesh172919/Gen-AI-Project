"""Unified LLM adapter for FaithForge.

Provides a single interface to call any supported LLM provider (Groq, Cerebras,
OpenRouter) without leaking provider-specific details into agent logic.

Swap providers by changing the LLM_PROVIDER env var — no code changes needed.

Features:
- Exponential backoff retry on 429 (rate limit) and 5xx errors
- Configurable timeout
- Request/response logging for debugging
"""

import asyncio
import json
from typing import Any, Optional

import httpx

from app.core.config import LLMProvider, settings
from app.core.logging import get_logger

logger = get_logger("faithforge.llm")

# Provider base URLs
PROVIDER_URLS = {
    LLMProvider.GROQ: "https://api.groq.com/openai/v1/chat/completions",
    LLMProvider.CEREBRAS: "https://api.cerebras.ai/v1/chat/completions",
    LLMProvider.OPENROUTER: "https://openrouter.ai/api/v1/chat/completions",
}

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
BACKOFF_MULTIPLIER = 2.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class LLMAdapter:
    """Stateless adapter for calling LLM APIs.

    Usage:
        llm = LLMAdapter()
        response = await llm.chat("You are a helpful assistant.", "What is RAG?")
    """

    def __init__(self, provider: Optional[LLMProvider] = None, timeout: float = 60.0):
        self.provider = provider or settings.llm_provider
        self.base_url = PROVIDER_URLS[self.provider]
        self._api_key = self._resolve_api_key()
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
        self._request_count = 0
        self._error_count = 0

    def _resolve_api_key(self) -> str:
        """Get the API key for the configured provider."""
        key_map = {
            LLMProvider.GROQ: settings.groq_api_key,
            LLMProvider.CEREBRAS: settings.cerebras_api_key,
            LLMProvider.OPENROUTER: settings.openrouter_api_key,
        }
        key = key_map.get(self.provider)
        if not key:
            raise ValueError(f"API key not set for provider: {self.provider.value}")
        return key

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request with automatic retry on transient errors.

        Args:
            system_prompt: System message content.
            user_message: User message content.
            model: Override the default model for this call.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            response_format: JSON mode spec (e.g., {"type": "json_object"}).

        Returns:
            The assistant's response text.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-retryable error.
            httpx.TimeoutException: If all retries are exhausted due to timeouts.
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter requires an extra header
        if self.provider == LLMProvider.OPENROUTER:
            headers["HTTP-Referer"] = "https://faithforge.dev"

        payload: dict[str, Any] = {
            "model": model or settings.generator_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature if temperature is not None else settings.generator_temperature,
            "max_tokens": max_tokens or settings.generator_max_tokens,
        }

        if response_format:
            payload["response_format"] = response_format

        self._request_count += 1
        logger.debug(
            "LLM call #%d: provider=%s model=%s",
            self._request_count, self.provider.value, payload["model"],
        )

        # Retry loop with exponential backoff
        last_exception: Optional[Exception] = None
        backoff = INITIAL_BACKOFF_S

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = await self._client.post(self.base_url, json=payload, headers=headers)

                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    retry_after = resp.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else backoff
                    logger.warning(
                        "LLM call got %d, retrying in %.1fs (attempt %d/%d)",
                        resp.status_code, wait, attempt + 1, MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    backoff *= BACKOFF_MULTIPLIER
                    continue

                resp.raise_for_status()

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug("LLM response length=%d chars", len(content))
                return content

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "LLM call timed out, retrying in %.1fs (attempt %d/%d)",
                        backoff, attempt + 1, MAX_RETRIES,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                    continue
                break

            except httpx.HTTPStatusError as e:
                if e.response.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    last_exception = e
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                    continue
                self._error_count += 1
                raise

        # All retries exhausted
        self._error_count += 1
        logger.error("LLM call failed after %d retries", MAX_RETRIES)
        if last_exception:
            raise last_exception
        raise RuntimeError("LLM call failed with no response")

    async def chat_json(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs,
    ) -> dict:
        """Call the LLM and parse the response as JSON.

        Convenience wrapper around chat() that requests JSON mode and
        parses the response.

        Returns:
            Parsed JSON response as a dict.
        """
        raw = await self.chat(
            system_prompt,
            user_message,
            response_format={"type": "json_object"},
            **kwargs,
        )
        return json.loads(raw)

    def get_stats(self) -> dict:
        """Get adapter usage statistics."""
        return {
            "provider": self.provider.value,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1),
        }

    async def close(self) -> None:
        """Shut down the HTTP client."""
        stats = self.get_stats()
        logger.info("LLM adapter closing: %s", stats)
        await self._client.aclose()


# ── Module-level singleton ───────────────────────────────────────────────────

_llm: Optional[LLMAdapter] = None


def get_llm() -> LLMAdapter:
    """Get or create the module-level LLMAdapter singleton."""
    global _llm
    if _llm is None:
        _llm = LLMAdapter()
    return _llm


async def close_llm() -> None:
    """Close the module-level LLM adapter (call at shutdown)."""
    global _llm
    if _llm is not None:
        await _llm.close()
        _llm = None
