# apps/agent/ag3nt_agent/model_fallback.py
"""Model fallback chain with cooldown tracking and error classification."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Backoff schedule in seconds: 30s, 1m, 5m, 15m, 60m
ERROR_BACKOFF_SCHEDULE = [30, 60, 300, 900, 3600]


def classify_error(error: Exception) -> str:
    """Classify an exception into an error category."""
    msg = str(error).lower()
    if "rate limit" in msg or "rate_limit" in msg or "429" in msg:
        return "rate_limit"
    if "auth" in msg or "401" in msg or "403" in msg or "api key" in msg:
        return "auth"
    if "context" in msg and ("window" in msg or "overflow" in msg or "length" in msg):
        return "context_overflow"
    if "billing" in msg or "payment" in msg or "quota" in msg:
        return "billing"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    return "unknown"


@dataclass
class ProviderState:
    """Tracks health state for a single provider."""
    consecutive_errors: int = 0
    last_failure_at: float = 0.0
    last_error_type: str = ""
    cooldown_until: float = 0.0


def _create_model_for_provider(provider_config: dict):
    """Create a LangChain model instance for a provider config."""
    from ag3nt_agent.model_config import create_model_for_provider
    return create_model_for_provider(
        provider_config["provider"],
        provider_config["model"],
    )


class ModelFallbackChain:
    """Manages a chain of model providers with fallback and cooldown logic."""

    def __init__(
        self,
        providers: list[dict],
        cooldown_seconds: float = 60.0,
    ):
        self.providers = providers
        self.cooldown_seconds = cooldown_seconds
        self._states: list[ProviderState] = [ProviderState() for _ in providers]
        self._model_cache: dict[int, object] = {}

    @classmethod
    def from_env(cls) -> ModelFallbackChain:
        """Build a fallback chain from environment variables."""
        providers = []
        provider_keys = [
            ("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-5-20250929"),
            ("openai", "OPENAI_API_KEY", "gpt-4o"),
            ("openrouter", "OPENROUTER_API_KEY", "moonshotai/kimi-k2.5"),
            ("google", "GOOGLE_API_KEY", "gemini-pro"),
        ]
        explicit_provider = os.environ.get("AG3NT_MODEL_PROVIDER")
        explicit_model = os.environ.get("AG3NT_MODEL_NAME")
        if explicit_provider:
            default_model = next(
                (m for p, _, m in provider_keys if p == explicit_provider),
                explicit_provider,
            )
            providers.append({
                "provider": explicit_provider,
                "model": explicit_model or default_model,
            })
        for provider, key_var, default_model in provider_keys:
            if os.environ.get(key_var):
                entry = {"provider": provider, "model": default_model}
                if not any(
                    p["provider"] == provider and p["model"] == default_model
                    for p in providers
                ):
                    providers.append(entry)
        if not providers:
            providers.append({
                "provider": "anthropic",
                "model": "claude-sonnet-4-5-20250929",
            })
        return cls(providers=providers)

    def get_model(self, index: int = 0):
        """Get or create a model instance for the given provider index."""
        if index in self._model_cache:
            return self._model_cache[index]
        model = _create_model_for_provider(self.providers[index])
        self._model_cache[index] = model
        return model

    def mark_failure(self, index: int, error_type: str) -> None:
        """Record a failure for a provider."""
        state = self._states[index]
        state.consecutive_errors += 1
        state.last_failure_at = time.time()
        state.last_error_type = error_type
        backoff_idx = min(state.consecutive_errors - 1, len(ERROR_BACKOFF_SCHEDULE) - 1)
        backoff = ERROR_BACKOFF_SCHEDULE[backoff_idx]
        cooldown = max(self.cooldown_seconds, backoff)
        state.cooldown_until = time.time() + cooldown
        logger.warning(
            "Provider %s failed (%s), cooldown %.0fs, errors=%d",
            self.providers[index]["provider"],
            error_type, cooldown, state.consecutive_errors,
        )

    def mark_success(self, index: int) -> None:
        """Reset error tracking after a successful call."""
        state = self._states[index]
        state.consecutive_errors = 0
        state.cooldown_until = 0.0
        state.last_error_type = ""

    def is_in_cooldown(self, index: int) -> bool:
        """Check if a provider is currently in cooldown."""
        return time.time() < self._states[index].cooldown_until

    def get_consecutive_errors(self, index: int) -> int:
        """Get the consecutive error count for a provider."""
        return self._states[index].consecutive_errors

    def get_next_available(self, start: int = 0) -> int | None:
        """Find the next provider not in cooldown, starting from index."""
        for i in range(len(self.providers)):
            idx = (start + i) % len(self.providers)
            if not self.is_in_cooldown(idx):
                return idx
        return None


def run_with_fallback(chain: ModelFallbackChain, action, max_attempts: int = 0):
    """Run an action with automatic model fallback on failure.

    Args:
        chain: The fallback chain to use.
        action: Callable that takes a model and returns a result.
        max_attempts: Max total attempts (0 = try each provider once).

    Returns:
        The result from the first successful action call.

    Raises:
        The last exception if all providers fail.
    """
    if max_attempts <= 0:
        max_attempts = len(chain.providers)
    last_error = None
    start_index = 0
    for attempt in range(max_attempts):
        idx = chain.get_next_available(start=start_index)
        if idx is None:
            break
        model = chain.get_model(idx)
        try:
            result = action(model)
            chain.mark_success(idx)
            return result
        except Exception as exc:
            error_type = classify_error(exc)
            chain.mark_failure(idx, error_type)
            last_error = exc
            logger.warning(
                "Attempt %d/%d failed (provider=%s, error=%s): %s",
                attempt + 1, max_attempts,
                chain.providers[idx]["provider"], error_type, exc,
            )
            start_index = (idx + 1) % len(chain.providers)
    if last_error:
        raise last_error
    raise RuntimeError("No available model providers")
