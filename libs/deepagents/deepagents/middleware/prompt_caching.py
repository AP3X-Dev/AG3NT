"""PromptCachingMiddleware - Provider-agnostic prompt caching.

This middleware provides prompt caching functionality that works across
different LLM providers. Currently wraps Anthropic's prompt caching
implementation but presents a provider-agnostic interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from typing import Self

# Import the underlying Anthropic implementation
try:
    from langchain_anthropic.middleware import (
        AnthropicPromptCachingMiddleware as _AnthropicPromptCachingMiddleware,
    )

    _HAS_ANTHROPIC = True
except ImportError:
    _AnthropicPromptCachingMiddleware = None  # type: ignore[misc, assignment]
    _HAS_ANTHROPIC = False


class PromptCachingMiddleware:
    """Provider-agnostic prompt caching middleware.

    Caches system prompts to reduce API costs and latency for subsequent calls.
    Currently supports Anthropic models with automatic fallback for unsupported providers.

    Args:
        unsupported_model_behavior: How to handle unsupported models.
            - "ignore": Silently pass through without caching (default)
            - "warn": Log a warning but continue
            - "error": Raise an exception

    Example:
        ```python
        from deepagents.middleware import PromptCachingMiddleware

        agent = create_deep_agent(
            model="claude-3-5-sonnet-latest",
            middleware=[
                PromptCachingMiddleware(unsupported_model_behavior="ignore"),
            ],
        )
        ```
    """

    def __new__(
        cls,
        unsupported_model_behavior: Literal["ignore", "warn", "error"] = "ignore",
    ) -> Self:
        """Create a new PromptCachingMiddleware instance.

        Returns the appropriate provider-specific implementation based on
        what's available in the environment.
        """
        if _HAS_ANTHROPIC and _AnthropicPromptCachingMiddleware is not None:
            # Return the Anthropic implementation directly
            return _AnthropicPromptCachingMiddleware(  # type: ignore[return-value]
                unsupported_model_behavior=unsupported_model_behavior
            )
        # Return a no-op middleware if no caching implementation is available
        return object.__new__(_NoOpCachingMiddleware)  # type: ignore[return-value]


class _NoOpCachingMiddleware(PromptCachingMiddleware):
    """No-op middleware when no prompt caching implementation is available."""

    def __new__(cls) -> _NoOpCachingMiddleware:  # noqa: PYI034
        """Create a no-op middleware instance."""
        return object.__new__(cls)


# Export the main class
__all__ = ["PromptCachingMiddleware"]
