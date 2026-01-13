"""PromptCachingMiddleware - Provider-agnostic prompt caching.

This middleware provides prompt caching functionality that works across
different LLM providers. Currently wraps Anthropic's prompt caching
implementation but presents a provider-agnostic interface.
"""

from __future__ import annotations

from typing import Literal

# Import the underlying Anthropic implementation
try:
    from langchain_anthropic.middleware import (
        AnthropicPromptCachingMiddleware as _AnthropicPromptCachingMiddleware,
    )
    _HAS_ANTHROPIC = True
except ImportError:
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
            ]
        )
        ```
    """
    
    def __new__(
        cls,
        unsupported_model_behavior: Literal["ignore", "warn", "error"] = "ignore",
    ) -> "PromptCachingMiddleware":
        """Create a new PromptCachingMiddleware instance.
        
        Returns the appropriate provider-specific implementation based on
        what's available in the environment.
        """
        if _HAS_ANTHROPIC:
            # Return the Anthropic implementation directly
            return _AnthropicPromptCachingMiddleware(
                unsupported_model_behavior=unsupported_model_behavior
            )
        else:
            # Return a no-op middleware if no caching implementation is available
            return _NoOpCachingMiddleware()


class _NoOpCachingMiddleware:
    """No-op middleware when no prompt caching implementation is available."""
    
    async def __call__(self, request, call_next):
        """Pass through without modification."""
        return await call_next(request)


# Export the main class
__all__ = ["PromptCachingMiddleware"]

