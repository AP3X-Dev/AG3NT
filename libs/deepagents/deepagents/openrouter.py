"""OpenRouter integration for deepagents.

This module provides integration with OpenRouter, supporting both:
1. Official OpenRouter Python SDK (recommended)
2. OpenAI SDK with OpenRouter endpoint (fallback)

Latest models as of January 2026:
- Claude Sonnet 4.5, Claude Opus 4.5
- GPT-5.2, GPT-5.1
- Gemini 3 Pro, Gemini 2.5 Pro
- And 300+ more models
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def load_env() -> None:
    """Load environment variables from .env file if it exists."""
    load_dotenv()


def is_openrouter_configured() -> bool:
    """Check if OpenRouter is configured via environment variables.

    Returns:
        True if OPENROUTER_API_KEY is set, False otherwise.
    """
    return bool(os.getenv("OPENROUTER_API_KEY"))


def get_openrouter_model(model_name: str | None = None, **kwargs: Any) -> BaseChatModel:
    """Create a ChatOpenAI instance configured for OpenRouter.

    Uses the OpenAI-compatible API endpoint provided by OpenRouter.
    This works with all 300+ models available on OpenRouter.

    Args:
        model_name: The model to use from OpenRouter. If None, uses OPENROUTER_MODEL
                   environment variable or defaults to "anthropic/claude-sonnet-4.5".
        **kwargs: Additional arguments to pass to ChatOpenAI.

    Returns:
        ChatOpenAI instance configured for OpenRouter.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.

    Examples:
        Latest models (2026):
        >>> model = get_openrouter_model("anthropic/claude-sonnet-4.5")
        >>> model = get_openrouter_model("anthropic/claude-opus-4.5")
        >>> model = get_openrouter_model("openai/gpt-5.2")
        >>> model = get_openrouter_model("google/gemini-3-pro")
        >>> model = get_openrouter_model("meta-llama/llama-3.1-405b-instruct")
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        msg = (
            "OPENROUTER_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment. "
            "Get your API key from: https://openrouter.ai/keys"
        )
        raise ValueError(msg)

    # Get model name from parameter, environment, or default
    # Updated default to Claude Sonnet 4.5 (latest as of 2026)
    if model_name is None:
        model_name = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")

    # Set default kwargs for OpenRouter
    openrouter_kwargs: dict[str, Any] = {
        "model": model_name,
        "openai_api_key": api_key,
        "openai_api_base": "https://openrouter.ai/api/v1",
        "default_headers": {
            "HTTP-Referer": "https://github.com/langchain-ai/deepagents",
            "X-Title": "DeepAgents",
        },
    }

    # Merge with user-provided kwargs (user kwargs take precedence)
    openrouter_kwargs.update(kwargs)

    return ChatOpenAI(**openrouter_kwargs)


def get_default_openrouter_model() -> BaseChatModel:
    """Get the default OpenRouter model configured via environment variables.

    This is a convenience function that uses the OPENROUTER_MODEL environment
    variable or defaults to "anthropic/claude-sonnet-4.5" (latest as of 2026).

    Returns:
        ChatOpenAI instance configured for OpenRouter with default settings.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    return get_openrouter_model()
