"""Model configuration and creation for AG3NT.

Extracted from deepagents_runtime.py for maintainability.
Provides model provider detection, configuration, and instance creation.
"""

from __future__ import annotations

import logging
import os

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("ag3nt.model")

# ---------------------------------------------------------------------------
# Patch langchain_openai to preserve reasoning_content through round-trips.
#
# OpenAI-compatible APIs with thinking/reasoning models (DeepSeek, Kimi K2
# Thinking, etc.) include a ``reasoning_content`` field on assistant messages.
# LangChain's ChatOpenAI does not capture this field when parsing responses
# and does not include it when constructing API requests, causing 400 errors
# on subsequent turns:
#
#   "thinking is enabled but reasoning_content is missing in assistant
#    tool call message at index N"
#
# The patches below fix the three conversion functions so that
# ``reasoning_content`` survives the full round-trip:
#   API response → AIMessage (additional_kwargs) → API request dict
# ---------------------------------------------------------------------------
_reasoning_patch_applied = False


def _apply_reasoning_content_patch() -> None:
    """Monkey-patch langchain_openai message conversion to preserve reasoning_content."""
    global _reasoning_patch_applied
    if _reasoning_patch_applied:
        return
    _reasoning_patch_applied = True

    try:
        import langchain_openai.chat_models.base as _oai_base
        from langchain_core.messages import AIMessage, AIMessageChunk
    except ImportError:
        logger.debug("langchain_openai not available, skipping reasoning_content patch")
        return

    # --- 1. Patch _convert_dict_to_message ---
    _orig_convert_dict = _oai_base._convert_dict_to_message

    def _patched_convert_dict_to_message(_dict, **kwargs):  # type: ignore[override]
        msg = _orig_convert_dict(_dict, **kwargs)
        if (
            _dict.get("role") == "assistant"
            and isinstance(msg, AIMessage)
            and "reasoning_content" in _dict
            and _dict["reasoning_content"]
        ):
            msg.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
        return msg

    _oai_base._convert_dict_to_message = _patched_convert_dict_to_message

    # --- 2. Patch _convert_delta_to_message_chunk ---
    _orig_convert_delta = _oai_base._convert_delta_to_message_chunk

    def _patched_convert_delta_to_message_chunk(_dict, default_class, **kwargs):  # type: ignore[override]
        chunk = _orig_convert_delta(_dict, default_class, **kwargs)
        if isinstance(chunk, AIMessageChunk) and "reasoning_content" in _dict:
            rc = _dict["reasoning_content"]
            if rc:
                chunk.additional_kwargs["reasoning_content"] = rc
        return chunk

    _oai_base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk

    # --- 3. Patch _convert_message_to_dict ---
    _orig_convert_msg_dict = _oai_base._convert_message_to_dict

    def _patched_convert_message_to_dict(message, *args, **kwargs):  # type: ignore[override]
        result = _orig_convert_msg_dict(message, *args, **kwargs)
        if isinstance(message, AIMessage):
            rc = message.additional_kwargs.get("reasoning_content")
            if rc is not None:
                result["reasoning_content"] = rc
        return result

    _oai_base._convert_message_to_dict = _patched_convert_message_to_dict

    logger.info("Applied reasoning_content round-trip patches to langchain_openai")


# Apply patches at import time so they're active before any model is created.
_apply_reasoning_content_patch()

# ---------------------------------------------------------------------------
# Model instance cache – avoids re-creating on every agent rebuild
# ---------------------------------------------------------------------------
_model_cache: dict[str, BaseChatModel] = {}


def get_model_config() -> tuple[str, str]:
    """Get the model provider and name from environment.

    Returns:
        Tuple of (provider, model_name)
    """
    provider = os.environ.get("AG3NT_MODEL_PROVIDER")
    model = os.environ.get("AG3NT_MODEL_NAME")

    if not provider:
        if os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "google"
        elif os.environ.get("KIMI_API_KEY"):
            provider = "kimi"
        else:
            provider = "anthropic"

    if not model:
        defaults = {
            "openrouter": "moonshotai/kimi-k2.5",
            "anthropic": "claude-sonnet-4-5-20250929",
            "openai": "gpt-4o",
            "google": "gemini-pro",
            "kimi": "kimi-for-coding",
        }
        model = defaults.get(provider, "claude-sonnet-4-5-20250929")

    return provider, model


def _create_openrouter_model(model_name: str) -> BaseChatModel:
    """Create a ChatOpenAI instance configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required when using OpenRouter. "
            "Get your API key from https://openrouter.ai/keys"
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        max_retries=3,
        default_headers={
            "HTTP-Referer": "https://github.com/ag3nt",
            "X-Title": "AG3NT",
        },
    )


def _create_kimi_model(model_name: str) -> BaseChatModel:
    """Create a ChatOpenAI instance configured for Kimi.

    Uses the Kimi Coding API (api.kimi.com/coding/v1) for kimi-for-coding,
    falls back to the legacy Moonshot API for other model names.
    """
    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        raise ValueError(
            "KIMI_API_KEY environment variable is required when using Kimi. "
            "Get your API key from https://www.kimi.com/code/docs/en/"
        )

    from langchain_openai import ChatOpenAI

    if model_name == "kimi-for-coding":
        return ChatOpenAI(
            model="kimi-for-coding",
            openai_api_key=api_key,
            openai_api_base="https://api.kimi.com/coding/v1",
            max_tokens=32768,
            max_retries=3,
            default_headers={
                "HTTP-Referer": "https://github.com/RooVetGit/Roo-Cline",
                "X-Title": "Roo Code",
                "User-Agent": "RooCode/1.0.0",
            },
        )

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://api.moonshot.cn/v1",
        max_retries=3,
    )


def create_model(*, use_cache: bool = True) -> BaseChatModel | str:
    """Create the appropriate model instance based on provider.

    Args:
        use_cache: If True (default), return a cached instance when the
                   provider+model combination has been created before.

    Returns:
        Either a BaseChatModel instance (for OpenRouter, Kimi) or a string
        in ``"provider:model"`` format for LangChain's ``init_chat_model()``.
    """
    provider, model_name = get_model_config()
    cache_key = f"{provider}:{model_name}"

    if use_cache and cache_key in _model_cache:
        logger.debug("Returning cached model instance for %s", cache_key)
        return _model_cache[cache_key]

    if provider == "openrouter":
        instance = _create_openrouter_model(model_name)
        _model_cache[cache_key] = instance
        return instance

    if provider == "kimi":
        instance = _create_kimi_model(model_name)
        _model_cache[cache_key] = instance
        return instance

    # For other providers, use init_chat_model with retry support
    try:
        from langchain.chat_models import init_chat_model

        model_str = f"{provider}:{model_name}"
        instance = init_chat_model(model_str, max_retries=3)
        _model_cache[cache_key] = instance
        return instance
    except (ImportError, Exception) as exc:
        logger.warning("init_chat_model with retries failed (%s), falling back to string", exc)
        return f"{provider}:{model_name}"


def create_model_for_provider(provider: str, model_name: str):
    """Create a model instance for a specific provider and model name.

    Used by ModelFallbackChain to create models on demand.
    """
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY required for openrouter provider")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://github.com/ag3nt-dev/ag3nt", "X-Title": "AG3NT"},
            max_retries=3,
        )
    elif provider == "kimi":
        api_key = os.environ.get("KIMI_API_KEY")
        if not api_key:
            raise ValueError("KIMI_API_KEY required for kimi provider")
        from langchain_openai import ChatOpenAI
        base_url = "https://api.kimi.com/coding/v1" if "coding" in model_name.lower() else "https://api.moonshot.cn/v1"
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            max_retries=3,
        )
    else:
        from langchain.chat_models import init_chat_model
        return init_chat_model(f"{provider}:{model_name}", max_retries=3)
