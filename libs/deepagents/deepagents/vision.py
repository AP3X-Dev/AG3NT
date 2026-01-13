"""Vision utilities for multimodal LLM capabilities.

This module provides utilities for using vision-capable LLMs to analyze images,
PDFs, and other visual content.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# MIME type mapping for common image formats
MIME_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
    "application/pdf": "pdf",
    "image/svg+xml": "svg",
}


def _get_media_type(file_type: str) -> str:
    """Get the media type string for the image_url."""
    # If already a MIME type, return as-is
    if "/" in file_type:
        return file_type
    # Map common extensions
    ext_to_mime = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "pdf": "application/pdf",
        "svg": "image/svg+xml",
    }
    return ext_to_mime.get(file_type.lower(), "application/octet-stream")


def invoke_vision_model(
    model: BaseChatModel,
    image_bytes: bytes,
    query: str,
    context: str | None = None,
    file_type: str = "image/png",
) -> str:
    """Invoke a vision-capable model to analyze an image.

    Args:
        model: A LangChain chat model that supports vision (e.g., ChatOpenAI with gpt-4-vision, ChatAnthropic with Claude 3+).
        image_bytes: Raw bytes of the image/file to analyze.
        query: The analysis query/question about the image.
        context: Optional additional context to help with analysis.
        file_type: MIME type or extension of the file (e.g., "image/png" or "png").

    Returns:
        The model's analysis response as a string.

    Raises:
        Exception: If the model invocation fails.
    """
    media_type = _get_media_type(file_type)
    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    # Build the prompt
    prompt_parts = [query]
    if context:
        prompt_parts.append(f"\nContext: {context}")
    prompt_text = "\n".join(prompt_parts)

    # Create multimodal message
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
            },
        ]
    )

    try:
        response = model.invoke([message])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Vision model invocation failed: {e}")
        raise


async def ainvoke_vision_model(
    model: BaseChatModel,
    image_bytes: bytes,
    query: str,
    context: str | None = None,
    file_type: str = "image/png",
) -> str:
    """Async version of invoke_vision_model.

    Args:
        model: A LangChain chat model that supports vision.
        image_bytes: Raw bytes of the image/file to analyze.
        query: The analysis query/question about the image.
        context: Optional additional context to help with analysis.
        file_type: MIME type or extension of the file.

    Returns:
        The model's analysis response as a string.
    """
    media_type = _get_media_type(file_type)
    b64_data = base64.b64encode(image_bytes).decode("utf-8")

    prompt_parts = [query]
    if context:
        prompt_parts.append(f"\nContext: {context}")
    prompt_text = "\n".join(prompt_parts)

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
            },
        ]
    )

    try:
        response = await model.ainvoke([message])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Async vision model invocation failed: {e}")
        raise
