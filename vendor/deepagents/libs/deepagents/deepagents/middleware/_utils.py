"""Utility functions for middleware."""

from langchain_core.messages import SystemMessage


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message.

    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.

    Returns:
        New SystemMessage with the text appended.
    """
    if system_message is None:
        return SystemMessage(content=[{"type": "text", "text": text}])
    # Access content_blocks once and spread — avoids intermediate list() copy
    blocks = system_message.content_blocks
    prefix = "\n\n" if blocks else ""
    return SystemMessage(content=[*blocks, {"type": "text", "text": f"{prefix}{text}"}])


def append_cached_text(
    system_message: SystemMessage | None,
    text: str,
    cache: bool = True,
) -> SystemMessage:
    """Like append_to_system_message but with optional cache_control.

    Marks the block with cache_control={"type": "ephemeral"} when cache=True.
    This enables Anthropic prompt caching for stable content (identity, skills).

    Args:
        system_message: Existing system message or None.
        text: Text to add to the system message.
        cache: If True, add cache_control to the block. Defaults to True.

    Returns:
        New SystemMessage with the text appended.
    """
    block: dict = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}

    if system_message is None:
        return SystemMessage(content=[block])

    blocks = system_message.content_blocks
    prefix = "\n\n" if blocks else ""
    block["text"] = f"{prefix}{text}" if prefix else text
    return SystemMessage(content=[*blocks, block])
