"""Agentic memory flush -- uses a lightweight LLM call to extract insights before compaction."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_FILE = Path.home() / ".ag3nt" / "MEMORY.md"

FLUSH_PROMPT = """You are about to lose older context due to compaction.
Review these messages and extract any information worth preserving:
- Decisions made and their rationale
- User preferences discovered
- Facts learned about the codebase/project
- Solutions that worked (or didn't)
- Error patterns and their resolutions

Write each as a concise bullet point. Only include genuinely important items.
If nothing is worth preserving, respond with "NONE".

Messages to review:
{messages}"""


def _format_messages_for_flush(messages: list) -> str:
    """Format messages into a string for the flush prompt."""
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", str(msg))
        if isinstance(content, list):
            # Handle structured content (list of blocks)
            text_parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            content = " ".join(text_parts)
        if content:
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}] {content}")
    return "\n".join(lines[-50:])  # Last 50 messages max


def _append_insights_to_memory(insights: str, session_id: str | None = None) -> int:
    """Append extracted insights to MEMORY.md. Returns count of insights."""
    if not insights or insights.strip().upper() == "NONE":
        return 0

    lines = [l.strip() for l in insights.strip().split("\n") if l.strip()]
    # Filter to bullet points only
    bullet_lines = [l for l in lines if l.startswith("-") or l.startswith("*")]
    if not bullet_lines:
        # Try treating all non-empty lines as insights
        bullet_lines = [f"- {l}" for l in lines if l and l.upper() != "NONE"]

    if not bullet_lines:
        return 0

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    session_str = f", session: {session_id}" if session_id else ""
    header = f"\n## Agentic flush ({timestamp}{session_str})\n"

    with open(MEMORY_FILE, "a", encoding="utf-8") as fh:
        fh.write(header)
        for line in bullet_lines:
            fh.write(f"{line}\n")

    return len(bullet_lines)


async def agentic_flush(
    messages: list,
    session_id: str | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> int:
    """Run agentic memory flush on messages about to be compacted.

    Uses a cheap/fast model to extract insights worth preserving.

    Args:
        messages: Messages about to be compacted away.
        session_id: Current session ID for logging.
        model: Model to use for extraction (default: cheapest available).

    Returns:
        Number of insights extracted and saved.
    """
    if not messages:
        return 0

    formatted = _format_messages_for_flush(messages)
    if not formatted.strip():
        return 0

    prompt = FLUSH_PROMPT.format(messages=formatted)

    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=model, max_tokens=1024, temperature=0)
        response = await llm.ainvoke(prompt)
        insights_text = response.content if isinstance(response.content, str) else str(response.content)

        count = _append_insights_to_memory(insights_text, session_id)
        if count > 0:
            logger.info("Agentic flush extracted %d insights for session %s", count, session_id)
        return count

    except Exception:
        logger.debug("Agentic flush failed", exc_info=True)
        return 0
