"""Context-window management tools.

Provides:
- dump_to_artifact / read_artifact — offload large content to the ArtifactStore.
- check_context_budget — query current context-window usage ratio.
- compact_now — request immediate context compaction.
- write_note / read_note / list_notes — persistent scratchpad notes.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.tools import tool

from ag3nt_agent.artifact_store import get_artifact_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Compaction trigger hook — set at agent start-up so tools can inspect usage.
# ---------------------------------------------------------------------------

_compaction_trigger = None


def set_compaction_trigger(trigger) -> None:
    """Wire the :class:`CompactionTrigger` instance into context tools."""
    global _compaction_trigger
    _compaction_trigger = trigger


def _get_compaction_trigger():
    """Return the current compaction trigger (or ``None``)."""
    return _compaction_trigger


@tool
def dump_to_artifact(content: str, label: str) -> str:
    """Store content in the artifact store and return a reference ID.

    Use this to offload large outputs (logs, API responses, file contents)
    out of the conversation context. You can retrieve the content later
    with read_artifact.

    Args:
        content: The text content to store.
        label: A short human-readable label describing the content.
    """
    store = get_artifact_store()
    meta = store.write_artifact(
        content=content,
        tool_name="dump_to_artifact",
        tags=[label] if label else [],
    )
    return (
        f"Stored as artifact {meta.artifact_id} "
        f"({meta.size_bytes} bytes, label={label!r}). "
        f"Use read_artifact with this ID to retrieve the content."
    )


@tool
def read_artifact(artifact_id: str) -> str:
    """Retrieve content previously stored with dump_to_artifact.

    Args:
        artifact_id: The artifact ID returned by dump_to_artifact.
    """
    store = get_artifact_store()
    content = store.read_artifact(artifact_id)
    if content is None:
        return f"Artifact {artifact_id!r} not found."
    return content


@tool
def check_context_budget() -> str:
    """Check current context window usage and compaction status.

    Returns usage ratio, active tier, and recommendation.
    Use to decide whether to offload content to artifacts or request compaction.
    """
    trigger = _get_compaction_trigger()
    if trigger is None:
        return "Context budget tracking not available."

    ratio = trigger.usage_ratio()
    cfg = trigger._config
    pct = ratio * 100

    # Determine tier based on ratio vs thresholds.
    if ratio >= cfg.compact_threshold:
        tier = "COMPACT"
        recommendation = (
            "Critical — context is nearly full. "
            "Run compact_now immediately and offload large content to artifacts."
        )
    elif ratio >= cfg.extract_threshold:
        tier = "EXTRACT"
        recommendation = (
            "High usage — consider offloading large outputs to artifacts "
            "with dump_to_artifact or running compact_now."
        )
    elif ratio >= cfg.prune_threshold:
        tier = "PRUNE"
        recommendation = (
            "Moderate usage — you can continue, but start offloading "
            "large tool outputs to artifacts to stay under budget."
        )
    else:
        tier = "OK"
        recommendation = "Usage is within budget. No action needed."

    return (
        f"Context usage: {pct:.1f}%\n"
        f"Tier: {tier}\n"
        f"Thresholds: prune={cfg.prune_threshold*100:.0f}%, "
        f"extract={cfg.extract_threshold*100:.0f}%, "
        f"compact={cfg.compact_threshold*100:.0f}%\n"
        f"Recommendation: {recommendation}"
    )


@tool
def compact_now() -> str:
    """Request immediate context compaction.

    Use when running low on context budget or before starting a large task
    that will generate significant output.
    """
    trigger = _get_compaction_trigger()
    if trigger is None:
        return "Compaction system not available."
    trigger.request_immediate()
    return "Compaction requested. Will apply at next opportunity."


_NOTES_DIR = Path.home() / ".ag3nt" / "notes"


def _set_notes_dir(path: str) -> None:
    """Override notes directory (for testing)."""
    global _NOTES_DIR
    _NOTES_DIR = Path(path)


def _sanitize_key(key: str) -> str:
    """Sanitize note key to prevent path traversal."""
    safe = re.sub(r"[/\\]", "-", key)
    safe = safe.replace("..", "")
    safe = safe.strip("-. ")
    return safe or "unnamed"


@tool
def write_note(key: str, content: str) -> str:
    """Write a note to your scratchpad.

    Notes persist in ~/.ag3nt/notes/ and are available across sessions.
    Use for tracking decisions, recording findings, or planning next steps.

    Args:
        key: Short identifier (e.g., "research-findings", "next-steps")
        content: The note content (markdown supported)
    """
    _NOTES_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = _sanitize_key(key)
    path = _NOTES_DIR / f"{safe_key}.md"
    path.write_text(content, encoding="utf-8")
    return f"Note '{safe_key}' saved ({len(content)} chars)."


@tool
def read_note(key: str) -> str:
    """Read a note from your scratchpad.

    Args:
        key: The note identifier used when writing
    """
    safe_key = _sanitize_key(key)
    path = _NOTES_DIR / f"{safe_key}.md"
    if not path.exists():
        return f"Note '{safe_key}' not found."
    return path.read_text(encoding="utf-8")


@tool
def list_notes() -> str:
    """List all notes in your scratchpad."""
    if not _NOTES_DIR.exists():
        return "No notes found."

    notes = sorted(_NOTES_DIR.glob("*.md"))
    if not notes:
        return "No notes found."

    lines = [f"- {p.stem}" for p in notes]
    return "Notes:\n" + "\n".join(lines)


def get_context_tools() -> list:
    """Factory function for the tool registry."""
    return [
        dump_to_artifact, read_artifact,
        check_context_budget, compact_now,
        write_note, read_note, list_notes,
    ]
