"""Context-window management tools.

Provides:
- dump_to_artifact / read_artifact — offload large content to the ArtifactStore.
- check_context_budget — query current context-window usage ratio.
- compact_now — request immediate context compaction.
"""
from __future__ import annotations

import logging

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


def get_context_tools() -> list:
    """Factory function for the tool registry."""
    return [dump_to_artifact, read_artifact, check_context_budget, compact_now]
