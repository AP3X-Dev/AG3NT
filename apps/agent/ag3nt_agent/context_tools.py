"""Context-window management tools: dump_to_artifact & read_artifact.

These tools expose the ArtifactStore to the agent so it can offload
large content out of context and retrieve it later by ID.
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from ag3nt_agent.artifact_store import get_artifact_store

logger = logging.getLogger(__name__)


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


def get_context_tools() -> list:
    """Factory function for the tool registry."""
    return [dump_to_artifact, read_artifact]
