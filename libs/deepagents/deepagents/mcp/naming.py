"""Tool naming utilities for MCP integration.

Handles tool name prefixing and collision detection to ensure
deterministic and unique tool names across multiple MCP servers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ToolNameInfo:
    """Information about a tool's naming."""

    original_name: str
    """Original tool name from MCP server."""

    server_name: str
    """Name of the MCP server providing this tool."""

    prefixed_name: str
    """Final tool name (may be prefixed or original)."""

    prefix: str
    """Prefix applied (empty string if none)."""


def normalize_tool_name(name: str) -> str:
    """Normalize a tool name for consistent comparison.

    Converts to lowercase and replaces invalid characters.

    Args:
        name: The tool name to normalize.

    Returns:
        Normalized tool name suitable for comparison.
    """
    # Convert to lowercase
    normalized = name.lower()

    # Replace any non-alphanumeric characters (except underscore) with underscore
    normalized = re.sub(r"[^a-z0-9_]", "_", normalized)

    # Remove consecutive underscores
    normalized = re.sub(r"_+", "_", normalized)

    # Remove leading/trailing underscores
    return normalized.strip("_")


def create_prefixed_name(server_name: str, tool_name: str, separator: str = "_") -> str:
    """Create a prefixed tool name.

    Args:
        server_name: The server name to use as prefix.
        tool_name: The original tool name.
        separator: Separator between prefix and name (default: underscore).

    Returns:
        Prefixed tool name in format: {server_name}{separator}{tool_name}
    """
    prefix = normalize_tool_name(server_name)
    normalized_tool = normalize_tool_name(tool_name)
    return f"{prefix}{separator}{normalized_tool}"


class ToolNameRegistry:
    """Registry for tracking tool names and detecting collisions.

    Ensures that all tool names are unique across MCP servers.
    Collisions are detected at agent build time.
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        # Map from final tool name -> ToolNameInfo
        self._tools: dict[str, ToolNameInfo] = {}

    def register_tool(
        self,
        original_name: str,
        server_name: str,
        prefix: str = "",
    ) -> ToolNameInfo:
        """Register a tool and check for collisions.

        Args:
            original_name: Original tool name from MCP server.
            server_name: Name of the MCP server.
            prefix: Prefix to apply (empty string for no prefix).

        Returns:
            ToolNameInfo with the final name.

        Raises:
            ValueError: If the tool name collides with an existing tool.
        """
        if prefix:
            prefixed_name = create_prefixed_name(prefix, original_name)
        else:
            prefixed_name = normalize_tool_name(original_name)

        # Check for collision
        if prefixed_name in self._tools:
            existing = self._tools[prefixed_name]
            msg = (
                f"Tool name collision: '{prefixed_name}' would be registered by "
                f"server '{server_name}' (tool: '{original_name}'), but it's already "
                f"registered by server '{existing.server_name}' (tool: '{existing.original_name}'). "
                f"Enable tool_name_prefix or use a custom name_prefix to resolve."
            )
            raise ValueError(msg)

        info = ToolNameInfo(
            original_name=original_name,
            server_name=server_name,
            prefixed_name=prefixed_name,
            prefix=prefix,
        )
        self._tools[prefixed_name] = info
        return info

    def get_all_tools(self) -> dict[str, ToolNameInfo]:
        """Get all registered tools.

        Returns:
            Dict mapping prefixed names to ToolNameInfo.
        """
        return dict(self._tools)

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The prefixed tool name to check.

        Returns:
            True if the tool is registered.
        """
        return name in self._tools
