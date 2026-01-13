"""Approval Policy - Tool risk classification and policy configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain.agents.middleware import InterruptOnConfig


class ToolRiskLevel(IntEnum):
    """Risk levels for tool operations.

    Higher values = more dangerous, more likely to need approval.
    """

    SAFE = 0  # Read-only, no side effects
    LOW = 1  # Minor side effects, easily reversible
    MEDIUM = 2  # Writes data, may need review
    HIGH = 3  # Destructive, external calls, irreversible
    CRITICAL = 4  # System-level, security-sensitive


# Default tool risk classifications
DEFAULT_TOOL_CLASSIFICATIONS: dict[str, ToolRiskLevel] = {
    # SAFE - read-only operations
    "read_file": ToolRiskLevel.SAFE,
    "ls": ToolRiskLevel.SAFE,
    "glob": ToolRiskLevel.SAFE,
    "grep": ToolRiskLevel.SAFE,
    "read_todos": ToolRiskLevel.SAFE,
    "read_artifact": ToolRiskLevel.SAFE,
    "search_artifacts": ToolRiskLevel.SAFE,
    "retrieve_snippets": ToolRiskLevel.SAFE,
    # LOW - minor effects
    "write_todos": ToolRiskLevel.LOW,
    "save_artifact": ToolRiskLevel.LOW,
    # MEDIUM - writes files
    "write_file": ToolRiskLevel.MEDIUM,
    "edit_file": ToolRiskLevel.MEDIUM,
    # HIGH - external calls, subagents
    "web_search": ToolRiskLevel.HIGH,
    "fetch_url": ToolRiskLevel.HIGH,
    "read_web_page": ToolRiskLevel.HIGH,
    "task": ToolRiskLevel.HIGH,
    "generate_image": ToolRiskLevel.HIGH,
    # CRITICAL - shell access
    "shell": ToolRiskLevel.CRITICAL,
    "execute": ToolRiskLevel.CRITICAL,
}


@dataclass
class ApprovalPolicy:
    """Policy defining which tools require human approval.

    Attributes:
        min_risk_for_approval: Minimum risk level that requires approval.
        tool_classifications: Override tool risk classifications.
        always_approve: Tools that always require approval regardless of risk.
        never_approve: Tools that never require approval regardless of risk.
        description_formatters: Custom description formatters per tool.
    """

    min_risk_for_approval: ToolRiskLevel = ToolRiskLevel.MEDIUM
    """Tools at or above this risk level require approval."""

    tool_classifications: dict[str, ToolRiskLevel] = field(default_factory=dict)
    """Override default tool classifications."""

    always_approve: set[str] = field(default_factory=set)
    """Tools that always require approval."""

    never_approve: set[str] = field(default_factory=set)
    """Tools that never require approval (unsafe, use with care)."""

    description_formatters: dict[str, Callable[[dict[str, Any]], str]] = field(default_factory=dict)
    """Custom formatters for approval descriptions."""

    def get_risk_level(self, tool_name: str) -> ToolRiskLevel:
        """Get risk level for a tool."""
        if tool_name in self.tool_classifications:
            return self.tool_classifications[tool_name]
        return DEFAULT_TOOL_CLASSIFICATIONS.get(tool_name, ToolRiskLevel.MEDIUM)

    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires approval."""
        if tool_name in self.never_approve:
            return False
        if tool_name in self.always_approve:
            return True
        return self.get_risk_level(tool_name) >= self.min_risk_for_approval

    def to_interrupt_on_config(self) -> dict[str, InterruptOnConfig]:
        """Convert policy to HumanInTheLoopMiddleware interrupt_on config."""
        config: dict[str, InterruptOnConfig] = {}

        # Get all known tools
        all_tools = set(DEFAULT_TOOL_CLASSIFICATIONS.keys())
        all_tools.update(self.tool_classifications.keys())
        all_tools.update(self.always_approve)

        for tool_name in all_tools:
            if not self.requires_approval(tool_name):
                continue

            tool_config: InterruptOnConfig = {
                "allowed_decisions": ["approve", "reject"],
            }

            if tool_name in self.description_formatters:
                tool_config["description"] = self.description_formatters[tool_name]

            config[tool_name] = tool_config

        return config

    def get_tools_by_risk(self, risk_level: ToolRiskLevel) -> list[str]:
        """Get all tools at a specific risk level."""
        return [name for name in DEFAULT_TOOL_CLASSIFICATIONS if self.get_risk_level(name) == risk_level]
