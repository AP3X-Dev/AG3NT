"""Subagent Containment - Contracts and config propagation for subagents.

This module provides:
1. DistilledReturnContract - Schema for subagent return values
2. SubagentConfig - Runtime configuration to propagate to subagents
3. ContainedSubAgentMiddleware - Extended SubAgentMiddleware with containment

Usage:
    from deepagents.subagent import ContainedSubAgentMiddleware, SubagentConfig

    middleware = ContainedSubAgentMiddleware(
        default_model="anthropic:claude-sonnet",
        config=SubagentConfig(
            max_output_tokens=2000,
            require_structured_output=True,
        ),
    )
"""

from deepagents.subagent.containment import (
    ContainedSubAgentMiddleware,
    DistilledReturnContract,
    SubagentConfig,
)

__all__ = [
    "ContainedSubAgentMiddleware",
    "DistilledReturnContract",
    "SubagentConfig",
]

