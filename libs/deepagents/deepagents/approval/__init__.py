"""Approval Policy - Centralized tool risk classification and approval management.

This module provides:
1. ToolRiskLevel enum for classifying tools
2. ApprovalPolicy for defining which tools need approval
3. ApprovalLedger for tracking approval decisions
4. Helper to generate interrupt_on config from policy

Usage:
    from deepagents.approval import ApprovalPolicy, ToolRiskLevel

    policy = ApprovalPolicy()
    interrupt_on = policy.to_interrupt_on_config()

    agent = create_deep_agent(
        middleware=[HumanInTheLoopMiddleware(interrupt_on=interrupt_on)]
    )
"""

from deepagents.approval.ledger import ApprovalLedger, ApprovalRecord
from deepagents.approval.policy import (
    DEFAULT_TOOL_CLASSIFICATIONS,
    ApprovalPolicy,
    ToolRiskLevel,
)

__all__ = [
    "DEFAULT_TOOL_CLASSIFICATIONS",
    "ApprovalLedger",
    "ApprovalPolicy",
    "ApprovalRecord",
    "ToolRiskLevel",
]
