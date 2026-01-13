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

from deepagents.approval.policy import (
    ApprovalPolicy,
    ToolRiskLevel,
    DEFAULT_TOOL_CLASSIFICATIONS,
)
from deepagents.approval.ledger import ApprovalLedger, ApprovalRecord

__all__ = [
    "ApprovalPolicy",
    "ToolRiskLevel",
    "DEFAULT_TOOL_CLASSIFICATIONS",
    "ApprovalLedger",
    "ApprovalRecord",
]

