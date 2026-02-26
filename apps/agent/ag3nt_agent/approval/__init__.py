"""Tiered approval engine for AG3NT.

Provides a 4-tier approval system for tool calls and other agent actions:

    AUTO            Always allow, no user interaction.
    AUTO_SNAPSHOT   Allow and take a workspace snapshot first.
    SOFT_CONFIRM    Ask the user but allow a quick "yes" (timeout = approve).
    HARD_CONFIRM    Require explicit approval, no auto-timeout.

The engine evaluates requests through a configurable policy pipeline and
supports trust escalation: after a user approves the same tool+context
combination enough times, subsequent requests are auto-approved.

Usage:
    from ag3nt_agent.approval import (
        TieredApprovalEngine,
        ApprovalRequest,
        ApprovalTier,
        ActionType,
    )

    engine = TieredApprovalEngine()
    request = ApprovalRequest(
        action_type=ActionType.TOOL_CALL,
        tool_name="write_file",
        context="src/",
    )
    result = await engine.evaluate(request)
    if result.approved:
        ...  # proceed
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Any

logger = logging.getLogger("ag3nt.approval")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ApprovalTier(IntEnum):
    """Severity tiers for approval decisions (lower = more permissive)."""

    AUTO = 0
    AUTO_SNAPSHOT = 1
    SOFT_CONFIRM = 2
    HARD_CONFIRM = 3


class ActionType(str, Enum):
    """Categories of actions that can be evaluated for approval."""

    TOOL_CALL = "tool_call"
    FILE_WRITE = "file_write"
    SHELL_COMMAND = "shell_command"
    NETWORK_REQUEST = "network_request"
    MEMORY_WRITE = "memory_write"
    CONFIG_CHANGE = "config_change"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequest:
    """Describes an action that requires approval evaluation.

    Parameters
    ----------
    action_type:
        Category of the action (see ``ActionType``).
    tool_name:
        Name of the tool being invoked.
    description:
        Human-readable summary of what the action will do.
    details:
        Arbitrary key-value metadata (e.g., ``{"depth": 3}``).
    context:
        Optional context string (e.g., file path prefix) used for
        trust-key grouping.
    timestamp:
        Unix timestamp of the request; defaults to ``time.time()``.
    """

    action_type: ActionType = ActionType.TOOL_CALL
    tool_name: str = ""
    description: str = ""
    details: dict[str, Any] | None = None
    context: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict."""
        return {
            "action_type": self.action_type.value,
            "tool_name": self.tool_name,
            "description": self.description,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp,
        }


@dataclass
class ApprovalResult:
    """Outcome of an approval evaluation.

    Parameters
    ----------
    approved:
        Whether the action is permitted.
    tier:
        The approval tier that was determined.
    reason:
        Human-readable explanation of the decision.
    approver:
        Who/what approved (e.g., ``"policy:AutoApprovePolicy"``,
        ``"user"``, ``"trust_escalation"``).
    snapshot_created:
        Whether a workspace snapshot was taken before approval.
    """

    approved: bool = False
    tier: ApprovalTier = ApprovalTier.SOFT_CONFIRM
    reason: str = ""
    approver: str = ""
    snapshot_created: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict."""
        return {
            "approved": self.approved,
            "tier": self.tier.name,
            "reason": self.reason,
            "approver": self.approver,
            "snapshot_created": self.snapshot_created,
        }


# ---------------------------------------------------------------------------
# TieredApprovalEngine
# ---------------------------------------------------------------------------

class TieredApprovalEngine:
    """Central approval engine that evaluates requests through a policy pipeline.

    Parameters
    ----------
    bus:
        Optional event bus for emitting approval events.
    trust_tracker:
        Optional ``SessionTrustTracker`` for trust escalation.  If not
        provided, a default tracker is created.
    """

    def __init__(
        self,
        bus: Any | None = None,
        trust_tracker: Any | None = None,
    ) -> None:
        from ag3nt_agent.approval.trust import SessionTrustTracker
        from ag3nt_agent.approval.policies import PolicyPipeline

        self._bus = bus
        self._trust_tracker: SessionTrustTracker = (
            trust_tracker if trust_tracker is not None
            else SessionTrustTracker()
        )
        self._pipeline = PolicyPipeline(trust_tracker=self._trust_tracker)

    @property
    def trust_tracker(self) -> Any:
        """Return the trust tracker instance."""
        return self._trust_tracker

    @property
    def pipeline(self) -> Any:
        """Return the policy pipeline instance."""
        return self._pipeline

    async def evaluate(self, request: ApprovalRequest) -> ApprovalResult:
        """Evaluate an approval request through the policy pipeline.

        Returns an ``ApprovalResult`` indicating whether the action is
        approved and at what tier.

        For ``AUTO`` and ``AUTO_SNAPSHOT`` tiers the action is
        automatically approved.  For ``SOFT_CONFIRM`` and
        ``HARD_CONFIRM`` tiers the action is not approved -- the caller
        is expected to prompt the user and then call
        ``record_user_decision``.
        """
        tier = self._pipeline.evaluate(request)

        # Determine automatic approval
        auto_approved = tier in (ApprovalTier.AUTO, ApprovalTier.AUTO_SNAPSHOT)
        snapshot = tier == ApprovalTier.AUTO_SNAPSHOT

        if auto_approved:
            approver = f"policy:{self._pipeline_decision_source(request)}"
        else:
            approver = ""

        result = ApprovalResult(
            approved=auto_approved,
            tier=tier,
            reason=self._build_reason(request, tier, auto_approved),
            approver=approver,
            snapshot_created=snapshot,
        )

        # Emit event if bus is available
        if self._bus is not None:
            try:
                await self._emit_event(request, result)
            except Exception:
                logger.debug("Failed to emit approval event", exc_info=True)

        logger.info(
            "Approval evaluated: tool=%s tier=%s approved=%s reason=%s",
            request.tool_name,
            tier.name,
            result.approved,
            result.reason,
        )

        return result

    def record_user_decision(
        self, request: ApprovalRequest, approved: bool
    ) -> None:
        """Record the user's approval or denial for trust tracking.

        Should be called after presenting a SOFT_CONFIRM or HARD_CONFIRM
        prompt to the user.
        """
        if approved:
            self._trust_tracker.record_approval(request)
            logger.info(
                "User approved %s (context=%s)", request.tool_name, request.context
            )
        else:
            self._trust_tracker.record_denial(request)
            logger.info(
                "User denied %s (context=%s)", request.tool_name, request.context
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pipeline_decision_source(self, request: ApprovalRequest) -> str:
        """Identify which policy made the decision (for logging)."""
        for policy in self._pipeline.policies:
            tier = policy.evaluate(request)
            if tier is not None:
                return policy.name
        return "default"

    @staticmethod
    def _build_reason(
        request: ApprovalRequest,
        tier: ApprovalTier,
        auto_approved: bool,
    ) -> str:
        """Build a human-readable reason string."""
        if auto_approved:
            if tier == ApprovalTier.AUTO:
                return f"{request.tool_name} auto-approved"
            return f"{request.tool_name} auto-approved with snapshot"
        if tier == ApprovalTier.HARD_CONFIRM:
            return f"{request.tool_name} requires explicit approval"
        return f"{request.tool_name} requires confirmation"

    async def _emit_event(
        self, request: ApprovalRequest, result: ApprovalResult
    ) -> None:
        """Emit an approval event on the bus."""
        payload = {
            "request": request.to_dict(),
            "result": result.to_dict(),
        }
        if hasattr(self._bus, "emit"):
            await self._bus.emit("APPROVAL_EVALUATED", payload)


__all__ = [
    "ApprovalTier",
    "ActionType",
    "ApprovalRequest",
    "ApprovalResult",
    "TieredApprovalEngine",
]
