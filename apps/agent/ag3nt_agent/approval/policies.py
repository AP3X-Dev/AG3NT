"""Five-policy approval chain for AG3NT.

Each policy inspects an ``ApprovalRequest`` and either returns a definitive
``ApprovalTier`` or ``None`` to defer to the next policy in the pipeline.

Policy order (first definitive answer wins):
    1. OwnerOnlyPolicy   -- HARD_CONFIRM for destructive tools
    2. GroupPolicy        -- placeholder (always defers)
    3. DepthLimitedPolicy -- SOFT_CONFIRM when call depth >= threshold
    4. AnomalyAwarePolicy -- HARD_CONFIRM when trust anomaly is active
    5. AutoApprovePolicy  -- AUTO for read-only tools; AUTO if trusted; else SOFT_CONFIRM

Usage:
    from ag3nt_agent.approval.policies import PolicyPipeline
    from ag3nt_agent.approval.trust import SessionTrustTracker

    pipeline = PolicyPipeline(trust_tracker=tracker)
    tier = pipeline.evaluate(request)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ag3nt_agent.approval import ApprovalTier

if TYPE_CHECKING:
    from ag3nt_agent.approval import ApprovalRequest
    from ag3nt_agent.approval.trust import SessionTrustTracker

logger = logging.getLogger("ag3nt.approval.policies")


# ---------------------------------------------------------------------------
# Destructive tool set used by OwnerOnlyPolicy
# ---------------------------------------------------------------------------

DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({
    "delete_file",
    "exec_command",
    "git_push",
    "send_message",
    "config_change",
})

# ---------------------------------------------------------------------------
# Read-only tool set used by AutoApprovePolicy
# ---------------------------------------------------------------------------

READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "glob_tool",
    "grep_tool",
    "codebase_search_tool",
    "memory_search",
    "list_directory",
    "read_directory",
    "lsp_tool",
    "deep_reasoning",
    "internet_search",
    "web_search",
})


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ApprovalPolicy(ABC):
    """Base class for a single approval policy."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def evaluate(self, request: ApprovalRequest) -> ApprovalTier | None:
        """Return a tier if this policy has a definitive opinion, else None."""


# ---------------------------------------------------------------------------
# 1. OwnerOnlyPolicy
# ---------------------------------------------------------------------------

class OwnerOnlyPolicy(ApprovalPolicy):
    """Require HARD_CONFIRM for destructive/high-risk tools.

    Any tool in ``DESTRUCTIVE_TOOLS`` must be explicitly approved by
    the session owner.
    """

    def evaluate(self, request: ApprovalRequest) -> ApprovalTier | None:
        if request.tool_name in DESTRUCTIVE_TOOLS:
            logger.debug(
                "OwnerOnlyPolicy: %s is destructive -> HARD_CONFIRM",
                request.tool_name,
            )
            return ApprovalTier.HARD_CONFIRM
        return None


# ---------------------------------------------------------------------------
# 2. GroupPolicy (placeholder)
# ---------------------------------------------------------------------------

class GroupPolicy(ApprovalPolicy):
    """Placeholder for future group/role-based approval logic.

    Always defers to the next policy.
    """

    def evaluate(self, request: ApprovalRequest) -> ApprovalTier | None:
        return None


# ---------------------------------------------------------------------------
# 3. DepthLimitedPolicy
# ---------------------------------------------------------------------------

class DepthLimitedPolicy(ApprovalPolicy):
    """Require confirmation when the agent call depth is high.

    If ``request.details.get("depth")`` is at or above ``max_depth``,
    a SOFT_CONFIRM is returned to ask the user before continuing.

    Parameters
    ----------
    max_depth:
        Depth threshold at which confirmation is required.
    """

    def __init__(self, max_depth: int = 5) -> None:
        self._max_depth = max_depth

    def evaluate(self, request: ApprovalRequest) -> ApprovalTier | None:
        details = request.details or {}
        depth = details.get("depth", 0)
        if isinstance(depth, int) and depth >= self._max_depth:
            logger.debug(
                "DepthLimitedPolicy: depth %d >= %d -> SOFT_CONFIRM",
                depth,
                self._max_depth,
            )
            return ApprovalTier.SOFT_CONFIRM
        return None


# ---------------------------------------------------------------------------
# 4. AnomalyAwarePolicy
# ---------------------------------------------------------------------------

class AnomalyAwarePolicy(ApprovalPolicy):
    """Escalate to HARD_CONFIRM while a trust anomaly is active.

    Parameters
    ----------
    trust_tracker:
        The ``SessionTrustTracker`` to query for anomaly state.
    """

    def __init__(self, trust_tracker: SessionTrustTracker | None = None) -> None:
        self._trust_tracker = trust_tracker

    def evaluate(self, request: ApprovalRequest) -> ApprovalTier | None:
        if self._trust_tracker and self._trust_tracker.is_anomaly_active():
            logger.debug(
                "AnomalyAwarePolicy: anomaly active -> HARD_CONFIRM"
            )
            return ApprovalTier.HARD_CONFIRM
        return None


# ---------------------------------------------------------------------------
# 5. AutoApprovePolicy
# ---------------------------------------------------------------------------

class AutoApprovePolicy(ApprovalPolicy):
    """Default policy: auto-approve read-only and trusted tools.

    Decision tree:
        - Read-only tool?  -> AUTO
        - Trusted (enough past approvals)?  -> AUTO
        - Otherwise  -> SOFT_CONFIRM

    Parameters
    ----------
    trust_tracker:
        The ``SessionTrustTracker`` to query for trust escalation.
    """

    def __init__(self, trust_tracker: SessionTrustTracker | None = None) -> None:
        self._trust_tracker = trust_tracker

    def evaluate(self, request: ApprovalRequest) -> ApprovalTier | None:
        # Read-only tools are always safe
        if request.tool_name in READ_ONLY_TOOLS:
            logger.debug(
                "AutoApprovePolicy: %s is read-only -> AUTO",
                request.tool_name,
            )
            return ApprovalTier.AUTO

        # Check trust escalation
        if self._trust_tracker and self._trust_tracker.is_trusted(
            request.tool_name, request.context
        ):
            logger.debug(
                "AutoApprovePolicy: %s trusted in context %r -> AUTO",
                request.tool_name,
                request.context,
            )
            return ApprovalTier.AUTO

        # Fallback: require soft confirmation
        logger.debug(
            "AutoApprovePolicy: %s not trusted -> SOFT_CONFIRM",
            request.tool_name,
        )
        return ApprovalTier.SOFT_CONFIRM


# ---------------------------------------------------------------------------
# PolicyPipeline
# ---------------------------------------------------------------------------

class PolicyPipeline:
    """Runs an ordered list of policies; first definitive tier wins.

    Parameters
    ----------
    trust_tracker:
        Shared ``SessionTrustTracker`` injected into policies that need it.
    policies:
        Optional override of the default 5-policy chain.
    """

    def __init__(
        self,
        trust_tracker: SessionTrustTracker | None = None,
        policies: list[ApprovalPolicy] | None = None,
    ) -> None:
        self._trust_tracker = trust_tracker
        if policies is not None:
            self._policies = policies
        else:
            self._policies = [
                OwnerOnlyPolicy(),
                GroupPolicy(),
                DepthLimitedPolicy(),
                AnomalyAwarePolicy(trust_tracker=trust_tracker),
                AutoApprovePolicy(trust_tracker=trust_tracker),
            ]

    @property
    def policies(self) -> list[ApprovalPolicy]:
        """Return the ordered policy list (read-only view)."""
        return list(self._policies)

    def evaluate(self, request: ApprovalRequest) -> ApprovalTier:
        """Evaluate *request* through the policy chain.

        Returns the tier from the first policy that has a definitive
        opinion.  If no policy decides, defaults to ``SOFT_CONFIRM``.
        """
        for policy in self._policies:
            tier = policy.evaluate(request)
            if tier is not None:
                logger.debug(
                    "PolicyPipeline: %s decided %s for %s",
                    policy.name,
                    tier.name,
                    request.tool_name,
                )
                return tier

        # Should not happen with the default chain, but be safe
        logger.warning(
            "PolicyPipeline: no policy decided for %s, defaulting to SOFT_CONFIRM",
            request.tool_name,
        )
        return ApprovalTier.SOFT_CONFIRM
