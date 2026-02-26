"""Session-scoped trust tracking for AG3NT approval system.

Tracks per-tool approval/denial history within a session and provides
trust escalation: once a user has approved the same tool+context combination
enough times, subsequent requests can be auto-approved.

Also supports a temporary "anomaly" state that forces higher approval tiers
for all operations (e.g., after detecting suspicious behaviour).

Usage:
    from ag3nt_agent.approval.trust import SessionTrustTracker

    tracker = SessionTrustTracker(threshold=3)
    tracker.record_approval(request)
    if tracker.is_trusted("write_file", "src/"):
        ...  # auto-approve
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ag3nt_agent.approval import ApprovalRequest

logger = logging.getLogger("ag3nt.approval.trust")


@dataclass
class TrustRecord:
    """Tracks approval and denial counts for a single tool+context pair."""

    approvals: int = 0
    denials: int = 0
    last_approval: float = 0.0
    last_denial: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict."""
        return {
            "approvals": self.approvals,
            "denials": self.denials,
            "last_approval": self.last_approval,
            "last_denial": self.last_denial,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrustRecord:
        """Deserialize from plain dict."""
        return cls(
            approvals=data.get("approvals", 0),
            denials=data.get("denials", 0),
            last_approval=data.get("last_approval", 0.0),
            last_denial=data.get("last_denial", 0.0),
        )


def _trust_key(tool_name: str, context: str | None) -> str:
    """Build a stable lookup key from tool name and optional context."""
    ctx = context or ""
    return f"{tool_name}::{ctx}"


class SessionTrustTracker:
    """Per-session tracker for approval trust escalation.

    Thread-safe via a ``threading.Lock``.  All public methods acquire
    the lock before mutating state.

    Parameters
    ----------
    threshold:
        Number of approvals required before a tool+context pair is
        considered trusted (auto-approve eligible).
    anomaly_cooldown_s:
        Duration in seconds that the anomaly flag stays active after
        ``set_anomaly()`` is called.
    """

    def __init__(
        self,
        threshold: int = 3,
        anomaly_cooldown_s: float = 60.0,
    ) -> None:
        self._threshold = threshold
        self._anomaly_cooldown_s = anomaly_cooldown_s
        self._records: dict[str, TrustRecord] = {}
        self._anomaly_until: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_approval(self, request: ApprovalRequest) -> None:
        """Record that the user approved *request*."""
        key = _trust_key(request.tool_name, request.context)
        now = time.time()
        with self._lock:
            rec = self._records.setdefault(key, TrustRecord())
            rec.approvals += 1
            rec.last_approval = now
        logger.debug(
            "Trust approval recorded for %s (total=%d)", key, rec.approvals
        )

    def record_denial(self, request: ApprovalRequest) -> None:
        """Record that the user denied *request*."""
        key = _trust_key(request.tool_name, request.context)
        now = time.time()
        with self._lock:
            rec = self._records.setdefault(key, TrustRecord())
            rec.denials += 1
            rec.last_denial = now
        logger.debug(
            "Trust denial recorded for %s (total=%d)", key, rec.denials
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def is_trusted(self, tool_name: str, context: str | None = None) -> bool:
        """Return ``True`` if the tool+context has been approved enough times.

        A tool is trusted when its approval count meets or exceeds the
        configured threshold.  If the anomaly flag is active, nothing is
        trusted.
        """
        if self.is_anomaly_active():
            return False
        key = _trust_key(tool_name, context)
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return False
            return rec.approvals >= self._threshold

    # ------------------------------------------------------------------
    # Anomaly management
    # ------------------------------------------------------------------

    def set_anomaly(self) -> None:
        """Activate the anomaly flag for ``anomaly_cooldown_s`` seconds."""
        with self._lock:
            self._anomaly_until = time.time() + self._anomaly_cooldown_s
        logger.warning(
            "Trust anomaly activated for %.0fs", self._anomaly_cooldown_s
        )

    def is_anomaly_active(self) -> bool:
        """Return ``True`` if the anomaly cooldown is still in effect."""
        with self._lock:
            return time.time() < self._anomaly_until

    # ------------------------------------------------------------------
    # Introspection / reset
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return a summary of trust state for debugging/monitoring."""
        with self._lock:
            records_snapshot = {
                k: v.to_dict() for k, v in self._records.items()
            }
            anomaly_remaining = max(0.0, self._anomaly_until - time.time())
        return {
            "threshold": self._threshold,
            "anomaly_cooldown_s": self._anomaly_cooldown_s,
            "anomaly_active": anomaly_remaining > 0,
            "anomaly_remaining_s": round(anomaly_remaining, 1),
            "records": records_snapshot,
            "total_records": len(records_snapshot),
        }

    def reset(self) -> None:
        """Clear all trust records and cancel any active anomaly."""
        with self._lock:
            self._records.clear()
            self._anomaly_until = 0.0
        logger.info("Trust tracker reset")
