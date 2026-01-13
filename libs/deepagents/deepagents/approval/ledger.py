"""Approval Ledger - Audit trail for approval decisions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRecord:
    """Record of a single approval decision.
    
    Attributes:
        tool_name: Name of the tool that required approval.
        tool_args: Arguments passed to the tool.
        decision: The approval decision made.
        timestamp: When the decision was made.
        session_id: Optional session identifier.
        reason: Optional reason for the decision.
    """
    
    tool_name: str
    tool_args: dict[str, Any]
    decision: Literal["approve", "reject", "edit", "auto"]
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "decision": self.decision,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalRecord":
        """Create from dictionary."""
        return cls(
            tool_name=data["tool_name"],
            tool_args=data["tool_args"],
            decision=data["decision"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data.get("session_id"),
            reason=data.get("reason"),
        )


class ApprovalLedger:
    """Ledger for tracking approval decisions.
    
    Provides:
    1. Persistent storage of approval decisions
    2. Query interface for audit trails
    3. Statistics on approval patterns
    """

    def __init__(self, ledger_path: Path | None = None) -> None:
        """Initialize ledger.
        
        Args:
            ledger_path: Path to ledger file. If None, uses in-memory only.
        """
        self.ledger_path = ledger_path
        self._records: list[ApprovalRecord] = []
        
        if ledger_path and ledger_path.exists():
            self._load()

    def record(self, record: ApprovalRecord) -> None:
        """Record an approval decision."""
        self._records.append(record)
        if self.ledger_path:
            self._append_to_file(record)
        logger.debug(f"Recorded {record.decision} for {record.tool_name}")

    def get_records(
        self,
        tool_name: str | None = None,
        decision: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
    ) -> list[ApprovalRecord]:
        """Query records with optional filters."""
        results = self._records
        
        if tool_name:
            results = [r for r in results if r.tool_name == tool_name]
        if decision:
            results = [r for r in results if r.decision == decision]
        if session_id:
            results = [r for r in results if r.session_id == session_id]
        if since:
            results = [r for r in results if r.timestamp >= since]
            
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get statistics on approval patterns."""
        if not self._records:
            return {"total": 0}
            
        by_tool: dict[str, dict[str, int]] = {}
        by_decision: dict[str, int] = {}
        
        for r in self._records:
            by_decision[r.decision] = by_decision.get(r.decision, 0) + 1
            if r.tool_name not in by_tool:
                by_tool[r.tool_name] = {}
            by_tool[r.tool_name][r.decision] = by_tool[r.tool_name].get(r.decision, 0) + 1
        
        return {
            "total": len(self._records),
            "by_decision": by_decision,
            "by_tool": by_tool,
        }

    def _load(self) -> None:
        """Load records from file."""
        if not self.ledger_path or not self.ledger_path.exists():
            return
        try:
            with open(self.ledger_path) as f:
                for line in f:
                    data = json.loads(line.strip())
                    self._records.append(ApprovalRecord.from_dict(data))
        except Exception as e:
            logger.warning(f"Failed to load ledger: {e}")

    def _append_to_file(self, record: ApprovalRecord) -> None:
        """Append a record to the ledger file."""
        if not self.ledger_path:
            return
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write to ledger: {e}")

