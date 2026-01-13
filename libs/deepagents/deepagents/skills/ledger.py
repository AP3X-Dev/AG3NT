"""Skill Usage Ledger for tracking skill usage.

The ledger records all skill usage events for auditing,
metrics, and debugging purposes.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.skills.models import SkillUsageRecord

if TYPE_CHECKING:
    from deepagents.skills.config import SkillsConfig

logger = logging.getLogger(__name__)


class SkillUsageLedger:
    """Ledger for tracking skill usage.

    Records all skill usage events to a JSONL file and
    maintains in-memory statistics.
    """

    def __init__(self, config: "SkillsConfig"):
        """Initialize the ledger.

        Args:
            config: Skills configuration.
        """
        self.config = config
        self._records: list[SkillUsageRecord] = []
        self._stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record(self, record: SkillUsageRecord) -> None:
        """Record a skill usage event.

        Args:
            record: Usage record to log.
        """
        self._records.append(record)

        # Update stats
        self._stats[record.skill_id]["total"] += 1
        self._stats[record.skill_id][record.action] += 1

        # Persist to file
        if self.config.enable_metrics:
            self._append_to_file(record)

        logger.debug(f"Recorded skill usage: {record.skill_id} - {record.action}")

    def _append_to_file(self, record: SkillUsageRecord) -> None:
        """Append record to ledger file.

        Args:
            record: Record to append.
        """
        try:
            ledger_path = self.config.get_ledger_path()
            with open(ledger_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to write to ledger: {e}")

    def get_records(
        self,
        skill_id: str | None = None,
        action: str | None = None,
        limit: int | None = None,
    ) -> list[SkillUsageRecord]:
        """Get usage records with optional filtering.

        Args:
            skill_id: Filter by skill ID.
            action: Filter by action type.
            limit: Maximum records to return.

        Returns:
            List of matching records.
        """
        records = self._records

        if skill_id:
            records = [r for r in records if r.skill_id == skill_id]

        if action:
            records = [r for r in records if r.action == action]

        if limit:
            records = records[-limit:]

        return records

    def get_stats(self, skill_id: str | None = None) -> dict[str, dict[str, int]]:
        """Get usage statistics.

        Args:
            skill_id: Filter by skill ID.

        Returns:
            Dict of skill_id -> action -> count.
        """
        if skill_id:
            return {skill_id: dict(self._stats.get(skill_id, {}))}
        return {k: dict(v) for k, v in self._stats.items()}

    def get_blocked_tools_report(self) -> dict[str, list[str]]:
        """Get report of tools blocked by skills.

        Returns:
            Dict of skill_id -> list of blocked tools.
        """
        report: dict[str, list[str]] = {}
        for record in self._records:
            if record.blocked_tools:
                if record.skill_id not in report:
                    report[record.skill_id] = []
                report[record.skill_id].extend(record.blocked_tools)
        return report

    def load_from_file(self) -> int:
        """Load records from ledger file.

        Returns:
            Number of records loaded.
        """
        ledger_path = self.config.get_ledger_path()
        if not ledger_path.exists():
            return 0

        count = 0
        with open(ledger_path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = SkillUsageRecord.model_validate_json(line.strip())
                    self._records.append(record)
                    self._stats[record.skill_id]["total"] += 1
                    self._stats[record.skill_id][record.action] += 1
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to parse ledger line: {e}")

        return count

    def clear(self) -> None:
        """Clear in-memory records and stats."""
        self._records.clear()
        self._stats.clear()

    def export_metrics(self) -> dict:
        """Export metrics summary.

        Returns:
            Dict with metrics summary.
        """
        return {
            "total_records": len(self._records),
            "skills_used": len(self._stats),
            "stats": self.get_stats(),
            "blocked_tools": self.get_blocked_tools_report(),
        }

