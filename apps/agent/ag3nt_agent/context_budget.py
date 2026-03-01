"""Proactive context budget tracker.

Monitors accumulated token usage across system prompt, messages, and tool results.
Provides status levels (green/yellow/red) and suggested output limits to prevent
context overflow before it happens.

TODO: Wire into deepagents_runtime.py to track token usage per turn and inject
budget_report() into system prompt when status is YELLOW or RED.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BudgetStatus(str, Enum):
    GREEN = "green"    # < 60% — no concern
    YELLOW = "yellow"  # 60-85% — getting tight, suggest shorter outputs
    RED = "red"        # > 85% — critical, compaction imminent


@dataclass
class ContextBudgetTracker:
    """Tracks token usage across context window components."""

    max_tokens: int = 200_000
    _usage: dict[str, int] = field(default_factory=dict)

    # Thresholds
    YELLOW_THRESHOLD: float = 0.60
    RED_THRESHOLD: float = 0.85

    def record(self, component: str, tokens: int) -> None:
        """Record token usage for a named component."""
        self._usage[component] = self._usage.get(component, 0) + tokens

    def total_tokens(self) -> int:
        """Total tokens used across all components."""
        return sum(self._usage.values())

    def remaining(self) -> int:
        """Tokens remaining in budget."""
        return max(0, self.max_tokens - self.total_tokens())

    def status(self) -> BudgetStatus:
        """Current budget status based on usage ratio."""
        ratio = self.total_tokens() / self.max_tokens if self.max_tokens > 0 else 1.0
        if ratio >= self.RED_THRESHOLD:
            return BudgetStatus.RED
        if ratio >= self.YELLOW_THRESHOLD:
            return BudgetStatus.YELLOW
        return BudgetStatus.GREEN

    def suggested_max_tool_output(self) -> int:
        """Suggested maximum tokens for the next tool output.

        Returns a fraction of remaining tokens to leave room for
        the model's response and future tool calls.
        """
        remaining = self.remaining()
        # Reserve 30% of remaining for model response
        return max(500, int(remaining * 0.70))

    def budget_report(self) -> str:
        """Human-readable budget report for injection into context."""
        total = self.total_tokens()
        status = self.status()
        parts = [f"Context: {total:,}/{self.max_tokens:,} tokens ({status.value})"]
        if status == BudgetStatus.YELLOW:
            parts.append("Consider using concise outputs.")
        elif status == BudgetStatus.RED:
            parts.append("Context nearly full — compaction imminent.")
        return " | ".join(parts)

    def reset(self) -> None:
        """Reset all usage tracking."""
        self._usage.clear()
