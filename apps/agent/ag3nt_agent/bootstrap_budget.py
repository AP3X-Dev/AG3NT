"""Bootstrap file budget manager for identity/system prompt files.

Implements per-file and total character budgets with head+tail truncation
(70% head / 20% tail / 10% truncation notice), matching OpenCode's approach.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BudgetConfig:
    """Configuration for bootstrap file budgets."""

    per_file_chars: int = 20_000  # 20KB per file
    total_chars: int = 150_000  # 150KB total
    head_ratio: float = 0.70  # 70% of budget for file head
    tail_ratio: float = 0.20  # 20% of budget for file tail


@dataclass
class _BudgetStats:
    """Internal tracking stats for budget usage."""

    files_loaded: int = 0
    files_truncated: int = 0
    total_chars_used: int = 0
    total_chars_budget: int = 0


class BootstrapBudgetManager:
    """Manages character budgets for bootstrap/identity files.

    Ensures that identity files loaded into the system prompt don't
    consume too much context window. Applies per-file limits and a
    running total limit across all files.
    """

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self._config = config or BudgetConfig()
        self._stats = _BudgetStats(total_chars_budget=self._config.total_chars)

    def apply_budget(self, filename: str, content: str) -> str:
        """Apply per-file and total budget to content.

        Truncates with head+tail strategy if content exceeds limits.
        """
        if not content:
            return ""

        self._stats.files_loaded += 1

        # Calculate effective per-file budget (capped by remaining total budget)
        remaining = self._config.total_chars - self._stats.total_chars_used
        effective_budget = min(self._config.per_file_chars, remaining)

        if effective_budget <= 0:
            self._stats.files_truncated += 1
            return f"[{filename}: skipped — total budget exhausted]"

        if len(content) <= effective_budget:
            self._stats.total_chars_used += len(content)
            return content

        # Head+tail truncation
        self._stats.files_truncated += 1
        truncated = self._head_tail_truncate(content, effective_budget)
        self._stats.total_chars_used += len(truncated)
        return truncated

    def _head_tail_truncate(self, content: str, budget: int) -> str:
        """Truncate content keeping head (70%) and tail (20%), with notice (10%)."""
        notice = "\n\n[... content truncated ...]\n\n"
        notice_len = len(notice)

        usable = budget - notice_len
        if usable <= 0:
            return content[:budget]

        head_chars = int(usable * self._config.head_ratio)
        tail_chars = int(usable * self._config.tail_ratio)

        # Snap to line boundaries where possible
        head = self._snap_to_line_end(content, head_chars)
        tail = self._snap_to_line_start(content, len(content) - tail_chars)

        return head + notice + tail

    @staticmethod
    def _snap_to_line_end(text: str, pos: int) -> str:
        """Snap position to nearest line end (don't cut mid-line)."""
        if pos >= len(text):
            return text
        # Look for newline near pos
        nl = text.rfind("\n", 0, pos + 1)
        if nl > pos * 0.8:  # Only snap if within 20% of target
            return text[: nl + 1]
        return text[:pos]

    @staticmethod
    def _snap_to_line_start(text: str, pos: int) -> str:
        """Snap position to nearest line start."""
        if pos <= 0:
            return text
        nl = text.find("\n", pos)
        tail_size = len(text) - pos
        if nl >= 0 and nl < pos + tail_size * 0.2:
            return text[nl + 1 :]
        return text[pos:]

    def get_stats(self) -> dict:
        """Return budget usage statistics."""
        return {
            "files_loaded": self._stats.files_loaded,
            "files_truncated": self._stats.files_truncated,
            "total_chars_used": self._stats.total_chars_used,
            "total_chars_budget": self._stats.total_chars_budget,
        }

    def reset(self) -> None:
        """Reset budget tracking for a new turn/session."""
        self._stats = _BudgetStats(total_chars_budget=self._config.total_chars)
