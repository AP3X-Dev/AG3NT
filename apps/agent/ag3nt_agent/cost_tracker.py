"""Token cost tracking for AG3NT sessions.

Tracks cumulative token usage and estimated cost per session.
Pricing is based on published Anthropic model pricing.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("ag3nt.cost")

# Pricing per 1M tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
    # Aliases
    "claude-opus-4-6-20250515": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # Older models (still in use)
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
}

# Fallback pricing when model is unknown
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}


@dataclass
class TurnCost:
    """Token usage and cost for a single turn or aggregate."""

    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    estimated_cost_usd: float = 0.0
    timestamp: str = ""


@dataclass
class ToolTiming:
    """Latency data for a single tool call."""

    tool_name: str
    duration_ms: float
    timestamp: str = ""


class CostTracker:
    """Tracks cumulative token usage and estimated cost per session."""

    def __init__(self) -> None:
        self._turns: list[TurnCost] = []
        self._tool_timings: list[ToolTiming] = []
        self._lock = threading.Lock()

    def record_turn(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> TurnCost:
        """Record token usage for a single LLM call.

        Returns the TurnCost for this call.
        """
        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )
        turn = TurnCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            estimated_cost_usd=round(cost, 6),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._turns.append(turn)
        return turn

    def record_tool_timing(self, tool_name: str, duration_ms: float) -> None:
        """Record latency for a tool call."""
        timing = ToolTiming(
            tool_name=tool_name,
            duration_ms=round(duration_ms, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._tool_timings.append(timing)

    def get_session_total(self) -> TurnCost:
        """Get cumulative token usage and cost for the session."""
        with self._lock:
            total_input = sum(t.input_tokens for t in self._turns)
            total_output = sum(t.output_tokens for t in self._turns)
            total_cost = sum(t.estimated_cost_usd for t in self._turns)
        return TurnCost(
            input_tokens=total_input,
            output_tokens=total_output,
            model="mixed" if len(set(t.model for t in self._turns)) > 1 else (self._turns[0].model if self._turns else ""),
            estimated_cost_usd=round(total_cost, 6),
        )

    def get_per_turn_breakdown(self) -> list[TurnCost]:
        """Get per-turn token usage breakdown."""
        with self._lock:
            return list(self._turns)

    def get_tool_timings(self) -> list[ToolTiming]:
        """Get all recorded tool timings."""
        with self._lock:
            return list(self._tool_timings)

    def get_slowest_tools(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Get the N slowest tool calls (name, duration_ms)."""
        with self._lock:
            timings = list(self._tool_timings)
        timings.sort(key=lambda t: t.duration_ms, reverse=True)
        return [(t.tool_name, t.duration_ms) for t in timings[:top_n]]

    def get_avg_tool_latency(self) -> dict[str, float]:
        """Get average latency per tool name."""
        with self._lock:
            timings = list(self._tool_timings)
        buckets: dict[str, list[float]] = {}
        for t in timings:
            buckets.setdefault(t.tool_name, []).append(t.duration_ms)
        return {
            name: round(sum(vals) / len(vals), 2)
            for name, vals in buckets.items()
        }

    @property
    def turn_count(self) -> int:
        """Number of recorded turns."""
        with self._lock:
            return len(self._turns)

    def format_summary(self) -> str:
        """Format a human-readable cost summary."""
        total = self.get_session_total()
        tokens = total.input_tokens + total.output_tokens
        if tokens >= 1_000_000:
            token_str = f"{tokens / 1_000_000:.1f}M"
        elif tokens >= 1_000:
            token_str = f"{tokens / 1_000:.1f}K"
        else:
            token_str = str(tokens)
        return f"${total.estimated_cost_usd:.2f} | {token_str} tokens"

    def reset(self) -> None:
        """Reset all tracking data."""
        with self._lock:
            self._turns.clear()
            self._tool_timings.clear()


# Module-level singleton
_instance: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get or create the singleton cost tracker."""
    global _instance
    if _instance is None:
        _instance = CostTracker()
    return _instance
