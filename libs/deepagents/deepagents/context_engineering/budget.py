"""Token budget tracking for context engineering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents.context_engineering.config import ContextEngineeringConfig

logger = logging.getLogger(__name__)


@dataclass
class TokenBudgetReport:
    """Report of token usage across context components."""

    total_budget: int
    """Total available tokens (model context limit)."""

    used_tokens: int
    """Total tokens currently used."""

    available_tokens: int
    """Remaining tokens available."""

    breakdown: dict[str, int] = field(default_factory=dict)
    """Token usage by component (e.g., messages, artifacts, system_prompt)."""

    summarization_recommended: bool = False
    """Whether summarization is recommended based on usage."""

    @property
    def usage_ratio(self) -> float:
        """Get ratio of used to total tokens."""
        return self.used_tokens / self.total_budget if self.total_budget > 0 else 0.0


class TokenBudgetTracker:
    """Tracks token usage across context components.

    Provides:
    1. Real-time token counting per component
    2. Budget reports with recommendations
    3. Summarization trigger detection
    """

    def __init__(self, config: ContextEngineeringConfig) -> None:
        self.config = config
        self._component_tokens: dict[str, int] = {}
        self._step_count: int = 0

    def update_component(self, component: str, token_count: int) -> None:
        """Update token count for a component.

        Args:
            component: Component name (e.g., 'messages', 'system_prompt').
            token_count: Current token count for this component.
        """
        self._component_tokens[component] = token_count

    def get_total_tokens(self) -> int:
        """Get total tokens across all components."""
        return sum(self._component_tokens.values())

    def should_summarize(self) -> bool:
        """Check if summarization should be triggered."""
        total = self.get_total_tokens()
        trigger = self.config.get_summarization_trigger_tokens()
        return total >= trigger

    def get_report(self) -> TokenBudgetReport:
        """Generate a token budget report."""
        total = self.get_total_tokens()
        limit = self.config.model_context_limit

        return TokenBudgetReport(
            total_budget=limit,
            used_tokens=total,
            available_tokens=max(0, limit - total),
            breakdown=dict(self._component_tokens),
            summarization_recommended=self.should_summarize(),
        )

    def log_report(self) -> None:
        """Log current budget report."""
        report = self.get_report()
        logger.info(
            f"Token budget: {report.used_tokens:,}/{report.total_budget:,} "
            f"({report.usage_ratio:.1%}) - "
            f"{'SUMMARIZE RECOMMENDED' if report.summarization_recommended else 'OK'}"
        )
        for component, tokens in report.breakdown.items():
            logger.debug(f"  {component}: {tokens:,} tokens")

    def increment_step(self) -> None:
        """Increment step counter."""
        self._step_count += 1

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count
