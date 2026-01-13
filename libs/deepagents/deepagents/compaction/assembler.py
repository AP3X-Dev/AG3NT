"""ContextAssembler for building final prompt context with budgets.

This module assembles the final context from various blocks:
- Working memory (highest priority)
- Plan state
- Decision ledger
- Reasoning state summaries
- Recent observations
- Masked placeholders
- Retrieved snippets

Each block has a configurable token budget.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents.compaction.config import CompactionConfig
    from deepagents.compaction.models import MaskedObservationPlaceholder, ReasoningState
    from deepagents.compaction.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ContextBlock:
    """A block of context with metadata."""

    name: str
    content: str
    priority: int
    token_estimate: int
    source: str = ""


@dataclass
class AssembledContext:
    """The assembled context ready for prompt inclusion."""

    blocks: list[ContextBlock]
    total_tokens: int
    budget_used: dict[str, int]
    blocks_truncated: list[str]
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert assembled context to text for prompt inclusion."""
        parts = []
        for block in sorted(self.blocks, key=lambda b: b.priority):
            if block.content.strip():
                parts.append(f"## {block.name}\n{block.content}")
        return "\n\n".join(parts)


class ContextAssembler:
    """Assembles context from various sources with budget management.

    Prioritizes blocks based on configuration and fits content within
    the total context budget.

    Args:
        config: Compaction configuration with budget settings.
    """

    def __init__(self, config: CompactionConfig) -> None:
        self.config = config
        self._assembly_count = 0

    def _truncate_to_budget(
        self,
        content: str,
        budget_tokens: int,
    ) -> tuple[str, bool]:
        """Truncate content to fit within token budget.

        Args:
            content: The content to truncate.
            budget_tokens: Maximum tokens allowed.

        Returns:
            Tuple of (truncated_content, was_truncated).
        """
        estimated = self.config.estimate_tokens(content)
        if estimated <= budget_tokens:
            return content, False

        # Truncate by characters (rough approximation)
        target_chars = budget_tokens * 4  # ~4 chars per token
        truncated = content[:target_chars]

        # Try to truncate at a sentence or line boundary
        last_newline = truncated.rfind("\n")
        last_period = truncated.rfind(". ")

        if last_newline > target_chars * 0.7:
            truncated = truncated[:last_newline]
        elif last_period > target_chars * 0.7:
            truncated = truncated[: last_period + 1]

        truncated += "\n[... truncated ...]"
        return truncated, True

    def _format_reasoning_state(
        self,
        state: ReasoningState,
    ) -> str:
        """Format a reasoning state for context inclusion."""
        lines = [
            f"*Summary at step {state.step_number}*",
            "",
            state.executive_summary,
        ]

        if state.confirmed_facts:
            lines.append("")
            lines.append("**Confirmed:**")
            for fact in state.confirmed_facts[:5]:
                lines.append(f"- {fact}")

        if state.hypotheses:
            lines.append("")
            lines.append("**Hypotheses:**")
            for hyp in state.hypotheses[:3]:
                lines.append(f"- {hyp}")

        if state.open_questions:
            lines.append("")
            lines.append("**Open Questions:**")
            for q in state.open_questions[:3]:
                lines.append(f"- {q}")

        return "\n".join(lines)

    def _format_placeholders(
        self,
        placeholders: list[MaskedObservationPlaceholder],
    ) -> str:
        """Format masked observation placeholders."""
        if not placeholders:
            return ""

        lines = [f"*{len(placeholders)} large outputs stored as artifacts:*", ""]
        for p in placeholders[-10:]:  # Show last 10
            lines.append(f"- **{p.artifact_id}** ({p.tool_name}): {p.digest[:100]}")

        return "\n".join(lines)

    def assemble(
        self,
        *,
        working_memory: str = "",
        plan_state: str = "",
        decision_ledger: str = "",
        reasoning_states: list[ReasoningState] | None = None,
        recent_observations: str = "",
        placeholders: list[MaskedObservationPlaceholder] | None = None,
        retrieved_snippets: list[RetrievalResult] | None = None,
        total_budget: int | None = None,
    ) -> AssembledContext:
        """Assemble context from various sources.

        Args:
            working_memory: Current working memory content.
            plan_state: Current plan/todo state.
            decision_ledger: Decision history.
            reasoning_states: List of reasoning state summaries.
            recent_observations: Recent unmasked observations.
            placeholders: Masked observation placeholders.
            retrieved_snippets: Retrieved snippet results.
            total_budget: Total token budget (defaults to config).

        Returns:
            AssembledContext with prioritized blocks.
        """
        self._assembly_count += 1

        if total_budget is None:
            total_budget = int(self.config.model_context_limit * 0.3)  # 30% for compacted context

        blocks: list[ContextBlock] = []
        budget_used: dict[str, int] = {}
        blocks_truncated: list[str] = []
        remaining_budget = total_budget

        # Define blocks with their budgets and priorities
        block_specs = [
            ("working_memory", working_memory, self.config.working_memory_token_budget),
            ("plan_state", plan_state, self.config.plan_state_token_budget),
            ("decision_ledger", decision_ledger, self.config.decision_ledger_token_budget),
        ]

        # Add reasoning state
        if reasoning_states:
            latest = reasoning_states[-1]
            reasoning_content = self._format_reasoning_state(latest)
            block_specs.append(("reasoning_state", reasoning_content, self.config.reasoning_state_token_budget))

        # Add recent observations
        if recent_observations:
            block_specs.append(("recent_observations", recent_observations, self.config.recent_observations_token_budget))

        # Add placeholders
        if placeholders:
            placeholder_content = self._format_placeholders(placeholders)
            block_specs.append(
                ("masked_placeholders", placeholder_content, 200)  # Small budget
            )

        # Add retrieved snippets
        if retrieved_snippets:
            snippets_content = self._format_retrieved_snippets(retrieved_snippets)
            block_specs.append(("retrieved_snippets", snippets_content, self.config.compressed_snippets_token_budget))

        # Process blocks in priority order
        for name, content, budget in block_specs:
            if not content.strip():
                continue

            priority = self.config.block_priorities.get(name, 10)
            block_budget = min(budget, remaining_budget)

            if block_budget <= 0:
                blocks_truncated.append(name)
                continue

            truncated_content, was_truncated = self._truncate_to_budget(content, block_budget)
            if was_truncated:
                blocks_truncated.append(name)

            token_estimate = self.config.estimate_tokens(truncated_content)
            remaining_budget -= token_estimate
            budget_used[name] = token_estimate

            blocks.append(
                ContextBlock(
                    name=name.replace("_", " ").title(),
                    content=truncated_content,
                    priority=priority,
                    token_estimate=token_estimate,
                    source=name,
                )
            )

        total_tokens = sum(b.token_estimate for b in blocks)

        return AssembledContext(
            blocks=blocks,
            total_tokens=total_tokens,
            budget_used=budget_used,
            blocks_truncated=blocks_truncated,
            debug_info={
                "assembly_number": self._assembly_count,
                "total_budget": total_budget,
                "remaining_budget": remaining_budget,
            },
        )

    def save_debug_artifact(
        self,
        context: AssembledContext,
        output_dir: Path | None = None,
    ) -> Path:
        """Save debug information about the assembled context.

        Args:
            context: The assembled context.
            output_dir: Directory to save to (defaults to workspace).

        Returns:
            Path to the saved debug file.
        """
        if output_dir is None:
            output_dir = self.config.get_workspace_dir() / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"context_assembly_{timestamp}.json"
        filepath = output_dir / filename

        debug_data = {
            "timestamp": timestamp,
            "total_tokens": context.total_tokens,
            "budget_used": context.budget_used,
            "blocks_truncated": context.blocks_truncated,
            "blocks": [
                {
                    "name": b.name,
                    "priority": b.priority,
                    "token_estimate": b.token_estimate,
                    "content_preview": b.content[:200] + "..." if len(b.content) > 200 else b.content,
                }
                for b in context.blocks
            ],
            **context.debug_info,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2)

        logger.debug(f"Saved context assembly debug to {filepath}")
        return filepath

    def get_assembly_count(self) -> int:
        """Get the number of context assemblies performed."""
        return self._assembly_count

    def _format_retrieved_snippets(
        self,
        results: list[RetrievalResult],
    ) -> str:
        """Format retrieved snippets."""
        if not results:
            return ""

        lines = ["*Retrieved snippets:*", ""]
        for r in results:
            lines.append(f"**[{r.artifact_id}:{r.line_number}]**")
            lines.append(r.snippet[:300])
            lines.append("")

        return "\n".join(lines)
