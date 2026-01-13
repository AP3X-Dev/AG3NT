"""ReasoningStateSummarizer for periodic structured summarization.

This module provides structured summarization of agent reasoning progress:
1. Triggered periodically based on step count or context size
2. Produces structured ReasoningState with evidence pointers
3. Compresses intermediate reasoning while preserving key facts
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from deepagents.compaction.models import ReasoningState

if TYPE_CHECKING:
    from deepagents.compaction.config import CompactionConfig
    from deepagents.compaction.observation_masker import ObservationMasker

logger = logging.getLogger(__name__)


class ReasoningStateSummarizer:
    """Summarizes agent reasoning progress into structured state.

    Creates periodic summaries that compress intermediate reasoning while
    preserving key facts, hypotheses, and evidence pointers.

    Args:
        config: Compaction configuration.
        masker: ObservationMasker for accessing evidence ledger.
    """

    def __init__(
        self,
        config: CompactionConfig,
        masker: ObservationMasker,
    ) -> None:
        self.config = config
        self.masker = masker
        self._summaries: list[ReasoningState] = []
        self._last_summary_step = 0

    def should_summarize(
        self,
        step_number: int,
        estimated_tokens: int | None = None,
    ) -> bool:
        """Check if summarization should be triggered.

        Args:
            step_number: Current agent step number.
            estimated_tokens: Estimated current context tokens.

        Returns:
            True if summarization should be triggered.
        """
        # Check step-based trigger
        steps_since_last = step_number - self._last_summary_step
        if steps_since_last >= self.config.summarize_every_steps:
            return True

        # Check token-based trigger
        if estimated_tokens is not None:
            threshold = self.config.model_context_limit * self.config.summarize_if_estimated_context_tokens_gt
            if estimated_tokens > threshold:
                return True

        return False

    def _extract_facts_from_messages(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[str], list[str], list[str]]:
        """Extract facts, hypotheses, and questions from messages.

        Returns:
            Tuple of (confirmed_facts, hypotheses, open_questions).
        """
        confirmed_facts: list[str] = []
        hypotheses: list[str] = []
        open_questions: list[str] = []

        # Patterns for extraction
        fact_patterns = [
            r"(?:confirmed|verified|found that|discovered that|it is|the answer is)[:\s]+(.+?)(?:\.|$)",
            r"(?:according to|based on|the data shows)[:\s]+(.+?)(?:\.|$)",
        ]
        hypothesis_patterns = [
            r"(?:hypothesis|theory|might be|could be|possibly|likely)[:\s]+(.+?)(?:\.|$)",
            r"(?:I think|I believe|it seems|appears to be)[:\s]+(.+?)(?:\.|$)",
        ]
        question_patterns = [
            r"(?:need to find|need to verify|question|unclear|unknown)[:\s]+(.+?)(?:\?|$)",
            r"(?:what|how|why|when|where|who)[^?]*\?",
        ]

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Extract facts
            for pattern in fact_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches[:3]:  # Limit per message
                    fact = match.strip()
                    if len(fact) > 20 and fact not in confirmed_facts:
                        confirmed_facts.append(fact[:200])

            # Extract hypotheses
            for pattern in hypothesis_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches[:2]:
                    hyp = match.strip()
                    if len(hyp) > 20 and hyp not in hypotheses:
                        hypotheses.append(hyp[:200])

            # Extract questions
            for pattern in question_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches[:2]:
                    q = match.strip()
                    if len(q) > 10 and q not in open_questions:
                        open_questions.append(q[:200])

        return (
            confirmed_facts[:10],
            hypotheses[:5],
            open_questions[:5],
        )

    def _generate_executive_summary(
        self,
        messages: list[BaseMessage],
        confirmed_facts: list[str],
    ) -> str:
        """Generate a high-level executive summary."""
        # Count message types
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))

        # Get the original task from first human message
        task = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                task = content[:200] + "..." if len(content) > 200 else content
                break

        summary_parts = [
            f"Task: {task}",
            f"Progress: {ai_count} reasoning steps, {tool_count} tool calls.",
        ]

        if confirmed_facts:
            summary_parts.append(f"Key findings: {len(confirmed_facts)} facts confirmed.")

        return " ".join(summary_parts)

    def summarize(
        self,
        messages: list[BaseMessage],
        step_number: int,
    ) -> ReasoningState:
        """Create a structured summary of the current reasoning state.

        Args:
            messages: The conversation messages to summarize.
            step_number: Current agent step number.

        Returns:
            ReasoningState with structured summary.
        """
        # Extract structured information
        confirmed_facts, hypotheses, open_questions = self._extract_facts_from_messages(messages)

        # Get visited sources from evidence ledger
        evidence = self.masker.get_evidence_ledger()
        visited_sources = [e.url for e in evidence[:20]]

        # Generate executive summary
        executive_summary = self._generate_executive_summary(messages, confirmed_facts)

        # Create reasoning state
        state = ReasoningState(
            executive_summary=executive_summary,
            confirmed_facts=confirmed_facts,
            hypotheses=hypotheses,
            open_questions=open_questions,
            visited_sources=visited_sources,
            next_actions=[],  # Will be populated by agent
            step_number=step_number,
        )

        self._summaries.append(state)
        self._last_summary_step = step_number

        logger.info(f"Created reasoning state summary at step {step_number}: {len(confirmed_facts)} facts, {len(hypotheses)} hypotheses")

        return state

    def get_latest_summary(self) -> ReasoningState | None:
        """Get the most recent reasoning state summary."""
        return self._summaries[-1] if self._summaries else None

    def get_all_summaries(self) -> list[ReasoningState]:
        """Get all reasoning state summaries."""
        return self._summaries.copy()

    def format_for_context(self, state: ReasoningState) -> str:
        """Format a reasoning state for inclusion in context.

        Args:
            state: The reasoning state to format.

        Returns:
            Formatted string for context inclusion.
        """
        lines = [
            "## Reasoning State Summary",
            f"*Step {state.step_number} | {state.created_at.strftime('%H:%M:%S')}*",
            "",
            f"**Summary:** {state.executive_summary}",
        ]

        if state.confirmed_facts:
            lines.append("")
            lines.append("**Confirmed Facts:**")
            for fact in state.confirmed_facts[:5]:
                lines.append(f"- {fact}")

        if state.hypotheses:
            lines.append("")
            lines.append("**Working Hypotheses:**")
            for hyp in state.hypotheses[:3]:
                lines.append(f"- {hyp}")

        if state.open_questions:
            lines.append("")
            lines.append("**Open Questions:**")
            for q in state.open_questions[:3]:
                lines.append(f"- {q}")

        if state.visited_sources:
            lines.append("")
            lines.append(f"**Sources Consulted:** {len(state.visited_sources)} sources")

        return "\n".join(lines)

    def estimate_tokens(self, state: ReasoningState) -> int:
        """Estimate token count for a reasoning state."""
        formatted = self.format_for_context(state)
        return self.config.estimate_tokens(formatted)
