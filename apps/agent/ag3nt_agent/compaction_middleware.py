"""Unified Compaction Middleware for AG3NT.

This module provides a unified middleware that integrates all context compaction
components into a single pipeline:

- Observation Masking: Replace large tool outputs with placeholders
- Memory Flush: Extract insights before compaction
- Context Pruning: Remove old/stale messages
- Progressive Summarization: Multi-stage summarization for large histories

Usage:
    from ag3nt_agent.compaction_middleware import CompactionMiddleware, CompactionConfig

    middleware = CompactionMiddleware(config)
    result = await middleware.compact(messages, token_count)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """Configuration for unified compaction middleware.

    Attributes:
        enabled: Master switch for compaction
        token_threshold: Token count to trigger compaction
        message_threshold: Message count to trigger compaction
        enable_masking: Enable observation masking
        enable_flush: Enable pre-compaction memory flush
        enable_pruning: Enable context pruning
        enable_progressive: Enable progressive summarization
        preserve_recent: Number of recent messages to preserve
    """

    enabled: bool = True
    token_threshold: int = 80000
    message_threshold: int = 100
    enable_masking: bool = True
    enable_flush: bool = True
    enable_pruning: bool = True
    enable_progressive: bool = True
    enable_agentic_flush: bool = True
    preserve_recent: int = 20


@dataclass
class CompactionMetrics:
    """Metrics from a compaction operation.

    Attributes:
        triggered: Whether compaction was triggered
        tokens_before: Token count before compaction
        tokens_after: Token count after compaction
        messages_before: Message count before compaction
        messages_after: Message count after compaction
        artifacts_created: Number of artifacts stored
        insights_flushed: Number of insights flushed to memory
        chunks_summarized: Number of chunks progressively summarized
        duration_ms: Time taken in milliseconds
        timestamp: When compaction occurred
    """

    triggered: bool = False
    tokens_before: int = 0
    tokens_after: int = 0
    messages_before: int = 0
    messages_after: int = 0
    artifacts_created: int = 0
    insights_flushed: int = 0
    agentic_insights_flushed: int = 0
    chunks_summarized: int = 0
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def compression_ratio(self) -> float:
        """Get the compression ratio (0-1)."""
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before


class CompactionMiddleware:
    """Unified compaction middleware for context management.

    Integrates observation masking, memory flush, pruning, and
    progressive summarization into a single pipeline.
    """

    def __init__(self, config: CompactionConfig | None = None) -> None:
        """Initialize the compaction middleware.

        Args:
            config: Configuration (uses defaults if not provided)
        """
        self._config = config or CompactionConfig()
        self._metrics_history: list[CompactionMetrics] = []
        self._total_compactions = 0

    @property
    def config(self) -> CompactionConfig:
        """Get the configuration."""
        return self._config

    def should_compact(
        self,
        messages: list[AnyMessage],
        token_count: int | None = None,
    ) -> bool:
        """Check if compaction should be triggered.

        Args:
            messages: Current messages
            token_count: Optional pre-computed token count

        Returns:
            True if compaction should be triggered
        """
        if not self._config.enabled:
            return False

        if token_count is None:
            token_count = count_tokens_approximately(messages)

        return (
            token_count >= self._config.token_threshold
            or len(messages) >= self._config.message_threshold
        )

    def _apply_masking(
        self,
        messages: list[AnyMessage],
        session_id: str | None = None,
    ) -> tuple[list[AnyMessage], int]:
        """Apply observation masking to messages.

        Returns:
            Tuple of (processed_messages, artifacts_created_count)
        """
        from ag3nt_agent.observation_masking import get_observation_masker

        masker = get_observation_masker()
        processed, results = masker.mask_messages(messages, session_id)
        artifacts_created = sum(1 for r in results if r.was_masked)
        return processed, artifacts_created

    def _apply_flush(
        self,
        messages: list[AnyMessage],
        token_count: int,
    ) -> int:
        """Apply pre-compaction memory flush.

        Returns:
            Number of insights flushed
        """
        from ag3nt_agent.memory_flush import get_memory_flusher

        flusher = get_memory_flusher()
        if not flusher.should_flush(token_count):
            return 0

        # Convert messages to dict format for flusher
        msg_dicts = [{"content": getattr(m, "content", str(m))} for m in messages]
        result = flusher.flush(msg_dicts)
        return result.insights_count if result.flushed else 0

    async def _apply_agentic_flush(
        self,
        messages: list[AnyMessage],
        token_count: int,
        session_id: str | None = None,
    ) -> int:
        """Apply LLM-powered agentic memory flush.

        Uses a cheap model to extract insights from messages about to be
        compacted away.  Only runs when the token count is at least 80% of
        the configured threshold (i.e. a "heavy" compaction).

        Returns:
            Number of insights extracted and persisted.
        """
        if token_count < self._config.token_threshold * 0.8:
            return 0

        try:
            from ag3nt_agent.agentic_flush import agentic_flush

            count = await agentic_flush(messages, session_id)
            return count
        except Exception:
            logger.debug("Agentic flush unavailable", exc_info=True)
            return 0

    def _apply_pruning(
        self,
        messages: list[AnyMessage],
        token_count: int,
    ) -> list[AnyMessage]:
        """Apply context pruning to messages.

        Returns:
            Pruned message list
        """
        from ag3nt_agent.context_summarization import get_auto_pruner

        pruner = get_auto_pruner()
        pruned_messages, result = pruner.prune_messages(messages)
        if result.pruned:
            logger.info(f"Pruned {result.messages_removed} messages")
            return pruned_messages
        return messages

    def _apply_progressive(
        self,
        messages: list[AnyMessage],
        summarize_fn: Callable[[list[AnyMessage]], str] | None = None,
    ) -> tuple[list[AnyMessage], int]:
        """Apply progressive summarization.

        Returns:
            Tuple of (processed_messages, chunks_summarized)
        """
        from langchain_core.messages import AIMessage, HumanMessage

        from ag3nt_agent.context_summarization import get_progressive_summarizer

        summarizer = get_progressive_summarizer()

        # Use default summarize function if not provided
        if summarize_fn is None:
            def summarize_fn(msgs: list[AnyMessage]) -> str:
                content_parts = []
                for m in msgs:
                    role = getattr(m, "type", "message")
                    content = getattr(m, "content", str(m))
                    if content:
                        content_parts.append(f"[{role}] {content[:200]}...")
                return f"Summary of {len(msgs)} messages:\n" + "\n".join(content_parts[:5])

        to_summarize, to_preserve = summarizer.get_preserved_messages(
            messages, self._config.preserve_recent
        )

        if not to_summarize:
            return messages, 0

        result = summarizer.summarize(to_summarize, summarize_fn)

        if not result.summarized:
            return messages, 0

        # Structure as a user→assistant pair so the model treats the summary
        # as its own prior context rather than user content to echo back.
        merged = summarizer.merge_summaries(result.summaries)
        summary_ask = HumanMessage(
            content="What have we been working on so far?",
            additional_kwargs={"lc_source": "summarization"},
        )
        summary_answer = AIMessage(
            content=f"Here's a summary of our conversation so far:\n\n{merged}",
            additional_kwargs={"lc_source": "summarization"},
        )

        return [summary_ask, summary_answer] + list(to_preserve), result.chunks_processed

    def compact(
        self,
        messages: list[AnyMessage],
        token_count: int | None = None,
        session_id: str | None = None,
        summarize_fn: Callable[[list[AnyMessage]], str] | None = None,
    ) -> tuple[list[AnyMessage], CompactionMetrics]:
        """Perform full compaction pipeline.

        Args:
            messages: Messages to compact
            token_count: Optional pre-computed token count
            session_id: Optional session ID for artifacts
            summarize_fn: Optional custom summarization function

        Returns:
            Tuple of (compacted_messages, metrics)
        """
        start_time = time.time()

        if token_count is None:
            token_count = count_tokens_approximately(messages)

        metrics = CompactionMetrics(
            tokens_before=token_count,
            messages_before=len(messages),
        )

        if not self.should_compact(messages, token_count):
            metrics.tokens_after = token_count
            metrics.messages_after = len(messages)
            return messages, metrics

        metrics.triggered = True
        processed = messages

        # Step 1: Observation masking
        if self._config.enable_masking:
            processed, artifacts = self._apply_masking(processed, session_id)
            metrics.artifacts_created = artifacts
            logger.debug(f"Masking: {artifacts} artifacts created")

        # Step 2: Memory flush
        if self._config.enable_flush:
            insights = self._apply_flush(processed, token_count)
            metrics.insights_flushed = insights
            logger.debug(f"Flush: {insights} insights flushed")

        # Step 2.5: Agentic flush (LLM-powered, only for heavy compaction)
        # NOTE: This is an async step.  When compact() is called from a
        # synchronous context the agentic flush is skipped silently.
        if self._config.enable_agentic_flush:
            try:
                import asyncio

                coro = self._apply_agentic_flush(processed, token_count, session_id)
                try:
                    loop = asyncio.get_running_loop()
                    # We are inside an event loop — schedule and await via Task
                    task = loop.create_task(coro)
                    # In a sync method we cannot truly await, so we rely on
                    # async_compact() for the real path.  Record zero here.
                    metrics.agentic_insights_flushed = 0
                except RuntimeError:
                    # No running loop — safe to use asyncio.run()
                    metrics.agentic_insights_flushed = asyncio.run(coro)
            except Exception:
                logger.debug("Agentic flush step skipped", exc_info=True)

        # Step 3: Pruning
        if self._config.enable_pruning:
            processed = self._apply_pruning(processed, token_count)
            logger.debug(f"Pruning: {len(processed)} messages remaining")

        # Step 4: Progressive summarization
        if self._config.enable_progressive:
            processed, chunks = self._apply_progressive(processed, summarize_fn)
            metrics.chunks_summarized = chunks
            logger.debug(f"Progressive: {chunks} chunks summarized")

        metrics.tokens_after = count_tokens_approximately(processed)
        metrics.messages_after = len(processed)
        metrics.duration_ms = (time.time() - start_time) * 1000

        self._metrics_history.append(metrics)
        self._total_compactions += 1

        logger.info(
            f"Compaction complete: {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.compression_ratio:.1%}), {metrics.duration_ms:.1f}ms"
        )

        return processed, metrics

    async def async_compact(
        self,
        messages: list[AnyMessage],
        token_count: int | None = None,
        session_id: str | None = None,
        summarize_fn: Callable[[list[AnyMessage]], str] | None = None,
    ) -> tuple[list[AnyMessage], CompactionMetrics]:
        """Async variant of compact() that properly awaits the agentic flush.

        Prefer this over compact() when called from an async context (e.g.
        the DeepAgents runtime) so that the LLM-powered agentic flush can
        run without hacks.

        Args:
            messages: Messages to compact
            token_count: Optional pre-computed token count
            session_id: Optional session ID for artifacts
            summarize_fn: Optional custom summarization function

        Returns:
            Tuple of (compacted_messages, metrics)
        """
        start_time = time.time()

        if token_count is None:
            token_count = count_tokens_approximately(messages)

        metrics = CompactionMetrics(
            tokens_before=token_count,
            messages_before=len(messages),
        )

        if not self.should_compact(messages, token_count):
            metrics.tokens_after = token_count
            metrics.messages_after = len(messages)
            return messages, metrics

        metrics.triggered = True
        processed = messages

        # Step 1: Observation masking
        if self._config.enable_masking:
            processed, artifacts = self._apply_masking(processed, session_id)
            metrics.artifacts_created = artifacts

        # Step 2: Memory flush (regex-based)
        if self._config.enable_flush:
            insights = self._apply_flush(processed, token_count)
            metrics.insights_flushed = insights

        # Step 2.5: Agentic flush (LLM-powered)
        if self._config.enable_agentic_flush:
            try:
                agentic_count = await self._apply_agentic_flush(
                    processed, token_count, session_id
                )
                metrics.agentic_insights_flushed = agentic_count
            except Exception:
                logger.debug("Agentic flush step skipped", exc_info=True)

        # Step 3: Pruning
        if self._config.enable_pruning:
            processed = self._apply_pruning(processed, token_count)

        # Step 4: Progressive summarization
        if self._config.enable_progressive:
            processed, chunks = self._apply_progressive(processed, summarize_fn)
            metrics.chunks_summarized = chunks

        metrics.tokens_after = count_tokens_approximately(processed)
        metrics.messages_after = len(processed)
        metrics.duration_ms = (time.time() - start_time) * 1000

        self._metrics_history.append(metrics)
        self._total_compactions += 1

        logger.info(
            f"Compaction complete: {metrics.tokens_before} -> {metrics.tokens_after} tokens "
            f"({metrics.compression_ratio:.1%}), {metrics.duration_ms:.1f}ms"
        )

        return processed, metrics

    def get_stats(self) -> dict[str, Any]:
        """Get compaction statistics.

        Returns:
            Dict with compaction stats
        """
        if not self._metrics_history:
            return {
                "total_compactions": 0,
                "avg_compression_ratio": 1.0,
                "total_artifacts": 0,
                "total_insights": 0,
            }

        return {
            "total_compactions": self._total_compactions,
            "avg_compression_ratio": sum(m.compression_ratio for m in self._metrics_history)
            / len(self._metrics_history),
            "total_artifacts": sum(m.artifacts_created for m in self._metrics_history),
            "total_insights": sum(m.insights_flushed for m in self._metrics_history),
            "total_agentic_insights": sum(
                m.agentic_insights_flushed for m in self._metrics_history
            ),
            "total_chunks": sum(m.chunks_summarized for m in self._metrics_history),
            "avg_duration_ms": sum(m.duration_ms for m in self._metrics_history)
            / len(self._metrics_history),
        }


# Preset configurations
COMPACTION_DISABLED = CompactionConfig(enabled=False)
COMPACTION_CONSERVATIVE = CompactionConfig(
    token_threshold=120000,
    message_threshold=150,
    preserve_recent=30,
)
COMPACTION_BALANCED = CompactionConfig()
COMPACTION_AGGRESSIVE = CompactionConfig(
    token_threshold=50000,
    message_threshold=60,
    preserve_recent=10,
)


# Global middleware instance
_compaction_middleware: CompactionMiddleware | None = None


def get_compaction_middleware() -> CompactionMiddleware:
    """Get the global compaction middleware instance.

    Returns:
        Global CompactionMiddleware instance
    """
    global _compaction_middleware
    if _compaction_middleware is None:
        _compaction_middleware = CompactionMiddleware()
    return _compaction_middleware


def reset_compaction_middleware() -> None:
    """Reset the global compaction middleware (for testing)."""
    global _compaction_middleware
    _compaction_middleware = None

