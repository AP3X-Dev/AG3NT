"""Compaction and session management loops for AG3NT autonomous system.

Monitors context window usage and orchestrates a 3-tier compaction cascade
when token consumption exceeds a configurable threshold:

    Tier 1: Tool-output pruning  (free — no LLM call)
    Tier 2: Memory extraction    (cheap — pull insights before trimming)
    Tier 3: Full compaction      (dedicated CompactionAgent LLM call)

The CompactionAgent is provided as a stub interface here. Actual LLM
integration is wired up separately (Task 29).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Optional, Protocol

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.loops.compaction")


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CompactionLoopConfig:
    """Tuning knobs for the compaction trigger and orchestrator.

    Attributes:
        threshold_ratio: Context-window usage ratio that triggers compaction
            (e.g. 0.75 means "compact when 75 % of the window is consumed").
        target_ratio: Desired usage ratio after compaction completes.
        chars_per_token: Rough character-to-token multiplier used for the
            heuristic token estimator.
        min_turns_before_compact: Minimum conversation turns that must occur
            before compaction is considered (avoids premature compaction of
            short conversations).
        preserve_recent_turns: Number of most-recent turns that are *never*
            pruned or summarised during compaction.
        prune_threshold: Usage ratio at which Tier 1 (tool-output pruning) is
            activated.
        extract_threshold: Usage ratio at which Tier 2 (memory extraction) is
            activated.
        compact_threshold: Usage ratio at which Tier 3 (full LLM compaction)
            is activated.
        max_extraction_entries: Maximum memory entries to extract in Tier 2.
        context_window: Default context-window size in tokens.
    """

    threshold_ratio: float = 0.75
    target_ratio: float = 0.50
    chars_per_token: float = 4.0
    min_turns_before_compact: int = 3
    preserve_recent_turns: int = 3
    prune_threshold: float = 0.60
    extract_threshold: float = 0.70
    compact_threshold: float = 0.80
    max_extraction_entries: int = 10
    context_window: int = 128_000


class CompactionTier(IntEnum):
    """Compaction cascade levels."""

    NONE = 0
    PRUNE = 1
    EXTRACT = 2
    COMPACT = 3


# ═══════════════════════════════════════════════════════════════════════════════
#  CompactionTrigger
# ═══════════════════════════════════════════════════════════════════════════════


class CompactionTrigger:
    """Monitors token usage and emits ``compaction.needed`` when estimated
    consumption crosses *threshold_ratio* of the context window.

    Usage::

        bus = EventBus()
        trigger = CompactionTrigger(CompactionLoopConfig(), bus)
        await trigger.start()

        # Feed turn data
        trigger.record_turn(char_count=4200)

        if trigger.needs_compaction():
            ...  # hand off to CompactionOrchestrator

    The trigger subscribes to ``turn.completed`` events on the bus so it can
    automatically track character counts published by the agent runtime.
    """

    def __init__(self, config: CompactionLoopConfig, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._context_window: int = config.context_window
        self._cumulative_chars: int = 0
        self._turn_count: int = 0
        self._sub_ids: list[str] = []
        self._running: bool = False
        self._force_requested: bool = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to relevant events and begin monitoring."""
        if self._running:
            return
        self._running = True

        # Listen for turn completions to auto-track character counts.
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_turn_completed,
                event_types={"turn.completed"},
            )
        )

        logger.info(
            "CompactionTrigger started (threshold=%.0f%%, target=%.0f%%)",
            self._config.threshold_ratio * 100,
            self._config.target_ratio * 100,
        )

    async def stop(self) -> None:
        """Unsubscribe from the bus and stop monitoring."""
        for sub_id in self._sub_ids:
            self._bus.unsubscribe(sub_id)
        self._sub_ids.clear()
        self._running = False
        logger.info("CompactionTrigger stopped")

    def request_immediate(self) -> None:
        """Request immediate compaction at the next opportunity.

        Sets a force flag that :meth:`needs_compaction` will honour
        regardless of the current usage ratio.  Also fires a
        ``compaction.needed`` event on the bus so listeners react
        immediately.
        """
        self._force_requested = True
        # Fire-and-forget: schedule the event on the running loop if one
        # exists; otherwise the flag alone is sufficient.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_compaction_needed())
        except RuntimeError:
            pass  # no running loop — the flag will be picked up synchronously
        logger.info("Immediate compaction requested")

    # -- public API ----------------------------------------------------------

    def set_context_window(self, tokens: int) -> None:
        """Update context-window size from provider information."""
        if tokens > 0:
            self._context_window = tokens
            logger.debug("Context window updated to %d tokens", tokens)

    def record_turn(self, char_count: int) -> None:
        """Record a conversation turn by its character count.

        Args:
            char_count: Total characters in the turn (user + assistant).
        """
        self._cumulative_chars += max(0, char_count)
        self._turn_count += 1

    def needs_compaction(self) -> bool:
        """Return ``True`` if current usage exceeds the threshold *and*
        enough turns have elapsed, or if an immediate compaction was
        requested via :meth:`request_immediate`."""
        if self._force_requested:
            self._force_requested = False
            return True
        estimated = self._estimate_tokens(self._cumulative_chars)
        threshold = int(self._context_window * self._config.threshold_ratio)
        return (
            estimated >= threshold
            and self._turn_count >= self._config.min_turns_before_compact
        )

    def usage_ratio(self) -> float:
        """Return current estimated context-window usage as a ratio (0.0–1.0)."""
        estimated = self._estimate_tokens(self._cumulative_chars)
        if self._context_window <= 0:
            return 0.0
        return min(estimated / self._context_window, 1.0)

    def reset(self) -> None:
        """Reset counters (call after compaction or when starting a new session)."""
        self._cumulative_chars = 0
        self._turn_count = 0

    def get_status(self) -> dict[str, Any]:
        """Return a status snapshot for dashboards / health checks."""
        estimated = self._estimate_tokens(self._cumulative_chars)
        return {
            "running": self._running,
            "turn_count": self._turn_count,
            "cumulative_chars": self._cumulative_chars,
            "estimated_tokens": estimated,
            "context_window": self._context_window,
            "usage_ratio": self.usage_ratio(),
            "needs_compaction": self.needs_compaction(),
        }

    # -- event handlers ------------------------------------------------------

    async def _on_turn_completed(self, event: Event) -> None:
        """Handle ``turn.completed`` events from the agent runtime."""
        char_count = event.payload.get("char_count", 0)
        if char_count > 0:
            self.record_turn(char_count)

        # Check and emit compaction.needed if threshold crossed.
        if self.needs_compaction():
            await self._emit_compaction_needed()

    async def _emit_compaction_needed(self) -> None:
        """Publish a ``compaction.needed`` event on the bus."""
        await self._bus.publish(
            Event(
                event_type="compaction.needed",
                source="compaction_trigger",
                payload={
                    "usage_ratio": self.usage_ratio(),
                    "estimated_tokens": self._estimate_tokens(self._cumulative_chars),
                    "context_window": self._context_window,
                    "turn_count": self._turn_count,
                },
                priority=EventPriority.HIGH,
            )
        )
        logger.info(
            "compaction.needed emitted (usage=%.1f%%, turns=%d)",
            self.usage_ratio() * 100,
            self._turn_count,
        )

    # -- internals -----------------------------------------------------------

    def _estimate_tokens(self, chars: int) -> int:
        """Estimate token count from character count."""
        cpt = self._config.chars_per_token
        if cpt <= 0:
            return chars  # fallback: 1 char ≈ 1 token
        return int(chars / cpt)


# ═══════════════════════════════════════════════════════════════════════════════
#  CompactionAgent (stub / interface)
# ═══════════════════════════════════════════════════════════════════════════════


class CompactionAgentProtocol(Protocol):
    """Protocol that a real CompactionAgent must satisfy."""

    def build_compaction_messages(
        self, original_messages: list[dict]
    ) -> list[dict]: ...

    def select_compaction_model(
        self,
        available_providers: set[str],
        config_provider: str = "",
        config_model: str = "",
    ) -> tuple[str, str] | None: ...


class CompactionAgent:
    """Stub CompactionAgent for context compaction.

    Uses a cheap/fast model specifically prompted for conversation
    summarisation. The actual LLM integration is wired up in Task 29;
    this stub provides the interface and system prompt.
    """

    COMPACTION_SYSTEM_PROMPT: str = (
        "You are a session compaction specialist. Your ONLY job is to produce a "
        "comprehensive summary of this conversation that will serve as the starting "
        "context for a continuation. The continuation will have NO access to the "
        "original messages — your summary is ALL it will know.\n\n"
        "Produce your summary with these sections:\n\n"
        "## Task\nWhat the user originally asked for. What is the overall goal.\n\n"
        "## Progress\nWhat has been completed. List specific files modified, tests passed.\n\n"
        "## Current State\nWhat we are actively working on RIGHT NOW. Include file paths, "
        "function names, and the exact state of any in-progress work.\n\n"
        "## Key Decisions\nArchitecture decisions, approach choices, user preferences expressed.\n\n"
        "## Issues & Workarounds\nErrors encountered, things that didn't work, workarounds applied.\n\n"
        "## Next Steps\nWhat still needs to be done to complete the task.\n\n"
        "## Working Files\nList every file the agent has read, modified, or created, with current status.\n\n"
        "Be exhaustive. Missing information here means it is LOST FOREVER."
    )

    _MODEL_PREFERENCE: list[tuple[str, str]] = [
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("google", "gemini-2.0-flash"),
        ("openai", "gpt-4o-mini"),
        ("deepseek", "deepseek-chat"),
    ]

    def select_compaction_model(
        self,
        available_providers: set[str],
        config_provider: str = "",
        config_model: str = "",
    ) -> tuple[str, str] | None:
        """Pick the cheapest available model for compaction work."""
        if config_provider and config_model:
            return (config_provider, config_model)

        for provider_id, model_id in self._MODEL_PREFERENCE:
            if provider_id in available_providers:
                return (provider_id, model_id)

        if available_providers:
            return (next(iter(available_providers)), "")
        return None

    def build_compaction_messages(
        self,
        original_messages: list[dict],
    ) -> list[dict]:
        """Build the message list for the compaction LLM call."""
        return [
            {"role": "system", "content": self.COMPACTION_SYSTEM_PROMPT},
            *original_messages,
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#  CompactionOrchestrator
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CompactionRecord:
    """Immutable record of a single compaction run."""

    timestamp: str
    session_id: str
    tier_reached: int
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "tier_reached": self.tier_reached,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "duration_ms": self.duration_ms,
        }


class CompactionOrchestrator:
    """Centralised 3-tier compaction cascade.

    Tier 1 — **Prune**: Truncate large tool outputs and remove stale
    system messages.  Free — no LLM call required.

    Tier 2 — **Extract**: Pull important facts into long-term memory
    before they are trimmed.  Cheap — may use a fast model.

    Tier 3 — **Compact**: Full session compaction via a dedicated
    ``CompactionAgent`` that summarises the conversation.

    The orchestrator emits ``compaction.started`` and
    ``compaction.completed`` events on the bus so that other subsystems
    (e.g. snapshots, dashboards) can react.

    Usage::

        bus = EventBus()
        config = CompactionLoopConfig()
        orch = CompactionOrchestrator(config, bus)
        await orch.start()

        result = await orch.run_cascade(messages, context_window=128_000)
    """

    def __init__(
        self,
        config: CompactionLoopConfig,
        bus: EventBus,
        *,
        pruner: Optional[Callable[[list[dict], int], int]] = None,
        memory_extractor: Optional[Callable[[list[dict], str], int]] = None,
        compaction_agent: Optional[CompactionAgent] = None,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            config: Compaction configuration.
            bus: EventBus for emitting lifecycle events.
            pruner: Optional callable ``(messages, preserve_recent) -> tokens_saved``.
                If ``None``, a default pruner is used that truncates large
                tool-output strings.
            memory_extractor: Optional callable
                ``(messages, session_id) -> entries_extracted``.
                If ``None``, Tier 2 is skipped gracefully.
            compaction_agent: Optional ``CompactionAgent`` for Tier 3.
                A stub is created if not provided.
        """
        self._config = config
        self._bus = bus
        self._pruner = pruner or self._default_pruner
        self._memory_extractor = memory_extractor
        self._agent = compaction_agent or CompactionAgent()
        self._history: list[CompactionRecord] = []
        self._sub_ids: list[str] = []
        self._running: bool = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to ``compaction.needed`` and start the orchestrator."""
        if self._running:
            return
        self._running = True

        self._sub_ids.append(
            self._bus.subscribe(
                self._on_compaction_needed,
                event_types={"compaction.needed"},
            )
        )
        logger.info("CompactionOrchestrator started")

    async def stop(self) -> None:
        """Unsubscribe and stop the orchestrator."""
        for sub_id in self._sub_ids:
            self._bus.unsubscribe(sub_id)
        self._sub_ids.clear()
        self._running = False
        logger.info("CompactionOrchestrator stopped")

    # -- tier selection ------------------------------------------------------

    def select_tier(self, usage_ratio: float) -> CompactionTier:
        """Determine which tier to run based on context-window usage.

        Returns:
            ``CompactionTier.NONE`` through ``CompactionTier.COMPACT``.
        """
        if usage_ratio >= self._config.compact_threshold:
            return CompactionTier.COMPACT
        if usage_ratio >= self._config.extract_threshold:
            return CompactionTier.EXTRACT
        if usage_ratio >= self._config.prune_threshold:
            return CompactionTier.PRUNE
        return CompactionTier.NONE

    # -- cascade runner ------------------------------------------------------

    async def run_cascade(
        self,
        messages: list[dict],
        context_window: int = 0,
        session_id: str = "",
    ) -> dict[str, Any]:
        """Run the 3-tier compaction cascade.

        Args:
            messages: Conversation messages (list of dicts with at least
                a ``"content"`` key).
            context_window: Context-window size in tokens.  Falls back to
                ``config.context_window`` if not provided.
            session_id: Optional session identifier for event payloads.

        Returns:
            Dict with keys ``tier_reached``, ``tokens_saved``,
            ``messages`` (possibly modified in-place).
        """
        start_time = time.monotonic()
        ctx_window = context_window or self._config.context_window

        # Estimate current usage.
        total_chars = sum(
            len(m.get("content", "")) for m in messages
        )
        cpt = self._config.chars_per_token or 4.0
        estimated_tokens = int(total_chars / cpt)
        usage_ratio = estimated_tokens / ctx_window if ctx_window > 0 else 0.0

        tier = self.select_tier(usage_ratio)
        tokens_saved = 0

        if tier == CompactionTier.NONE:
            return {
                "tier_reached": 0,
                "tokens_saved": 0,
                "messages": messages,
            }

        # -- emit started event --
        await self._bus.publish(
            Event(
                event_type="compaction.started",
                source="compaction_orchestrator",
                payload={
                    "session_id": session_id,
                    "usage_ratio": usage_ratio,
                    "estimated_tokens": estimated_tokens,
                    "tier_selected": tier.value,
                },
                priority=EventPriority.MEDIUM,
            )
        )

        # Tier 1: Tool-output pruning (free).
        if tier >= CompactionTier.PRUNE:
            saved = self._pruner(messages, self._config.preserve_recent_turns)
            tokens_saved += saved
            logger.info("Tier 1 (prune): saved ~%d tokens", saved)

            # Re-estimate after pruning.
            total_chars = sum(len(m.get("content", "")) for m in messages)
            estimated_tokens = int(total_chars / cpt)
            usage_ratio = estimated_tokens / ctx_window if ctx_window > 0 else 0.0

            # If pruning was enough, stop early.
            if usage_ratio < self._config.extract_threshold:
                return await self._finish(
                    CompactionTier.PRUNE,
                    tokens_saved,
                    messages,
                    session_id,
                    estimated_tokens,
                    start_time,
                )

        # Tier 2: Memory extraction.
        if tier >= CompactionTier.EXTRACT and self._memory_extractor is not None:
            logger.info("Tier 2 (extract): extracting memories")
            try:
                entries = self._memory_extractor(messages, session_id)
                tokens_saved += entries * 100  # rough estimate
                logger.info("Tier 2: extracted %d entries", entries)
            except Exception as exc:
                logger.warning("Tier 2 extraction failed: %s", exc)

        # Tier 3: Full compaction via CompactionAgent.
        if tier >= CompactionTier.COMPACT:
            logger.info("Tier 3 (compact): running dedicated compaction agent")
            keep = max(self._config.preserve_recent_turns, 1)
            if len(messages) > keep:
                dropped = messages[:-keep]
                summary_text = (
                    f"[Compacted: {len(dropped)} earlier messages removed to save context]"
                )
                summary_msg = {
                    "role": "system",
                    "content": summary_text,
                }
                kept = [summary_msg] + messages[-keep:]

                # Estimate tokens saved from dropped messages.
                for msg in dropped:
                    content = msg.get("content", "")
                    tokens_saved += int(len(content) / cpt)

                messages.clear()
                messages.extend(kept)

        # Re-estimate after all tiers.
        total_chars = sum(len(m.get("content", "")) for m in messages)
        final_tokens = int(total_chars / cpt)

        return await self._finish(
            tier,
            tokens_saved,
            messages,
            session_id,
            final_tokens,
            start_time,
        )

    # -- history / status ----------------------------------------------------

    @property
    def history(self) -> list[CompactionRecord]:
        """Return the compaction history (most recent last)."""
        return list(self._history)

    def get_status(self) -> dict[str, Any]:
        """Return a status snapshot for dashboards / health checks."""
        return {
            "running": self._running,
            "total_compactions": len(self._history),
            "last_compaction": (
                self._history[-1].to_dict() if self._history else None
            ),
        }

    def for_subagent(
        self,
        pruner: Optional[Callable[[list[dict], int], int]] = None,
        memory_extractor: Optional[Callable[[list[dict], str], int]] = None,
    ) -> "CompactionOrchestrator":
        """Create a scoped orchestrator for a sub-agent.

        Shares config and bus with the parent but gets independent pruner,
        memory extractor, and history.
        """
        return CompactionOrchestrator(
            config=self._config,
            bus=self._bus,
            pruner=pruner or self._pruner,
            memory_extractor=memory_extractor or self._memory_extractor,
            compaction_agent=CompactionAgent(),
        )

    # -- event handler -------------------------------------------------------

    async def _on_compaction_needed(self, event: Event) -> None:
        """React to ``compaction.needed`` events.

        This handler is a hook point — concrete consumers should wire the
        messages list in.  By default, it logs the event so the dashboard
        can show that compaction was requested.
        """
        logger.info(
            "Received compaction.needed (usage=%.1f%%, tokens=%d)",
            event.payload.get("usage_ratio", 0.0) * 100,
            event.payload.get("estimated_tokens", 0),
        )

    # -- internal helpers ----------------------------------------------------

    async def _finish(
        self,
        tier: CompactionTier,
        tokens_saved: int,
        messages: list[dict],
        session_id: str,
        final_tokens: int,
        start_time: float,
    ) -> dict[str, Any]:
        """Emit ``compaction.completed``, record history, return result."""
        duration_ms = (time.monotonic() - start_time) * 1000

        record = CompactionRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            tier_reached=tier.value,
            tokens_before=final_tokens + tokens_saved,
            tokens_after=final_tokens,
            tokens_saved=tokens_saved,
            duration_ms=round(duration_ms, 2),
        )
        self._history.append(record)

        await self._bus.publish(
            Event(
                event_type="compaction.completed",
                source="compaction_orchestrator",
                payload={
                    "session_id": session_id,
                    "tier_reached": tier.value,
                    "tokens_saved": tokens_saved,
                    "duration_ms": record.duration_ms,
                },
                priority=EventPriority.MEDIUM,
            )
        )

        logger.info(
            "Compaction completed: tier=%d, saved=%d tokens, %.1fms",
            tier.value,
            tokens_saved,
            duration_ms,
        )

        return {
            "tier_reached": tier.value,
            "tokens_saved": tokens_saved,
            "messages": messages,
        }

    @staticmethod
    def _default_pruner(messages: list[dict], preserve_recent: int) -> int:
        """Built-in Tier-1 pruner: truncate large tool outputs.

        Scans all messages except the most recent *preserve_recent* and
        replaces any ``content`` longer than 2000 characters with a
        truncated version.

        Returns the estimated number of tokens saved.
        """
        saved_chars = 0
        cutoff = max(len(messages) - preserve_recent, 0)
        max_content_len = 2000

        for msg in messages[:cutoff]:
            content = msg.get("content", "")
            role = msg.get("role", "")
            # Only prune tool / system outputs (not user messages).
            if role in ("tool", "system", "function") and len(content) > max_content_len:
                original_len = len(content)
                msg["content"] = (
                    content[:max_content_len]
                    + f"\n\n[... truncated {original_len - max_content_len} chars ...]"
                )
                saved_chars += original_len - len(msg["content"])

        # Convert chars saved to approximate tokens.
        return int(saved_chars / 4.0) if saved_chars > 0 else 0
