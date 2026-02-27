"""
ResearchReconLoop — Novelty-Gated Research Triggering.

Triggers research when new imports or novel errors are detected.
Uses a :class:`NoveltyGate` to avoid re-researching the same topic
within configurable cooldown windows.

Subscribes to:
- ``sentinel.indexed`` — scans for new imports in indexed files.
- ``autofix.failed``   — scans for novel, unresolved errors.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NoveltyGate — cooldown-based deduplication
# ---------------------------------------------------------------------------

# Cooldown durations per category (seconds)
_COOLDOWNS: dict[str, float] = {
    "import": 3600.0,   # 1 hour
    "error": 600.0,     # 10 minutes
}

_DEFAULT_COOLDOWN = 1800.0  # 30 minutes


class NoveltyGate:
    """Tracks what has already been researched and enforces cooldowns.

    Each ``(key, category)`` pair has a timestamp recording when it was
    last researched.  :meth:`is_novel` returns ``True`` only if the
    cooldown for that category has elapsed (or the key has never been
    seen).
    """

    def __init__(self) -> None:
        # category -> {key -> last_researched_timestamp}
        self._import_gate: dict[str, float] = {}
        self._error_gate: dict[str, float] = {}

    def _gate_for(self, category: str) -> dict[str, float]:
        """Return the gate dict for *category*."""
        if category == "import":
            return self._import_gate
        if category == "error":
            return self._error_gate
        # Fallback: create a dynamic gate (should not happen often)
        return self._import_gate

    def is_novel(self, key: str, category: str) -> bool:
        """Return ``True`` if *key* has not been researched within cooldown."""
        gate = self._gate_for(category)
        last_ts = gate.get(key)
        if last_ts is None:
            return True
        cooldown = _COOLDOWNS.get(category, _DEFAULT_COOLDOWN)
        return (time.time() - last_ts) >= cooldown

    def mark_researched(self, key: str, category: str) -> None:
        """Record that *key* was just researched."""
        gate = self._gate_for(category)
        gate[key] = time.time()

    def get_stats(self) -> dict[str, int]:
        """Return count of tracked keys per category."""
        return {
            "import_keys": len(self._import_gate),
            "error_keys": len(self._error_gate),
        }


# ---------------------------------------------------------------------------
# ResearchReconLoop
# ---------------------------------------------------------------------------

class ResearchReconLoop:
    """Reactive loop that triggers research on novel imports and errors.

    Subscribes to EventBus events and filters them through a
    :class:`NoveltyGate` before queuing research.

    Usage::

        bus = EventBus()
        loop = ResearchReconLoop(bus)
        await loop.start()
        # ... events flow ...
        await loop.stop()
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._gate = NoveltyGate()
        self._subscription_ids: list[str] = []
        self._running = False

        # Stats
        self._imports_detected = 0
        self._errors_detected = 0
        self._research_queued = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to sentinel.indexed and autofix.failed events."""
        if self._running:
            return

        self._running = True

        sub_imports = self._bus.subscribe(
            handler=self._on_new_imports,
            event_types={"sentinel.indexed"},
        )
        self._subscription_ids.append(sub_imports)

        sub_errors = self._bus.subscribe(
            handler=self._on_novel_error,
            event_types={"autofix.failed"},
        )
        self._subscription_ids.append(sub_errors)

        logger.info(
            "ResearchReconLoop started (subscriptions=%d)",
            len(self._subscription_ids),
        )

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self._running = False
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("ResearchReconLoop stopped")

    # -- event handlers ------------------------------------------------------

    async def _on_new_imports(self, event: Event) -> None:
        """Check for novel imports in newly indexed files."""
        payload = event.payload or {}
        path = payload.get("path", "")
        if not path:
            return

        # Extract imports from the event payload if available,
        # otherwise we treat the file itself as the topic.
        imports = payload.get("imports", [])
        if not imports:
            # If sentinel doesn't provide imports list, use path as key
            imports = [path]

        for imp in imports:
            self._imports_detected += 1
            if self._gate.is_novel(imp, "import"):
                self._gate.mark_researched(imp, "import")
                self._queue_research(
                    topic=f"import:{imp}",
                    context={"source_path": path, "import": imp},
                )

    async def _on_novel_error(self, event: Event) -> None:
        """Check if an autofix failure represents a novel error."""
        payload = event.payload or {}
        error_msg = payload.get("error", payload.get("message", ""))
        if not error_msg:
            return

        self._errors_detected += 1

        # Use first 200 chars as key to normalise minor variations
        error_key = error_msg[:200].strip()
        if self._gate.is_novel(error_key, "error"):
            self._gate.mark_researched(error_key, "error")
            self._queue_research(
                topic=f"error:{error_key[:80]}",
                context={
                    "error": error_msg,
                    "component": payload.get("component", "unknown"),
                    "file": payload.get("file", ""),
                },
            )

    # -- research queue (stub) -----------------------------------------------

    def _queue_research(self, topic: str, context: dict[str, Any]) -> None:
        """Queue a topic for research.

        Currently logs the research request.  A future implementation
        will dispatch to an LLM-backed research agent.
        """
        self._research_queued += 1
        logger.info(
            "ResearchReconLoop: queued research — topic=%s context_keys=%s",
            topic, list(context.keys()),
        )

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current status and statistics."""
        return {
            "running": self._running,
            "subscriptions": len(self._subscription_ids),
            "imports_detected": self._imports_detected,
            "errors_detected": self._errors_detected,
            "research_queued": self._research_queued,
            "novelty_gate": self._gate.get_stats(),
        }
