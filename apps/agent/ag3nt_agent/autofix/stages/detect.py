"""ReactiveDetector — collects errors from EventBus and creates FixContext batches."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from ag3nt_agent.autofix.context import FixContext

if TYPE_CHECKING:
    from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)

# Event types that indicate errors worth collecting.
_ERROR_EVENT_TYPES = frozenset({
    "tool.completed",
    "provider.error",
    "agent.error",
})


class ReactiveDetector:
    """Collects error events from the EventBus, debounces, and creates FixContext batches.

    Parameters
    ----------
    bus:
        An ``EventBus`` instance to subscribe to.  May be *None* if the
        detector is used in a standalone / testing context.
    debounce_s:
        Minimum seconds between drain operations (advisory; enforced by
        callers).
    max_batch:
        Maximum number of errors to include in a single FixContext.
    """

    def __init__(
        self,
        bus: Optional["EventBus"] = None,
        debounce_s: float = 2.0,
        max_batch: int = 10,
    ) -> None:
        self.debounce_s = debounce_s
        self.max_batch = max_batch
        self._error_buffer: list[dict[str, Any]] = []
        self._bus = bus

        if bus is not None:
            self._subscribe(bus)

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    def _subscribe(self, bus: "EventBus") -> None:
        """Register handlers on the event bus for error-related events."""
        for event_type in _ERROR_EVENT_TYPES:
            bus.subscribe(
                handler=self._on_error_event,
                event_types={event_type},
            )
        logger.debug("ReactiveDetector subscribed to %s", _ERROR_EVENT_TYPES)

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _on_error_event(self, event: "Event") -> None:
        """Append an incoming error event to the internal buffer."""
        payload = event.payload or {}

        # Only keep events that look like failures.
        # For tool.completed, we check for an explicit success=False or
        # the presence of an 'error' key.
        if event.event_type == "tool.completed":
            if payload.get("success", True) and "error" not in payload:
                return

        entry: dict[str, Any] = {
            "event_type": event.event_type,
            "source": event.source,
            "message": payload.get("error", payload.get("message", str(payload))),
            "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp),
            "payload": payload,
        }
        self._error_buffer.append(entry)
        logger.debug(
            "ReactiveDetector buffered error from %s (%d buffered)",
            event.source,
            len(self._error_buffer),
        )

    # ------------------------------------------------------------------
    # Drain / batch creation
    # ------------------------------------------------------------------

    def drain_to_context(self, max_batch: int | None = None) -> FixContext | None:
        """Create a :class:`FixContext` from buffered errors and clear the buffer.

        Returns *None* when the buffer is empty.
        """
        if not self._error_buffer:
            return None

        batch_size = max_batch or self.max_batch
        batch = self._error_buffer[:batch_size]
        self._error_buffer = self._error_buffer[batch_size:]

        ctx = FixContext(
            errors=batch,
            trigger="reactive_detection",
        )
        logger.info(
            "ReactiveDetector drained %d errors into FixContext %s",
            len(batch),
            ctx.id,
        )
        return ctx

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    async def process(self, ctx: FixContext) -> FixContext:
        """Pass-through — detection happens externally via ``drain_to_context``."""
        return ctx
