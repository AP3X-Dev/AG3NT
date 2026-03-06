"""
Reactive Security Loop — anomaly detection for tool usage patterns.

Subscribes to tool and approval events on the EventBus and detects
anomalous patterns such as:
- Rapid-fire tool calls (>10 calls in 5 seconds)
- Failure storms (>3 consecutive tool failures)
- Denied-then-retry (re-calling a denied tool within 30 seconds)

When an anomaly is detected, calls trust_tracker.set_anomaly() to force
HARD_CONFIRM approval for a cooldown period.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ReactiveSecurityLoop:
    """
    Anomaly detection loop that monitors tool usage patterns via EventBus.

    Thresholds:
        RAPID_FIRE_COUNT: Maximum tool calls allowed within the rapid-fire window.
        RAPID_FIRE_WINDOW_S: Time window (seconds) for rapid-fire detection.
        FAILURE_STORM_COUNT: Consecutive failures that trigger a failure storm.
        DENIED_RETRY_WINDOW_S: Window (seconds) within which retrying a denied
            tool triggers an anomaly.
    """

    RAPID_FIRE_COUNT: int = 10
    RAPID_FIRE_WINDOW_S: float = 5.0
    FAILURE_STORM_COUNT: int = 3
    DENIED_RETRY_WINDOW_S: float = 30.0

    def __init__(self, bus: Any, trust_tracker: Optional[Any] = None) -> None:
        """
        Initialize the security loop.

        Args:
            bus: EventBus instance to subscribe to for tool/approval events.
            trust_tracker: Optional trust tracker with a set_anomaly() method.
                When an anomaly is detected, set_anomaly() is called to force
                HARD_CONFIRM approval for a cooldown period.
        """
        self._bus = bus
        self._trust_tracker = trust_tracker

        # Tracking state
        self._call_times: deque = deque(maxlen=100)
        self._consecutive_failures: int = 0
        self._denied_tools: dict[str, float] = {}  # tool_name -> denial monotonic time

        # Thread safety
        self._lock = threading.Lock()

        # Subscription IDs for cleanup
        self._subscription_ids: list[str] = []

    async def start(self) -> None:
        """Subscribe to tool and approval events on the EventBus."""
        sub_tool_called = self._bus.subscribe(
            handler=self._on_tool_called,
            event_types={"tool.called"},
        )
        sub_tool_completed = self._bus.subscribe(
            handler=self._on_tool_completed,
            event_types={"tool.completed"},
        )
        sub_approval_denied = self._bus.subscribe(
            handler=self._on_approval_denied,
            event_types={"approval.denied"},
        )

        self._subscription_ids = [
            sub_tool_called,
            sub_tool_completed,
            sub_approval_denied,
        ]

        logger.info("ReactiveSecurityLoop started — subscribed to tool and approval events")

    async def stop(self) -> None:
        """Unsubscribe from all events on the EventBus."""
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)

        self._subscription_ids.clear()
        logger.info("ReactiveSecurityLoop stopped — unsubscribed from all events")

    async def _on_tool_called(self, event: Any) -> None:
        """
        Handle tool.called events.

        Checks for:
        1. Rapid-fire: More than RAPID_FIRE_COUNT calls within RAPID_FIRE_WINDOW_S.
        2. Denied-then-retry: Calling a tool that was denied within DENIED_RETRY_WINDOW_S.
        """
        now = time.monotonic()
        tool_name = event.payload.get("tool_name", "") if hasattr(event, "payload") else ""

        with self._lock:
            # Reset consecutive failures on any new call
            self._consecutive_failures = 0

            # Record call time
            self._call_times.append(now)

            # --- Rapid-fire detection ---
            cutoff = now - self.RAPID_FIRE_WINDOW_S
            recent_calls = sum(1 for t in self._call_times if t >= cutoff)
            if recent_calls > self.RAPID_FIRE_COUNT:
                self._trigger_anomaly(
                    f"Rapid-fire detected: {recent_calls} tool calls in "
                    f"{self.RAPID_FIRE_WINDOW_S}s (threshold: {self.RAPID_FIRE_COUNT})"
                )

            # --- Denied-then-retry detection ---
            if tool_name and tool_name in self._denied_tools:
                denial_time = self._denied_tools[tool_name]
                elapsed = now - denial_time
                if elapsed < self.DENIED_RETRY_WINDOW_S:
                    self._trigger_anomaly(
                        f"Denied-then-retry detected: tool '{tool_name}' was denied "
                        f"{elapsed:.1f}s ago (window: {self.DENIED_RETRY_WINDOW_S}s)"
                    )
                else:
                    # Denial has expired; clean it up
                    del self._denied_tools[tool_name]

    async def _on_tool_completed(self, event: Any) -> None:
        """
        Handle tool.completed events.

        Checks for failure storms: More than FAILURE_STORM_COUNT consecutive
        tool failures trigger an anomaly.
        """
        success = event.payload.get("success", True) if hasattr(event, "payload") else True

        with self._lock:
            if not success:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.FAILURE_STORM_COUNT:
                    self._trigger_anomaly(
                        f"Failure storm detected: {self._consecutive_failures} "
                        f"consecutive tool failures (threshold: {self.FAILURE_STORM_COUNT})"
                    )
            else:
                # Success resets the counter
                self._consecutive_failures = 0

    async def _on_approval_denied(self, event: Any) -> None:
        """
        Handle approval.denied events.

        Records the denial time for each tool so that denied-then-retry
        detection can fire if the same tool is called again within the window.
        """
        tool_name = event.payload.get("tool_name", "") if hasattr(event, "payload") else ""

        if tool_name:
            with self._lock:
                self._denied_tools[tool_name] = time.monotonic()

    def _trigger_anomaly(self, reason: str) -> None:
        """
        Trigger an anomaly response.

        Logs a warning and, if a trust tracker is available, calls
        set_anomaly() to force HARD_CONFIRM for a cooldown period.

        Note: Callers must already hold self._lock.

        Args:
            reason: Human-readable description of the anomaly.
        """
        logger.warning("SECURITY ANOMALY: %s", reason)

        if self._trust_tracker is not None:
            try:
                self._trust_tracker.set_anomaly()
            except Exception:
                logger.exception("Failed to set anomaly on trust tracker")
