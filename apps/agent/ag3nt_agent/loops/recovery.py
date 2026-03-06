"""
Self-Healing Recovery with 4-Level Cascade for AG3NT Autonomous System

When a component fails, the recovery system tries increasingly aggressive
recovery strategies:

    1. Retry     -- re-run the same operation (up to 3 times)
    2. Failover  -- switch to a backup / alternative (up to 2 times)
    3. Rollback  -- revert to last known good state (up to 1 time)
    4. Escalate  -- alert a human / higher-level system (up to 1 time)

Successful recovery at *any* level resets all counters so the component
starts fresh on the next failure.
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Recovery levels
# ------------------------------------------------------------------

class RecoveryLevel(IntEnum):
    """Ordered recovery levels — tried from least to most aggressive."""

    RETRY = 1
    FAILOVER = 2
    ROLLBACK = 3
    ESCALATE = 4


# ------------------------------------------------------------------
# Per-component recovery state
# ------------------------------------------------------------------

@dataclass
class RecoveryState:
    """Tracks how many times each recovery level has been attempted."""

    retry_count: int = 0
    failover_count: int = 0
    rollback_count: int = 0
    escalate_count: int = 0

    # Maximum attempts per level before moving to the next
    MAX_RETRY: int = field(default=3, repr=False)
    MAX_FAILOVER: int = field(default=2, repr=False)
    MAX_ROLLBACK: int = field(default=1, repr=False)
    MAX_ESCALATE: int = field(default=1, repr=False)

    def next_level(self) -> Optional[RecoveryLevel]:
        """Return the next available recovery level, or ``None`` if exhausted."""
        if self.retry_count < self.MAX_RETRY:
            return RecoveryLevel.RETRY
        if self.failover_count < self.MAX_FAILOVER:
            return RecoveryLevel.FAILOVER
        if self.rollback_count < self.MAX_ROLLBACK:
            return RecoveryLevel.ROLLBACK
        if self.escalate_count < self.MAX_ESCALATE:
            return RecoveryLevel.ESCALATE
        return None

    def record(self, level: RecoveryLevel) -> None:
        """Increment the counter for *level*."""
        if level == RecoveryLevel.RETRY:
            self.retry_count += 1
        elif level == RecoveryLevel.FAILOVER:
            self.failover_count += 1
        elif level == RecoveryLevel.ROLLBACK:
            self.rollback_count += 1
        elif level == RecoveryLevel.ESCALATE:
            self.escalate_count += 1

    def reset(self) -> None:
        """Reset all counters to zero (called after successful recovery)."""
        self.retry_count = 0
        self.failover_count = 0
        self.rollback_count = 0
        self.escalate_count = 0


# ------------------------------------------------------------------
# Type alias for recovery handler functions
# ------------------------------------------------------------------

RecoveryHandler = Callable[[str, Exception], Coroutine[Any, Any, bool]]


# ------------------------------------------------------------------
# SelfHealingRecovery
# ------------------------------------------------------------------

class SelfHealingRecovery:
    """
    4-level cascade recovery system.

    Usage::

        bus = EventBus()
        recovery = SelfHealingRecovery(bus)

        async def my_retry(component: str, error: Exception) -> bool:
            # ... attempt to retry ...
            return True  # or False

        recovery.set_handlers(retry=my_retry, failover=my_failover)

        ok = await recovery.recover("web-server", some_exception)
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._states: dict[str, RecoveryState] = {}

        # Handler functions (all async (component, error) -> bool)
        self._retry_handler: Optional[RecoveryHandler] = None
        self._failover_handler: Optional[RecoveryHandler] = None
        self._rollback_handler: Optional[RecoveryHandler] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_handlers(
        self,
        retry: Optional[RecoveryHandler] = None,
        failover: Optional[RecoveryHandler] = None,
        rollback: Optional[RecoveryHandler] = None,
    ) -> None:
        """Register async handler functions for recovery levels."""
        if retry is not None:
            self._retry_handler = retry
        if failover is not None:
            self._failover_handler = failover
        if rollback is not None:
            self._rollback_handler = rollback

    # ------------------------------------------------------------------
    # Main recovery entry point
    # ------------------------------------------------------------------

    async def recover(self, component: str, error: Exception) -> bool:
        """
        Attempt to recover *component* from *error* using the cascade.

        Walks through recovery levels in order. On success at any level,
        resets all counters and returns ``True``. Returns ``False`` when
        all levels are exhausted.
        """
        state = self._states.setdefault(component, RecoveryState())

        # Emit recovery.started
        await self._emit(
            "recovery.started",
            component,
            error,
            priority=EventPriority.HIGH,
        )

        while True:
            level = state.next_level()
            if level is None:
                # All levels exhausted
                logger.error(
                    "Recovery exhausted for component '%s': %s",
                    component,
                    error,
                )
                await self._emit(
                    "recovery.failed",
                    component,
                    error,
                    priority=EventPriority.CRITICAL,
                )
                return False

            state.record(level)
            logger.info(
                "Attempting %s recovery for '%s' (error: %s)",
                level.name,
                component,
                error,
            )

            success = await self._execute_level(level, component, error)

            if success:
                logger.info(
                    "Recovery succeeded at %s level for '%s'",
                    level.name,
                    component,
                )
                state.reset()
                await self._emit(
                    "recovery.succeeded",
                    component,
                    error,
                    priority=EventPriority.HIGH,
                    extra={"level": level.name},
                )
                return True

            # Level failed — loop continues to try the next available level

    # ------------------------------------------------------------------
    # Level execution
    # ------------------------------------------------------------------

    async def _execute_level(
        self,
        level: RecoveryLevel,
        component: str,
        error: Exception,
    ) -> bool:
        """
        Execute recovery at the given *level*.

        Returns ``True`` on success, ``False`` on failure.
        ESCALATE always emits a ``recovery.escalated`` event.
        """
        try:
            if level == RecoveryLevel.RETRY:
                if self._retry_handler:
                    return await self._retry_handler(component, error)
                return False

            elif level == RecoveryLevel.FAILOVER:
                if self._failover_handler:
                    return await self._failover_handler(component, error)
                return False

            elif level == RecoveryLevel.ROLLBACK:
                if self._rollback_handler:
                    return await self._rollback_handler(component, error)
                return False

            elif level == RecoveryLevel.ESCALATE:
                await self._emit(
                    "recovery.escalated",
                    component,
                    error,
                    priority=EventPriority.CRITICAL,
                )
                # Escalation itself is considered a "handled" outcome —
                # return False so the cascade reports overall failure
                # (a human must intervene).
                return False

        except Exception as exc:
            logger.warning(
                "Handler for %s raised an exception for '%s': %s",
                level.name,
                component,
                exc,
            )
            return False

    # ------------------------------------------------------------------
    # Event emission helpers
    # ------------------------------------------------------------------

    async def _emit(
        self,
        event_type: str,
        component: str,
        error: Exception,
        priority: EventPriority = EventPriority.HIGH,
        extra: Optional[dict] = None,
    ) -> None:
        """Publish a recovery event on the bus."""
        payload: dict[str, Any] = {
            "component": component,
            "error": str(error),
            "error_type": type(error).__name__,
        }
        if extra:
            payload.update(extra)

        event = Event(
            event_type=event_type,
            source="self_healing_recovery",
            payload=payload,
            priority=priority,
        )
        await self._bus.publish(event)
