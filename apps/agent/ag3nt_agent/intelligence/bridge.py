"""EventBridge — bidirectional main<->worker thread event forwarding.

Prevents echo by marking bridged events with a _bridged flag.
"""

import asyncio
import logging
from typing import Any

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger("ag3nt.intelligence.bridge")

_BRIDGED_KEY = "_bridged"


class EventBridge:
    """Forwards events between main and worker EventBus instances."""

    def __init__(
        self,
        main_bus: EventBus,
        worker_bus: EventBus,
        forward_patterns: list[str] | None = None,
    ) -> None:
        self._main_bus = main_bus
        self._worker_bus = worker_bus
        self._forward_patterns = forward_patterns or [
            "file.*", "tool.*", "quality.*", "sentinel.*",
            "autofix.*", "recovery.*", "meta.*",
        ]
        self._sub_ids: list[str] = []

    async def start(self) -> None:
        # Main -> Worker
        for pattern in self._forward_patterns:
            sub_id = self._main_bus.subscribe_pattern(
                pattern, self._forward_to_worker
            )
            self._sub_ids.append(sub_id)
        # Worker -> Main
        for pattern in self._forward_patterns:
            sub_id = self._worker_bus.subscribe_pattern(
                pattern, self._forward_to_main
            )
            self._sub_ids.append(sub_id)
        logger.info("EventBridge started with %d patterns", len(self._forward_patterns))

    async def stop(self) -> None:
        for sub_id in self._sub_ids:
            self._main_bus.unsubscribe(sub_id)
            self._worker_bus.unsubscribe(sub_id)
        self._sub_ids.clear()

    async def _forward_to_worker(self, event: Event) -> None:
        if event.metadata.get(_BRIDGED_KEY):
            return  # Prevent echo
        bridged = Event(
            event_type=event.event_type,
            source=event.source,
            payload=event.payload,
            priority=event.priority,
            metadata={**event.metadata, _BRIDGED_KEY: True},
        )
        await self._worker_bus.publish(bridged)

    async def _forward_to_main(self, event: Event) -> None:
        if event.metadata.get(_BRIDGED_KEY):
            return
        bridged = Event(
            event_type=event.event_type,
            source=event.source,
            payload=event.payload,
            priority=event.priority,
            metadata={**event.metadata, _BRIDGED_KEY: True},
        )
        await self._main_bus.publish(bridged)
