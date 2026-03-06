"""MCPHealthLoop — periodic health monitoring for MCP servers.

Periodically pings connected MCP (Model Context Protocol) servers and attempts
reconnection with exponential backoff when failures are detected.  Server status
is tracked as one of ``healthy``, ``degraded``, or ``disconnected``.

Events consumed:
    mcp.server.connected    — payload: {server}
    mcp.server.disconnected — payload: {server}
    mcp.pong                — payload: {server}

Events emitted:
    mcp.ping                — payload: {server}
    mcp.health_failed       — payload: {server, consecutive_failures}
    mcp.reconnect_requested — payload: {server, attempt}
    mcp.reconnect_exhausted — payload: {server, attempts}
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.loops.mcp_health")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class MCPHealthConfig:
    """Configuration for MCPHealthLoop."""

    check_interval: float = 30.0
    max_reconnect_attempts: int = 4
    backoff_schedule: list[float] = field(
        default_factory=lambda: [5.0, 15.0, 60.0, 300.0]
    )
    ping_timeout: float = 5.0
    reconnect_timeout: float = 15.0


# ------------------------------------------------------------------
# MCPHealthLoop
# ------------------------------------------------------------------

class MCPHealthLoop:
    """Background monitoring of MCP (Model Context Protocol) servers.

    Periodically pings connected servers and attempts reconnection with
    exponential backoff when failures are detected.

    Usage::

        bus = EventBus()
        loop = MCPHealthLoop(bus)
        await loop.start()
        # ... later ...
        await loop.stop()
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        config: MCPHealthConfig | None = None,
    ) -> None:
        cfg = config or MCPHealthConfig()
        self._bus = bus
        self._interval = cfg.check_interval
        self._max_reconnects = cfg.max_reconnect_attempts
        self._backoff = list(cfg.backoff_schedule)
        self._ping_timeout = cfg.ping_timeout
        self._reconnect_timeout = cfg.reconnect_timeout

        # server_id -> {status, last_check, consecutive_failures}
        self._server_status: dict[str, dict[str, Any]] = {}
        self._reconnect_counts: dict[str, int] = {}
        self._task: Optional[asyncio.Task[None]] = None
        self._sub_ids: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Start the background health check loop."""
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_connected,
                event_types={"mcp.server.connected"},
            )
        )
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_disconnected,
                event_types={"mcp.server.disconnected"},
            )
        )
        self._task = asyncio.create_task(
            self._health_loop(), name="mcp_health_loop"
        )
        logger.info("MCPHealthLoop started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        """Stop the health check loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for sub_id in self._sub_ids:
            self._bus.unsubscribe(sub_id)
        self._sub_ids.clear()
        logger.info("MCPHealthLoop stopped")

    # -- event handlers ------------------------------------------------------

    async def _on_connected(self, event: Event) -> None:
        server_id = event.payload.get("server", "")
        if server_id:
            self._server_status[server_id] = {
                "status": "healthy",
                "last_check": time.monotonic(),
                "consecutive_failures": 0,
            }
            self._reconnect_counts[server_id] = 0

    async def _on_disconnected(self, event: Event) -> None:
        server_id = event.payload.get("server", "")
        if server_id and server_id in self._server_status:
            self._server_status[server_id]["status"] = "disconnected"

    # -- background loop -----------------------------------------------------

    async def _health_loop(self) -> None:
        """Periodic health check of all connected servers."""
        while True:
            try:
                await asyncio.sleep(self._interval)
                for server_id in list(self._server_status.keys()):
                    status = self._server_status[server_id]
                    if status["status"] == "disconnected":
                        reconnected = await self._attempt_reconnect(server_id)
                        if reconnected:
                            status["status"] = "healthy"
                            status["consecutive_failures"] = 0
                        continue

                    healthy = await self._check_server(server_id)
                    status["last_check"] = time.monotonic()
                    if healthy:
                        status["consecutive_failures"] = 0
                        status["status"] = "healthy"
                    else:
                        status["consecutive_failures"] = (
                            status.get("consecutive_failures", 0) + 1
                        )
                        if status["consecutive_failures"] >= 3:
                            status["status"] = "disconnected"
                            fail_event = Event(
                                event_type="mcp.health_failed",
                                source="mcp_health",
                                payload={
                                    "server": server_id,
                                    "consecutive_failures": status[
                                        "consecutive_failures"
                                    ],
                                },
                                priority=EventPriority.HIGH,
                            )
                            await self._bus.publish(fail_event)
                        else:
                            status["status"] = "degraded"
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "MCPHealthLoop iteration error: %s", exc, exc_info=True
                )
                await asyncio.sleep(5.0)

    async def _check_server(self, server_id: str) -> bool:
        """Ping a single MCP server.  Returns True if healthy.

        Emits ``mcp.ping`` and waits for a corresponding ``mcp.pong`` within
        a short timeout.  In the absence of a real transport layer we treat
        the server as healthy unless previously disconnected.
        """
        pong_received = asyncio.Event()

        async def _on_pong(event: Event) -> None:
            if event.payload.get("server") == server_id:
                pong_received.set()

        sub_id = self._bus.subscribe(
            _on_pong, event_types={"mcp.pong"}
        )
        try:
            ping_event = Event(
                event_type="mcp.ping",
                source="mcp_health",
                payload={"server": server_id},
            )
            await self._bus.publish(ping_event)
            try:
                await asyncio.wait_for(
                    pong_received.wait(), timeout=self._ping_timeout
                )
                return True
            except asyncio.TimeoutError:
                # No pong — assume still healthy if status was healthy
                # (ping may not be implemented by all servers)
                return (
                    self._server_status.get(server_id, {}).get("status")
                    == "healthy"
                )
        finally:
            self._bus.unsubscribe(sub_id)

    async def _attempt_reconnect(self, server_id: str) -> bool:
        """Try to reconnect a disconnected server with exponential backoff."""
        attempt = self._reconnect_counts.get(server_id, 0)
        if attempt >= self._max_reconnects:
            logger.warning(
                "MCP server %s: max reconnect attempts (%d) exhausted",
                server_id,
                self._max_reconnects,
            )
            exhausted_event = Event(
                event_type="mcp.reconnect_exhausted",
                source="mcp_health",
                payload={"server": server_id, "attempts": attempt},
                priority=EventPriority.HIGH,
            )
            await self._bus.publish(exhausted_event)
            return False

        backoff_idx = min(attempt, len(self._backoff) - 1)
        delay = self._backoff[backoff_idx]
        logger.info(
            "MCP server %s: reconnect attempt %d (backoff=%.1fs)",
            server_id,
            attempt + 1,
            delay,
        )
        await asyncio.sleep(delay)

        # Emit reconnect request and check for success
        reconnect_done = asyncio.Event()
        success_flag: list[bool] = [False]

        async def _on_reconnected(event: Event) -> None:
            if event.payload.get("server") == server_id:
                success_flag[0] = True
                reconnect_done.set()

        sub_id = self._bus.subscribe(
            _on_reconnected, event_types={"mcp.server.connected"}
        )
        try:
            req_event = Event(
                event_type="mcp.reconnect_requested",
                source="mcp_health",
                payload={"server": server_id, "attempt": attempt + 1},
            )
            await self._bus.publish(req_event)
            try:
                await asyncio.wait_for(
                    reconnect_done.wait(), timeout=self._reconnect_timeout
                )
            except asyncio.TimeoutError:
                pass
        finally:
            self._bus.unsubscribe(sub_id)

        self._reconnect_counts[server_id] = attempt + 1
        if success_flag[0]:
            self._reconnect_counts[server_id] = 0
            logger.info("MCP server %s: reconnected successfully", server_id)
        return success_flag[0]

    # -- introspection -------------------------------------------------------

    def get_server_status(self) -> dict[str, dict[str, Any]]:
        """Return a copy of all tracked server statuses."""
        return {k: dict(v) for k, v in self._server_status.items()}

    def register_server(self, server_id: str) -> None:
        """Manually register a server for tracking (e.g. at startup)."""
        if server_id not in self._server_status:
            self._server_status[server_id] = {
                "status": "healthy",
                "last_check": time.monotonic(),
                "consecutive_failures": 0,
            }
            self._reconnect_counts[server_id] = 0
