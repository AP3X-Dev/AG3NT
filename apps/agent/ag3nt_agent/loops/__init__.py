"""Reactive loop engine for AG3NT autonomous system."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ag3nt_agent.loops.compaction import (  # noqa: F401
    CompactionAgent,
    CompactionLoopConfig,
    CompactionOrchestrator,
    CompactionRecord,
    CompactionTier,
    CompactionTrigger,
)
from ag3nt_agent.loops.failover import ProviderFailoverLoop  # noqa: F401
from ag3nt_agent.loops.lane_queue import Lane, LaneAwareQueue, LaneItem  # noqa: F401
from ag3nt_agent.loops.lsp_health import LSPHealthLoop  # noqa: F401
from ag3nt_agent.loops.mcp_health import MCPHealthLoop  # noqa: F401
from ag3nt_agent.loops.peva import PEVAOrchestrator

logger = logging.getLogger("ag3nt.loops")


# ---------------------------------------------------------------
#  LoopManager Configuration
# ---------------------------------------------------------------

@dataclass
class LoopManagerConfig:
    """Central configuration for all managed reactive loops."""

    quality_gate_enabled: bool = True
    recovery_enabled: bool = True
    snapshot_enabled: bool = True
    failover_enabled: bool = True
    mcp_health_enabled: bool = True
    lsp_health_enabled: bool = True
    security_enabled: bool = True
    compaction_enabled: bool = True

    # Provider failover needs a fallback chain; empty = skip
    failover_chain: list[str] = field(default_factory=list)


# ---------------------------------------------------------------
#  LoopManager
# ---------------------------------------------------------------

class LoopManager:
    """
    Central manager for all reactive loops in the AG3NT autonomous system.

    Provides ``start_all()`` / ``stop_all()`` lifecycle and ``get_status()``
    introspection.  Each loop is created internally from an EventBus and
    optional configuration.

    Managed loops:
        - QualityGateLoop
        - SelfHealingRecovery
        - SnapshotAutoTrigger
        - ProviderFailoverLoop
        - MCPHealthLoop
        - LSPHealthLoop
        - ReactiveSecurityLoop
        - CompactionTrigger
        - CompactionOrchestrator

    Usage::

        from ag3nt_agent.autonomous.event_bus import EventBus
        from ag3nt_agent.loops import LoopManager

        bus = EventBus()
        manager = LoopManager(bus)
        await manager.start_all()
        status = manager.get_status()
        await manager.stop_all()
    """

    def __init__(
        self,
        bus: Any,
        config: Optional[LoopManagerConfig] = None,
    ) -> None:
        self._bus = bus
        self._config = config or LoopManagerConfig()
        self._loops: dict[str, Any] = {}
        self._running = False

        self._build_loops()

    # -- internal construction ----------------------------------

    def _build_loops(self) -> None:
        """Instantiate each loop based on configuration flags.

        Imports are done lazily inside this method so that missing
        optional dependencies don't break the entire module.
        """
        cfg = self._config

        if cfg.quality_gate_enabled:
            try:
                from ag3nt_agent.loops.quality_gate import QualityGateLoop
                self._loops["QualityGateLoop"] = QualityGateLoop(self._bus)
            except Exception as exc:
                logger.warning("Failed to create QualityGateLoop: %s", exc)

        if cfg.recovery_enabled:
            try:
                from ag3nt_agent.loops.recovery import SelfHealingRecovery
                self._loops["SelfHealingRecovery"] = SelfHealingRecovery(self._bus)
            except Exception as exc:
                logger.warning("Failed to create SelfHealingRecovery: %s", exc)

        if cfg.snapshot_enabled:
            try:
                from ag3nt_agent.loops.snapshot import SnapshotAutoTrigger
                self._loops["SnapshotAutoTrigger"] = SnapshotAutoTrigger(self._bus)
            except Exception as exc:
                logger.warning("Failed to create SnapshotAutoTrigger: %s", exc)

        if cfg.failover_enabled and cfg.failover_chain:
            try:
                self._loops["ProviderFailoverLoop"] = ProviderFailoverLoop(
                    fallback_chain=cfg.failover_chain,
                    bus=self._bus,
                )
            except Exception as exc:
                logger.warning("Failed to create ProviderFailoverLoop: %s", exc)

        if cfg.mcp_health_enabled:
            try:
                self._loops["MCPHealthLoop"] = MCPHealthLoop(self._bus)
            except Exception as exc:
                logger.warning("Failed to create MCPHealthLoop: %s", exc)

        if cfg.lsp_health_enabled:
            try:
                self._loops["LSPHealthLoop"] = LSPHealthLoop(self._bus)
            except Exception as exc:
                logger.warning("Failed to create LSPHealthLoop: %s", exc)

        if cfg.security_enabled:
            try:
                from ag3nt_agent.approval.security import ReactiveSecurityLoop
                self._loops["ReactiveSecurityLoop"] = ReactiveSecurityLoop(self._bus)
            except Exception as exc:
                logger.warning("Failed to create ReactiveSecurityLoop: %s", exc)

        if cfg.compaction_enabled:
            try:
                compaction_config = CompactionLoopConfig()
                self._loops["CompactionTrigger"] = CompactionTrigger(
                    compaction_config, self._bus,
                )
                self._loops["CompactionOrchestrator"] = CompactionOrchestrator(
                    compaction_config, self._bus,
                )
            except Exception as exc:
                logger.warning("Failed to create compaction loops: %s", exc)

    # -- lifecycle ----------------------------------------------

    async def start_all(self) -> None:
        """Start every managed loop that has a ``start()`` coroutine."""
        if self._running:
            logger.warning("LoopManager is already running")
            return

        started: list[str] = []
        for name, loop in self._loops.items():
            start_fn = getattr(loop, "start", None)
            if start_fn is None or not asyncio.iscoroutinefunction(start_fn):
                logger.debug("Skipping %s (no async start method)", name)
                continue
            try:
                await start_fn()
                started.append(name)
            except Exception as exc:
                logger.error("Failed to start %s: %s", name, exc)

        self._running = True
        logger.info(
            "LoopManager started %d/%d loops: %s",
            len(started),
            len(self._loops),
            ", ".join(started) if started else "(none)",
        )

    async def stop_all(self) -> None:
        """Stop every managed loop that has a ``stop()`` coroutine."""
        if not self._running:
            return

        stopped: list[str] = []
        for name, loop in self._loops.items():
            stop_fn = getattr(loop, "stop", None)
            if stop_fn is None or not asyncio.iscoroutinefunction(stop_fn):
                continue
            try:
                await stop_fn()
                stopped.append(name)
            except Exception as exc:
                logger.error("Failed to stop %s: %s", name, exc)

        self._running = False
        logger.info(
            "LoopManager stopped %d loops: %s",
            len(stopped),
            ", ".join(stopped) if stopped else "(none)",
        )

    # -- introspection ------------------------------------------

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Return the running state of each managed loop.

        Returns a dict keyed by loop name, with each value containing
        at least ``{"running": bool}``.  If a loop exposes additional
        status via a ``get_status()`` or ``get_*()`` method, those
        details are merged in.
        """
        result: dict[str, dict[str, Any]] = {}

        for name, loop in self._loops.items():
            entry: dict[str, Any] = {"running": self._running}

            # Try to pull extra status from the loop itself
            for method_name in ("get_status", "get_server_status", "get_health_report", "get_server_health"):
                fn = getattr(loop, method_name, None)
                if fn is not None and callable(fn):
                    try:
                        extra = fn()
                        if isinstance(extra, dict):
                            entry["details"] = extra
                    except Exception:
                        pass
                    break

            result[name] = entry

        return result

    @property
    def running(self) -> bool:
        """True if ``start_all()`` has been called and ``stop_all()`` has not."""
        return self._running

    @property
    def loop_names(self) -> list[str]:
        """Names of all managed loops (whether started or not)."""
        return list(self._loops.keys())


__all__ = [
    "CompactionAgent",
    "CompactionLoopConfig",
    "CompactionOrchestrator",
    "CompactionRecord",
    "CompactionTier",
    "CompactionTrigger",
    "Lane",
    "LaneAwareQueue",
    "LaneItem",
    "LoopManager",
    "LoopManagerConfig",
    "LSPHealthLoop",
    "MCPHealthLoop",
    "PEVAOrchestrator",
    "ProviderFailoverLoop",
]
