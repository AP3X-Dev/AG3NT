"""Phase Hook Manager for AG3NT lifecycle events.

Provides a priority-ordered hook system with 8 lifecycle phases and
cancellation support. Hooks are async callables that receive a context
dict and return None (continue) or CANCEL (abort the phase).

Phases:
    PRE_TURN / POST_TURN       — Before/after each agent turn
    PRE_TOOL / POST_TOOL       — Before/after each tool invocation
    PRE_COMPACTION / POST_COMPACTION — Before/after context compaction
    PRE_MEMORY_STORE / POST_MEMORY_STORE — Before/after memory persistence

Usage:
    from ag3nt_agent.hooks import PhaseHookManager, HookPhase, CANCEL

    mgr = PhaseHookManager()
    mgr.start()

    async def my_hook(ctx: dict) -> str | None:
        if ctx.get("dangerous"):
            return CANCEL
        return None

    mgr.register(HookPhase.PRE_TOOL, my_hook, priority=10, name="safety_check")
    result = await mgr.run(HookPhase.PRE_TOOL, {"tool": "exec_command"})
    if result.cancelled:
        print(f"Blocked by {result.cancelled_by}: {result.cancel_reason}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger("ag3nt.hooks")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANCEL = "CANCEL"
"""Sentinel value returned by a hook to cancel the current phase."""

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

HookFn = Callable[[dict[str, Any]], Awaitable[str | None]]
"""Async callable that receives a context dict and returns None or CANCEL."""

# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class HookPhase(str, Enum):
    """Lifecycle phases where hooks can be registered."""

    PRE_TURN = "pre_turn"
    POST_TURN = "post_turn"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    PRE_COMPACTION = "pre_compaction"
    POST_COMPACTION = "post_compaction"
    PRE_MEMORY_STORE = "pre_memory_store"
    POST_MEMORY_STORE = "post_memory_store"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HookRegistration:
    """A registered hook bound to a lifecycle phase."""

    phase: HookPhase
    handler: HookFn
    priority: int = 100  # lower = runs first
    name: str = ""
    source: str = "static"  # "static" or "dynamic"


@dataclass
class HookResult:
    """Outcome of running all hooks for a single phase invocation."""

    cancelled: bool = False
    cancel_reason: str = ""
    cancelled_by: str = ""
    results: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class PhaseHookManager:
    """Registry and executor for lifecycle hooks.

    Hooks are executed in priority order (lower priority number runs first).
    If any hook returns :data:`CANCEL`, execution stops immediately and
    the returned :class:`HookResult` has ``cancelled=True``.
    """

    def __init__(self) -> None:
        self._hooks: dict[HookPhase, list[HookRegistration]] = {
            phase: [] for phase in HookPhase
        }
        self._running: bool = False

    # -- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Activate the hook manager.  Hooks will only execute when running."""
        self._running = True
        logger.info("PhaseHookManager started")

    def stop(self) -> None:
        """Deactivate the hook manager.  Subsequent :meth:`run` calls are no-ops."""
        self._running = False
        logger.info("PhaseHookManager stopped")

    # -- Registration --------------------------------------------------------

    def register(
        self,
        phase: HookPhase,
        handler: HookFn,
        priority: int = 100,
        name: str = "",
        source: str = "static",
    ) -> None:
        """Register a hook for *phase*.

        Parameters
        ----------
        phase:
            The lifecycle phase to attach the hook to.
        handler:
            Async callable ``(ctx) -> str | None``.
        priority:
            Execution order — lower values run first.  Default ``100``.
        name:
            Human-readable identifier.  Used by :meth:`unregister`.
            Defaults to the handler's ``__name__`` attribute.
        source:
            Origin tag — ``"static"`` (code-registered) or ``"dynamic"``
            (registered at runtime, e.g. from config).
        """
        resolved_name = name or getattr(handler, "__name__", "")
        reg = HookRegistration(
            phase=phase,
            handler=handler,
            priority=priority,
            name=resolved_name,
            source=source,
        )
        self._hooks[phase].append(reg)
        self._hooks[phase].sort(key=lambda r: r.priority)
        logger.debug(
            "Registered hook %r on %s (priority=%d, source=%s)",
            resolved_name,
            phase.value,
            priority,
            source,
        )

    def unregister(self, phase: HookPhase, name: str) -> bool:
        """Remove a hook by *name* from *phase*.

        Returns ``True`` if a hook was found and removed.
        """
        hooks = self._hooks[phase]
        for i, reg in enumerate(hooks):
            if reg.name == name:
                hooks.pop(i)
                logger.debug("Unregistered hook %r from %s", name, phase.value)
                return True
        return False

    # -- Execution -----------------------------------------------------------

    async def run(self, phase: HookPhase, context: dict[str, Any]) -> HookResult:
        """Execute all hooks registered for *phase* in priority order.

        If the manager is not running, returns an empty :class:`HookResult`.
        If any hook returns :data:`CANCEL`, remaining hooks are skipped and
        the result is marked as cancelled.

        Parameters
        ----------
        phase:
            The lifecycle phase to execute.
        context:
            Arbitrary data dict passed to each hook handler.

        Returns
        -------
        HookResult
            Aggregated execution result.
        """
        result = HookResult()

        if not self._running:
            return result

        for reg in self._hooks[phase]:
            try:
                ret = await reg.handler(context)
            except Exception:
                logger.exception(
                    "Hook %r on %s raised an exception", reg.name, phase.value
                )
                result.results.append(None)
                continue

            result.results.append(ret)

            if ret == CANCEL:
                result.cancelled = True
                result.cancelled_by = reg.name
                result.cancel_reason = (
                    f"Hook '{reg.name}' cancelled phase {phase.value}"
                )
                logger.warning(
                    "Phase %s cancelled by hook %r (priority=%d)",
                    phase.value,
                    reg.name,
                    reg.priority,
                )
                break

        return result

    # -- Introspection -------------------------------------------------------

    def get_hooks(self, phase: HookPhase) -> list[dict[str, Any]]:
        """Return a serialisable list of hooks registered for *phase*."""
        return [
            {
                "name": reg.name,
                "priority": reg.priority,
                "source": reg.source,
                "phase": reg.phase.value,
            }
            for reg in self._hooks[phase]
        ]

    def get_stats(self) -> dict[str, int]:
        """Return a mapping of phase name to hook count."""
        return {phase.value: len(hooks) for phase, hooks in self._hooks.items()}
