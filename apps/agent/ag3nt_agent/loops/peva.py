"""
PEVA Orchestrator — Plan, Execute, Verify, Adapt cycle for AG3NT.

Implements a structured four-phase loop:

    1. **Plan**    — classify task complexity and build a dependency-aware
                     execution plan (trivial / moderate / complex).
    2. **Execute** — run plan steps, respecting declared dependencies and
                     executing independent steps in parallel where possible.
    3. **Verify**  — check results of each step (success / partial / failure).
    4. **Adapt**   — on failure, apply one of three strategies:
                     ``targeted_fix``, ``revise_approach``, or ``expand_plan``.
                     Adaptation is capped at ``max_adapt_rounds`` (default 3).

Events are emitted via the :class:`EventBus` at every state transition so
that other subsystems (quality gates, snapshots, recovery) can react.

Usage::

    bus = EventBus()
    peva = PEVAOrchestrator(bus)
    await peva.start()

    result = await peva.run(steps=[...])
    # result.success / result.history / ...

    await peva.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional
from uuid import uuid4

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.loops.peva")


# =========================================================================
# Enums
# =========================================================================

class Complexity(str, Enum):
    """Task complexity classification."""

    TRIVIAL = "trivial"
    MODERATE = "moderate"
    COMPLEX = "complex"


class Phase(str, Enum):
    """PEVA lifecycle phases."""

    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    ADAPT = "adapt"


class StepStatus(str, Enum):
    """Outcome of an individual step execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class AdaptStrategy(str, Enum):
    """Adaptation strategies available when verification detects failures."""

    TARGETED_FIX = "targeted_fix"
    REVISE_APPROACH = "revise_approach"
    EXPAND_PLAN = "expand_plan"


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class PlanStep:
    """A single step in a PEVA execution plan.

    Parameters
    ----------
    step_id:
        Unique identifier for this step.
    description:
        Human-readable summary of what this step does.
    action:
        Async callable ``(context: dict) -> dict`` that performs the work.
        Must return a dict with at least ``{"success": bool}``.
    depends_on:
        Set of step IDs that must complete before this one can start.
    timeout_seconds:
        Maximum wall-clock time for this step.
    metadata:
        Arbitrary key-value metadata attached to the step.
    """

    step_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    action: Optional[Callable[..., Coroutine[Any, Any, dict]]] = None
    depends_on: set[str] = field(default_factory=set)
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Runtime state (mutated by the orchestrator)
    status: StepStatus = StepStatus.PENDING
    result: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at is not None and self.finished_at is not None:
            return (self.finished_at - self.started_at) * 1000.0
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "depends_on": sorted(self.depends_on),
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class PEVAConfig:
    """Configuration for the :class:`PEVAOrchestrator`."""

    enabled: bool = True

    # Complexity thresholds (step count)
    trivial_threshold: int = 2
    moderate_threshold: int = 6

    # Adaptation limits
    max_adapt_rounds: int = 3

    # Parallelism
    max_parallel: int = 4

    # Default step timeout (seconds)
    default_step_timeout: float = 60.0

    # Overall cycle timeout (seconds) — 0 means no limit
    cycle_timeout: float = 0.0


@dataclass
class AdaptationRecord:
    """Records a single adaptation attempt for audit / telemetry."""

    round_number: int
    strategy: AdaptStrategy
    failed_steps: list[str]
    outcome: str = ""  # "success", "partial", "failed"
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "strategy": self.strategy.value,
            "failed_steps": self.failed_steps,
            "outcome": self.outcome,
        }


@dataclass
class PEVAResult:
    """Outcome of a complete PEVA cycle."""

    cycle_id: str = field(default_factory=lambda: str(uuid4()))
    success: bool = False
    complexity: Complexity = Complexity.TRIVIAL
    steps: list[PlanStep] = field(default_factory=list)
    adaptations: list[AdaptationRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: Optional[float] = None
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.finished_at is not None:
            return (self.finished_at - self.started_at) * 1000.0
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "success": self.success,
            "complexity": self.complexity.value,
            "steps": [s.to_dict() for s in self.steps],
            "adaptations": [a.to_dict() for a in self.adaptations],
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# =========================================================================
# Orchestrator
# =========================================================================

class PEVAOrchestrator:
    """Plan-Execute-Verify-Adapt orchestrator.

    Drives a list of :class:`PlanStep` objects through the four PEVA phases,
    automatically adapting on failure up to ``max_adapt_rounds`` times.

    Integration points
    ------------------
    * **EventBus** — emits ``peva.*`` events at every state transition so
      quality gates, snapshots, and the meta-loop can react.
    * **Pluggable adapters** — callers can register custom ``adapt_fn``
      callables to control how adaptation works (LLM-driven, rule-based,
      etc.).
    """

    def __init__(
        self,
        bus: EventBus,
        config: Optional[PEVAConfig] = None,
    ) -> None:
        self._bus = bus
        self._config = config or PEVAConfig()
        self._running = False
        self._sub_ids: list[str] = []

        # Pluggable adaptation function.  Signature:
        #   async (strategy, failed_steps, context) -> list[PlanStep]
        # Returns replacement / additional steps, or empty list if nothing
        # can be done.
        self._adapt_fn: Optional[
            Callable[
                [AdaptStrategy, list[PlanStep], dict[str, Any]],
                Coroutine[Any, Any, list[PlanStep]],
            ]
        ] = None

        # History of completed cycles (kept in memory for the session)
        self._history: list[PEVAResult] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the orchestrator and announce readiness on the bus."""
        if not self._config.enabled:
            logger.info("PEVAOrchestrator disabled — skipping start")
            return

        self._running = True
        await self._emit("peva.started", {})
        logger.info("PEVAOrchestrator started")

    async def stop(self) -> None:
        """Tear down subscriptions and stop accepting new cycles."""
        self._running = False
        for sub_id in self._sub_ids:
            self._bus.unsubscribe(sub_id)
        self._sub_ids.clear()

        await self._emit("peva.stopped", {})
        logger.info("PEVAOrchestrator stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_adapt_fn(
        self,
        fn: Callable[
            [AdaptStrategy, list[PlanStep], dict[str, Any]],
            Coroutine[Any, Any, list[PlanStep]],
        ],
    ) -> None:
        """Register a custom adaptation function.

        The function receives ``(strategy, failed_steps, context)`` and
        must return a list of replacement or additional :class:`PlanStep`
        objects.  An empty list signals that adaptation could not produce
        new steps.
        """
        self._adapt_fn = fn

    async def run(
        self,
        steps: list[PlanStep],
        context: Optional[dict[str, Any]] = None,
    ) -> PEVAResult:
        """Execute a full PEVA cycle.

        Parameters
        ----------
        steps:
            The initial plan steps to execute.
        context:
            Arbitrary context dict passed through to step actions and the
            adaptation function.

        Returns
        -------
        PEVAResult
            Immutable result describing the cycle outcome, including any
            adaptations that were attempted.
        """
        if not self._running:
            raise RuntimeError("PEVAOrchestrator is not running — call start() first")

        ctx = context or {}
        result = PEVAResult(steps=list(steps))

        try:
            # Phase 1 — PLAN
            result.complexity = await self._phase_plan(result, ctx)

            adapt_round = 0
            while True:
                # Phase 2 — EXECUTE
                await self._phase_execute(result, ctx)

                # Phase 3 — VERIFY
                all_ok, failed = await self._phase_verify(result, ctx)

                if all_ok:
                    result.success = True
                    break

                # Phase 4 — ADAPT (if budget remains)
                adapt_round += 1
                if adapt_round > self._config.max_adapt_rounds:
                    logger.warning(
                        "Adaptation budget exhausted after %d rounds — cycle %s failing",
                        self._config.max_adapt_rounds,
                        result.cycle_id,
                    )
                    result.error = (
                        f"Max adaptation rounds ({self._config.max_adapt_rounds}) exceeded"
                    )
                    break

                adapted = await self._phase_adapt(
                    result, failed, adapt_round, ctx
                )
                if not adapted:
                    result.error = "Adaptation produced no actionable steps"
                    break

        except asyncio.CancelledError:
            result.error = "Cycle cancelled"
            raise
        except Exception as exc:
            logger.exception("Unexpected error in PEVA cycle %s", result.cycle_id)
            result.error = str(exc)
        finally:
            result.finished_at = time.monotonic()
            self._history.append(result)
            await self._emit(
                "peva.cycle_completed",
                {
                    "cycle_id": result.cycle_id,
                    "success": result.success,
                    "complexity": result.complexity.value,
                    "duration_ms": result.duration_ms,
                    "adapt_rounds": len(result.adaptations),
                },
                priority=(
                    EventPriority.LOW if result.success else EventPriority.HIGH
                ),
            )

        return result

    @property
    def history(self) -> list[PEVAResult]:
        """Return the list of completed PEVA cycles (read-only view)."""
        return list(self._history)

    def get_status(self) -> dict[str, Any]:
        """Return a status snapshot for health-check / dashboard."""
        return {
            "running": self._running,
            "enabled": self._config.enabled,
            "max_adapt_rounds": self._config.max_adapt_rounds,
            "max_parallel": self._config.max_parallel,
            "cycles_completed": len(self._history),
            "cycles_succeeded": sum(1 for r in self._history if r.success),
        }

    # ------------------------------------------------------------------
    # Phase 1: PLAN
    # ------------------------------------------------------------------

    async def _phase_plan(
        self,
        result: PEVAResult,
        ctx: dict[str, Any],
    ) -> Complexity:
        """Classify complexity and validate the dependency graph."""
        await self._emit_phase_transition(result.cycle_id, Phase.PLAN)

        complexity = self._classify_complexity(result.steps)
        logger.info(
            "Cycle %s: classified as %s (%d steps)",
            result.cycle_id,
            complexity.value,
            len(result.steps),
        )

        # Validate dependency graph — detect missing or circular deps
        self._validate_dependencies(result.steps)

        await self._emit(
            "peva.plan_ready",
            {
                "cycle_id": result.cycle_id,
                "complexity": complexity.value,
                "step_count": len(result.steps),
                "step_ids": [s.step_id for s in result.steps],
            },
        )

        return complexity

    def _classify_complexity(self, steps: list[PlanStep]) -> Complexity:
        """Return complexity based on step count and dependency depth."""
        n = len(steps)
        if n <= self._config.trivial_threshold:
            return Complexity.TRIVIAL

        # Check dependency depth — deep chains increase complexity
        depth = self._max_dependency_depth(steps)
        if n <= self._config.moderate_threshold and depth <= 2:
            return Complexity.MODERATE

        return Complexity.COMPLEX

    def _max_dependency_depth(self, steps: list[PlanStep]) -> int:
        """Compute the longest dependency chain length."""
        step_map = {s.step_id: s for s in steps}
        cache: dict[str, int] = {}

        def _depth(sid: str) -> int:
            if sid in cache:
                return cache[sid]
            step = step_map.get(sid)
            if step is None or not step.depends_on:
                cache[sid] = 0
                return 0
            d = 1 + max(_depth(dep) for dep in step.depends_on)
            cache[sid] = d
            return d

        if not steps:
            return 0
        return max(_depth(s.step_id) for s in steps)

    def _validate_dependencies(self, steps: list[PlanStep]) -> None:
        """Raise if the dependency graph is invalid.

        Checks for:
        - References to non-existent step IDs.
        - Circular dependencies (via DFS).
        """
        known_ids = {s.step_id for s in steps}
        adj: dict[str, set[str]] = {s.step_id: set(s.depends_on) for s in steps}

        for sid, deps in adj.items():
            unknown = deps - known_ids
            if unknown:
                raise ValueError(
                    f"Step {sid!r} depends on unknown step(s): {unknown}"
                )

        # Cycle detection via iterative DFS with coloring
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {sid: WHITE for sid in known_ids}

        for start in known_ids:
            if color[start] != WHITE:
                continue
            stack: list[tuple[str, bool]] = [(start, False)]
            while stack:
                node, processed = stack.pop()
                if processed:
                    color[node] = BLACK
                    continue
                if color[node] == GRAY:
                    raise ValueError(
                        f"Circular dependency detected involving step {node!r}"
                    )
                color[node] = GRAY
                stack.append((node, True))
                for dep in adj.get(node, set()):
                    if color[dep] == GRAY:
                        raise ValueError(
                            f"Circular dependency detected: {node!r} -> {dep!r}"
                        )
                    if color[dep] == WHITE:
                        stack.append((dep, False))

    # ------------------------------------------------------------------
    # Phase 2: EXECUTE
    # ------------------------------------------------------------------

    async def _phase_execute(
        self,
        result: PEVAResult,
        ctx: dict[str, Any],
    ) -> None:
        """Execute plan steps respecting dependencies and parallelism."""
        await self._emit_phase_transition(result.cycle_id, Phase.EXECUTE)

        pending = {
            s.step_id: s
            for s in result.steps
            if s.status in (StepStatus.PENDING, StepStatus.FAILED)
        }

        if not pending:
            return

        # Reset failed steps so they can be retried
        for step in pending.values():
            step.status = StepStatus.PENDING
            step.error = None
            step.result = {}

        completed: set[str] = {
            s.step_id
            for s in result.steps
            if s.status == StepStatus.SUCCESS
        }

        sem = asyncio.Semaphore(self._config.max_parallel)

        while pending:
            # Find steps whose dependencies are all satisfied
            ready = [
                s for s in pending.values()
                if s.depends_on.issubset(completed)
            ]

            if not ready:
                # All remaining steps have unsatisfied dependencies.
                # This can happen if a dependency failed — mark as skipped.
                for step in pending.values():
                    step.status = StepStatus.SKIPPED
                    step.error = "Dependency not satisfied"
                    await self._emit(
                        "peva.step_skipped",
                        {
                            "cycle_id": result.cycle_id,
                            "step_id": step.step_id,
                            "reason": "dependency_not_satisfied",
                        },
                    )
                break

            # Launch ready steps with bounded parallelism
            tasks: list[asyncio.Task] = []
            for step in ready:
                step.status = StepStatus.RUNNING
                task = asyncio.create_task(
                    self._run_step(step, ctx, sem, result.cycle_id)
                )
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

            # Update bookkeeping
            for step in ready:
                del pending[step.step_id]
                if step.status == StepStatus.SUCCESS:
                    completed.add(step.step_id)

    async def _run_step(
        self,
        step: PlanStep,
        ctx: dict[str, Any],
        sem: asyncio.Semaphore,
        cycle_id: str,
    ) -> None:
        """Execute a single plan step within the semaphore."""
        async with sem:
            await self._emit(
                "peva.step_started",
                {
                    "cycle_id": cycle_id,
                    "step_id": step.step_id,
                    "description": step.description,
                },
            )

            step.started_at = time.monotonic()
            timeout = step.timeout_seconds or self._config.default_step_timeout

            try:
                if step.action is None:
                    # No action callable — treat as a no-op success
                    step.result = {"success": True, "note": "no-op"}
                    step.status = StepStatus.SUCCESS
                else:
                    step.result = await asyncio.wait_for(
                        step.action(ctx),
                        timeout=timeout,
                    )
                    if step.result.get("success", False):
                        step.status = StepStatus.SUCCESS
                    elif step.result.get("partial", False):
                        step.status = StepStatus.PARTIAL
                    else:
                        step.status = StepStatus.FAILED
                        step.error = step.result.get("error", "Step reported failure")

            except asyncio.TimeoutError:
                step.status = StepStatus.FAILED
                step.error = f"Timed out after {timeout}s"
                logger.warning(
                    "Step %s timed out after %.1fs in cycle %s",
                    step.step_id,
                    timeout,
                    cycle_id,
                )
            except Exception as exc:
                step.status = StepStatus.FAILED
                step.error = str(exc)
                logger.exception(
                    "Step %s raised an exception in cycle %s",
                    step.step_id,
                    cycle_id,
                )
            finally:
                step.finished_at = time.monotonic()
                await self._emit(
                    "peva.step_completed",
                    {
                        "cycle_id": cycle_id,
                        "step_id": step.step_id,
                        "status": step.status.value,
                        "duration_ms": step.duration_ms,
                        "error": step.error,
                    },
                    priority=(
                        EventPriority.LOW
                        if step.status == StepStatus.SUCCESS
                        else EventPriority.HIGH
                    ),
                )

    # ------------------------------------------------------------------
    # Phase 3: VERIFY
    # ------------------------------------------------------------------

    async def _phase_verify(
        self,
        result: PEVAResult,
        ctx: dict[str, Any],
    ) -> tuple[bool, list[PlanStep]]:
        """Check step outcomes and return ``(all_ok, failed_steps)``."""
        await self._emit_phase_transition(result.cycle_id, Phase.VERIFY)

        failed: list[PlanStep] = []
        for step in result.steps:
            if step.status in (StepStatus.FAILED, StepStatus.PARTIAL):
                failed.append(step)

        all_ok = len(failed) == 0

        await self._emit(
            "peva.verify_completed",
            {
                "cycle_id": result.cycle_id,
                "all_ok": all_ok,
                "total_steps": len(result.steps),
                "succeeded": sum(
                    1 for s in result.steps if s.status == StepStatus.SUCCESS
                ),
                "failed": len(failed),
                "failed_ids": [s.step_id for s in failed],
            },
            priority=EventPriority.LOW if all_ok else EventPriority.HIGH,
        )

        return all_ok, failed

    # ------------------------------------------------------------------
    # Phase 4: ADAPT
    # ------------------------------------------------------------------

    async def _phase_adapt(
        self,
        result: PEVAResult,
        failed_steps: list[PlanStep],
        round_number: int,
        ctx: dict[str, Any],
    ) -> bool:
        """Attempt adaptation. Returns True if new/fixed steps were produced."""
        await self._emit_phase_transition(result.cycle_id, Phase.ADAPT)

        strategy = self._choose_strategy(failed_steps, round_number)
        record = AdaptationRecord(
            round_number=round_number,
            strategy=strategy,
            failed_steps=[s.step_id for s in failed_steps],
        )

        logger.info(
            "Cycle %s adapt round %d: strategy=%s, failed=%s",
            result.cycle_id,
            round_number,
            strategy.value,
            [s.step_id for s in failed_steps],
        )

        await self._emit(
            "peva.adapt_started",
            {
                "cycle_id": result.cycle_id,
                "round": round_number,
                "strategy": strategy.value,
                "failed_step_ids": [s.step_id for s in failed_steps],
            },
        )

        adapted = False

        if strategy == AdaptStrategy.TARGETED_FIX:
            adapted = await self._adapt_targeted_fix(
                result, failed_steps, ctx
            )

        elif strategy == AdaptStrategy.REVISE_APPROACH:
            adapted = await self._adapt_revise_approach(
                result, failed_steps, ctx
            )

        elif strategy == AdaptStrategy.EXPAND_PLAN:
            adapted = await self._adapt_expand_plan(
                result, failed_steps, ctx
            )

        record.outcome = "success" if adapted else "failed"
        result.adaptations.append(record)

        await self._emit(
            "peva.adapt_completed",
            {
                "cycle_id": result.cycle_id,
                "round": round_number,
                "strategy": strategy.value,
                "adapted": adapted,
            },
            priority=EventPriority.LOW if adapted else EventPriority.HIGH,
        )

        return adapted

    def _choose_strategy(
        self,
        failed_steps: list[PlanStep],
        round_number: int,
    ) -> AdaptStrategy:
        """Select an adaptation strategy based on failure pattern and round.

        Heuristic:
        - Round 1 with few failures  -> targeted_fix
        - Round 2 or many failures   -> revise_approach
        - Round 3+                   -> expand_plan
        """
        if round_number >= 3:
            return AdaptStrategy.EXPAND_PLAN
        if round_number >= 2 or len(failed_steps) > 2:
            return AdaptStrategy.REVISE_APPROACH
        return AdaptStrategy.TARGETED_FIX

    async def _adapt_targeted_fix(
        self,
        result: PEVAResult,
        failed_steps: list[PlanStep],
        ctx: dict[str, Any],
    ) -> bool:
        """Reset failed steps so they can be re-executed.

        If a custom ``adapt_fn`` is registered, it may return replacement
        steps.
        """
        if self._adapt_fn is not None:
            try:
                replacements = await self._adapt_fn(
                    AdaptStrategy.TARGETED_FIX, failed_steps, ctx
                )
                if replacements:
                    self._replace_steps(result, failed_steps, replacements)
                    return True
            except Exception:
                logger.exception("Custom adapt_fn raised during targeted_fix")

        # Default: simply mark failed steps as pending for retry
        for step in failed_steps:
            step.status = StepStatus.PENDING
            step.error = None
            step.result = {}

        return True

    async def _adapt_revise_approach(
        self,
        result: PEVAResult,
        failed_steps: list[PlanStep],
        ctx: dict[str, Any],
    ) -> bool:
        """Attempt a broader revision via the custom adapt function.

        Falls back to resetting failed steps if no custom function is
        available.
        """
        if self._adapt_fn is not None:
            try:
                replacements = await self._adapt_fn(
                    AdaptStrategy.REVISE_APPROACH, failed_steps, ctx
                )
                if replacements:
                    self._replace_steps(result, failed_steps, replacements)
                    return True
            except Exception:
                logger.exception("Custom adapt_fn raised during revise_approach")

        # Fallback: reset failed steps
        for step in failed_steps:
            step.status = StepStatus.PENDING
            step.error = None
            step.result = {}

        return True

    async def _adapt_expand_plan(
        self,
        result: PEVAResult,
        failed_steps: list[PlanStep],
        ctx: dict[str, Any],
    ) -> bool:
        """Add new steps to the plan to work around failures.

        Requires a custom adapt function to produce additional steps.
        Without one, the orchestrator cannot synthesise new work and
        adaptation fails.
        """
        if self._adapt_fn is not None:
            try:
                additional = await self._adapt_fn(
                    AdaptStrategy.EXPAND_PLAN, failed_steps, ctx
                )
                if additional:
                    # Mark old failed steps as skipped and append new ones
                    for step in failed_steps:
                        step.status = StepStatus.SKIPPED
                        step.error = "Superseded by expanded plan"
                    result.steps.extend(additional)
                    return True
            except Exception:
                logger.exception("Custom adapt_fn raised during expand_plan")

        # Cannot expand without a custom function
        logger.warning("No adapt_fn registered — cannot expand plan")
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _replace_steps(
        self,
        result: PEVAResult,
        old_steps: list[PlanStep],
        new_steps: list[PlanStep],
    ) -> None:
        """Replace *old_steps* in the result with *new_steps*."""
        old_ids = {s.step_id for s in old_steps}
        result.steps = [
            s for s in result.steps if s.step_id not in old_ids
        ] + list(new_steps)

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    async def _emit(
        self,
        event_type: str,
        payload: dict[str, Any],
        priority: EventPriority = EventPriority.MEDIUM,
    ) -> None:
        """Publish an event on the bus."""
        event = Event(
            event_type=event_type,
            source="peva_orchestrator",
            payload=payload,
            priority=priority,
        )
        await self._bus.publish(event)

    async def _emit_phase_transition(
        self,
        cycle_id: str,
        phase: Phase,
    ) -> None:
        """Emit a ``peva.phase_changed`` event."""
        await self._emit(
            "peva.phase_changed",
            {"cycle_id": cycle_id, "phase": phase.value},
        )
