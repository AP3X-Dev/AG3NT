"""AutofixEngine — main orchestrator for the self-healing error pipeline."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from ag3nt_agent.autofix.circuit_breaker import CircuitBreaker
from ag3nt_agent.autofix.context import AttemptRecord, FixContext
from ag3nt_agent.autofix.error_queue import ErrorQueue
from ag3nt_agent.autofix.learning import FixHistory
from ag3nt_agent.autofix.pipeline import Pipeline
from ag3nt_agent.autofix.stages.detect import ReactiveDetector
from ag3nt_agent.autofix.stages.classify import ErrorClassifier
from ag3nt_agent.autofix.stages.enrich import ContextEnricher
from ag3nt_agent.autofix.stages.score import ConfidenceScorer
from ag3nt_agent.autofix.stages.fix import AgentFixer
from ag3nt_agent.autofix.stages.verify import FixVerifier
from ag3nt_agent.autofix.stages.learn import LearningRecorder
from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.autofix.engine")


@dataclass
class AutofixConfig:
    """Configuration for the AutofixEngine."""

    enabled: bool = True
    debounce_s: float = 2.0
    max_batch: int = 10
    max_attempts: int = 4
    health_check_interval_s: float = 30.0
    failure_threshold: int = 3
    reset_timeout_s: float = 300.0
    working_dir: str = "."


class AutofixEngine:
    """Main orchestrator for the self-healing pipeline.

    Assembles the 7-stage pipeline (Detect -> Classify -> Enrich -> Score ->
    Fix -> Verify -> Learn), continuously processes error batches from
    ReactiveDetector, runs a health loop for proactive checks, and integrates
    a circuit breaker with up to ``max_attempts`` retries per batch.
    """

    def __init__(self, bus: EventBus, config: AutofixConfig | None = None) -> None:
        self._bus = bus
        self._config = config or AutofixConfig()

        # Components
        self._history = FixHistory()
        self._error_queue = ErrorQueue()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=self._config.failure_threshold,
            reset_timeout_s=self._config.reset_timeout_s,
        )
        self._detector = ReactiveDetector(bus, debounce_s=self._config.debounce_s)

        # Build pipeline (stages 2-7; detection is handled externally)
        self._pipeline = Pipeline()
        self._pipeline.add_stage(ErrorClassifier())
        self._pipeline.add_stage(ContextEnricher(working_dir=self._config.working_dir))
        self._pipeline.add_stage(ConfidenceScorer(history=self._history))
        self._pipeline.add_stage(AgentFixer(working_dir=self._config.working_dir))
        self._pipeline.add_stage(FixVerifier(working_dir=self._config.working_dir))
        self._pipeline.add_stage(LearningRecorder(history=self._history))

        self._main_task: asyncio.Task | None = None
        self._health_task: asyncio.Task | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the main processing and health loops."""
        if not self._config.enabled:
            logger.info("AutofixEngine disabled")
            return
        self._running = True
        self._main_task = asyncio.create_task(self._main_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("AutofixEngine started")

    async def stop(self) -> None:
        """Stop all background tasks."""
        self._running = False
        if self._main_task:
            self._main_task.cancel()
        if self._health_task:
            self._health_task.cancel()
        logger.info("AutofixEngine stopped")

    # ------------------------------------------------------------------
    # Main processing loop
    # ------------------------------------------------------------------

    async def _main_loop(self) -> None:
        """Continuously drain error batches from the detector and run them
        through the pipeline with retry logic."""
        while self._running:
            try:
                await asyncio.sleep(self._config.debounce_s)
                ctx = self._detector.drain_to_context(self._config.max_batch)
                if ctx is None:
                    continue

                if not self._circuit_breaker.allow("autofix"):
                    logger.warning("Circuit breaker OPEN, skipping autofix batch")
                    continue

                # Emit start event
                self._bus.emit_sync(Event(
                    event_type="autofix.started",
                    source="autofix:engine",
                    payload={"error_count": len(ctx.errors)},
                    priority=EventPriority.MEDIUM,
                ))

                # Run pipeline with retry
                for attempt in range(self._config.max_attempts):
                    ctx.attempt_number = attempt + 1
                    result = await self._pipeline.run(ctx)

                    if result.outcome in ("fixed", "success"):
                        self._circuit_breaker.record_success("autofix")
                        self._bus.emit_sync(Event(
                            event_type="autofix.completed",
                            source="autofix:engine",
                            payload={
                                "outcome": result.outcome,
                                "attempts": attempt + 1,
                            },
                            priority=EventPriority.MEDIUM,
                        ))
                        break
                    elif result.outcome == "error":
                        self._circuit_breaker.record_failure("autofix")
                        break  # Pipeline stage error, don't retry
                    else:
                        # Verification failed or skipped — retry with
                        # conversational feedback from the prior attempt.
                        ctx.prior_attempts.append(AttemptRecord(
                            attempt_number=attempt + 1,
                            diff_summary=(
                                result.fix_response[:500]
                                if result.fix_response
                                else ""
                            ),
                            verification_stage_failed=str(
                                result.validation.get("failed_stage", "")
                            ),
                            failure_detail=str(
                                result.validation.get("error", "")
                            ),
                        ))
                        if attempt + 1 >= self._config.max_attempts:
                            self._circuit_breaker.record_failure("autofix")
                            self._bus.emit_sync(Event(
                                event_type="autofix.failed",
                                source="autofix:engine",
                                payload={
                                    "outcome": result.outcome,
                                    "attempts": attempt + 1,
                                },
                                priority=EventPriority.HIGH,
                            ))

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("AutofixEngine main loop error: %s", exc)

    # ------------------------------------------------------------------
    # Health loop
    # ------------------------------------------------------------------

    async def _health_loop(self) -> None:
        """Periodic health checks (runs every ``health_check_interval_s``)."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval_s)
                # Check circuit breaker state
                state = self._circuit_breaker.get_state("autofix")
                if state["state"] == "open":
                    logger.warning("Autofix circuit breaker is OPEN")
                # Check error queue size
                queue_size = self._error_queue.size()
                if queue_size > 50:
                    logger.warning("Error queue growing: %d entries", queue_size)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Health loop error: %s", exc)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of the engine's current state."""
        return {
            "enabled": self._config.enabled,
            "running": self._running,
            "circuit_breaker": self._circuit_breaker.get_state("autofix"),
            "error_queue_size": self._error_queue.size(),
        }
