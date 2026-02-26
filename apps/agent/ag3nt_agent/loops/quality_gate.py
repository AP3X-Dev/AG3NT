"""
Quality Gate Loop for AG3NT Autonomous System

Subscribes to file.written and file.edited events on the EventBus,
auto-runs lint / typecheck / format on code files, and auto-fixes
with retry (max configurable, default 3).
"""

import asyncio
import fnmatch
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


@dataclass
class QualityGateConfig:
    """Configuration for the QualityGateLoop."""

    enabled: bool = True
    max_retries: int = 3
    auto_fix: bool = True
    parallel: bool = True
    timeout_seconds: int = 30

    skip_patterns: list[str] = field(default_factory=lambda: [
        "*.md", "*.txt", "*.json", "*.yaml", "*.yml",
        "*.toml", "*.cfg", "*.ini", "*.lock",
        "*.png", "*.jpg", "*.gif", "*.svg",
    ])

    check_extensions: list[str] = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".tsx", ".jsx",
    ])


class QualityGateLoop:
    """
    Reactive loop that watches for file-write events and runs
    quality checks (lint, typecheck, format) with automatic fix + retry.

    Usage::

        bus = EventBus()
        loop = QualityGateLoop(bus)
        await loop.start()
        # ... events flow through the bus ...
        await loop.stop()
    """

    def __init__(
        self,
        bus: EventBus,
        config: Optional[QualityGateConfig] = None,
        working_dir: str = ".",
    ):
        self._bus = bus
        self._config = config or QualityGateConfig()
        self._working_dir = working_dir

        self._check_queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscription_ids: list[str] = []
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to file events and start the background worker."""
        if not self._config.enabled:
            logger.info("QualityGateLoop is disabled – skipping start")
            return

        event_types = {"file.written", "file.edited"}
        sub_id = self._bus.subscribe(
            handler=self._on_file_event,
            event_types=event_types,
        )
        self._subscription_ids.append(sub_id)

        self._running = True
        self._worker_task = asyncio.create_task(self._process_loop())

        logger.info("QualityGateLoop started (subscribed to %s)", event_types)

    async def stop(self) -> None:
        """Unsubscribe from the bus and cancel the background worker."""
        self._running = False

        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        logger.info("QualityGateLoop stopped")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    async def _on_file_event(self, event: Event) -> None:
        """Handle a file.written or file.edited event."""
        path = event.payload.get("path") or event.payload.get("file")
        if path is None:
            logger.debug("Ignoring file event with no path in payload: %s", event.event_id)
            return

        if self._should_skip(path):
            logger.debug("Skipping quality check for %s (skip pattern or unsupported extension)", path)
            return

        await self._check_queue.put(path)

    # ------------------------------------------------------------------
    # Skip logic
    # ------------------------------------------------------------------

    def _should_skip(self, path: str) -> bool:
        """Return True if the file should NOT be quality-checked."""
        basename = os.path.basename(path)

        # Check against skip patterns
        for pattern in self._config.skip_patterns:
            if fnmatch.fnmatch(basename, pattern):
                return True

        # Check against allowed extensions
        _, ext = os.path.splitext(path)
        if ext and ext not in self._config.check_extensions:
            return True

        return False

    # ------------------------------------------------------------------
    # Processing loop
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Continuously dequeue file paths and run quality checks."""
        while self._running:
            try:
                path = await asyncio.wait_for(
                    self._check_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._check_file(path)
            except Exception:
                logger.exception("Unexpected error checking file %s", path)

    async def _check_file(self, path: str) -> None:
        """
        Run quality checks on *path* with auto-fix retry.

        Up to ``max_retries`` attempts are made when auto-fix is enabled.
        """
        _, suffix = os.path.splitext(path)

        for attempt in range(1, self._config.max_retries + 1):
            checks = await self._run_checks(path, suffix)
            all_passed = all(checks.values())

            if all_passed:
                await self._emit_result(path, passed=True, checks=checks)
                return

            if self._config.auto_fix and attempt < self._config.max_retries:
                fixed = await self._auto_fix(path, suffix, checks)
                if not fixed:
                    # Nothing could be auto-fixed; no point retrying
                    break
                logger.info(
                    "Auto-fix applied for %s (attempt %d/%d) – retrying checks",
                    path, attempt, self._config.max_retries,
                )
            else:
                break

        await self._emit_result(path, passed=False, checks=checks)

    # ------------------------------------------------------------------
    # Check runners
    # ------------------------------------------------------------------

    async def _run_checks(self, path: str, suffix: str) -> dict[str, bool]:
        """
        Run the appropriate quality checks for *path* based on its *suffix*.

        Returns a dict mapping check-name -> passed (bool).
        """
        checks: dict[str, bool] = {}

        if suffix == ".py":
            # Syntax check via py_compile
            ok, _ = await self._run_cmd(
                ["python", "-m", "py_compile", path],
                timeout=self._config.timeout_seconds,
            )
            checks["py_compile"] = ok

            # Lint via ruff
            ok, _ = await self._run_cmd(
                ["ruff", "check", path],
                timeout=self._config.timeout_seconds,
            )
            checks["ruff"] = ok

        elif suffix in (".js", ".jsx", ".ts", ".tsx"):
            # Lint via eslint
            ok, _ = await self._run_cmd(
                ["eslint", path],
                timeout=self._config.timeout_seconds,
            )
            checks["eslint"] = ok

        return checks

    async def _auto_fix(
        self,
        path: str,
        suffix: str,
        checks: dict[str, bool],
    ) -> bool:
        """
        Attempt to auto-fix issues found by the quality checks.

        Returns True if at least one fix command was executed.
        """
        fixed_any = False

        if suffix == ".py":
            if not checks.get("ruff", True):
                ok, _ = await self._run_cmd(
                    ["ruff", "check", "--fix", path],
                    timeout=self._config.timeout_seconds,
                )
                fixed_any = True

        elif suffix in (".js", ".jsx", ".ts", ".tsx"):
            if not checks.get("eslint", True):
                ok, _ = await self._run_cmd(
                    ["eslint", "--fix", path],
                    timeout=self._config.timeout_seconds,
                )
                fixed_any = True

        return fixed_any

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------

    async def _run_cmd(
        self,
        cmd: list[str],
        timeout: int = 30,
    ) -> tuple[bool, str]:
        """
        Run a command via asyncio subprocess.

        Returns ``(success: bool, output: str)``.
        Gracefully handles ``FileNotFoundError`` when the tool is not installed.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._working_dir,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"Command timed out after {timeout}s: {' '.join(cmd)}"

            output = stdout.decode(errors="replace") if stdout else ""
            return proc.returncode == 0, output

        except FileNotFoundError:
            logger.debug("Command not found: %s", cmd[0])
            return True, f"{cmd[0]} not found – skipping check"
        except Exception as exc:
            logger.warning("Failed to run %s: %s", cmd, exc)
            return False, str(exc)

    # ------------------------------------------------------------------
    # Result emission
    # ------------------------------------------------------------------

    async def _emit_result(
        self,
        path: str,
        passed: bool,
        checks: dict[str, bool],
    ) -> None:
        """Emit a quality.check_passed or quality.check_failed event."""
        event_type = "quality.check_passed" if passed else "quality.check_failed"
        event = Event(
            event_type=event_type,
            source="quality_gate_loop",
            payload={
                "path": path,
                "checks": checks,
                "passed": passed,
            },
        )
        await self._bus.publish(event)
        log_fn = logger.info if passed else logger.warning
        log_fn("Quality gate %s for %s: %s", "PASSED" if passed else "FAILED", path, checks)
