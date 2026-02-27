"""
AG3NT Heartbeat -- Periodic background task engine.

Runs registered callbacks on a configurable interval.
Tracks tick count, errors, and last-tick timestamp.
Zero external dependencies -- pure asyncio.

Ported from ONI-BOT heartbeat module.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

log = logging.getLogger("ag3nt.heartbeat")


# ===================================================================
#  Types
# ===================================================================

HeartbeatCallback = Callable[[], Awaitable[None]]


# ===================================================================
#  Data Models
# ===================================================================


@dataclass
class HeartbeatTask:
    """A single task from the HEARTBEAT.md checklist."""

    description: str
    id: str = ""
    command: Optional[str] = None
    check_type: str = "command"  # "command", "file_exists", "file_changed", "custom"
    interval_seconds: int = 300
    last_run: Optional[float] = None  # monotonic time
    last_result: Optional[str] = None
    enabled: bool = True
    failure_count: int = 0
    max_failures: int = 5

    def __post_init__(self) -> None:
        if not self.id:
            raw = f"{self.description}:{self.command or ''}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class HeartbeatSchedule:
    """Manages timing for heartbeat tasks."""

    tasks: list[HeartbeatTask] = field(default_factory=list)
    global_interval_seconds: int = 300
    last_cycle: Optional[float] = None
    jitter_seconds: int = 30


@dataclass
class HeartbeatResult:
    """Result of a single heartbeat task execution."""

    task_id: str
    success: bool
    output: str
    duration_ms: float
    executed_at: str  # ISO timestamp
    action_taken: Optional[str] = None


@dataclass
class HeartbeatCycleResult:
    """Result of a complete heartbeat cycle."""

    cycle_number: int
    results: list[HeartbeatResult]
    started_at: str
    completed_at: str
    duration_ms: float
    tasks_run: int
    tasks_passed: int
    tasks_failed: int


# ===================================================================
#  Heartbeat Engine
# ===================================================================


class HeartbeatEngine:
    """
    Periodic background task engine.

    Registers async callbacks and invokes them every ``interval_seconds``.
    Errors in individual callbacks are logged but never crash the loop.

    Default interval: 1800 s (30 minutes).
    """

    def __init__(
        self,
        interval_seconds: float = 1800,
        name: str = "heartbeat",
    ):
        self._interval = interval_seconds
        self._name = name
        self._callbacks: list[tuple[HeartbeatCallback, str]] = []
        self._task: Optional[asyncio.Task] = None
        self._tick_count: int = 0
        self._error_count: int = 0
        self._last_tick: Optional[datetime] = None

    # ----- Public API -----

    def register(self, callback: HeartbeatCallback, name: str = "") -> None:
        """Register a callback to run each tick."""
        label = name or getattr(callback, "__name__", repr(callback))
        self._callbacks.append((callback, label))
        log.debug("[%s] registered callback: %s", self._name, label)

    async def start(self) -> None:
        """Start the heartbeat loop as a background asyncio task."""
        if self._task is not None and not self._task.done():
            log.warning("[%s] already running -- ignoring start()", self._name)
            return

        self._task = asyncio.create_task(self._loop(), name=self._name)
        log.info(
            "[%s] started  interval=%ss  callbacks=%d",
            self._name,
            self._interval,
            len(self._callbacks),
        )

    async def stop(self) -> None:
        """Cancel the background task gracefully."""
        if self._task is None or self._task.done():
            log.debug("[%s] not running -- nothing to stop", self._name)
            return

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

        self._task = None
        log.info(
            "[%s] stopped  ticks=%d  errors=%d",
            self._name,
            self._tick_count,
            self._error_count,
        )

    # ----- Properties -----

    @property
    def is_running(self) -> bool:
        """True if the background loop task is alive."""
        return self._task is not None and not self._task.done()

    @property
    def last_tick(self) -> Optional[datetime]:
        """Timestamp of the most recent completed tick, or None."""
        return self._last_tick

    @property
    def tick_count(self) -> int:
        """Total number of ticks completed since creation."""
        return self._tick_count

    @property
    def error_count(self) -> int:
        """Total number of callback errors since creation."""
        return self._error_count

    # ----- Internal loop -----

    async def _loop(self) -> None:
        """Sleep interval, run all callbacks, log errors, continue."""
        log.debug("[%s] loop entered", self._name)
        try:
            while True:
                await asyncio.sleep(self._interval)

                self._tick_count += 1
                self._last_tick = datetime.now(timezone.utc)
                log.debug("[%s] tick #%d", self._name, self._tick_count)

                for callback, label in self._callbacks:
                    try:
                        await callback()
                    except Exception as exc:
                        self._error_count += 1
                        log.error(
                            "[%s] callback '%s' failed on tick #%d: %s",
                            self._name,
                            label,
                            self._tick_count,
                            exc,
                            exc_info=True,
                        )
        except asyncio.CancelledError:
            log.debug("[%s] loop cancelled", self._name)
            raise


# ===================================================================
#  Heartbeat File Parser
# ===================================================================


class HeartbeatFileParser:
    """Parses HEARTBEAT.md files into HeartbeatTask lists.

    Expected format::

        # Heartbeat Checklist

        ## Every 5 minutes
        - [ ] Check build status: `make check`
        - [ ] Verify server running: `curl -s http://localhost:8080/health`

        ## Every 15 minutes
        - [ ] Run quick tests: `python -m pytest tests/quick -q`
        - [x] Check disk space: `df -h /`  (checked = disabled)

        ## Every hour
        - [ ] Full test suite: `python -m pytest`
    """

    INTERVAL_PATTERNS: dict[str, int] = {
        "every 1 minute": 60,
        "every minute": 60,
        "every 5 minutes": 300,
        "every 10 minutes": 600,
        "every 15 minutes": 900,
        "every 30 minutes": 1800,
        "every hour": 3600,
        "every 1 hour": 3600,
    }

    _RE_N_MINUTES = re.compile(r"every\s+(\d+)\s+minutes?", re.IGNORECASE)
    _RE_N_HOURS = re.compile(r"every\s+(\d+)\s+hours?", re.IGNORECASE)
    _RE_TASK_LINE = re.compile(r"^\s*-\s+\[([ xX])\]\s+(.+)$")
    _RE_BACKTICK_CMD = re.compile(r"`([^`]+)`")

    def parse(self, content: str) -> list[HeartbeatTask]:
        """Parse HEARTBEAT.md content into tasks.

        Section headers (``## ...``) set the interval for subsequent tasks.
        Checklist items (``- [ ]`` enabled, ``- [x]`` disabled) define tasks.
        Commands are extracted from backtick-quoted portions of each line.
        """
        tasks: list[HeartbeatTask] = []
        current_interval = 300  # default 5 minutes

        for line in content.splitlines():
            stripped = line.strip()

            # Section header -- update interval
            if stripped.startswith("## "):
                header_text = stripped[3:].strip()
                current_interval = self._parse_interval(header_text)
                continue

            # Task line
            match = self._RE_TASK_LINE.match(stripped)
            if match:
                body = match.group(2)
                disabled = self._is_disabled(stripped)
                command = self._extract_command(body)
                description = self._extract_description(body)

                if command:
                    check_type = "command"
                else:
                    # Heuristic: determine check_type from description
                    lower_desc = description.lower()
                    if "file exist" in lower_desc or "exists" in lower_desc:
                        check_type = "file_exists"
                    elif "file change" in lower_desc or "changed" in lower_desc:
                        check_type = "file_changed"
                    else:
                        check_type = "custom"

                task = HeartbeatTask(
                    description=description,
                    command=command,
                    check_type=check_type,
                    interval_seconds=current_interval,
                    enabled=not disabled,
                )
                tasks.append(task)

        return tasks

    def parse_file(self, path: str) -> list[HeartbeatTask]:
        """Read and parse a HEARTBEAT.md file."""
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        return self.parse(content)

    @staticmethod
    def _parse_interval(header: str) -> int:
        """Parse interval from section header.  Default 300s."""
        lower = header.lower().strip()

        # Check static patterns first
        for pattern, seconds in HeartbeatFileParser.INTERVAL_PATTERNS.items():
            if lower == pattern:
                return seconds

        # Dynamic N minutes
        m = HeartbeatFileParser._RE_N_MINUTES.search(lower)
        if m:
            return int(m.group(1)) * 60

        # Dynamic N hours
        m = HeartbeatFileParser._RE_N_HOURS.search(lower)
        if m:
            return int(m.group(1)) * 3600

        return 300  # default

    @staticmethod
    def _extract_command(line: str) -> Optional[str]:
        """Extract command from backtick-quoted portion of line."""
        m = HeartbeatFileParser._RE_BACKTICK_CMD.search(line)
        return m.group(1) if m else None

    @staticmethod
    def _extract_description(line: str) -> str:
        """Extract description text (before backtick command).

        If there is a backtick command, the description is the text
        preceding it, stripped of trailing colons and whitespace.
        """
        idx = line.find("`")
        if idx >= 0:
            desc = line[:idx].rstrip().rstrip(":")
        else:
            desc = line
        return desc.strip()

    @staticmethod
    def _is_disabled(line: str) -> bool:
        """Check if line uses ``[x]`` (disabled) vs ``[ ]`` (enabled)."""
        m = HeartbeatFileParser._RE_TASK_LINE.match(line.strip())
        if m:
            return m.group(1).lower() == "x"
        return False


# ===================================================================
#  Heartbeat Executor
# ===================================================================


class HeartbeatExecutor:
    """Executes individual heartbeat tasks."""

    DEFAULT_TIMEOUT_MS = 30_000  # 30 second timeout per task

    def __init__(self, workspace_dir: str = "."):
        self._workspace = workspace_dir
        self._logger = logging.getLogger("ag3nt.heartbeat.executor")

    async def execute_task(self, task: HeartbeatTask) -> HeartbeatResult:
        """Execute a single heartbeat task and return result.

        Dispatches to the appropriate handler based on ``task.check_type``.
        All exceptions are caught so a result is always returned.
        """
        started = time.monotonic()
        executed_at = datetime.now(timezone.utc).isoformat()

        try:
            if task.check_type == "command" and task.command:
                success, output = await self._run_command(
                    task.command, self.DEFAULT_TIMEOUT_MS
                )
            elif task.check_type == "file_exists" and task.command:
                success, output = await self._check_file_exists(task.command)
            elif task.check_type == "file_changed" and task.command:
                since = task.last_run if task.last_run is not None else 0.0
                success, output = await self._check_file_changed(
                    task.command, since
                )
            elif task.check_type == "custom":
                success = True
                output = f"Custom task noted for agent attention: {task.description}"
            else:
                success = False
                output = (
                    f"Unsupported check_type '{task.check_type}' "
                    f"or missing command"
                )
        except Exception as exc:
            success = False
            output = f"Unhandled exception: {exc}"
            self._logger.exception("Task %s failed unexpectedly", task.id)

        elapsed_ms = (time.monotonic() - started) * 1000.0
        return HeartbeatResult(
            task_id=task.id,
            success=success,
            output=output,
            duration_ms=round(elapsed_ms, 2),
            executed_at=executed_at,
            action_taken=None,
        )

    async def _run_command(
        self, command: str, timeout_ms: int
    ) -> tuple[bool, str]:
        """Run a shell command with timeout.  Returns ``(success, output)``."""
        timeout_s = timeout_ms / 1000.0
        self._logger.debug("Running command: %s (timeout=%.1fs)", command, timeout_s)

        _extra: dict = {}
        if sys.platform == "win32":
            import subprocess as _sp

            _extra["creationflags"] = _sp.CREATE_NO_WINDOW

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._workspace,
                **_extra,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, f"Command timed out after {timeout_ms}ms"

            output = stdout.decode("utf-8", errors="replace").strip() if stdout else ""
            return (proc.returncode == 0, output)

        except OSError as exc:
            return False, f"Failed to start command: {exc}"

    async def _check_file_exists(self, path: str) -> tuple[bool, str]:
        """Check if file exists.  Returns ``(exists, message)``."""
        resolved = Path(self._workspace) / path
        exists = resolved.exists()
        if exists:
            return True, f"File exists: {resolved}"
        return False, f"File not found: {resolved}"

    async def _check_file_changed(
        self, path: str, since: float
    ) -> tuple[bool, str]:
        """Check if file was modified since given monotonic time.

        Compares file mtime against the wall-clock equivalent of *since*.
        If *since* is 0 the file is treated as changed.
        """
        resolved = Path(self._workspace) / path
        if not resolved.exists():
            return False, f"File not found: {resolved}"

        mtime = resolved.stat().st_mtime
        if since == 0.0:
            return True, f"File exists (first check): {resolved}"

        # Convert monotonic 'since' to an approximate epoch time
        now_mono = time.monotonic()
        now_epoch = time.time()
        since_epoch = now_epoch - (now_mono - since)

        if mtime > since_epoch:
            return True, (
                f"File changed (mtime={mtime:.1f} > since={since_epoch:.1f}): "
                f"{resolved}"
            )
        return False, f"File unchanged: {resolved}"


# ===================================================================
#  Heartbeat Daemon
# ===================================================================


class HeartbeatDaemon:
    """Main heartbeat daemon -- runs background cycles on schedule.

    Lifecycle:
        1. Load ``HEARTBEAT.md`` from workspace.
        2. Parse into tasks.
        3. Start background loop.
        4. Each cycle: check which tasks are due, execute them, record results.
        5. Reload ``HEARTBEAT.md`` periodically (every 10 cycles) to pick up changes.
        6. Emit events for monitoring.

    Can be started/stopped.  Respects session state (pauses during active
    user turns).  Config-gated: disabled by default.
    """

    HEARTBEAT_FILENAME = "HEARTBEAT.md"
    RELOAD_EVERY_N_CYCLES = 10
    MAX_CONCURRENT_TASKS = 3

    def __init__(
        self,
        workspace_dir: str,
        bus: object = None,
        session_key: Optional[str] = None,
    ):
        self._workspace = workspace_dir
        self._bus = bus
        self._session_key = session_key
        self._parser = HeartbeatFileParser()
        self._executor = HeartbeatExecutor(workspace_dir)
        self._schedule: Optional[HeartbeatSchedule] = None
        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._history: list[HeartbeatCycleResult] = []
        self._max_history = 50
        self._logger = logging.getLogger("ag3nt.heartbeat.daemon")

    # ----- Lifecycle -----

    async def start(self) -> None:
        """Start the heartbeat daemon.

        Loads and parses ``HEARTBEAT.md``.  If the file does not exist a
        default template is written first.  Then kicks off the background
        loop as an ``asyncio.Task``.
        """
        if self._running:
            self._logger.warning("Daemon already running -- ignoring start()")
            return

        heartbeat_path = Path(self._workspace) / self.HEARTBEAT_FILENAME
        if not heartbeat_path.exists():
            self._logger.info(
                "No %s found -- creating default template", self.HEARTBEAT_FILENAME
            )
            heartbeat_path.write_text(
                self._create_default_heartbeat(), encoding="utf-8"
            )

        tasks = self._parser.parse_file(str(heartbeat_path))
        self._schedule = HeartbeatSchedule(tasks=tasks)
        self._running = True
        self._task = asyncio.create_task(
            self._daemon_loop(), name="heartbeat-daemon"
        )
        self._logger.info(
            "Heartbeat daemon started -- %d tasks loaded", len(tasks)
        )

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        if not self._running:
            return
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._logger.info(
            "Heartbeat daemon stopped after %d cycles", self._cycle_count
        )

    def pause(self) -> None:
        """Pause execution (during active user turns)."""
        self._paused = True
        self._logger.debug("Daemon paused")

    def resume(self) -> None:
        """Resume execution after pause."""
        self._paused = False
        self._logger.debug("Daemon resumed")

    # ----- Main loop -----

    async def _daemon_loop(self) -> None:
        """Main daemon loop.

        Sleeps in 10-second increments, checking whether enough time has
        elapsed for the next cycle.  Respects pause state and periodically
        reloads the checklist file.
        """
        self._logger.debug("Daemon loop entered")
        try:
            while self._running:
                # Respect pause
                if self._paused:
                    await asyncio.sleep(1)
                    continue

                # Check global interval (with jitter)
                if (
                    self._schedule is not None
                    and self._schedule.last_cycle is not None
                ):
                    jitter = random.uniform(0, self._schedule.jitter_seconds)
                    elapsed = time.monotonic() - self._schedule.last_cycle
                    if elapsed < self._schedule.global_interval_seconds + jitter:
                        await asyncio.sleep(10)
                        continue

                # Find and run due tasks
                due = self._find_due_tasks()
                if due:
                    cycle_result = await self._run_cycle(due)
                    self._cycle_count += 1

                    # Emit event if bus is available
                    if self._bus is not None and hasattr(self._bus, "emit"):
                        try:
                            self._bus.emit(
                                "heartbeat.cycle",
                                {
                                    "cycle": self._cycle_count,
                                    "tasks_run": cycle_result.tasks_run,
                                    "tasks_passed": cycle_result.tasks_passed,
                                    "tasks_failed": cycle_result.tasks_failed,
                                },
                            )
                        except Exception:
                            self._logger.debug("Bus emit failed", exc_info=True)

                    # Periodic reload
                    if self._cycle_count % self.RELOAD_EVERY_N_CYCLES == 0:
                        await self._reload_checklist()
                else:
                    # No tasks due -- still mark cycle time so we don't spin
                    if self._schedule is not None:
                        self._schedule.last_cycle = time.monotonic()

                await asyncio.sleep(10)

        except asyncio.CancelledError:
            self._logger.debug("Daemon loop cancelled")
            raise

    async def _run_cycle(
        self, due_tasks: list[HeartbeatTask]
    ) -> HeartbeatCycleResult:
        """Run a single heartbeat cycle with the given due tasks.

        Executes in batches of ``MAX_CONCURRENT_TASKS`` to bound
        concurrency.  Updates task state and records results.
        """
        started_at = datetime.now(timezone.utc).isoformat()
        started_mono = time.monotonic()
        all_results: list[HeartbeatResult] = []

        # Process in batches
        for i in range(0, len(due_tasks), self.MAX_CONCURRENT_TASKS):
            batch = due_tasks[i : i + self.MAX_CONCURRENT_TASKS]
            batch_results = await asyncio.gather(
                *(self._executor.execute_task(t) for t in batch),
                return_exceptions=True,
            )
            for task, result in zip(batch, batch_results):
                if isinstance(result, BaseException):
                    # Shouldn't happen since executor catches, but just in case
                    result = HeartbeatResult(
                        task_id=task.id,
                        success=False,
                        output=f"Gather exception: {result}",
                        duration_ms=0.0,
                        executed_at=started_at,
                    )
                # Update task state
                task.last_run = time.monotonic()
                task.last_result = result.output[:500]
                if result.success:
                    task.failure_count = 0
                else:
                    task.failure_count += 1
                    if task.failure_count >= task.max_failures:
                        task.enabled = False
                        self._logger.warning(
                            "Task %s disabled after %d consecutive failures",
                            task.id,
                            task.failure_count,
                        )
                all_results.append(result)

        completed_at = datetime.now(timezone.utc).isoformat()
        elapsed_ms = (time.monotonic() - started_mono) * 1000.0
        passed = sum(1 for r in all_results if r.success)
        failed = sum(1 for r in all_results if not r.success)

        if self._schedule is not None:
            self._schedule.last_cycle = time.monotonic()

        cycle_result = HeartbeatCycleResult(
            cycle_number=self._cycle_count + 1,
            results=all_results,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=round(elapsed_ms, 2),
            tasks_run=len(all_results),
            tasks_passed=passed,
            tasks_failed=failed,
        )

        # Add to history, trimming if needed
        self._history.append(cycle_result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        self._logger.info(
            "Cycle %d complete: %d run, %d passed, %d failed (%.0fms)",
            cycle_result.cycle_number,
            cycle_result.tasks_run,
            passed,
            failed,
            elapsed_ms,
        )
        return cycle_result

    def _find_due_tasks(self) -> list[HeartbeatTask]:
        """Find tasks that are due for execution.

        A task is due if it is enabled and either has never been run or
        enough time has elapsed since its last run.
        """
        if self._schedule is None:
            return []

        now = time.monotonic()
        due: list[HeartbeatTask] = []
        for task in self._schedule.tasks:
            if not task.enabled:
                continue
            if task.last_run is None:
                due.append(task)
            elif (now - task.last_run) >= task.interval_seconds:
                due.append(task)
        return due

    async def _reload_checklist(self) -> None:
        """Reload HEARTBEAT.md, merging with existing task state.

        New tasks are added, deleted tasks are removed, and existing tasks
        retain their runtime state (``last_run``, ``failure_count``, etc.).
        """
        heartbeat_path = Path(self._workspace) / self.HEARTBEAT_FILENAME
        if not heartbeat_path.exists():
            self._logger.warning("HEARTBEAT.md not found during reload")
            return

        new_tasks = self._parser.parse_file(str(heartbeat_path))
        if self._schedule is None:
            self._schedule = HeartbeatSchedule(tasks=new_tasks)
            return

        # Build lookup of existing tasks by ID
        existing: dict[str, HeartbeatTask] = {
            t.id: t for t in self._schedule.tasks
        }

        merged: list[HeartbeatTask] = []
        for new_task in new_tasks:
            if new_task.id in existing:
                old = existing[new_task.id]
                # Preserve runtime state
                new_task.last_run = old.last_run
                new_task.last_result = old.last_result
                new_task.failure_count = old.failure_count
                # Enabled state comes from the file (checkbox)
                # but if it was auto-disabled due to failures keep it disabled
                if old.failure_count >= old.max_failures:
                    new_task.enabled = False
            merged.append(new_task)

        added = len(merged) - len(
            [t for t in merged if t.id in existing]
        )
        removed = len(existing) - len(
            [t for t in merged if t.id in existing]
        )
        self._schedule.tasks = merged
        self._logger.info(
            "Reloaded HEARTBEAT.md: %d tasks (%d added, %d removed)",
            len(merged),
            added,
            removed,
        )

    def _create_default_heartbeat(self) -> str:
        """Create a default HEARTBEAT.md template."""
        return (
            "# Heartbeat Checklist\n"
            "\n"
            "## Every 5 minutes\n"
            "- [ ] Check build status: `echo 'build OK'`\n"
            "- [ ] Verify workspace exists: `ls .`\n"
            "\n"
            "## Every 15 minutes\n"
            "- [ ] Run quick tests: `echo 'tests placeholder'`\n"
            "\n"
            "## Every hour\n"
            "- [ ] Full health check: `echo 'full check placeholder'`\n"
            "- [x] Example disabled task: `echo 'I am disabled'`\n"
        )

    # ----- Manual trigger / introspection -----

    async def trigger_manual(
        self, task_id: Optional[str] = None
    ) -> HeartbeatCycleResult:
        """Manually trigger a heartbeat cycle or a specific task.

        If *task_id* is provided only that task is executed, otherwise
        all enabled tasks are run.
        """
        if self._schedule is None:
            return HeartbeatCycleResult(
                cycle_number=0,
                results=[],
                started_at=datetime.now(timezone.utc).isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_ms=0.0,
                tasks_run=0,
                tasks_passed=0,
                tasks_failed=0,
            )

        if task_id is not None:
            targets = [t for t in self._schedule.tasks if t.id == task_id]
        else:
            targets = [t for t in self._schedule.tasks if t.enabled]

        return await self._run_cycle(targets)

    def get_status(self) -> dict:
        """Get daemon status as a dictionary."""
        task_count = 0
        enabled_count = 0
        recent_failures = 0
        if self._schedule is not None:
            task_count = len(self._schedule.tasks)
            enabled_count = sum(
                1 for t in self._schedule.tasks if t.enabled
            )
            recent_failures = sum(
                1
                for t in self._schedule.tasks
                if t.failure_count > 0 and t.enabled
            )

        last_cycle_time: Optional[str] = None
        next_cycle_estimate: Optional[str] = None
        if self._schedule is not None and self._schedule.last_cycle is not None:
            # Approximate wall-clock from monotonic
            mono_elapsed = time.monotonic() - self._schedule.last_cycle
            last_epoch = time.time() - mono_elapsed
            last_cycle_time = datetime.fromtimestamp(
                last_epoch, tz=timezone.utc
            ).isoformat()
            next_epoch = last_epoch + self._schedule.global_interval_seconds
            next_cycle_estimate = datetime.fromtimestamp(
                next_epoch, tz=timezone.utc
            ).isoformat()

        return {
            "running": self._running,
            "paused": self._paused,
            "cycle_count": self._cycle_count,
            "task_count": task_count,
            "enabled_count": enabled_count,
            "last_cycle": last_cycle_time,
            "next_cycle_estimate": next_cycle_estimate,
            "recent_failures": recent_failures,
        }

    def get_history(self, limit: int = 10) -> list[HeartbeatCycleResult]:
        """Get recent cycle history."""
        return self._history[-limit:]

    def enable_task(self, task_id: str) -> bool:
        """Enable a disabled task.  Returns True if found and enabled."""
        if self._schedule is None:
            return False
        for task in self._schedule.tasks:
            if task.id == task_id:
                task.enabled = True
                task.failure_count = 0
                self._logger.info("Task %s enabled", task_id)
                return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task.  Returns True if found and disabled."""
        if self._schedule is None:
            return False
        for task in self._schedule.tasks:
            if task.id == task_id:
                task.enabled = False
                self._logger.info("Task %s disabled", task_id)
                return True
        return False
