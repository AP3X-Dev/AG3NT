"""
AG3NT Cron -- Scheduled task execution engine.

SQLite-backed persistent cron store with async scheduler loop.
Supports standard 5-field cron expressions (minute hour day_of_month month day_of_week).
Zero external dependencies -- pure asyncio + sqlite3.

Ported from ONI-BOT cron module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Optional

log = logging.getLogger("ag3nt.cron")


# ===================================================================
#  Cron Expression Parser
# ===================================================================


class CronExpression:
    """
    Parse and evaluate standard 5-field cron expressions.

    Fields: minute hour day_of_month month day_of_week
    Supports: * (any), */N (step), N (exact), N-M (range), N,M,O (list),
              N-M/S (range with step).

    Named shortcuts: ``@hourly``, ``@daily``, ``@weekly``, ``@monthly``,
    ``@yearly``, ``@annually``, ``@midnight``.  ``@reboot`` is accepted
    but treated as a sentinel that never matches (must be checked by the
    scheduler at startup).

    Day-of-week: 0=Sunday .. 6=Saturday
    """

    SHORTCUTS: dict[str, str] = {
        "@yearly": "0 0 1 1 *",
        "@annually": "0 0 1 1 *",
        "@monthly": "0 0 1 * *",
        "@weekly": "0 0 * * 0",
        "@daily": "0 0 * * *",
        "@midnight": "0 0 * * *",
        "@hourly": "0 * * * *",
    }

    FIELD_RANGES: list[tuple[int, int]] = [
        (0, 59),   # minute
        (0, 23),   # hour
        (1, 31),   # day of month
        (1, 12),   # month
        (0, 6),    # day of week (0=Sun)
    ]

    FIELD_NAMES = ["minute", "hour", "day_of_month", "month", "day_of_week"]

    def __init__(self, expression: str):
        self._raw = expression.strip()
        resolved = self.SHORTCUTS.get(self._raw, self._raw)
        if self._raw == "@reboot":
            # Sentinel: fields that never match any datetime
            self._fields: list[set[int]] = [set() for _ in range(5)]
            self._is_reboot = True
            return
        self._is_reboot = False
        parts = resolved.split()
        if len(parts) != 5:
            raise ValueError(
                f"Cron expression must have 5 fields, got {len(parts)}: "
                f"'{self._raw}'"
            )
        self._fields = []
        for i, part in enumerate(parts):
            min_val, max_val = self.FIELD_RANGES[i]
            self._fields.append(self._parse_field(part, min_val, max_val))

    def __repr__(self) -> str:
        return f"CronExpression('{self._raw}')"

    def __str__(self) -> str:
        return self._raw

    # ----- Field parser -----

    @staticmethod
    def _parse_field(field_str: str, min_val: int, max_val: int) -> set[int]:
        """
        Parse a single cron field to a set of valid integer values.

        Supported syntax:
            *       -- all values in [min_val, max_val]
            */N     -- every N-th value starting from min_val
            N       -- single value
            N-M     -- inclusive range
            N-M/S   -- range with step
            N,M,O   -- comma-separated list (each element may be a range or step)
        """
        result: set[int] = set()

        for token in field_str.split(","):
            token = token.strip()

            # */N -- step over full range
            if token.startswith("*/"):
                step_str = token[2:]
                if not step_str.isdigit() or int(step_str) == 0:
                    raise ValueError(f"Invalid step value: '{token}'")
                step = int(step_str)
                result.update(range(min_val, max_val + 1, step))

            # * -- wildcard
            elif token == "*":
                result.update(range(min_val, max_val + 1))

            # N-M or N-M/S -- range with optional step
            elif "-" in token:
                range_part, _, step_part = token.partition("/")
                parts = range_part.split("-", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid range: '{token}'")

                lo_str, hi_str = parts
                if not lo_str.isdigit() or not hi_str.isdigit():
                    raise ValueError(f"Invalid range values: '{token}'")

                lo, hi = int(lo_str), int(hi_str)
                if lo < min_val or hi > max_val or lo > hi:
                    raise ValueError(
                        f"Range {lo}-{hi} out of bounds "
                        f"[{min_val}-{max_val}]"
                    )

                step = 1
                if step_part:
                    if not step_part.isdigit() or int(step_part) == 0:
                        raise ValueError(f"Invalid step in range: '{token}'")
                    step = int(step_part)

                result.update(range(lo, hi + 1, step))

            # N -- single value
            elif token.isdigit():
                val = int(token)
                if val < min_val or val > max_val:
                    raise ValueError(
                        f"Value {val} out of bounds [{min_val}-{max_val}]"
                    )
                result.add(val)

            else:
                raise ValueError(f"Unrecognised cron field token: '{token}'")

        return result

    # ----- Match & next-run -----

    @staticmethod
    def weekday_to_cron(dt: datetime) -> int:
        """Convert Python weekday (Mon=0) to cron weekday (Sun=0)."""
        return dt.isoweekday() % 7  # isoweekday: Mon=1..Sun=7 -> Sun=0

    def matches(self, dt: datetime) -> bool:
        """Return True if *dt* matches every field of this cron expression."""
        if self._is_reboot:
            return False
        cron_dow = self.weekday_to_cron(dt)
        return (
            dt.minute in self._fields[0]
            and dt.hour in self._fields[1]
            and dt.day in self._fields[2]
            and dt.month in self._fields[3]
            and cron_dow in self._fields[4]
        )

    def next_run(self, after: Optional[datetime] = None) -> datetime:
        """
        Compute the next datetime that matches this cron expression.

        Search starts one minute after *after* (default: now) and uses
        skip-ahead logic for months, days, and hours to avoid brute-force
        minute iteration.  Gives up after scanning ~366 days.
        """
        if self._is_reboot:
            raise RuntimeError("@reboot expression has no recurring next run")

        if after is None:
            after = datetime.now(timezone.utc)

        # Start from the next whole minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Upper bound: ~366 days of minutes
        limit = 366 * 24 * 60

        for _ in range(limit):
            # Skip entire month if month does not match
            if candidate.month not in self._fields[3]:
                if candidate.month == 12:
                    candidate = candidate.replace(
                        year=candidate.year + 1, month=1, day=1,
                        hour=0, minute=0,
                    )
                else:
                    candidate = candidate.replace(
                        month=candidate.month + 1, day=1,
                        hour=0, minute=0,
                    )
                continue

            # Skip day if neither dom nor dow match
            cron_dow = candidate.isoweekday() % 7
            if (
                candidate.day not in self._fields[2]
                or cron_dow not in self._fields[4]
            ):
                candidate = (candidate + timedelta(days=1)).replace(
                    hour=0, minute=0,
                )
                continue

            # Skip hour if hour does not match
            if candidate.hour not in self._fields[1]:
                candidate = candidate.replace(minute=0) + timedelta(hours=1)
                continue

            if candidate.minute in self._fields[0]:
                return candidate
            candidate += timedelta(minutes=1)

        raise RuntimeError(
            f"Could not find next run for '{self._raw}' within 366 days "
            f"after {after.isoformat()}"
        )


# ===================================================================
#  Cron Job Data Model
# ===================================================================


@dataclass
class CronJob:
    """A single scheduled task stored in the SQLite backend."""

    id: int
    name: str
    schedule: str           # cron expression string (5 fields)
    action: str             # message to send to the agent
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_count: int = 0
    last_error: Optional[str] = None


# ===================================================================
#  SQLite-Backed Cron Store
# ===================================================================


class CronStore:
    """
    Persistent storage for cron jobs.

    Uses a single SQLite file.  Thread-safe for single-writer usage.
    Default location: ``~/.ag3nt/cron.db``
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".ag3nt" / "cron.db")
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ----- Schema -----

    def _init_schema(self) -> None:
        """Create the cron_jobs table if it does not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cron_jobs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL,
                schedule    TEXT    NOT NULL,
                action      TEXT    NOT NULL,
                enabled     INTEGER NOT NULL DEFAULT 1,
                last_run    TEXT,
                next_run    TEXT,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                run_count   INTEGER NOT NULL DEFAULT 0,
                last_error  TEXT
            )
        """)
        self._conn.commit()

    # ----- CRUD -----

    def add(self, job: CronJob) -> int:
        """Insert a new cron job and return its ID."""
        # Validate the expression early
        cron = CronExpression(job.schedule)
        computed_next = cron.next_run()

        cur = self._conn.execute(
            """INSERT INTO cron_jobs
                   (name, schedule, action, enabled, last_run, next_run,
                    created_at, run_count, last_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.name,
                job.schedule,
                job.action,
                int(job.enabled),
                job.last_run.isoformat() if job.last_run else None,
                computed_next.isoformat(),
                job.created_at.isoformat(),
                job.run_count,
                job.last_error,
            ),
        )
        self._conn.commit()
        job_id = cur.lastrowid or 0
        log.info("Added cron job #%d: '%s' [%s]", job_id, job.name, job.schedule)
        return job_id

    def get(self, job_id: int) -> Optional[CronJob]:
        """Fetch a single job by ID, or None."""
        row = self._conn.execute(
            "SELECT * FROM cron_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def list_all(self) -> list[CronJob]:
        """Return every cron job."""
        rows = self._conn.execute(
            "SELECT * FROM cron_jobs ORDER BY id"
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def list_enabled(self) -> list[CronJob]:
        """Return only enabled cron jobs."""
        rows = self._conn.execute(
            "SELECT * FROM cron_jobs WHERE enabled = 1 ORDER BY id"
        ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def update(self, job: CronJob) -> None:
        """Overwrite an existing job (matched by id)."""
        self._conn.execute(
            """UPDATE cron_jobs
               SET name = ?, schedule = ?, action = ?, enabled = ?,
                   last_run = ?, next_run = ?, run_count = ?, last_error = ?
               WHERE id = ?""",
            (
                job.name,
                job.schedule,
                job.action,
                int(job.enabled),
                job.last_run.isoformat() if job.last_run else None,
                job.next_run.isoformat() if job.next_run else None,
                job.run_count,
                job.last_error,
                job.id,
            ),
        )
        self._conn.commit()

    def remove(self, job_id: int) -> bool:
        """Delete a job by ID.  Returns True if a row was removed."""
        cur = self._conn.execute(
            "DELETE FROM cron_jobs WHERE id = ?", (job_id,)
        )
        self._conn.commit()
        removed = cur.rowcount > 0
        if removed:
            log.info("Removed cron job #%d", job_id)
        return removed

    # ----- Run recording -----

    def record_run(
        self,
        job_id: int,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """
        Update last_run, run_count, next_run, and last_error after execution.

        Re-computes next_run from the job's schedule expression.
        """
        job = self.get(job_id)
        if job is None:
            log.warning("record_run: job #%d not found", job_id)
            return

        now = datetime.now(timezone.utc)
        try:
            cron = CronExpression(job.schedule)
            next_dt = cron.next_run(after=now)
        except Exception as exc:
            next_dt = None
            log.error("Failed to compute next_run for job #%d: %s", job_id, exc)

        self._conn.execute(
            """UPDATE cron_jobs
               SET last_run   = ?,
                   run_count  = run_count + 1,
                   next_run   = ?,
                   last_error = ?
               WHERE id = ?""",
            (
                now.isoformat(),
                next_dt.isoformat() if next_dt else None,
                error if not success else None,
                job_id,
            ),
        )
        self._conn.commit()
        log.debug(
            "Recorded run for job #%d: success=%s  next_run=%s",
            job_id,
            success,
            next_dt,
        )

    # ----- Helpers -----

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> CronJob:
        """Convert a sqlite3.Row to a CronJob dataclass."""
        return CronJob(
            id=row["id"],
            name=row["name"],
            schedule=row["schedule"],
            action=row["action"],
            enabled=bool(row["enabled"]),
            last_run=(
                datetime.fromisoformat(row["last_run"])
                if row["last_run"] else None
            ),
            next_run=(
                datetime.fromisoformat(row["next_run"])
                if row["next_run"] else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
            run_count=row["run_count"],
            last_error=row["last_error"],
        )

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()


# ===================================================================
#  Async Cron Scheduler
# ===================================================================


class CronScheduler:
    """
    Background scheduler that checks for due cron jobs every 60 seconds.

    When a job is due (now >= next_run), the scheduler invokes the
    registered async *callback* with the job's action string and records
    the result in the store.

    Config-gated: disabled by default.  The caller is responsible for
    creating a ``CronStore`` and passing an async callback.
    """

    TICK_INTERVAL = 60  # seconds

    def __init__(
        self,
        store: CronStore,
        callback: Callable[[str], Awaitable[str]],
    ):
        self._store = store
        self._callback = callback
        self._task: Optional[asyncio.Task] = None

    # ----- Lifecycle -----

    async def start(self) -> None:
        """Launch the scheduler as a background asyncio task."""
        if self._task is not None and not self._task.done():
            log.warning("CronScheduler already running -- ignoring start()")
            return

        self._task = asyncio.create_task(self._loop(), name="ag3nt.cron")
        log.info("CronScheduler started  tick_interval=%ds", self.TICK_INTERVAL)

    async def stop(self) -> None:
        """Cancel the background scheduler gracefully."""
        if self._task is None or self._task.done():
            log.debug("CronScheduler not running -- nothing to stop")
            return

        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

        self._task = None
        log.info("CronScheduler stopped")

    # ----- Properties -----

    @property
    def is_running(self) -> bool:
        """True if the background scheduler task is alive."""
        return self._task is not None and not self._task.done()

    # ----- Tick logic -----

    async def _loop(self) -> None:
        """Sleep -> tick -> repeat."""
        log.debug("CronScheduler loop entered")
        try:
            while True:
                await asyncio.sleep(self.TICK_INTERVAL)
                await self._tick()
        except asyncio.CancelledError:
            log.debug("CronScheduler loop cancelled")
            raise

    async def _tick(self) -> None:
        """
        Check all enabled jobs and execute those whose next_run has passed.
        """
        now = datetime.now(timezone.utc)
        jobs = self._store.list_enabled()

        for job in jobs:
            if job.next_run is None:
                # Compute if missing
                try:
                    cron = CronExpression(job.schedule)
                    job.next_run = cron.next_run(after=now)
                    self._store.update(job)
                except Exception as exc:
                    log.error(
                        "Bad schedule for job #%d '%s': %s", job.id, job.name, exc
                    )
                continue

            if now >= job.next_run:
                log.info("Cron job #%d '%s' is due -- executing", job.id, job.name)
                await self._execute_job(job)

    async def _execute_job(self, job: CronJob) -> None:
        """Run a single job's action via the callback and record the outcome."""
        try:
            result = await self._callback(job.action)
            self._store.record_run(job.id, success=True)
            log.info(
                "Cron job #%d '%s' completed  result_len=%d",
                job.id,
                job.name,
                len(result),
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            self._store.record_run(job.id, success=False, error=error_msg)
            log.error(
                "Cron job #%d '%s' failed: %s",
                job.id,
                job.name,
                error_msg,
                exc_info=True,
            )

    # ----- Manual trigger -----

    async def run_now(self, job_id: int) -> str:
        """
        Manually trigger a job regardless of its schedule.

        Returns the callback result string, or raises if the job is not found.
        """
        job = self._store.get(job_id)
        if job is None:
            raise ValueError(f"Cron job #{job_id} not found")

        log.info("Manual trigger: job #%d '%s'", job.id, job.name)
        try:
            result = await self._callback(job.action)
            self._store.record_run(job.id, success=True)
            return result
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            self._store.record_run(job.id, success=False, error=error_msg)
            raise


# ===================================================================
#  Isolated Session Support
# ===================================================================


@dataclass
class IsolatedSessionConfig:
    """Configuration for a cron job's isolated session.

    Controls how the agent session is set up when a cron job fires,
    including history limits, model selection, and timeout.
    """

    session_key_prefix: str = "cron"
    max_history_turns: int = 20
    system_prompt: Optional[str] = None
    model_override: Optional[str] = None
    timeout_seconds: int = 120
    persist_history: bool = False
    workspace_dir: Optional[str] = None


@dataclass
class CronJobDefinition:
    """A cron job definition with its schedule and session config.

    Stores everything needed to schedule and execute a recurring task
    inside an isolated agent session.
    """

    id: str
    name: str
    expression: CronExpression
    command: str
    session_config: IsolatedSessionConfig
    enabled: bool = True
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_run: Optional[str] = None
    last_result: Optional[str] = None
    run_count: int = 0
    failure_count: int = 0
    max_failures: int = 10
    tags: list[str] = field(default_factory=list)


@dataclass
class CronJobResult:
    """Result of a single cron job execution."""

    job_id: str
    session_key: str
    success: bool
    output: str
    duration_ms: float
    started_at: str
    completed_at: str
    error: Optional[str] = None


# ===================================================================
#  Cron Session Manager
# ===================================================================


class CronSessionManager:
    """Manages isolated sessions for cron jobs.

    Each cron job gets its own *session_key* derived from the job ID.
    Sessions can optionally persist message history between runs so that
    the agent retains context of prior executions.
    """

    def __init__(self, workspace_dir: str = "."):
        self._sessions: dict[str, list[dict]] = {}
        self._workspace = workspace_dir
        self._logger = logging.getLogger("ag3nt.cron.sessions")

    def get_session_key(self, job: CronJobDefinition) -> str:
        """Generate a deterministic session key for a cron job.

        Format: ``cron:<job_id>:<name_slug>`` where *name_slug* is a
        lowercased, hyphen-separated, alphanumeric-only version of the
        job name.
        """
        slug = re.sub(r"[^a-z0-9]+", "-", job.name.lower()).strip("-") or "job"
        return f"cron:{job.id}:{slug}"

    def get_history(self, session_key: str) -> list[dict]:
        """Return a *copy* of the message history for *session_key*."""
        return list(self._sessions.get(session_key, []))

    def append_history(
        self, session_key: str, message: dict, max_turns: int = 20
    ) -> None:
        """Append *message* to the session's history.

        If the history exceeds *max_turns* messages the oldest entries
        are dropped to keep it within limit.
        """
        history = self._sessions.setdefault(session_key, [])
        history.append(message)
        # Trim from the front while preserving max_turns most-recent
        while len(history) > max_turns:
            history.pop(0)
        self._logger.debug(
            "Session %s history: %d messages", session_key, len(history)
        )

    def clear_history(self, session_key: str) -> None:
        """Remove all messages from the session."""
        self._sessions.pop(session_key, None)
        self._logger.debug("Cleared history for session %s", session_key)

    def build_system_prompt(self, job: CronJobDefinition) -> str:
        """Build a system prompt tailored for a cron session.

        Includes the job name, workspace info, and a directive to be
        concise.  If the job's session config specifies a custom system
        prompt it is appended verbatim.
        """
        workspace = job.session_config.workspace_dir or self._workspace
        parts = [
            f"You are running as a scheduled task named '{job.name}'.",
            "Be concise and action-oriented. Report results clearly.",
            f"Workspace directory: {workspace}",
            f"Job ID: {job.id}",
        ]
        if job.tags:
            parts.append(f"Tags: {', '.join(job.tags)}")
        if job.session_config.system_prompt:
            parts.append("")
            parts.append(job.session_config.system_prompt)
        return "\n".join(parts)


# ===================================================================
#  Isolated-session Cron Scheduler
# ===================================================================


class IsolatedCronScheduler:
    """Cron scheduler with isolated session support.

    Features:
    - Extended cron expression parsing (shortcuts + standard 5-field)
    - Isolated sessions per job -- no user conversation pollution
    - Persistent job storage via a JSON file
    - Concurrent job execution with configurable limits
    - Failure tracking with automatic disable after *max_failures*
    - Event emission (via an optional event bus) for monitoring
    """

    STORAGE_FILENAME = ".ag3nt_cron_jobs.json"
    MAX_CONCURRENT_JOBS = 3
    CHECK_INTERVAL_SECONDS = 30

    def __init__(
        self,
        workspace_dir: str = ".",
        bus: object = None,
        agent_runner: Optional[Callable[..., Awaitable[str]]] = None,
    ):
        self._workspace = workspace_dir
        self._bus = bus
        self._agent_runner = agent_runner
        self._session_mgr = CronSessionManager(workspace_dir)
        self._jobs: dict[str, CronJobDefinition] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._active_runs: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._history: list[CronJobResult] = []
        self._max_history = 100
        self._logger = logging.getLogger("ag3nt.cron.scheduler")

    # ----- Lifecycle -----

    async def start(self) -> None:
        """Start the cron scheduler.

        Loads persisted job definitions and launches the background
        check loop.
        """
        if self._running:
            self._logger.warning("Scheduler already running")
            return
        self._load_jobs()
        self._running = True
        self._task = asyncio.create_task(
            self._check_loop(), name="ag3nt.cron.isolated"
        )
        self._logger.info(
            "IsolatedCronScheduler started  jobs=%d  interval=%ds",
            len(self._jobs),
            self.CHECK_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        """Stop gracefully: cancel pending, wait for active jobs."""
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Wait for in-progress job runs to finish (with a timeout)
        for job_id, task in list(self._active_runs.items()):
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        self._active_runs.clear()
        self._save_jobs()
        self._logger.info("IsolatedCronScheduler stopped")

    # ----- Job management -----

    def add_job(
        self,
        name: str,
        expression: str,
        command: str,
        session_config: Optional[IsolatedSessionConfig] = None,
        tags: Optional[list[str]] = None,
    ) -> CronJobDefinition:
        """Add a new cron job and return the created definition.

        The expression is validated immediately; a ``ValueError`` is
        raised for invalid cron syntax.
        """
        cron_expr = CronExpression(expression)
        config = session_config or IsolatedSessionConfig()
        job = CronJobDefinition(
            id=str(uuid.uuid4()),
            name=name,
            expression=cron_expr,
            command=command,
            session_config=config,
            tags=list(tags) if tags else [],
        )
        self._jobs[job.id] = job
        self._save_jobs()
        self._logger.info("Added job '%s' [%s]  id=%s", name, expression, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """Remove a cron job by ID.  Returns ``True`` if found and removed."""
        if job_id in self._jobs:
            removed = self._jobs.pop(job_id)
            session_key = self._session_mgr.get_session_key(removed)
            self._session_mgr.clear_history(session_key)
            self._save_jobs()
            self._logger.info("Removed job '%s' id=%s", removed.name, job_id)
            return True
        return False

    def enable_job(self, job_id: str) -> bool:
        """Enable a previously disabled job."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.enabled = True
        self._save_jobs()
        self._logger.info("Enabled job '%s' id=%s", job.name, job_id)
        return True

    def disable_job(self, job_id: str) -> bool:
        """Disable a job so it will not be scheduled."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.enabled = False
        self._save_jobs()
        self._logger.info("Disabled job '%s' id=%s", job.name, job_id)
        return True

    def list_jobs(self, tag: Optional[str] = None) -> list[CronJobDefinition]:
        """List all jobs, optionally filtered by *tag*."""
        jobs = list(self._jobs.values())
        if tag is not None:
            jobs = [j for j in jobs if tag in j.tags]
        return jobs

    async def trigger_job(self, job_id: str) -> Optional[CronJobResult]:
        """Manually trigger a job immediately, bypassing the schedule."""
        job = self._jobs.get(job_id)
        if job is None:
            self._logger.warning("trigger_job: unknown job_id=%s", job_id)
            return None
        return await self._execute_job(job)

    # ----- Check loop -----

    async def _check_loop(self) -> None:
        """Main loop: check for due jobs every CHECK_INTERVAL_SECONDS."""
        self._logger.debug("Check loop entered")
        try:
            while self._running:
                await self._tick()
                await asyncio.sleep(self.CHECK_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            self._logger.debug("Check loop cancelled")
            raise

    async def _tick(self) -> None:
        """Evaluate all enabled jobs against the current time."""
        now = datetime.now(timezone.utc)
        # Clean up completed tasks
        finished = [
            jid for jid, t in self._active_runs.items() if t.done()
        ]
        for jid in finished:
            self._active_runs.pop(jid, None)

        for job in self._jobs.values():
            if not job.enabled:
                continue
            if job.id in self._active_runs:
                continue  # already running
            if len(self._active_runs) >= self.MAX_CONCURRENT_JOBS:
                break  # concurrency limit reached
            if job.expression.matches(now):
                # Avoid re-triggering within the same minute
                if job.last_run:
                    last_dt = datetime.fromisoformat(job.last_run)
                    if (now - last_dt).total_seconds() < 60:
                        continue
                self._logger.info(
                    "Job '%s' is due -- launching  id=%s", job.name, job.id
                )
                task = asyncio.create_task(
                    self._execute_job(job),
                    name=f"ag3nt.cron.run.{job.id[:8]}",
                )
                self._active_runs[job.id] = task

    # ----- Execution -----

    async def _execute_job(self, job: CronJobDefinition) -> CronJobResult:
        """Execute a single cron job inside its isolated session."""
        session_key = self._session_mgr.get_session_key(job)
        system_prompt = self._session_mgr.build_system_prompt(job)
        started_at = datetime.now(timezone.utc)
        t_start = time.monotonic()

        # Prepare user message
        user_msg: dict = {"role": "user", "content": job.command}

        # Retrieve history and append user message
        if not job.session_config.persist_history:
            self._session_mgr.clear_history(session_key)
        self._session_mgr.append_history(
            session_key, user_msg,
            max_turns=job.session_config.max_history_turns,
        )

        output = ""
        error_text: Optional[str] = None
        success = True

        try:
            if self._agent_runner is not None:
                output = await asyncio.wait_for(
                    self._agent_runner(session_key, job.command, {
                        "system_prompt": system_prompt,
                        "history": self._session_mgr.get_history(session_key),
                        "model_override": job.session_config.model_override,
                        "workspace_dir": (
                            job.session_config.workspace_dir or self._workspace
                        ),
                    }),
                    timeout=job.session_config.timeout_seconds,
                )
            else:
                output = "[no agent_runner configured -- dry run]"
                self._logger.warning(
                    "No agent_runner for job '%s'; performing dry run", job.name
                )
        except asyncio.TimeoutError:
            success = False
            error_text = (
                f"Job timed out after {job.session_config.timeout_seconds}s"
            )
            self._logger.error("Job '%s' timed out  id=%s", job.name, job.id)
        except Exception as exc:
            success = False
            error_text = f"{type(exc).__name__}: {exc}"
            self._logger.error(
                "Job '%s' failed: %s", job.name, error_text, exc_info=True
            )

        t_end = time.monotonic()
        completed_at = datetime.now(timezone.utc)
        duration_ms = (t_end - t_start) * 1000.0

        # Record assistant response in session history
        assistant_msg: dict = {"role": "assistant", "content": output or ""}
        self._session_mgr.append_history(
            session_key, assistant_msg,
            max_turns=job.session_config.max_history_turns,
        )

        # Update job state
        job.last_run = completed_at.isoformat()
        job.last_result = output[:500] if output else error_text
        job.run_count += 1
        if not success:
            job.failure_count += 1
            if job.failure_count >= job.max_failures:
                job.enabled = False
                self._logger.warning(
                    "Auto-disabled job '%s' after %d failures  id=%s",
                    job.name, job.failure_count, job.id,
                )

        # Build result object
        result = CronJobResult(
            job_id=job.id,
            session_key=session_key,
            success=success,
            output=output,
            duration_ms=round(duration_ms, 2),
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            error=error_text,
        )

        # Append to scheduler history (bounded)
        self._history.append(result)
        while len(self._history) > self._max_history:
            self._history.pop(0)

        # Emit event if bus is available
        if self._bus is not None and hasattr(self._bus, "emit"):
            try:
                self._bus.emit("cron.job.completed", {
                    "job_id": job.id,
                    "job_name": job.name,
                    "success": success,
                    "duration_ms": result.duration_ms,
                    "error": error_text,
                })
            except Exception:
                self._logger.debug("Failed to emit cron event", exc_info=True)

        self._save_jobs()
        return result

    # ----- Persistence -----

    def _save_jobs(self) -> None:
        """Persist all job definitions to a JSON file."""
        storage_path = Path(self._workspace) / self.STORAGE_FILENAME
        data: list[dict] = []
        for job in self._jobs.values():
            data.append({
                "id": job.id,
                "name": job.name,
                "expression": str(job.expression),
                "command": job.command,
                "enabled": job.enabled,
                "created_at": job.created_at,
                "last_run": job.last_run,
                "last_result": job.last_result,
                "run_count": job.run_count,
                "failure_count": job.failure_count,
                "max_failures": job.max_failures,
                "tags": job.tags,
                "session_config": {
                    "session_key_prefix": job.session_config.session_key_prefix,
                    "max_history_turns": job.session_config.max_history_turns,
                    "system_prompt": job.session_config.system_prompt,
                    "model_override": job.session_config.model_override,
                    "timeout_seconds": job.session_config.timeout_seconds,
                    "persist_history": job.session_config.persist_history,
                    "workspace_dir": job.session_config.workspace_dir,
                },
            })
        try:
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            self._logger.error("Failed to save jobs: %s", exc)

    def _load_jobs(self) -> None:
        """Load job definitions from the JSON storage file."""
        storage_path = Path(self._workspace) / self.STORAGE_FILENAME
        if not storage_path.exists():
            self._logger.debug("No job storage file at %s", storage_path)
            return
        try:
            raw = storage_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            self._logger.error("Failed to load jobs: %s", exc)
            return

        for entry in data:
            try:
                sc_data = entry.get("session_config", {})
                config = IsolatedSessionConfig(
                    session_key_prefix=sc_data.get("session_key_prefix", "cron"),
                    max_history_turns=sc_data.get("max_history_turns", 20),
                    system_prompt=sc_data.get("system_prompt"),
                    model_override=sc_data.get("model_override"),
                    timeout_seconds=sc_data.get("timeout_seconds", 120),
                    persist_history=sc_data.get("persist_history", False),
                    workspace_dir=sc_data.get("workspace_dir"),
                )
                job = CronJobDefinition(
                    id=entry["id"],
                    name=entry["name"],
                    expression=CronExpression(entry["expression"]),
                    command=entry["command"],
                    session_config=config,
                    enabled=entry.get("enabled", True),
                    created_at=entry.get("created_at", ""),
                    last_run=entry.get("last_run"),
                    last_result=entry.get("last_result"),
                    run_count=entry.get("run_count", 0),
                    failure_count=entry.get("failure_count", 0),
                    max_failures=entry.get("max_failures", 10),
                    tags=entry.get("tags", []),
                )
                self._jobs[job.id] = job
            except Exception as exc:
                self._logger.warning(
                    "Skipping invalid job entry: %s -- %s",
                    entry.get("id"),
                    exc,
                )

        self._logger.info("Loaded %d jobs from storage", len(self._jobs))

    # ----- Stats & history -----

    def get_stats(self) -> dict:
        """Return scheduler statistics as a plain dictionary."""
        enabled = [j for j in self._jobs.values() if j.enabled]
        disabled = [j for j in self._jobs.values() if not j.enabled]
        total_runs = sum(j.run_count for j in self._jobs.values())
        total_failures = sum(j.failure_count for j in self._jobs.values())
        details = []
        for j in self._jobs.values():
            details.append({
                "id": j.id,
                "name": j.name,
                "expression": str(j.expression),
                "enabled": j.enabled,
                "run_count": j.run_count,
                "failure_count": j.failure_count,
                "last_run": j.last_run,
            })
        return {
            "total_jobs": len(self._jobs),
            "enabled": len(enabled),
            "disabled": len(disabled),
            "active_runs": len(self._active_runs),
            "total_runs": total_runs,
            "total_failures": total_failures,
            "job_details": details,
        }

    def get_job_history(
        self,
        job_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[CronJobResult]:
        """Return execution history, optionally filtered to a single job.

        Results are ordered most-recent-first, capped at *limit*.
        """
        results = self._history
        if job_id is not None:
            results = [r for r in results if r.job_id == job_id]
        # Most recent first
        return list(reversed(results[-limit:]))
