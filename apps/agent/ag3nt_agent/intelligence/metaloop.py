"""MetaLoop — central tick-based orchestrator for intelligence loops.

Ticks every N seconds, allocating resource budgets to registered
intelligence loops and monitoring their health.  Persists state
in a WAL-mode SQLite database at ``~/.ag3nt/metaloop.db``.
"""

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("ag3nt.intelligence.metaloop")

_DEFAULT_DB_PATH = Path.home() / ".ag3nt" / "metaloop.db"

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class MetaLoopDB:
    """WAL-mode SQLite store for MetaLoop state and metrics."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = str(path or _DEFAULT_DB_PATH)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    # -- schema -------------------------------------------------------------

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS loop_status (
                    name        TEXT PRIMARY KEY,
                    enabled     INTEGER NOT NULL DEFAULT 1,
                    last_tick   REAL,
                    tick_count  INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    budget      REAL NOT NULL DEFAULT 0.1
                );
                CREATE TABLE IF NOT EXISTS health_metrics (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   REAL NOT NULL,
                    loop_name   TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value       REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       REAL NOT NULL,
                    decision_type   TEXT NOT NULL,
                    reasoning       TEXT NOT NULL,
                    data            TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    # -- loop_status --------------------------------------------------------

    def update_loop_status(self, name: str, **kwargs: Any) -> None:
        """Upsert fields in *loop_status* for *name*."""
        # Ensure the row exists first.
        self._conn.execute(
            "INSERT OR IGNORE INTO loop_status (name) VALUES (?)", (name,)
        )
        if kwargs:
            assignments = ", ".join(f"{k} = ?" for k in kwargs)
            values = list(kwargs.values()) + [name]
            self._conn.execute(
                f"UPDATE loop_status SET {assignments} WHERE name = ?", values
            )
        self._conn.commit()

    def get_loop_status(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM loop_status WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_loop_status(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM loop_status").fetchall()
        return [dict(r) for r in rows]

    # -- decisions ----------------------------------------------------------

    def log_decision(
        self, decision_type: str, reasoning: str, data: dict | None = None
    ) -> None:
        self._conn.execute(
            "INSERT INTO decisions (timestamp, decision_type, reasoning, data) VALUES (?, ?, ?, ?)",
            (time.time(), decision_type, reasoning, json.dumps(data or {})),
        )
        self._conn.commit()

    # -- health_metrics -----------------------------------------------------

    def record_metric(
        self, loop_name: str, metric_name: str, value: float
    ) -> None:
        self._conn.execute(
            "INSERT INTO health_metrics (timestamp, loop_name, metric_name, value) VALUES (?, ?, ?, ?)",
            (time.time(), loop_name, metric_name, value),
        )
        self._conn.commit()

    # -- aggregate stats ----------------------------------------------------

    def get_stats(self) -> dict:
        total_ticks = (
            self._conn.execute(
                "SELECT COALESCE(SUM(tick_count), 0) FROM loop_status"
            ).fetchone()[0]
        )
        total_errors = (
            self._conn.execute(
                "SELECT COALESCE(SUM(error_count), 0) FROM loop_status"
            ).fetchone()[0]
        )
        decision_count = (
            self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        )
        return {
            "total_ticks": total_ticks,
            "total_errors": total_errors,
            "decision_count": decision_count,
        }

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MetaLoop:
    """Central tick-based orchestrator that drives intelligence loops."""

    def __init__(
        self,
        bus: Any,
        tick_interval_s: float = 10.0,
        db: MetaLoopDB | None = None,
    ) -> None:
        self._bus = bus
        self._tick_interval_s = tick_interval_s
        self._db = db or MetaLoopDB()

        self._loops: dict[str, Any] = {}
        self._budgets: dict[str, float] = {}
        self._whisper_mode: bool = False

        self._tick_task: asyncio.Task | None = None
        self._running = False

        # Track per-loop timing for runaway detection
        self._last_durations: dict[str, float] = {}

    # -- registration -------------------------------------------------------

    def register_loop(self, name: str, loop: Any, budget: float = 0.1) -> None:
        """Register an intelligence loop with a resource budget fraction."""
        self._loops[name] = loop
        self._budgets[name] = budget
        self._db.update_loop_status(name, budget=budget)
        logger.info("Registered intelligence loop '%s' (budget=%.2f)", name, budget)

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the tick loop."""
        if self._running:
            return
        self._running = True
        self._tick_task = asyncio.create_task(self._tick_loop())
        logger.info("MetaLoop started (interval=%.1fs)", self._tick_interval_s)

    async def stop(self) -> None:
        """Stop the tick loop and close the database."""
        self._running = False
        if self._tick_task:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
        self._db.close()
        logger.info("MetaLoop stopped")

    # -- tick ---------------------------------------------------------------

    async def _tick_loop(self) -> None:
        """Repeatedly run ``_tick`` at the configured interval."""
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("MetaLoop tick error: %s", exc, exc_info=True)
            await asyncio.sleep(self._tick_interval_s)

    async def _tick(self) -> None:
        """Run one tick: check health, run eligible loops, detect runaways."""
        for name, loop in self._loops.items():
            status = self._db.get_loop_status(name)
            if status and not status.get("enabled", True):
                continue

            # Whisper mode — skip every other tick per loop
            if self._whisper_mode:
                tick_count = (status or {}).get("tick_count", 0)
                if tick_count % 2 == 1:
                    self._db.update_loop_status(name, tick_count=tick_count + 1)
                    continue

            t0 = time.time()
            try:
                # If loop exposes an async tick method, call it
                tick_fn = getattr(loop, "tick", None)
                if tick_fn and asyncio.iscoroutinefunction(tick_fn):
                    await tick_fn()
                elif tick_fn:
                    tick_fn()

                duration = time.time() - t0
                self._last_durations[name] = duration
                self._db.record_metric(name, "tick_duration", duration)

                current = self._db.get_loop_status(name) or {}
                self._db.update_loop_status(
                    name,
                    last_tick=time.time(),
                    tick_count=current.get("tick_count", 0) + 1,
                )

                # Runaway detection
                if self._detect_runaway(name):
                    self._db.log_decision(
                        "runaway_detected",
                        f"Loop '{name}' exceeded 5x budget",
                        {"duration": duration, "budget": self._budgets.get(name, 0)},
                    )
                    logger.warning(
                        "Runaway detected for loop '%s' (%.2fs)", name, duration
                    )

            except Exception as exc:
                duration = time.time() - t0
                self._last_durations[name] = duration
                current = self._db.get_loop_status(name) or {}
                self._db.update_loop_status(
                    name,
                    last_tick=time.time(),
                    tick_count=current.get("tick_count", 0) + 1,
                    error_count=current.get("error_count", 0) + 1,
                )
                logger.error("Loop '%s' error: %s", name, exc)

    # -- whisper mode -------------------------------------------------------

    def set_whisper_mode(self, enabled: bool) -> None:
        """Toggle whisper mode (throttle loops to ~50%)."""
        self._whisper_mode = enabled
        self._db.log_decision(
            "whisper_mode",
            f"Whisper mode {'enabled' if enabled else 'disabled'}",
        )
        logger.info("Whisper mode %s", "enabled" if enabled else "disabled")

    # -- runaway detection --------------------------------------------------

    def _detect_runaway(self, name: str) -> bool:
        """Return True if *name*'s last tick took >5x its budget share of the interval."""
        budget = self._budgets.get(name, 0.1)
        allowed = self._tick_interval_s * budget * 5
        duration = self._last_durations.get(name, 0.0)
        return duration > allowed

    # -- status -------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a snapshot of the orchestrator state."""
        return {
            "running": self._running,
            "whisper_mode": self._whisper_mode,
            "tick_interval_s": self._tick_interval_s,
            "loops": list(self._loops.keys()),
            "budgets": dict(self._budgets),
            "last_durations": dict(self._last_durations),
            "db_stats": self._db.get_stats(),
        }
