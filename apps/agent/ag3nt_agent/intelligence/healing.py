"""
Self-Healing Intelligence Loop for AG3NT.

Monitors recovery events and builds a database of failure patterns.
After >=3 occurrences of the same error pattern, auto-generates a
remediation playbook.  Also detects regressions (previously fixed errors
reappearing).

Architecture
------------
EventBus --> SelfHealingIntelligenceLoop --> HealingDB (SQLite WAL)
                                        --> RemediationEngine
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PLAYBOOK_THRESHOLD = 3  # minimum occurrences before playbook generation

# Regex patterns used by classify_error to normalise messages
_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_ABS_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?(?:/|\\)(?:[\w.\-]+(?:/|\\))+[\w.\-]*"
)
_LINE_NUMBER_RE = re.compile(r"(?:line\s+)\d+|(?::\d+(?::\d+)?)")
_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")

# Error class detection — order matters (first match wins)
_ERROR_CLASS_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("import_error", re.compile(r"(?:ModuleNotFoundError|ImportError)", re.I)),
    ("connection_error", re.compile(
        r"(?:ConnectionRefusedError|ConnectionResetError|ConnectionError|"
        r"ConnectionAbortedError|BrokenPipeError|socket\.error|"
        r"requests\.exceptions\.ConnectionError|ECONNREFUSED|ECONNRESET)", re.I,
    )),
    ("permission_error", re.compile(
        r"(?:PermissionError|EACCES|AccessDenied|PermissionDenied|permission denied)", re.I,
    )),
    ("type_error", re.compile(r"(?:TypeError|AttributeError)", re.I)),
    ("timeout_error", re.compile(
        r"(?:TimeoutError|asyncio\.TimeoutError|ReadTimeout|ConnectTimeout|"
        r"timed?\s*out)", re.I,
    )),
    ("resource_error", re.compile(
        r"(?:MemoryError|OSError|IOError|ResourceExhausted|"
        r"FileNotFoundError|ENOSPC|ENOMEM|disk full)", re.I,
    )),
]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_error(error_msg: str) -> tuple[str, str]:
    """Normalise and classify an error message.

    Strips UUIDs, absolute paths, line numbers, timestamps, and hex
    addresses so that semantically identical errors produce the same hash
    regardless of runtime-specific details.

    Returns
    -------
    (error_hash, error_class)
        *error_hash* is a deterministic SHA-256 (hex, 16 chars) of the
        normalised message.  *error_class* is one of the known categories
        or ``"unknown"``.
    """
    normalised = error_msg.strip()

    # Strip variable parts
    normalised = _UUID_RE.sub("<UUID>", normalised)
    normalised = _TIMESTAMP_RE.sub("<TS>", normalised)
    normalised = _ABS_PATH_RE.sub("<PATH>", normalised)
    normalised = _LINE_NUMBER_RE.sub("<LN>", normalised)
    normalised = _HEX_ADDR_RE.sub("<ADDR>", normalised)

    # Collapse whitespace
    normalised = re.sub(r"\s+", " ", normalised).strip()

    error_hash = hashlib.sha256(normalised.encode()).hexdigest()[:16]

    # Classify
    error_class = "unknown"
    for cls_name, pattern in _ERROR_CLASS_PATTERNS:
        if pattern.search(error_msg):
            error_class = cls_name
            break

    return error_hash, error_class


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    """A single failure / recovery event."""

    id: str
    component: str
    error_class: str
    error_hash: str
    error_message: str
    recovery_level: int
    recovery_action: str
    success: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class FailurePattern:
    """Aggregate statistics for a recurring error pattern."""

    error_hash: str
    error_class: str
    occurrence_count: int
    first_seen: float
    last_seen: float
    success_count: int
    failure_count: int


# ---------------------------------------------------------------------------
# HealingDB — WAL-mode SQLite
# ---------------------------------------------------------------------------

class HealingDB:
    """Persistent store for failure logs, patterns, and remediation playbooks.

    Uses SQLite in WAL mode at ``~/.ag3nt/healing.db``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            base = Path.home() / ".ag3nt"
            base.mkdir(parents=True, exist_ok=True)
            db_path = str(base / "healing.db")
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    # -- lifecycle -----------------------------------------------------------

    def init_db(self) -> None:
        """Create tables and enable WAL mode."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS failure_logs (
                id           TEXT PRIMARY KEY,
                component    TEXT NOT NULL,
                error_class  TEXT NOT NULL,
                error_hash   TEXT NOT NULL,
                error_message TEXT NOT NULL,
                recovery_level INTEGER NOT NULL,
                recovery_action TEXT NOT NULL DEFAULT '',
                success      INTEGER NOT NULL DEFAULT 0,
                timestamp    REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_failure_logs_hash
                ON failure_logs(error_hash);
            CREATE INDEX IF NOT EXISTS idx_failure_logs_ts
                ON failure_logs(timestamp);

            CREATE TABLE IF NOT EXISTS failure_patterns (
                error_hash       TEXT PRIMARY KEY,
                error_class      TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 0,
                first_seen       REAL NOT NULL,
                last_seen        REAL NOT NULL,
                success_count    INTEGER NOT NULL DEFAULT 0,
                failure_count    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS predictive_signals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                error_hash   TEXT NOT NULL,
                signal_type  TEXT NOT NULL,
                signal_value TEXT NOT NULL,
                timestamp    REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pred_hash
                ON predictive_signals(error_hash);

            CREATE TABLE IF NOT EXISTS remediation_playbooks (
                error_hash   TEXT PRIMARY KEY,
                playbook     TEXT NOT NULL,
                created_at   REAL NOT NULL,
                updated_at   REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- writes --------------------------------------------------------------

    def log_failure(self, record: FailureRecord) -> None:
        """Insert a failure record into ``failure_logs``."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO failure_logs
                (id, component, error_class, error_hash, error_message,
                 recovery_level, recovery_action, success, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.component,
                record.error_class,
                record.error_hash,
                record.error_message,
                record.recovery_level,
                record.recovery_action,
                int(record.success),
                record.timestamp,
            ),
        )
        self._conn.commit()

    def update_pattern(
        self, error_hash: str, error_class: str, success: bool
    ) -> FailurePattern:
        """Upsert aggregate pattern stats.

        Returns the updated :class:`FailurePattern`.
        """
        assert self._conn is not None
        now = time.time()

        row = self._conn.execute(
            "SELECT * FROM failure_patterns WHERE error_hash = ?",
            (error_hash,),
        ).fetchone()

        if row is None:
            self._conn.execute(
                """
                INSERT INTO failure_patterns
                    (error_hash, error_class, occurrence_count,
                     first_seen, last_seen, success_count, failure_count)
                VALUES (?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    error_hash,
                    error_class,
                    now,
                    now,
                    int(success),
                    0 if success else 1,
                ),
            )
        else:
            self._conn.execute(
                """
                UPDATE failure_patterns
                SET occurrence_count = occurrence_count + 1,
                    last_seen = ?,
                    success_count = success_count + ?,
                    failure_count = failure_count + ?
                WHERE error_hash = ?
                """,
                (now, int(success), 0 if success else 1, error_hash),
            )

        self._conn.commit()
        return self.get_pattern(error_hash)  # type: ignore[return-value]

    # -- reads ---------------------------------------------------------------

    def get_pattern(self, error_hash: str) -> Optional[FailurePattern]:
        """Return the pattern for *error_hash*, or ``None``."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM failure_patterns WHERE error_hash = ?",
            (error_hash,),
        ).fetchone()
        if row is None:
            return None
        return FailurePattern(
            error_hash=row[0],
            error_class=row[1],
            occurrence_count=row[2],
            first_seen=row[3],
            last_seen=row[4],
            success_count=row[5],
            failure_count=row[6],
        )

    def get_playbook(self, error_hash: str) -> Optional[str]:
        """Return the remediation playbook text, or ``None``."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT playbook FROM remediation_playbooks WHERE error_hash = ?",
            (error_hash,),
        ).fetchone()
        return row[0] if row else None

    def save_playbook(self, error_hash: str, playbook: str) -> None:
        """Save (upsert) a remediation playbook."""
        assert self._conn is not None
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO remediation_playbooks (error_hash, playbook, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(error_hash)
            DO UPDATE SET playbook = excluded.playbook, updated_at = excluded.updated_at
            """,
            (error_hash, playbook, now, now),
        )
        self._conn.commit()

    def detect_regression(self, error_hash: str) -> bool:
        """Return ``True`` if this error was previously fixed but has reappeared.

        A "fixed" error is one where the most recent *N* records were all
        successes, followed by a new failure.  We approximate this by
        checking whether the pattern had at least one success and the
        latest log entry is a failure.
        """
        assert self._conn is not None
        pattern = self.get_pattern(error_hash)
        if pattern is None or pattern.success_count == 0:
            return False

        # Check the latest log entry for this hash
        row = self._conn.execute(
            """
            SELECT success FROM failure_logs
            WHERE error_hash = ?
            ORDER BY timestamp DESC LIMIT 1
            """,
            (error_hash,),
        ).fetchone()

        if row is None:
            return False

        latest_success = bool(row[0])
        if latest_success:
            return False

        # There was at least one prior success and the latest is a failure
        # Check if there was a streak of successes before this failure
        rows = self._conn.execute(
            """
            SELECT success FROM failure_logs
            WHERE error_hash = ?
            ORDER BY timestamp DESC LIMIT 5
            """,
            (error_hash,),
        ).fetchall()

        if len(rows) < 2:
            return False

        # Latest is failure (rows[0]), check if the one before was a success
        return bool(rows[1][0])

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        assert self._conn is not None

        total_failures = self._conn.execute(
            "SELECT COUNT(*) FROM failure_logs"
        ).fetchone()[0]
        total_patterns = self._conn.execute(
            "SELECT COUNT(*) FROM failure_patterns"
        ).fetchone()[0]
        total_playbooks = self._conn.execute(
            "SELECT COUNT(*) FROM remediation_playbooks"
        ).fetchone()[0]
        total_signals = self._conn.execute(
            "SELECT COUNT(*) FROM predictive_signals"
        ).fetchone()[0]

        # Success rate
        successes = self._conn.execute(
            "SELECT COUNT(*) FROM failure_logs WHERE success = 1"
        ).fetchone()[0]
        success_rate = (successes / total_failures) if total_failures > 0 else 0.0

        # Top error classes
        top_classes = self._conn.execute(
            """
            SELECT error_class, SUM(occurrence_count) AS total
            FROM failure_patterns
            GROUP BY error_class
            ORDER BY total DESC
            LIMIT 5
            """
        ).fetchall()

        return {
            "total_failure_logs": total_failures,
            "total_patterns": total_patterns,
            "total_playbooks": total_playbooks,
            "total_signals": total_signals,
            "success_rate": round(success_rate, 4),
            "top_error_classes": [
                {"error_class": row[0], "count": row[1]} for row in top_classes
            ],
        }


# ---------------------------------------------------------------------------
# RemediationEngine
# ---------------------------------------------------------------------------

_PLAYBOOK_TEMPLATES: dict[str, str] = {
    "import_error": (
        "## Import Error Remediation\n"
        "1. Verify the module is installed: `pip install <module>`\n"
        "2. Check virtual environment is activated\n"
        "3. Verify PYTHONPATH includes the module's parent directory\n"
        "4. Check for circular imports\n"
        "5. Review requirements.txt / pyproject.toml for missing dependency"
    ),
    "connection_error": (
        "## Connection Error Remediation\n"
        "1. Verify the target service is running and reachable\n"
        "2. Check firewall / security group rules\n"
        "3. Verify DNS resolution\n"
        "4. Check for port conflicts\n"
        "5. Increase connection timeout if transient\n"
        "6. Implement exponential back-off retry"
    ),
    "permission_error": (
        "## Permission Error Remediation\n"
        "1. Check file / directory permissions: `ls -la <path>`\n"
        "2. Verify the process user has required access\n"
        "3. Check SELinux / AppArmor policies\n"
        "4. Review container or sandbox restrictions\n"
        "5. Use `chmod` / `chown` to fix ownership"
    ),
    "type_error": (
        "## Type / Attribute Error Remediation\n"
        "1. Review the function signature and expected argument types\n"
        "2. Check for None values being passed unexpectedly\n"
        "3. Verify API version compatibility\n"
        "4. Add type guards or isinstance checks\n"
        "5. Run mypy / pyright for static analysis"
    ),
    "timeout_error": (
        "## Timeout Error Remediation\n"
        "1. Increase timeout threshold if the operation is legitimately slow\n"
        "2. Check network latency and bandwidth\n"
        "3. Profile the slow operation\n"
        "4. Add caching to reduce repeated slow calls\n"
        "5. Implement circuit breaker to fail fast"
    ),
    "resource_error": (
        "## Resource Error Remediation\n"
        "1. Check available disk space: `df -h`\n"
        "2. Check available memory: `free -m`\n"
        "3. Close unused file handles\n"
        "4. Increase resource limits (ulimit)\n"
        "5. Implement resource pooling / recycling"
    ),
    "unknown": (
        "## General Error Remediation\n"
        "1. Review the full stack trace for root cause\n"
        "2. Check recent code changes that may have introduced the error\n"
        "3. Search issue tracker for known issues\n"
        "4. Add more granular error handling\n"
        "5. Enable debug logging for more context"
    ),
}


class RemediationEngine:
    """Retrieves or generates remediation playbooks based on failure patterns."""

    def __init__(self, db: HealingDB) -> None:
        self._db = db

    def suggest_recovery(self, error_hash: str) -> dict[str, Any]:
        """Return a recovery suggestion for the given error pattern.

        Returns
        -------
        dict with keys:
            - ``has_playbook`` (bool)
            - ``playbook`` (str | None)
            - ``suggested_level`` (int) — suggested recovery cascade level
            - ``is_regression`` (bool)
            - ``pattern`` (dict | None)
        """
        pattern = self._db.get_pattern(error_hash)
        playbook = self._db.get_playbook(error_hash)
        is_regression = self._db.detect_regression(error_hash)

        # Suggest recovery level based on history
        if pattern is None:
            suggested_level = 1
        elif pattern.failure_count > pattern.success_count * 2:
            # Mostly failing — escalate
            suggested_level = min(4, 2 + pattern.failure_count // 5)
        elif is_regression:
            # Regression — bump level
            suggested_level = 2
        else:
            suggested_level = 1

        return {
            "has_playbook": playbook is not None,
            "playbook": playbook,
            "suggested_level": suggested_level,
            "is_regression": is_regression,
            "pattern": {
                "error_hash": pattern.error_hash,
                "error_class": pattern.error_class,
                "occurrence_count": pattern.occurrence_count,
                "success_count": pattern.success_count,
                "failure_count": pattern.failure_count,
            } if pattern else None,
        }

    def _generate_playbook(self, pattern: FailurePattern) -> str:
        """Generate a text playbook for the given pattern."""
        template = _PLAYBOOK_TEMPLATES.get(
            pattern.error_class, _PLAYBOOK_TEMPLATES["unknown"]
        )

        header = (
            f"# Remediation Playbook — {pattern.error_class}\n"
            f"**Pattern hash:** `{pattern.error_hash}`\n"
            f"**Occurrences:** {pattern.occurrence_count}\n"
            f"**Success rate:** "
            f"{pattern.success_count}/{pattern.occurrence_count}\n\n"
        )

        return header + template


# ---------------------------------------------------------------------------
# SelfHealingIntelligenceLoop
# ---------------------------------------------------------------------------

class SelfHealingIntelligenceLoop:
    """Reactive loop that monitors recovery events.

    Subscribes to ``recovery.started``, ``recovery.succeeded``, and
    ``recovery.failed`` event types on the :class:`EventBus`.  For each
    event it:

    1. Logs the failure record.
    2. Updates the aggregate pattern.
    3. Checks whether a playbook should be generated (>= 3 occurrences).
    4. Detects regressions.
    """

    def __init__(
        self,
        bus: EventBus,
        db: Optional[HealingDB] = None,
    ) -> None:
        self._bus = bus
        self._db = db or HealingDB()
        self._engine = RemediationEngine(self._db)
        self._subscription_ids: list[str] = []
        self._running = False
        self._events_processed = 0
        self._playbooks_generated = 0
        self._regressions_detected = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to recovery events on the bus."""
        if self._running:
            return

        self._running = True

        event_types = {"recovery.started", "recovery.succeeded", "recovery.failed"}
        sub_id = self._bus.subscribe(
            handler=self._on_recovery_event,
            event_types=event_types,
        )
        self._subscription_ids.append(sub_id)

        logger.info("SelfHealingIntelligenceLoop started (subscriptions=%d)", len(self._subscription_ids))

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self._running = False
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("SelfHealingIntelligenceLoop stopped")

    # -- event handling ------------------------------------------------------

    async def _on_recovery_event(self, event: Event) -> None:
        """Handle a single recovery event."""
        try:
            payload = event.payload or {}
            error_msg = payload.get("error", payload.get("message", ""))
            component = payload.get("component", event.source or "unknown")
            level = payload.get("level", 0)
            action = payload.get("action", "")

            is_success = event.event_type == "recovery.succeeded"

            # Classify the error
            error_hash, error_class = classify_error(error_msg) if error_msg else ("no_error", "unknown")

            # Build record
            record = FailureRecord(
                id=event.event_id,
                component=component,
                error_class=error_class,
                error_hash=error_hash,
                error_message=error_msg[:2000],  # truncate
                recovery_level=level,
                recovery_action=action,
                success=is_success,
                timestamp=event.timestamp.timestamp() if hasattr(event.timestamp, "timestamp") else time.time(),
            )

            # Persist
            self._db.log_failure(record)
            pattern = self._db.update_pattern(error_hash, error_class, is_success)

            self._events_processed += 1

            # Check for playbook generation
            if (
                pattern is not None
                and pattern.occurrence_count >= _PLAYBOOK_THRESHOLD
                and self._db.get_playbook(error_hash) is None
            ):
                playbook = self._engine._generate_playbook(pattern)
                self._db.save_playbook(error_hash, playbook)
                self._playbooks_generated += 1
                logger.info(
                    "Generated remediation playbook for %s (hash=%s, occurrences=%d)",
                    error_class,
                    error_hash,
                    pattern.occurrence_count,
                )

            # Check for regression
            if self._db.detect_regression(error_hash):
                self._regressions_detected += 1
                logger.warning(
                    "Regression detected for %s (hash=%s)",
                    error_class,
                    error_hash,
                )

        except Exception:
            logger.exception("Error processing recovery event %s", event.event_id)

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current loop status and statistics."""
        db_stats = self._db.get_stats()
        return {
            "running": self._running,
            "subscriptions": len(self._subscription_ids),
            "events_processed": self._events_processed,
            "playbooks_generated": self._playbooks_generated,
            "regressions_detected": self._regressions_detected,
            "db": db_stats,
        }
