"""
QualityGuardianLoop -- Reactive File Health Scoring.

Monitors file change events on the EventBus, runs heuristic code-quality
checks, and maintains per-file health scores in a WAL-mode SQLite database
at ``~/.ag3nt/quality.db``.

Architecture
------------
EventBus --> QualityGuardianLoop --> QualityDB (SQLite WAL)
                                --> emits quality.health_updated
                                --> emits quality.degradation_alert (score < 60)
"""

from __future__ import annotations

import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FileHealth:
    """Health record for a single file."""

    path: str
    score: float = 100.0
    raw_errors: int = 0
    raw_warnings: int = 0
    raw_hints: int = 0
    type_score: float = 100.0
    test_coverage: float = 0.0
    last_checked: str = ""
    issue_count: int = 0

    def __post_init__(self) -> None:
        if not self.last_checked:
            self.last_checked = datetime.now(timezone.utc).isoformat()


@dataclass
class Issue:
    """A single quality issue found in a file."""

    path: str
    line: int
    severity: str  # "error" | "warning" | "hint"
    source: str  # "lint" | "type" | "test"
    message: str
    first_seen: str = ""
    resolved_at: str | None = None

    def __post_init__(self) -> None:
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# QualityDB -- WAL-mode SQLite
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = os.path.join(os.path.expanduser("~"), ".ag3nt", "quality.db")


class QualityDB:
    """Persistent store for file health scores and issues.

    Uses SQLite in WAL mode at ``~/.ag3nt/quality.db``.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    # -- lifecycle -----------------------------------------------------------

    def init_db(self) -> None:
        """Create tables and enable WAL mode."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row

        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS file_health (
                path          TEXT PRIMARY KEY,
                score         REAL NOT NULL DEFAULT 100.0,
                raw_errors    INTEGER NOT NULL DEFAULT 0,
                raw_warnings  INTEGER NOT NULL DEFAULT 0,
                raw_hints     INTEGER NOT NULL DEFAULT 0,
                type_score    REAL NOT NULL DEFAULT 100.0,
                test_coverage REAL NOT NULL DEFAULT 0.0,
                last_checked  TEXT NOT NULL,
                issue_count   INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS issues (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                path        TEXT NOT NULL,
                line        INTEGER NOT NULL,
                severity    TEXT NOT NULL,
                source      TEXT NOT NULL,
                message     TEXT NOT NULL,
                first_seen  TEXT NOT NULL,
                resolved_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_issues_path
                ON issues(path);
            CREATE INDEX IF NOT EXISTS idx_issues_severity
                ON issues(severity);

            CREATE TABLE IF NOT EXISTS history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                path       TEXT NOT NULL,
                score      REAL NOT NULL,
                timestamp  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_history_path
                ON history(path);
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- file_health ---------------------------------------------------------

    def upsert_health(self, health: FileHealth) -> None:
        """Insert or update a file health record."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO file_health
                (path, score, raw_errors, raw_warnings, raw_hints,
                 type_score, test_coverage, last_checked, issue_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                score=excluded.score,
                raw_errors=excluded.raw_errors,
                raw_warnings=excluded.raw_warnings,
                raw_hints=excluded.raw_hints,
                type_score=excluded.type_score,
                test_coverage=excluded.test_coverage,
                last_checked=excluded.last_checked,
                issue_count=excluded.issue_count
            """,
            (
                health.path,
                health.score,
                health.raw_errors,
                health.raw_warnings,
                health.raw_hints,
                health.type_score,
                health.test_coverage,
                health.last_checked,
                health.issue_count,
            ),
        )
        # Record history entry
        self._conn.execute(
            "INSERT INTO history (path, score, timestamp) VALUES (?, ?, ?)",
            (health.path, health.score, health.last_checked),
        )
        self._conn.commit()

    def get_health(self, path: str) -> FileHealth | None:
        """Return the FileHealth for *path*, or None."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM file_health WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None
        return FileHealth(
            path=row["path"],
            score=row["score"],
            raw_errors=row["raw_errors"],
            raw_warnings=row["raw_warnings"],
            raw_hints=row["raw_hints"],
            type_score=row["type_score"],
            test_coverage=row["test_coverage"],
            last_checked=row["last_checked"],
            issue_count=row["issue_count"],
        )

    def get_all_health(self) -> list[FileHealth]:
        """Return all file health records."""
        assert self._conn is not None
        rows = self._conn.execute("SELECT * FROM file_health").fetchall()
        return [
            FileHealth(
                path=r["path"],
                score=r["score"],
                raw_errors=r["raw_errors"],
                raw_warnings=r["raw_warnings"],
                raw_hints=r["raw_hints"],
                type_score=r["type_score"],
                test_coverage=r["test_coverage"],
                last_checked=r["last_checked"],
                issue_count=r["issue_count"],
            )
            for r in rows
        ]

    # -- issues --------------------------------------------------------------

    def add_issue(self, issue: Issue) -> None:
        """Insert a new issue record."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO issues (path, line, severity, source, message, first_seen, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                issue.path,
                issue.line,
                issue.severity,
                issue.source,
                issue.message,
                issue.first_seen,
                issue.resolved_at,
            ),
        )
        self._conn.commit()

    def resolve_issue(self, path: str, line: int, message: str) -> None:
        """Mark an issue as resolved."""
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            UPDATE issues SET resolved_at = ?
            WHERE path = ? AND line = ? AND message = ? AND resolved_at IS NULL
            """,
            (now, path, line, message),
        )
        self._conn.commit()

    # -- stats ---------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate quality statistics."""
        assert self._conn is not None

        total_files = self._conn.execute(
            "SELECT COUNT(*) FROM file_health"
        ).fetchone()[0]

        avg_score_row = self._conn.execute(
            "SELECT AVG(score) FROM file_health"
        ).fetchone()
        avg_score = avg_score_row[0] if avg_score_row[0] is not None else 0.0

        total_issues = self._conn.execute(
            "SELECT COUNT(*) FROM issues WHERE resolved_at IS NULL"
        ).fetchone()[0]

        total_resolved = self._conn.execute(
            "SELECT COUNT(*) FROM issues WHERE resolved_at IS NOT NULL"
        ).fetchone()[0]

        # Issues by severity
        severity_rows = self._conn.execute(
            """
            SELECT severity, COUNT(*) as cnt
            FROM issues WHERE resolved_at IS NULL
            GROUP BY severity
            """
        ).fetchall()
        by_severity = {r["severity"]: r["cnt"] for r in severity_rows}

        # Lowest-scoring files
        worst_rows = self._conn.execute(
            "SELECT path, score FROM file_health ORDER BY score ASC LIMIT 5"
        ).fetchall()
        worst_files = [{"path": r["path"], "score": r["score"]} for r in worst_rows]

        return {
            "total_files": total_files,
            "average_score": round(avg_score, 2),
            "open_issues": total_issues,
            "resolved_issues": total_resolved,
            "issues_by_severity": by_severity,
            "worst_files": worst_files,
        }


# ---------------------------------------------------------------------------
# Heuristic check patterns
# ---------------------------------------------------------------------------

_TODO_RE = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
_BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:", re.MULTILINE)
_NESTING_INDENT_CHARS = 4  # spaces per indent level


def _count_nesting_depth(line: str) -> int:
    """Return the indentation-based nesting depth of a line."""
    stripped = line.lstrip(" ")
    if not stripped or stripped.startswith("#"):
        return 0
    indent = len(line) - len(stripped)
    return indent // _NESTING_INDENT_CHARS


# ---------------------------------------------------------------------------
# QualityGuardianLoop -- reactive loop
# ---------------------------------------------------------------------------


class QualityGuardianLoop:
    """Reactive loop that monitors file events and maintains health scores.

    Subscribes to file change events and ``sentinel.indexed`` on the
    :class:`EventBus`.  For each file, runs heuristic checks and updates
    the :class:`QualityDB`.

    Emits:
    - ``quality.health_updated`` after every check
    - ``quality.degradation_alert`` when score drops below 60
    """

    def __init__(
        self,
        bus: EventBus,
        working_dir: str = ".",
        db: QualityDB | None = None,
    ) -> None:
        self._bus = bus
        self._working_dir = os.path.abspath(working_dir)
        self._db = db or QualityDB()

        self._subscription_ids: list[str] = []
        self._running = False

        # Stats
        self._files_checked = 0
        self._alerts_emitted = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to file events and sentinel.indexed."""
        if self._running:
            return

        self._running = True

        # Subscribe to file change events
        file_sub = self._bus.subscribe(
            handler=self._on_file_event,
            event_types={"file.written", "file.edited", "file.external_modified"},
        )
        self._subscription_ids.append(file_sub)

        # Subscribe to sentinel.indexed events
        sentinel_sub = self._bus.subscribe(
            handler=self._on_file_event,
            event_types={"sentinel.indexed"},
        )
        self._subscription_ids.append(sentinel_sub)

        logger.info(
            "QualityGuardianLoop started (working_dir=%s, subscriptions=%d)",
            self._working_dir,
            len(self._subscription_ids),
        )

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self._running = False
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("QualityGuardianLoop stopped")

    # -- event handler -------------------------------------------------------

    async def _on_file_event(self, event: Event) -> None:
        """Handle a file event by checking the file."""
        path = event.payload.get("path") or event.payload.get("file")
        if path is None:
            return

        try:
            await self._check_file(path)
        except Exception:
            logger.exception("Error checking file %s", path)

    # -- file checking -------------------------------------------------------

    async def _check_file(self, path: str) -> None:
        """Run heuristic checks on a file and update the database."""
        abs_path = os.path.abspath(path)

        if not os.path.isfile(abs_path):
            logger.debug("Skipping non-existent file: %s", abs_path)
            return

        # Only check text source files
        _, ext = os.path.splitext(abs_path)
        if ext not in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs"}:
            return

        try:
            content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", abs_path, exc)
            return

        errors, warnings, hints = self._heuristic_checks(abs_path, content)
        score = self._calculate_score(errors, warnings, hints)

        now = datetime.now(timezone.utc).isoformat()
        health = FileHealth(
            path=abs_path,
            score=score,
            raw_errors=errors,
            raw_warnings=warnings,
            raw_hints=hints,
            last_checked=now,
            issue_count=errors + warnings + hints,
        )

        # Get previous health for comparison
        prev_health = self._db.get_health(abs_path)

        self._db.upsert_health(health)
        self._files_checked += 1

        # Emit health_updated event
        await self._bus.publish(Event(
            event_type="quality.health_updated",
            source="quality_guardian",
            payload={
                "path": abs_path,
                "score": score,
                "errors": errors,
                "warnings": warnings,
                "hints": hints,
                "previous_score": prev_health.score if prev_health else None,
            },
        ))

        # Emit degradation alert if score < 60
        if score < 60:
            self._alerts_emitted += 1
            await self._bus.publish(Event(
                event_type="quality.degradation_alert",
                source="quality_guardian",
                priority=EventPriority.HIGH,
                payload={
                    "path": abs_path,
                    "score": score,
                    "errors": errors,
                    "warnings": warnings,
                    "hints": hints,
                },
            ))
            logger.warning(
                "Quality degradation alert: %s score=%.1f (errors=%d, warnings=%d, hints=%d)",
                abs_path, score, errors, warnings, hints,
            )

    # -- heuristic checks ----------------------------------------------------

    def _heuristic_checks(self, path: str, content: str) -> tuple[int, int, int]:
        """Run heuristic code-quality checks on file content.

        Returns
        -------
        (errors, warnings, hints)
        """
        errors = 0
        warnings = 0
        hints = 0

        lines = content.splitlines()

        # File-level checks
        # Files > 500 lines -> warning
        if len(lines) > 500:
            warnings += 1

        # Track function lengths
        func_start: int | None = None
        func_indent: int = 0

        for i, line in enumerate(lines, start=1):
            # TODO/FIXME/HACK/XXX -> hint
            if _TODO_RE.search(line):
                hints += 1

            # Bare except -> error
            if _BARE_EXCEPT_RE.match(line):
                errors += 1

            # Lines > 120 chars -> hint
            if len(line) > 120:
                hints += 1

            # Nesting > 4 levels deep -> warning
            depth = _count_nesting_depth(line)
            if depth > 4 and line.strip():
                warnings += 1

            # Track function definitions and their length
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if stripped.startswith("def ") or stripped.startswith("async def "):
                # If we were tracking a previous function, check its length
                if func_start is not None:
                    func_len = i - 1 - func_start
                    if func_len > 50:
                        warnings += 1
                func_start = i
                func_indent = indent
            elif func_start is not None:
                # Check if we've exited the function (same or lower indent with content)
                if stripped and not stripped.startswith("#") and indent <= func_indent and not stripped.startswith("@"):
                    func_len = i - 1 - func_start
                    if func_len > 50:
                        warnings += 1
                    func_start = None

        # Check last function if file ended while tracking one
        if func_start is not None:
            func_len = len(lines) - func_start
            if func_len > 50:
                warnings += 1

        return errors, warnings, hints

    # -- scoring -------------------------------------------------------------

    @staticmethod
    def _calculate_score(errors: int, warnings: int, hints: int) -> float:
        """Calculate a health score from issue counts.

        Uses exponential decay: ``100 * exp(-errors*0.1 - warnings*0.03 - hints*0.01)``
        """
        return 100.0 * math.exp(-errors * 0.1 - warnings * 0.03 - hints * 0.01)

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return current loop status and statistics."""
        db_stats = self._db.get_stats()
        return {
            "running": self._running,
            "working_dir": self._working_dir,
            "files_checked": self._files_checked,
            "alerts_emitted": self._alerts_emitted,
            "subscriptions": len(self._subscription_ids),
            "db": db_stats,
        }
