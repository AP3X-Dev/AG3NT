"""
ProjectFileWatcher -- Enhanced File Watcher with Grace Period

Higher-level file watcher that sits above the basic autonomous FileWatcher
and provides:

1. **Agent-write grace period** (2 000 ms suppression after agent writes)
   so that files the agent itself creates/edits are not reported as
   "external" changes.

2. **Event coalescing** (200 ms batch delay) to avoid flooding downstream
   consumers with many rapid events from a single save-and-format cycle.

3. **Typed external events** emitted on the EventBus:
   - ``file.external_created``
   - ``file.external_modified``
   - ``file.external_deleted``

Downstream consumers (QualityGateLoop, SentinelLoop, etc.) subscribe to
these events to react only to changes made *outside* the agent.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


class ChangeType(str, Enum):
    """Kind of filesystem change detected."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class FileChangeEvent:
    """
    Lightweight description of a single file change.

    Attributes:
        path:        Absolute (or working-dir-relative) path to the file.
        change_type: One of created / modified / deleted.
        detected_at: ``time.time()`` when the change was first noticed.
        source:      ``"external"`` for user edits; ``"agent"`` when the
                     agent wrote the file (these are normally suppressed).
    """
    path: str
    change_type: ChangeType
    detected_at: float
    source: str = "external"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE_PATTERNS: list[str] = [
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "*.pyc",
    ".ag3nt",
    ".claude",
]


@dataclass
class _FileState:
    """Snapshot of one file's metadata used for change detection."""
    mtime: float
    size: int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ProjectFileWatcher:
    """
    Enhanced file watcher with agent-write grace period and event batching.

    Usage::

        bus = EventBus()
        watcher = ProjectFileWatcher(bus, "/path/to/project")
        await watcher.start()

        # When the agent writes a file, call this so the change is suppressed:
        watcher.notify_agent_write("/path/to/project/foo.py")

        # Later:
        await watcher.stop()
    """

    # Class-level constants (ms converted to seconds internally)
    GRACE_PERIOD_MS: int = 2000
    BATCH_DELAY_MS: int = 200

    def __init__(
        self,
        bus: EventBus,
        working_dir: str,
        poll_interval: float = 1.0,
        ignore_patterns: Optional[list[str]] = None,
    ) -> None:
        """
        Args:
            bus:             The EventBus used to publish file-change events.
            working_dir:     Root directory to watch (recursively).
            poll_interval:   Seconds between filesystem polls (default 1 s).
            ignore_patterns: Extra glob patterns to ignore on top of the
                             built-in defaults (.git, __pycache__, ...).
        """
        self._bus = bus
        self._working_dir = os.path.abspath(working_dir)
        self._poll_interval = poll_interval
        self._ignore_patterns: list[str] = list(_DEFAULT_IGNORE_PATTERNS)
        if ignore_patterns:
            self._ignore_patterns.extend(ignore_patterns)

        # Snapshot: path -> _FileState
        self._snapshot: dict[str, _FileState] = {}

        # Grace period tracking: normalised path -> timestamp of agent write
        self._grace: dict[str, float] = {}

        # Pending batch: path -> FileChangeEvent  (last event wins per path)
        self._batch: dict[str, FileChangeEvent] = {}

        # Async control
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._batch_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Grace period
    # ------------------------------------------------------------------

    def notify_agent_write(self, path: str) -> None:
        """
        Record that the *agent* just wrote ``path``.

        For the next :pyattr:`GRACE_PERIOD_MS` milliseconds any detected
        change on this path will be silently ignored instead of being
        emitted as an external event.
        """
        normalised = os.path.normpath(os.path.abspath(path))
        self._grace[normalised] = time.time()
        logger.debug("Grace period set for %s", normalised)

    def _in_grace_period(self, path: str) -> bool:
        """Return ``True`` if *path* is still within the agent-write grace window."""
        normalised = os.path.normpath(os.path.abspath(path))
        write_time = self._grace.get(normalised)
        if write_time is None:
            return False
        elapsed_ms = (time.time() - write_time) * 1000.0
        if elapsed_ms < self.GRACE_PERIOD_MS:
            return True
        # Expired -- clean up eagerly
        del self._grace[normalised]
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Take the initial snapshot and start the poll + batch loops."""
        if self._running:
            logger.warning("ProjectFileWatcher is already running")
            return

        self._running = True
        self._take_snapshot()
        self._poll_task = asyncio.create_task(self._poll_loop())
        self._batch_task = asyncio.create_task(self._batch_loop())
        logger.info(
            "ProjectFileWatcher started — watching %s (%d files tracked)",
            self._working_dir,
            len(self._snapshot),
        )

    async def stop(self) -> None:
        """Stop the poll/batch loops and flush any remaining events."""
        if not self._running:
            return

        self._running = False

        for task in (self._poll_task, self._batch_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._poll_task = None
        self._batch_task = None

        # Flush whatever is left in the batch
        await self._flush_batch()
        logger.info("ProjectFileWatcher stopped")

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> None:
        """Walk the working directory and record mtime + size for every file."""
        new_snapshot: dict[str, _FileState] = {}
        for file_path in self._walk():
            try:
                stat = os.stat(file_path)
                new_snapshot[file_path] = _FileState(
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                )
            except OSError:
                # File may have been deleted between walk and stat
                continue
        self._snapshot = new_snapshot

    def _walk(self):
        """Yield absolute file paths under ``_working_dir``, respecting ignore patterns."""
        for dirpath, dirnames, filenames in os.walk(self._working_dir):
            # Prune ignored directories *in place* so os.walk skips them
            dirnames[:] = [
                d for d in dirnames
                if not self._is_ignored(d)
            ]

            for filename in filenames:
                if self._is_ignored(filename):
                    continue
                yield os.path.normpath(os.path.join(dirpath, filename))

    def _is_ignored(self, name: str) -> bool:
        """Check whether a file or directory *name* matches any ignore pattern."""
        for pattern in self._ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Periodically compare the filesystem to the snapshot."""
        while self._running:
            try:
                self._detect_changes()
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in ProjectFileWatcher poll loop")
                await asyncio.sleep(self._poll_interval)

    def _detect_changes(self) -> None:
        """
        Compare the current filesystem state to ``_snapshot``.

        New or changed files that are *not* in the grace period are added
        to the pending batch.  The snapshot is updated in place.
        """
        current: dict[str, _FileState] = {}
        for file_path in self._walk():
            try:
                stat = os.stat(file_path)
                current[file_path] = _FileState(mtime=stat.st_mtime, size=stat.st_size)
            except OSError:
                continue

        now = time.time()

        # --- Created files ---
        for path in current:
            if path not in self._snapshot:
                if not self._in_grace_period(path):
                    self._batch[path] = FileChangeEvent(
                        path=path,
                        change_type=ChangeType.CREATED,
                        detected_at=now,
                    )

        # --- Modified files ---
        for path, state in current.items():
            if path in self._snapshot:
                prev = self._snapshot[path]
                if state.mtime != prev.mtime or state.size != prev.size:
                    if not self._in_grace_period(path):
                        self._batch[path] = FileChangeEvent(
                            path=path,
                            change_type=ChangeType.MODIFIED,
                            detected_at=now,
                        )

        # --- Deleted files ---
        for path in self._snapshot:
            if path not in current:
                if not self._in_grace_period(path):
                    self._batch[path] = FileChangeEvent(
                        path=path,
                        change_type=ChangeType.DELETED,
                        detected_at=now,
                    )

        # Update snapshot to current state
        self._snapshot = current

    # ------------------------------------------------------------------
    # Batching / flushing
    # ------------------------------------------------------------------

    async def _batch_loop(self) -> None:
        """Flush the event batch every :pyattr:`BATCH_DELAY_MS` milliseconds."""
        delay_seconds = self.BATCH_DELAY_MS / 1000.0
        while self._running:
            try:
                await asyncio.sleep(delay_seconds)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in ProjectFileWatcher batch loop")

    async def _flush_batch(self) -> None:
        """
        Emit one EventBus event per pending file change, then clear the batch.

        Event types emitted:
        - ``file.external_created``
        - ``file.external_modified``
        - ``file.external_deleted``
        """
        if not self._batch:
            return

        # Swap out the batch atomically so new changes can accumulate while
        # we are emitting.
        to_emit = self._batch
        self._batch = {}

        for path, change_event in to_emit.items():
            event_type = f"file.external_{change_event.change_type.value}"
            event = Event(
                event_type=event_type,
                source="project_file_watcher",
                payload={
                    "path": change_event.path,
                    "change_type": change_event.change_type.value,
                    "detected_at": change_event.detected_at,
                    "source": change_event.source,
                },
                priority=EventPriority.MEDIUM,
            )
            try:
                await self._bus.publish(event)
                logger.debug("Emitted %s for %s", event_type, path)
            except Exception:
                logger.exception("Failed to publish %s for %s", event_type, path)

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """
        Return a status dict suitable for health dashboards and debugging.

        Keys:
            running            -- whether the watcher loop is active
            watched_dir        -- absolute path being watched
            tracked_files      -- number of files in the current snapshot
            pending_batch      -- number of events waiting to be flushed
            grace_period_entries -- number of paths in the grace-period map
        """
        return {
            "running": self._running,
            "watched_dir": self._working_dir,
            "tracked_files": len(self._snapshot),
            "pending_batch": len(self._batch),
            "grace_period_entries": len(self._grace),
        }
