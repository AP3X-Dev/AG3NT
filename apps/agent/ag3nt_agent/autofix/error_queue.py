"""Persistent error queue for the AutoFix pipeline.

Stores queued errors at ``~/.ag3nt/autofix_queue.json`` with priority
ordering, deduplication within a configurable window, and a per-error
requeue limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".ag3nt" / "autofix_queue.json"
_MAX_ENTRIES = 100
_MAX_REQUEUES = 3
_DEDUP_WINDOW_S = 300  # 5 minutes

# Severity -> numeric priority (lower = higher priority).
_PRIORITY_MAP: dict[str, int] = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


def _error_hash(entry: dict[str, Any]) -> str:
    """Compute a stable hash for an error entry (for dedup)."""
    msg = entry.get("message", entry.get("error", str(entry)))
    cat = entry.get("category", "")
    raw = f"{cat}:{msg}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ErrorQueue:
    """Priority-based persistent error queue.

    Parameters
    ----------
    path:
        Location of the JSON backing store.  Defaults to
        ``~/.ag3nt/autofix_queue.json``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH
        self._entries: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, error_entry: dict[str, Any]) -> bool:
        """Add an error to the queue.

        Returns *True* if accepted, *False* if deduplicated or full.
        Entries are automatically prioritised by severity.
        """
        h = _error_hash(error_entry)
        now = time.time()

        # Dedup: reject if an identical error was enqueued recently.
        for existing in self._entries:
            if existing.get("_hash") == h:
                enqueued_at = existing.get("_enqueued_at", 0)
                if now - enqueued_at < _DEDUP_WINDOW_S:
                    logger.debug("ErrorQueue: deduplicated error %s", h)
                    return False

        # Requeue limit.
        requeue_count = error_entry.get("_requeue_count", 0)
        if requeue_count >= _MAX_REQUEUES:
            logger.info("ErrorQueue: dropping error %s (requeue limit reached)", h)
            return False

        entry = {
            **error_entry,
            "_hash": h,
            "_enqueued_at": now,
            "_priority": _PRIORITY_MAP.get(
                error_entry.get("severity", "low"), 2
            ),
            "_requeue_count": requeue_count,
        }
        self._entries.append(entry)

        # Sort by priority (lower = more urgent).
        self._entries.sort(key=lambda e: e.get("_priority", 2))

        # Trim to max size.
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[:_MAX_ENTRIES]

        self._save()
        return True

    def dequeue(self, max_count: int = 10) -> list[dict[str, Any]]:
        """Remove and return the highest-priority entries."""
        batch = self._entries[:max_count]
        self._entries = self._entries[max_count:]
        self._save()
        return batch

    def size(self) -> int:
        """Return the number of queued entries."""
        return len(self._entries)

    def peek(self, max_count: int = 10) -> list[dict[str, Any]]:
        """Return the highest-priority entries without removing them."""
        return list(self._entries[:max_count])

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the queue from disk."""
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._entries = data if isinstance(data, list) else []
                logger.debug("ErrorQueue: loaded %d entries from %s", len(self._entries), self._path)
            except Exception as exc:
                logger.warning("ErrorQueue: failed to load %s: %s", self._path, exc)
                self._entries = []
        else:
            self._entries = []

    def _save(self) -> None:
        """Persist the queue to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._entries, default=str, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("ErrorQueue: failed to save %s: %s", self._path, exc)
