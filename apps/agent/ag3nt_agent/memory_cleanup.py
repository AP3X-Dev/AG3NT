"""Memory cleanup with TTL-based expiration.

Removes memory files older than a configurable TTL (default 90 days).
Designed to run at agent startup or on a periodic schedule.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("ag3nt.memory_cleanup")

DEFAULT_TTL_DAYS = 90
_MEMORY_DIR = Path.home() / ".ag3nt" / "memory"
_BLUEPRINTS_DIR = Path.home() / ".ag3nt" / "blueprints"


class MemoryCleanup:
    """Removes expired memory entries based on TTL."""

    def __init__(
        self,
        memory_dir: Path | None = None,
        blueprints_dir: Path | None = None,
        ttl_days: int | None = None,
    ):
        self._memory_dir = memory_dir or _MEMORY_DIR
        self._blueprints_dir = blueprints_dir or _BLUEPRINTS_DIR
        self._ttl_days = ttl_days or int(
            os.environ.get("AG3NT_MEMORY_TTL_DAYS", str(DEFAULT_TTL_DAYS))
        )

    @property
    def ttl_days(self) -> int:
        return self._ttl_days

    def cleanup_memory(self) -> int:
        """Remove memory markdown files older than TTL.

        Memory files are expected to have date-based names like YYYY-MM-DD.md
        or YYYY-MM-DD-topic.md.

        Returns:
            Number of files removed.
        """
        if not self._memory_dir.exists():
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._ttl_days)
        removed = 0

        for path in list(self._memory_dir.glob("*.md")):
            file_date = self._parse_date_from_filename(path.stem)
            if file_date is None:
                # Can't determine age — use file mtime as fallback
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    if mtime < cutoff:
                        path.unlink()
                        removed += 1
                        logger.info("Removed expired memory: %s (mtime)", path.name)
                except OSError:
                    pass
                continue

            if file_date < cutoff:
                try:
                    path.unlink()
                    removed += 1
                    logger.info("Removed expired memory: %s", path.name)
                except OSError:
                    logger.debug("Failed to remove %s", path.name, exc_info=True)

        return removed

    def cleanup_blueprints(self) -> int:
        """Remove blueprint JSON files older than TTL (uses file mtime).

        Returns:
            Number of files removed.
        """
        if not self._blueprints_dir.exists():
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(days=self._ttl_days)
        removed = 0

        for path in list(self._blueprints_dir.glob("*.json")):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if mtime < cutoff:
                    path.unlink()
                    removed += 1
                    logger.info("Removed expired blueprint: %s", path.name)
            except OSError:
                logger.debug("Failed to remove %s", path.name, exc_info=True)

        return removed

    def cleanup_all(self) -> dict[str, int]:
        """Run all cleanup tasks.

        Returns:
            Dict with counts per category.
        """
        return {
            "memory": self.cleanup_memory(),
            "blueprints": self.cleanup_blueprints(),
        }

    @staticmethod
    def _parse_date_from_filename(stem: str) -> datetime | None:
        """Extract date from filename like '2026-03-01' or '2026-03-01-topic'."""
        match = re.match(r"(\d{4}-\d{2}-\d{2})", stem)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
