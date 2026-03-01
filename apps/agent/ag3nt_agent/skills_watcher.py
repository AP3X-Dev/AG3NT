"""Skills hot-reload via file system watching.

Monitors skills directories for changes (new skills, modified SKILL.md files)
and invalidates cached skill metadata to enable hot-reload without restart.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class SkillsWatcher:
    """Watches skills directories and notifies on changes to SKILL.md files."""

    def __init__(self, watch_dirs: list[Path] | None = None) -> None:
        self._watch_dirs = [d for d in (watch_dirs or self._default_dirs()) if d.exists()]
        self._callbacks: list[Callable[[], None]] = []

    @staticmethod
    def _default_dirs() -> list[Path]:
        """Default skills directories to watch."""
        home = Path.home()
        cwd = Path.cwd()
        return [
            cwd / "skills",
            home / ".ag3nt" / "skills",
            cwd / ".ag3nt" / "skills",
        ]

    @property
    def watch_dirs(self) -> list[Path]:
        return list(self._watch_dirs)

    def on_change(self, callback: Callable[[], None]) -> None:
        """Register a callback for skill file changes."""
        self._callbacks.append(callback)

    def _handle_change(self, file_path: str) -> None:
        """Handle a file system change event. Called by file watcher."""
        path = Path(file_path)
        # Only react to SKILL.md files
        if path.name != "SKILL.md":
            return
        logger.info("[SkillsWatcher] Skill changed: %s", path.parent.name)
        for cb in self._callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning("[SkillsWatcher] Callback error: %s", e)

    def start(self) -> None:
        """Start watching (connects to existing file watcher infrastructure).

        TODO: Connect to watchdog Observer or ProjectFileWatcher to get real
        filesystem events. Currently requires manual _handle_change() calls.
        """
        logger.info(
            "[SkillsWatcher] Watching %d directories for skill changes",
            len(self._watch_dirs),
        )

    def stop(self) -> None:
        """Stop watching."""
        logger.info("[SkillsWatcher] Stopped")
