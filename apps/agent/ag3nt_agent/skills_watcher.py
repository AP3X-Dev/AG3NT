"""Skills hot-reload via file system watching.

Monitors skills directories for changes (new skills, modified SKILL.md files)
and invalidates cached skill metadata to enable hot-reload without restart.

Uses the existing FileWatcher singleton for actual filesystem events.
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
        self._registered = False

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

    def _handle_change(self, file_path: str, _event_type: str = "") -> None:
        """Handle a file system change event.

        Called by FileWatcher or manually. Filters for SKILL.md files
        and invokes all registered callbacks.
        """
        path = Path(file_path)
        # Only react to SKILL.md files
        if path.name != "SKILL.md":
            return
        # Only react to files within our watch directories
        if not any(self._is_under(path, d) for d in self._watch_dirs):
            return
        logger.info("[SkillsWatcher] Skill changed: %s", path.parent.name)
        for cb in self._callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning("[SkillsWatcher] Callback error: %s", e)

    @staticmethod
    def _is_under(path: Path, directory: Path) -> bool:
        """Check if path is under directory."""
        try:
            path.resolve().relative_to(directory.resolve())
            return True
        except ValueError:
            return False

    def start(self) -> None:
        """Start watching by registering with the FileWatcher singleton.

        If no FileWatcher is available or no watch directories exist,
        logs a message and remains in manual mode.
        """
        if self._registered or not self._watch_dirs:
            logger.info(
                "[SkillsWatcher] Watching %d directories for skill changes (manual mode)",
                len(self._watch_dirs),
            )
            return

        try:
            from ag3nt_agent.file_watcher import FileWatcher

            fw = FileWatcher.get_instance()
            fw.on_change(self._handle_change)
            self._registered = True
            logger.info(
                "[SkillsWatcher] Registered with FileWatcher for %d skill directories",
                len(self._watch_dirs),
            )
        except Exception:
            logger.debug(
                "[SkillsWatcher] FileWatcher not available, using manual mode",
                exc_info=True,
            )

    def stop(self) -> None:
        """Stop watching by unregistering from the FileWatcher."""
        if self._registered:
            try:
                from ag3nt_agent.file_watcher import FileWatcher

                fw = FileWatcher.get_instance()
                fw.remove_callback(self._handle_change)
                self._registered = False
            except Exception:
                pass
        logger.info("[SkillsWatcher] Stopped")
