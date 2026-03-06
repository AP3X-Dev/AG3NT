"""
AG3NT Daemon -- Process lifecycle and signal management.

Provides:
    GracefulShutdown  -- cooperative shutdown via asyncio.Event
    DaemonManager     -- PID files, signal handling, cross-platform lifecycle
    DaemonStatus      -- snapshot dataclass of daemon state
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Coroutine, Optional

log = logging.getLogger("ag3nt.daemon")


# ---------------------------------------------------------------
#  Data Models
# ---------------------------------------------------------------

@dataclass
class DaemonStatus:
    """Snapshot of daemon process state."""

    pid: int
    running: bool
    uptime_seconds: float
    started_at: Optional[datetime]
    pid_file: str


# ---------------------------------------------------------------
#  Graceful Shutdown
# ---------------------------------------------------------------

class GracefulShutdown:
    """
    Cooperative shutdown primitive backed by an asyncio.Event.

    Use as an async context manager so that the shutdown event is
    automatically set when the block exits (normally or via exception).

    Usage::

        async with GracefulShutdown() as shutdown:
            await shutdown.wait()
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    # -- public API ---------------------------------------------

    def request_shutdown(self) -> None:
        """Signal that the process should shut down."""
        if not self._event.is_set():
            log.info("Shutdown requested")
            self._event.set()

    async def wait(self) -> None:
        """Block until shutdown is requested."""
        await self._event.wait()

    @property
    def is_shutting_down(self) -> bool:
        """True once shutdown has been requested."""
        return self._event.is_set()

    # -- async context manager ----------------------------------

    async def __aenter__(self) -> GracefulShutdown:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.request_shutdown()
        return None


# ---------------------------------------------------------------
#  Daemon Manager
# ---------------------------------------------------------------

class DaemonManager:
    """
    Manages an AG3NT daemon process: PID-file bookkeeping, signal
    handling, start/stop/restart lifecycle.

    All state is persisted via a PID file in *pid_dir* (default
    ``~/.ag3nt``).  Cross-platform: works on POSIX and Windows.
    """

    def __init__(self, pid_dir: Path = Path.home() / ".ag3nt") -> None:
        self._pid_dir = pid_dir
        self._started_at: Optional[datetime] = None
        self._shutdown: Optional[GracefulShutdown] = None

    # -- PID file path ------------------------------------------

    @property
    def _pid_file(self) -> Path:
        """Absolute path to the PID file."""
        return self._pid_dir / "ag3nt.pid"

    # -- process queries ----------------------------------------

    def get_pid(self) -> Optional[int]:
        """Read the PID stored on disk, or ``None`` if unavailable."""
        try:
            text = self._pid_file.read_text(encoding="utf-8").strip()
            return int(text)
        except (FileNotFoundError, ValueError, OSError):
            return None

    def is_running(self) -> bool:
        """Return ``True`` if the PID file exists **and** the process is alive."""
        pid = self.get_pid()
        if pid is None:
            return False
        return self._is_process_alive(pid)

    def status(self) -> DaemonStatus:
        """Build a ``DaemonStatus`` snapshot."""
        pid = self.get_pid()
        running = pid is not None and self._is_process_alive(pid)

        if running and self._started_at is not None:
            uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
        else:
            uptime = 0.0

        return DaemonStatus(
            pid=pid if pid is not None else 0,
            running=running,
            uptime_seconds=uptime,
            started_at=self._started_at,
            pid_file=str(self._pid_file),
        )

    # -- PID file I/O -------------------------------------------

    def write_pid(self) -> None:
        """Write the current process PID to the PID file."""
        self._pid_dir.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(os.getpid()), encoding="utf-8")
        log.debug("Wrote PID %d to %s", os.getpid(), self._pid_file)

    def remove_pid(self) -> None:
        """Delete the PID file if it exists."""
        try:
            self._pid_file.unlink()
            log.debug("Removed PID file %s", self._pid_file)
        except FileNotFoundError:
            pass

    # -- signal handling ----------------------------------------

    def setup_signal_handlers(self, shutdown_callback: Callable) -> None:
        """
        Register *shutdown_callback* for SIGTERM and SIGINT.

        On Windows, SIGTERM may not be available; in that case only
        SIGINT is registered.
        """
        def _handler(signum: int, _frame: object) -> None:
            sig_name = signal.Signals(signum).name
            log.info("Received %s -- initiating shutdown", sig_name)
            shutdown_callback()

        signal.signal(signal.SIGINT, _handler)

        # SIGTERM is not reliably available on Windows
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

        log.debug("Signal handlers installed")

    # -- lifecycle ----------------------------------------------

    async def start(self, runtime_coro: Coroutine) -> None:
        """
        Full daemon start sequence:

        1. Write PID file.
        2. Install signal handlers that trigger graceful shutdown.
        3. Await *runtime_coro*.
        4. Clean up PID file on exit.
        """
        self._shutdown = GracefulShutdown()
        self.write_pid()
        self._started_at = datetime.now(timezone.utc)
        self.setup_signal_handlers(self._shutdown.request_shutdown)

        log.info(
            "Daemon started  pid=%d  pid_file=%s",
            os.getpid(),
            self._pid_file,
        )

        try:
            await runtime_coro
        except asyncio.CancelledError:
            log.info("Runtime coroutine cancelled")
        except Exception:
            log.exception("Runtime coroutine failed")
            raise
        finally:
            self.remove_pid()
            self._started_at = None
            log.info("Daemon stopped")

    def stop(self) -> bool:
        """
        Send SIGTERM (or SIGINT on Windows) to the running daemon.

        Returns ``True`` if the signal was delivered successfully.
        """
        pid = self.get_pid()
        if pid is None:
            log.warning("No PID file found -- daemon may not be running")
            return False

        if not self._is_process_alive(pid):
            log.warning("PID %d is not alive -- cleaning stale PID file", pid)
            self.remove_pid()
            return False

        try:
            # Prefer SIGTERM on POSIX; fall back to SIGINT on Windows
            sig = getattr(signal, "SIGTERM", signal.SIGINT)
            os.kill(pid, sig)
            log.info("Sent %s to PID %d", signal.Signals(sig).name, pid)
            return True
        except OSError as exc:
            log.error("Failed to signal PID %d: %s", pid, exc)
            return False

    async def restart(self, runtime_coro: Coroutine) -> None:
        """Stop the running daemon (if any) then start a new one."""
        if self.is_running():
            self.stop()
            # Give the old process a moment to clean up
            await asyncio.sleep(1.0)
        await self.start(runtime_coro)

    # -- internals ----------------------------------------------

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        """
        Cross-platform check for whether *pid* refers to a running process.

        - POSIX: ``os.kill(pid, 0)`` succeeds or raises ``PermissionError``
          (process exists but is owned by another user).
        - Windows: same ``os.kill(pid, 0)`` convention is supported by
          CPython on Windows.
        """
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            # Process exists but we lack permission -- still alive
            return True
        except OSError:
            return False


__all__ = [
    "DaemonManager",
    "DaemonStatus",
    "GracefulShutdown",
]
