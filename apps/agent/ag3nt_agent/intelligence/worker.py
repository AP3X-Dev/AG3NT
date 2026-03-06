"""IntelligenceWorker — daemon thread for intelligence loops.

Runs the MetaLoop and all registered intelligence loops in a
separate thread with its own asyncio event loop.
"""

import asyncio
import logging
import threading
from typing import Any

logger = logging.getLogger("ag3nt.intelligence.worker")


class IntelligenceWorker:
    """Daemon thread running intelligence loops."""

    def __init__(self, metaloop: Any) -> None:
        self._metaloop = metaloop
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="intelligence-worker"
        )
        self._thread.start()
        logger.info("IntelligenceWorker started")

    def stop(self) -> None:
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("IntelligenceWorker stopped")

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._metaloop.start())
            self._loop.run_forever()
        except Exception as exc:
            logger.error("IntelligenceWorker error: %s", exc)
        finally:
            try:
                self._loop.run_until_complete(self._metaloop.stop())
            except Exception:
                pass
            self._loop.close()

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        return {
            "running": self.is_running,
            "thread_name": self._thread.name if self._thread else None,
        }
