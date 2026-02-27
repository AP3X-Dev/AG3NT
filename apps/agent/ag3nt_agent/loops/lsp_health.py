"""LSPHealthLoop — periodic health monitoring for LSP servers.

Tracks per-language server health, restarts unresponsive servers, and shuts
down servers that have been idle beyond a configurable timeout.

Events consumed:
    lsp.server_started       — payload: {language}
    lsp.server_stopped       — payload: {language}
    lsp.diagnostics_received — payload: {language?, file?}
    lsp.pong                 — payload: {language}

Events emitted:
    lsp.ping              — payload: {language}
    lsp.idle_shutdown     — payload: {language, idle_seconds}
    lsp.restart_requested — payload: {language, attempt}
    lsp.restart_exhausted — payload: {language, attempts}
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.loops.lsp_health")


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class LSPHealthConfig:
    """Configuration for LSPHealthLoop."""

    check_interval: float = 60.0
    max_restart_attempts: int = 3
    idle_shutdown_minutes: int = 30
    ping_timeout: float = 5.0


# ------------------------------------------------------------------
# Language inference
# ------------------------------------------------------------------

_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".lua": "lua",
    ".zig": "zig",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".r": "r",
    ".jl": "julia",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
}


def lang_from_path(path: str) -> str:
    """Infer language from file path extension.

    Returns an empty string if the extension is not recognised.
    """
    dot_pos = path.rfind(".")
    if dot_pos == -1:
        return ""
    ext = path[dot_pos:].lower()
    return _EXT_TO_LANGUAGE.get(ext, "")


# ------------------------------------------------------------------
# LSPHealthLoop
# ------------------------------------------------------------------

class LSPHealthLoop:
    """Background monitoring of LSP (Language Server Protocol) servers.

    Tracks per-language server health, restarts unresponsive servers, and
    shuts down servers that have been idle beyond a configurable timeout.

    Usage::

        bus = EventBus()
        loop = LSPHealthLoop(bus)
        await loop.start()
        # ... later ...
        await loop.stop()
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        config: LSPHealthConfig | None = None,
    ) -> None:
        cfg = config or LSPHealthConfig()
        self._bus = bus
        self._interval = cfg.check_interval
        self._max_restarts = cfg.max_restart_attempts
        self._idle_shutdown = cfg.idle_shutdown_minutes * 60.0
        self._ping_timeout = cfg.ping_timeout

        # language -> {status, last_check, consecutive_failures}
        self._server_health: dict[str, dict[str, Any]] = {}
        self._restart_counts: dict[str, int] = {}
        self._last_activity: dict[str, float] = {}
        self._task: Optional[asyncio.Task[None]] = None
        self._sub_ids: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to LSP events and start the background loop."""
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_lsp_started,
                event_types={"lsp.server_started"},
            )
        )
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_lsp_stopped,
                event_types={"lsp.server_stopped"},
            )
        )
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_lsp_activity,
                event_types={"lsp.diagnostics_received"},
            )
        )
        self._task = asyncio.create_task(
            self._health_loop(), name="lsp_health_loop"
        )
        logger.info(
            "LSPHealthLoop started (interval=%.1fs, idle_shutdown=%dm)",
            self._interval,
            int(self._idle_shutdown / 60),
        )

    async def stop(self) -> None:
        """Cancel the background loop and unsubscribe from all events."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for sub_id in self._sub_ids:
            self._bus.unsubscribe(sub_id)
        self._sub_ids.clear()
        logger.info("LSPHealthLoop stopped")

    # -- event handlers ------------------------------------------------------

    async def _on_lsp_started(self, event: Event) -> None:
        lang = event.payload.get("language", "")
        if lang:
            now = time.monotonic()
            self._server_health[lang] = {
                "status": "healthy",
                "last_check": now,
                "consecutive_failures": 0,
            }
            self._last_activity[lang] = now
            self._restart_counts.setdefault(lang, 0)

    async def _on_lsp_stopped(self, event: Event) -> None:
        lang = event.payload.get("language", "")
        if lang and lang in self._server_health:
            self._server_health[lang]["status"] = "stopped"

    async def _on_lsp_activity(self, event: Event) -> None:
        """Any diagnostics event counts as activity."""
        lang = event.payload.get("language", "")
        if not lang:
            # Try to infer from file extension
            file_path = event.payload.get("file", "")
            lang = lang_from_path(file_path)
        if lang:
            self._last_activity[lang] = time.monotonic()

    # -- background loop -----------------------------------------------------

    async def _health_loop(self) -> None:
        """Periodic health check and idle shutdown for all tracked servers."""
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._check_idle_servers()
                for lang in list(self._server_health.keys()):
                    health = self._server_health[lang]
                    if health["status"] == "stopped":
                        continue  # don't check stopped servers
                    alive = await self._ping_server(lang)
                    health["last_check"] = time.monotonic()
                    if alive:
                        health["consecutive_failures"] = 0
                        health["status"] = "healthy"
                    else:
                        health["consecutive_failures"] = (
                            health.get("consecutive_failures", 0) + 1
                        )
                        if health["consecutive_failures"] >= 2:
                            await self._attempt_restart(lang)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "LSPHealthLoop iteration error: %s", exc, exc_info=True
                )
                await asyncio.sleep(5.0)

    async def record_activity(self, language: str) -> None:
        """Manually record activity for a language server."""
        self._last_activity[language] = time.monotonic()

    async def _check_idle_servers(self) -> None:
        """Shut down servers that have been idle beyond the configured timeout."""
        now = time.monotonic()
        for lang in list(self._last_activity.keys()):
            last = self._last_activity.get(lang, now)
            health = self._server_health.get(lang, {})
            if health.get("status") != "healthy":
                continue
            idle_seconds = now - last
            if idle_seconds >= self._idle_shutdown:
                logger.info(
                    "LSP server %s idle for %.0fs, requesting shutdown",
                    lang,
                    idle_seconds,
                )
                shutdown_event = Event(
                    event_type="lsp.idle_shutdown",
                    source="lsp_health",
                    payload={
                        "language": lang,
                        "idle_seconds": idle_seconds,
                    },
                )
                await self._bus.publish(shutdown_event)
                self._server_health[lang]["status"] = "stopped"

    async def _ping_server(self, language: str) -> bool:
        """Ping an LSP server. Uses event bus ping/pong pattern."""
        pong_received = asyncio.Event()

        async def _on_pong(event: Event) -> None:
            if event.payload.get("language") == language:
                pong_received.set()

        sub_id = self._bus.subscribe(
            _on_pong, event_types={"lsp.pong"}
        )
        try:
            ping_event = Event(
                event_type="lsp.ping",
                source="lsp_health",
                payload={"language": language},
            )
            await self._bus.publish(ping_event)
            try:
                await asyncio.wait_for(
                    pong_received.wait(), timeout=self._ping_timeout
                )
                return True
            except asyncio.TimeoutError:
                return (
                    self._server_health.get(language, {}).get("status")
                    == "healthy"
                )
        finally:
            self._bus.unsubscribe(sub_id)

    async def _attempt_restart(self, language: str) -> None:
        """Attempt to restart a failed LSP server."""
        attempts = self._restart_counts.get(language, 0)
        if attempts >= self._max_restarts:
            logger.warning(
                "LSP server %s: max restarts (%d) exhausted",
                language,
                self._max_restarts,
            )
            self._server_health[language]["status"] = "stopped"
            exhausted_event = Event(
                event_type="lsp.restart_exhausted",
                source="lsp_health",
                payload={"language": language, "attempts": attempts},
                priority=EventPriority.HIGH,
            )
            await self._bus.publish(exhausted_event)
            return

        self._restart_counts[language] = attempts + 1
        logger.info(
            "LSP server %s: restart attempt %d/%d",
            language,
            attempts + 1,
            self._max_restarts,
        )
        restart_event = Event(
            event_type="lsp.restart_requested",
            source="lsp_health",
            payload={"language": language, "attempt": attempts + 1},
        )
        await self._bus.publish(restart_event)

    # -- introspection -------------------------------------------------------

    def get_server_health(self) -> dict[str, dict[str, Any]]:
        """Return a copy of all tracked server health statuses."""
        return {k: dict(v) for k, v in self._server_health.items()}

    def register_server(self, language: str) -> None:
        """Manually register a language server for tracking."""
        if language not in self._server_health:
            now = time.monotonic()
            self._server_health[language] = {
                "status": "healthy",
                "last_check": now,
                "consecutive_failures": 0,
            }
            self._last_activity[language] = now
            self._restart_counts.setdefault(language, 0)
