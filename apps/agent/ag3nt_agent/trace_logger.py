"""Lightweight local trace logger.

Two-tier approach:
1. If LANGSMITH_API_KEY is set: Full LangSmith tracing (existing behavior)
2. If not set: Write JSONL traces to ~/.ag3nt/traces/

Records: timestamp, tool name, duration_ms, input_tokens, output_tokens, error, session_id
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger("ag3nt.trace")

_TRACES_DIR = Path.home() / ".ag3nt" / "traces"


@dataclass
class TraceEntry:
    """Single trace log entry."""
    ts: str
    tool: str
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    session_id: str = ""


class LocalTraceLogger:
    """Lightweight JSONL trace logger for when LangSmith isn't configured."""

    def __init__(self, traces_dir: Path | None = None):
        self._dir = traces_dir or _TRACES_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session_id = ""

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def log_tool_call(
        self,
        tool_name: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        entry = TraceEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            tool=tool_name,
            duration_ms=round(duration_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            session_id=self._session_id,
        )
        try:
            path = self._dir / f"{date.today().isoformat()}.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except OSError:
            logger.debug("Failed to write trace entry", exc_info=True)

    def log_model_call(
        self,
        model: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error: str | None = None,
    ) -> None:
        """Log an LLM model call (not a tool call)."""
        self.log_tool_call(
            tool_name=f"llm:{model}",
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
        )

    def read_traces(self, day: date | None = None, limit: int = 100) -> list[dict]:
        """Read trace entries for a given day (default: today)."""
        target = day or date.today()
        path = self._dir / f"{target.isoformat()}.jsonl"
        if not path.exists():
            return []
        entries = []
        try:
            for line in path.read_text(encoding="utf-8").strip().splitlines():
                if line.strip():
                    entries.append(json.loads(line))
                    if len(entries) >= limit:
                        break
        except (OSError, json.JSONDecodeError):
            pass
        return entries


def is_langsmith_configured() -> bool:
    """Check if LangSmith tracing is available."""
    return bool(os.environ.get("LANGSMITH_API_KEY"))


# Module-level singleton
_instance: LocalTraceLogger | None = None


def get_trace_logger() -> LocalTraceLogger:
    """Get or create the singleton trace logger."""
    global _instance
    if _instance is None:
        _instance = LocalTraceLogger()
    return _instance
