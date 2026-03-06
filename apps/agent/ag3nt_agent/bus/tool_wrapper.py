"""EventEmittingToolWrapper — wraps LangChain tools to emit bus events.

Every tool invocation emits:
- ``tool.called`` before execution (with tool_name, arguments)
- ``tool.completed`` after execution (with tool_name, success, duration_ms, output_size)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.bus.tool_wrapper")


class EventEmittingToolWrapper:
    """Wraps a LangChain tool to emit events on invoke."""

    def __init__(self, tool: Any, bus: EventBus) -> None:
        self._tool = tool
        self._bus = bus
        self.name = getattr(tool, "name", str(tool))
        self.description = getattr(tool, "description", "")
        self.args_schema = getattr(tool, "args_schema", None)
        self.return_direct = getattr(tool, "return_direct", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tool, name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        args = input if isinstance(input, dict) else {"input": input}
        self._emit_called(args)
        start = time.monotonic()
        try:
            result = self._tool.invoke(input, config, **kwargs)
            duration_ms = int((time.monotonic() - start) * 1000)
            self._emit_completed(True, duration_ms, result)
            return result
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._emit_completed(False, duration_ms, None, str(exc))
            raise

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        args = input if isinstance(input, dict) else {"input": input}
        self._emit_called(args)
        start = time.monotonic()
        try:
            result = await self._tool.ainvoke(input, config, **kwargs)
            duration_ms = int((time.monotonic() - start) * 1000)
            self._emit_completed(True, duration_ms, result)
            return result
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            self._emit_completed(False, duration_ms, None, str(exc))
            raise

    def _emit_called(self, arguments: dict) -> None:
        event = Event(
            event_type="tool.called",
            source=f"tool:{self.name}",
            payload={"tool_name": self.name, "arguments": arguments},
            priority=EventPriority.LOW,
        )
        self._bus.emit_sync(event)

    def _emit_completed(self, success: bool, duration_ms: int, result: Any, error: str | None = None) -> None:
        output_size = len(str(result)) if result is not None else 0
        event = Event(
            event_type="tool.completed",
            source=f"tool:{self.name}",
            payload={
                "tool_name": self.name,
                "success": success,
                "duration_ms": duration_ms,
                "output_size": output_size,
                "error": error,
            },
            priority=EventPriority.LOW,
        )
        self._bus.emit_sync(event)


def wrap_tools_with_events(tools: list, bus: EventBus) -> list:
    """Wrap a list of LangChain tools with event emission.
    Tools that are already wrapped are skipped.
    """
    wrapped = []
    for tool in tools:
        if isinstance(tool, EventEmittingToolWrapper):
            wrapped.append(tool)
        else:
            wrapped.append(EventEmittingToolWrapper(tool, bus))
    return wrapped
