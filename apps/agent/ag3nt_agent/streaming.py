"""Tool execution streaming for real-time progress updates.

This module provides infrastructure for streaming tool execution events
to the Gateway, enabling real-time UI updates during long-running operations.

Events:
- tool_start: Tool execution begins
- tool_progress: Intermediate progress update (for long operations)
- tool_end: Tool execution completed successfully
- tool_error: Tool execution failed

Usage:
    from ag3nt_agent.streaming import get_stream_manager

    manager = get_stream_manager()
    manager.emit(ToolEvent(...))
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("ag3nt.streaming")


class EventType(str, Enum):
    """Types of streaming events."""

    TOOL_START = "tool_start"
    TOOL_PROGRESS = "tool_progress"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"


@dataclass
class ToolEvent:
    """Event emitted during tool execution."""

    event_type: EventType
    session_id: str
    tool_name: str
    tool_call_id: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    # Cached dict representation; computed on first to_dict() call.
    # ToolEvents are immutable after creation, so caching is safe.
    _cached_dict: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (cached)."""
        if self._cached_dict is None:
            self._cached_dict = {
                "event_type": self.event_type.value,
                "session_id": self.session_id,
                "tool_name": self.tool_name,
                "tool_call_id": self.tool_call_id,
                "timestamp": self.timestamp,
                **self.data,
            }
        return self._cached_dict


class StreamManager:
    """Manages streaming subscriptions and event dispatch.

    Singleton that handles:
    - WebSocket connections from Gateway
    - Event routing to appropriate subscribers
    - Buffering for disconnected clients
    """

    _instance: StreamManager | None = None

    def __init__(self) -> None:
        # session_id -> list of callbacks
        self._subscribers: dict[str, list[Callable[[ToolEvent], None]]] = {}
        # session_id -> list of buffered events (for reconnection)
        self._event_buffer: dict[str, list[ToolEvent]] = {}
        self._buffer_max_size = 100
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> StreamManager:
        """Get the singleton StreamManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def subscribe(
        self,
        session_id: str,
        callback: Callable[[ToolEvent], None],
    ) -> Callable[[], None]:
        """Subscribe to events for a session.

        Args:
            session_id: Session to subscribe to
            callback: Function called for each event

        Returns:
            Unsubscribe function
        """
        if session_id not in self._subscribers:
            self._subscribers[session_id] = []

        self._subscribers[session_id].append(callback)
        logger.debug(f"Subscribed to session {session_id[:16]}...")

        # Send any buffered events
        if session_id in self._event_buffer:
            for event in self._event_buffer[session_id]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error sending buffered event: {e}")
            # Clear buffer after sending
            del self._event_buffer[session_id]

        def unsubscribe() -> None:
            if session_id in self._subscribers:
                try:
                    self._subscribers[session_id].remove(callback)
                    if not self._subscribers[session_id]:
                        del self._subscribers[session_id]
                except ValueError:
                    pass
            logger.debug(f"Unsubscribed from session {session_id[:16]}...")

        return unsubscribe

    def emit(self, event: ToolEvent) -> None:
        """Emit an event to all subscribers.

        If no subscribers, buffers the event for later delivery.
        """
        session_id = event.session_id

        if session_id in self._subscribers:
            for callback in self._subscribers[session_id]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
        else:
            # Buffer event for later delivery
            if session_id not in self._event_buffer:
                self._event_buffer[session_id] = []

            self._event_buffer[session_id].append(event)

            # Trim buffer if too large
            if len(self._event_buffer[session_id]) > self._buffer_max_size:
                self._event_buffer[session_id] = self._event_buffer[session_id][
                    -self._buffer_max_size :
                ]

        logger.debug(
            f"Event {event.event_type.value} for {event.tool_name} "
            f"(session {session_id[:16]}...)"
        )

    def get_subscriber_count(self, session_id: str) -> int:
        """Get number of subscribers for a session."""
        return len(self._subscribers.get(session_id, []))

    def clear_buffer(self, session_id: str) -> None:
        """Clear buffered events for a session."""
        self._event_buffer.pop(session_id, None)


def get_stream_manager() -> StreamManager:
    """Get the global StreamManager instance."""
    return StreamManager.get_instance()


