"""Monitoring and resource management for subagents.

This module provides:
- SubagentExecution: Record of a subagent execution
- SubagentMonitor: Tracks subagent executions for debugging and analytics
- SubagentEventType: Lifecycle event types for subagent execution
- SubagentEvent: Event data for lifecycle callbacks
- Persistence: Save/load subagent runs to disk for resume after restart
- AnnounceQueue: Priority queue for subagent announcements (Moltbot parity)
- CrossSessionBus: Message bus for cross-session communication (Moltbot parity)
- DeliveryTracker: Track message delivery context and status (Moltbot parity)

Matches and exceeds Moltbot's subagent-registry.ts capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from collections import defaultdict
import heapq
import json
import threading
import uuid
import logging

logger = logging.getLogger(__name__)


class SubagentEventType(str, Enum):
    """Lifecycle event types for subagent execution.

    Matches Moltbot's onAgentEvent phases.
    """
    STARTED = "started"  # Subagent execution began
    TURN_COMPLETED = "turn_completed"  # A conversation turn completed
    TOOL_CALLED = "tool_called"  # A tool was invoked
    COMPLETED = "completed"  # Subagent finished successfully
    FAILED = "failed"  # Subagent encountered an error
    TIMEOUT = "timeout"  # Subagent exceeded time/turn limits


@dataclass
class SubagentEvent:
    """Event data for subagent lifecycle callbacks.

    Attributes:
        event_type: The type of lifecycle event.
        execution_id: ID of the subagent execution.
        subagent_type: Type of subagent (researcher, coder, etc.).
        timestamp: When the event occurred.
        data: Additional event-specific data.
    """
    event_type: SubagentEventType
    execution_id: str
    subagent_type: str
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "execution_id": self.execution_id,
            "subagent_type": self.subagent_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


# Type alias for event callbacks
SubagentEventCallback = Callable[[SubagentEvent], None]


# =============================================================================
# ANNOUNCE QUEUE SYSTEM (Moltbot Parity)
# =============================================================================


class AnnouncePriority(int, Enum):
    """Priority levels for announcements.

    Higher values = higher priority (dequeued first).
    """
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class AnnounceMessage:
    """A message in the announce queue.

    Attributes:
        id: Unique message identifier.
        source_id: ID of the subagent that published this.
        source_session_id: Session where the message originated.
        priority: Message priority for queue ordering.
        topic: Topic/channel for filtering (e.g., "findings", "errors", "progress").
        content: The announcement content (any JSON-serializable data).
        created_at: When the message was created.
        expires_at: When the message expires (None = never).
        metadata: Additional metadata for the announcement.
    """
    id: str
    source_id: str
    source_session_id: str
    priority: AnnouncePriority
    topic: str
    content: Any
    created_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_session_id": self.source_session_id,
            "priority": self.priority.value,
            "topic": self.topic,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }


# =============================================================================
# CROSS-SESSION COMMUNICATION (Moltbot Parity)
# =============================================================================


@dataclass
class SessionMessage:
    """A message for cross-session communication.

    Attributes:
        id: Unique message identifier.
        from_session: Source session ID.
        to_session: Target session ID (None = broadcast to all).
        topic: Message topic for filtering.
        payload: Message content.
        created_at: When the message was sent.
        acknowledged: Whether the message has been acknowledged.
    """
    id: str
    from_session: str
    to_session: str | None  # None = broadcast
    topic: str
    payload: Any
    created_at: datetime
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "from_session": self.from_session,
            "to_session": self.to_session,
            "topic": self.topic,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
        }


# =============================================================================
# DELIVERY CONTEXT TRACKING (Moltbot Parity)
# =============================================================================


class DeliveryStatus(str, Enum):
    """Status of message delivery."""
    PENDING = "pending"  # Message queued, not yet delivered
    DELIVERED = "delivered"  # Message delivered to recipient
    ACKNOWLEDGED = "acknowledged"  # Recipient acknowledged receipt
    EXPIRED = "expired"  # Message expired before delivery
    FAILED = "failed"  # Delivery failed after retries


@dataclass
class DeliveryContext:
    """Tracks delivery status for a message.

    Attributes:
        message_id: ID of the message being tracked.
        recipient_id: Target recipient (session or subagent ID).
        status: Current delivery status.
        attempts: Number of delivery attempts.
        created_at: When tracking started.
        delivered_at: When message was delivered (if applicable).
        acknowledged_at: When recipient acknowledged (if applicable).
        last_attempt_at: When last delivery attempt was made.
        error: Error message if delivery failed.
    """
    message_id: str
    recipient_id: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    delivered_at: datetime | None = None
    acknowledged_at: datetime | None = None
    last_attempt_at: datetime | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "recipient_id": self.recipient_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "error": self.error,
        }


@dataclass
class SubagentExecution:
    """Record of a subagent execution.

    Attributes:
        id: Unique identifier for this execution.
        parent_id: ID of the parent agent/execution that spawned this subagent.
        subagent_type: Type of subagent (researcher, coder, reviewer, planner).
        task: The task description given to the subagent.
        started_at: When the execution started.
        ended_at: When the execution ended (None if still running).
        turns: Number of conversation turns.
        tool_calls: List of tool calls made during execution.
        result: Final result from the subagent (None if still running or failed).
        error: Error message if execution failed.
        tokens_used: Total tokens consumed during execution.
    """
    id: str
    parent_id: str
    subagent_type: str
    task: str
    started_at: datetime
    ended_at: datetime | None = None
    turns: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    result: str | None = None
    error: str | None = None
    tokens_used: int = 0

    @property
    def duration_seconds(self) -> float | None:
        """Get execution duration in seconds."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()

    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.ended_at is not None

    @property
    def is_success(self) -> bool:
        """Check if execution completed successfully."""
        return self.is_complete and self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "subagent_type": self.subagent_type,
            "task": self.task,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "result": self.result,
            "error": self.error,
            "tokens_used": self.tokens_used,
            "duration_seconds": self.duration_seconds,
            "is_success": self.is_success,
        }


class SubagentMonitor:
    """Monitors subagent executions for debugging and resource tracking.

    This class maintains a history of subagent executions and provides
    statistics for monitoring and debugging purposes.

    Features matching Moltbot:
    - Disk persistence for subagent runs (save/load)
    - Lifecycle event callbacks (started, completed, failed, timeout)
    - Resume capability after restart
    """

    # Default persistence path
    DEFAULT_PERSISTENCE_PATH = Path.home() / ".ag3nt" / "subagent_runs.json"

    def __init__(
        self,
        max_history: int = 100,
        persistence_path: Path | str | None = None,
        auto_persist: bool = True,
    ):
        """Initialize the monitor.

        Args:
            max_history: Maximum number of completed executions to retain.
            persistence_path: Path to save/load executions. None disables persistence.
            auto_persist: If True, automatically save after each execution ends.
        """
        self.executions: list[SubagentExecution] = []
        self.max_history = max_history
        self.active_subagents: dict[str, SubagentExecution] = {}
        self.persistence_path = (
            Path(persistence_path) if persistence_path else self.DEFAULT_PERSISTENCE_PATH
        )
        self.auto_persist = auto_persist
        self._lock = threading.Lock()

        # Lifecycle event callbacks
        self._event_callbacks: dict[SubagentEventType, list[SubagentEventCallback]] = {
            event_type: [] for event_type in SubagentEventType
        }
        self._global_callbacks: list[SubagentEventCallback] = []

    def on_event(
        self,
        event_type: SubagentEventType | None = None,
        callback: SubagentEventCallback | None = None,
    ) -> Callable[[SubagentEventCallback], SubagentEventCallback] | None:
        """Register a callback for lifecycle events.

        Can be used as a decorator or called directly.

        Args:
            event_type: Specific event type to listen for, or None for all events.
            callback: The callback function. If None, returns a decorator.

        Returns:
            The callback (for decorator use) or None.

        Example:
            # As decorator
            @monitor.on_event(SubagentEventType.COMPLETED)
            def on_complete(event):
                print(f"Subagent {event.execution_id} completed")

            # Direct registration
            monitor.on_event(SubagentEventType.FAILED, my_error_handler)
        """
        def register(cb: SubagentEventCallback) -> SubagentEventCallback:
            with self._lock:
                if event_type is None:
                    self._global_callbacks.append(cb)
                else:
                    self._event_callbacks[event_type].append(cb)
            return cb

        if callback is not None:
            register(callback)
            return None
        return register

    def remove_callback(
        self,
        callback: SubagentEventCallback,
        event_type: SubagentEventType | None = None,
    ) -> bool:
        """Remove a registered callback.

        Args:
            callback: The callback to remove.
            event_type: Specific event type, or None to remove from global.

        Returns:
            True if callback was found and removed.
        """
        with self._lock:
            if event_type is None:
                if callback in self._global_callbacks:
                    self._global_callbacks.remove(callback)
                    return True
            else:
                if callback in self._event_callbacks[event_type]:
                    self._event_callbacks[event_type].remove(callback)
                    return True
            return False

    def _emit_event(
        self,
        event_type: SubagentEventType,
        execution_id: str,
        subagent_type: str,
        data: dict[str, Any] | None = None,
    ) -> SubagentEvent:
        """Emit a lifecycle event to all registered callbacks.

        Args:
            event_type: The type of event.
            execution_id: ID of the subagent execution.
            subagent_type: Type of subagent.
            data: Additional event data.

        Returns:
            The emitted event.
        """
        event = SubagentEvent(
            event_type=event_type,
            execution_id=execution_id,
            subagent_type=subagent_type,
            timestamp=datetime.now(),
            data=data or {},
        )

        # Snapshot callback lists under lock, iterate outside lock
        with self._lock:
            type_callbacks = list(self._event_callbacks[event_type])
            global_callbacks = list(self._global_callbacks)

        # Call type-specific callbacks
        for cb in type_callbacks:
            try:
                cb(event)
            except (TypeError, ValueError, RuntimeError):
                pass  # Don't let callback errors break execution

        # Call global callbacks
        for cb in global_callbacks:
            try:
                cb(event)
            except (TypeError, ValueError, RuntimeError):
                pass

        return event

    def start_execution(
        self,
        parent_id: str,
        subagent_type: str,
        task: str,
        execution_id: str | None = None,
    ) -> SubagentExecution:
        """Record the start of a subagent execution.

        Args:
            parent_id: ID of the parent agent/execution.
            subagent_type: Type of subagent being spawned.
            task: The task description for the subagent.
            execution_id: Optional custom execution ID (auto-generated if None).

        Returns:
            The created SubagentExecution record.
        """
        exec_id = execution_id or f"subagent_{uuid.uuid4().hex[:12]}"
        execution = SubagentExecution(
            id=exec_id,
            parent_id=parent_id,
            subagent_type=subagent_type,
            task=task,
            started_at=datetime.now(),
        )
        with self._lock:
            self.active_subagents[exec_id] = execution

        # Emit STARTED event (callbacks invoked outside lock)
        self._emit_event(
            SubagentEventType.STARTED,
            exec_id,
            subagent_type,
            {"parent_id": parent_id, "task": task},
        )

        return execution

    def record_turn(self, execution_id: str) -> None:
        """Record a turn in the subagent conversation.

        Args:
            execution_id: The execution to update.
        """
        with self._lock:
            if execution_id not in self.active_subagents:
                return
            execution = self.active_subagents[execution_id]
            execution.turns += 1
            turn_number = execution.turns
            subagent_type = execution.subagent_type

        # Emit TURN_COMPLETED event (outside lock)
        self._emit_event(
            SubagentEventType.TURN_COMPLETED,
            execution_id,
            subagent_type,
            {"turn_number": turn_number},
        )

    def record_tool_call(
        self,
        execution_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
    ) -> None:
        """Record a tool call made by the subagent.

        Args:
            execution_id: The execution to update.
            tool_name: Name of the tool called.
            args: Arguments passed to the tool.
            result: Result returned by the tool.
        """
        with self._lock:
            if execution_id not in self.active_subagents:
                return
            execution = self.active_subagents[execution_id]
            tool_call_record = {
                "tool": tool_name,
                "args": args,
                "result_preview": str(result)[:200],
                "timestamp": datetime.now().isoformat(),
            }
            execution.tool_calls.append(tool_call_record)
            subagent_type = execution.subagent_type

        # Emit TOOL_CALLED event (outside lock)
        self._emit_event(
            SubagentEventType.TOOL_CALLED,
            execution_id,
            subagent_type,
            {"tool_name": tool_name, "args": args},
        )

    def record_tokens(self, execution_id: str, tokens: int) -> None:
        """Record tokens used by the subagent.

        Args:
            execution_id: The execution to update.
            tokens: Number of tokens to add.
        """
        with self._lock:
            if execution_id in self.active_subagents:
                self.active_subagents[execution_id].tokens_used += tokens

    def end_execution(
        self,
        execution_id: str,
        result: str | None = None,
        error: str | None = None,
        tokens_used: int = 0,
        timeout: bool = False,
    ) -> SubagentExecution | None:
        """Record the end of a subagent execution.

        Args:
            execution_id: The execution to complete.
            result: Final result from the subagent.
            error: Error message if execution failed.
            tokens_used: Additional tokens to add to the total.
            timeout: Whether the execution ended due to timeout.

        Returns:
            The completed SubagentExecution, or None if not found.
        """
        with self._lock:
            if execution_id not in self.active_subagents:
                return None

            execution = self.active_subagents[execution_id]
            execution.ended_at = datetime.now()
            execution.result = result
            execution.error = error
            execution.tokens_used += tokens_used

            # Append to history BEFORE removing from active to prevent data loss
            self.executions.append(execution)
            # Trim history if needed
            if len(self.executions) > self.max_history:
                self.executions = self.executions[-self.max_history:]

            subagent_type = execution.subagent_type
            emit_data = {
                "result": result[:200] if result else None,
                "error": error,
                "duration_seconds": execution.duration_seconds,
                "tokens_used": execution.tokens_used,
                "turns": execution.turns,
            }

        # Determine event type outside lock (immutable data)
        if timeout:
            event_type = SubagentEventType.TIMEOUT
        elif error:
            event_type = SubagentEventType.FAILED
        else:
            event_type = SubagentEventType.COMPLETED

        # Emit event outside lock (callbacks may re-enter)
        self._emit_event(
            event_type,
            execution_id,
            subagent_type,
            emit_data,
        )

        try:
            # Auto-persist if enabled
            if self.auto_persist:
                self.save_to_disk()
        finally:
            # Remove from active only after save attempt completes
            with self._lock:
                self.active_subagents.pop(execution_id, None)

        return execution

    def get_execution(self, execution_id: str) -> SubagentExecution | None:
        """Get an execution by ID.

        Args:
            execution_id: The execution ID to look up.

        Returns:
            The execution if found, None otherwise.
        """
        with self._lock:
            # Check active first
            if execution_id in self.active_subagents:
                return self.active_subagents[execution_id]
            # Check completed
            for execution in self.executions:
                if execution.id == execution_id:
                    return execution
            return None

    def get_active_count(self) -> int:
        """Get number of active subagents.

        Returns:
            Number of currently running subagents.
        """
        with self._lock:
            return len(self.active_subagents)

    def get_active_executions(self) -> list[SubagentExecution]:
        """Get all active subagent executions.

        Returns:
            List of currently running executions.
        """
        with self._lock:
            return list(self.active_subagents.values())

    def get_recent_executions(self, limit: int = 10) -> list[SubagentExecution]:
        """Get recent completed executions.

        Args:
            limit: Maximum number of executions to return.

        Returns:
            List of recent completed executions (newest first).
        """
        with self._lock:
            return list(reversed(self.executions[-limit:]))

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution statistics.
        """
        with self._lock:
            if not self.executions:
                return {
                    "total_executions": 0,
                    "active_count": len(self.active_subagents),
                }

            executions_snapshot = list(self.executions)
            active_count = len(self.active_subagents)

        successful = [e for e in executions_snapshot if e.is_success]
        failed = [e for e in executions_snapshot if e.error is not None]
        durations = [e.duration_seconds for e in executions_snapshot if e.duration_seconds]

        counts: dict[str, int] = {}
        for e in executions_snapshot:
            counts[e.subagent_type] = counts.get(e.subagent_type, 0) + 1

        return {
            "total_executions": len(executions_snapshot),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(executions_snapshot) if executions_snapshot else 0,
            "active_count": active_count,
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 0,
            "total_tokens": sum(e.tokens_used for e in executions_snapshot),
            "avg_turns": sum(e.turns for e in executions_snapshot) / len(executions_snapshot),
            "avg_tool_calls": sum(len(e.tool_calls) for e in executions_snapshot) / len(executions_snapshot),
            "by_type": counts,
        }

    def clear_history(self) -> int:
        """Clear completed execution history.

        Returns:
            Number of executions cleared.
        """
        with self._lock:
            count = len(self.executions)
            self.executions.clear()
        if self.auto_persist:
            self.save_to_disk()
        return count

    # =========================================================================
    # PERSISTENCE METHODS (Matching Moltbot's subagent-registry.store.ts)
    # =========================================================================

    def save_to_disk(self) -> bool:
        """Save completed executions to disk.

        Returns:
            True if save was successful.
        """
        try:
            # Ensure directory exists
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            # Serialize executions
            data = {
                "version": 1,
                "saved_at": datetime.now().isoformat(),
                "max_history": self.max_history,
                "executions": [e.to_dict() for e in self.executions],
            }

            # Write atomically (write to temp, then rename)
            temp_path = self.persistence_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.persistence_path)

            return True
        except (OSError, TypeError, ValueError):
            return False

    def load_from_disk(self) -> int:
        """Load executions from disk.

        Returns:
            Number of executions loaded.
        """
        if not self.persistence_path.exists():
            return 0

        try:
            with open(self.persistence_path, encoding="utf-8") as f:
                data = json.load(f)

            # Parse executions
            loaded_executions = []
            for e_dict in data.get("executions", []):
                execution = SubagentExecution(
                    id=e_dict["id"],
                    parent_id=e_dict["parent_id"],
                    subagent_type=e_dict["subagent_type"],
                    task=e_dict["task"],
                    started_at=datetime.fromisoformat(e_dict["started_at"]),
                    ended_at=(
                        datetime.fromisoformat(e_dict["ended_at"])
                        if e_dict.get("ended_at")
                        else None
                    ),
                    turns=e_dict.get("turns", 0),
                    tool_calls=e_dict.get("tool_calls", []),
                    result=e_dict.get("result"),
                    error=e_dict.get("error"),
                    tokens_used=e_dict.get("tokens_used", 0),
                )
                loaded_executions.append(execution)

            self.executions = loaded_executions[-self.max_history:]
            return len(self.executions)
        except (OSError, KeyError, ValueError):
            return 0

    def delete_persistence_file(self) -> bool:
        """Delete the persistence file.

        Returns:
            True if file was deleted or didn't exist.
        """
        try:
            if self.persistence_path.exists():
                self.persistence_path.unlink()
            return True
        except OSError:
            return False


# =============================================================================
# ANNOUNCE QUEUE SYSTEM
# =============================================================================


class AnnounceQueue:
    """Priority queue for subagent announcements.

    Subagents can publish announcements to topics, and subscribers can poll
    for messages by topic. Messages are ordered by priority (highest first),
    then by creation time (oldest first within same priority).

    Features:
    - Priority-based ordering (URGENT > HIGH > NORMAL > LOW)
    - Topic-based filtering
    - Automatic expiration handling
    - Thread-safe operations

    Example:
        queue = AnnounceQueue()

        # Publisher (subagent)
        queue.publish(
            source_id="researcher_1",
            source_session_id="session_123",
            topic="findings",
            content={"key_insight": "..."},
            priority=AnnouncePriority.HIGH,
        )

        # Subscriber (parent agent)
        messages = queue.poll(topic="findings", limit=5)
    """

    def __init__(self, max_size: int = 1000, default_ttl_seconds: int = 3600):
        """Initialize the announce queue.

        Args:
            max_size: Maximum number of messages to retain.
            default_ttl_seconds: Default time-to-live for messages (1 hour).
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds

        # Priority queue: (negative_priority, sequence, message)
        # Negative priority ensures higher priority comes first in min-heap
        # Sequence ensures FIFO order for same priority and prevents message comparison
        self._queue: list[tuple[int, int, AnnounceMessage]] = []
        self._sequence = 0  # Monotonically increasing counter for FIFO tie-breaking
        self._lock = threading.Lock()

        # Topic subscriptions: topic -> list of session_ids
        self._subscriptions: dict[str, set[str]] = defaultdict(set)

    def publish(
        self,
        source_id: str,
        source_session_id: str,
        topic: str,
        content: Any,
        priority: AnnouncePriority = AnnouncePriority.NORMAL,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AnnounceMessage:
        """Publish an announcement to the queue.

        Args:
            source_id: ID of the publishing subagent.
            source_session_id: Session ID where announcement originated.
            topic: Topic/channel for the announcement.
            content: The announcement content.
            priority: Message priority (default: NORMAL).
            ttl_seconds: Time-to-live in seconds (None = use default).
            metadata: Additional metadata.

        Returns:
            The created AnnounceMessage.
        """
        now = datetime.now()
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expires_at = now + timedelta(seconds=ttl) if ttl > 0 else None

        message = AnnounceMessage(
            id=f"announce_{uuid.uuid4().hex[:12]}",
            source_id=source_id,
            source_session_id=source_session_id,
            priority=priority,
            topic=topic,
            content=content,
            created_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        with self._lock:
            # Add to priority queue (negative priority for max-heap behavior)
            # Use sequence counter as tiebreaker for same priority (FIFO order)
            seq = self._sequence
            self._sequence += 1
            heapq.heappush(
                self._queue,
                (-priority.value, seq, message)
            )

            # Trim if over max size (remove oldest low-priority)
            if len(self._queue) > self.max_size:
                # Sort to find lowest priority, oldest messages
                self._queue.sort(key=lambda x: (x[0], x[1]))  # Lowest priority, oldest sequence
                self._queue = self._queue[:self.max_size]
                heapq.heapify(self._queue)

        return message

    def subscribe(self, session_id: str, topic: str) -> None:
        """Subscribe a session to a topic.

        Args:
            session_id: Session ID to subscribe.
            topic: Topic to subscribe to.
        """
        with self._lock:
            self._subscriptions[topic].add(session_id)

    def unsubscribe(self, session_id: str, topic: str | None = None) -> None:
        """Unsubscribe a session from topics.

        Args:
            session_id: Session ID to unsubscribe.
            topic: Specific topic, or None to unsubscribe from all.
        """
        with self._lock:
            if topic is None:
                empty_topics = []
                for t, subscribers in self._subscriptions.items():
                    subscribers.discard(session_id)
                    if not subscribers:
                        empty_topics.append(t)
                for t in empty_topics:
                    del self._subscriptions[t]
            else:
                self._subscriptions[topic].discard(session_id)
                if not self._subscriptions[topic]:
                    del self._subscriptions[topic]

    def poll(
        self,
        topic: str | None = None,
        session_id: str | None = None,
        limit: int = 10,
        remove: bool = True,
    ) -> list[AnnounceMessage]:
        """Poll for announcements.

        Args:
            topic: Filter by topic (None = all topics).
            session_id: Filter by source session (None = all sessions).
            limit: Maximum messages to return.
            remove: Whether to remove messages from queue after polling.

        Returns:
            List of matching announcements (highest priority first).
        """
        with self._lock:
            # Clean expired messages first
            self._cleanup_expired()

            # Find matching messages
            matches: list[AnnounceMessage] = []
            remaining: list[tuple[int, float, AnnounceMessage]] = []

            for item in self._queue:
                _, _, msg = item
                is_match = True

                if topic is not None and msg.topic != topic:
                    is_match = False
                if session_id is not None and msg.source_session_id != session_id:
                    is_match = False

                if is_match and len(matches) < limit:
                    matches.append(msg)
                    if not remove:
                        remaining.append(item)
                else:
                    remaining.append(item)

            if remove:
                self._queue = remaining
                heapq.heapify(self._queue)

            return matches

    def poll_all(self, topic: str | None = None) -> list[AnnounceMessage]:
        """Poll all matching announcements.

        Args:
            topic: Filter by topic (None = all topics).

        Returns:
            All matching announcements.
        """
        return self.poll(topic=topic, limit=self.max_size, remove=True)

    def peek(self, topic: str | None = None, limit: int = 10) -> list[AnnounceMessage]:
        """Peek at announcements without removing them.

        Args:
            topic: Filter by topic (None = all topics).
            limit: Maximum messages to return.

        Returns:
            List of matching announcements.
        """
        return self.poll(topic=topic, limit=limit, remove=False)

    def get_subscribers(self, topic: str) -> set[str]:
        """Get all subscribers for a topic.

        Args:
            topic: Topic to check.

        Returns:
            Set of subscribed session IDs.
        """
        with self._lock:
            return self._subscriptions[topic].copy()

    def get_topics(self) -> list[str]:
        """Get all topics with subscribers.

        Returns:
            List of topic names.
        """
        with self._lock:
            return [t for t, subs in self._subscriptions.items() if subs]

    def count(self, topic: str | None = None) -> int:
        """Count messages in queue.

        Args:
            topic: Filter by topic (None = count all).

        Returns:
            Number of messages.
        """
        with self._lock:
            if topic is None:
                return len(self._queue)
            return sum(1 for _, _, msg in self._queue if msg.topic == topic)

    def clear(self, topic: str | None = None) -> int:
        """Clear messages from queue.

        Args:
            topic: Topic to clear (None = clear all).

        Returns:
            Number of messages cleared.
        """
        with self._lock:
            if topic is None:
                count = len(self._queue)
                self._queue.clear()
                return count

            original_count = len(self._queue)
            self._queue = [
                item for item in self._queue
                if item[2].topic != topic
            ]
            heapq.heapify(self._queue)
            return original_count - len(self._queue)

    def _cleanup_expired(self) -> int:
        """Remove expired messages. Called with lock held.

        Returns:
            Number of messages removed.
        """
        original_count = len(self._queue)
        self._queue = [
            item for item in self._queue
            if not item[2].is_expired()
        ]
        heapq.heapify(self._queue)
        return original_count - len(self._queue)




# =============================================================================
# CROSS-SESSION COMMUNICATION
# =============================================================================


class CrossSessionBus:
    """Message bus for cross-session communication.

    Enables subagents and sessions to communicate across boundaries.
    Supports direct messaging, broadcast, and topic-based subscriptions.

    Features:
    - Session-to-session direct messaging
    - Broadcast to all sessions
    - Topic-based subscription and filtering
    - Message acknowledgment tracking
    - Thread-safe operations

    Example:
        bus = CrossSessionBus()

        # Session A sends to Session B
        bus.send(
            from_session="session_a",
            to_session="session_b",
            topic="task_result",
            payload={"status": "complete", "data": {...}},
        )

        # Session B polls for messages
        messages = bus.get_messages("session_b")
        for msg in messages:
            process(msg)
            bus.acknowledge(msg.id)
    """

    def __init__(self, max_messages_per_session: int = 100):
        """Initialize the message bus.

        Args:
            max_messages_per_session: Maximum messages to retain per session.
        """
        self.max_messages_per_session = max_messages_per_session

        # Messages by target session: session_id -> list of SessionMessage
        self._mailboxes: dict[str, list[SessionMessage]] = defaultdict(list)

        # Topic subscriptions: topic -> set of session_ids
        self._topic_subscriptions: dict[str, set[str]] = defaultdict(set)

        # Broadcast messages (for all sessions)
        self._broadcast_messages: list[SessionMessage] = []

        self._lock = threading.Lock()

    def send(
        self,
        from_session: str,
        to_session: str,
        topic: str,
        payload: Any,
    ) -> SessionMessage:
        """Send a message to a specific session.

        Args:
            from_session: Source session ID.
            to_session: Target session ID.
            topic: Message topic.
            payload: Message content.

        Returns:
            The created SessionMessage.
        """
        message = SessionMessage(
            id=f"msg_{uuid.uuid4().hex[:12]}",
            from_session=from_session,
            to_session=to_session,
            topic=topic,
            payload=payload,
            created_at=datetime.now(),
        )

        with self._lock:
            mailbox = self._mailboxes[to_session]
            mailbox.append(message)

            # Trim if over limit
            if len(mailbox) > self.max_messages_per_session:
                dropped = len(mailbox) - self.max_messages_per_session
                logger.warning(
                    f"CrossSessionBus: mailbox for {to_session} overflow, dropping {dropped} oldest message(s)"
                )
                self._mailboxes[to_session] = mailbox[-self.max_messages_per_session:]

        return message

    def broadcast(
        self,
        from_session: str,
        topic: str,
        payload: Any,
    ) -> SessionMessage:
        """Broadcast a message to all sessions.

        Args:
            from_session: Source session ID.
            topic: Message topic.
            payload: Message content.

        Returns:
            The created SessionMessage.
        """
        message = SessionMessage(
            id=f"bcast_{uuid.uuid4().hex[:12]}",
            from_session=from_session,
            to_session=None,  # None = broadcast
            topic=topic,
            payload=payload,
            created_at=datetime.now(),
        )

        with self._lock:
            self._broadcast_messages.append(message)

            # Trim broadcasts
            if len(self._broadcast_messages) > self.max_messages_per_session:
                dropped = len(self._broadcast_messages) - self.max_messages_per_session
                logger.warning(
                    f"CrossSessionBus: broadcast overflow, dropping {dropped} oldest message(s)"
                )
                self._broadcast_messages = self._broadcast_messages[
                    -self.max_messages_per_session:
                ]

        return message

    def subscribe_topic(self, session_id: str, topic: str) -> None:
        """Subscribe a session to a topic.

        Args:
            session_id: Session ID to subscribe.
            topic: Topic to subscribe to.
        """
        with self._lock:
            self._topic_subscriptions[topic].add(session_id)

    def unsubscribe_topic(self, session_id: str, topic: str | None = None) -> None:
        """Unsubscribe a session from topics.

        Args:
            session_id: Session ID to unsubscribe.
            topic: Specific topic, or None for all topics.
        """
        with self._lock:
            if topic is None:
                empty_topics = []
                for t, subscribers in self._topic_subscriptions.items():
                    subscribers.discard(session_id)
                    if not subscribers:
                        empty_topics.append(t)
                for t in empty_topics:
                    del self._topic_subscriptions[t]
            else:
                self._topic_subscriptions[topic].discard(session_id)
                if not self._topic_subscriptions[topic]:
                    del self._topic_subscriptions[topic]

    def get_messages(
        self,
        session_id: str,
        topic: str | None = None,
        include_broadcasts: bool = True,
        unacknowledged_only: bool = False,
    ) -> list[SessionMessage]:
        """Get messages for a session.

        Args:
            session_id: Session to get messages for.
            topic: Filter by topic (None = all topics).
            include_broadcasts: Whether to include broadcast messages.
            unacknowledged_only: Only return unacknowledged messages.

        Returns:
            List of messages (newest first).
        """
        with self._lock:
            messages: list[SessionMessage] = []

            # Direct messages
            for msg in self._mailboxes.get(session_id, []):
                if topic is not None and msg.topic != topic:
                    continue
                if unacknowledged_only and msg.acknowledged:
                    continue
                messages.append(msg)

            # Broadcast messages (if requested)
            if include_broadcasts:
                subscribed_topics = set()
                for t, subs in self._topic_subscriptions.items():
                    if session_id in subs:
                        subscribed_topics.add(t)

                for msg in self._broadcast_messages:
                    # Include if subscribed to topic or topic filter matches
                    if topic is not None and msg.topic != topic:
                        continue
                    if unacknowledged_only and msg.acknowledged:
                        continue
                    # Include broadcasts that match subscribed topics
                    if topic is None and msg.topic not in subscribed_topics:
                        continue
                    messages.append(msg)

            # Sort by creation time (newest first)
            messages.sort(key=lambda m: m.created_at, reverse=True)
            return messages

    def acknowledge(self, message_id: str) -> bool:
        """Acknowledge a message.

        Args:
            message_id: ID of the message to acknowledge.

        Returns:
            True if message was found and acknowledged.
        """
        with self._lock:
            # Check direct messages
            for mailbox in self._mailboxes.values():
                for msg in mailbox:
                    if msg.id == message_id:
                        msg.acknowledged = True
                        return True

            # Check broadcasts
            for msg in self._broadcast_messages:
                if msg.id == message_id:
                    msg.acknowledged = True
                    return True

            return False

    def clear_session(self, session_id: str) -> int:
        """Clear all messages for a session.

        Args:
            session_id: Session to clear.

        Returns:
            Number of messages cleared.
        """
        with self._lock:
            count = len(self._mailboxes.get(session_id, []))
            self._mailboxes[session_id] = []
            return count

    def get_statistics(self) -> dict[str, Any]:
        """Get message bus statistics.

        Returns:
            Statistics dictionary.
        """
        with self._lock:
            total_direct = sum(len(m) for m in self._mailboxes.values())
            total_broadcast = len(self._broadcast_messages)
            active_sessions = len([m for m in self._mailboxes.values() if m])

            return {
                "total_direct_messages": total_direct,
                "total_broadcast_messages": total_broadcast,
                "active_sessions": active_sessions,
                "topic_subscriptions": {
                    t: len(s) for t, s in self._topic_subscriptions.items() if s
                },
            }


# =============================================================================
# DELIVERY CONTEXT TRACKING
# =============================================================================


class DeliveryTracker:
    """Tracks message delivery context and status.

    Provides visibility into message delivery lifecycle including
    pending, delivered, acknowledged, and failed states.

    Features:
    - Delivery status tracking per message/recipient
    - Retry attempt counting
    - Acknowledgment recording
    - Failed delivery handling

    Example:
        tracker = DeliveryTracker()

        # Track a new message
        tracker.track("msg_123", "session_456")

        # Mark as delivered
        tracker.mark_delivered("msg_123", "session_456")

        # Later, mark as acknowledged
        tracker.acknowledge("msg_123", "session_456")

        # Get pending deliveries for retry
        pending = tracker.get_pending(max_attempts=3)
    """

    def __init__(self, max_history: int = 500):
        """Initialize the delivery tracker.

        Args:
            max_history: Maximum delivery contexts to retain.
        """
        self.max_history = max_history

        # Delivery contexts: (message_id, recipient_id) -> DeliveryContext
        self._contexts: dict[tuple[str, str], DeliveryContext] = {}
        self._lock = threading.Lock()

    def track(self, message_id: str, recipient_id: str) -> DeliveryContext:
        """Start tracking delivery for a message.

        Args:
            message_id: ID of the message.
            recipient_id: ID of the recipient.

        Returns:
            The created DeliveryContext.
        """
        context = DeliveryContext(
            message_id=message_id,
            recipient_id=recipient_id,
            status=DeliveryStatus.PENDING,
            created_at=datetime.now(),
        )

        with self._lock:
            key = (message_id, recipient_id)
            self._contexts[key] = context
            self._trim_history()

        return context

    def mark_delivered(self, message_id: str, recipient_id: str) -> bool:
        """Mark a message as delivered.

        Args:
            message_id: ID of the message.
            recipient_id: ID of the recipient.

        Returns:
            True if context was found and updated.
        """
        with self._lock:
            key = (message_id, recipient_id)
            if key not in self._contexts:
                return False

            ctx = self._contexts[key]
            ctx.status = DeliveryStatus.DELIVERED
            ctx.delivered_at = datetime.now()
            ctx.attempts += 1
            ctx.last_attempt_at = ctx.delivered_at
            return True

    def acknowledge(self, message_id: str, recipient_id: str) -> bool:
        """Mark a message as acknowledged by recipient.

        Args:
            message_id: ID of the message.
            recipient_id: ID of the recipient.

        Returns:
            True if context was found and updated.
        """
        with self._lock:
            key = (message_id, recipient_id)
            if key not in self._contexts:
                return False

            ctx = self._contexts[key]
            ctx.status = DeliveryStatus.ACKNOWLEDGED
            ctx.acknowledged_at = datetime.now()
            return True

    def mark_failed(self, message_id: str, recipient_id: str, error: str) -> bool:
        """Mark a delivery as failed.

        Args:
            message_id: ID of the message.
            recipient_id: ID of the recipient.
            error: Error message describing the failure.

        Returns:
            True if context was found and updated.
        """
        with self._lock:
            key = (message_id, recipient_id)
            if key not in self._contexts:
                return False

            ctx = self._contexts[key]
            ctx.status = DeliveryStatus.FAILED
            ctx.error = error
            ctx.attempts += 1
            ctx.last_attempt_at = datetime.now()
            return True

    def mark_expired(self, message_id: str, recipient_id: str) -> bool:
        """Mark a message as expired.

        Args:
            message_id: ID of the message.
            recipient_id: ID of the recipient.

        Returns:
            True if context was found and updated.
        """
        with self._lock:
            key = (message_id, recipient_id)
            if key not in self._contexts:
                return False

            ctx = self._contexts[key]
            ctx.status = DeliveryStatus.EXPIRED
            return True

    def record_attempt(self, message_id: str, recipient_id: str) -> bool:
        """Record a delivery attempt.

        Args:
            message_id: ID of the message.
            recipient_id: ID of the recipient.

        Returns:
            True if context was found and updated.
        """
        with self._lock:
            key = (message_id, recipient_id)
            if key not in self._contexts:
                return False

            ctx = self._contexts[key]
            ctx.attempts += 1
            ctx.last_attempt_at = datetime.now()
            return True

    def get_context(self, message_id: str, recipient_id: str) -> DeliveryContext | None:
        """Get delivery context for a message/recipient pair.

        Args:
            message_id: ID of the message.
            recipient_id: ID of the recipient.

        Returns:
            DeliveryContext if found, None otherwise.
        """
        with self._lock:
            key = (message_id, recipient_id)
            return self._contexts.get(key)

    def get_pending(
        self,
        max_attempts: int | None = None,
        older_than_seconds: float | None = None,
    ) -> list[DeliveryContext]:
        """Get pending deliveries.

        Args:
            max_attempts: Only return contexts with fewer attempts.
            older_than_seconds: Only return contexts older than this.

        Returns:
            List of pending DeliveryContext objects.
        """
        with self._lock:
            now = datetime.now()
            results: list[DeliveryContext] = []

            for ctx in self._contexts.values():
                if ctx.status != DeliveryStatus.PENDING:
                    continue
                if max_attempts is not None and ctx.attempts >= max_attempts:
                    continue
                if older_than_seconds is not None:
                    age = (now - ctx.created_at).total_seconds()
                    if age < older_than_seconds:
                        continue
                results.append(ctx)

            return results

    def get_failed(self) -> list[DeliveryContext]:
        """Get all failed deliveries.

        Returns:
            List of failed DeliveryContext objects.
        """
        with self._lock:
            return [
                ctx for ctx in self._contexts.values()
                if ctx.status == DeliveryStatus.FAILED
            ]

    def get_statistics(self) -> dict[str, Any]:
        """Get delivery statistics.

        Returns:
            Statistics dictionary.
        """
        with self._lock:
            status_counts: dict[str, int] = defaultdict(int)
            total_attempts = 0

            for ctx in self._contexts.values():
                status_counts[ctx.status.value] += 1
                total_attempts += ctx.attempts

            return {
                "total_tracked": len(self._contexts),
                "by_status": dict(status_counts),
                "total_attempts": total_attempts,
            }

    def clear(self) -> int:
        """Clear all delivery contexts.

        Returns:
            Number of contexts cleared.
        """
        with self._lock:
            count = len(self._contexts)
            self._contexts.clear()
            return count

    def _trim_history(self) -> None:
        """Trim to max_history. Called with lock held."""
        if len(self._contexts) <= self.max_history:
            return

        # Keep most recent by created_at
        sorted_keys = sorted(
            self._contexts.keys(),
            key=lambda k: self._contexts[k].created_at,
            reverse=True,
        )
        keep_keys = set(sorted_keys[:self.max_history])
        self._contexts = {
            k: v for k, v in self._contexts.items()
            if k in keep_keys
        }


# =============================================================================
# GLOBAL SINGLETON INSTANCES
# =============================================================================

# Global announce queue for cross-subagent announcements
_global_announce_queue: AnnounceQueue | None = None

# Global cross-session bus for inter-session communication
_global_cross_session_bus: CrossSessionBus | None = None

# Global delivery tracker
_global_delivery_tracker: DeliveryTracker | None = None

# Lock for thread-safe singleton creation
_global_singleton_lock = threading.Lock()


def get_announce_queue() -> AnnounceQueue:
    """Get or create the global announce queue.

    Returns:
        The global AnnounceQueue instance.
    """
    global _global_announce_queue
    if _global_announce_queue is None:
        with _global_singleton_lock:
            if _global_announce_queue is None:
                _global_announce_queue = AnnounceQueue()
    return _global_announce_queue


def get_cross_session_bus() -> CrossSessionBus:
    """Get or create the global cross-session bus.

    Returns:
        The global CrossSessionBus instance.
    """
    global _global_cross_session_bus
    if _global_cross_session_bus is None:
        with _global_singleton_lock:
            if _global_cross_session_bus is None:
                _global_cross_session_bus = CrossSessionBus()
    return _global_cross_session_bus


def get_delivery_tracker() -> DeliveryTracker:
    """Get or create the global delivery tracker.

    Returns:
        The global DeliveryTracker instance.
    """
    global _global_delivery_tracker
    if _global_delivery_tracker is None:
        with _global_singleton_lock:
            if _global_delivery_tracker is None:
                _global_delivery_tracker = DeliveryTracker()
    return _global_delivery_tracker


def reset_global_instances() -> None:
    """Reset all global instances. Useful for testing."""
    global _global_announce_queue, _global_cross_session_bus, _global_delivery_tracker
    with _global_singleton_lock:
        _global_announce_queue = None
        _global_cross_session_bus = None
        _global_delivery_tracker = None
