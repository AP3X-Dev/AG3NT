"""
Event Bus for AG3NT Autonomous System

Central event routing system that:
- Receives events from various sources (HTTP monitors, file watchers, etc.)
- Routes events to registered handlers
- Supports priority-based event processing
- Provides event deduplication and dead letter queue
"""

import asyncio
import fnmatch
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class EventPriority(IntEnum):
    """Event priority levels for queue ordering."""
    CRITICAL = 0  # Security alerts, system failures
    HIGH = 1      # Service outages, errors
    MEDIUM = 2    # Warnings, degraded performance
    LOW = 3       # Informational, routine events


@dataclass
class Event:
    """
    Represents an event in the autonomous system.

    Events are the primary communication mechanism between
    event sources and the goal/decision system.
    """
    event_type: str
    source: str
    payload: dict = field(default_factory=dict)
    priority: EventPriority = EventPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict = field(default_factory=dict)

    # Deduplication
    dedup_key: Optional[str] = None
    dedup_window_seconds: int = 60

    def __post_init__(self):
        """Normalize event after creation."""
        # Ensure payload is a dict
        if self.payload is None:
            self.payload = {}

        # Generate dedup key if not provided
        if self.dedup_key is None:
            self.dedup_key = self._generate_dedup_key()

    def _generate_dedup_key(self) -> str:
        """Generate a deduplication key based on event content."""
        # Create a hash of event_type, source, and key payload fields
        content = f"{self.event_type}:{self.source}"

        # Add sorted payload keys for consistent hashing
        for key in sorted(self.payload.keys()):
            content += f":{key}={self.payload[key]}"

        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "payload": self.payload,
            "priority": self.priority.name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "dedup_key": self.dedup_key
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Event":
        """Create event from dictionary."""
        priority = EventPriority[data.get("priority", "MEDIUM")]
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow()

        return cls(
            event_id=data.get("event_id", str(uuid4())),
            event_type=data["event_type"],
            source=data["source"],
            payload=data.get("payload", {}),
            priority=priority,
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            dedup_key=data.get("dedup_key")
        )


# Type alias for event handlers
EventHandler = Callable[[Event], Any]


@dataclass
class Subscription:
    """Represents a subscription to events."""
    handler: EventHandler
    event_types: set[str] = field(default_factory=set)  # Empty = all types
    priority_filter: Optional[EventPriority] = None  # None = all priorities
    source_filter: Optional[str] = None  # None = all sources
    subscription_id: str = field(default_factory=lambda: str(uuid4()))


class EventBus:
    """
    Central event bus for the autonomous system.

    Features:
    - Priority queue for event ordering
    - Subscription-based routing
    - Event deduplication
    - Dead letter queue for failed events
    - Metrics tracking
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        dedup_window_seconds: int = 60,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0
    ):
        """
        Initialize the event bus.

        Args:
            max_queue_size: Maximum events in the queue
            dedup_window_seconds: Window for deduplication
            max_retries: Max retry attempts for failed handlers
            retry_delay_seconds: Delay between retries
        """
        self.max_queue_size = max_queue_size
        self.dedup_window_seconds = dedup_window_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Event queue (priority queue using asyncio.PriorityQueue)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)

        # Sequence number for stable priority queue ordering (avoids comparing Event objects)
        self._seq = 0

        # Subscriptions
        self._subscriptions: dict[str, Subscription] = {}
        self._handlers_by_type: dict[str, list[Subscription]] = defaultdict(list)
        self._global_handlers: list[Subscription] = []
        self._pattern_handlers: dict[str, list[Subscription]] = defaultdict(list)

        # Deduplication cache: dedup_key -> expiry_time
        self._dedup_cache: dict[str, datetime] = {}

        # Dead letter queue
        self._dlq: list[tuple[Event, Exception]] = []
        self._dlq_max_size: int = 1000

        # Metrics
        self._metrics = {
            "events_received": 0,
            "events_processed": 0,
            "events_deduplicated": 0,
            "events_failed": 0,
            "handlers_invoked": 0
        }

        # Control
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the event bus processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        self._cleanup_task = asyncio.create_task(self._cleanup_dedup_cache())

        logger.info("Event bus started")

    async def stop(self):
        """Stop the event bus processor."""
        self._running = False

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Event bus stopped")

    def subscribe(
        self,
        handler: EventHandler,
        event_types: Optional[set[str]] = None,
        priority_filter: Optional[EventPriority] = None,
        source_filter: Optional[str] = None
    ) -> str:
        """
        Subscribe a handler to events.

        Args:
            handler: Async or sync callable to handle events
            event_types: Set of event types to handle (None = all)
            priority_filter: Only handle events of this priority or higher
            source_filter: Only handle events from this source

        Returns:
            Subscription ID for unsubscribing
        """
        subscription = Subscription(
            handler=handler,
            event_types=event_types or set(),
            priority_filter=priority_filter,
            source_filter=source_filter
        )

        self._subscriptions[subscription.subscription_id] = subscription

        if not subscription.event_types:
            # Global handler
            self._global_handlers.append(subscription)
        else:
            # Type-specific handlers
            for event_type in subscription.event_types:
                self._handlers_by_type[event_type].append(subscription)

        logger.debug(f"Subscribed handler {subscription.subscription_id} for types: {event_types or 'all'}")

        return subscription.subscription_id

    def subscribe_pattern(
        self,
        pattern: str,
        handler: EventHandler,
        priority_filter: Optional[EventPriority] = None,
        source_filter: Optional[str] = None,
    ) -> str:
        """
        Subscribe to events matching a glob pattern (e.g., 'file.*').

        Uses fnmatch-style glob matching so 'file.*' matches 'file.written',
        'file.deleted', etc. but not 'tool.called'.

        Args:
            pattern: fnmatch glob pattern for event types
            handler: Async or sync callable to handle events
            priority_filter: Only handle events of this priority or higher
            source_filter: Only handle events from this source

        Returns:
            Subscription ID for unsubscribing
        """
        sub_id = str(uuid4())
        sub = Subscription(
            subscription_id=sub_id,
            handler=handler,
            event_types={pattern},
            priority_filter=priority_filter,
            source_filter=source_filter,
        )
        self._pattern_handlers[pattern].append(sub)
        self._subscriptions[sub_id] = sub
        logger.debug(f"Subscribed pattern handler {sub_id} for pattern: {pattern}")
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe a handler.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if unsubscribed, False if not found
        """
        subscription = self._subscriptions.pop(subscription_id, None)
        if not subscription:
            return False

        # Remove from global handlers
        self._global_handlers = [s for s in self._global_handlers if s.subscription_id != subscription_id]

        # Remove from type-specific handlers
        for event_type in list(self._handlers_by_type.keys()):
            self._handlers_by_type[event_type] = [
                s for s in self._handlers_by_type[event_type]
                if s.subscription_id != subscription_id
            ]

        # Remove from pattern handlers
        for pattern in list(self._pattern_handlers.keys()):
            self._pattern_handlers[pattern] = [
                s for s in self._pattern_handlers[pattern]
                if s.subscription_id != subscription_id
            ]
            # Clean up empty pattern lists
            if not self._pattern_handlers[pattern]:
                del self._pattern_handlers[pattern]

        logger.debug(f"Unsubscribed handler {subscription_id}")
        return True

    async def publish(self, event: Event) -> bool:
        """
        Publish an event to the bus.

        Args:
            event: The event to publish

        Returns:
            True if accepted, False if deduplicated or queue full
        """
        self._metrics["events_received"] += 1

        # Check deduplication
        if self._is_duplicate(event):
            self._metrics["events_deduplicated"] += 1
            logger.debug(f"Deduplicated event: {event.dedup_key}")
            return False

        # Add to dedup cache
        expiry = datetime.utcnow() + timedelta(seconds=event.dedup_window_seconds)
        self._dedup_cache[event.dedup_key] = expiry

        # Add to queue with priority
        # PriorityQueue sorts by first element, then second, etc.
        # Using (priority, seq, event) to ensure stable ordering without comparing Event objects
        try:
            self._seq += 1
            self._queue.put_nowait((event.priority.value, self._seq, event))
            logger.debug(f"Published event: {event.event_type} from {event.source}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event: {event.event_id}")
            return False

    def emit_sync(self, event: Event) -> None:
        """
        Fire-and-forget event emission from synchronous code.

        Enqueues the event without awaiting. Use from sync contexts
        (e.g., tool wrappers, middleware). Unlike publish(), this skips
        deduplication to keep the call non-async and fast.

        Args:
            event: The event to enqueue
        """
        try:
            self._seq += 1
            self._queue.put_nowait((event.priority.value, self._seq, event))
            self._metrics["events_received"] += 1
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event: %s", event.event_type)

    def _is_duplicate(self, event: Event) -> bool:
        """Check if event is a duplicate within the dedup window."""
        if event.dedup_key in self._dedup_cache:
            expiry = self._dedup_cache[event.dedup_key]
            if datetime.utcnow() < expiry:
                return True
        return False

    async def _process_events(self):
        """Main event processing loop."""
        while self._running:
            try:
                # Get next event with timeout
                try:
                    priority, seq, event = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Find matching handlers
                handlers = self._get_handlers_for_event(event)

                # Invoke handlers
                for subscription in handlers:
                    await self._invoke_handler(subscription, event)

                self._metrics["events_processed"] += 1
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processor: {e}", exc_info=True)

    def _get_handlers_for_event(self, event: Event) -> list[Subscription]:
        """Get all handlers that should receive this event."""
        handlers = []

        # Type-specific handlers
        handlers.extend(self._handlers_by_type.get(event.event_type, []))

        # Pattern-matched handlers
        for pattern, subs in self._pattern_handlers.items():
            if fnmatch.fnmatch(event.event_type, pattern):
                handlers.extend(subs)

        # Global handlers
        handlers.extend(self._global_handlers)

        # Filter by priority and source
        filtered = []
        for sub in handlers:
            # Priority filter
            if sub.priority_filter is not None and event.priority > sub.priority_filter:
                continue

            # Source filter
            if sub.source_filter is not None and event.source != sub.source_filter:
                continue

            filtered.append(sub)

        return filtered

    async def _invoke_handler(self, subscription: Subscription, event: Event):
        """Invoke a handler with retry logic."""
        for attempt in range(self.max_retries):
            try:
                result = subscription.handler(event)
                if asyncio.iscoroutine(result):
                    await result

                self._metrics["handlers_invoked"] += 1
                return

            except Exception as e:
                logger.warning(
                    f"Handler {subscription.subscription_id} failed (attempt {attempt + 1}): {e}"
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay_seconds)
                else:
                    # Send to dead letter queue
                    self._add_to_dlq(event, e)
                    self._metrics["events_failed"] += 1

    def _add_to_dlq(self, event: Event, error: Exception):
        """Add failed event to dead letter queue."""
        self._dlq.append((event, error))

        # Trim DLQ if too large
        if len(self._dlq) > self._dlq_max_size:
            self._dlq = self._dlq[-self._dlq_max_size:]

        logger.error(f"Event {event.event_id} sent to DLQ: {error}")

    async def _cleanup_dedup_cache(self):
        """Periodically clean up expired dedup entries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Clean every minute

                now = datetime.utcnow()
                expired = [
                    key for key, expiry in self._dedup_cache.items()
                    if expiry < now
                ]

                for key in expired:
                    del self._dedup_cache[key]

                if expired:
                    logger.debug(f"Cleaned {len(expired)} expired dedup entries")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dedup cleanup: {e}")

    def get_metrics(self) -> dict:
        """Get event bus metrics."""
        return {
            **self._metrics,
            "queue_size": self._queue.qsize(),
            "subscriptions": len(self._subscriptions),
            "dlq_size": len(self._dlq),
            "dedup_cache_size": len(self._dedup_cache)
        }

    def health_report(self) -> dict:
        """
        Return a health report compatible with MetaLoop monitoring.

        Provides a snapshot of bus state for the ONI dashboard
        and MetaLoop health checks.
        """
        metrics = self.get_metrics()
        return {
            "subscription_count": len(self._subscriptions),
            "pattern_subscription_count": sum(
                len(v) for v in self._pattern_handlers.values()
            ),
            "dispatch_rate": metrics["events_processed"],
            "dlq_size": metrics["dlq_size"],
            "queue_size": metrics["queue_size"],
            "dedup_cache_size": metrics["dedup_cache_size"],
        }

    def get_dlq(self, limit: int = 100) -> list[dict]:
        """Get recent dead letter queue entries."""
        entries = []
        for event, error in self._dlq[-limit:]:
            entries.append({
                "event": event.to_dict(),
                "error": str(error),
                "error_type": type(error).__name__
            })
        return entries

    async def replay_from_dlq(self, event_id: str) -> bool:
        """Replay a specific event from the DLQ."""
        for i, (event, _) in enumerate(self._dlq):
            if event.event_id == event_id:
                # Remove from DLQ
                self._dlq.pop(i)
                # Republish
                return await self.publish(event)
        return False

    @property
    def is_running(self) -> bool:
        """Check if the event bus is running."""
        return self._running


# Convenience function for creating events
def create_event(
    event_type: str,
    source: str,
    payload: Optional[dict] = None,
    priority: EventPriority = EventPriority.MEDIUM,
    **metadata
) -> Event:
    """
    Create a new event.

    Args:
        event_type: Type of event (e.g., "http_check", "log_pattern")
        source: Source identifier (e.g., "http_monitor:website-health")
        payload: Event data
        priority: Event priority
        **metadata: Additional metadata

    Returns:
        New Event instance
    """
    return Event(
        event_type=event_type,
        source=source,
        payload=payload or {},
        priority=priority,
        metadata=metadata
    )


# ── Module-level singleton ──────────────────────────────────────────────
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Return the shared EventBus singleton.

    Creates the instance on first call.  Every caller that needs the
    *shared* bus (bootstrap, emit helpers, loops) should use this
    instead of ``EventBus()``.
    """
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
