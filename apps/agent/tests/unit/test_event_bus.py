"""
Tests for Event Bus.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock

from ag3nt_agent.autonomous.event_bus import (
    Event,
    EventBus,
    EventPriority,
    create_event,
)


class TestEvent:
    """Tests for Event dataclass."""

    def test_create_event_basic(self):
        """Test basic event creation."""
        event = Event(
            event_type="test_event",
            source="test_source",
            payload={"key": "value"}
        )

        assert event.event_type == "test_event"
        assert event.source == "test_source"
        assert event.payload == {"key": "value"}
        assert event.priority == EventPriority.MEDIUM
        assert event.event_id is not None
        assert event.dedup_key is not None

    def test_create_event_with_priority(self):
        """Test event creation with custom priority."""
        event = Event(
            event_type="critical_event",
            source="monitor",
            priority=EventPriority.CRITICAL
        )

        assert event.priority == EventPriority.CRITICAL

    def test_dedup_key_generation(self):
        """Test dedup key is consistent for same content."""
        event1 = Event(
            event_type="test",
            source="src",
            payload={"a": 1}
        )
        event2 = Event(
            event_type="test",
            source="src",
            payload={"a": 1}
        )

        assert event1.dedup_key == event2.dedup_key

    def test_dedup_key_different_for_different_content(self):
        """Test dedup key differs for different content."""
        event1 = Event(
            event_type="test",
            source="src",
            payload={"a": 1}
        )
        event2 = Event(
            event_type="test",
            source="src",
            payload={"a": 2}
        )

        assert event1.dedup_key != event2.dedup_key

    def test_to_dict(self):
        """Test event serialization."""
        event = Event(
            event_type="test",
            source="src",
            payload={"key": "value"}
        )

        data = event.to_dict()

        assert data["event_type"] == "test"
        assert data["source"] == "src"
        assert data["payload"] == {"key": "value"}
        assert "event_id" in data
        assert "timestamp" in data

    def test_from_dict(self):
        """Test event deserialization."""
        data = {
            "event_type": "test",
            "source": "src",
            "payload": {"key": "value"},
            "priority": "HIGH",
            "timestamp": datetime.utcnow().isoformat()
        }

        event = Event.from_dict(data)

        assert event.event_type == "test"
        assert event.source == "src"
        assert event.priority == EventPriority.HIGH


class TestEventBus:
    """Tests for EventBus."""

    @pytest.fixture
    def event_bus(self):
        """Create a test event bus."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_start_stop(self, event_bus):
        """Test starting and stopping the event bus."""
        await event_bus.start()
        assert event_bus.is_running

        await event_bus.stop()
        assert not event_bus.is_running

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus):
        """Test subscribing to and publishing events."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe(handler)
        await event_bus.start()

        event = Event(event_type="test", source="src")
        await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].event_type == "test"

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_subscribe_with_type_filter(self, event_bus):
        """Test subscribing to specific event types."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe(handler, event_types={"type_a"})
        await event_bus.start()

        # Publish matching event
        await event_bus.publish(Event(event_type="type_a", source="src"))

        # Publish non-matching event
        await event_bus.publish(Event(event_type="type_b", source="src"))

        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].event_type == "type_a"

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing handlers."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        sub_id = event_bus.subscribe(handler)
        await event_bus.start()

        # Publish first event
        await event_bus.publish(Event(event_type="test", source="src"))
        await asyncio.sleep(0.1)

        # Unsubscribe
        result = event_bus.unsubscribe(sub_id)
        assert result is True

        # Publish second event
        await event_bus.publish(Event(event_type="test2", source="src"))
        await asyncio.sleep(0.1)

        # Should only have first event
        assert len(received_events) == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_deduplication(self, event_bus):
        """Test event deduplication."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe(handler)
        await event_bus.start()

        # Publish same event twice
        event = Event(event_type="test", source="src", payload={"key": "value"})
        await event_bus.publish(event)

        event2 = Event(event_type="test", source="src", payload={"key": "value"})
        result = await event_bus.publish(event2)

        await asyncio.sleep(0.1)

        # Second publish should be deduplicated
        assert result is False
        assert len(received_events) == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_priority_ordering(self, event_bus):
        """Test that higher priority events are processed first."""
        received_events = []

        async def handler(event):
            received_events.append(event.priority)

        event_bus.subscribe(handler)

        # Publish events in reverse priority order
        await event_bus.publish(Event(event_type="low", source="src", priority=EventPriority.LOW))
        await event_bus.publish(Event(event_type="critical", source="src", priority=EventPriority.CRITICAL))
        await event_bus.publish(Event(event_type="medium", source="src", priority=EventPriority.MEDIUM))

        await event_bus.start()
        await asyncio.sleep(0.2)
        await event_bus.stop()

        # Critical should be first
        assert received_events[0] == EventPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, event_bus):
        """Test failed events go to DLQ."""
        async def failing_handler(event):
            raise ValueError("Handler failed")

        # Create event bus with shorter retry delay for faster tests
        fast_bus = EventBus(max_retries=2, retry_delay_seconds=0.1)
        fast_bus.subscribe(failing_handler)
        await fast_bus.start()

        await fast_bus.publish(Event(event_type="test", source="src"))
        await asyncio.sleep(0.5)  # Wait for retries (2 retries * 0.1s + processing)

        dlq = fast_bus.get_dlq()
        assert len(dlq) == 1
        assert "Handler failed" in dlq[0]["error"]

        await fast_bus.stop()

    @pytest.mark.asyncio
    async def test_get_metrics(self, event_bus):
        """Test metrics collection."""
        received = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe(handler)
        await event_bus.start()

        await event_bus.publish(Event(event_type="test", source="src"))
        await asyncio.sleep(0.1)

        metrics = event_bus.get_metrics()

        assert metrics["events_received"] == 1
        assert metrics["events_processed"] == 1
        assert metrics["subscriptions"] == 1

        await event_bus.stop()


class TestPatternSubscriptions:
    """Tests for subscribe_pattern() — fnmatch-style glob routing."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_pattern_matches(self, event_bus):
        """Pattern 'file.*' should match file.written and file.edited."""
        received = []

        async def handler(event):
            received.append(event.event_type)

        event_bus.subscribe_pattern("file.*", handler)
        await event_bus.start()

        await event_bus.publish(Event(event_type="file.written", source="test"))
        await event_bus.publish(Event(event_type="file.edited", source="test"))
        await event_bus.publish(Event(event_type="tool.called", source="test"))

        await asyncio.sleep(0.2)
        await event_bus.stop()

        assert "file.written" in received
        assert "file.edited" in received
        assert "tool.called" not in received

    @pytest.mark.asyncio
    async def test_pattern_star_matches_all(self, event_bus):
        """Pattern '*' should match all event types."""
        received = []

        async def handler(event):
            received.append(event.event_type)

        event_bus.subscribe_pattern("*", handler)
        await event_bus.start()

        await event_bus.publish(Event(event_type="any.event", source="test"))
        await asyncio.sleep(0.1)
        await event_bus.stop()

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_pattern_unsubscribe(self, event_bus):
        """Unsubscribing a pattern handler should stop delivery."""
        received = []

        async def handler(event):
            received.append(event)

        sub_id = event_bus.subscribe_pattern("test.*", handler)
        await event_bus.start()

        await event_bus.publish(Event(event_type="test.a", source="s"))
        await asyncio.sleep(0.1)
        assert len(received) == 1

        event_bus.unsubscribe(sub_id)
        await event_bus.publish(Event(event_type="test.b", source="s"))
        await asyncio.sleep(0.1)
        assert len(received) == 1

        await event_bus.stop()


class TestEmitSync:
    """Tests for emit_sync() — fire-and-forget from sync contexts."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.mark.asyncio
    async def test_emit_sync_basic(self, event_bus):
        """emit_sync should enqueue an event that gets processed."""
        received = []

        async def handler(event):
            received.append(event)

        event_bus.subscribe(handler)
        await event_bus.start()

        event = Event(event_type="sync.test", source="test")
        event_bus.emit_sync(event)

        await asyncio.sleep(0.1)
        await event_bus.stop()

        assert len(received) == 1
        assert received[0].event_type == "sync.test"

    @pytest.mark.asyncio
    async def test_emit_sync_updates_metrics(self, event_bus):
        """emit_sync should increment events_received."""
        event_bus.emit_sync(Event(event_type="metric.test", source="test"))
        metrics = event_bus.get_metrics()
        assert metrics["events_received"] == 1

    def test_emit_sync_queue_full(self):
        """emit_sync on a full queue should not raise."""
        bus = EventBus(max_queue_size=1)
        bus.emit_sync(Event(event_type="a", source="s"))
        # Should not raise, just log warning
        bus.emit_sync(Event(event_type="b", source="s"))


class TestHealthReport:
    """Tests for health_report() — MetaLoop-compatible status."""

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    def test_health_report_structure(self, event_bus):
        report = event_bus.health_report()
        assert "subscription_count" in report
        assert "pattern_subscription_count" in report
        assert "dispatch_rate" in report
        assert "dlq_size" in report
        assert "queue_size" in report
        assert "dedup_cache_size" in report

    def test_health_report_counts_subscriptions(self, event_bus):
        event_bus.subscribe(lambda e: None)
        event_bus.subscribe(lambda e: None, event_types={"foo"})
        report = event_bus.health_report()
        assert report["subscription_count"] == 2

    def test_health_report_counts_pattern_subs(self, event_bus):
        event_bus.subscribe_pattern("file.*", lambda e: None)
        event_bus.subscribe_pattern("tool.*", lambda e: None)
        report = event_bus.health_report()
        assert report["pattern_subscription_count"] == 2


class TestDLQReplay:
    """Tests for dead letter queue replay."""

    @pytest.mark.asyncio
    async def test_replay_from_dlq(self):
        """Replaying a DLQ event should requeue it."""
        # Use a short dedup window so it expires before replay
        bus = EventBus(max_retries=1, retry_delay_seconds=0.01, dedup_window_seconds=1)

        async def failing_handler(event):
            raise ValueError("always fails")

        bus.subscribe(failing_handler)
        await bus.start()

        event = Event(event_type="replay.test", source="test", dedup_window_seconds=1)
        await bus.publish(event)
        await asyncio.sleep(0.3)

        # Should be in DLQ
        dlq = bus.get_dlq()
        assert len(dlq) == 1
        event_id = dlq[0]["event"]["event_id"]

        # Clear dedup cache so replay is accepted
        bus._dedup_cache.clear()

        replayed = await bus.replay_from_dlq(event_id)
        assert replayed is True

        await asyncio.sleep(0.2)
        await bus.stop()

    @pytest.mark.asyncio
    async def test_replay_nonexistent(self):
        bus = EventBus()
        result = await bus.replay_from_dlq("nonexistent-id")
        assert result is False


class TestCreateEvent:
    """Tests for create_event helper."""

    def test_create_event_simple(self):
        """Test simple event creation."""
        event = create_event("test_type", "test_source")

        assert event.event_type == "test_type"
        assert event.source == "test_source"
        assert event.priority == EventPriority.MEDIUM

    def test_create_event_with_all_params(self):
        """Test event creation with all parameters."""
        event = create_event(
            event_type="http_check",
            source="monitor",
            payload={"status": 500},
            priority=EventPriority.HIGH,
            custom_field="value"
        )

        assert event.event_type == "http_check"
        assert event.payload == {"status": 500}
        assert event.priority == EventPriority.HIGH
        assert event.metadata["custom_field"] == "value"


class TestGetEventBus:
    """Tests for get_event_bus() singleton getter."""

    def setup_method(self):
        """Reset singleton between tests."""
        import ag3nt_agent.autonomous.event_bus as mod
        mod._event_bus = None

    def test_returns_event_bus_instance(self):
        from ag3nt_agent.autonomous.event_bus import get_event_bus
        bus = get_event_bus()
        assert isinstance(bus, EventBus)

    def test_returns_same_instance(self):
        from ag3nt_agent.autonomous.event_bus import get_event_bus
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2

    def test_reset_creates_new_instance(self):
        from ag3nt_agent.autonomous.event_bus import get_event_bus
        import ag3nt_agent.autonomous.event_bus as mod
        bus1 = get_event_bus()
        mod._event_bus = None
        bus2 = get_event_bus()
        assert bus1 is not bus2
