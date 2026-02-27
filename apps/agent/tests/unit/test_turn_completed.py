"""Test that run_turn emits turn.completed event."""
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

try:
    from ag3nt_agent.deepagents_runtime import _emit_turn_completed
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False

pytestmark = pytest.mark.skipif(
    not _HAS_RUNTIME,
    reason="deepagents_runtime requires langchain (not installed)",
)


def test_emit_turn_completed_creates_event():
    """Verify _emit_turn_completed creates and publishes the right event."""
    mock_bus = MagicMock()
    mock_bus.publish = AsyncMock(return_value=True)

    with patch("ag3nt_agent.deepagents_runtime.EventBus", return_value=mock_bus):
        _emit_turn_completed(session_id="test-123", char_count=5000)

    mock_bus.publish.assert_called_once()
    event = mock_bus.publish.call_args[0][0]
    assert event.event_type == "turn.completed"
    assert event.source == "deepagents_runtime"
    assert event.payload["session_id"] == "test-123"
    assert event.payload["char_count"] == 5000


def test_emit_turn_completed_swallows_errors():
    """Verify _emit_turn_completed doesn't raise on failure."""
    with patch("ag3nt_agent.deepagents_runtime.EventBus", side_effect=RuntimeError("boom")):
        # Should not raise
        _emit_turn_completed(session_id="test-123", char_count=100)
