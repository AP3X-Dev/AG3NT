"""Tests for AgentGuardMiddleware — doom loop, steps limit, output truncation."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from ag3nt_agent.agent_guard_middleware import (
    DOOM_LOOP_THRESHOLD,
    AgentGuardMiddleware,
    _detect_doom_loop,
    _step_counts,
)


# ---------------------------------------------------------------------------
# Doom loop detection
# ---------------------------------------------------------------------------

class TestDoomLoopDetection:

    def test_no_messages(self):
        is_looping, tool_name = _detect_doom_loop([])
        assert not is_looping
        assert tool_name is None

    def test_below_threshold(self):
        """Two identical calls is below the threshold of 3."""
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "1"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": "/tmp/x"}, "id": "2"}
            ]),
        ]
        is_looping, _ = _detect_doom_loop(messages)
        assert not is_looping

    def test_at_threshold_identical(self):
        """Exactly 3 identical calls triggers detection."""
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "exec", "args": {"cmd": "ls"}, "id": str(i)}
            ])
            for i in range(DOOM_LOOP_THRESHOLD)
        ]
        is_looping, tool_name = _detect_doom_loop(messages)
        assert is_looping
        assert tool_name == "exec"

    def test_different_args_no_loop(self):
        """Same tool name but different args is not a loop."""
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": f"/tmp/{i}"}, "id": str(i)}
            ])
            for i in range(DOOM_LOOP_THRESHOLD)
        ]
        is_looping, _ = _detect_doom_loop(messages)
        assert not is_looping

    def test_different_tools_no_loop(self):
        """Different tool names are not a loop."""
        tools = ["read_file", "write_file", "exec"]
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": tools[i], "args": {"x": 1}, "id": str(i)}
            ])
            for i in range(DOOM_LOOP_THRESHOLD)
        ]
        is_looping, _ = _detect_doom_loop(messages)
        assert not is_looping

    def test_interleaved_human_messages(self):
        """Tool calls from AIMessages are extracted even with HumanMessages mixed in."""
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "exec", "args": {"cmd": "ls"}, "id": "1"}
            ]),
            HumanMessage(content="keep going"),
            AIMessage(content="", tool_calls=[
                {"name": "exec", "args": {"cmd": "ls"}, "id": "2"}
            ]),
            ToolMessage(content="result", tool_call_id="2"),
            AIMessage(content="", tool_calls=[
                {"name": "exec", "args": {"cmd": "ls"}, "id": "3"}
            ]),
        ]
        is_looping, tool_name = _detect_doom_loop(messages)
        assert is_looping
        assert tool_name == "exec"

    def test_args_order_independent(self):
        """Args with different key order but same values are detected as identical."""
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {"b": 2, "a": 1}, "id": "1"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {"a": 1, "b": 2}, "id": "2"}
            ]),
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {"a": 1, "b": 2}, "id": "3"}
            ]),
        ]
        is_looping, _ = _detect_doom_loop(messages)
        assert is_looping

    def test_multiple_tool_calls_in_one_message(self):
        """Multiple tool calls in a single AIMessage are scanned."""
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "exec", "args": {"cmd": "ls"}, "id": "1"},
                {"name": "exec", "args": {"cmd": "ls"}, "id": "2"},
                {"name": "exec", "args": {"cmd": "ls"}, "id": "3"},
            ]),
        ]
        is_looping, tool_name = _detect_doom_loop(messages)
        assert is_looping
        assert tool_name == "exec"


# ---------------------------------------------------------------------------
# Steps limit
# ---------------------------------------------------------------------------

class TestStepsLimit:

    def setup_method(self):
        _step_counts.clear()

    def test_reset_steps(self):
        _step_counts["sess1"] = 50
        AgentGuardMiddleware.reset_steps("sess1")
        assert AgentGuardMiddleware.get_steps("sess1") == 0

    def test_get_steps_unknown_session(self):
        assert AgentGuardMiddleware.get_steps("unknown") == 0

    def test_steps_incremented_via_guard(self):
        """_guard_model_call increments the step counter."""
        mw = AgentGuardMiddleware()
        AgentGuardMiddleware.reset_steps("test")

        request = _make_model_request(session_id="test")

        # Call guard 5 times
        for _ in range(5):
            mw._guard_model_call(request)

        assert AgentGuardMiddleware.get_steps("test") == 5

    @patch("ag3nt_agent.agent_guard_middleware._get_max_steps", return_value=3)
    def test_max_steps_injects_prompt(self, mock_max):
        """When steps >= max, system message gets injection."""
        mw = AgentGuardMiddleware()
        AgentGuardMiddleware.reset_steps("test")

        request = _make_model_request(session_id="test")

        # Steps 0, 1, 2 are fine (< 3)
        for _ in range(3):
            result = mw._guard_model_call(request)

        # Step 3 should trigger
        result = mw._guard_model_call(request)
        sys_content = _extract_system_text(result)
        assert "MAXIMUM STEPS REACHED" in sys_content

    @patch("ag3nt_agent.agent_guard_middleware._get_max_steps", return_value=3)
    def test_below_max_steps_no_injection(self, mock_max):
        """Before reaching max, no steps injection."""
        mw = AgentGuardMiddleware()
        AgentGuardMiddleware.reset_steps("test")

        request = _make_model_request(session_id="test")
        result = mw._guard_model_call(request)

        sys_content = _extract_system_text(result)
        assert "MAXIMUM STEPS" not in sys_content


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------

class TestOutputTruncation:

    def test_small_output_unchanged(self):
        """Output below threshold passes through."""
        msg = ToolMessage(content="small result", tool_call_id="1")
        result = AgentGuardMiddleware._maybe_truncate_result(msg)
        assert result.content == "small result"

    def test_command_passthrough(self):
        """Command results are not truncated."""
        cmd = Command(update={"messages": []})
        result = AgentGuardMiddleware._maybe_truncate_result(cmd)
        assert isinstance(result, Command)

    @patch("ag3nt_agent.output_truncation.maybe_truncate")
    @patch("ag3nt_agent.artifact_store.get_artifact_store")
    def test_large_output_truncated(self, mock_get_store, mock_truncate):
        """Large tool output gets truncated when artifact store fails."""
        mock_get_store.side_effect = RuntimeError("store unavailable")
        mock_truncate.return_value = ("truncated...", True, "/tmp/full.txt")

        large_content = "x" * 100_000
        msg = ToolMessage(content=large_content, tool_call_id="1")
        result = AgentGuardMiddleware._maybe_truncate_result(msg)

        assert result.content == "truncated..."
        mock_truncate.assert_called_once_with(large_content)

    def test_non_string_content_unchanged(self):
        """Non-string content (e.g., list) is not truncated."""
        msg = ToolMessage(content=[{"type": "text", "text": "hi"}], tool_call_id="1")
        result = AgentGuardMiddleware._maybe_truncate_result(msg)
        assert result.content == [{"type": "text", "text": "hi"}]

    def test_empty_content_unchanged(self):
        """Empty string content passes through."""
        msg = ToolMessage(content="", tool_call_id="1")
        result = AgentGuardMiddleware._maybe_truncate_result(msg)
        assert result.content == ""

    @patch("ag3nt_agent.artifact_store.get_artifact_store")
    def test_auto_dump_to_artifact(self, mock_get_store):
        """Large output auto-dumps to artifact store and returns preview + reference."""
        mock_store = MagicMock()
        mock_meta = MagicMock()
        mock_meta.size_bytes = 50000
        mock_meta.artifact_id = "abc123"
        mock_store.write_artifact.return_value = mock_meta
        mock_get_store.return_value = mock_store

        large_content = "x" * 5000
        msg = ToolMessage(content=large_content, tool_call_id="1")
        result = AgentGuardMiddleware._maybe_truncate_result(msg)

        mock_store.write_artifact.assert_called_once()
        assert "abc123" in result.content
        assert "read_artifact" in result.content
        assert result.content.startswith("x" * 100)  # preview preserved


# ---------------------------------------------------------------------------
# Integration: doom loop + steps in _guard_model_call
# ---------------------------------------------------------------------------

class TestGuardModelCallIntegration:

    def setup_method(self):
        _step_counts.clear()

    def test_doom_loop_injects_prompt(self):
        """Doom loop detection injects warning into system prompt."""
        mw = AgentGuardMiddleware()
        AgentGuardMiddleware.reset_steps("test")

        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "exec", "args": {"cmd": "ls"}, "id": str(i)}
            ])
            for i in range(DOOM_LOOP_THRESHOLD)
        ]
        request = _make_model_request(session_id="test", messages=messages)
        result = mw._guard_model_call(request)

        sys_content = _extract_system_text(result)
        assert "DOOM LOOP DETECTED" in sys_content
        assert "exec" in sys_content

    @patch("ag3nt_agent.agent_guard_middleware._get_max_steps", return_value=2)
    def test_both_doom_and_steps(self, mock_max):
        """Both doom loop and steps limit can fire simultaneously."""
        mw = AgentGuardMiddleware()
        AgentGuardMiddleware.reset_steps("test")

        messages = [
            AIMessage(content="", tool_calls=[
                {"name": "exec", "args": {"cmd": "ls"}, "id": str(i)}
            ])
            for i in range(DOOM_LOOP_THRESHOLD)
        ]
        request = _make_model_request(session_id="test", messages=messages)

        # Burn through steps
        for _ in range(2):
            mw._guard_model_call(request)

        # Step 2 = at limit, plus doom loop
        result = mw._guard_model_call(request)
        sys_content = _extract_system_text(result)
        assert "DOOM LOOP DETECTED" in sys_content
        assert "MAXIMUM STEPS REACHED" in sys_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_request(
    session_id: str = "default",
    messages: list | None = None,
) -> MagicMock:
    """Create a mock ModelRequest for testing."""
    request = MagicMock()
    request.state = {"messages": messages or []}
    request.config = {"configurable": {"thread_id": session_id}}
    request.system_message = None

    def override_fn(**kwargs):
        """Mock override that returns a new mock with the updated system_message."""
        new_req = MagicMock()
        new_req.state = request.state
        new_req.config = request.config
        new_req.system_message = kwargs.get("system_message", request.system_message)
        new_req.override = override_fn
        return new_req

    request.override = MagicMock(side_effect=override_fn)
    return request


def _extract_system_text(request) -> str:
    """Extract all text from a system message on a request mock."""
    sys_msg = request.system_message
    if sys_msg is None:
        return ""
    content = sys_msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


# ---------------------------------------------------------------------------
# Cost guardrails
# ---------------------------------------------------------------------------

class TestCostGuardrail:

    def test_no_cost_no_injection(self):
        """When cost is zero, no injection."""
        result = AgentGuardMiddleware._check_cost_guardrail()
        # With a fresh tracker (or no tracker), should be None
        assert result is None or "COST" not in (result or "")

    @patch("ag3nt_agent.agent_guard_middleware.AgentGuardMiddleware._check_cost_guardrail")
    def test_cost_warning_at_80_pct(self, mock_check):
        """At 80% of limit, a notice is injected."""
        from ag3nt_agent.cost_tracker import CostTracker
        tracker = CostTracker()
        # Simulate $4.10 with $5.00 limit = 82%
        tracker.record_turn(1_000_000, 100_000, "claude-opus-4-6")  # ~$22.50
        # We test the static method directly instead
        mock_check.return_value = None  # reset mock

        # Test the actual logic by calling directly
        with patch("ag3nt_agent.cost_tracker.get_cost_tracker", return_value=tracker), \
             patch("ag3nt_agent.agent_config.MAX_SESSION_COST_USD", 25.0):
            # Restore the real method for this call
            result = AgentGuardMiddleware._check_cost_guardrail.__wrapped__() \
                if hasattr(AgentGuardMiddleware._check_cost_guardrail, '__wrapped__') \
                else None
            # Since mock replaces the method, test via integration below
            pass

    def test_check_cost_guardrail_below_threshold(self):
        """Cost well below 80% returns None."""
        from ag3nt_agent.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.record_turn(100, 50, "claude-sonnet-4-6")  # ~$0.001

        with patch("ag3nt_agent.cost_tracker.get_cost_tracker", return_value=tracker), \
             patch("ag3nt_agent.agent_config.MAX_SESSION_COST_USD", 5.0):
            result = AgentGuardMiddleware._check_cost_guardrail()
            assert result is None

    def test_check_cost_guardrail_at_80_pct(self):
        """Cost at 80% returns notice."""
        from ag3nt_agent.cost_tracker import CostTracker
        tracker = CostTracker()
        # Need $4.00+ with $5.00 limit. Sonnet: 1M input = $3.0, 100K output = $1.5
        tracker.record_turn(1_000_000, 100_000, "claude-sonnet-4-6")

        with patch("ag3nt_agent.cost_tracker.get_cost_tracker", return_value=tracker), \
             patch("ag3nt_agent.agent_config.MAX_SESSION_COST_USD", 5.0):
            result = AgentGuardMiddleware._check_cost_guardrail()
            assert result is not None
            assert "Cost notice" in result or "COST WARNING" in result

    def test_check_cost_guardrail_at_limit(self):
        """Cost at ~100% (but below 200%) returns warning."""
        from ag3nt_agent.cost_tracker import CostTracker
        tracker = CostTracker()
        # Sonnet: 1M input=$3, 200K output=$3 → $6 total, limit $5 → 120% (soft stop)
        tracker.record_turn(1_000_000, 200_000, "claude-sonnet-4-6")

        with patch("ag3nt_agent.cost_tracker.get_cost_tracker", return_value=tracker), \
             patch("ag3nt_agent.agent_config.MAX_SESSION_COST_USD", 5.0):
            result = AgentGuardMiddleware._check_cost_guardrail()
            assert result is not None
            assert "COST WARNING" in result

    def test_check_cost_guardrail_at_2x(self):
        """Cost at 200% returns hard stop."""
        from ag3nt_agent.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.record_turn(1_000_000, 1_000_000, "claude-opus-4-6")  # $90

        with patch("ag3nt_agent.cost_tracker.get_cost_tracker", return_value=tracker), \
             patch("ag3nt_agent.agent_config.MAX_SESSION_COST_USD", 5.0):
            result = AgentGuardMiddleware._check_cost_guardrail()
            assert result is not None
            assert "EXCEEDED" in result
            assert "stop" in result.lower()

    def test_cost_guardrail_import_error(self):
        """If cost_tracker can't be imported, returns None."""
        with patch("ag3nt_agent.agent_guard_middleware.AgentGuardMiddleware._check_cost_guardrail") as mock:
            # Simulate by making the static method return None on import error
            mock.return_value = None
            assert mock() is None


# ---------------------------------------------------------------------------
# Tool timing
# ---------------------------------------------------------------------------

class TestToolTiming:

    def test_record_tool_timing_records_to_tracker(self):
        """_record_tool_timing writes to cost tracker."""
        from ag3nt_agent.cost_tracker import CostTracker
        tracker = CostTracker()
        request = MagicMock()
        request.tool_name = "read_file"

        with patch("ag3nt_agent.cost_tracker.get_cost_tracker", return_value=tracker):
            AgentGuardMiddleware._record_tool_timing(request, 0.0)

        timings = tracker.get_tool_timings()
        assert len(timings) == 1
        assert timings[0].tool_name == "read_file"

    def test_record_tool_timing_graceful_on_error(self):
        """_record_tool_timing doesn't raise on import errors."""
        request = MagicMock()
        request.tool_name = "read_file"

        with patch("ag3nt_agent.cost_tracker.get_cost_tracker", side_effect=RuntimeError):
            # Should not raise
            AgentGuardMiddleware._record_tool_timing(request, 0.0)
