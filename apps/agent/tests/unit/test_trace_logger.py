"""Tests for the lightweight local trace logger."""

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from ag3nt_agent.trace_logger import (
    LocalTraceLogger,
    TraceEntry,
    get_trace_logger,
    is_langsmith_configured,
)


class TestLocalTraceLoggerInit:
    """Tests for LocalTraceLogger initialization."""

    def test_creates_traces_directory(self, tmp_path):
        """__init__ creates the traces directory if it does not exist."""
        traces_dir = tmp_path / "traces"
        assert not traces_dir.exists()
        LocalTraceLogger(traces_dir=traces_dir)
        assert traces_dir.is_dir()

    def test_uses_existing_directory(self, tmp_path):
        """__init__ works with an already-existing directory."""
        traces_dir = tmp_path / "traces"
        traces_dir.mkdir()
        logger = LocalTraceLogger(traces_dir=traces_dir)
        assert logger._dir == traces_dir

    def test_creates_nested_directory(self, tmp_path):
        """__init__ creates nested parent directories."""
        traces_dir = tmp_path / "a" / "b" / "c" / "traces"
        LocalTraceLogger(traces_dir=traces_dir)
        assert traces_dir.is_dir()


class TestLogToolCall:
    """Tests for log_tool_call method."""

    def test_writes_valid_jsonl(self, tmp_path):
        """log_tool_call writes a valid JSONL entry."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call("exec_tool", duration_ms=123.456)

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        assert path.exists()

        content = path.read_text(encoding="utf-8").strip()
        entry = json.loads(content)
        assert entry["tool"] == "exec_tool"
        assert entry["duration_ms"] == 123.46  # rounded to 2 decimals
        assert entry["input_tokens"] == 0
        assert entry["output_tokens"] == 0
        assert entry["error"] is None
        assert "ts" in entry

    def test_records_token_counts(self, tmp_path):
        """log_tool_call records input and output token counts."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call(
            "search_tool",
            duration_ms=50.0,
            input_tokens=100,
            output_tokens=200,
        )

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["input_tokens"] == 100
        assert entry["output_tokens"] == 200

    def test_records_error(self, tmp_path):
        """log_tool_call records an error string."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call("bad_tool", duration_ms=10.0, error="Something broke")

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["error"] == "Something broke"

    def test_appends_multiple_entries(self, tmp_path):
        """log_tool_call appends entries to the same file."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call("tool_a", duration_ms=10.0)
        logger.log_tool_call("tool_b", duration_ms=20.0)
        logger.log_tool_call("tool_c", duration_ms=30.0)

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

        tools = [json.loads(line)["tool"] for line in lines]
        assert tools == ["tool_a", "tool_b", "tool_c"]

    def test_timestamp_is_utc_iso(self, tmp_path):
        """log_tool_call records an ISO-format UTC timestamp."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call("ts_tool", duration_ms=1.0)

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        ts = entry["ts"]
        # Should be parseable and contain timezone info
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_duration_rounded_to_two_decimals(self, tmp_path):
        """log_tool_call rounds duration_ms to 2 decimal places."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call("round_tool", duration_ms=1.23456789)

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["duration_ms"] == 1.23


class TestLogModelCall:
    """Tests for log_model_call method."""

    def test_prefixes_with_llm(self, tmp_path):
        """log_model_call prefixes the model name with 'llm:'."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_model_call("gpt-4", duration_ms=500.0, input_tokens=50, output_tokens=100)

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["tool"] == "llm:gpt-4"
        assert entry["input_tokens"] == 50
        assert entry["output_tokens"] == 100

    def test_model_call_with_error(self, tmp_path):
        """log_model_call records errors for model calls."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_model_call("claude-3", duration_ms=100.0, error="rate limited")

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["tool"] == "llm:claude-3"
        assert entry["error"] == "rate limited"


class TestReadTraces:
    """Tests for read_traces method."""

    def test_reads_written_entries(self, tmp_path):
        """read_traces returns entries that were written."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call("tool_x", duration_ms=10.0)
        logger.log_tool_call("tool_y", duration_ms=20.0)

        entries = logger.read_traces()
        assert len(entries) == 2
        assert entries[0]["tool"] == "tool_x"
        assert entries[1]["tool"] == "tool_y"

    def test_respects_limit(self, tmp_path):
        """read_traces respects the limit parameter."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        for i in range(20):
            logger.log_tool_call(f"tool_{i}", duration_ms=float(i))

        entries = logger.read_traces(limit=5)
        assert len(entries) == 5
        assert entries[0]["tool"] == "tool_0"
        assert entries[4]["tool"] == "tool_4"

    def test_returns_empty_for_missing_day(self, tmp_path):
        """read_traces returns empty list for a day with no traces."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        entries = logger.read_traces(day=date(2020, 1, 1))
        assert entries == []

    def test_reads_specific_day(self, tmp_path):
        """read_traces can read traces for a specific date."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        # Manually write a JSONL file for a specific date
        target_date = date(2025, 6, 15)
        path = tmp_path / f"{target_date.isoformat()}.jsonl"
        entry = {"ts": "2025-06-15T12:00:00+00:00", "tool": "old_tool", "duration_ms": 5.0,
                 "input_tokens": 0, "output_tokens": 0, "error": None, "session_id": ""}
        path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        entries = logger.read_traces(day=target_date)
        assert len(entries) == 1
        assert entries[0]["tool"] == "old_tool"

    def test_returns_empty_for_empty_file(self, tmp_path):
        """read_traces returns empty list for an empty JSONL file."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        path.write_text("", encoding="utf-8")

        entries = logger.read_traces()
        assert entries == []

    def test_handles_corrupt_json_gracefully(self, tmp_path):
        """read_traces handles corrupt JSON without crashing."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        path.write_text("not valid json\n", encoding="utf-8")

        # Should not raise, just return empty or partial
        entries = logger.read_traces()
        # Implementation catches JSONDecodeError, returns empty
        assert isinstance(entries, list)


class TestSetSessionId:
    """Tests for set_session_id method."""

    def test_session_id_included_in_entries(self, tmp_path):
        """set_session_id makes subsequent entries include the session id."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.set_session_id("session-abc-123")
        logger.log_tool_call("tool_s", duration_ms=5.0)

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["session_id"] == "session-abc-123"

    def test_default_session_id_is_empty(self, tmp_path):
        """Without set_session_id, session_id defaults to empty string."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.log_tool_call("tool_t", duration_ms=5.0)

        path = tmp_path / f"{date.today().isoformat()}.jsonl"
        entry = json.loads(path.read_text(encoding="utf-8").strip())
        assert entry["session_id"] == ""

    def test_session_id_can_be_changed(self, tmp_path):
        """set_session_id can be called multiple times."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        logger.set_session_id("session-1")
        logger.log_tool_call("tool_1", duration_ms=1.0)
        logger.set_session_id("session-2")
        logger.log_tool_call("tool_2", duration_ms=2.0)

        entries = logger.read_traces()
        assert entries[0]["session_id"] == "session-1"
        assert entries[1]["session_id"] == "session-2"


class TestIsLangsmithConfigured:
    """Tests for is_langsmith_configured function."""

    def test_returns_true_when_key_set(self):
        """is_langsmith_configured returns True when LANGSMITH_API_KEY is set."""
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": "lsv2_abc123"}):
            assert is_langsmith_configured() is True

    def test_returns_false_when_key_missing(self):
        """is_langsmith_configured returns False when LANGSMITH_API_KEY is not set."""
        env = os.environ.copy()
        env.pop("LANGSMITH_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            assert is_langsmith_configured() is False

    def test_returns_false_when_key_empty(self):
        """is_langsmith_configured returns False when LANGSMITH_API_KEY is empty."""
        with patch.dict(os.environ, {"LANGSMITH_API_KEY": ""}):
            assert is_langsmith_configured() is False


class TestGetTraceLogger:
    """Tests for get_trace_logger singleton."""

    def test_returns_singleton(self):
        """get_trace_logger returns the same instance on repeated calls."""
        import ag3nt_agent.trace_logger as mod
        # Reset singleton
        mod._instance = None
        try:
            logger1 = get_trace_logger()
            logger2 = get_trace_logger()
            assert logger1 is logger2
        finally:
            mod._instance = None

    def test_returns_local_trace_logger_instance(self):
        """get_trace_logger returns a LocalTraceLogger instance."""
        import ag3nt_agent.trace_logger as mod
        mod._instance = None
        try:
            logger = get_trace_logger()
            assert isinstance(logger, LocalTraceLogger)
        finally:
            mod._instance = None


class TestGracefulErrorHandling:
    """Tests for graceful handling of write errors."""

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not effective on Windows")
    def test_log_tool_call_handles_readonly_dir(self, tmp_path):
        """log_tool_call does not raise on write errors (read-only dir)."""
        traces_dir = tmp_path / "readonly_traces"
        traces_dir.mkdir()
        logger = LocalTraceLogger(traces_dir=traces_dir)
        # Make directory read-only
        traces_dir.chmod(0o444)
        try:
            # Should not raise
            logger.log_tool_call("fail_tool", duration_ms=1.0)
        finally:
            traces_dir.chmod(0o755)

    def test_log_tool_call_handles_write_error_via_mock(self, tmp_path):
        """log_tool_call silently handles OSError when writing."""
        logger = LocalTraceLogger(traces_dir=tmp_path)
        with patch("builtins.open", side_effect=OSError("disk full")):
            # Should not raise
            logger.log_tool_call("fail_tool", duration_ms=1.0)


class TestTraceEntry:
    """Tests for the TraceEntry dataclass."""

    def test_creation(self):
        """TraceEntry can be created with all fields."""
        entry = TraceEntry(
            ts="2025-01-01T00:00:00+00:00",
            tool="test_tool",
            duration_ms=42.5,
            input_tokens=10,
            output_tokens=20,
            error="some error",
            session_id="sess-1",
        )
        assert entry.ts == "2025-01-01T00:00:00+00:00"
        assert entry.tool == "test_tool"
        assert entry.duration_ms == 42.5
        assert entry.input_tokens == 10
        assert entry.output_tokens == 20
        assert entry.error == "some error"
        assert entry.session_id == "sess-1"

    def test_defaults(self):
        """TraceEntry has sensible defaults."""
        entry = TraceEntry(ts="2025-01-01T00:00:00+00:00", tool="t", duration_ms=0.0)
        assert entry.input_tokens == 0
        assert entry.output_tokens == 0
        assert entry.error is None
        assert entry.session_id == ""
