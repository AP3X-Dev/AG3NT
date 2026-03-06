"""Tests for context_tools (dump_to_artifact & read_artifact)."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch


def _ensure_mock_module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


# Ensure langchain_core.tools is available
_lc_core = _ensure_mock_module("langchain_core")
_lc_core_tools = _ensure_mock_module("langchain_core.tools")
if not hasattr(_lc_core_tools, "tool"):

    def _mock_tool_decorator(fn):
        fn.invoke = lambda args: fn(**args)
        return fn

    _lc_core_tools.tool = _mock_tool_decorator

from ag3nt_agent.context_tools import (
    dump_to_artifact,
    read_artifact,
    check_context_budget,
    compact_now,
    write_note,
    read_note,
    list_notes,
    get_context_tools,
    _set_notes_dir,
)


class TestDumpToArtifact:
    def test_stores_and_returns_reference(self):
        mock_store = MagicMock()
        mock_store.write_artifact.return_value = MagicMock(
            artifact_id="abc123_ff",
            size_bytes=42,
        )
        with patch("ag3nt_agent.context_tools.get_artifact_store", return_value=mock_store):
            result = dump_to_artifact.invoke({"content": "hello world", "label": "test-label"})

        assert "abc123_ff" in result
        assert "42 bytes" in result
        mock_store.write_artifact.assert_called_once_with(
            content="hello world",
            tool_name="dump_to_artifact",
            tags=["test-label"],
        )

    def test_empty_content_still_stores(self):
        mock_store = MagicMock()
        mock_store.write_artifact.return_value = MagicMock(
            artifact_id="empty_001",
            size_bytes=0,
        )
        with patch("ag3nt_agent.context_tools.get_artifact_store", return_value=mock_store):
            result = dump_to_artifact.invoke({"content": "", "label": "empty"})

        assert "empty_001" in result
        mock_store.write_artifact.assert_called_once_with(
            content="",
            tool_name="dump_to_artifact",
            tags=["empty"],
        )


class TestReadArtifact:
    def test_returns_stored_content(self):
        mock_store = MagicMock()
        mock_store.read_artifact.return_value = "the stored content"
        with patch("ag3nt_agent.context_tools.get_artifact_store", return_value=mock_store):
            result = read_artifact.invoke({"artifact_id": "abc123_ff"})

        assert result == "the stored content"
        mock_store.read_artifact.assert_called_once_with("abc123_ff")

    def test_returns_not_found_for_missing_artifact(self):
        mock_store = MagicMock()
        mock_store.read_artifact.return_value = None
        with patch("ag3nt_agent.context_tools.get_artifact_store", return_value=mock_store):
            result = read_artifact.invoke({"artifact_id": "nonexistent_id"})

        assert "not found" in result.lower()
        assert "nonexistent_id" in result


class TestGetContextTools:
    def test_returns_all_tools(self):
        tools = get_context_tools()
        assert isinstance(tools, list)
        assert len(tools) == 7
        assert dump_to_artifact in tools
        assert read_artifact in tools
        assert check_context_budget in tools
        assert compact_now in tools
        assert write_note in tools
        assert read_note in tools
        assert list_notes in tools


class TestCheckContextBudget:
    def test_returns_budget_when_available(self):
        mock_trigger = MagicMock()
        mock_trigger.usage_ratio.return_value = 0.45
        mock_trigger._config.prune_threshold = 0.60
        mock_trigger._config.extract_threshold = 0.70
        mock_trigger._config.compact_threshold = 0.80
        with patch("ag3nt_agent.context_tools._get_compaction_trigger", return_value=mock_trigger):
            result = check_context_budget.invoke({})
        assert "45" in result
        assert "OK" in result

    def test_returns_warning_when_high(self):
        mock_trigger = MagicMock()
        mock_trigger.usage_ratio.return_value = 0.72
        mock_trigger._config.prune_threshold = 0.60
        mock_trigger._config.extract_threshold = 0.70
        mock_trigger._config.compact_threshold = 0.80
        with patch("ag3nt_agent.context_tools._get_compaction_trigger", return_value=mock_trigger):
            result = check_context_budget.invoke({})
        assert "72" in result
        assert "EXTRACT" in result
        assert "artifact" in result.lower()

    def test_unavailable_when_no_trigger(self):
        with patch("ag3nt_agent.context_tools._get_compaction_trigger", return_value=None):
            result = check_context_budget.invoke({})
        assert "not available" in result.lower()


class TestCompactNow:
    def test_requests_compaction(self):
        mock_trigger = MagicMock()
        with patch("ag3nt_agent.context_tools._get_compaction_trigger", return_value=mock_trigger):
            result = compact_now.invoke({})
        mock_trigger.request_immediate.assert_called_once()
        assert "requested" in result.lower()

    def test_unavailable_when_no_trigger(self):
        with patch("ag3nt_agent.context_tools._get_compaction_trigger", return_value=None):
            result = compact_now.invoke({})
        assert "not available" in result.lower()


import shutil
import tempfile


class TestScratchpadNotes:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        _set_notes_dir(self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_and_read_note(self):
        write_result = write_note.invoke({"key": "findings", "content": "Found a bug in auth"})
        assert "saved" in write_result.lower()

        read_result = read_note.invoke({"key": "findings"})
        assert "Found a bug in auth" in read_result

    def test_read_nonexistent_note(self):
        result = read_note.invoke({"key": "nonexistent"})
        assert "not found" in result.lower()

    def test_list_notes_empty(self):
        result = list_notes.invoke({})
        assert "no notes" in result.lower()

    def test_list_notes_shows_keys(self):
        write_note.invoke({"key": "alpha", "content": "first"})
        write_note.invoke({"key": "beta", "content": "second"})
        result = list_notes.invoke({})
        assert "alpha" in result
        assert "beta" in result

    def test_write_overwrites_existing(self):
        write_note.invoke({"key": "test", "content": "version 1"})
        write_note.invoke({"key": "test", "content": "version 2"})
        result = read_note.invoke({"key": "test"})
        assert "version 2" in result
        assert "version 1" not in result

    def test_key_sanitized(self):
        write_note.invoke({"key": "my/dangerous/../key", "content": "safe"})
        result = read_note.invoke({"key": "my/dangerous/../key"})
        assert "safe" in result
