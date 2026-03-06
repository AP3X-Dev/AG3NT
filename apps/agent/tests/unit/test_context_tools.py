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

from ag3nt_agent.context_tools import dump_to_artifact, read_artifact, get_context_tools


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
    def test_returns_both_tools(self):
        tools = get_context_tools()
        assert isinstance(tools, list)
        assert len(tools) == 2
        assert dump_to_artifact in tools
        assert read_artifact in tools
