"""Tests for the 1K-token auto-dump rule in AgentGuardMiddleware."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from ag3nt_agent.agent_guard_middleware import AgentGuardMiddleware
from ag3nt_agent.artifact_store import ArtifactMeta


def _make_tool_message(content: str, name: str = "test_tool") -> ToolMessage:
    """Create a ToolMessage with the given content."""
    return ToolMessage(content=content, tool_call_id="call_123", name=name)


def _make_artifact_meta(
    artifact_id: str = "abc123_deadbeef",
    size_bytes: int = 5000,
    tool_name: str = "test_tool",
) -> ArtifactMeta:
    return ArtifactMeta(
        artifact_id=artifact_id,
        tool_name=tool_name,
        content_hash="fakehash",
        size_bytes=size_bytes,
        created_at="2026-03-06T00:00:00+00:00",
    )


class TestShortOutputPassesThrough:
    """ToolMessage with <4000 chars should pass through unchanged."""

    def test_short_output_unchanged(self):
        msg = _make_tool_message("x" * 3999)
        result = AgentGuardMiddleware._maybe_truncate_result(msg)
        assert result.content == "x" * 3999

    def test_empty_content_passes_through(self):
        msg = _make_tool_message("")
        result = AgentGuardMiddleware._maybe_truncate_result(msg)
        assert result.content == ""


class TestLargeOutputGetsArtifactReference:
    """ToolMessage with >4000 chars should be stored as artifact."""

    @patch("ag3nt_agent.artifact_store.get_artifact_store")
    def test_large_output_gets_artifact_reference(self, mock_get_store):
        large_content = "A" * 5000
        meta = _make_artifact_meta(size_bytes=5000)

        mock_store = MagicMock()
        mock_store.write_artifact.return_value = meta
        mock_get_store.return_value = mock_store

        msg = _make_tool_message(large_content, name="shell_execute")
        result = AgentGuardMiddleware._maybe_truncate_result(msg)

        # Should contain preview (first 500 chars)
        assert result.content.startswith("A" * 500)
        # Should contain artifact reference
        assert meta.artifact_id in result.content
        assert "5000 bytes" in result.content
        assert "read_artifact" in result.content

        # Verify store was called correctly
        mock_store.write_artifact.assert_called_once_with(
            large_content, tool_name="shell_execute"
        )

    @patch("ag3nt_agent.artifact_store.get_artifact_store")
    def test_preview_is_truncated_to_500_chars(self, mock_get_store):
        large_content = "B" * 6000
        meta = _make_artifact_meta(size_bytes=6000)

        mock_store = MagicMock()
        mock_store.write_artifact.return_value = meta
        mock_get_store.return_value = mock_store

        msg = _make_tool_message(large_content)
        result = AgentGuardMiddleware._maybe_truncate_result(msg)

        # Preview should be exactly 500 chars of 'B' followed by artifact info
        lines = result.content.split("\n\n")
        assert lines[0] == "B" * 500

    @patch("ag3nt_agent.artifact_store.get_artifact_store")
    def test_unknown_tool_name_fallback(self, mock_get_store):
        """When ToolMessage has no name, falls back to 'unknown_tool'."""
        large_content = "C" * 5000
        meta = _make_artifact_meta(tool_name="unknown_tool", size_bytes=5000)

        mock_store = MagicMock()
        mock_store.write_artifact.return_value = meta
        mock_get_store.return_value = mock_store

        # Create a ToolMessage without a name attribute
        msg = ToolMessage(content=large_content, tool_call_id="call_456")
        # Remove name if set
        if hasattr(msg, "name"):
            msg.name = None

        AgentGuardMiddleware._maybe_truncate_result(msg)
        mock_store.write_artifact.assert_called_once_with(
            large_content, tool_name="unknown_tool"
        )


class TestOutputAtThresholdPassesThrough:
    """Exactly 4000 chars should NOT be dumped (only > threshold)."""

    def test_exactly_at_threshold_not_dumped(self):
        msg = _make_tool_message("D" * 4000)
        result = AgentGuardMiddleware._maybe_truncate_result(msg)
        # Content should remain unchanged (no artifact reference)
        assert "artifact" not in result.content.lower()
        assert result.content == "D" * 4000


class TestCommandResultsPassThrough:
    """Command objects should pass through unchanged."""

    def test_command_passes_through(self):
        cmd = Command(update={"messages": []})
        result = AgentGuardMiddleware._maybe_truncate_result(cmd)
        assert result is cmd
        assert isinstance(result, Command)


class TestArtifactStoreFailureFallsBack:
    """If the artifact store raises, we fall back to truncation gracefully."""

    @patch("ag3nt_agent.artifact_store.get_artifact_store")
    def test_store_exception_falls_back(self, mock_get_store):
        mock_store = MagicMock()
        mock_store.write_artifact.side_effect = RuntimeError("disk full")
        mock_get_store.return_value = mock_store

        large_content = "E" * 5000
        msg = _make_tool_message(large_content)
        result = AgentGuardMiddleware._maybe_truncate_result(msg)

        # Should not crash; result should still be a ToolMessage
        assert isinstance(result, ToolMessage)
        # The content should either be the original or truncated by fallback,
        # but NOT contain an artifact reference
        assert "artifact" not in result.content.lower() or result.content == large_content

    @patch("ag3nt_agent.artifact_store.get_artifact_store")
    def test_import_error_falls_back(self, mock_get_store):
        mock_get_store.side_effect = ImportError("module not found")

        large_content = "F" * 5000
        msg = _make_tool_message(large_content)
        result = AgentGuardMiddleware._maybe_truncate_result(msg)

        # Should not crash
        assert isinstance(result, ToolMessage)
