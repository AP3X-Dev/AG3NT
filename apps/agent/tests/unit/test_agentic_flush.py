"""Tests for agentic memory flush -- LLM-powered insight extraction before compaction."""
from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_msg(role: str = "human", content: str | list = "hello"):
    """Create a minimal message-like object with .type and .content."""
    msg = types.SimpleNamespace(type=role, content=content)
    return msg


# ---------------------------------------------------------------------------
# _format_messages_for_flush
# ---------------------------------------------------------------------------

class TestFormatMessagesForFlush:

    def test_format_messages_basic(self):
        from ag3nt_agent.agentic_flush import _format_messages_for_flush

        msgs = [_make_msg("human", "Hi there"), _make_msg("ai", "Hello!")]
        result = _format_messages_for_flush(msgs)
        assert "[human] Hi there" in result
        assert "[ai] Hello!" in result

    def test_format_empty_messages(self):
        from ag3nt_agent.agentic_flush import _format_messages_for_flush

        result = _format_messages_for_flush([])
        assert result == ""

    def test_format_truncates_long_messages(self):
        from ag3nt_agent.agentic_flush import _format_messages_for_flush

        long_content = "x" * 1000
        msgs = [_make_msg("human", long_content)]
        result = _format_messages_for_flush(msgs)
        # Should be truncated to 500 chars + "..."
        assert result.endswith("...")
        # The content portion after "[human] " should be 503 chars (500 + "...")
        content_part = result.split("] ", 1)[1]
        assert len(content_part) == 503

    def test_format_handles_structured_content(self):
        from ag3nt_agent.agentic_flush import _format_messages_for_flush

        content = [{"text": "part1"}, {"text": "part2"}]
        msgs = [_make_msg("ai", content)]
        result = _format_messages_for_flush(msgs)
        assert "part1 part2" in result

    def test_format_handles_string_blocks_in_list(self):
        from ag3nt_agent.agentic_flush import _format_messages_for_flush

        content = ["chunk1", "chunk2"]
        msgs = [_make_msg("ai", content)]
        result = _format_messages_for_flush(msgs)
        assert "chunk1 chunk2" in result

    def test_format_limits_to_50_messages(self):
        from ag3nt_agent.agentic_flush import _format_messages_for_flush

        msgs = [_make_msg("human", f"msg-{i}") for i in range(100)]
        result = _format_messages_for_flush(msgs)
        lines = result.strip().split("\n")
        assert len(lines) == 50
        # Should keep the last 50 (msg-50 through msg-99)
        assert "msg-50" in result
        assert "msg-99" in result

    def test_format_skips_empty_content(self):
        from ag3nt_agent.agentic_flush import _format_messages_for_flush

        msgs = [_make_msg("human", ""), _make_msg("ai", "hello")]
        result = _format_messages_for_flush(msgs)
        assert "[human]" not in result
        assert "[ai] hello" in result


# ---------------------------------------------------------------------------
# _append_insights_to_memory
# ---------------------------------------------------------------------------

class TestAppendInsightsToMemory:

    def test_append_insights_to_memory(self, tmp_path):
        from ag3nt_agent.agentic_flush import _append_insights_to_memory

        mem_file = tmp_path / "MEMORY.md"
        with patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file):
            count = _append_insights_to_memory(
                "- Insight one\n- Insight two", session_id="sess-123"
            )
        assert count == 2
        content = mem_file.read_text(encoding="utf-8")
        assert "Agentic flush" in content
        assert "sess-123" in content
        assert "- Insight one" in content
        assert "- Insight two" in content

    def test_append_insights_none(self):
        from ag3nt_agent.agentic_flush import _append_insights_to_memory

        assert _append_insights_to_memory("NONE") == 0
        assert _append_insights_to_memory("  none  ") == 0

    def test_append_insights_empty(self):
        from ag3nt_agent.agentic_flush import _append_insights_to_memory

        assert _append_insights_to_memory("") == 0
        assert _append_insights_to_memory(None) == 0

    def test_append_insights_bullet_points(self, tmp_path):
        from ag3nt_agent.agentic_flush import _append_insights_to_memory

        mem_file = tmp_path / "MEMORY.md"
        with patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file):
            count = _append_insights_to_memory(
                "- First bullet\n* Second bullet\n- Third bullet"
            )
        assert count == 3
        content = mem_file.read_text(encoding="utf-8")
        assert "- First bullet" in content
        assert "* Second bullet" in content
        assert "- Third bullet" in content

    def test_append_insights_non_bullet_lines(self, tmp_path):
        """Lines without bullet prefixes get auto-prefixed with '- '."""
        from ag3nt_agent.agentic_flush import _append_insights_to_memory

        mem_file = tmp_path / "MEMORY.md"
        with patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file):
            count = _append_insights_to_memory("User prefers dark mode")
        assert count == 1
        content = mem_file.read_text(encoding="utf-8")
        assert "- User prefers dark mode" in content

    def test_append_creates_parent_dir(self, tmp_path):
        from ag3nt_agent.agentic_flush import _append_insights_to_memory

        mem_file = tmp_path / "subdir" / "MEMORY.md"
        with patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file):
            count = _append_insights_to_memory("- An insight")
        assert count == 1
        assert mem_file.exists()

    def test_append_no_session_id(self, tmp_path):
        from ag3nt_agent.agentic_flush import _append_insights_to_memory

        mem_file = tmp_path / "MEMORY.md"
        with patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file):
            _append_insights_to_memory("- Fact learned")
        content = mem_file.read_text(encoding="utf-8")
        assert "session:" not in content


# ---------------------------------------------------------------------------
# agentic_flush (async, mocked LLM)
# ---------------------------------------------------------------------------

class TestAgenticFlush:

    async def test_agentic_flush_calls_llm(self, tmp_path):
        from ag3nt_agent.agentic_flush import agentic_flush

        mock_response = MagicMock()
        mock_response.content = "- Decision: use pytest\n- Preference: dark mode"

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        mem_file = tmp_path / "MEMORY.md"

        with (
            patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file),
            patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm) as mock_cls,
        ):
            msgs = [_make_msg("human", "Let's use pytest"), _make_msg("ai", "Good idea")]
            count = await agentic_flush(msgs, session_id="test-session")

        assert count == 2
        mock_cls.assert_called_once_with(
            model="claude-haiku-4-5-20251001", max_tokens=1024, temperature=0
        )
        mock_llm.ainvoke.assert_called_once()
        # Verify the prompt was passed
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert "Messages to review:" in call_args

        content = mem_file.read_text(encoding="utf-8")
        assert "- Decision: use pytest" in content
        assert "- Preference: dark mode" in content

    async def test_agentic_flush_empty_messages(self):
        from ag3nt_agent.agentic_flush import agentic_flush

        result = await agentic_flush([])
        assert result == 0

    async def test_agentic_flush_handles_error(self):
        from ag3nt_agent.agentic_flush import agentic_flush

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm):
            msgs = [_make_msg("human", "test")]
            result = await agentic_flush(msgs)

        assert result == 0

    async def test_agentic_flush_llm_returns_none_text(self, tmp_path):
        """When the LLM says NONE, no insights are persisted."""
        from ag3nt_agent.agentic_flush import agentic_flush

        mock_response = MagicMock()
        mock_response.content = "NONE"

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        mem_file = tmp_path / "MEMORY.md"
        with (
            patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file),
            patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm),
        ):
            msgs = [_make_msg("human", "nothing important")]
            count = await agentic_flush(msgs)

        assert count == 0
        assert not mem_file.exists()

    async def test_agentic_flush_custom_model(self, tmp_path):
        from ag3nt_agent.agentic_flush import agentic_flush

        mock_response = MagicMock()
        mock_response.content = "- An insight"

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        mem_file = tmp_path / "MEMORY.md"
        with (
            patch("ag3nt_agent.agentic_flush.MEMORY_FILE", mem_file),
            patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm) as mock_cls,
        ):
            msgs = [_make_msg("human", "hello")]
            await agentic_flush(msgs, model="claude-sonnet-4-20250514")

        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-20250514", max_tokens=1024, temperature=0
        )

    async def test_agentic_flush_empty_content_messages(self):
        """Messages with empty content should result in 0 insights."""
        from ag3nt_agent.agentic_flush import agentic_flush

        msgs = [_make_msg("human", ""), _make_msg("ai", "")]
        result = await agentic_flush(msgs)
        assert result == 0
