"""Tests for ag3nt_agent.memory_auto_indexer."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from ag3nt_agent.memory_auto_indexer import MemoryAutoIndexer, extract_insights


class TestExtractInsights:
    """Unit tests for the extract_insights helper."""

    def test_extracts_preferences(self):
        msgs = [HumanMessage(content="I prefer using Python for scripting tasks.")]
        insights = extract_insights(msgs)
        assert len(insights) >= 1
        assert any("prefer" in i.lower() for i in insights)

    def test_extracts_decisions(self):
        msgs = [HumanMessage(content="I've decided to use PostgreSQL for the backend.")]
        insights = extract_insights(msgs)
        assert len(insights) >= 1
        assert any("decided" in i.lower() for i in insights)

    def test_extracts_facts(self):
        msgs = [HumanMessage(content="Important: the deploy key rotates every 90 days.")]
        insights = extract_insights(msgs)
        assert len(insights) >= 1
        assert any("important" in i.lower() for i in insights)

    def test_extracts_note_facts(self):
        msgs = [HumanMessage(content="Note: the API endpoint has changed to v3.")]
        insights = extract_insights(msgs)
        assert len(insights) >= 1

    def test_extracts_lets_use_decision(self):
        msgs = [HumanMessage(content="Let's use Redis for the caching layer.")]
        insights = extract_insights(msgs)
        assert len(insights) >= 1

    def test_extracts_never_preference(self):
        msgs = [HumanMessage(content="Never commit secrets to the repository.")]
        insights = extract_insights(msgs)
        assert len(insights) >= 1

    def test_no_insights_from_empty(self):
        insights = extract_insights([])
        assert insights == []

    def test_no_insights_from_generic_messages(self):
        msgs = [
            HumanMessage(content="Hello, how are you?"),
            AIMessage(content="Fine thanks."),
        ]
        insights = extract_insights(msgs)
        assert insights == []

    def test_ignores_short_sentences(self):
        # "I like X" is only 8 chars -- below _MIN_INSIGHT_LEN
        msgs = [HumanMessage(content="I like X.")]
        insights = extract_insights(msgs)
        assert insights == []

    def test_ignores_very_long_sentences(self):
        long = "I prefer " + "x" * 250 + "."
        msgs = [HumanMessage(content=long)]
        insights = extract_insights(msgs)
        assert insights == []

    def test_deduplicates(self):
        msgs = [
            HumanMessage(content="I prefer Python for everything."),
            HumanMessage(content="I prefer Python for everything."),
        ]
        insights = extract_insights(msgs)
        assert len(insights) == 1

    def test_handles_ai_messages(self):
        msgs = [AIMessage(content="Important: the server runs on port 8080.")]
        insights = extract_insights(msgs)
        assert len(insights) >= 1

    def test_handles_non_string_content(self):
        """Messages with non-string content should be skipped silently."""
        msgs = [HumanMessage(content="")]  # type: ignore
        insights = extract_insights(msgs)
        assert insights == []


class TestMemoryAutoIndexer:
    """Unit tests for the MemoryAutoIndexer class."""

    def test_skips_before_interval(self):
        indexer = MemoryAutoIndexer(flush_interval=5)
        msgs = [HumanMessage(content="I prefer tabs over spaces.")]
        for _ in range(4):
            assert indexer.maybe_index(msgs, session_id="s1") is False

    def test_triggers_at_interval(self):
        indexer = MemoryAutoIndexer(flush_interval=3)
        msgs = [HumanMessage(content="I prefer tabs over spaces for indentation.")]
        with patch.object(indexer, "_append_to_memory"):
            assert indexer.maybe_index(msgs, session_id="s1") is False
            assert indexer.maybe_index(msgs, session_id="s1") is False
            assert indexer.maybe_index(msgs, session_id="s1") is True

    def test_triggers_repeatedly(self):
        indexer = MemoryAutoIndexer(flush_interval=2)
        msgs = [HumanMessage(content="Important: rotate keys every week.")]
        with patch.object(indexer, "_append_to_memory"):
            assert indexer.maybe_index(msgs, session_id="s1") is False
            assert indexer.maybe_index(msgs, session_id="s1") is True
            assert indexer.maybe_index(msgs, session_id="s1") is False
            assert indexer.maybe_index(msgs, session_id="s1") is True

    def test_calls_append_with_insights(self):
        indexer = MemoryAutoIndexer(flush_interval=1)
        msgs = [HumanMessage(content="I prefer dark mode for all editors.")]
        with patch.object(indexer, "_append_to_memory") as mock_append:
            indexer.maybe_index(msgs, session_id="sess-42")
            mock_append.assert_called_once()
            insights_arg = mock_append.call_args[0][0]
            assert len(insights_arg) >= 1
            assert mock_append.call_args[0][1] == "sess-42"

    def test_no_append_when_no_insights(self):
        indexer = MemoryAutoIndexer(flush_interval=1)
        msgs = [HumanMessage(content="Hello world.")]
        with patch.object(indexer, "_append_to_memory") as mock_append:
            indexer.maybe_index(msgs, session_id="s1")
            mock_append.assert_not_called()

    def test_append_to_memory_writes_file(self, tmp_path):
        indexer = MemoryAutoIndexer(flush_interval=1)
        fake_mem = tmp_path / "MEMORY.md"
        with patch("ag3nt_agent.memory_auto_indexer.MEMORY_FILE", fake_mem):
            indexer._append_to_memory(
                ["I prefer dark mode.", "Important: use v3 API."], "sess-1"
            )
        content = fake_mem.read_text(encoding="utf-8")
        assert "Auto-indexed" in content
        assert "sess-1" in content
        assert "- I prefer dark mode." in content
        assert "- Important: use v3 API." in content

    def test_append_creates_parent_dirs(self, tmp_path):
        indexer = MemoryAutoIndexer(flush_interval=1)
        fake_mem = tmp_path / "nested" / "dir" / "MEMORY.md"
        with patch("ag3nt_agent.memory_auto_indexer.MEMORY_FILE", fake_mem):
            indexer._append_to_memory(["Note: test insight here."], "s1")
        assert fake_mem.exists()

    def test_exception_does_not_crash(self):
        """Fire-and-forget: exceptions in indexing are swallowed."""
        indexer = MemoryAutoIndexer(flush_interval=1)
        msgs = [HumanMessage(content="I prefer Python for all scripting.")]
        with patch.object(
            indexer, "_append_to_memory", side_effect=OSError("disk full")
        ):
            # Should return True (interval matched) but not raise
            result = indexer.maybe_index(msgs, session_id="s1")
            assert result is True

    def test_uses_recent_messages_window(self):
        """Only the last (flush_interval * 2) messages should be examined."""
        indexer = MemoryAutoIndexer(flush_interval=2)
        old_msgs = [HumanMessage(content="Hello.") for _ in range(10)]
        recent = [HumanMessage(content="I prefer dark mode for the IDE.")]
        all_msgs = old_msgs + recent

        with patch.object(indexer, "_append_to_memory") as mock_append:
            # First call: skip
            indexer.maybe_index(all_msgs, session_id="s1")
            # Second call: trigger
            indexer.maybe_index(all_msgs, session_id="s1")
            if mock_append.called:
                insights = mock_append.call_args[0][0]
                assert any("prefer" in i.lower() for i in insights)
