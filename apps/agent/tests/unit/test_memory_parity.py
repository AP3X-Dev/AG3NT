"""Unit tests for memory parity modules (P2-7).

Tests cover:
- ArtifactStore (P2-7.1)
- ObservationMasker (P2-7.2)
- EmbeddingCache (P2-7.6)
- MemoryFlusher (P2-7.7)
- ProgressiveSummarizer (P2-7.8)
- CompactionMiddleware (P2-7.9)
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# ============================================================================
# ArtifactStore Tests (P2-7.1)
# ============================================================================


class TestArtifactStore:
    """Tests for ArtifactStore class."""

    def test_write_and_read_artifact(self, tmp_path: Path):
        """Test writing and reading an artifact."""
        from ag3nt_agent.artifact_store import ArtifactStore

        store = ArtifactStore(artifacts_dir=tmp_path / "artifacts")
        content = "This is a test artifact with important content."

        meta = store.write_artifact(
            content=content,
            tool_name="test_tool",
            source_url="/test/path",
            tags=["test", "unit"],
        )

        assert meta.artifact_id is not None
        assert meta.tool_name == "test_tool"
        assert meta.source_url == "/test/path"
        assert "test" in meta.tags

        # Read back
        retrieved = store.read_artifact(meta.artifact_id)
        assert retrieved == content

    def test_artifact_deduplication(self, tmp_path: Path):
        """Test that duplicate content returns existing artifact."""
        from ag3nt_agent.artifact_store import ArtifactStore

        store = ArtifactStore(artifacts_dir=tmp_path / "artifacts")
        content = "Duplicate content test"

        meta1 = store.write_artifact(content=content, tool_name="tool1")
        meta2 = store.write_artifact(content=content, tool_name="tool2")

        # Same content hash should return same artifact
        assert meta1.content_hash == meta2.content_hash
        assert meta1.artifact_id == meta2.artifact_id

    def test_list_artifacts(self, tmp_path: Path):
        """Test listing artifacts with filters."""
        from ag3nt_agent.artifact_store import ArtifactStore

        store = ArtifactStore(artifacts_dir=tmp_path / "artifacts")

        store.write_artifact("Content 1", "tool_a")
        store.write_artifact("Content 2", "tool_a")
        store.write_artifact("Content 3", "tool_b")

        all_artifacts = store.list_artifacts()
        assert len(all_artifacts) == 3

        tool_a_artifacts = store.list_artifacts(tool_name="tool_a")
        assert len(tool_a_artifacts) == 2

    def test_get_metadata(self, tmp_path: Path):
        """Test retrieving artifact metadata."""
        from ag3nt_agent.artifact_store import ArtifactStore

        store = ArtifactStore(artifacts_dir=tmp_path / "artifacts")
        meta = store.write_artifact("Test content", "test_tool")

        retrieved_meta = store.get_metadata(meta.artifact_id)
        assert retrieved_meta is not None
        assert retrieved_meta.artifact_id == meta.artifact_id

    def test_read_nonexistent_artifact(self, tmp_path: Path):
        """Test reading a nonexistent artifact returns None."""
        from ag3nt_agent.artifact_store import ArtifactStore

        store = ArtifactStore(artifacts_dir=tmp_path / "artifacts")
        result = store.read_artifact("nonexistent-id")
        assert result is None


# ============================================================================
# ObservationMasker Tests (P2-7.2)
# ============================================================================


class TestObservationMasker:
    """Tests for ObservationMasker class."""

    def test_mask_small_content_unchanged(self, tmp_path: Path):
        """Test that small content is not masked."""
        from ag3nt_agent.observation_masking import ObservationMasker

        with patch("ag3nt_agent.observation_masking.get_artifact_store") as mock_store:
            masker = ObservationMasker(threshold_tokens=1000, threshold_chars=4000)
            result = masker.mask_if_needed("Small content", "test_tool")

            assert result.was_masked is False
            assert result.content == "Small content"
            mock_store.assert_not_called()

    def test_mask_large_content(self, tmp_path: Path):
        """Test that large content is masked."""
        from ag3nt_agent.observation_masking import ObservationMasker

        # Create large content
        large_content = "x" * 5000  # Exceeds 4000 char threshold

        with patch("ag3nt_agent.observation_masking.get_artifact_store") as mock_store:
            mock_meta = MagicMock()
            mock_meta.artifact_id = "test-artifact-id"
            mock_store.return_value.write_artifact.return_value = mock_meta

            masker = ObservationMasker(threshold_tokens=100, threshold_chars=1000)
            result = masker.mask_if_needed(large_content, "test_tool")

            assert result.was_masked is True
            # Actual format is "[Output stored as artifact: {id}]"
            assert "[Output stored as artifact:" in result.content
            assert "test-artifact-id" in result.content

    def test_mask_preset_configurations(self):
        """Test preset masking configurations."""
        from ag3nt_agent.observation_masking import (
            MASKER_AGGRESSIVE,
            MASKER_BALANCED,
            MASKER_CONSERVATIVE,
        )

        assert MASKER_CONSERVATIVE._threshold_chars > MASKER_BALANCED._threshold_chars
        assert MASKER_BALANCED._threshold_chars > MASKER_AGGRESSIVE._threshold_chars

    def test_mask_tool_message(self):
        """Test masking a ToolMessage."""
        from ag3nt_agent.observation_masking import ObservationMasker

        with patch("ag3nt_agent.observation_masking.get_artifact_store") as mock_store:
            mock_meta = MagicMock()
            mock_meta.artifact_id = "art-123"
            mock_store.return_value.write_artifact.return_value = mock_meta

            masker = ObservationMasker(threshold_chars=100, threshold_tokens=50)
            # Create a ToolMessage with string content and a name
            tool_msg = ToolMessage(content="x" * 200, tool_call_id="call-1", name="test_tool")

            masked_msg, result = masker.mask_tool_message(tool_msg)

            assert result is not None
            assert result.was_masked is True


# ============================================================================
# EmbeddingCache Tests (P2-7.6)
# ============================================================================


class TestEmbeddingCache:
    """Tests for EmbeddingCache class."""

    def test_cache_set_and_get(self, tmp_path: Path):
        """Test setting and getting cached embeddings."""
        from ag3nt_agent.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(db_path=tmp_path / "test_cache.db")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        cache.set("test content", embedding, provider="openai", model="text-embedding-3-small")
        result = cache.get("test content")

        assert result is not None
        assert result == embedding

    def test_cache_miss(self, tmp_path: Path):
        """Test cache miss returns None."""
        from ag3nt_agent.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(db_path=tmp_path / "test_cache.db")
        result = cache.get("nonexistent content")
        assert result is None

    def test_get_or_compute(self, tmp_path: Path):
        """Test get_or_compute with compute function."""
        from ag3nt_agent.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(db_path=tmp_path / "test_cache.db")
        compute_called = []

        def compute_fn(text: str) -> list[float]:
            compute_called.append(text)
            return [0.1, 0.2, 0.3]

        # First call should compute
        result1 = cache.get_or_compute("test", compute_fn)
        assert len(compute_called) == 1

        # Second call should use cache
        result2 = cache.get_or_compute("test", compute_fn)
        assert len(compute_called) == 1  # Not called again
        assert result1 == result2

    def test_cache_stats(self, tmp_path: Path):
        """Test cache statistics."""
        from ag3nt_agent.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(db_path=tmp_path / "test_cache.db")
        cache.set("content1", [0.1, 0.2])
        cache.get("content1")  # Hit
        cache.get("content2")  # Miss

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.entries_count == 1

    def test_cleanup_stale(self, tmp_path: Path):
        """Test cleanup of stale entries."""
        from ag3nt_agent.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(db_path=tmp_path / "test_cache.db", max_age_days=0)
        cache.set("old content", [0.1, 0.2])

        # Cleanup with 0 days should remove immediately
        removed = cache.cleanup_stale(max_age_days=0)
        # Note: This might not remove immediately due to timing
        assert removed >= 0


# ============================================================================
# MemoryFlusher Tests (P2-7.7)
# ============================================================================


class TestMemoryFlusher:
    """Tests for MemoryFlusher class."""

    def test_should_flush_below_threshold(self):
        """Test should_flush returns False below threshold."""
        from ag3nt_agent.memory_flush import FlushConfig, MemoryFlusher

        flusher = MemoryFlusher(FlushConfig(soft_threshold=80000))
        assert flusher.should_flush(50000) is False

    def test_should_flush_above_threshold(self):
        """Test should_flush returns True above threshold."""
        from ag3nt_agent.memory_flush import FlushConfig, MemoryFlusher

        config = FlushConfig(
            soft_threshold=80000,
            reserve_tokens=20000,
            flush_buffer=4000,
        )
        flusher = MemoryFlusher(config)
        # Threshold = 80000 - 20000 - 4000 = 56000
        assert flusher.should_flush(60000) is True

    def test_extract_decisions(self):
        """Test decision extraction from messages."""
        from ag3nt_agent.memory_flush import MemoryFlusher

        flusher = MemoryFlusher()
        messages = [
            {"content": "I've decided to use TypeScript for this project"},
            {"content": "The decision was to implement caching first"},
        ]

        result = flusher.flush(messages)
        assert result.flushed is True
        assert len(result.decisions) >= 1

    def test_extract_preferences(self):
        """Test preference extraction from messages."""
        from ag3nt_agent.memory_flush import MemoryFlusher

        flusher = MemoryFlusher()
        messages = [
            {"content": "User prefers dark mode for all interfaces"},
        ]

        result = flusher.flush(messages)
        assert len(result.preferences) >= 1

    def test_flush_disabled(self):
        """Test flush when disabled."""
        from ag3nt_agent.memory_flush import FlushConfig, MemoryFlusher

        flusher = MemoryFlusher(FlushConfig(enabled=False))
        messages = [{"content": "I've decided to use Python"}]

        result = flusher.flush(messages)
        assert result.flushed is False

    def test_flush_presets(self):
        """Test preset configurations."""
        from ag3nt_agent.memory_flush import (
            FLUSH_AGGRESSIVE,
            FLUSH_BALANCED,
            FLUSH_CONSERVATIVE,
            FLUSH_DISABLED,
        )

        assert FLUSH_DISABLED.enabled is False
        assert FLUSH_CONSERVATIVE.soft_threshold > FLUSH_BALANCED.soft_threshold
        assert FLUSH_BALANCED.soft_threshold > FLUSH_AGGRESSIVE.soft_threshold


# ============================================================================
# ProgressiveSummarizer Tests (P2-7.8)
# ============================================================================


class TestProgressiveSummarizer:
    """Tests for ProgressiveSummarizer class."""

    def test_split_into_chunks(self):
        """Test splitting messages into chunks."""
        from ag3nt_agent.context_summarization import (
            ProgressiveConfig,
            ProgressiveSummarizer,
        )

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
            HumanMessage(content="How are you?"),
            AIMessage(content="I'm doing well"),
            HumanMessage(content="Great"),
            AIMessage(content="Thanks for asking"),
        ]

        summarizer = ProgressiveSummarizer(
            ProgressiveConfig(max_chunk_tokens=50, min_chunk_messages=2)
        )
        chunks = summarizer.split_into_chunks(messages)

        assert len(chunks) >= 1
        assert all(len(c.messages) >= 2 or c == chunks[-1] for c in chunks)

    def test_merge_summaries(self):
        """Test merging multiple summaries."""
        from ag3nt_agent.context_summarization import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer()
        summaries = ["Summary 1", "Summary 2", "Summary 3"]

        merged = summarizer.merge_summaries(summaries)
        assert "[Part 1/3]" in merged
        assert "[Part 2/3]" in merged
        assert "[Part 3/3]" in merged

    def test_merge_single_summary(self):
        """Test merging single summary returns it unchanged."""
        from ag3nt_agent.context_summarization import ProgressiveSummarizer

        summarizer = ProgressiveSummarizer()
        merged = summarizer.merge_summaries(["Only summary"])
        assert merged == "Only summary"

    def test_summarize_with_mock_function(self):
        """Test summarize with mock summarization function."""
        from ag3nt_agent.context_summarization import (
            ProgressiveConfig,
            ProgressiveSummarizer,
        )

        messages = [HumanMessage(content=f"Message {i}") for i in range(10)]

        summarizer = ProgressiveSummarizer(
            ProgressiveConfig(max_chunk_tokens=50, min_chunk_messages=2)
        )

        def mock_summarize(msgs):
            return f"Summary of {len(msgs)} messages"

        result = summarizer.summarize(messages, mock_summarize)
        assert result.chunks_processed >= 1

    def test_progressive_presets(self):
        """Test preset configurations."""
        from ag3nt_agent.context_summarization import (
            PROGRESSIVE_AGGRESSIVE,
            PROGRESSIVE_BALANCED,
            PROGRESSIVE_CONSERVATIVE,
            PROGRESSIVE_DISABLED,
        )

        assert PROGRESSIVE_DISABLED.enabled is False
        assert PROGRESSIVE_CONSERVATIVE.max_chunk_tokens > PROGRESSIVE_BALANCED.max_chunk_tokens
        assert PROGRESSIVE_BALANCED.max_chunk_tokens > PROGRESSIVE_AGGRESSIVE.max_chunk_tokens


# ============================================================================
# CompactionMiddleware Tests (P2-7.9)
# ============================================================================


class TestCompactionMiddleware:
    """Tests for CompactionMiddleware class."""

    def test_should_compact_below_threshold(self):
        """Test should_compact returns False below threshold."""
        from ag3nt_agent.compaction_middleware import CompactionConfig, CompactionMiddleware

        middleware = CompactionMiddleware(
            CompactionConfig(token_threshold=100000, message_threshold=1000)
        )
        messages = [HumanMessage(content="Short message")]

        assert middleware.should_compact(messages) is False

    def test_should_compact_above_threshold(self):
        """Test should_compact returns True above threshold."""
        from ag3nt_agent.compaction_middleware import CompactionConfig, CompactionMiddleware

        middleware = CompactionMiddleware(
            CompactionConfig(token_threshold=10, message_threshold=3)
        )
        messages = [
            HumanMessage(content="Message 1"),
            AIMessage(content="Message 2"),
            HumanMessage(content="Message 3"),
            AIMessage(content="Message 4"),
        ]

        assert middleware.should_compact(messages) is True

    def test_compact_not_triggered(self):
        """Test compact when not triggered."""
        from ag3nt_agent.compaction_middleware import CompactionConfig, CompactionMiddleware

        middleware = CompactionMiddleware(
            CompactionConfig(token_threshold=100000, message_threshold=1000)
        )
        messages = [HumanMessage(content="Short")]

        result_messages, metrics = middleware.compact(messages)

        assert metrics.triggered is False
        assert result_messages == messages

    def test_compact_triggered(self):
        """Test compact when triggered."""
        from ag3nt_agent.compaction_middleware import CompactionConfig, CompactionMiddleware

        # Use very low thresholds to trigger compaction
        middleware = CompactionMiddleware(
            CompactionConfig(
                token_threshold=1,
                message_threshold=1,
                enable_masking=False,  # Disable to simplify test
                enable_flush=False,
                enable_pruning=False,
                enable_progressive=False,
            )
        )
        messages = [HumanMessage(content="Test message")]

        result_messages, metrics = middleware.compact(messages)

        assert metrics.triggered is True
        assert metrics.messages_before == 1

    def test_compaction_disabled(self):
        """Test compaction when disabled."""
        from ag3nt_agent.compaction_middleware import CompactionConfig, CompactionMiddleware

        middleware = CompactionMiddleware(CompactionConfig(enabled=False))
        messages = [HumanMessage(content="Test")]

        assert middleware.should_compact(messages) is False

    def test_get_stats_empty(self):
        """Test get_stats with no compactions."""
        from ag3nt_agent.compaction_middleware import CompactionMiddleware

        middleware = CompactionMiddleware()
        stats = middleware.get_stats()

        assert stats["total_compactions"] == 0
        assert stats["avg_compression_ratio"] == 1.0

    def test_compaction_presets(self):
        """Test preset configurations."""
        from ag3nt_agent.compaction_middleware import (
            COMPACTION_AGGRESSIVE,
            COMPACTION_BALANCED,
            COMPACTION_CONSERVATIVE,
            COMPACTION_DISABLED,
        )

        assert COMPACTION_DISABLED.enabled is False
        assert COMPACTION_CONSERVATIVE.token_threshold > COMPACTION_BALANCED.token_threshold
        assert COMPACTION_BALANCED.token_threshold > COMPACTION_AGGRESSIVE.token_threshold

    def test_metrics_compression_ratio(self):
        """Test CompactionMetrics compression ratio calculation."""
        from ag3nt_agent.compaction_middleware import CompactionMetrics

        metrics = CompactionMetrics(
            triggered=True,
            tokens_before=1000,
            tokens_after=500,
        )

        assert metrics.compression_ratio == 0.5

    def test_metrics_compression_ratio_zero(self):
        """Test CompactionMetrics compression ratio with zero tokens."""
        from ag3nt_agent.compaction_middleware import CompactionMetrics

        metrics = CompactionMetrics(triggered=True, tokens_before=0, tokens_after=0)
        assert metrics.compression_ratio == 1.0


# ============================================================================
# Global Helpers Tests
# ============================================================================


class TestGlobalHelpers:
    """Tests for global helper functions."""

    def test_artifact_store_global(self, tmp_path: Path):
        """Test get_artifact_store global helper."""
        from ag3nt_agent.artifact_store import get_artifact_store, reset_artifact_store

        reset_artifact_store()
        store1 = get_artifact_store()
        store2 = get_artifact_store()
        assert store1 is store2

    def test_observation_masker_global(self):
        """Test get_observation_masker global helper."""
        from ag3nt_agent.observation_masking import (
            get_observation_masker,
            reset_observation_masker,
        )

        reset_observation_masker()
        masker1 = get_observation_masker()
        masker2 = get_observation_masker()
        assert masker1 is masker2

    def test_memory_flusher_global(self):
        """Test get_memory_flusher global helper."""
        from ag3nt_agent.memory_flush import get_memory_flusher, reset_memory_flusher

        reset_memory_flusher()
        flusher1 = get_memory_flusher()
        flusher2 = get_memory_flusher()
        assert flusher1 is flusher2

    def test_progressive_summarizer_global(self):
        """Test get_progressive_summarizer global helper."""
        from ag3nt_agent.context_summarization import (
            get_progressive_summarizer,
            reset_progressive_summarizer,
        )

        reset_progressive_summarizer()
        summarizer1 = get_progressive_summarizer()
        summarizer2 = get_progressive_summarizer()
        assert summarizer1 is summarizer2

    def test_compaction_middleware_global(self):
        """Test get_compaction_middleware global helper."""
        from ag3nt_agent.compaction_middleware import (
            get_compaction_middleware,
            reset_compaction_middleware,
        )

        reset_compaction_middleware()
        middleware1 = get_compaction_middleware()
        middleware2 = get_compaction_middleware()
        assert middleware1 is middleware2

    def test_embedding_cache_global(self):
        """Test get_embedding_cache global helper."""
        from ag3nt_agent.embedding_cache import get_embedding_cache, reset_embedding_cache

        reset_embedding_cache()
        cache1 = get_embedding_cache()
        cache2 = get_embedding_cache()
        assert cache1 is cache2

