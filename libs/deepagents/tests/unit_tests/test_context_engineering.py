"""Tests for Context Engineering module."""

import pytest

from deepagents.context_engineering import (
    CacheStats,
    ContextEngineeringConfig,
    ContextEngineeringMiddleware,
    PromptAssemblyCache,
    TokenBudgetTracker,
)


class TestContextEngineeringConfig:
    """Tests for ContextEngineeringConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ContextEngineeringConfig()
        assert config.model_context_limit == 128_000
        assert config.summarization_trigger_ratio == 0.85
        assert config.mask_tool_output_if_chars_gt == 8000
        assert config.keep_last_unmasked_tool_outputs == 5

    def test_get_summarization_trigger_tokens(self):
        """Test summarization trigger calculation."""
        config = ContextEngineeringConfig(
            model_context_limit=100_000,
            summarization_trigger_ratio=0.8,
        )
        assert config.get_summarization_trigger_tokens() == 80_000

    def test_workspace_dir_auto_creation(self):
        """Test workspace directory is auto-created if None."""
        config = ContextEngineeringConfig(workspace_dir=None)
        workspace = config.get_workspace_dir()
        assert workspace.exists()
        assert "deepagents_context_" in str(workspace)

    def test_estimate_tokens(self):
        """Test token estimation."""
        config = ContextEngineeringConfig()
        # ~4 chars per token
        assert config.estimate_tokens("a" * 100) == 25


class TestTokenBudgetTracker:
    """Tests for TokenBudgetTracker."""

    def test_update_component(self):
        """Test updating component token counts."""
        config = ContextEngineeringConfig(model_context_limit=10000)
        tracker = TokenBudgetTracker(config)

        tracker.update_component("messages", 5000)
        tracker.update_component("system_prompt", 1000)

        assert tracker.get_total_tokens() == 6000

    def test_should_summarize(self):
        """Test summarization trigger detection."""
        config = ContextEngineeringConfig(
            model_context_limit=10000,
            summarization_trigger_ratio=0.8,
        )
        tracker = TokenBudgetTracker(config)

        # Under threshold
        tracker.update_component("messages", 7000)
        assert not tracker.should_summarize()

        # Over threshold
        tracker.update_component("messages", 9000)
        assert tracker.should_summarize()

    def test_get_report(self):
        """Test budget report generation."""
        config = ContextEngineeringConfig(model_context_limit=10000)
        tracker = TokenBudgetTracker(config)

        tracker.update_component("messages", 3000)
        tracker.update_component("system_prompt", 500)

        report = tracker.get_report()
        assert report.total_budget == 10000
        assert report.used_tokens == 3500
        assert report.available_tokens == 6500
        assert report.usage_ratio == 0.35
        assert report.breakdown == {"messages": 3000, "system_prompt": 500}

    def test_step_counting(self):
        """Test step counter."""
        config = ContextEngineeringConfig()
        tracker = TokenBudgetTracker(config)

        assert tracker.step_count == 0
        tracker.increment_step()
        tracker.increment_step()
        assert tracker.step_count == 2


class TestContextEngineeringMiddleware:
    """Tests for ContextEngineeringMiddleware."""

    @pytest.fixture
    def workspace_dir(self, tmp_path):
        """Create a workspace directory for tests."""
        workspace = tmp_path / "context_workspace"
        workspace.mkdir()
        return workspace

    def test_initialization(self, workspace_dir):
        """Test middleware initializes correctly."""
        config = ContextEngineeringConfig(workspace_dir=workspace_dir)
        middleware = ContextEngineeringMiddleware(config=config)

        assert middleware.config is config
        assert middleware.budget_tracker is not None
        assert len(middleware.tools) == 2  # read_artifact, search_artifacts

        # Close the retrieval index to release file handles
        if hasattr(middleware, "retrieval_index"):
            middleware.retrieval_index.close()

    def test_tool_names(self, workspace_dir):
        """Test middleware provides expected tools."""
        config = ContextEngineeringConfig(workspace_dir=workspace_dir)
        middleware = ContextEngineeringMiddleware(config=config)

        tool_names = [t.name for t in middleware.tools]
        assert "read_artifact" in tool_names
        assert "search_artifacts" in tool_names

        if hasattr(middleware, "retrieval_index"):
            middleware.retrieval_index.close()

    def test_get_budget_report(self, workspace_dir):
        """Test getting budget report from middleware."""
        config = ContextEngineeringConfig(workspace_dir=workspace_dir)
        middleware = ContextEngineeringMiddleware(config=config)

        report = middleware.get_budget_report()
        assert "total_budget" in report
        assert "used_tokens" in report
        assert "step_count" in report

        if hasattr(middleware, "retrieval_index"):
            middleware.retrieval_index.close()


class TestPromptAssemblyCache:
    """Tests for PromptAssemblyCache."""

    def test_hash_content(self):
        """Test content hashing."""
        cache = PromptAssemblyCache()
        hash1 = cache.hash_content("hello world")
        hash2 = cache.hash_content("hello world")
        hash3 = cache.hash_content("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # Truncated hash

    def test_cache_hit(self):
        """Test cache hit scenario."""
        cache = PromptAssemblyCache()
        content_hash = cache.hash_content("test content")

        cache.set("key1", {"data": "value"}, content_hash)
        result = cache.get("key1", content_hash)

        assert result == {"data": "value"}
        assert cache.get_stats().hits == 1

    def test_cache_miss_wrong_hash(self):
        """Test cache miss when hash changes."""
        cache = PromptAssemblyCache()
        old_hash = cache.hash_content("old content")
        new_hash = cache.hash_content("new content")

        cache.set("key1", {"data": "old"}, old_hash)
        result = cache.get("key1", new_hash)

        assert result is None
        assert cache.get_stats().misses == 1

    def test_cache_miss_no_entry(self):
        """Test cache miss when key doesn't exist."""
        cache = PromptAssemblyCache()
        result = cache.get("nonexistent", "somehash")

        assert result is None
        assert cache.get_stats().misses == 1

    def test_eviction(self):
        """Test cache eviction when at capacity."""
        cache = PromptAssemblyCache(max_entries=2)

        cache.set("key1", "value1", "hash1")
        cache.set("key2", "value2", "hash2")
        cache.set("key3", "value3", "hash3")

        stats = cache.get_stats()
        assert stats.evictions == 1
        assert stats.total_entries == 2

    def test_clear(self):
        """Test cache clearing."""
        cache = PromptAssemblyCache()
        cache.set("key1", "value1", "hash1")
        cache.set("key2", "value2", "hash2")

        cache.clear()

        assert cache.get_stats().total_entries == 0

    def test_stats_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=3, misses=1)
        assert stats.hit_rate == 0.75

    def test_hash_files(self, tmp_path):
        """Test file hashing."""
        import time

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        hash1 = PromptAssemblyCache.hash_files([file1, file2])
        hash2 = PromptAssemblyCache.hash_files([file1, file2])

        assert hash1 == hash2

        # Wait for mtime to change (Windows has low resolution)
        time.sleep(0.1)

        # Modify file with different content and size
        file1.write_text("modified content that is longer")
        hash3 = PromptAssemblyCache.hash_files([file1, file2])

        assert hash1 != hash3
