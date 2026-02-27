"""Tests for tool result caching."""

import pytest
import time
from unittest.mock import patch

from ag3nt_agent.tool_cache import (
    ToolResultCache,
    get_tool_cache,
    cached_tool,
    cached_tool_async,
    CacheStats,
)


class TestToolResultCache:
    """Tests for ToolResultCache class."""

    def test_cache_miss_for_uncached_key(self):
        """Test that cache returns miss for uncached key."""
        cache = ToolResultCache()
        hit, value = cache.get("read_file", {"path": "/foo/bar.txt"})
        assert hit is False
        assert value is None

    def test_cache_hit_after_set(self):
        """Test that cache returns hit after setting value."""
        cache = ToolResultCache()
        cache.set("read_file", {"path": "/foo/bar.txt"}, "file contents")

        hit, value = cache.get("read_file", {"path": "/foo/bar.txt"})
        assert hit is True
        assert value == "file contents"

    def test_cache_miss_for_different_args(self):
        """Test that different args produce cache miss."""
        cache = ToolResultCache()
        cache.set("read_file", {"path": "/foo/bar.txt"}, "contents1")

        hit, value = cache.get("read_file", {"path": "/foo/other.txt"})
        assert hit is False
        assert value is None

    def test_cache_ignores_non_cacheable_tools(self):
        """Test that non-cacheable tools are not cached."""
        cache = ToolResultCache()
        cache.set("shell_tool", {"command": "ls"}, "output")

        hit, value = cache.get("shell_tool", {"command": "ls"})
        assert hit is False

    def test_cache_ttl_expiration(self):
        """Test that cache entries expire after TTL."""
        cache = ToolResultCache(ttl_seconds=0.1)
        cache.set("read_file", {"path": "/foo.txt"}, "contents")

        # Should hit immediately
        hit, _ = cache.get("read_file", {"path": "/foo.txt"})
        assert hit is True

        # Should miss after TTL
        time.sleep(0.15)
        hit, _ = cache.get("read_file", {"path": "/foo.txt"})
        assert hit is False

    def test_cache_eviction_by_count(self):
        """Test that cache evicts entries when max count reached."""
        cache = ToolResultCache(max_entries=3)

        # Fill cache
        for i in range(5):
            cache.set("read_file", {"path": f"/file{i}.txt"}, f"contents{i}")

        # Only last 3 should be cached
        stats = cache.get_stats()
        assert stats.entry_count == 3
        assert stats.evictions == 2

    def test_cache_eviction_by_size(self):
        """Test that cache evicts entries when max size reached."""
        cache = ToolResultCache(max_size_bytes=100)

        # Add entries that exceed size limit
        cache.set("read_file", {"path": "/a.txt"}, "x" * 40)
        cache.set("read_file", {"path": "/b.txt"}, "y" * 40)
        cache.set("read_file", {"path": "/c.txt"}, "z" * 40)

        stats = cache.get_stats()
        assert stats.entry_count <= 2  # At least one should be evicted

    def test_cache_invalidate_all(self):
        """Test that invalidate clears all entries."""
        cache = ToolResultCache()
        cache.set("read_file", {"path": "/a.txt"}, "a")
        cache.set("read_file", {"path": "/b.txt"}, "b")

        count = cache.invalidate()
        assert count == 2

        stats = cache.get_stats()
        assert stats.entry_count == 0

    def test_cache_invalidate_path(self):
        """Test that invalidate_path clears cache."""
        cache = ToolResultCache()
        cache.set("read_file", {"path": "/a.txt"}, "a")

        count = cache.invalidate_path("/a.txt")
        assert count >= 1

    def test_cache_stats(self):
        """Test that cache stats are tracked correctly."""
        cache = ToolResultCache()

        # Generate some activity
        cache.get("read_file", {"path": "/miss.txt"})  # Miss
        cache.set("read_file", {"path": "/hit.txt"}, "contents")
        cache.get("read_file", {"path": "/hit.txt"})  # Hit
        cache.get("read_file", {"path": "/hit.txt"})  # Hit

        stats = cache.get_stats()
        assert stats.misses == 1
        assert stats.hits == 2
        assert stats.entry_count == 1
        assert stats.hit_rate == 2 / 3

    def test_cache_handles_large_values(self):
        """Test that very large values are not cached."""
        # Cache with 1000 byte limit, threshold is 100 bytes (10%)
        cache = ToolResultCache(max_size_bytes=1000)

        # Small value (under threshold) should be cached
        small_value = "x" * 50
        cache.set("read_file", {"path": "/small.txt"}, small_value)
        hit, _ = cache.get("read_file", {"path": "/small.txt"})
        assert hit is True

        # Large value (over 10% of max) should NOT be cached
        large_value = "y" * 200  # 200 > 100 threshold
        cache.set("read_file", {"path": "/large.txt"}, large_value)
        hit, _ = cache.get("read_file", {"path": "/large.txt"})
        assert hit is False


class TestCachedToolDecorator:
    """Tests for the @cached_tool decorator."""

    def setup_method(self):
        """Clear the global cache before each test."""
        cache = get_tool_cache()
        cache.invalidate()

    def test_cached_tool_returns_cached_value(self):
        """Test that decorated function returns cached value on second call."""
        call_count = 0

        @cached_tool
        def read_file(path: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"contents of {path}"

        # Use unique path to avoid interference from other tests
        unique_path = f"/test_cached_{id(self)}.txt"

        # First call
        result1 = read_file(path=unique_path)
        assert result1 == f"contents of {unique_path}"
        assert call_count == 1

        # Second call should use cache
        result2 = read_file(path=unique_path)
        assert result2 == f"contents of {unique_path}"
        assert call_count == 1  # Not incremented

    def test_cached_tool_different_args_not_cached(self):
        """Test that different args result in separate cache entries."""
        call_count = 0

        @cached_tool
        def read_file(path: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"contents of {path}"

        # Use unique paths
        unique_id = id(self)
        path1 = f"/test_diff_a_{unique_id}.txt"
        path2 = f"/test_diff_b_{unique_id}.txt"

        result1 = read_file(path=path1)
        result2 = read_file(path=path2)

        assert call_count == 2
        assert result1 != result2


class TestCachedToolAsyncDecorator:
    """Tests for the @cached_tool_async decorator."""

    def setup_method(self):
        """Clear the global cache before each test."""
        cache = get_tool_cache()
        cache.invalidate()

    @pytest.mark.asyncio
    async def test_cached_tool_async_returns_cached_value(self):
        """Test that async decorated function returns cached value."""
        call_count = 0

        @cached_tool_async
        async def read_file(path: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"contents of {path}"

        # Use unique path
        unique_path = f"/test_async_{id(self)}.txt"

        # First call
        result1 = await read_file(path=unique_path)
        assert result1 == f"contents of {unique_path}"
        assert call_count == 1

        # Second call should use cache
        result2 = await read_file(path=unique_path)
        assert result2 == f"contents of {unique_path}"
        assert call_count == 1


class TestGetToolCache:
    """Tests for the global cache singleton."""

    def test_get_tool_cache_returns_singleton(self):
        """Test that get_tool_cache returns the same instance."""
        cache1 = get_tool_cache()
        cache2 = get_tool_cache()
        assert cache1 is cache2

    def test_cache_stats_dict_format(self):
        """Test that stats can be converted to dict."""
        stats = CacheStats(hits=10, misses=5, evictions=2, invalidations=1)
        d = stats.to_dict()

        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["hitRate"] == 10 / 15
