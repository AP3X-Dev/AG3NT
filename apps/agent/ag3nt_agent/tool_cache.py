"""Tool result caching for deterministic tool operations.

This module provides LRU caching for read-only, deterministic tool results
to avoid redundant file reads, glob operations, and grep searches.

Usage:
    from ag3nt_agent.tool_cache import get_tool_cache, cached_tool

    # Manual caching
    cache = get_tool_cache()
    hit, value = cache.get("read_file", {"path": "/foo/bar.txt"})
    if not hit:
        value = read_file("/foo/bar.txt")
        cache.set("read_file", {"path": "/foo/bar.txt"}, value)

    # Decorator-based caching
    @cached_tool
    def read_file(path: str) -> str:
        ...
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger("ag3nt.tool_cache")

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class CacheEntry:
    """A single cache entry with value and metadata."""

    value: Any
    timestamp: float
    hits: int = 0
    size_bytes: int = 0


@dataclass
class CacheStats:
    """Statistics about cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "hitRate": self.hit_rate,
            "totalSizeBytes": self.total_size_bytes,
            "entryCount": self.entry_count,
        }


class ToolResultCache:
    """LRU cache for deterministic tool results.

    Thread-safe implementation with TTL support and size limits.
    """

    # Tools that are safe to cache (read-only, deterministic)
    CACHEABLE_TOOLS: set[str] = {
        "read_file",
        "glob_tool",
        "grep_tool",
        "codebase_search_tool",
        "list_directory",
    }

    def __init__(
        self,
        max_entries: int = 1000,
        max_size_bytes: int = 50 * 1024 * 1024,  # 50MB
        ttl_seconds: int = 300,  # 5 minutes
    ):
        """Initialize the cache.

        Args:
            max_entries: Maximum number of cache entries
            max_size_bytes: Maximum total size of cached values
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def _make_key(self, tool_name: str, args: dict[str, Any]) -> str:
        """Create a deterministic cache key from tool name and args."""
        # Sort args for consistent hashing
        try:
            normalized = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fall back to repr for non-JSON-serializable args
            normalized = repr(sorted(args.items()))

        key_data = f"{tool_name}:{normalized}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:24]

    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            return len(value)
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(v) for v in value)
        elif isinstance(value, dict):
            return sum(
                self._estimate_size(k) + self._estimate_size(v)
                for k, v in value.items()
            )
        else:
            # Rough estimate for other types
            try:
                return len(json.dumps(value, default=str).encode())
            except (TypeError, ValueError):
                return 100  # Default estimate

    def get(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, Any]:
        """Get a cached result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Tuple of (cache_hit, value). If cache_hit is False, value is None.
        """
        if tool_name not in self.CACHEABLE_TOOLS:
            return False, None

        key = self._make_key(tool_name, args)

        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return False, None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._remove_entry(key)
                self._stats.misses += 1
                return False, None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1

            return True, entry.value

    def set(self, tool_name: str, args: dict[str, Any], value: Any) -> None:
        """Cache a tool result.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            value: Result to cache
        """
        if tool_name not in self.CACHEABLE_TOOLS:
            return

        key = self._make_key(tool_name, args)
        size = self._estimate_size(value)

        # Skip caching very large values
        if size > self.max_size_bytes // 10:
            logger.debug(f"Skipping cache for large result: {size} bytes")
            return

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Evict entries if needed
            self._evict_if_needed(size)

            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size,
            )
            self._cache[key] = entry
            self._stats.total_size_bytes += size
            self._stats.entry_count = len(self._cache)

    def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            pattern: Optional pattern to match (currently supports tool name prefix).
                    If None, clears entire cache.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                self._stats.total_size_bytes = 0
                self._stats.entry_count = 0
                self._stats.invalidations += count
                logger.info(f"Cache cleared: {count} entries")
                return count

            # Pattern-based invalidation
            # For now, just clear all (can implement smarter matching later)
            count = len(self._cache)
            self._cache.clear()
            self._stats.total_size_bytes = 0
            self._stats.entry_count = 0
            self._stats.invalidations += count
            return count

    def invalidate_path(self, path: str) -> int:
        """Invalidate cache entries related to a file path.

        Call this when a file is modified to ensure stale data isn't served.

        Args:
            path: File path that was modified

        Returns:
            Number of entries invalidated.
        """
        # For now, invalidate all entries - can be optimized later
        # to track which entries depend on which paths
        return self.invalidate()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                invalidations=self._stats.invalidations,
                total_size_bytes=self._stats.total_size_bytes,
                entry_count=len(self._cache),
            )

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from the cache (must hold lock)."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.total_size_bytes -= entry.size_bytes

    def _evict_if_needed(self, new_size: int) -> None:
        """Evict entries to make room for new entry (must hold lock)."""
        # Evict by count
        while len(self._cache) >= self.max_entries and self._cache:
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._stats.evictions += 1

        # Evict by size
        while (
            self._stats.total_size_bytes + new_size > self.max_size_bytes
            and self._cache
        ):
            key = next(iter(self._cache))
            self._remove_entry(key)
            self._stats.evictions += 1


# Global cache instance
_tool_cache: ToolResultCache | None = None
_cache_lock = threading.Lock()


def get_tool_cache() -> ToolResultCache:
    """Get or create the global tool cache instance."""
    global _tool_cache

    if _tool_cache is None:
        with _cache_lock:
            if _tool_cache is None:
                _tool_cache = ToolResultCache()

    return _tool_cache


def cached_tool(fn: F) -> F:
    """Decorator to add caching to a tool function.

    The decorated function must have keyword arguments that can be
    serialized to JSON for cache key generation.

    Example:
        @cached_tool
        def read_file(path: str) -> str:
            with open(path) as f:
                return f.read()
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = fn.__name__
        cache = get_tool_cache()

        # Check cache
        hit, cached_value = cache.get(tool_name, kwargs)
        if hit:
            logger.debug(f"Cache hit for {tool_name}")
            return cached_value

        # Execute and cache
        result = fn(*args, **kwargs)
        cache.set(tool_name, kwargs, result)

        return result

    return wrapper  # type: ignore


def cached_tool_async(fn: F) -> F:
    """Decorator to add caching to an async tool function.

    Example:
        @cached_tool_async
        async def read_file(path: str) -> str:
            async with aiofiles.open(path) as f:
                return await f.read()
    """
    import asyncio

    @wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        tool_name = fn.__name__
        cache = get_tool_cache()

        # Check cache
        hit, cached_value = cache.get(tool_name, kwargs)
        if hit:
            logger.debug(f"Cache hit for {tool_name}")
            return cached_value

        # Execute and cache
        result = await fn(*args, **kwargs)
        cache.set(tool_name, kwargs, result)

        return result

    return wrapper  # type: ignore
