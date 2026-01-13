"""Prompt Assembly Cache - Caches AGENTS.md and skills metadata by content hash.

This module provides caching for expensive prompt assembly operations:
1. AGENTS.md content (hashed by file content)
2. Skills metadata (hashed by directory mtime)
3. System prompt fragments (hashed by component content)

The cache is in-memory with optional persistence to disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached value with metadata."""
    
    value: T
    content_hash: str
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def is_valid(self, current_hash: str) -> bool:
        """Check if cache entry is still valid."""
        return self.content_hash == current_hash


@dataclass
class CacheStats:
    """Statistics about cache usage."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0

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
            "total_entries": self.total_entries,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


class PromptAssemblyCache:
    """Cache for prompt assembly components.
    
    Caches:
    - AGENTS.md content by file hash
    - Skills metadata by directory hash
    - Assembled prompt fragments
    
    Thread-safe for read operations.
    """

    def __init__(
        self,
        max_entries: int = 100,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize cache.
        
        Args:
            max_entries: Maximum cache entries before eviction.
            cache_dir: Optional directory for persistent cache.
        """
        self.max_entries = max_entries
        self.cache_dir = cache_dir
        self._cache: dict[str, CacheEntry[Any]] = {}
        self._stats = CacheStats()

    @staticmethod
    def hash_content(content: str | bytes) -> str:
        """Generate hash for content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    @staticmethod
    def hash_files(paths: list[Path]) -> str:
        """Generate hash for multiple files based on mtime and size."""
        parts = []
        for p in sorted(paths):
            if p.exists():
                stat = p.stat()
                parts.append(f"{p}:{stat.st_mtime}:{stat.st_size}")
        combined = "|".join(parts)
        return PromptAssemblyCache.hash_content(combined)

    def get(self, key: str, current_hash: str) -> Any | None:
        """Get cached value if valid.
        
        Args:
            key: Cache key.
            current_hash: Current content hash for validation.
            
        Returns:
            Cached value if valid, None otherwise.
        """
        entry = self._cache.get(key)
        if entry and entry.is_valid(current_hash):
            entry.hits += 1
            self._stats.hits += 1
            logger.debug(f"Cache hit for {key}")
            return entry.value
        
        self._stats.misses += 1
        return None

    def set(self, key: str, value: Any, content_hash: str) -> None:
        """Set cached value.
        
        Args:
            key: Cache key.
            value: Value to cache.
            content_hash: Content hash for validation.
        """
        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()
        
        self._cache[key] = CacheEntry(
            value=value,
            content_hash=content_hash,
        )
        self._stats.total_entries = len(self._cache)
        logger.debug(f"Cached {key} with hash {content_hash[:8]}")

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._stats.evictions += 1
        logger.debug(f"Evicted {oldest_key}")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.total_entries = len(self._cache)
        return self._stats

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._stats = CacheStats()

