"""Contextual retrieval for AG3NT memory search.

This module adds LLM-generated context prefixes to memory chunks before
embedding, following Anthropic's contextual retrieval approach. Each chunk
is enriched with a short description of where it fits within its source
document, dramatically improving retrieval accuracy.

Components:
- ContextCache: SQLite-backed cache for context prefixes (avoids re-calling LLM)
- ContextEnricher: Orchestrates LLM-based chunk enrichment with caching

Usage:
    from ag3nt_agent.context_enrichment import ContextEnricher, get_context_enricher

    enricher = get_context_enricher()
    enriched_chunks = enricher.enrich_chunks(chunks, source_documents)
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = Path.home() / ".ag3nt" / "cache"
SOURCE_DOC_MAX_CHARS = 5000
CONTEXT_PREFIX_MAX_TOKENS = 100


def _create_model():
    """Create the LLM model for context generation.

    Delegates to deepagents_runtime._create_model().
    """
    from ag3nt_agent.deepagents_runtime import _create_model as _dm_create_model

    return _dm_create_model()


def _truncate_document(text: str, max_chars: int = SOURCE_DOC_MAX_CHARS) -> str:
    """Truncate a source document while preserving start and end.

    Keeps 80% from the beginning and 20% from the end, with a truncation
    marker in between.

    Args:
        text: The full source document text.
        max_chars: Maximum character length (default 5000).

    Returns:
        The truncated text, or the original if within limit.
    """
    if len(text) <= max_chars:
        return text

    start_chars = int(max_chars * 0.8)
    end_chars = int(max_chars * 0.2)

    return text[:start_chars] + "\n[...]\n" + text[-end_chars:]


def _build_context_prompt(source_document: str, chunk_text: str) -> str:
    """Build the LLM prompt for contextual retrieval.

    Uses Anthropic's recommended prompt structure: provide the full (or
    truncated) document and ask the model to situate the chunk.

    Args:
        source_document: The full source document (possibly truncated).
        chunk_text: The specific chunk to contextualize.

    Returns:
        The formatted prompt string.
    """
    return (
        "<document>\n"
        f"{source_document}\n"
        "</document>\n"
        "Here is the chunk we want to situate within the whole document:\n"
        "<chunk>\n"
        f"{chunk_text}\n"
        "</chunk>\n"
        "Please give a short succinct context to situate this chunk within "
        "the overall document for the purposes of improving search retrieval "
        "of the chunk. Answer only with the succinct context and nothing else."
    )


class ContextCache:
    """SQLite-backed cache for context prefixes.

    Stores LLM-generated context prefixes keyed by content hash so that
    unchanged chunks do not require repeated LLM calls.

    Schema:
        context_cache(
            content_hash TEXT PRIMARY KEY,
            context_prefix TEXT NOT NULL,
            source TEXT,
            created_at REAL NOT NULL
        )
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the context cache.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                ~/.ag3nt/cache/context_enrichment.db
        """
        self._db_path = db_path or (CACHE_DIR / "context_enrichment.db")
        self._conn: sqlite3.Connection | None = None
        self._initialized = False
        self._hits = 0
        self._misses = 0

    def _ensure_initialized(self) -> None:
        """Initialize database connection and create schema if needed."""
        if self._initialized:
            return

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS context_cache (
                content_hash TEXT PRIMARY KEY,
                context_prefix TEXT NOT NULL,
                source TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()
        self._initialized = True
        logger.debug("ContextCache initialized at %s", self._db_path)

    def get(self, content_hash: str) -> str | None:
        """Retrieve a cached context prefix.

        Args:
            content_hash: The hash key for the chunk.

        Returns:
            The cached context prefix string, or None if not found.
        """
        self._ensure_initialized()

        cursor = self._conn.execute(
            "SELECT context_prefix FROM context_cache WHERE content_hash = ?",
            (content_hash,),
        )
        row = cursor.fetchone()

        if row is None:
            self._misses += 1
            return None

        self._hits += 1
        return row["context_prefix"]

    def put(self, content_hash: str, context_prefix: str, source: str | None = None) -> None:
        """Store or update a context prefix in the cache.

        Args:
            content_hash: The hash key for the chunk.
            context_prefix: The LLM-generated context prefix.
            source: Optional source identifier (e.g. file name).
        """
        self._ensure_initialized()

        self._conn.execute(
            """
            INSERT OR REPLACE INTO context_cache
            (content_hash, context_prefix, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (content_hash, context_prefix, source, time.time()),
        )
        self._conn.commit()

    def invalidate(self, content_hash: str) -> None:
        """Remove a cached entry.

        Args:
            content_hash: The hash key to remove. No-op if not present.
        """
        self._ensure_initialized()

        self._conn.execute(
            "DELETE FROM context_cache WHERE content_hash = ?",
            (content_hash,),
        )
        self._conn.commit()

    def get_stats(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dict with keys: hits, misses, stored.
        """
        self._ensure_initialized()

        count = self._conn.execute(
            "SELECT COUNT(*) FROM context_cache"
        ).fetchone()[0]

        return {
            "hits": self._hits,
            "misses": self._misses,
            "stored": count,
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False


class ContextEnricher:
    """Enriches memory chunks with LLM-generated context prefixes.

    For each chunk, calls the LLM with the source document and chunk text
    to produce a short context description. The result is prepended to the
    chunk text as ``contextualized_text`` for improved retrieval.

    Features:
    - Caches context prefixes in ContextCache to avoid repeated LLM calls
    - Graceful degradation: on any failure, falls back to original text
    - Feature flag: set AG3NT_CONTEXTUAL_RETRIEVAL=0 to disable enrichment
    - Truncates large source documents to keep prompts manageable
    """

    def __init__(self, cache: ContextCache | None = None) -> None:
        """Initialize the enricher.

        Args:
            cache: Optional ContextCache instance. Creates a default one
                if not provided.
        """
        self._cache = cache or ContextCache()

    def enrich_chunks(
        self,
        chunks: list[dict[str, Any]],
        source_documents: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Enrich chunks with contextualized text.

        Each chunk dict gets a new ``contextualized_text`` field containing
        the LLM-generated context prefix followed by the original text.

        Args:
            chunks: List of chunk dicts (must have ``text``, ``source``,
                ``content_hash`` keys).
            source_documents: Mapping of source name to full document text.

        Returns:
            The same list of chunk dicts, each augmented with
            ``contextualized_text``.
        """
        # Check feature flag
        if os.environ.get("AG3NT_CONTEXTUAL_RETRIEVAL", "1") == "0":
            logger.info("Contextual retrieval disabled via AG3NT_CONTEXTUAL_RETRIEVAL=0")
            for chunk in chunks:
                chunk["contextualized_text"] = chunk["text"]
            return chunks

        # Try to create the model once for all chunks
        model = None
        try:
            model = _create_model()
        except Exception as e:
            logger.warning("Failed to create model for contextual retrieval: %s", e)
            for chunk in chunks:
                chunk["contextualized_text"] = chunk["text"]
            return chunks

        for chunk in chunks:
            chunk["contextualized_text"] = self._enrich_single_chunk(
                chunk, source_documents, model
            )

        return chunks

    def _enrich_single_chunk(
        self,
        chunk: dict[str, Any],
        source_documents: dict[str, str],
        model: Any,
    ) -> str:
        """Enrich a single chunk, using cache when available.

        Args:
            chunk: The chunk dict.
            source_documents: Source name -> full text mapping.
            model: The LLM model instance.

        Returns:
            The contextualized text (prefix + original), or just the
            original text on failure.
        """
        content_hash = chunk.get("content_hash", "")
        original_text = chunk["text"]

        # Check cache first
        cached_prefix = self._cache.get(content_hash)
        if cached_prefix is not None:
            return f"{cached_prefix}\n\n{original_text}"

        # Build prompt with (possibly truncated) source document
        source_name = chunk.get("source", "")
        source_doc = source_documents.get(source_name, "")
        truncated_doc = _truncate_document(source_doc)
        prompt = _build_context_prompt(truncated_doc, original_text)

        try:
            response = model.invoke(prompt)
            prefix = response.content if hasattr(response, "content") else str(response)

            # Cache the result
            self._cache.put(content_hash, prefix, source_name)

            return f"{prefix}\n\n{original_text}"
        except Exception as e:
            logger.warning("LLM enrichment failed for chunk %s: %s", content_hash, e)
            return original_text


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_context_cache: ContextCache | None = None
_context_enricher: ContextEnricher | None = None


def get_context_cache() -> ContextCache:
    """Get the global ContextCache instance."""
    global _context_cache
    if _context_cache is None:
        _context_cache = ContextCache()
    return _context_cache


def get_context_enricher() -> ContextEnricher:
    """Get the global ContextEnricher instance."""
    global _context_enricher
    if _context_enricher is None:
        _context_enricher = ContextEnricher(cache=get_context_cache())
    return _context_enricher


def reset_context_enricher() -> None:
    """Reset the global instances (for testing)."""
    global _context_cache, _context_enricher
    if _context_cache is not None:
        _context_cache.close()
    _context_cache = None
    _context_enricher = None
