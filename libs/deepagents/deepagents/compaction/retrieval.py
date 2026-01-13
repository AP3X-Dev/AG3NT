"""RetrievalIndex for on-demand retrieval of relevant snippets.

This module provides full-text search over stored artifacts using SQLite FTS5.
Enables efficient retrieval of relevant content without loading entire artifacts.

Thread Safety:
    This module uses thread-local storage for SQLite connections to ensure
    safe operation in multi-threaded environments. Each thread gets its own
    connection that is lazily initialized on first use.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents.compaction.artifact_store import ArtifactStore
    from deepagents.compaction.config import CompactionConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with snippet and metadata."""

    artifact_id: str
    snippet: str
    line_number: int
    score: float
    context_before: str = ""
    context_after: str = ""


class RetrievalIndex:
    """Full-text search index over stored artifacts.

    Uses SQLite FTS5 for efficient text search. Indexes artifacts
    as they are stored and provides query-based retrieval.

    Thread Safety:
        Uses thread-local storage for SQLite connections. Each thread gets
        its own connection, lazily initialized on first access. A threading
        lock protects the shared indexed_artifacts set.

    Args:
        config: Compaction configuration.
        store: ArtifactStore for reading artifact content.
    """

    def __init__(
        self,
        config: CompactionConfig,
        store: ArtifactStore,
    ) -> None:
        self.config = config
        self.store = store
        self._db_path = config.get_index_path()

        # Thread-local storage for SQLite connections
        self._local = threading.local()

        # Lock for protecting shared state
        self._lock = threading.Lock()

        # Shared set of indexed artifacts (protected by _lock)
        self._indexed_artifacts: set[str] = set()

        # Initialize in current thread and load existing artifacts
        self._ensure_tables_exist()
        self._load_indexed_artifacts()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection, creating if needed.

        Returns:
            sqlite3.Connection for the current thread.
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._db_path))
            logger.debug(f"Created new SQLite connection for thread {threading.current_thread().name}")
        return self._local.conn

    def _ensure_tables_exist(self) -> None:
        """Ensure the FTS5 tables exist (called once per thread on first use)."""
        conn = self._get_connection()
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS artifact_chunks USING fts5(
                artifact_id,
                line_number,
                content,
                tokenize='porter unicode61'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_artifacts (
                artifact_id TEXT PRIMARY KEY,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    def _load_indexed_artifacts(self) -> None:
        """Load the set of already-indexed artifacts from the database."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT artifact_id FROM indexed_artifacts")
        with self._lock:
            self._indexed_artifacts = {row[0] for row in cursor.fetchall()}

    def _chunk_content(
        self,
        content: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[tuple[int, str]]:
        """Split content into overlapping chunks with line numbers.

        Args:
            content: The content to chunk.
            chunk_size: Target size of each chunk in characters.
            overlap: Overlap between chunks.

        Returns:
            List of (line_number, chunk_text) tuples.
        """
        lines = content.split("\n")
        chunks: list[tuple[int, str]] = []

        current_chunk: list[str] = []
        current_size = 0
        chunk_start_line = 1

        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > chunk_size and current_chunk:
                # Save current chunk
                chunks.append((chunk_start_line, "\n".join(current_chunk)))

                # Start new chunk with overlap
                overlap_lines = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_lines.copy()
                current_size = sum(len(l) + 1 for l in current_chunk)
                chunk_start_line = max(1, i - len(overlap_lines))

            current_chunk.append(line)
            current_size += line_size

        # Don't forget the last chunk
        if current_chunk:
            chunks.append((chunk_start_line, "\n".join(current_chunk)))

        return chunks

    def index_artifact(self, artifact_id: str) -> int:
        """Index an artifact for full-text search.

        Thread-safe: Uses thread-local connection and lock for shared state.

        Args:
            artifact_id: The artifact ID to index.

        Returns:
            Number of chunks indexed.
        """
        # Quick check without lock (may have false negatives, but safe)
        with self._lock:
            if artifact_id in self._indexed_artifacts:
                return 0

        content = self.store.read_artifact(artifact_id)
        if content is None or isinstance(content, bytes):
            logger.warning(f"Cannot index artifact {artifact_id}: not text content")
            return 0

        chunks = self._chunk_content(content)

        # Get thread-local connection
        conn = self._get_connection()

        for line_num, chunk_text in chunks:
            conn.execute(
                "INSERT INTO artifact_chunks (artifact_id, line_number, content) VALUES (?, ?, ?)",
                (artifact_id, line_num, chunk_text),
            )

        conn.execute(
            "INSERT OR REPLACE INTO indexed_artifacts (artifact_id) VALUES (?)",
            (artifact_id,),
        )
        conn.commit()

        # Update shared set with lock
        with self._lock:
            self._indexed_artifacts.add(artifact_id)

        logger.debug(f"Indexed artifact {artifact_id}: {len(chunks)} chunks")
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        artifact_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant snippets matching a query.

        Thread-safe: Uses thread-local SQLite connection.

        Args:
            query: The search query.
            artifact_id: Optional artifact ID to limit search to.
            top_k: Maximum number of results (defaults to config).

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        if top_k is None:
            top_k = self.config.retrieval_top_k

        # Escape special FTS5 characters
        safe_query = re.sub(r'[^\w\s]', ' ', query)
        terms = safe_query.split()
        if not terms:
            return []

        # Build FTS5 query
        fts_query = " OR ".join(terms)

        sql = """
            SELECT artifact_id, line_number, content, bm25(artifact_chunks) as score
            FROM artifact_chunks
            WHERE artifact_chunks MATCH ?
        """
        params: list[str | int] = [fts_query]

        if artifact_id:
            sql += " AND artifact_id = ?"
            params.append(artifact_id)

        sql += " ORDER BY score LIMIT ?"
        params.append(top_k)

        # Get thread-local connection
        conn = self._get_connection()
        cursor = conn.execute(sql, params)
        results = []

        for row in cursor.fetchall():
            results.append(RetrievalResult(
                artifact_id=row[0],
                snippet=row[2],
                line_number=row[1],
                score=abs(row[3]),  # BM25 returns negative scores
            ))

        return results

    def search_with_context(
        self,
        query: str,
        *,
        artifact_id: str | None = None,
        top_k: int | None = None,
        context_lines: int = 3,
    ) -> list[RetrievalResult]:
        """Search with additional context lines around matches.

        Args:
            query: The search query.
            artifact_id: Optional artifact ID to limit search to.
            top_k: Maximum number of results.
            context_lines: Number of context lines to include.

        Returns:
            List of RetrievalResult objects with context.
        """
        results = self.search(query, artifact_id=artifact_id, top_k=top_k)

        # Add context from original artifacts
        for result in results:
            content = self.store.read_artifact(result.artifact_id)
            if content is None or isinstance(content, bytes):
                continue

            lines = content.split("\n")
            start = max(0, result.line_number - 1 - context_lines)
            end = min(len(lines), result.line_number + context_lines)

            if start < result.line_number - 1:
                result.context_before = "\n".join(lines[start:result.line_number - 1])
            if end > result.line_number:
                result.context_after = "\n".join(lines[result.line_number:end])

        return results

    def index_all_artifacts(self) -> int:
        """Index all artifacts that haven't been indexed yet.

        Returns:
            Total number of chunks indexed.
        """
        total_chunks = 0
        for meta in self.store.list_artifacts(limit=1000):
            with self._lock:
                already_indexed = meta.artifact_id in self._indexed_artifacts
            if not already_indexed:
                chunks = self.index_artifact(meta.artifact_id)
                total_chunks += chunks
        return total_chunks

    def get_indexed_count(self) -> int:
        """Get the number of indexed artifacts."""
        with self._lock:
            return len(self._indexed_artifacts)

    def close(self) -> None:
        """Close the current thread's database connection.

        Note: This only closes the connection for the calling thread.
        Other threads' connections remain open until they call close()
        or are garbage collected.
        """
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
            logger.debug(f"Closed SQLite connection for thread {threading.current_thread().name}")

    def close_all(self) -> None:
        """Close all thread-local connections.

        Note: This method should be called when the RetrievalIndex is being
        disposed of. It only closes the current thread's connection directly;
        other threads' connections will be closed when they are garbage collected.
        """
        self.close()
