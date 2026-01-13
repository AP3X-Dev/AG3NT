"""Semantic search engine with embedding support.

Provides:
- In-memory vector search for code snippets
- LangChain embeddings integration
- Fallback to keyword search when embeddings unavailable
"""

from __future__ import annotations

import logging
import math
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result with relevance score."""

    path: str
    """File path."""

    line: int
    """Line number (1-based)."""

    text: str
    """Matched text content."""

    score: float
    """Relevance score (0-1, higher is better)."""

    match_type: str = "keyword"
    """Type of match: 'semantic' or 'keyword'."""

    context: str = ""
    """Additional context about why this matched."""


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @abstractmethod
    async def embed_query(self, query: str) -> list[float]:
        """Embed a query for search.

        Args:
            query: Query text.

        Returns:
            Embedding vector.
        """
        ...


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings via LangChain."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """Initialize with embedding model.

        Args:
            model: OpenAI embedding model name.
        """
        self.model = model
        self._embeddings: Any = None

    def _get_embeddings(self) -> Any:
        """Lazy load embeddings."""
        if self._embeddings is None:
            try:
                from langchain_openai import OpenAIEmbeddings

                self._embeddings = OpenAIEmbeddings(model=self.model)
            except ImportError:
                raise ImportError(
                    "langchain-openai not installed. "
                    "Install with: pip install langchain-openai"
                )
        return self._embeddings

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI."""
        embeddings = self._get_embeddings()
        return await embeddings.aembed_documents(texts)

    async def embed_query(self, query: str) -> list[float]:
        """Embed query using OpenAI."""
        embeddings = self._get_embeddings()
        return await embeddings.aembed_query(query)


@dataclass
class CodeChunk:
    """A chunk of code for indexing."""

    path: str
    line_start: int
    line_end: int
    text: str
    embedding: list[float] | None = None


class CodeSearchEngine:
    """Semantic code search engine with embedding support."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        chunk_size: int = 20,
        chunk_overlap: int = 5,
    ):
        """Initialize the search engine.

        Args:
            embedding_provider: Provider for text embeddings (optional).
            chunk_size: Number of lines per chunk.
            chunk_overlap: Overlap between chunks.
        """
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._index: list[CodeChunk] = []
        self._indexed_files: set[str] = set()

    @property
    def has_embeddings(self) -> bool:
        """Check if embeddings are available."""
        return self.embedding_provider is not None

    def clear_index(self) -> None:
        """Clear the search index."""
        self._index.clear()
        self._indexed_files.clear()

    async def index_file(self, path: str, content: str) -> int:
        """Index a file's content.

        Args:
            path: File path.
            content: File content.

        Returns:
            Number of chunks indexed.
        """
        if path in self._indexed_files:
            return 0

        lines = content.split("\n")
        chunks = []

        # Create overlapping chunks
        for i in range(0, len(lines), self.chunk_size - self.chunk_overlap):
            chunk_lines = lines[i : i + self.chunk_size]
            if not chunk_lines:
                continue

            chunk = CodeChunk(
                path=path,
                line_start=i + 1,
                line_end=i + len(chunk_lines),
                text="\n".join(chunk_lines),
            )
            chunks.append(chunk)

        # Embed chunks if provider available
        if self.embedding_provider and chunks:
            try:
                texts = [c.text for c in chunks]
                embeddings = await self.embedding_provider.embed_texts(texts)
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
            except Exception as e:
                logger.warning(f"Failed to embed chunks for {path}: {e}")

        self._index.extend(chunks)
        self._indexed_files.add(path)
        return len(chunks)

    async def search(
        self,
        query: str,
        limit: int = 20,
        min_score: float = 0.1,
    ) -> list[SearchResult]:
        """Search the index.

        Args:
            query: Search query.
            limit: Maximum results.
            min_score: Minimum relevance score.

        Returns:
            List of search results.
        """
        if not self._index:
            return []

        # Try semantic search if embeddings available
        if self.has_embeddings and any(c.embedding for c in self._index):
            try:
                return await self._semantic_search(query, limit, min_score)
            except Exception as e:
                logger.warning(f"Semantic search failed, falling back to keyword: {e}")

        # Fall back to keyword search
        return self._keyword_search(query, limit, min_score)

    async def _semantic_search(
        self,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Perform semantic search using embeddings."""
        query_embedding = await self.embedding_provider.embed_query(query)

        results = []
        for chunk in self._index:
            if chunk.embedding is None:
                continue

            score = self._cosine_similarity(query_embedding, chunk.embedding)
            if score >= min_score:
                results.append(
                    SearchResult(
                        path=chunk.path,
                        line=chunk.line_start,
                        text=chunk.text[:500],  # Truncate
                        score=score,
                        match_type="semantic",
                        context=f"Lines {chunk.line_start}-{chunk.line_end}",
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: -r.score)
        return results[:limit]

    def _keyword_search(
        self,
        query: str,
        limit: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Perform keyword-based search."""
        # Extract keywords
        words = re.findall(r"\b\w+\b", query.lower())
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "for", "and", "or",
            "but", "in", "on", "at", "to", "from", "with", "by", "that", "this",
            "it", "its", "of", "all", "find", "where", "how", "what", "when",
        }
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        if not keywords:
            return []

        results = []
        for chunk in self._index:
            chunk_lower = chunk.text.lower()
            matches = sum(1 for kw in keywords if kw in chunk_lower)
            if matches == 0:
                continue

            score = matches / len(keywords)
            if score >= min_score:
                results.append(
                    SearchResult(
                        path=chunk.path,
                        line=chunk.line_start,
                        text=chunk.text[:500],
                        score=score,
                        match_type="keyword",
                        context=f"Matched {matches}/{len(keywords)} keywords",
                    )
                )

        results.sort(key=lambda r: -r.score)
        return results[:limit]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

