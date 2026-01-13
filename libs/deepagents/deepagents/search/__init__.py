"""Semantic search engine for codebase search.

Provides embedding-based semantic search with fallback to keyword search.
"""

from deepagents.search.engine import (
    CodeSearchEngine,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    SearchResult,
)

__all__ = [
    "CodeSearchEngine",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SearchResult",
]
