"""Semantic search engine for codebase search.

Provides embedding-based semantic search with fallback to keyword search.
"""

from deepagents.search.engine import (
    CodeSearchEngine,
    SearchResult,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
)

__all__ = [
    "CodeSearchEngine",
    "SearchResult",
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
]

