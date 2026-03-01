"""Tests for query expansion in memory search."""

import pytest
from unittest.mock import patch, MagicMock

from ag3nt_agent.memory_search import _expand_query, _STOP_WORDS


class TestExpandQuery:
    """Tests for the _expand_query() function."""

    def test_returns_original_plus_keywords(self):
        """Result includes original query, len >= 2."""
        result = _expand_query("python async patterns")
        assert result[0] == "python async patterns"
        assert len(result) >= 2

    def test_single_word_query(self):
        """Single word returns list with at least the original."""
        result = _expand_query("asyncio")
        assert result[0] == "asyncio"
        assert len(result) >= 1

    def test_empty_query_returns_list_with_original(self):
        """Empty string -> ['']."""
        result = _expand_query("")
        assert result == [""]

    def test_stop_words_filtered(self):
        """Stop words like 'how', 'to', 'the' don't appear as standalone variants."""
        result = _expand_query("how to use the database")
        # The original query is always first (and may contain stop words)
        # But standalone expanded variants should not be pure stop words
        for variant in result[1:]:
            # Each variant should contain at least one non-stop-word token
            tokens = variant.lower().split()
            non_stop = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
            assert len(non_stop) > 0, f"Variant '{variant}' contains only stop words"

    def test_generates_keyword_subsets(self):
        """Bigrams from keywords present in results."""
        result = _expand_query("python async database migration")
        # Should generate bigrams from keywords
        # At least some 2-word combinations should appear
        bigrams_found = [v for v in result[1:] if len(v.split()) == 2]
        assert len(bigrams_found) > 0, "No bigrams found in expanded queries"


class TestHybridSearchWithExpansion:
    """Tests that _expand_query is wired into hybrid search."""

    def test_expansion_called_during_search(self):
        """Monkeypatch to verify _expand_query is called by _hybrid_search."""
        from ag3nt_agent.memory_search import MemoryVectorStore

        store = MemoryVectorStore()
        # Set up minimal state so _hybrid_search doesn't crash before reaching expansion
        store._initialized = True
        store._bm25_index = MagicMock()
        store._bm25_index.get_normalized_score = MagicMock(return_value=0.5)
        store._embeddings = MagicMock()
        store._embeddings.embed_query = MagicMock(return_value=[0.1] * 128)
        store._metadata = [
            {"text": "some memory text", "source": "test.md", "mtime": 1000000.0}
        ]

        # Mock FAISS index
        import numpy as np

        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.5]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )
        store._index = mock_index
        store._search_config = MagicMock()
        store._search_config.enable_hybrid = True
        store._search_config.enable_bm25 = True
        store._search_config.semantic_weight = 0.5
        store._search_config.bm25_weight = 0.25
        store._search_config.keyword_weight = 0.15
        store._search_config.recency_weight = 0.10
        store._search_config.recency_decay_days = 30

        with patch(
            "ag3nt_agent.memory_search._expand_query",
            return_value=["test query", "test", "query"],
        ) as mock_expand:
            store._hybrid_search("test query", top_k=5)
            mock_expand.assert_called_once_with("test query")
