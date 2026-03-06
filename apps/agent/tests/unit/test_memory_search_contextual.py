"""Tests for contextual retrieval integration in memory search."""
import pytest
from unittest.mock import MagicMock, patch


class TestBM25UsesContextualizedText:
    """Test that BM25 index is built on contextualized_text."""

    def test_bm25_build_uses_contextualized_text(self):
        """_build_bm25_index should use contextualized_text when available."""
        from ag3nt_agent.memory_search import MemoryVectorStore, SearchConfig

        store = MemoryVectorStore.__new__(MemoryVectorStore)
        store._search_config = SearchConfig()
        store._metadata = [
            {"text": "raw", "contextualized_text": "enriched: raw"},
            {"text": "plain", "contextualized_text": "enriched: plain"},
        ]

        store._build_bm25_index()

        assert store._bm25_index is not None
        # "enriched" should score positively since it's in contextualized_text
        score = store._bm25_index.score("enriched", 0)
        assert score > 0

    def test_bm25_falls_back_to_text_without_contextualized(self):
        """_build_bm25_index should use text when contextualized_text is absent."""
        from ag3nt_agent.memory_search import MemoryVectorStore, SearchConfig

        store = MemoryVectorStore.__new__(MemoryVectorStore)
        store._search_config = SearchConfig()
        store._metadata = [
            {"text": "original content only"},
        ]

        store._build_bm25_index()

        assert store._bm25_index is not None
        score = store._bm25_index.score("original", 0)
        assert score > 0


class TestKeywordScoreUsesOriginalText:
    """Test that keyword scoring in hybrid search uses original text."""

    def test_keyword_score_on_original_text(self):
        """_compute_keyword_score should match against original text."""
        from ag3nt_agent.memory_search import _compute_keyword_score

        # Keyword score should match "raw" in original text
        score = _compute_keyword_score("raw", "raw content here")
        assert score > 0

        # Should NOT match contextual prefix that only exists in contextualized_text
        score = _compute_keyword_score("enriched", "raw content here")
        assert score == 0.0
