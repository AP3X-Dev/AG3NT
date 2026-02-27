"""Tests for contextual retrieval: ContextCache and ContextEnricher."""

import hashlib
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestContextCache:
    """Test suite for the SQLite-backed ContextCache."""

    def _make_cache(self, tmp_path):
        """Helper to create a ContextCache with a temp DB path."""
        from ag3nt_agent.context_enrichment import ContextCache

        db_path = tmp_path / "test_context.db"
        return ContextCache(db_path=db_path)

    def test_get_returns_none_for_missing_key(self, tmp_path):
        """get() returns None when the content_hash is not in the cache."""
        cache = self._make_cache(tmp_path)
        result = cache.get("nonexistent_hash_abc123")
        assert result is None

    def test_put_and_get_roundtrip(self, tmp_path):
        """put() stores a value that get() can retrieve."""
        cache = self._make_cache(tmp_path)
        cache.put("hash_001", "This chunk is about user preferences.", "memory")
        result = cache.get("hash_001")
        assert result == "This chunk is about user preferences."

    def test_put_overwrites_existing(self, tmp_path):
        """put() with the same hash overwrites the previous value."""
        cache = self._make_cache(tmp_path)
        cache.put("hash_002", "Original context prefix.", "memory")
        cache.put("hash_002", "Updated context prefix.", "memory")
        result = cache.get("hash_002")
        assert result == "Updated context prefix."

    def test_invalidate_removes_entry(self, tmp_path):
        """invalidate() removes the entry so get() returns None."""
        cache = self._make_cache(tmp_path)
        cache.put("hash_003", "Some context.", "memory")
        assert cache.get("hash_003") is not None
        cache.invalidate("hash_003")
        assert cache.get("hash_003") is None

    def test_invalidate_nonexistent_is_noop(self, tmp_path):
        """invalidate() on a non-existent key does not raise."""
        cache = self._make_cache(tmp_path)
        # Should not raise
        cache.invalidate("does_not_exist")

    def test_stats_tracking(self, tmp_path):
        """get_stats() accurately counts hits, misses, and stored entries."""
        cache = self._make_cache(tmp_path)

        # Start: 0 entries, 0 hits, 0 misses
        stats = cache.get_stats()
        assert stats["stored"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

        # Miss
        cache.get("missing_key")
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

        # Put two entries
        cache.put("key_a", "prefix a", "src")
        cache.put("key_b", "prefix b", "src")
        stats = cache.get_stats()
        assert stats["stored"] == 2

        # Hit
        cache.get("key_a")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestContextEnricher:
    """Test suite for the LLM-based ContextEnricher."""

    def _make_enricher(self, tmp_path):
        """Helper to create a ContextEnricher with a temp cache."""
        from ag3nt_agent.context_enrichment import ContextEnricher, ContextCache

        cache = ContextCache(db_path=tmp_path / "enricher_cache.db")
        return ContextEnricher(cache=cache)

    def _make_chunks(self, texts, source="MEMORY.md"):
        """Helper to build chunk dicts like memory_search produces."""
        chunks = []
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "source": source,
                "chunk_index": i,
                "content_hash": hashlib.md5(text.encode()).hexdigest(),
            })
        return chunks

    def _source_docs(self):
        """Return a simple source_documents mapping."""
        return {"MEMORY.md": "This is a document about user preferences and settings."}

    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_adds_contextualized_text(self, mock_create, tmp_path):
        """enrich_chunks() adds a contextualized_text field combining prefix + original."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "This chunk discusses user theme preferences."
        mock_model.invoke.return_value = mock_response
        mock_create.return_value = mock_model

        enricher = self._make_enricher(tmp_path)
        chunks = self._make_chunks(["User prefers dark theme."])
        result = enricher.enrich_chunks(chunks, self._source_docs())

        assert len(result) == 1
        assert "contextualized_text" in result[0]
        # The contextualized_text should contain both the prefix and the original text
        assert "This chunk discusses user theme preferences." in result[0]["contextualized_text"]
        assert "User prefers dark theme." in result[0]["contextualized_text"]

    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_preserves_original_text(self, mock_create, tmp_path):
        """enrich_chunks() does not modify the original 'text' field."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "Context prefix here."
        mock_model.invoke.return_value = mock_response
        mock_create.return_value = mock_model

        enricher = self._make_enricher(tmp_path)
        original_text = "User prefers dark theme."
        chunks = self._make_chunks([original_text])
        result = enricher.enrich_chunks(chunks, self._source_docs())

        assert result[0]["text"] == original_text

    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_uses_cache(self, mock_create, tmp_path):
        """Second call with same chunks uses cache, does not call LLM again."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "Cached prefix."
        mock_model.invoke.return_value = mock_response
        mock_create.return_value = mock_model

        enricher = self._make_enricher(tmp_path)
        chunks = self._make_chunks(["Some text to enrich."])

        # First call: LLM invoked
        enricher.enrich_chunks(chunks, self._source_docs())
        assert mock_model.invoke.call_count == 1

        # Second call: should use cache, no additional LLM call
        result = enricher.enrich_chunks(chunks, self._source_docs())
        assert mock_model.invoke.call_count == 1
        assert "contextualized_text" in result[0]

    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_graceful_degradation_on_llm_failure(self, mock_create, tmp_path):
        """If LLM raises, contextualized_text falls back to original text."""
        mock_model = Mock()
        mock_model.invoke.side_effect = RuntimeError("LLM unavailable")
        mock_create.return_value = mock_model

        enricher = self._make_enricher(tmp_path)
        chunks = self._make_chunks(["Fallback text here."])
        result = enricher.enrich_chunks(chunks, self._source_docs())

        assert result[0]["contextualized_text"] == "Fallback text here."

    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_graceful_degradation_no_model(self, mock_create, tmp_path):
        """If _create_model raises, contextualized_text falls back to original text."""
        mock_create.side_effect = Exception("No model configured")

        enricher = self._make_enricher(tmp_path)
        chunks = self._make_chunks(["No model text."])
        result = enricher.enrich_chunks(chunks, self._source_docs())

        assert result[0]["contextualized_text"] == "No model text."

    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_truncates_large_source_docs(self, mock_create, tmp_path):
        """Source documents exceeding 5000 chars are truncated in the prompt."""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.content = "Truncated context."
        mock_model.invoke.return_value = mock_response
        mock_create.return_value = mock_model

        enricher = self._make_enricher(tmp_path)
        chunks = self._make_chunks(["Short chunk."])
        # Create a source doc longer than 5000 chars
        large_doc = "A" * 6000
        source_docs = {"MEMORY.md": large_doc}

        enricher.enrich_chunks(chunks, source_docs)

        # Check that the LLM was called with a prompt containing a truncated doc
        call_args = mock_model.invoke.call_args[0][0]
        # The prompt should NOT contain the full 6000 chars
        assert len(call_args) < len(large_doc)
        # Should contain start (80% of 5000 = 4000) and end (20% of 5000 = 1000) markers
        assert "A" * 100 in call_args  # start portion present
        assert "[...]" in call_args  # truncation marker

    @patch.dict(os.environ, {"AG3NT_CONTEXTUAL_RETRIEVAL": "0"})
    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_disabled_via_env(self, mock_create, tmp_path):
        """AG3NT_CONTEXTUAL_RETRIEVAL=0 skips enrichment entirely."""
        mock_model = Mock()
        mock_create.return_value = mock_model

        enricher = self._make_enricher(tmp_path)
        chunks = self._make_chunks(["Skip enrichment."])
        result = enricher.enrich_chunks(chunks, self._source_docs())

        # LLM should not be called
        mock_model.invoke.assert_not_called()
        # contextualized_text should just be the original text
        assert result[0]["contextualized_text"] == "Skip enrichment."

    @patch("ag3nt_agent.context_enrichment._create_model")
    def test_enrich_multiple_chunks_same_source(self, mock_create, tmp_path):
        """Multiple chunks from the same source are all enriched with individual LLM calls."""
        mock_model = Mock()
        call_count = [0]

        def fake_invoke(prompt):
            call_count[0] += 1
            resp = Mock()
            resp.content = f"Context for chunk {call_count[0]}."
            return resp

        mock_model.invoke.side_effect = fake_invoke
        mock_create.return_value = mock_model

        enricher = self._make_enricher(tmp_path)
        chunks = self._make_chunks([
            "First chunk of content.",
            "Second chunk of content.",
            "Third chunk of content.",
        ])
        result = enricher.enrich_chunks(chunks, self._source_docs())

        # All 3 chunks should be enriched
        assert len(result) == 3
        for chunk in result:
            assert "contextualized_text" in chunk
            assert chunk["contextualized_text"] != chunk["text"]

        # LLM should have been called once per chunk
        assert mock_model.invoke.call_count == 3
