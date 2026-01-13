"""Tests for Distiller and Extractor."""

import pytest

from deepagents.compaction.models import Confidence
from deepagents.research import (
    Distiller,
    ExtractionResult,
    Extractor,
    PageContent,
    ResearchConfig,
)


class TestExtractor:
    """Tests for Extractor."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ResearchConfig()

    @pytest.fixture
    def extractor(self, config):
        """Create an extractor."""
        return Extractor(config)

    def test_extract_basic(self, extractor):
        """Test basic extraction."""
        content = PageContent(
            url="https://example.com",
            title="Test Article",
            content="This is a test article about Python programming.",
        )

        result = extractor.extract(content, "art_123")

        assert result.source_url == "https://example.com"
        assert result.artifact_id == "art_123"
        assert result.error is None

    def test_extract_key_facts(self, extractor):
        """Test key fact extraction."""
        content = PageContent(
            url="https://example.com",
            title="Stats Article",
            content="""
            According to research, Python usage increased by 25% in 2024.
            The company announced a new product launch.
            Revenue grew by $1,000,000 last quarter.
            """,
        )

        result = extractor.extract(content, "art_123")

        assert len(result.key_facts) > 0
        # Should find percentage, announcement, and dollar amount

    def test_extract_quotes(self, extractor):
        """Test quote extraction."""
        content = PageContent(
            url="https://example.com",
            title="Interview",
            content="""
            The CEO said "This is a significant milestone for our company and we are excited about the future."
            Another expert noted "The technology is revolutionary and will change everything."
            """,
        )

        result = extractor.extract(content, "art_123")

        assert len(result.quotes) >= 1

    def test_extract_entities(self, extractor):
        """Test entity extraction."""
        content = PageContent(
            url="https://example.com",
            title="News",
            content="Microsoft Corporation announced that Google Cloud and Amazon Web Services are competitors.",
        )

        result = extractor.extract(content, "art_123")

        assert len(result.entities) > 0
        assert any("Microsoft" in e for e in result.entities)

    def test_extract_with_error(self, extractor):
        """Test extraction with error content."""
        content = PageContent(
            url="https://example.com",
            title=None,
            content="",
            error="Connection failed",
        )

        result = extractor.extract(content, "art_123")

        assert result.error == "Connection failed"
        assert len(result.key_facts) == 0

    def test_relevance_calculation(self, extractor):
        """Test relevance calculation."""
        content = PageContent(
            url="https://example.com",
            title="Python Guide",
            content="Python programming language is great for data science and machine learning.",
        )

        result = extractor.extract(content, "art_123", goal="Python programming")

        assert result.topic_relevance > 0.5


class TestDistiller:
    """Tests for Distiller."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ResearchConfig(bundle_token_budget=500)

    @pytest.fixture
    def distiller(self, config):
        """Create a distiller."""
        return Distiller(config)

    def test_distill_basic(self, distiller):
        """Test basic distillation."""
        extractions = [
            ExtractionResult(
                source_url="https://example1.com",
                artifact_id="art_1",
                key_facts=["Python is popular", "Python is easy to learn"],
                quotes=["Python is the best language for beginners"],
            ),
        ]

        findings = distiller.distill(extractions, "Python programming")

        assert len(findings) > 0
        assert all(f.evidence_artifact_ids for f in findings)

    def test_distill_corroboration(self, distiller):
        """Test that corroborated facts get higher confidence."""
        extractions = [
            ExtractionResult(
                source_url="https://example1.com",
                artifact_id="art_1",
                key_facts=["Python usage increased by 25%"],
            ),
            ExtractionResult(
                source_url="https://example2.com",
                artifact_id="art_2",
                key_facts=["Python usage increased by 25%"],
            ),
            ExtractionResult(
                source_url="https://example3.com",
                artifact_id="art_3",
                key_facts=["Python usage increased by 25%"],
            ),
        ]

        findings = distiller.distill(extractions, "Python growth")

        # Should have high confidence due to corroboration
        high_conf = [f for f in findings if f.confidence == Confidence.HIGH]
        assert len(high_conf) > 0

    def test_distill_token_budget(self, distiller):
        """Test token budget enforcement."""
        # Create many extractions
        extractions = [
            ExtractionResult(
                source_url=f"https://example{i}.com",
                artifact_id=f"art_{i}",
                key_facts=[f"Fact {i} with some longer text to consume tokens"] * 5,
                quotes=[f"Quote {i} that is also quite long"] * 3,
            )
            for i in range(10)
        ]

        findings = distiller.distill(extractions, "test", token_budget=200)

        # Should be limited by token budget
        total_chars = sum(len(f.claim) for f in findings)
        assert total_chars < 200 * 4 + 500  # Some overhead allowed
