"""Tests for SourceCollector."""

import pytest

from deepagents.research import (
    MockSearchProvider,
    ResearchBrief,
    ResearchConfig,
    ResearchMode,
    SourceCollector,
    SourceReasonCode,
)
from deepagents.research.source_collector import (
    SearchResult,
    get_domain_authority,
)


class TestDomainAuthority:
    """Tests for domain authority scoring."""
    
    def test_known_domains(self):
        """Test authority scores for known domains."""
        assert get_domain_authority("docs.python.org") == 0.95
        assert get_domain_authority("stackoverflow.com") == 0.80
        assert get_domain_authority("github.com") == 0.80
    
    def test_subdomain_matching(self):
        """Test that subdomains match parent domains."""
        assert get_domain_authority("api.github.com") == 0.80
        assert get_domain_authority("en.wikipedia.org") == 0.80
    
    def test_unknown_domains(self):
        """Test default score for unknown domains."""
        assert get_domain_authority("random-blog.com") == 0.5
        assert get_domain_authority("unknown-site.net") == 0.5


class TestSourceCollector:
    """Tests for SourceCollector."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ResearchConfig(
            max_sources=10,
            source_diversity_min_domains=3,
            allowed_search_providers=["mock"],
        )
    
    @pytest.fixture
    def collector(self, config):
        """Create a source collector with mock provider."""
        return SourceCollector(config, providers=[MockSearchProvider()])
    
    def test_generate_queries(self, collector):
        """Test query generation from brief."""
        brief = ResearchBrief(
            goal="AWS Lambda pricing",
            required_outputs=["free_tier", "compute_costs"],
        )
        
        queries = collector.generate_queries(brief)
        
        assert len(queries) >= 1
        assert "AWS Lambda pricing" in queries[0]
    
    def test_generate_queries_with_recency(self, collector):
        """Test query generation with recency constraint."""
        brief = ResearchBrief(
            goal="Latest AI news",
            recency_days=7,
        )
        
        queries = collector.generate_queries(brief)
        
        # Should include a recency-modified query
        assert any("latest" in q.lower() or "2024" in q for q in queries)
    
    @pytest.mark.asyncio
    async def test_collect_sources(self, collector):
        """Test collecting sources."""
        brief = ResearchBrief(
            goal="Python programming",
            max_sources=5,
        )
        
        sources = await collector.collect(brief)
        
        assert len(sources) <= 5
        assert all(s.url for s in sources)
        assert all(s.rank_score >= 0 for s in sources)
    
    @pytest.mark.asyncio
    async def test_deduplication(self, config):
        """Test URL deduplication."""
        # Create collector with duplicate results
        collector = SourceCollector(config, providers=[MockSearchProvider()])
        
        brief = ResearchBrief(goal="test query", max_sources=10)
        
        sources = await collector.collect(brief)
        
        # Check no duplicate URLs
        urls = [s.url for s in sources]
        assert len(urls) == len(set(urls))
    
    @pytest.mark.asyncio
    async def test_domain_filtering(self, config):
        """Test domain allowlist/denylist."""
        config.domain_denylist = ["example0.com"]
        collector = SourceCollector(config, providers=[MockSearchProvider()])
        
        brief = ResearchBrief(goal="test", max_sources=10)
        sources = await collector.collect(brief)
        
        # Should not include denied domain
        for source in sources:
            assert "example0.com" not in source.url
    
    def test_rank_score_calculation(self, collector):
        """Test rank score calculation."""
        brief = ResearchBrief(goal="test")
        
        result = SearchResult(
            url="https://docs.python.org/test",
            title="Test",
            snippet="Test snippet",
            rank=0,
        )
        
        score, reasons = collector._calculate_rank_score(result, brief, set())
        
        assert score > 0
        assert SourceReasonCode.AUTHORITY in reasons  # High authority domain
        assert SourceReasonCode.DIVERSITY in reasons  # First domain seen


class TestSearchResult:
    """Tests for SearchResult."""
    
    def test_domain_extraction(self):
        """Test domain extraction from URL."""
        result = SearchResult(
            url="https://docs.python.org/3/library/index.html",
            title="Test",
            snippet="Test",
        )
        
        assert result.domain == "docs.python.org"
    
    def test_domain_extraction_with_port(self):
        """Test domain extraction with port."""
        result = SearchResult(
            url="http://localhost:8080/test",
            title="Test",
            snippet="Test",
        )
        
        assert result.domain == "localhost:8080"

