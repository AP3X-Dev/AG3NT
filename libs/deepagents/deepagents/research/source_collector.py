"""Source Collector for gathering and ranking research sources.

The SourceCollector is responsible for:
- Generating search queries from ResearchBrief
- Executing searches across configured providers
- Deduplicating and ranking sources by authority, recency, and relevance
- Creating a source queue with reason codes for each candidate
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from deepagents.research.config import ResearchConfig
from deepagents.research.models import (
    ResearchBrief,
    ResearchMode,
    SourceQueueItem,
    SourceReasonCode,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SearchResult:
    """A single search result from a provider."""
    
    def __init__(
        self,
        url: str,
        title: str,
        snippet: str,
        *,
        source: str = "unknown",
        publish_date: datetime | None = None,
        rank: int = 0,
    ):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.source = source
        self.publish_date = publish_date
        self.rank = rank
    
    @property
    def domain(self) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(self.url)
            return parsed.netloc.lower()
        except Exception:
            return ""


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...
    
    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        """Execute a search query.
        
        Args:
            query: The search query.
            max_results: Maximum number of results.
            recency_days: Only return results from last N days.
            
        Returns:
            List of search results.
        """
        ...


class MockSearchProvider(SearchProvider):
    """Mock search provider for testing."""
    
    @property
    def name(self) -> str:
        return "mock"
    
    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        recency_days: int | None = None,
    ) -> list[SearchResult]:
        """Return mock search results."""
        # Generate deterministic mock results based on query
        results = []
        for i in range(min(5, max_results)):
            results.append(SearchResult(
                url=f"https://example{i}.com/{query.replace(' ', '-')}",
                title=f"Result {i+1} for {query}",
                snippet=f"This is a sample result about {query}...",
                source="mock",
                rank=i,
            ))
        return results


# Authority scores for known domains (higher = more authoritative)
DOMAIN_AUTHORITY_SCORES: dict[str, float] = {
    # Official/documentation sites
    "docs.python.org": 0.95,
    "developer.mozilla.org": 0.95,
    "docs.microsoft.com": 0.90,
    "cloud.google.com": 0.90,
    "aws.amazon.com": 0.90,
    "docs.github.com": 0.90,
    
    # Academic/research
    "arxiv.org": 0.95,
    "scholar.google.com": 0.90,
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    
    # News/tech
    "techcrunch.com": 0.70,
    "theverge.com": 0.65,
    "arstechnica.com": 0.75,
    "wired.com": 0.70,
    
    # Stack Overflow / Q&A
    "stackoverflow.com": 0.80,
    "superuser.com": 0.75,
    
    # Wikipedia
    "en.wikipedia.org": 0.80,
    "wikipedia.org": 0.75,
    
    # GitHub
    "github.com": 0.80,
}


def get_domain_authority(domain: str) -> float:
    """Get authority score for a domain."""
    domain = domain.lower()
    
    # Check exact match
    if domain in DOMAIN_AUTHORITY_SCORES:
        return DOMAIN_AUTHORITY_SCORES[domain]
    
    # Check partial match (e.g., subdomain)
    for known_domain, score in DOMAIN_AUTHORITY_SCORES.items():
        if known_domain in domain or domain.endswith(known_domain):
            return score
    
    # Default score
    return 0.5


class SourceCollector:
    """Collects and ranks sources for research.

    The SourceCollector:
    1. Generates search queries from the research brief
    2. Executes searches across configured providers
    3. Deduplicates results by URL and content hash
    4. Ranks sources by authority, recency, and relevance
    5. Returns a prioritized source queue

    Args:
        config: Research configuration.
        providers: Optional list of search providers.
    """

    def __init__(
        self,
        config: ResearchConfig,
        providers: list[SearchProvider] | None = None,
    ) -> None:
        self.config = config
        self.providers = providers or [MockSearchProvider()]
        self._seen_urls: set[str] = set()
        self._content_hashes: set[str] = set()

    def generate_queries(self, brief: ResearchBrief) -> list[str]:
        """Generate search queries from a research brief.

        Args:
            brief: The research brief.

        Returns:
            List of search queries.
        """
        queries = []

        # Primary query from goal
        queries.append(brief.goal)

        # Add variations based on required outputs
        for output in brief.required_outputs[:3]:
            queries.append(f"{brief.goal} {output}")

        # Add recency modifier if needed
        if brief.recency_days and brief.recency_days <= 30:
            queries.append(f"{brief.goal} latest 2024")

        # Add geography modifier if specified
        if brief.geography:
            queries.append(f"{brief.goal} {brief.geography}")

        return queries[:5]  # Limit to 5 queries

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        url = url.lower().strip()
        # Remove trailing slashes
        url = url.rstrip("/")
        # Remove common tracking parameters
        if "?" in url:
            base, params = url.split("?", 1)
            # Keep only essential params
            url = base
        return url

    def _content_hash(self, title: str, snippet: str) -> str:
        """Generate content hash for deduplication."""
        content = f"{title.lower()}{snippet.lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _is_duplicate(self, result: SearchResult) -> bool:
        """Check if a result is a duplicate."""
        if not self.config.enable_source_deduplication:
            return False

        # Check URL
        norm_url = self._normalize_url(result.url)
        if norm_url in self._seen_urls:
            return True

        # Check content hash
        content_hash = self._content_hash(result.title, result.snippet)
        if content_hash in self._content_hashes:
            return True

        # Mark as seen
        self._seen_urls.add(norm_url)
        self._content_hashes.add(content_hash)
        return False

    def _calculate_rank_score(
        self,
        result: SearchResult,
        brief: ResearchBrief,
        seen_domains: set[str],
    ) -> tuple[float, list[SourceReasonCode]]:
        """Calculate rank score and reason codes for a result.

        Args:
            result: The search result.
            brief: The research brief.
            seen_domains: Set of already-seen domains.

        Returns:
            Tuple of (score, reason_codes).
        """
        score = 0.0
        reasons = []

        # Authority score (0-0.3)
        authority = get_domain_authority(result.domain)
        if authority >= self.config.min_source_authority_score:
            score += authority * 0.3
            if authority >= 0.8:
                reasons.append(SourceReasonCode.AUTHORITY)

        # Recency score (0-0.2)
        if result.publish_date:
            days_old = (_utcnow() - result.publish_date).days
            if days_old <= 7:
                score += 0.2
                reasons.append(SourceReasonCode.RECENCY)
            elif days_old <= 30:
                score += 0.15
            elif days_old <= 90:
                score += 0.1

        # Relevance score based on search rank (0-0.3)
        rank_score = max(0, 0.3 - (result.rank * 0.03))
        score += rank_score
        if result.rank <= 2:
            reasons.append(SourceReasonCode.RELEVANCE)

        # Diversity bonus (0-0.2)
        if result.domain not in seen_domains:
            score += 0.2
            if len(seen_domains) < self.config.source_diversity_min_domains:
                reasons.append(SourceReasonCode.DIVERSITY)

        return min(1.0, score), reasons

    async def collect(
        self,
        brief: ResearchBrief,
    ) -> list[SourceQueueItem]:
        """Collect and rank sources for a research brief.

        Args:
            brief: The research brief.

        Returns:
            Prioritized list of SourceQueueItems.
        """
        all_results: list[SearchResult] = []

        # Generate queries
        queries = self.generate_queries(brief)
        logger.info(f"Generated {len(queries)} queries for research")

        # Execute searches
        for query in queries:
            for provider in self.providers:
                if provider.name not in self.config.allowed_search_providers:
                    continue

                try:
                    results = await provider.search(
                        query,
                        max_results=10,
                        recency_days=brief.recency_days,
                    )
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Search failed for {provider.name}: {e}")

        # Deduplicate and filter
        filtered_results = []
        for result in all_results:
            # Check domain allowlist/denylist
            if not self.config.is_domain_allowed(result.domain):
                continue

            # Check brief domain restrictions
            if brief.disallowed_domains:
                if any(d in result.domain for d in brief.disallowed_domains):
                    continue
            if brief.allowed_domains:
                if not any(d in result.domain for d in brief.allowed_domains):
                    continue

            # Check for duplicates
            if self._is_duplicate(result):
                continue

            filtered_results.append(result)

        # Rank results
        seen_domains: set[str] = set()
        queue_items: list[SourceQueueItem] = []

        for result in filtered_results:
            score, reasons = self._calculate_rank_score(result, brief, seen_domains)
            seen_domains.add(result.domain)

            item = SourceQueueItem(
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                rank_score=score,
                reason_codes=reasons,
                mode=brief.mode_preference,
                domain=result.domain,
                publish_date=result.publish_date,
            )
            queue_items.append(item)

        # Sort by rank score (descending)
        queue_items.sort(key=lambda x: x.rank_score, reverse=True)

        # Limit to max sources
        queue_items = queue_items[:brief.max_sources]

        logger.info(f"Collected {len(queue_items)} sources from {len(seen_domains)} domains")
        return queue_items
