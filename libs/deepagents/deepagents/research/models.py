"""Data models for the Research Agents v2 system.

These models define the core data structures used throughout the research
system, including research briefs, source queue items, and session state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Get current UTC time in a timezone-aware manner."""
    return datetime.now(UTC)


class ResearchMode(str, Enum):
    """Research mode for source processing."""

    READER_ONLY = "reader_only"
    BROWSER_ALLOWED = "browser_allowed"
    BROWSER_REQUIRED = "browser_required"


class SourceStatus(str, Enum):
    """Status of a source in the queue."""

    QUEUED = "queued"
    READING = "reading"
    READ = "read"
    BROWSER_NEEDED = "browser_needed"
    BROWSING = "browsing"
    BROWSED = "browsed"
    REJECTED = "rejected"
    ERRORED = "errored"


class SourceReasonCode(str, Enum):
    """Reason codes for source ranking."""

    AUTHORITY = "authority"  # High-authority source
    RECENCY = "recency"  # Recently published
    RELEVANCE = "relevance"  # Highly relevant to query
    DIVERSITY = "diversity"  # Different domain for diversity
    PRIMARY = "primary"  # Primary/official source
    CITED = "cited"  # Cited by other sources


class ResearchBrief(BaseModel):
    """Brief describing a research task.

    A ResearchBrief fully specifies what the research should accomplish,
    including constraints, required outputs, and budgets.
    """

    goal: str = Field(..., description="The research goal or question to answer")

    # Constraints
    constraints: dict[str, Any] = Field(default_factory=dict, description="Constraints like recency, geography, allowed_domains, etc.")
    recency_days: int | None = Field(None, description="Maximum age of sources in days (None = no limit)")
    geography: str | None = Field(None, description="Geographic focus (e.g., 'US', 'EU', 'Global')")
    allowed_domains: list[str] | None = Field(None, description="Only allow these domains")
    disallowed_domains: list[str] = Field(default_factory=list, description="Block these domains")

    # Required outputs
    required_outputs: list[str] = Field(default_factory=list, description="Specific items that must be in the final bundle")

    # Preferred sources
    preferred_sources: list[str] = Field(default_factory=list, description="Preferred source categories (e.g., 'official_docs', 'academic')")

    # Mode preference
    mode_preference: ResearchMode = Field(default=ResearchMode.BROWSER_ALLOWED, description="Whether browser mode is allowed/required")

    # Budgets
    max_sources: int = Field(default=12, ge=1)
    max_steps: int = Field(default=40, ge=1)
    bundle_token_budget: int = Field(default=1200, ge=100)

    # Context for the research
    context: str | None = Field(None, description="Additional context to help guide the research")

    created_at: datetime = Field(default_factory=_utcnow)


class SourceQueueItem(BaseModel):
    """An item in the research source queue."""

    url: str = Field(..., description="URL of the source")
    title: str | None = Field(None, description="Title from search results")
    snippet: str | None = Field(None, description="Snippet from search results")

    # Ranking
    rank_score: float = Field(default=0.5, ge=0.0, le=1.0)
    reason_codes: list[SourceReasonCode] = Field(default_factory=list)

    # Mode and status
    mode: ResearchMode = Field(default=ResearchMode.BROWSER_ALLOWED)
    status: SourceStatus = Field(default=SourceStatus.QUEUED)

    # Processing metadata
    artifact_id: str | None = Field(None, description="Artifact ID if processed")
    error_message: str | None = Field(None, description="Error message if failed")
    retry_count: int = Field(default=0)

    # Timestamps
    queued_at: datetime = Field(default_factory=_utcnow)
    processed_at: datetime | None = Field(None)

    # Source metadata
    domain: str | None = Field(None, description="Extracted domain")
    publish_date: datetime | None = Field(None)

    def mark_read(self, artifact_id: str) -> None:
        """Mark source as successfully read."""
        self.status = SourceStatus.READ
        self.artifact_id = artifact_id
        self.processed_at = _utcnow()

    def mark_error(self, error: str) -> None:
        """Mark source as errored."""
        self.status = SourceStatus.ERRORED
        self.error_message = error
        self.retry_count += 1
        self.processed_at = _utcnow()

    def needs_browser(self) -> None:
        """Mark source as needing browser mode."""
        self.status = SourceStatus.BROWSER_NEEDED
