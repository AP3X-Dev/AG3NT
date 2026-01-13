"""Configuration for the Research Agents v2 system.

ResearchConfig provides all tunable parameters for research sessions,
source collection, browser operation, and quality gates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ResearchConfig(BaseModel):
    """Configuration for research sessions.

    This configuration controls:
    - Source collection and ranking parameters
    - Step and token budgets for research sessions
    - Browser mode escalation triggers
    - Quality gate thresholds
    - Search provider settings

    Args:
        max_sources: Maximum number of sources to collect and process.
        max_steps: Maximum number of research steps before stopping.
        source_diversity_min_domains: Minimum number of different domains to consult.
        reader_fail_escalation_count: Number of reader failures before escalating to browser.
        browser_step_budget: Maximum steps for browser-based operations.
        bundle_token_budget: Maximum tokens for the final ResearchBundle.
        allowed_search_providers: List of allowed search provider names.
        domain_allowlist: Optional list of allowed domains (whitelist mode).
        domain_denylist: Optional list of blocked domains.
        citation_required: Whether every non-obvious claim requires evidence.
        default_mode: Default research mode (reader_only, browser_allowed, browser_required).
        recency_threshold_days: Max age in days for sources (0 = no limit).
        min_source_authority_score: Minimum authority score for sources (0.0-1.0).
        enable_source_deduplication: Whether to deduplicate sources by content hash.
        workspace_base_dir: Base directory for research session workspaces.
    """

    # Source collection
    max_sources: int = Field(default=12, ge=1, le=100)
    source_diversity_min_domains: int = Field(default=4, ge=1)
    enable_source_deduplication: bool = Field(default=True)

    # Step and budget limits
    max_steps: int = Field(default=40, ge=1, le=500)
    browser_step_budget: int = Field(default=15, ge=1)
    bundle_token_budget: int = Field(default=1200, ge=100, le=10000)

    # Mode escalation
    reader_fail_escalation_count: int = Field(default=2, ge=1)
    default_mode: Literal["reader_only", "browser_allowed", "browser_required"] = Field(default="browser_allowed")

    # Search providers
    allowed_search_providers: list[str] = Field(default_factory=lambda: ["google", "bing", "duckduckgo"])

    # Domain filtering
    domain_allowlist: list[str] | None = Field(default=None)
    domain_denylist: list[str] = Field(
        default_factory=lambda: [
            "facebook.com",
            "twitter.com",
            "instagram.com",
            "tiktok.com",
        ]
    )

    # Quality gates
    citation_required: bool = Field(default=True)
    min_source_authority_score: float = Field(default=0.3, ge=0.0, le=1.0)
    recency_threshold_days: int = Field(default=0)  # 0 = no limit

    # Workspace
    workspace_base_dir: Path = Field(default_factory=lambda: Path("./research_sessions"))

    # Timeout settings
    page_fetch_timeout_seconds: int = Field(default=30, ge=5)
    browser_action_timeout_seconds: int = Field(default=60, ge=10)

    # Content extraction
    max_content_chars_per_source: int = Field(default=50000)
    extract_publish_dates: bool = Field(default=True)
    extract_author_info: bool = Field(default=True)

    model_config = {"extra": "forbid"}

    def get_session_dir(self, session_id: str) -> Path:
        """Get the workspace directory for a specific session."""
        session_dir = self.workspace_base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def is_domain_allowed(self, domain: str) -> bool:
        """Check if a domain is allowed based on allowlist/denylist."""
        domain = domain.lower()

        # Check denylist first
        for denied in self.domain_denylist:
            if denied.lower() in domain:
                return False

        # If allowlist is set, domain must be in it
        if self.domain_allowlist:
            return any(allowed.lower() in domain for allowed in self.domain_allowlist)

        return True
