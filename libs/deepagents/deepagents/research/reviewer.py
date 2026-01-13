"""Reviewer for validating research quality and triggering follow-ups.

The Reviewer is responsible for:
- Validating evidence coverage for required outputs
- Checking source diversity requirements
- Identifying gaps that need follow-up research
- Generating follow-up tasks when quality gates aren't met
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

from deepagents.compaction.models import Confidence, ResearchBundle
from deepagents.research.config import ResearchConfig
from deepagents.research.models import ResearchBrief

if TYPE_CHECKING:
    from deepagents.research.evidence_ledger import EvidenceLedger

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ReviewStatus(str, Enum):
    """Status of a review."""

    PASSED = "passed"
    NEEDS_FOLLOWUP = "needs_followup"
    FAILED = "failed"


class GapType(str, Enum):
    """Types of gaps identified in research."""

    MISSING_OUTPUT = "missing_output"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_SOURCES = "insufficient_sources"
    DOMAIN_DIVERSITY = "domain_diversity"
    RECENCY = "recency"
    UNCITED_CLAIM = "uncited_claim"


@dataclass
class Gap:
    """A gap identified in the research."""

    gap_type: GapType
    description: str
    severity: str = "medium"  # low, medium, high
    suggested_query: str | None = None


@dataclass
class FollowUpTask:
    """A follow-up task to address a gap."""

    goal: str
    reason: str
    priority: int = 1  # 1 = highest
    suggested_queries: list[str] = field(default_factory=list)


@dataclass
class ReviewResult:
    """Result of reviewing research quality."""

    status: ReviewStatus

    # Quality metrics
    evidence_coverage: float = 0.0
    source_diversity: float = 0.0
    confidence_distribution: dict[str, int] = field(default_factory=dict)

    # Gaps and follow-ups
    gaps: list[Gap] = field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = field(default_factory=list)

    # Summary
    summary: str = ""


class Reviewer:
    """Reviews research quality and generates follow-up tasks.

    The Reviewer validates that research meets quality gates:
    - All required outputs are covered with evidence
    - Source diversity requirements are met
    - Confidence levels are acceptable
    - Claims are properly cited

    Args:
        config: Research configuration.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config

    def review(
        self,
        bundle: ResearchBundle,
        brief: ResearchBrief,
        ledger: EvidenceLedger,
    ) -> ReviewResult:
        """Review a research bundle for quality.

        Args:
            bundle: The research bundle to review.
            brief: The original research brief.
            ledger: The evidence ledger.

        Returns:
            ReviewResult with gaps and follow-up tasks.
        """
        gaps: list[Gap] = []

        # Check required outputs
        output_gaps = self._check_required_outputs(bundle, brief)
        gaps.extend(output_gaps)

        # Check source diversity
        diversity_gaps = self._check_source_diversity(ledger, brief)
        gaps.extend(diversity_gaps)

        # Check confidence levels
        confidence_gaps = self._check_confidence_levels(bundle)
        gaps.extend(confidence_gaps)

        # Check citations
        if self.config.citation_required:
            citation_gaps = self._check_citations(bundle)
            gaps.extend(citation_gaps)

        # Calculate metrics
        evidence_coverage = self._calculate_evidence_coverage(bundle, brief)
        source_diversity = self._calculate_source_diversity(ledger)
        confidence_dist = self._get_confidence_distribution(bundle)

        # Generate follow-up tasks
        follow_ups = self._generate_follow_ups(gaps, brief)

        # Determine status
        high_severity_gaps = [g for g in gaps if g.severity == "high"]
        if high_severity_gaps:
            status = ReviewStatus.NEEDS_FOLLOWUP
        elif gaps:
            status = ReviewStatus.PASSED  # Minor gaps are OK
        else:
            status = ReviewStatus.PASSED

        # Generate summary
        summary = self._generate_summary(bundle, gaps, evidence_coverage)

        return ReviewResult(
            status=status,
            evidence_coverage=evidence_coverage,
            source_diversity=source_diversity,
            confidence_distribution=confidence_dist,
            gaps=gaps,
            follow_up_tasks=follow_ups,
            summary=summary,
        )

    def _check_required_outputs(
        self,
        bundle: ResearchBundle,
        brief: ResearchBrief,
    ) -> list[Gap]:
        """Check if required outputs are covered."""
        gaps = []

        for required in brief.required_outputs:
            # Check if any finding mentions this output
            found = False
            for finding in bundle.findings:
                if required.lower() in finding.claim.lower():
                    found = True
                    break

            if not found:
                gaps.append(
                    Gap(
                        gap_type=GapType.MISSING_OUTPUT,
                        description=f"Required output not found: {required}",
                        severity="high",
                        suggested_query=f"{brief.goal} {required}",
                    )
                )

        return gaps

    def _check_source_diversity(
        self,
        ledger: EvidenceLedger,
        brief: ResearchBrief,
    ) -> list[Gap]:
        """Check source diversity requirements."""
        gaps = []

        unique_domains = ledger.get_unique_domains()
        min_domains = self.config.source_diversity_min_domains

        if len(unique_domains) < min_domains:
            gaps.append(
                Gap(
                    gap_type=GapType.DOMAIN_DIVERSITY,
                    description=f"Only {len(unique_domains)} domains consulted, need {min_domains}",
                    severity="medium",
                    suggested_query=f"{brief.goal} site:different-domain.com",
                )
            )

        return gaps

    def _check_confidence_levels(self, bundle: ResearchBundle) -> list[Gap]:
        """Check confidence level distribution."""
        gaps = []

        if not bundle.findings:
            gaps.append(
                Gap(
                    gap_type=GapType.LOW_CONFIDENCE,
                    description="No findings extracted from sources",
                    severity="high",
                )
            )
            return gaps

        # Count by confidence
        high = sum(1 for f in bundle.findings if f.confidence == Confidence.HIGH)
        total = len(bundle.findings)

        if high / total < 0.2:
            gaps.append(
                Gap(
                    gap_type=GapType.LOW_CONFIDENCE,
                    description=f"Only {high}/{total} findings have high confidence",
                    severity="medium",
                )
            )

        return gaps

    def _check_citations(self, bundle: ResearchBundle) -> list[Gap]:
        """Check that claims have citations."""
        gaps = []

        for finding in bundle.findings:
            if not finding.evidence_artifact_ids:
                gaps.append(
                    Gap(
                        gap_type=GapType.UNCITED_CLAIM,
                        description=f"Uncited claim: {finding.claim[:50]}...",
                        severity="medium",
                    )
                )

        return gaps

    def _calculate_evidence_coverage(
        self,
        bundle: ResearchBundle,
        brief: ResearchBrief,
    ) -> float:
        """Calculate evidence coverage score."""
        if not brief.required_outputs:
            return 1.0 if bundle.findings else 0.0

        covered = 0
        for required in brief.required_outputs:
            for finding in bundle.findings:
                if required.lower() in finding.claim.lower():
                    covered += 1
                    break

        return covered / len(brief.required_outputs)

    def _calculate_source_diversity(self, ledger: EvidenceLedger) -> float:
        """Calculate source diversity score."""
        unique_domains = ledger.get_unique_domains()
        min_domains = self.config.source_diversity_min_domains

        return min(1.0, len(unique_domains) / min_domains)

    def _get_confidence_distribution(
        self,
        bundle: ResearchBundle,
    ) -> dict[str, int]:
        """Get distribution of confidence levels."""
        dist = {"high": 0, "medium": 0, "low": 0}

        for finding in bundle.findings:
            dist[finding.confidence.value] = dist.get(finding.confidence.value, 0) + 1

        return dist

    def _generate_follow_ups(
        self,
        gaps: list[Gap],
        brief: ResearchBrief,
    ) -> list[FollowUpTask]:
        """Generate follow-up tasks from gaps."""
        follow_ups = []

        # Group gaps by type
        high_severity = [g for g in gaps if g.severity == "high"]

        for gap in high_severity:
            queries = []
            if gap.suggested_query:
                queries.append(gap.suggested_query)

            follow_ups.append(
                FollowUpTask(
                    goal=f"Address gap: {gap.description}",
                    reason=f"Gap type: {gap.gap_type.value}",
                    priority=1 if gap.severity == "high" else 2,
                    suggested_queries=queries,
                )
            )

        return follow_ups

    def _generate_summary(
        self,
        bundle: ResearchBundle,
        gaps: list[Gap],
        coverage: float,
    ) -> str:
        """Generate review summary."""
        parts = []

        parts.append(f"Reviewed {len(bundle.findings)} findings from {len(bundle.evidence)} sources.")
        parts.append(f"Evidence coverage: {coverage:.0%}.")

        if gaps:
            high = sum(1 for g in gaps if g.severity == "high")
            parts.append(f"Found {len(gaps)} gaps ({high} high severity).")
        else:
            parts.append("All quality gates passed.")

        return " ".join(parts)
