"""Tests for Reviewer."""

import tempfile
from pathlib import Path

import pytest

from deepagents.compaction.models import Confidence, EvidenceRecord, Finding, ResearchBundle
from deepagents.research import (
    EvidenceLedger,
    GapType,
    ResearchBrief,
    ResearchConfig,
    Reviewer,
    ReviewStatus,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config():
    """Create test configuration."""
    return ResearchConfig(
        source_diversity_min_domains=3,
        citation_required=True,
    )


@pytest.fixture
def reviewer(config):
    """Create a reviewer."""
    return Reviewer(config)


@pytest.fixture
def ledger(temp_workspace, config):
    """Create an evidence ledger with some records."""
    ledger = EvidenceLedger(temp_workspace, config)

    # Add some evidence from different domains
    ledger.add_record(
        url="https://example1.com/article",
        artifact_id="art_1",
        title="Article 1",
    )
    ledger.add_record(
        url="https://example2.com/article",
        artifact_id="art_2",
        title="Article 2",
    )
    ledger.add_record(
        url="https://example3.com/article",
        artifact_id="art_3",
        title="Article 3",
    )

    return ledger


class TestReviewer:
    """Tests for Reviewer."""

    def test_review_passing(self, reviewer, ledger):
        """Test review that passes all checks."""
        bundle = ResearchBundle(
            executive_summary="Research complete",
            findings=[
                Finding(
                    claim="Python is popular",
                    confidence=Confidence.HIGH,
                    evidence_artifact_ids=["art_1"],
                ),
                Finding(
                    claim="Python is easy to learn",
                    confidence=Confidence.HIGH,
                    evidence_artifact_ids=["art_2"],
                ),
            ],
            evidence=[
                EvidenceRecord(url="https://example1.com", artifact_id="art_1"),
                EvidenceRecord(url="https://example2.com", artifact_id="art_2"),
            ],
        )

        brief = ResearchBrief(goal="Python programming")

        result = reviewer.review(bundle, brief, ledger)

        assert result.status == ReviewStatus.PASSED
        assert result.source_diversity >= 1.0

    def test_review_missing_output(self, reviewer, ledger):
        """Test review with missing required output."""
        bundle = ResearchBundle(
            executive_summary="Research complete",
            findings=[
                Finding(
                    claim="Python is popular",
                    confidence=Confidence.HIGH,
                    evidence_artifact_ids=["art_1"],
                ),
            ],
            evidence=[
                EvidenceRecord(url="https://example1.com", artifact_id="art_1"),
            ],
        )

        brief = ResearchBrief(
            goal="Python programming",
            required_outputs=["pricing_table", "installation_guide"],
        )

        result = reviewer.review(bundle, brief, ledger)

        # Should have gaps for missing outputs
        missing_gaps = [g for g in result.gaps if g.gap_type == GapType.MISSING_OUTPUT]
        assert len(missing_gaps) >= 1

    def test_review_low_diversity(self, reviewer, temp_workspace, config):
        """Test review with low source diversity."""
        # Create ledger with only one domain
        ledger = EvidenceLedger(temp_workspace, config)
        ledger.add_record(
            url="https://example.com/page1",
            artifact_id="art_1",
        )
        ledger.add_record(
            url="https://example.com/page2",
            artifact_id="art_2",
        )

        bundle = ResearchBundle(
            executive_summary="Research complete",
            findings=[],
            evidence=[],
        )

        brief = ResearchBrief(goal="Test")

        result = reviewer.review(bundle, brief, ledger)

        # Should have diversity gap
        diversity_gaps = [g for g in result.gaps if g.gap_type == GapType.DOMAIN_DIVERSITY]
        assert len(diversity_gaps) >= 1

    def test_review_uncited_claims(self, reviewer, ledger):
        """Test review with uncited claims."""
        bundle = ResearchBundle(
            executive_summary="Research complete",
            findings=[
                Finding(
                    claim="Uncited claim here",
                    confidence=Confidence.MEDIUM,
                    evidence_artifact_ids=[],  # No citations!
                ),
            ],
            evidence=[],
        )

        brief = ResearchBrief(goal="Test")

        result = reviewer.review(bundle, brief, ledger)

        # Should have uncited claim gap
        uncited_gaps = [g for g in result.gaps if g.gap_type == GapType.UNCITED_CLAIM]
        assert len(uncited_gaps) >= 1

    def test_follow_up_generation(self, reviewer, ledger):
        """Test follow-up task generation."""
        bundle = ResearchBundle(
            executive_summary="Research complete",
            findings=[],
            evidence=[],
        )

        brief = ResearchBrief(
            goal="AWS Lambda pricing",
            required_outputs=["pricing_table"],
        )

        result = reviewer.review(bundle, brief, ledger)

        # Should generate follow-up for missing output
        assert len(result.follow_up_tasks) >= 1
        assert any("pricing_table" in t.goal.lower() for t in result.follow_up_tasks)

    def test_confidence_distribution(self, reviewer, ledger):
        """Test confidence distribution calculation."""
        bundle = ResearchBundle(
            executive_summary="Research complete",
            findings=[
                Finding(claim="High 1", confidence=Confidence.HIGH, evidence_artifact_ids=["art_1"]),
                Finding(claim="High 2", confidence=Confidence.HIGH, evidence_artifact_ids=["art_2"]),
                Finding(claim="Medium 1", confidence=Confidence.MEDIUM, evidence_artifact_ids=["art_3"]),
                Finding(claim="Low 1", confidence=Confidence.LOW, evidence_artifact_ids=["art_1"]),
            ],
            evidence=[],
        )

        brief = ResearchBrief(goal="Test")

        result = reviewer.review(bundle, brief, ledger)

        assert result.confidence_distribution["high"] == 2
        assert result.confidence_distribution["medium"] == 1
        assert result.confidence_distribution["low"] == 1
