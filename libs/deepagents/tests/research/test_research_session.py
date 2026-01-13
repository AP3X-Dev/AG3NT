"""Tests for ResearchSession."""

import tempfile
from pathlib import Path

import pytest

from deepagents.research import (
    ResearchBrief,
    ResearchConfig,
    ResearchMode,
    ResearchSession,
    SourceQueueItem,
    SourceStatus,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_workspace):
    """Create a test configuration."""
    return ResearchConfig(workspace_base_dir=temp_workspace)


class TestResearchSession:
    """Tests for ResearchSession."""

    def test_create_session(self, temp_workspace, config):
        """Test creating a new session."""
        session = ResearchSession.create(
            workspace_dir=temp_workspace / "test_session",
            config=config,
        )

        assert session.session_id is not None
        assert session.status == "created"
        assert session.current_step == 0

    def test_session_persistence(self, temp_workspace, config):
        """Test that session state persists."""
        # Create session using config's workspace_base_dir
        session = ResearchSession.create(
            config=config,
            session_id="test_persist",
        )

        # Set some state
        brief = ResearchBrief(goal="Test research goal")
        session.set_brief(brief)
        session.set_status("running")
        session.increment_step()

        # Load session using same config
        loaded = ResearchSession.load("test_persist", config)

        assert loaded.session_id == "test_persist"
        assert loaded.status == "running"
        assert loaded.current_step == 1
        assert loaded.get_brief().goal == "Test research goal"

    def test_source_queue_management(self, temp_workspace, config):
        """Test source queue operations."""
        session = ResearchSession.create(
            workspace_dir=temp_workspace / "queue_test",
            config=config,
        )

        # Add sources
        source1 = SourceQueueItem(url="https://example1.com", title="Example 1")
        source2 = SourceQueueItem(url="https://example2.com", title="Example 2")

        session.add_source(source1)
        session.add_source(source2)

        queue = session.get_source_queue()
        assert len(queue) == 2

        # Get pending sources
        pending = session.get_pending_sources()
        assert len(pending) == 2

        # Update source
        session.update_source("https://example1.com", status=SourceStatus.READ)
        pending = session.get_pending_sources()
        assert len(pending) == 1

    def test_evidence_ledger_integration(self, temp_workspace, config):
        """Test evidence ledger integration."""
        session = ResearchSession.create(
            workspace_dir=temp_workspace / "ledger_test",
            config=config,
        )

        # Add evidence
        record = session.evidence_ledger.add_record(
            url="https://example.com/article",
            artifact_id="art_123",
            title="Test Article",
            notes="Key finding from this source",
        )

        assert record.artifact_id == "art_123"
        assert session.evidence_ledger.count() == 1

        # Retrieve by URL
        found = session.evidence_ledger.get_by_url("https://example.com/article")
        assert found is not None
        assert found.title == "Test Article"

    def test_metrics(self, temp_workspace, config):
        """Test session metrics."""
        session = ResearchSession.create(
            workspace_dir=temp_workspace / "metrics_test",
            config=config,
        )

        # Record some activity
        session.increment_step()
        session.increment_step()
        session.record_error("Test error")
        session.record_browser_escalation()

        metrics = session.get_metrics()

        assert metrics["current_step"] == 2
        assert metrics["error_count"] == 1
        assert metrics["browser_escalations"] == 1


class TestResearchBrief:
    """Tests for ResearchBrief."""

    def test_brief_creation(self):
        """Test creating a research brief."""
        brief = ResearchBrief(
            goal="Find AWS Lambda pricing",
            constraints={"recency": "last_30_days"},
            required_outputs=["pricing_table", "free_tier_limits"],
            mode_preference=ResearchMode.BROWSER_ALLOWED,
        )

        assert brief.goal == "Find AWS Lambda pricing"
        assert brief.required_outputs == ["pricing_table", "free_tier_limits"]
        assert brief.mode_preference == ResearchMode.BROWSER_ALLOWED

    def test_brief_defaults(self):
        """Test brief default values."""
        brief = ResearchBrief(goal="Simple research")

        assert brief.max_sources == 12
        assert brief.max_steps == 40
        assert brief.mode_preference == ResearchMode.BROWSER_ALLOWED
