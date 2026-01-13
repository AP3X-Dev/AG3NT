"""Tests for Testing Harness module."""

import json

import pytest

from deepagents.testing import (
    AgentFixture,
    MetricsCollector,
    RegressionRunner,
    RunMetrics,
)
from deepagents.testing.fixtures import FixtureMessage
from deepagents.testing.runner import RunConfig


class TestFixtureMessage:
    """Tests for FixtureMessage."""

    def test_to_dict(self):
        """Test message serialization."""
        msg = FixtureMessage(
            role="user",
            content="Hello",
        )
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"

    def test_from_dict(self):
        """Test message deserialization."""
        data = {"role": "assistant", "content": "Hi there"}
        msg = FixtureMessage.from_dict(data)
        assert msg.role == "assistant"
        assert msg.content == "Hi there"


class TestAgentFixture:
    """Tests for AgentFixture."""

    def test_create_fixture(self):
        """Test fixture creation."""
        fixture = AgentFixture(
            name="test_fixture",
            description="A test fixture",
            messages=[
                FixtureMessage(role="user", content="Hello"),
            ],
            expected_output="Hi there",
        )
        assert fixture.name == "test_fixture"
        assert len(fixture.messages) == 1

    def test_save_and_load(self, tmp_path):
        """Test fixture persistence."""
        fixture = AgentFixture(
            name="persist_test",
            description="Test persistence",
            messages=[
                FixtureMessage(role="user", content="Test"),
            ],
            expected_output="Response",
            tags=["test", "unit"],
        )

        path = tmp_path / "fixture.json"
        fixture.save(path)

        loaded = AgentFixture.load(path)
        assert loaded.name == "persist_test"
        assert loaded.expected_output == "Response"
        assert "test" in loaded.tags

    def test_get_input_messages(self):
        """Test filtering input messages."""
        fixture = AgentFixture(
            name="filter_test",
            description="Test filtering",
            messages=[
                FixtureMessage(role="user", content="Q1"),
                FixtureMessage(role="assistant", content="A1"),
                FixtureMessage(role="user", content="Q2"),
            ],
        )
        inputs = fixture.get_input_messages()
        assert len(inputs) == 2
        assert all(m.role == "user" for m in inputs)


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_metrics(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        collector.start_run()

        collector.record(
            RunMetrics(
                fixture_name="test1",
                passed=True,
                duration_ms=100,
                token_count=500,
                tool_calls=2,
                steps=3,
            )
        )

        agg = collector.get_aggregate()
        assert agg.total_tests == 1
        assert agg.passed_tests == 1

    def test_aggregate_metrics(self):
        """Test metric aggregation."""
        collector = MetricsCollector()
        collector.start_run()

        collector.record(RunMetrics("t1", True, 100, 500, 2, 3))
        collector.record(RunMetrics("t2", False, 200, 600, 3, 4, error="Failed"))
        collector.record(RunMetrics("t3", True, 150, 400, 1, 2))

        agg = collector.get_aggregate()
        assert agg.total_tests == 3
        assert agg.passed_tests == 2
        assert agg.failed_tests == 1
        assert agg.pass_rate == pytest.approx(0.667, rel=0.01)

    def test_save_metrics(self, tmp_path):
        """Test saving metrics to file."""
        collector = MetricsCollector(output_dir=tmp_path)
        collector.start_run()
        collector.record(RunMetrics("t1", True, 100, 500, 2, 3))

        path = collector.save()
        assert path is not None
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert "aggregate" in data
        assert "tests" in data


class TestRegressionRunner:
    """Tests for RegressionRunner."""

    def test_load_fixtures(self, tmp_path):
        """Test loading fixtures from directory."""
        # Create test fixtures
        fixture1 = AgentFixture("test1", "Test 1", [])
        fixture2 = AgentFixture("test2", "Test 2", [], tags=["smoke"])

        fixture1.save(tmp_path / "test1.json")
        fixture2.save(tmp_path / "test2.json")

        config = RunConfig(fixtures_dir=tmp_path)
        runner = RegressionRunner(config)

        fixtures = runner.load_fixtures()
        assert len(fixtures) == 2

    def test_run_all_pass(self, tmp_path):
        """Test running all fixtures with passing results."""
        fixture = AgentFixture("test1", "Test 1", [], expected_output="OK")
        fixture.save(tmp_path / "test1.json")

        config = RunConfig(fixtures_dir=tmp_path)
        runner = RegressionRunner(config)

        result = runner.run_all()
        assert result.total == 1
        assert result.passed == 1
        assert result.success

    def test_tag_filtering(self, tmp_path):
        """Test filtering fixtures by tag."""
        AgentFixture("t1", "Test 1", [], tags=["smoke"]).save(tmp_path / "t1.json")
        AgentFixture("t2", "Test 2", [], tags=["full"]).save(tmp_path / "t2.json")

        config = RunConfig(fixtures_dir=tmp_path, tags_filter=["smoke"])
        runner = RegressionRunner(config)

        fixtures = runner.load_fixtures()
        assert len(fixtures) == 1
        assert fixtures[0].name == "t1"
