"""Testing Harness - Fixture-based offline testing and benchmarking.

This module provides:
1. TestFixture - Recorded agent interactions for replay
2. MetricsCollector - Collect and aggregate test metrics
3. RegressionRunner - Run fixtures and compare results

Usage:
    from deepagents.testing import TestFixture, RegressionRunner, MetricsCollector

    # Record a fixture
    fixture = TestFixture.record(agent, "test_task", messages)
    fixture.save("fixtures/test_task.json")

    # Run regression tests
    runner = RegressionRunner(fixtures_dir="fixtures/")
    results = runner.run_all()
"""

from deepagents.testing.fixtures import AgentFixture, FixtureResult
from deepagents.testing.metrics import MetricsCollector, RunMetrics
from deepagents.testing.runner import RegressionRunner

__all__ = [
    "AgentFixture",
    "FixtureResult",
    "MetricsCollector",
    "RunMetrics",
    "RegressionRunner",
]

