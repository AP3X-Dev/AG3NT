"""Regression Runner - Execute fixtures and compare results."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepagents.testing.fixtures import AgentFixture, FixtureResult
from deepagents.testing.metrics import MetricsCollector, RunMetrics

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for regression runs."""

    fixtures_dir: Path
    output_dir: Path | None = None
    tags_filter: list[str] = field(default_factory=list)
    fail_fast: bool = False
    verbose: bool = False


@dataclass
class RunResult:
    """Result of a regression run."""

    total: int
    passed: int
    failed: int
    skipped: int
    results: list[FixtureResult]
    duration_ms: float

    @property
    def success(self) -> bool:
        """Check if all tests passed."""
        return self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "results": [r.to_dict() for r in self.results],
        }


class RegressionRunner:
    """Runs fixture-based regression tests.

    Loads fixtures from a directory, executes them against an agent,
    and compares results to expected outputs.
    """

    def __init__(
        self,
        config: RunConfig,
        executor: Callable[[AgentFixture], FixtureResult] | None = None,
    ) -> None:
        """Initialize runner.

        Args:
            config: Run configuration.
            executor: Function to execute a fixture. If None, uses mock executor.
        """
        self.config = config
        self.executor = executor or self._mock_executor
        self.metrics = MetricsCollector(config.output_dir)

    def _mock_executor(self, fixture: AgentFixture) -> FixtureResult:
        """Mock executor for testing the runner itself."""
        return FixtureResult(
            fixture_name=fixture.name,
            passed=True,
            expected_output=fixture.expected_output,
            actual_output=fixture.expected_output,
            duration_ms=0.0,
        )

    def load_fixtures(self) -> list[AgentFixture]:
        """Load all fixtures from the fixtures directory."""
        fixtures = []
        if not self.config.fixtures_dir.exists():
            logger.warning(f"Fixtures directory not found: {self.config.fixtures_dir}")
            return fixtures

        for path in self.config.fixtures_dir.glob("*.json"):
            try:
                fixture = AgentFixture.load(path)

                # Apply tag filter
                if self.config.tags_filter:
                    if not any(t in fixture.tags for t in self.config.tags_filter):
                        continue

                fixtures.append(fixture)
            except Exception as e:
                logger.warning(f"Failed to load fixture {path}: {e}")

        logger.info(f"Loaded {len(fixtures)} fixtures")
        return fixtures

    def run_all(self) -> RunResult:
        """Run all fixtures and return results."""
        start_time = time.time()
        self.metrics.start_run()

        fixtures = self.load_fixtures()
        results: list[FixtureResult] = []
        passed = 0
        failed = 0
        skipped = 0

        for fixture in fixtures:
            try:
                result = self.executor(fixture)
                results.append(result)

                # Record metrics
                self.metrics.record(
                    RunMetrics(
                        fixture_name=fixture.name,
                        passed=result.passed,
                        duration_ms=result.duration_ms,
                        token_count=result.token_count,
                        tool_calls=len(fixture.expected_tool_calls),
                        steps=len(fixture.messages),
                        error=result.error,
                    )
                )

                if result.passed:
                    passed += 1
                    if self.config.verbose:
                        logger.info(f"✓ {fixture.name}")
                else:
                    failed += 1
                    logger.warning(f"✗ {fixture.name}: {result.error or 'Output mismatch'}")

                    if self.config.fail_fast:
                        break

            except Exception as e:
                failed += 1
                results.append(
                    FixtureResult(
                        fixture_name=fixture.name,
                        passed=False,
                        expected_output=fixture.expected_output,
                        actual_output=None,
                        error=str(e),
                    )
                )
                logger.error(f"✗ {fixture.name}: {e}")

        duration_ms = (time.time() - start_time) * 1000

        # Save metrics
        if self.config.output_dir:
            self.metrics.save()

        return RunResult(
            total=len(fixtures),
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            duration_ms=duration_ms,
        )
