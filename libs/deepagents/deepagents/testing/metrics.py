"""Metrics Collection - Aggregate test metrics and statistics."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """Metrics for a single test run."""
    
    fixture_name: str
    passed: bool
    duration_ms: float
    token_count: int
    tool_calls: int
    steps: int
    timestamp: datetime = field(default_factory=datetime.now)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_name": self.fixture_name,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "token_count": self.token_count,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
        }


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple test runs."""
    
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_tool_calls: int = 0
    total_steps: int = 0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average duration."""
        return self.total_duration_ms / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def avg_tokens(self) -> float:
        """Calculate average tokens per test."""
        return self.total_tokens / self.total_tests if self.total_tests > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "pass_rate": f"{self.pass_rate:.1%}",
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.avg_tokens,
            "total_tool_calls": self.total_tool_calls,
            "total_steps": self.total_steps,
        }


class MetricsCollector:
    """Collects and aggregates test metrics.
    
    Provides:
    - Per-test metrics recording
    - Aggregate statistics
    - Persistence to JSON
    - Comparison between runs
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize collector.
        
        Args:
            output_dir: Directory for metrics output files.
        """
        self.output_dir = output_dir
        self._metrics: list[RunMetrics] = []
        self._run_start: float | None = None

    def start_run(self) -> None:
        """Mark the start of a test run."""
        self._run_start = time.time()
        self._metrics = []

    def record(self, metrics: RunMetrics) -> None:
        """Record metrics for a single test."""
        self._metrics.append(metrics)
        logger.debug(f"Recorded metrics for {metrics.fixture_name}")

    def get_aggregate(self) -> AggregateMetrics:
        """Get aggregated metrics."""
        agg = AggregateMetrics()
        for m in self._metrics:
            agg.total_tests += 1
            if m.passed:
                agg.passed_tests += 1
            else:
                agg.failed_tests += 1
            agg.total_duration_ms += m.duration_ms
            agg.total_tokens += m.token_count
            agg.total_tool_calls += m.tool_calls
            agg.total_steps += m.steps
        return agg

    def save(self, filename: str = "metrics.json") -> Path | None:
        """Save metrics to file."""
        if not self.output_dir:
            return None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        
        data = {
            "run_timestamp": datetime.now().isoformat(),
            "aggregate": self.get_aggregate().to_dict(),
            "tests": [m.to_dict() for m in self._metrics],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved metrics to {path}")
        return path

    def get_failed_tests(self) -> list[RunMetrics]:
        """Get list of failed tests."""
        return [m for m in self._metrics if not m.passed]

