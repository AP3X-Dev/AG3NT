"""Circuit breaker for preventing cascading failures.

Uses per-category state tracking with three states:
  CLOSED  (normal)   -> calls allowed
  OPEN    (failing)  -> calls blocked until reset timeout elapses
  HALF_OPEN (testing) -> one test call allowed to probe recovery
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


class CircuitState(enum.Enum):
    """State of a circuit breaker category."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CategoryState:
    """Per-category tracking for the circuit breaker."""

    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    state: CircuitState = CircuitState.CLOSED
    total_successes: int = 0
    total_failures: int = 0


class CircuitBreaker:
    """Thread-safe circuit breaker with per-category state tracking.

    Parameters
    ----------
    failure_threshold:
        Number of consecutive failures before opening the circuit.
    reset_timeout_s:
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
    success_threshold:
        Number of consecutive successes in HALF_OPEN required to close.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout_s: float = 300.0,
        success_threshold: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout_s = reset_timeout_s
        self._success_threshold = success_threshold
        self._categories: Dict[str, CategoryState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_category(self, category: str) -> CategoryState:
        """Return the CategoryState for *category*, creating if needed.

        Must be called while holding ``self._lock``.
        """
        if category not in self._categories:
            self._categories[category] = CategoryState()
        return self._categories[category]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self, category: str = "default") -> bool:
        """Return whether a call is allowed for *category*.

        * CLOSED   -> always allowed.
        * OPEN     -> blocked unless the reset timeout has elapsed, in which
                      case the state transitions to HALF_OPEN and the call is
                      allowed (one test call).
        * HALF_OPEN -> allowed (probe call).
        """
        with self._lock:
            cat = self._get_category(category)

            if cat.state is CircuitState.CLOSED:
                return True

            if cat.state is CircuitState.OPEN:
                elapsed = time.monotonic() - cat.last_failure_time
                if elapsed >= self._reset_timeout_s:
                    cat.state = CircuitState.HALF_OPEN
                    return True
                return False

            # HALF_OPEN — allow one probe call
            return True

    def record_success(self, category: str = "default") -> None:
        """Record a successful call for *category*.

        Resets consecutive failures.  If the circuit is HALF_OPEN and
        the success threshold is met, transitions to CLOSED.
        """
        with self._lock:
            cat = self._get_category(category)
            cat.consecutive_failures = 0
            cat.total_successes += 1
            cat.last_success_time = time.monotonic()

            if cat.state is CircuitState.HALF_OPEN:
                cat.state = CircuitState.CLOSED

    def record_failure(self, category: str = "default") -> None:
        """Record a failed call for *category*.

        Increments failures.  HALF_OPEN immediately re-opens.  CLOSED
        opens once the failure threshold is reached.
        """
        with self._lock:
            cat = self._get_category(category)
            cat.consecutive_failures += 1
            cat.total_failures += 1
            cat.last_failure_time = time.monotonic()

            if cat.state is CircuitState.HALF_OPEN:
                cat.state = CircuitState.OPEN
            elif cat.state is CircuitState.CLOSED:
                if cat.consecutive_failures >= self._failure_threshold:
                    cat.state = CircuitState.OPEN

    def get_state(self, category: str = "default") -> dict:
        """Return a snapshot of the state for *category*."""
        with self._lock:
            cat = self._get_category(category)
            return {
                "state": cat.state.value,
                "consecutive_failures": cat.consecutive_failures,
                "total_successes": cat.total_successes,
                "total_failures": cat.total_failures,
            }

    def reset(self, category: Optional[str] = None) -> None:
        """Reset one category or all categories to initial state."""
        with self._lock:
            if category is not None:
                if category in self._categories:
                    self._categories[category] = CategoryState()
            else:
                self._categories.clear()
