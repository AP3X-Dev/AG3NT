"""ProviderFailoverLoop — LLM provider health monitoring with circuit breakers.

Monitors LLM provider health and selects the best available provider from a
configured fallback chain.  Integrates per-provider circuit breakers (reusing
``ag3nt_agent.autofix.circuit_breaker.CircuitBreaker``) and tracks latency via
an exponential moving average (EMA).

Events consumed:
    provider.error    — payload: {provider, error}
    provider.response — payload: {provider, latency_ms}

Events emitted:
    provider.failover — payload: {from_provider, to_provider, reason}
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ag3nt_agent.autofix.circuit_breaker import CircuitBreaker
from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.loops.failover")


# ------------------------------------------------------------------
# Provider health dataclass
# ------------------------------------------------------------------

@dataclass
class ProviderHealth:
    """Per-provider health tracking."""

    provider_name: str
    status: str = "healthy"  # "healthy", "degraded", "down"
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    rate_limit_until: float = 0.0
    total_requests: int = 0
    total_errors: int = 0

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_errors / self.total_requests

    @property
    def is_available(self) -> bool:
        return self.status != "down" and time.monotonic() >= self.rate_limit_until

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "status": self.status,
            "consecutive_failures": self.consecutive_failures,
            "avg_latency_ms": self.avg_latency_ms,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": self.error_rate,
        }


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class FailoverConfig:
    """Configuration for ProviderFailoverLoop."""

    fallback_chain: list[str] = field(default_factory=list)
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0
    ema_alpha: float = 0.3


# ------------------------------------------------------------------
# ProviderFailoverLoop
# ------------------------------------------------------------------

class ProviderFailoverLoop:
    """Monitors LLM provider health and selects the best available provider
    from a configured fallback chain.  Integrates circuit-breaker logic to
    avoid hammering failing providers.

    Usage::

        bus = EventBus()
        loop = ProviderFailoverLoop(
            fallback_chain=["openai", "anthropic", "local"],
            bus=bus,
        )
        await loop.start()
        best = loop.select_provider()
    """

    def __init__(
        self,
        fallback_chain: list[str],
        bus: EventBus,
        *,
        config: FailoverConfig | None = None,
    ) -> None:
        cfg = config or FailoverConfig(fallback_chain=list(fallback_chain))
        self._chain = list(fallback_chain)
        self._bus = bus
        self._ema_alpha = cfg.ema_alpha

        self._health: dict[str, ProviderHealth] = {
            name: ProviderHealth(provider_name=name) for name in fallback_chain
        }
        self._current_index: int = 0

        # Reuse AG3NT's thread-safe CircuitBreaker with per-provider categories
        self._cb = CircuitBreaker(
            failure_threshold=cfg.failure_threshold,
            reset_timeout_s=cfg.recovery_timeout_s,
        )

        self._sub_ids: list[str] = []
        self._subscribed = False

        # Optional external callback: (provider_name) -> None
        self.on_failover: Callable[[str], None] | None = None

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to provider events on the bus."""
        if self._subscribed:
            return
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_provider_failed,
                event_types={"provider.error"},
            )
        )
        self._sub_ids.append(
            self._bus.subscribe(
                self._on_provider_response,
                event_types={"provider.response"},
            )
        )
        self._subscribed = True
        logger.info("ProviderFailoverLoop started with chain: %s", self._chain)

    async def stop(self) -> None:
        """Unsubscribe from all events."""
        for sub_id in self._sub_ids:
            self._bus.unsubscribe(sub_id)
        self._sub_ids.clear()
        self._subscribed = False
        logger.info("ProviderFailoverLoop stopped")

    # -- event handlers ------------------------------------------------------

    async def _on_provider_failed(self, event: Event) -> None:
        """Handle provider failure — update health and trigger failover."""
        name = event.payload.get("provider", "")
        error = event.payload.get("error", "")
        health = self._health.get(name)
        if health is None:
            return

        now = time.monotonic()
        health.consecutive_failures += 1
        health.total_errors += 1
        health.total_requests += 1
        health.last_failure = now

        self._cb.record_failure(category=name)

        if health.consecutive_failures >= 3:
            health.status = "down"
        elif health.consecutive_failures >= 1:
            health.status = "degraded"

        logger.warning(
            "Provider %s failed (%d consecutive): %s",
            name,
            health.consecutive_failures,
            str(error)[:200],
        )

        # If the current provider failed, trigger failover
        if (
            self._chain
            and self._current_index < len(self._chain)
            and self._chain[self._current_index] == name
        ):
            next_provider = self.select_provider()
            if next_provider and next_provider != name:
                failover_event = Event(
                    event_type="provider.failover",
                    source="provider_failover",
                    payload={
                        "from_provider": name,
                        "to_provider": next_provider,
                        "reason": str(error),
                    },
                    priority=EventPriority.HIGH,
                )
                await self._bus.publish(failover_event)
                logger.info("Failover: %s -> %s", name, next_provider)
                if self.on_failover:
                    self.on_failover(next_provider)

    async def _on_provider_response(self, event: Event) -> None:
        """Record successful response — update health stats."""
        name = event.payload.get("provider", "")
        health = self._health.get(name)
        if health is None:
            return

        health.consecutive_failures = 0
        health.total_requests += 1
        health.last_success = time.monotonic()
        health.status = "healthy"

        latency = event.payload.get("latency_ms", 0.0)
        if isinstance(latency, (int, float)) and latency > 0:
            alpha = self._ema_alpha
            if health.avg_latency_ms == 0.0:
                health.avg_latency_ms = float(latency)
            else:
                health.avg_latency_ms = (
                    alpha * float(latency) + (1 - alpha) * health.avg_latency_ms
                )

        self._cb.record_success(category=name)

    # -- selection -----------------------------------------------------------

    def select_provider(self) -> Optional[str]:
        """Select the best available provider based on health and circuit-breaker state."""
        for idx, name in enumerate(self._chain):
            health = self._health[name]
            if health.is_available and self._cb.allow(category=name):
                self._current_index = idx
                return name

        # All providers down — try the one with the fewest consecutive failures
        fallback = (
            min(self._chain, key=lambda n: self._health[n].consecutive_failures)
            if self._chain
            else None
        )
        if fallback:
            self._current_index = self._chain.index(fallback)
        return fallback

    @property
    def current_provider(self) -> Optional[str]:
        """Return the currently selected provider, if any."""
        if self._chain and self._current_index < len(self._chain):
            return self._chain[self._current_index]
        return None

    def get_health_report(self) -> dict[str, dict[str, Any]]:
        """Return health status of all providers."""
        return {name: h.to_dict() for name, h in self._health.items()}
