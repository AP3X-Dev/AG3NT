"""ConfidenceScorer — calculates fix confidence, risk level, and verification depth."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ag3nt_agent.autofix.context import FixContext

if TYPE_CHECKING:
    from ag3nt_agent.autofix.learning import FixHistory

logger = logging.getLogger(__name__)

# Base confidence scores per error category.
_BASE_CONFIDENCE: dict[str, float] = {
    "syntax_error": 0.9,
    "import_error": 0.7,
    "key_error": 0.65,
    "attribute_error": 0.6,
    "value_error": 0.55,
    "type_error": 0.5,
    "permission_error": 0.45,
    "runtime_error": 0.4,
    "connection_error": 0.4,
    "timeout_error": 0.35,
    "resource_error": 0.3,
    "unknown": 0.2,
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class ConfidenceScorer:
    """Score a :class:`FixContext` with confidence, risk, and verification depth.

    Factors considered:
    * Base confidence for the error category.
    * Enrichment quality — presence of stack frames, relevant files.
    * Historical success rate (when a :class:`FixHistory` is supplied).

    Implements the ``PipelineStage`` protocol.
    """

    def __init__(self, history: Optional["FixHistory"] = None) -> None:
        self._history = history

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    async def process(self, ctx: FixContext) -> FixContext:
        """Calculate confidence and derived risk / depth on *ctx*."""
        base = _BASE_CONFIDENCE.get(ctx.category, 0.2)

        # Enrichment bonus: up to +0.15 for good context.
        enrichment = 0.0
        if ctx.stack_frames:
            enrichment += 0.07
        if ctx.relevant_files:
            enrichment += 0.05
        if ctx.git_context and ctx.git_context.diffs:
            enrichment += 0.03

        # History bonus / penalty.
        history_adj = 0.0
        if self._history is not None:
            rate = self._history.get_success_rate(ctx.category)
            if rate >= 0:
                # rate is 0.0-1.0; shift so 0.5 is neutral.
                history_adj = (rate - 0.5) * 0.2  # -0.1 .. +0.1

        ctx.confidence = _clamp(base + enrichment + history_adj)
        ctx.risk_level = self._risk_level(ctx.confidence)
        ctx.verification_depth = self._verification_depth(ctx.confidence, ctx.severity)

        logger.info(
            "Scored FixContext %s: confidence=%.2f risk=%s depth=%s",
            ctx.id,
            ctx.confidence,
            ctx.risk_level,
            ctx.verification_depth,
        )
        return ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_level(confidence: float) -> str:
        """Map confidence to a risk label."""
        if confidence >= 0.8:
            return "low"
        if confidence >= 0.5:
            return "medium"
        if confidence >= 0.3:
            return "high"
        return "critical"

    @staticmethod
    def _verification_depth(confidence: float, severity: str) -> str:
        """Determine how thorough verification should be."""
        if confidence >= 0.8 and severity != "high":
            return "minimal"
        if confidence >= 0.5:
            return "standard"
        return "thorough"
