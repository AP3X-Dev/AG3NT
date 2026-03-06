"""LearningRecorder — records fix outcomes for future scoring."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from ag3nt_agent.autofix.context import FixContext

if TYPE_CHECKING:
    from ag3nt_agent.autofix.learning import FixHistory

logger = logging.getLogger(__name__)


class LearningRecorder:
    """Record the outcome of a fix pipeline run into :class:`FixHistory`.

    Implements the ``PipelineStage`` protocol.
    """

    def __init__(self, history: Optional["FixHistory"] = None) -> None:
        self._history = history

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    async def process(self, ctx: FixContext) -> FixContext:
        """Record the outcome and log a summary."""
        if self._history is not None:
            try:
                self._history.record(ctx)
                logger.info(
                    "Recorded outcome for FixContext %s: category=%s outcome=%s",
                    ctx.id,
                    ctx.category,
                    ctx.outcome,
                )
            except Exception as exc:
                logger.error("Failed to record fix history: %s", exc)
        else:
            logger.debug(
                "No FixHistory configured — skipping recording for %s", ctx.id
            )

        # Log a human-readable summary regardless.
        logger.info(
            "AutoFix summary [%s]: category=%s severity=%s confidence=%.2f "
            "risk=%s outcome=%s duration=%dms",
            ctx.id,
            ctx.category,
            ctx.severity,
            ctx.confidence,
            ctx.risk_level,
            ctx.outcome or "unknown",
            ctx.fix_duration_ms,
        )
        return ctx
