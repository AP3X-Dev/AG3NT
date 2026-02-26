"""Pipeline runner and PipelineStage protocol for the autofix system."""

import logging
import time
from typing import Protocol, runtime_checkable

from ag3nt_agent.autofix.context import FixContext

logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineStage(Protocol):
    async def process(self, ctx: FixContext) -> FixContext: ...


class Pipeline:
    def __init__(self) -> None:
        self._stages: list[PipelineStage] = []

    def add_stage(self, stage: PipelineStage) -> None:
        self._stages.append(stage)

    async def run(self, ctx: FixContext) -> FixContext:
        for stage in self._stages:
            stage_name = type(stage).__name__
            start = time.monotonic()
            try:
                ctx = await stage.process(ctx)
                elapsed = int((time.monotonic() - start) * 1000)
                logger.debug("Stage %s completed in %dms", stage_name, elapsed)
            except Exception as exc:
                logger.error("Stage %s failed: %s", stage_name, exc)
                ctx.outcome = "error"
                break
        return ctx
