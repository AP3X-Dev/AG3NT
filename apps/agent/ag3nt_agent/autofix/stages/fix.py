"""AgentFixer — builds LLM prompts for fix generation.

Actual LLM integration will be wired in Task 29 (integration).  For now
the stage builds the prompt, logs it, and returns the context with
``fix_response=""`` and ``outcome="skipped"``.
"""

from __future__ import annotations

import logging
from textwrap import dedent
from typing import Any

from ag3nt_agent.autofix.context import FixContext

logger = logging.getLogger(__name__)


class AgentFixer:
    """Generate an LLM prompt describing the fix task.

    Implements the ``PipelineStage`` protocol.

    Parameters
    ----------
    working_dir:
        Root directory of the project being fixed.
    """

    def __init__(self, working_dir: str = ".") -> None:
        self._working_dir = working_dir

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    async def process(self, ctx: FixContext) -> FixContext:
        """Build the fix prompt and (for now) mark the context as skipped."""
        if ctx.prior_attempts:
            prompt = self._build_retry_prompt(ctx)
        else:
            prompt = self._build_prompt(ctx)

        logger.info(
            "AgentFixer built prompt for FixContext %s (%d chars). "
            "LLM call not yet wired — marking as skipped.",
            ctx.id,
            len(prompt),
        )
        logger.debug("Prompt:\n%s", prompt)

        ctx.fix_response = ""
        ctx.outcome = "skipped"
        return ctx

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_prompt(self, ctx: FixContext) -> str:
        """Construct the initial fix prompt from context."""
        error_block = self._format_errors(ctx.errors)
        stack_block = self._format_stack_frames(ctx)
        files_block = self._format_relevant_files(ctx)
        git_block = self._format_git_context(ctx)

        return dedent(f"""\
            You are an expert software engineer tasked with fixing an error.

            ## Error Information
            Category: {ctx.category}
            Severity: {ctx.severity}
            Confidence: {ctx.confidence:.2f}

            ### Error Details
            {error_block}

            ### Stack Trace
            {stack_block}

            ### Relevant Files
            {files_block}

            ### Recent Git Changes
            {git_block}

            ## Instructions
            1. Analyse the error and its root cause.
            2. Propose a minimal, targeted fix.
            3. Return the fix as a unified diff that can be applied with `git apply`.
            4. Explain your reasoning briefly.
        """)

    def _build_retry_prompt(self, ctx: FixContext) -> str:
        """Construct a retry prompt that includes prior attempt feedback."""
        base = self._build_prompt(ctx)

        attempts_block = "\n".join(
            f"- Attempt {a.attempt_number}: failed at {a.verification_stage_failed} — {a.failure_detail}"
            for a in ctx.prior_attempts
        )

        return base + dedent(f"""\

            ## Prior Attempts (all failed)
            {attempts_block}

            Please avoid the approaches above and try a different strategy.
        """)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_errors(errors: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for i, err in enumerate(errors, 1):
            msg = err.get("message", err.get("error", str(err)))
            lines.append(f"{i}. {msg}")
        return "\n".join(lines) or "(no error messages)"

    @staticmethod
    def _format_stack_frames(ctx: FixContext) -> str:
        if not ctx.stack_frames:
            return "(no stack trace available)"
        lines: list[str] = []
        for f in ctx.stack_frames:
            lines.append(f'  File "{f.file}", line {f.line}, in {f.function}')
            if f.source_snippet:
                lines.append(f"    {f.source_snippet}")
        return "\n".join(lines)

    @staticmethod
    def _format_relevant_files(ctx: FixContext) -> str:
        if not ctx.relevant_files:
            return "(none found)"
        return "\n".join(f"- {f}" for f in ctx.relevant_files)

    @staticmethod
    def _format_git_context(ctx: FixContext) -> str:
        parts: list[str] = []
        if ctx.git_context.recent_logs:
            parts.append("Recent commits:\n" + "\n".join(f"  {l}" for l in ctx.git_context.recent_logs))
        if ctx.git_context.diffs:
            diff_preview = ctx.git_context.diffs[:2000]
            parts.append(f"Diff (first 2000 chars):\n{diff_preview}")
        return "\n".join(parts) or "(no git context)"
