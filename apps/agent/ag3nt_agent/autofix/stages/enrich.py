"""ContextEnricher — gathers stack-trace, git, and file-system context."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from ag3nt_agent.autofix.context import FixContext, GitContext, StackFrame

logger = logging.getLogger(__name__)

# Regex for standard Python traceback lines:
#   File "/some/path.py", line 42, in func_name
_TB_LINE_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)'
)


class ContextEnricher:
    """Populate enrichment fields on :class:`FixContext`.

    Gathers:
    * Stack frames parsed from error tracebacks.
    * Git diff / recent log lines (graceful degradation if git unavailable).
    * Relevant source files for the affected modules.

    Implements the ``PipelineStage`` protocol.
    """

    def __init__(self, working_dir: str = ".") -> None:
        self._working_dir = working_dir

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    async def process(self, ctx: FixContext) -> FixContext:
        """Enrich *ctx* with stack frames, git context, and relevant files."""
        # Stack frames
        ctx.stack_frames = self._parse_stack_traces(ctx.errors)

        # Relevant files (from affected_modules + stack frames)
        all_modules = set(ctx.affected_modules)
        for frame in ctx.stack_frames:
            if frame.file:
                all_modules.add(frame.file)
        ctx.relevant_files = self._find_relevant_files(
            list(all_modules), self._working_dir
        )

        # Git context (uses the files we found)
        ctx.git_context = await self._get_git_context(ctx.relevant_files)

        logger.info(
            "Enriched FixContext %s: %d stack frames, %d relevant files",
            ctx.id,
            len(ctx.stack_frames),
            len(ctx.relevant_files),
        )
        return ctx

    # ------------------------------------------------------------------
    # Stack-trace parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_stack_traces(errors: list[dict[str, Any]]) -> list[StackFrame]:
        """Extract :class:`StackFrame` objects from error dicts."""
        frames: list[StackFrame] = []
        seen: set[tuple[str, int, str]] = set()

        for err in errors:
            text = err.get("traceback", err.get("message", str(err)))
            for m in _TB_LINE_RE.finditer(str(text)):
                key = (m.group("file"), int(m.group("line")), m.group("func"))
                if key in seen:
                    continue
                seen.add(key)
                frames.append(
                    StackFrame(
                        file=m.group("file"),
                        line=int(m.group("line")),
                        function=m.group("func"),
                    )
                )
        return frames

    # ------------------------------------------------------------------
    # Git context
    # ------------------------------------------------------------------

    async def _get_git_context(self, files: list[str]) -> GitContext:
        """Run ``git diff`` and ``git log --oneline -5`` in *working_dir*.

        Returns an empty :class:`GitContext` if git is unavailable or the
        directory is not a repository.
        """
        ctx = GitContext()
        try:
            diff_cmd = ["git", "diff", "--", *files] if files else ["git", "diff"]
            proc = await asyncio.create_subprocess_exec(
                *diff_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )
            stdout, _ = await proc.communicate()
            ctx.diffs = stdout.decode(errors="replace").strip()

            log_proc = await asyncio.create_subprocess_exec(
                "git", "log", "--oneline", "-5",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )
            log_out, _ = await log_proc.communicate()
            ctx.recent_logs = [
                line for line in log_out.decode(errors="replace").strip().splitlines()
                if line
            ]
        except Exception as exc:
            logger.debug("Git context unavailable: %s", exc)
        return ctx

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_relevant_files(modules: list[str], working_dir: str) -> list[str]:
        """Resolve module names / paths to real files on disk.

        Handles:
        * Absolute paths already on disk.
        * Dotted module names converted to ``<module>/...py`` paths.
        * Simple file names searched under *working_dir*.
        """
        found: list[str] = []
        base = Path(working_dir).resolve()

        for mod in modules:
            # Already an absolute or relative path?
            p = Path(mod)
            if p.is_absolute() and p.is_file():
                found.append(str(p))
                continue

            # Relative to working_dir?
            candidate = base / mod
            if candidate.is_file():
                found.append(str(candidate))
                continue

            # Dotted module -> path (e.g. "foo.bar" -> "foo/bar.py")
            as_path = mod.replace(".", os.sep) + ".py"
            candidate = base / as_path
            if candidate.is_file():
                found.append(str(candidate))
                continue

            # Package init?
            pkg_init = base / mod.replace(".", os.sep) / "__init__.py"
            if pkg_init.is_file():
                found.append(str(pkg_init))

        return sorted(set(found))
