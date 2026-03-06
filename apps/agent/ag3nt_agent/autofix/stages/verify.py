"""FixVerifier — verifies fixes via compile, import, and test checks."""

from __future__ import annotations

import asyncio
import logging
import py_compile
import tempfile
from pathlib import Path
from typing import Any

from ag3nt_agent.autofix.context import FixContext

logger = logging.getLogger(__name__)


class FixVerifier:
    """Run verification checks (compile, import, test) on the fix result.

    Implements the ``PipelineStage`` protocol.

    Parameters
    ----------
    working_dir:
        Root directory of the project.
    """

    def __init__(self, working_dir: str = ".") -> None:
        self._working_dir = working_dir

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    async def process(self, ctx: FixContext) -> FixContext:
        """Run all verification checks and populate ``ctx.validation``."""
        results: dict[str, Any] = {}

        # Compile check on relevant .py files
        py_files = [f for f in ctx.relevant_files if f.endswith(".py")]
        if py_files:
            ok, detail = await self._compile_check(py_files)
            results["compile"] = {"passed": ok, "detail": detail}
        else:
            results["compile"] = {"passed": True, "detail": "no .py files to check"}

        # Import check on affected modules
        if ctx.affected_modules:
            ok, detail = await self._import_check(ctx.affected_modules)
            results["import"] = {"passed": ok, "detail": detail}
        else:
            results["import"] = {"passed": True, "detail": "no modules to check"}

        # Test check
        test_files = self._find_test_files(ctx.relevant_files)
        if test_files:
            ok, detail = await self._run_tests(test_files)
            results["test"] = {"passed": ok, "detail": detail}
        else:
            results["test"] = {"passed": True, "detail": "no test files found"}

        ctx.validation = results

        all_passed = all(v.get("passed", False) for v in results.values())
        if all_passed and ctx.outcome != "skipped":
            ctx.outcome = "success"
        elif not all_passed:
            ctx.outcome = "verification_failed"
            failed_stages = [k for k, v in results.items() if not v.get("passed")]
            logger.warning(
                "Verification failed for FixContext %s: %s",
                ctx.id,
                failed_stages,
            )

        logger.info(
            "Verified FixContext %s: %s",
            ctx.id,
            {k: v["passed"] for k, v in results.items()},
        )
        return ctx

    # ------------------------------------------------------------------
    # Compile check
    # ------------------------------------------------------------------

    async def _compile_check(self, files: list[str]) -> tuple[bool, str]:
        """Run ``py_compile`` on each file.  Returns (all_ok, detail)."""
        failures: list[str] = []
        for filepath in files:
            try:
                py_compile.compile(filepath, doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{filepath}: {exc}")
            except Exception as exc:
                failures.append(f"{filepath}: {exc}")

        if failures:
            return False, "; ".join(failures)
        return True, f"all {len(files)} files compiled OK"

    # ------------------------------------------------------------------
    # Import check
    # ------------------------------------------------------------------

    async def _import_check(self, modules: list[str]) -> tuple[bool, str]:
        """Try to import each module name.  Returns (all_ok, detail).

        Runs in a subprocess to avoid polluting the current interpreter.
        """
        failures: list[str] = []
        for mod in modules:
            # Skip file-path-style entries.
            if "/" in mod or "\\" in mod or mod.endswith(".py"):
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    "python", "-c", f"import {mod}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._working_dir,
                )
                _, stderr = await proc.communicate()
                if proc.returncode != 0:
                    failures.append(f"{mod}: {stderr.decode(errors='replace').strip()}")
            except Exception as exc:
                failures.append(f"{mod}: {exc}")

        if failures:
            return False, "; ".join(failures)
        return True, "all modules importable"

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    async def _run_tests(self, test_files: list[str]) -> tuple[bool, str]:
        """Run ``pytest`` on the given test files.  Returns (all_ok, detail)."""
        try:
            cmd = ["python", "-m", "pytest", "--tb=short", "-q", "--no-header"] + test_files
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode(errors="replace").strip()
            if proc.returncode == 0:
                return True, output or "tests passed"
            return False, output or stderr.decode(errors="replace").strip()
        except Exception as exc:
            return False, f"test runner error: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_test_files(relevant_files: list[str]) -> list[str]:
        """Filter *relevant_files* to those that look like test files."""
        return [
            f
            for f in relevant_files
            if Path(f).name.startswith("test_") or Path(f).name.endswith("_test.py")
        ]
