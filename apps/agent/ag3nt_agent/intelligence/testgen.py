"""
TestGenerationLoop — Coverage Analysis and Test Skeleton Generation.

Maps source functions to their test counterparts, identifies missing
coverage, and generates test skeleton stubs.  Subscribes to
``sentinel.indexed`` events on the EventBus so it reacts whenever the
SentinelLoop finishes indexing a file.

Whisper-mode aware: when ``_whisper_mode`` is True, suppresses
automatic skeleton generation (analysis still runs).
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestMapping:
    """Maps a source function to its test counterpart."""

    source_func: str
    test_func: Optional[str] = None
    covered: bool = False


# ---------------------------------------------------------------------------
# TestGenerationLoop
# ---------------------------------------------------------------------------

class TestGenerationLoop:
    """Reactive loop that analyses test coverage and generates skeletons.

    Subscribes to ``sentinel.indexed`` events.  For each indexed Python
    source file it:

    1. Locates the corresponding test file (if any).
    2. Compares the function lists to find untested functions.
    3. Optionally generates test skeleton stubs (suppressed in whisper mode).

    Usage::

        bus = EventBus()
        loop = TestGenerationLoop(bus, working_dir="/path/to/repo")
        await loop.start()
        # ... sentinel events flow ...
        await loop.stop()
    """

    def __init__(self, bus: EventBus, working_dir: str = ".") -> None:
        self._bus = bus
        self._working_dir = os.path.abspath(working_dir)
        self._subscription_ids: list[str] = []
        self._running = False
        self._whisper_mode = False

        # Stats
        self._files_analysed = 0
        self._mappings: list[TestMapping] = []
        self._skeletons_generated = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to sentinel.indexed events."""
        if self._running:
            return

        self._running = True
        sub_id = self._bus.subscribe(
            handler=self._on_sentinel_indexed,
            event_types={"sentinel.indexed"},
        )
        self._subscription_ids.append(sub_id)

        logger.info("TestGenerationLoop started (working_dir=%s)", self._working_dir)

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self._running = False
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("TestGenerationLoop stopped")

    # -- event handler -------------------------------------------------------

    async def _on_sentinel_indexed(self, event: Event) -> None:
        """Check for missing test coverage when a file is indexed."""
        path = event.payload.get("path", "")
        if not path or not path.endswith(".py"):
            return

        # Skip test files themselves
        basename = os.path.basename(path)
        if basename.startswith("test_") or basename.startswith("conftest"):
            return

        try:
            test_path = self._find_test_file(path)
            mappings = self._analyze_coverage(path, test_path)
            self._mappings.extend(mappings)
            self._files_analysed += 1

            uncovered = [m for m in mappings if not m.covered]
            if uncovered:
                logger.info(
                    "TestGenerationLoop: %d uncovered functions in %s",
                    len(uncovered), path,
                )

                if not self._whisper_mode:
                    for mapping in uncovered:
                        skeleton = self._generate_skeleton(mapping.source_func, path)
                        if skeleton:
                            self._skeletons_generated += 1
                            logger.debug(
                                "Generated test skeleton for %s in %s",
                                mapping.source_func, path,
                            )
        except Exception:
            logger.exception("Error analysing coverage for %s", path)

    # -- analysis helpers ----------------------------------------------------

    def _find_test_file(self, source_path: str) -> Optional[str]:
        """Map a source file to its test file.

        Checks the following conventional locations:

        1. ``tests/test_<module>.py`` relative to the source directory.
        2. ``test_<module>.py`` in the same directory.
        3. ``tests/unit/test_<module>.py`` relative to the source directory.

        Returns ``None`` if no test file is found.
        """
        source_dir = os.path.dirname(source_path)
        basename = os.path.basename(source_path)
        test_name = f"test_{basename}"

        candidates = [
            os.path.join(source_dir, "tests", test_name),
            os.path.join(source_dir, test_name),
            os.path.join(source_dir, "tests", "unit", test_name),
        ]

        # Also check project-level tests/ directory
        project_root = self._working_dir
        rel = os.path.relpath(source_path, project_root)
        candidates.append(os.path.join(project_root, "tests", test_name))
        candidates.append(os.path.join(project_root, "tests", "unit", test_name))

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        return None

    def _analyze_coverage(
        self, source_path: str, test_path: Optional[str]
    ) -> list[TestMapping]:
        """Compare function lists between source and test file.

        Extracts top-level and class-method function names from the source
        file via AST parsing, then checks whether corresponding
        ``test_<func>`` names exist in the test file.

        Returns a list of :class:`TestMapping` for every source function.
        """
        source_funcs = self._extract_functions(source_path)
        if not source_funcs:
            return []

        test_funcs: set[str] = set()
        if test_path and os.path.isfile(test_path):
            test_funcs = set(self._extract_functions(test_path))

        mappings: list[TestMapping] = []
        for func in source_funcs:
            # Skip private/dunder helpers
            bare_name = func.split(".")[-1]  # handle Class.method
            expected_test = f"test_{bare_name}"

            # Check if any test function matches
            matched_test = None
            for tf in test_funcs:
                tf_bare = tf.split(".")[-1]
                if tf_bare == expected_test or tf_bare.startswith(f"test_{bare_name}_"):
                    matched_test = tf
                    break

            mappings.append(TestMapping(
                source_func=func,
                test_func=matched_test,
                covered=matched_test is not None,
            ))

        return mappings

    def _generate_skeleton(self, source_func: str, source_path: str) -> str:
        """Generate a test skeleton stub for the given source function.

        Returns a string containing a pytest-style test function skeleton.
        """
        bare_name = source_func.split(".")[-1]
        module_name = os.path.splitext(os.path.basename(source_path))[0]

        skeleton = (
            f"def test_{bare_name}():\n"
            f"    \"\"\"Test {source_func} from {module_name}.\"\"\"\n"
            f"    # TODO: implement test for {source_func}\n"
            f"    raise NotImplementedError\n"
        )
        return skeleton

    # -- internal utilities --------------------------------------------------

    @staticmethod
    def _extract_functions(file_path: str) -> list[str]:
        """Extract function/method names from a Python file using AST.

        Returns a list of qualified names (``ClassName.method`` for methods).
        """
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return []

        names: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.append(f"{node.name}.{item.name}")
        return names

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current status and statistics."""
        total = len(self._mappings)
        covered = sum(1 for m in self._mappings if m.covered)
        return {
            "running": self._running,
            "whisper_mode": self._whisper_mode,
            "working_dir": self._working_dir,
            "files_analysed": self._files_analysed,
            "total_mappings": total,
            "covered_mappings": covered,
            "uncovered_mappings": total - covered,
            "skeletons_generated": self._skeletons_generated,
        }
