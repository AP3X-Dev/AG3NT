"""ErrorClassifier — categorises errors by type, severity, and affected modules."""

from __future__ import annotations

import logging
import re
from typing import Any

from ag3nt_agent.autofix.context import FixContext

logger = logging.getLogger(__name__)

# ---- Pattern table -------------------------------------------------------
# Each entry is (compiled_regex, category).  First match wins.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"SyntaxError|IndentationError|TabError", re.I), "syntax_error"),
    (re.compile(r"ModuleNotFoundError|ImportError|No module named", re.I), "import_error"),
    (re.compile(r"ConnectionError|ConnectionRefused|ConnectionReset|URLError|socket\.error|ECONNREFUSED", re.I), "connection_error"),
    (re.compile(r"PermissionError|EACCES|Permission denied|Access denied", re.I), "permission_error"),
    (re.compile(r"TypeError", re.I), "type_error"),
    (re.compile(r"TimeoutError|Timeout|ETIMEDOUT|deadline exceeded", re.I), "timeout_error"),
    (re.compile(r"MemoryError|ResourceExhausted|OutOfMemory|OOM", re.I), "resource_error"),
    (re.compile(r"KeyError", re.I), "key_error"),
    (re.compile(r"AttributeError", re.I), "attribute_error"),
    (re.compile(r"ValueError", re.I), "value_error"),
    (re.compile(r"RuntimeError", re.I), "runtime_error"),
]

# ---- Severity map ---------------------------------------------------------
_SEVERITY: dict[str, str] = {
    "syntax_error": "high",
    "permission_error": "high",
    "resource_error": "high",
    "import_error": "medium",
    "connection_error": "medium",
    "timeout_error": "medium",
    "type_error": "medium",
    "key_error": "low",
    "attribute_error": "low",
    "value_error": "low",
    "runtime_error": "medium",
    "unknown": "low",
}

# Regex helpers for extracting module names from error messages.
_MODULE_RE = re.compile(r"No module named ['\"]?(\S+?)['\"]?(?:\s|$)")
_FILE_RE = re.compile(r'File "([^"]+)"')


class ErrorClassifier:
    """Classify errors by category, severity, and affected modules.

    Implements the ``PipelineStage`` protocol (``async process(ctx) -> FixContext``).
    """

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    async def process(self, ctx: FixContext) -> FixContext:
        """Analyse errors in *ctx* and set category / severity / affected_modules."""
        categories: list[str] = []
        modules: set[str] = set()

        for err in ctx.errors:
            msg = self._error_message(err)
            cat = self._classify_message(msg)
            categories.append(cat)
            modules.update(self._extract_modules(msg))

        # Pick the most common category, or the first one.
        ctx.category = self._dominant_category(categories)
        ctx.severity = _SEVERITY.get(ctx.category, "low")
        ctx.affected_modules = sorted(modules) if modules else ctx.affected_modules

        logger.info(
            "Classified FixContext %s: category=%s severity=%s modules=%s",
            ctx.id,
            ctx.category,
            ctx.severity,
            ctx.affected_modules,
        )
        return ctx

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error_message(err: dict[str, Any]) -> str:
        """Extract a displayable error message from an error dict."""
        for key in ("message", "error", "detail", "msg"):
            if key in err and err[key]:
                return str(err[key])
        return str(err)

    @staticmethod
    def _classify_message(msg: str) -> str:
        """Return the error category for a given message string."""
        for pattern, category in _PATTERNS:
            if pattern.search(msg):
                return category
        return "unknown"

    @staticmethod
    def _extract_modules(msg: str) -> set[str]:
        """Extract module / file names from an error message."""
        found: set[str] = set()
        for m in _MODULE_RE.finditer(msg):
            found.add(m.group(1))
        for m in _FILE_RE.finditer(msg):
            found.add(m.group(1))
        return found

    @staticmethod
    def _dominant_category(categories: list[str]) -> str:
        """Return the most frequently occurring category."""
        if not categories:
            return "unknown"
        counts: dict[str, int] = {}
        for c in categories:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=lambda k: counts[k])
