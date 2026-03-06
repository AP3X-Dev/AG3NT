"""Safety hooks for AG3NT lifecycle.

Provides three hooks that register on :class:`PhaseHookManager`:

* **protect_core** (PRE_TOOL) -- blocks writes to critical files.
* **block_danger** (PRE_TOOL) -- blocks dangerous shell commands.
* **compile_check** (POST_TOOL) -- verifies syntax of edited Python files.

All hooks use only Python stdlib modules.

Usage::

    from ag3nt_agent.hooks import PhaseHookManager
    from ag3nt_agent.hooks.safety import register_safety_hooks

    mgr = PhaseHookManager()
    register_safety_hooks(mgr)
    mgr.start()
"""

from __future__ import annotations

import logging
import py_compile
import re
import tempfile
from fnmatch import fnmatch
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from ag3nt_agent.hooks import CANCEL, HookPhase, PhaseHookManager

logger = logging.getLogger("ag3nt.hooks.safety")

# ---------------------------------------------------------------------------
# Configuration — protected file patterns
# ---------------------------------------------------------------------------

#: Glob patterns matched against the file path to decide whether a write
#: should be blocked.  Matching is case-insensitive and uses :func:`fnmatch`.
DEFAULT_PROTECTED_PATTERNS: list[str] = [
    # Environment / secrets
    ".env",
    ".env.*",
    ".env.local",
    ".env.production",
    "*.env",
    # PID / lock files
    "*.pid",
    "*.lock",
    # Core package init files (root-level packages only)
    "__init__.py",
    # CI/CD configuration
    ".github/workflows/*",
    ".gitlab-ci.yml",
    ".gitlab-ci.yaml",
    "Jenkinsfile",
    ".circleci/config.yml",
    ".travis.yml",
    "azure-pipelines.yml",
    "bitbucket-pipelines.yml",
    # Docker
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
]

# ---------------------------------------------------------------------------
# Configuration — dangerous command patterns
# ---------------------------------------------------------------------------

#: Regex patterns that flag a shell command as dangerous.  Each pattern is
#: compiled with ``re.IGNORECASE``.
DEFAULT_DANGER_PATTERNS: list[str] = [
    # Destructive filesystem operations
    r"rm\s+(-\w*\s+)*-\w*r\w*\s+/\s*$",  # rm -rf /
    r"rm\s+(-\w*\s+)*-\w*r\w*\s+/\s+",  # rm -rf / (with trailing content)
    r"rm\s+-rf\s+/(?:\s|$)",  # explicit rm -rf /
    r"rm\s+-rf\s+\*",  # rm -rf *
    r"rm\s+-rf\s+~",  # rm -rf ~
    # SQL injection / destructive SQL
    r"DROP\s+(TABLE|DATABASE|SCHEMA|INDEX)",
    r"TRUNCATE\s+TABLE",
    r"DELETE\s+FROM\s+\S+\s*;?\s*$",  # DELETE FROM table (no WHERE)
    # Disk formatting
    r"format\s+[a-zA-Z]:",
    r"mkfs\b",
    r"dd\s+if=",
    # Fork bomb
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    # Shutdown / reboot
    r"shutdown\b",
    r"reboot\b",
    r"init\s+0",
    # chmod 777 recursively
    r"chmod\s+(-\w+\s+)*777\s+/",
    # Overwrite disk
    r">\s*/dev/sd[a-z]",
]

# ---------------------------------------------------------------------------
# Tool name sets
# ---------------------------------------------------------------------------

#: Tool names considered "write" tools whose file paths should be checked
#: against the protected patterns.
WRITE_TOOL_NAMES: set[str] = {
    "Edit",
    "Write",
    "edit",
    "write",
    "file_edit",
    "file_write",
    "create_file",
    "write_file",
    "edit_file",
}

#: Tool names considered "exec" tools whose commands should be checked
#: against the dangerous patterns.
EXEC_TOOL_NAMES: set[str] = {
    "Bash",
    "bash",
    "exec",
    "exec_command",
    "shell",
    "run_command",
    "terminal",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_file_path(arguments: dict[str, Any] | str | None) -> str | None:
    """Best-effort extraction of a file path from tool arguments."""
    if arguments is None:
        return None
    if isinstance(arguments, str):
        return arguments
    for key in ("file_path", "path", "filepath", "filename", "file", "target"):
        val = arguments.get(key)
        if val and isinstance(val, str):
            return val
    return None


def _extract_command(arguments: dict[str, Any] | str | None) -> str | None:
    """Best-effort extraction of a shell command from tool arguments."""
    if arguments is None:
        return None
    if isinstance(arguments, str):
        return arguments
    for key in ("command", "cmd", "script", "code", "input"):
        val = arguments.get(key)
        if val and isinstance(val, str):
            return val
    return None


def _matches_protected(
    file_path: str, patterns: list[str] | None = None
) -> str | None:
    """Return the first matching pattern if *file_path* matches a protected pattern.

    Matching is performed against the **filename** component (``PurePosixPath``
    or ``PureWindowsPath``, whichever parses more segments) and also against
    the full relative path for patterns that contain ``/``.
    """
    patterns = patterns if patterns is not None else DEFAULT_PROTECTED_PATTERNS
    # Normalise to forward slashes for consistent matching.
    normalised = file_path.replace("\\", "/")

    # Try both posix and windows parsing to get a sensible filename.
    posix_name = PurePosixPath(normalised).name
    win_name = PureWindowsPath(file_path).name
    # Use whichever produced a non-empty name, preferring the normalised one.
    filename = posix_name or win_name

    for pat in patterns:
        # Pattern contains a directory separator — match against the full path.
        if "/" in pat:
            if fnmatch(normalised.lower(), f"*/{pat.lower()}") or fnmatch(
                normalised.lower(), pat.lower()
            ):
                return pat
        else:
            if fnmatch(filename.lower(), pat.lower()):
                return pat
    return None


def _matches_danger(command: str, patterns: list[str] | None = None) -> str | None:
    """Return the first matching pattern if *command* matches a dangerous pattern."""
    patterns = patterns if patterns is not None else DEFAULT_DANGER_PATTERNS
    for pat in patterns:
        try:
            if re.search(pat, command, re.IGNORECASE):
                return pat
        except re.error:
            logger.warning("Invalid danger regex pattern: %s", pat)
            continue
    return None


# ---------------------------------------------------------------------------
# Hook: protect_core (PRE_TOOL)
# ---------------------------------------------------------------------------


async def protect_core(ctx: dict[str, Any]) -> str | None:
    """Block writes to critical/protected files.

    Inspects the ``tool_name`` and ``arguments`` keys of *ctx*.  If the tool
    is a write-type tool and the target file matches a protected pattern, the
    hook returns :data:`CANCEL`.

    The set of protected patterns can be overridden via the ``protected_patterns``
    key in *ctx* (a list of glob strings).
    """
    tool_name: str = ctx.get("tool_name") or ctx.get("tool", "")
    if tool_name not in WRITE_TOOL_NAMES:
        return None

    arguments = ctx.get("arguments", ctx.get("args"))
    file_path = _extract_file_path(arguments)
    if not file_path:
        return None

    patterns = ctx.get("protected_patterns")
    match = _matches_protected(file_path, patterns)
    if match:
        logger.warning(
            "protect_core: BLOCKED write to protected file %r (matched pattern %r)",
            file_path,
            match,
        )
        return CANCEL

    return None


# ---------------------------------------------------------------------------
# Hook: block_danger (PRE_TOOL)
# ---------------------------------------------------------------------------


async def block_danger(ctx: dict[str, Any]) -> str | None:
    """Block dangerous shell commands.

    Inspects the ``tool_name`` and ``arguments`` keys of *ctx*.  If the tool
    is an exec-type tool and the command matches a dangerous pattern, the hook
    returns :data:`CANCEL`.

    The set of dangerous patterns can be overridden via the ``danger_patterns``
    key in *ctx* (a list of regex strings).
    """
    tool_name: str = ctx.get("tool_name") or ctx.get("tool", "")
    if tool_name not in EXEC_TOOL_NAMES:
        return None

    arguments = ctx.get("arguments", ctx.get("args"))
    command = _extract_command(arguments)
    if not command:
        return None

    patterns = ctx.get("danger_patterns")
    match = _matches_danger(command, patterns)
    if match:
        logger.warning(
            "block_danger: BLOCKED dangerous command %r (matched pattern %r)",
            command[:120],
            match,
        )
        return CANCEL

    return None


# ---------------------------------------------------------------------------
# Hook: compile_check (POST_TOOL)
# ---------------------------------------------------------------------------


async def compile_check(ctx: dict[str, Any]) -> str | None:
    """Verify syntax of Python files after edit/write operations.

    Runs :func:`py_compile.compile` on the target file.  If the file has a
    syntax error the hook logs a warning but does **not** cancel (POST_TOOL
    cannot undo the edit — this is informational / for metrics).

    If the file content is available in ``ctx["result"]`` but the file doesn't
    exist on disk, a temporary file is used for the syntax check.
    """
    tool_name: str = ctx.get("tool_name") or ctx.get("tool", "")
    if tool_name not in WRITE_TOOL_NAMES:
        return None

    arguments = ctx.get("arguments", ctx.get("args"))
    file_path = _extract_file_path(arguments)
    if not file_path:
        return None

    # Only check Python files.
    if not file_path.endswith(".py"):
        return None

    try:
        py_compile.compile(file_path, doraise=True)
        logger.debug("compile_check: %s OK", file_path)
    except py_compile.PyCompileError as exc:
        logger.warning("compile_check: syntax error in %s — %s", file_path, exc)
    except OSError:
        # File may not exist on disk yet (e.g. in-memory preview).
        # Attempt to check content from the result if available.
        result = ctx.get("result")
        content: str | None = None
        if isinstance(result, dict):
            content = result.get("content") or result.get("output")
        elif isinstance(result, str):
            content = result

        if content:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False
                ) as tmp:
                    tmp.write(content)
                    tmp.flush()
                    py_compile.compile(tmp.name, doraise=True)
                    logger.debug(
                        "compile_check: %s (from result content) OK", file_path
                    )
            except py_compile.PyCompileError as exc:
                logger.warning(
                    "compile_check: syntax error in %s (from result content) — %s",
                    file_path,
                    exc,
                )
            except Exception:
                logger.debug(
                    "compile_check: could not verify %s from result content",
                    file_path,
                )
        else:
            logger.debug("compile_check: file %s not found on disk, skipping", file_path)

    return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_safety_hooks(
    hook_manager: PhaseHookManager,
    *,
    source: str = "static",
) -> None:
    """Register all safety hooks on *hook_manager*.

    Parameters
    ----------
    hook_manager:
        The :class:`PhaseHookManager` instance to register on.
    source:
        Origin tag passed to each :meth:`~PhaseHookManager.register` call.
        Defaults to ``"static"``.
    """
    hook_manager.register(
        HookPhase.PRE_TOOL,
        protect_core,
        priority=10,
        name="protect_core",
        source=source,
    )
    hook_manager.register(
        HookPhase.PRE_TOOL,
        block_danger,
        priority=10,
        name="block_danger",
        source=source,
    )
    hook_manager.register(
        HookPhase.POST_TOOL,
        compile_check,
        priority=50,
        name="compile_check",
        source=source,
    )

    logger.info(
        "Safety hooks registered: protect_core, block_danger, compile_check "
        "(source=%s)",
        source,
    )
