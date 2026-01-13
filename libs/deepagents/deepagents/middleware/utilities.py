"""Middleware for utility tools: undo_edit, format_file, get_diagnostics, mermaid."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)

# --- Tool Descriptions ---

UNDO_EDIT_DESCRIPTION = """Undo the last edit made to a file.

Parameters:
- path: Absolute path to the file (required)

Returns a git-style diff showing the undone changes.

Use this tool when:
- You made an incorrect edit and need to revert it
- The user asks to undo a recent change
- You want to try a different approach after seeing the result of an edit

Note: Only the last edit can be undone. Multiple undos are not supported."""

FORMAT_FILE_DESCRIPTION = """Format a file using the configured formatter for its file type.

Parameters:
- path: Absolute path to the file or directory (required)

The formatter used depends on the file extension:
- .py files: Uses configured Python formatter (e.g., black, ruff)
- .js/.ts/.jsx/.tsx files: Uses Prettier or configured JS formatter
- .json files: Uses JSON formatter
- Other files: Uses available language-specific formatters

Use this tool after making edits to ensure consistent code style."""

GET_DIAGNOSTICS_DESCRIPTION = """Get IDE diagnostics (errors, warnings, hints) for files or directories.

Parameters:
- path: Absolute path to a file or directory (required)

Returns diagnostic information including:
- Errors (syntax errors, type errors, etc.)
- Warnings (unused variables, deprecated usage, etc.)
- Hints (suggestions for improvement)

Best practices:
- Run on a directory for efficiency when checking multiple files
- The output is shown directly to the user - don't repeat or summarize it
- Use after making changes to verify the code is valid

Note: Requires IDE integration to function. Returns empty if no IDE is connected."""

MERMAID_DESCRIPTION = """Render a Mermaid diagram for visualization.

Parameters:
- code: Mermaid diagram syntax (required)
- citations: Optional dict mapping node/edge IDs to file:// URIs

Proactively use this tool for visualizing:
- System architecture and component relationships
- Workflow and process flows
- Algorithm flowcharts
- Class hierarchies and relationships
- State machines and transitions
- Sequence diagrams for interactions

Styling guidelines:
- Use dark fills with light strokes and text for good contrast
- Keep node labels concise
- Use consistent shapes for similar element types

Example:
```
graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action 1]
    B -->|No| D[Action 2]
    C --> E[End]
    D --> E
```"""

OPEN_FILE_DESCRIPTION = """Open a file in the user's default application.

Parameters:
- path: Absolute path to the file to open (required)

This opens the file using the system's default application:
- Images (.jpg, .png, etc.) → Default image viewer
- PDFs → Default PDF reader
- Documents → Default document editor
- URLs → Default web browser
- Any file → Associated application

Use this when:
- The user asks to "open" or "show" a file
- You've created/edited a file and want the user to view it
- The user wants to see a generated image, document, or output

Note: This is for the USER to view the file, not for the agent to analyze it.
For agent analysis of images/media, use the look_at tool instead."""


# --- File Edit History State ---


class EditHistoryEntry:
    """Represents a single edit that can be undone."""

    def __init__(self, path: str, old_content: str, new_content: str):
        self.path = path
        self.old_content = old_content
        self.new_content = new_content


# Global edit history (per-session, last edit only)
_last_edit: dict[str, EditHistoryEntry] = {}


def record_edit(path: str, old_content: str, new_content: str) -> None:
    """Record an edit for potential undo. Called by edit_file tool."""
    _last_edit[path] = EditHistoryEntry(path, old_content, new_content)


def get_last_edit(path: str) -> EditHistoryEntry | None:
    """Get the last edit for a file, if any."""
    return _last_edit.get(path)


def clear_edit_history(path: str) -> None:
    """Clear the edit history for a file."""
    _last_edit.pop(path, None)


# --- Tool Generators ---


def _undo_edit_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the undo_edit tool."""
    tool_description = custom_description or UNDO_EDIT_DESCRIPTION

    def _get_backend(runtime: ToolRuntime) -> BackendProtocol:
        if callable(backend):
            return backend(runtime)
        return backend

    async def async_undo_edit(path: str, runtime: ToolRuntime) -> str:
        """Undo the last edit to a file."""
        last_edit = get_last_edit(path)
        if last_edit is None:
            return f"No edit history found for {path}. Cannot undo."

        resolved_backend = _get_backend(runtime)

        # Write the old content back
        try:
            await resolved_backend.awrite(path, last_edit.old_content)
        except Exception as e:
            return f"Failed to undo edit: {e}"

        # Generate diff showing what was undone
        from difflib import unified_diff

        diff_lines = list(
            unified_diff(
                last_edit.new_content.splitlines(keepends=True),
                last_edit.old_content.splitlines(keepends=True),
                fromfile=f"a{path}",
                tofile=f"b{path}",
            )
        )
        diff_output = "".join(diff_lines) if diff_lines else "(No changes)"

        # Clear history after successful undo
        clear_edit_history(path)

        return f"Successfully undone the last edit to {path}.\n\n```diff\n{diff_output}\n```"

    def sync_undo_edit(path: str, runtime: ToolRuntime) -> str:
        """Sync version of undo_edit."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(async_undo_edit(path, runtime))

    return StructuredTool.from_function(
        name="undo_edit",
        description=tool_description,
        func=sync_undo_edit,
        coroutine=async_undo_edit,
    )


def _mermaid_tool_generator(
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the mermaid diagram rendering tool."""
    tool_description = custom_description or MERMAID_DESCRIPTION

    def render_mermaid(
        code: str,
        citations: dict[str, str] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        """Render a Mermaid diagram."""
        # The actual rendering happens on the client/frontend side
        # This tool returns structured data that the UI can render
        result = {
            "type": "mermaid",
            "code": code,
            "citations": citations or {},
        }

        # Return as JSON for the frontend to handle
        return json.dumps(result, indent=2)

    return StructuredTool.from_function(
        name="mermaid",
        description=tool_description,
        func=render_mermaid,
    )


def _format_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the format_file tool."""
    tool_description = custom_description or FORMAT_FILE_DESCRIPTION

    def _get_backend(runtime: ToolRuntime) -> BackendProtocol:
        if callable(backend):
            return backend(runtime)
        return backend

    async def async_format_file(path: str, runtime: ToolRuntime) -> str:
        """Format a file using the appropriate formatter."""
        resolved_backend = _get_backend(runtime)

        # Check if backend supports formatting
        if hasattr(resolved_backend, "aformat_file"):
            try:
                result = await resolved_backend.aformat_file(path)
                if result.get("formatted"):
                    return f"Successfully formatted {path}"
                if result.get("error"):
                    return f"Formatting failed: {result['error']}"
                return f"No formatting changes needed for {path}"
            except Exception as e:
                return f"Failed to format file: {e}"
        else:
            return "File formatting is not supported by the current backend."

    def sync_format_file(path: str, runtime: ToolRuntime) -> str:
        """Sync version of format_file."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(async_format_file(path, runtime))

    return StructuredTool.from_function(
        name="format_file",
        description=tool_description,
        func=sync_format_file,
        coroutine=async_format_file,
    )


def _get_diagnostics_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the get_diagnostics tool."""
    tool_description = custom_description or GET_DIAGNOSTICS_DESCRIPTION

    def _get_backend(runtime: ToolRuntime) -> BackendProtocol:
        if callable(backend):
            return backend(runtime)
        return backend

    async def async_get_diagnostics(path: str, runtime: ToolRuntime) -> str:
        """Get IDE diagnostics for a file or directory."""
        resolved_backend = _get_backend(runtime)

        # Check if backend supports diagnostics
        if hasattr(resolved_backend, "aget_diagnostics"):
            try:
                diagnostics = await resolved_backend.aget_diagnostics(path)
                if not diagnostics:
                    return f"No diagnostics found for {path}"

                # Format diagnostics
                output_lines = [f"Diagnostics for {path}:"]
                for diag in diagnostics:
                    severity = diag.get("severity", "info")
                    message = diag.get("message", "")
                    file_path = diag.get("path", path)
                    line = diag.get("line", 0)
                    col = diag.get("column", 0)

                    output_lines.append(f"  [{severity.upper()}] {file_path}:{line}:{col}: {message}")

                return "\n".join(output_lines)
            except Exception as e:
                return f"Failed to get diagnostics: {e}"
        else:
            return "IDE diagnostics are not supported by the current backend."

    def sync_get_diagnostics(path: str, runtime: ToolRuntime) -> str:
        """Sync version of get_diagnostics."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(async_get_diagnostics(path, runtime))

    return StructuredTool.from_function(
        name="get_diagnostics",
        description=tool_description,
        func=sync_get_diagnostics,
        coroutine=async_get_diagnostics,
    )


def _open_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the open_file tool for opening files in default applications."""
    tool_description = custom_description or OPEN_FILE_DESCRIPTION

    def open_file(path: str) -> str:
        """Open a file in the user's default application.

        Args:
            path: Absolute path to the file to open.

        Returns:
            Success or error message.
        """
        import os
        import platform
        import subprocess
        from pathlib import Path

        file_path = Path(path)

        # Check if file exists
        if not file_path.exists():
            return f"File not found: {path}"

        try:
            system = platform.system()

            if system == "Windows":
                # Windows: use os.startfile
                os.startfile(str(file_path))
            elif system == "Darwin":
                # macOS: use 'open' command
                subprocess.run(["open", str(file_path)], check=True)
            else:
                # Linux/Unix: use 'xdg-open'
                subprocess.run(["xdg-open", str(file_path)], check=True)

            return f"Opened {path} in default application."

        except Exception as e:
            return f"Failed to open {path}: {e}"

    return StructuredTool.from_function(
        name="open_file",
        description=tool_description,
        func=open_file,
    )


# --- Tool Registry ---

UTILITY_TOOL_GENERATORS = {
    "undo_edit": _undo_edit_tool_generator,
    "mermaid": lambda backend, desc: _mermaid_tool_generator(desc),
    "format_file": _format_file_tool_generator,
    "get_diagnostics": _get_diagnostics_tool_generator,
    "open_file": _open_file_tool_generator,
}


def _get_utility_tools(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_tool_descriptions: dict[str, str] | None = None,
    enabled_tools: list[str] | None = None,
) -> list[BaseTool]:
    """Get utility tools.

    Args:
        backend: Backend for file operations.
        custom_tool_descriptions: Optional custom descriptions for tools.
        enabled_tools: Optional list of tool names to enable. If None, all tools are enabled.

    Returns:
        List of configured utility tools.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}

    tools_to_generate = enabled_tools or list(UTILITY_TOOL_GENERATORS.keys())
    tools = []

    for tool_name in tools_to_generate:
        if tool_name in UTILITY_TOOL_GENERATORS:
            tool_generator = UTILITY_TOOL_GENERATORS[tool_name]
            tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
            tools.append(tool)

    return tools


UTILITIES_SYSTEM_PROMPT = """## Utility Tools

You have access to these utility tools:

- undo_edit: Revert the last edit made to a file (shows git-style diff of undone changes)
- format_file: Format a file using the configured formatter for its type
- get_diagnostics: Get IDE errors, warnings, and hints for files/directories
- mermaid: Render Mermaid diagrams for visualization (architecture, workflows, etc.)
- open_file: Open a file in the user's default application (image viewer, PDF reader, etc.)
"""


class UtilitiesMiddleware(AgentMiddleware):
    """Middleware for utility tools: undo_edit, format_file, get_diagnostics, mermaid, open_file.

    This middleware adds utility tools for common development operations.

    Args:
        backend: Backend for file operations. Required for undo_edit and format_file.
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        enabled_tools: Optional list of tool names to enable. Defaults to all tools.

    Example:
        ```python
        from deepagents.middleware.utilities import UtilitiesMiddleware
        from langchain.agents import create_agent

        # Enable all utility tools
        agent = create_agent(middleware=[UtilitiesMiddleware()])

        # Enable only specific tools
        agent = create_agent(middleware=[UtilitiesMiddleware(enabled_tools=["undo_edit", "mermaid"])])
        ```
    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol] | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        enabled_tools: list[str] | None = None,
    ) -> None:
        """Initialize the utilities middleware."""
        from deepagents.backends import StateBackend

        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))
        self._custom_system_prompt = system_prompt
        self.tools = _get_utility_tools(
            self.backend,
            custom_tool_descriptions,
            enabled_tools,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Add utility tools system prompt."""
        system_prompt = self._custom_system_prompt or UTILITIES_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Add utility tools system prompt."""
        system_prompt = self._custom_system_prompt or UTILITIES_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return await handler(request)
