"""Middleware for providing filesystem tools to an agent."""

import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.backends import StateBackend

# Re-export type here for backwards compatibility
from deepagents.backends.protocol import BACKEND_TYPES as BACKEND_TYPES
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.utils import (
    format_content_with_line_numbers,
    format_grep_matches,
    sanitize_tool_call_id,
    truncate_if_too_long,
)

EMPTY_CONTENT_WARNING = "System reminder: File exists but has empty contents"
MAX_LINE_LENGTH = 2000
LINE_NUMBER_WIDTH = 6
DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 500


class FileData(TypedDict):
    """Data structure for storing file contents with metadata."""

    content: list[str]
    """Lines of the file."""

    created_at: str
    """ISO 8601 timestamp of file creation."""

    modified_at: str
    """ISO 8601 timestamp of last modification."""


def _file_data_reducer(left: dict[str, FileData] | None, right: dict[str, FileData | None]) -> dict[str, FileData]:
    """Merge file updates with support for deletions.

    This reducer enables file deletion by treating `None` values in the right
    dictionary as deletion markers. It's designed to work with LangGraph's
    state management where annotated reducers control how state updates merge.

    Args:
        left: Existing files dictionary. May be `None` during initialization.
        right: New files dictionary to merge. Files with `None` values are
            treated as deletion markers and removed from the result.

    Returns:
        Merged dictionary where right overwrites left for matching keys,
        and `None` values in right trigger deletions.

    Example:
        ```python
        existing = {"/file1.txt": FileData(...), "/file2.txt": FileData(...)}
        updates = {"/file2.txt": None, "/file3.txt": FileData(...)}
        result = file_data_reducer(existing, updates)
        # Result: {"/file1.txt": FileData(...), "/file3.txt": FileData(...)}
        ```
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _validate_path(
    path: str,
    *,
    allowed_prefixes: Sequence[str] | None = None,
    allow_native_absolute: bool = False,
) -> str:
    r"""Validate and normalize file path for security.

    Ensures paths are safe to use by preventing directory traversal attacks
    and enforcing consistent formatting. All paths are normalized to use
    forward slashes and start with a leading slash.

    This function is designed for virtual filesystem paths and by default rejects
    Windows absolute paths (e.g., C:/..., F:/...) to maintain consistency
    and prevent path format ambiguity. However, when `allow_native_absolute=True`,
    native absolute paths (including Windows paths) are allowed and returned as-is
    after basic security validation.

    Args:
        path: The path to validate and normalize.
        allowed_prefixes: Optional list of allowed path prefixes. If provided,
            the normalized path must start with one of these prefixes.
        allow_native_absolute: If True, allow native absolute paths (including
            Windows paths like C:\...) and return them normalized but not
            converted to virtual paths. This is useful for local filesystem
            backends that can handle native paths directly.

    Returns:
        Normalized canonical path. For virtual paths, starts with `/` and uses
        forward slashes. For native absolute paths (when allowed), returns the
        normalized native path.

    Raises:
        ValueError: If path contains traversal sequences (`..` or `~`), is a
            Windows absolute path when not allowed, or does not start with an
            allowed prefix when `allowed_prefixes` is specified.

    Example:
        ```python
        _validate_path("foo/bar")  # Returns: "/foo/bar"
        _validate_path("/./foo//bar")  # Returns: "/foo/bar"
        _validate_path("../etc/passwd")  # Raises ValueError
        _validate_path(r"C:\\Users\\file.txt")  # Raises ValueError
        _validate_path(r"C:\\Users\\file.txt", allow_native_absolute=True)  # Returns: "C:/Users/file.txt"
        _validate_path("/data/file.txt", allowed_prefixes=["/data/"])  # OK
        _validate_path("/etc/file.txt", allowed_prefixes=["/data/"])  # Raises ValueError
        ```
    """
    if ".." in path or path.startswith("~"):
        msg = f"Path traversal not allowed: {path}"
        raise ValueError(msg)

    # Check for Windows absolute paths (e.g., C:\..., D:/...)
    is_windows_absolute = bool(re.match(r"^[a-zA-Z]:", path))

    if is_windows_absolute:
        if not allow_native_absolute:
            msg = f"Windows absolute paths are not supported: {path}. Please use virtual paths starting with / (e.g., /workspace/file.txt)"
            raise ValueError(msg)
        # For native absolute paths, normalize but preserve the drive letter
        normalized = os.path.normpath(path)
        normalized = normalized.replace("\\", "/")
        # Skip prefix checks for native absolute paths
        return normalized

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None and not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
        msg = f"Path must start with one of {allowed_prefixes}: {path}"
        raise ValueError(msg)

    return normalized


class FilesystemState(AgentState):
    """State for the filesystem middleware."""

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """Files in the filesystem."""


LIST_FILES_TOOL_DESCRIPTION = """Lists all files in the filesystem, filtering by directory.

Usage:
- The path parameter must be an absolute path, not a relative path
- The list_files tool will return a list of all files in the specified directory.
- This is very useful for exploring the file system and finding the right file to read or edit.
- You should almost ALWAYS use this tool before using the Read or Edit tools."""

READ_FILE_TOOL_DESCRIPTION = """Read a file or list a directory from the filesystem.

Capabilities:
- Read text files with line numbers (cat -n format)
- List directory contents with trailing "/" for subdirectories
- Read images (PNG, JPEG, GIF) and present them visually to the model
- If the file or directory doesn't exist, an error is returned

Parameters:
- file_path: MUST be an absolute path (starting with "/")
- offset: Line number to start reading from (default: 0)
- limit: Maximum number of lines to read (default: 500)

Directory listing:
- If path is a directory, returns entries one per line
- Subdirectories have trailing "/" (e.g., "subdir/")
- No line numbers for directory listings

Usage:
- By default, reads up to 500 lines starting from the beginning
- **IMPORTANT for large files**: Use pagination with offset and limit to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit when necessary for editing
- Any lines longer than 2000 characters will be truncated
- Results are returned with line numbers starting at 1 (e.g., "1: abc\\n2: def")

Parallel reading:
- Call this tool in parallel for all files you need to read
- It is better to speculatively read multiple files as a batch

Best practices:
- Use the glob tool to look up filenames by pattern if unsure of the correct path
- ALWAYS read a file before editing it
- If a file exists but is empty, you'll receive a system reminder warning"""

EDIT_FILE_TOOL_DESCRIPTION = """Performs exact string replacements in files. Returns a git-style diff showing changes made.

Returns:
- A git-style diff showing the changes made as formatted markdown
- The line range ([startLine, endLine]) of the changed content
- The diff is also shown to the user

Usage:
- The file specified by `path` MUST exist, and MUST be an absolute path
- You must use your `read_file` tool at least once before editing. This tool will error if you attempt an edit without reading the file first
- `old_string` MUST exist in the file and MUST be different from `new_string`
- When editing text from read_file output, preserve exact indentation (tabs/spaces) as it appears AFTER the line number prefix
- The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match
- Never include any part of the line number prefix in old_string or new_string

Uniqueness requirements:
- The edit will FAIL if `old_string` is not unique in the file
- To make the string unique, add more surrounding context (additional lines)
- Set `replace_all=true` to replace ALL occurrences of `old_string` in the file
- Use `replace_all` for renaming variables or patterns across the file

Best practices:
- ALWAYS prefer editing existing files over creating new ones
- If you need to replace the ENTIRE contents of a file, use `write_file` instead (more efficient)
- Only use emojis if the user explicitly requests it"""


WRITE_FILE_TOOL_DESCRIPTION = """Create or overwrite a file in the filesystem.

Use this tool when:
- Creating a new file from scratch with given content
- Replacing the entire contents of an existing file
- Prefer this over `edit_file` when you want to overwrite the entire file

Parameters:
- file_path: MUST be an absolute path (starting with "/")
- content: The full file contents as a string

Usage:
- Creates a new file if it doesn't exist
- Overwrites existing file completely if it already exists
- For partial modifications, use `edit_file` instead (more efficient, shows diffs)
- Prefer editing existing files over creating new ones when possible

Best practices:
- Use `edit_file` for small changes to preserve file history context
- Use `write_file` for complete file replacement or new file creation"""


GLOB_TOOL_DESCRIPTION = """Fast file pattern matching tool that works with any codebase size.

Returns matching file paths sorted by most recent modification time first.

File pattern syntax:
- `*` - Matches any characters within a path segment
- `**` - Matches any directories (recursive)
- `?` - Matches a single character
- `{js,ts}` - Matches either `js` or `ts` (alternative patterns)
- `[a-z]` - Matches any lowercase letter (character classes)
- `[0-9]` - Matches any digit

Parameters:
- pattern: The glob pattern to match
- path: Starting directory (default: "/")
- limit: Maximum number of results to return (optional)

Examples:
- `**/*.js` - All JavaScript files in any directory
- `src/**/*.ts` - All TypeScript files under src directory (searches only in src)
- `*.json` - All JSON files in the current directory
- `**/*test*` - All files with "test" in their name
- `web/src/**/*` - All files under the web/src directory
- `**/*.{js,ts}` - All JavaScript AND TypeScript files
- `src/[a-z]*/*.ts` - TypeScript files in src subdirectories starting with lowercase letters"""

GREP_TOOL_DESCRIPTION = """Search for exact text patterns in files using fast keyword search (ripgrep-based).

When to use this tool:
- Finding exact text matches (variable names, function calls, specific strings)
- For semantic/conceptual searches, consider using specialized search tools

Parameters:
- pattern: The text to search for (literal string by default)
- path: Directory to search in (default is workspace root)
- glob: Glob pattern to filter files (e.g., `*.py`, `**/*.test.ts`)
- output_mode: Controls the output format:
  - `files_with_matches`: List only file paths containing matches (default)
  - `content`: Show matching lines with file path and line numbers
  - `count`: Show count of matches per file
- caseSensitive: Set to true for case-sensitive matching (default: false)
- literal: Set to true for exact literal text search (default: true)

Strategy:
- Use 'path' or 'glob' to narrow searches; run multiple focused calls rather than one broad search
- Uses Rust-style regex when literal=false (escape `{` and `}` with `\\`)

Constraints:
- Results are limited to 100 matches (up to 10 per file)
- Lines are truncated at 200 characters

Examples:
- Find specific function: `grep(pattern="registerTool", path="core/src")`
- Search interface definitions: `grep(pattern="interface ToolDefinition", path="core/src/tools")`
- Case-sensitive search: `grep(pattern="ERROR:", caseSensitive=true)`
- Find TODOs in frontend: `grep(pattern="TODO:", path="web/src")`
- Search test files: `grep(pattern="restoreThreads", glob="**/*.test.ts")`
- Find REST endpoints (regex): `grep(pattern="app\\\\.(get|post|put|delete)\\\\(", path="server", literal=false)`"""

EXECUTE_TOOL_DESCRIPTION = """Executes a given shell command in the sandbox environment.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use the ls tool to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use ls to check that "foo" exists

2. Command Execution:
   - ALWAYS quote file paths that contain spaces: `cat "path with spaces/file.txt"`
   - Examples of proper quoting:
     - cd "/Users/name/My Documents" (correct)
     - cd /Users/name/My Documents (incorrect - will fail)
     - python "/path/with spaces/script.py" (correct)

Usage notes:
  - The command parameter is required
  - Commands run in the workspace root by default
  - Returns combined stdout/stderr output with exit code
  - Output is truncated to the last 50000 characters; rerun with grep or head/tail filter if needed
  - On Windows, use PowerShell commands and `\\` path separators

IMPORTANT - Use specialized tools instead of shell commands:
  - Use `glob` instead of `find` for file pattern matching
  - Use `grep` tool instead of shell `grep` for text search
  - Use `read_file` instead of `cat`, `head`, `tail` for reading files
  - Use `edit_file` instead of `sed` for file modifications

Command chaining:
  - Do NOT chain commands with `;` or `&&` - make separate tool calls instead
  - Do NOT use `&` for background processes
  - Environment variables and `cd` do not persist between commands
  - Never use `cd dir && cmd`; use absolute paths instead

Git restrictions:
  - Only run `git commit` and `git push` if explicitly instructed by the user

Examples:
  Good examples:
    - execute(command="pytest /foo/bar/tests")
    - execute(command="python /path/to/script.py")
    - execute(command="npm test")

  Bad examples (avoid these):
    - execute(command="cd /foo/bar && pytest tests")  # Use absolute path instead
    - execute(command="cat file.txt")  # Use read_file tool instead
    - execute(command="find . -name '*.py'")  # Use glob tool instead
    - execute(command="grep -r 'pattern' .")  # Use grep tool instead

Note: This tool is only available if the backend supports execution (SandboxBackendProtocol).
If execution is not supported, the tool will return an error message."""

MOVE_FILE_TOOL_DESCRIPTION = """Move or rename a file or directory.

Use this tool to:
- Move a file to a different directory
- Rename a file or directory
- Move a directory and all its contents

Args:
    source: Absolute path of the file/directory to move (must start with /)
    destination: Absolute path of the new location (must start with /)

Returns:
    Success message with source and destination paths, or error message.

Examples:
    - Move file: move_file(source="/Downloads/image.png", destination="/Desktop/image.png")
    - Rename file: move_file(source="/docs/old_name.txt", destination="/docs/new_name.txt")
    - Move to directory: move_file(source="/file.txt", destination="/backup/")

Note: If destination is a directory, the file will be moved into that directory with the same name."""

COPY_FILE_TOOL_DESCRIPTION = """Copy a file or directory.

Use this tool to:
- Create a copy of a file
- Duplicate a directory and all its contents

Args:
    source: Absolute path of the file/directory to copy (must start with /)
    destination: Absolute path of the copy (must start with /)

Returns:
    Success message with source and destination paths, or error message.

Examples:
    - Copy file: copy_file(source="/config.json", destination="/config_backup.json")
    - Copy directory: copy_file(source="/src", destination="/src_backup")

Note: If destination is a directory, the file will be copied into that directory with the same name."""

DELETE_FILE_TOOL_DESCRIPTION = """Delete a file or directory.

Use this tool to:
- Remove a file
- Remove a directory and all its contents (use with caution!)

Args:
    path: Absolute path of the file/directory to delete (must start with /)

Returns:
    Success message with deleted path, or error message.

CAUTION: This action is permanent and cannot be undone!

Examples:
    - Delete file: delete_file(path="/temp/old_file.txt")
    - Delete directory: delete_file(path="/old_project")"""

MKDIR_TOOL_DESCRIPTION = """Create a new directory.

Use this tool to:
- Create a new directory
- Create nested directories (parent directories are created automatically)

Args:
    path: Absolute path of the directory to create (must start with /)
    parents: If True (default), create parent directories as needed

Returns:
    Success message with created path, or error message.

Examples:
    - Create directory: mkdir(path="/projects/new_project")
    - Create nested: mkdir(path="/a/b/c/d")  # Creates all parent directories"""

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `move_file`, `copy_file`, `delete_file`, `mkdir`

You have access to a filesystem which you can interact with using these tools.
All file paths must start with a /.

- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a file in the filesystem
- edit_file: edit a file in the filesystem
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files
- move_file: move or rename a file/directory
- copy_file: copy a file/directory
- delete_file: delete a file/directory
- mkdir: create a new directory"""

EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

You have access to an `execute` tool for running shell commands in a sandboxed environment.
Use this tool to run commands, scripts, tests, builds, and other shell operations.

- execute: run a shell command in the sandbox (returns output and exit code)"""


def _get_backend(backend: BACKEND_TYPES, runtime: ToolRuntime) -> BackendProtocol:
    """Get the resolved backend instance from backend or factory.

    Args:
        backend: Backend instance or factory function.
        runtime: The tool runtime context.

    Returns:
        Resolved backend instance.
    """
    if callable(backend):
        return backend(runtime)
    return backend


def _supports_native_paths(resolved_backend: BackendProtocol) -> bool:
    """Check if the backend supports native absolute paths (like Windows paths).

    FilesystemBackend with virtual_mode=False supports native paths.
    CompositeBackend inherits this from its default backend.
    Other backends (StateBackend, SandboxBackend, StoreBackend) use virtual paths.

    Args:
        resolved_backend: The resolved backend instance to check.

    Returns:
        True if native absolute paths are supported, False otherwise.
    """
    # Import here to avoid circular imports
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.filesystem import FilesystemBackend

    # Check if it's a FilesystemBackend with virtual_mode=False
    if isinstance(resolved_backend, FilesystemBackend):
        return not resolved_backend.virtual_mode

    # Check if it's a CompositeBackend - check its default backend
    if isinstance(resolved_backend, CompositeBackend):
        return _supports_native_paths(resolved_backend.default)

    return False


def _ls_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the ls (list files) tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured ls tool that lists files using the backend.
    """
    tool_description = custom_description or LIST_FILES_TOOL_DESCRIPTION

    def sync_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """Synchronous wrapper for ls tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        validated_path = _validate_path(path, allow_native_absolute=allow_native)
        infos = resolved_backend.ls_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_ls(runtime: ToolRuntime[None, FilesystemState], path: str) -> str:
        """Asynchronous wrapper for ls tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        validated_path = _validate_path(path, allow_native_absolute=allow_native)
        infos = await resolved_backend.als_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="ls",
        description=tool_description,
        func=sync_ls,
        coroutine=async_ls,
    )


def _read_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the read_file tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured read_file tool that reads files using the backend.
    """
    tool_description = custom_description or READ_FILE_TOOL_DESCRIPTION

    def sync_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """Synchronous wrapper for read_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        file_path = _validate_path(file_path, allow_native_absolute=allow_native)
        return resolved_backend.read(file_path, offset=offset, limit=limit)

    async def async_read_file(
        file_path: str,
        runtime: ToolRuntime[None, FilesystemState],
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
    ) -> str:
        """Asynchronous wrapper for read_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        file_path = _validate_path(file_path, allow_native_absolute=allow_native)
        return await resolved_backend.aread(file_path, offset=offset, limit=limit)

    return StructuredTool.from_function(
        name="read_file",
        description=tool_description,
        func=sync_read_file,
        coroutine=async_read_file,
    )


def _write_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the write_file tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured write_file tool that creates new files using the backend.
    """
    tool_description = custom_description or WRITE_FILE_TOOL_DESCRIPTION

    def sync_write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        """Synchronous wrapper for write_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        file_path = _validate_path(file_path, allow_native_absolute=allow_native)
        res: WriteResult = resolved_backend.write(file_path, content)
        if res.error:
            return res.error
        # If backend returns state update, wrap into Command with ToolMessage
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Updated file {res.path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Updated file {res.path}"

    async def async_write_file(
        file_path: str,
        content: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> Command | str:
        """Asynchronous wrapper for write_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        file_path = _validate_path(file_path, allow_native_absolute=allow_native)
        res: WriteResult = await resolved_backend.awrite(file_path, content)
        if res.error:
            return res.error
        # If backend returns state update, wrap into Command with ToolMessage
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Updated file {res.path}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Updated file {res.path}"

    return StructuredTool.from_function(
        name="write_file",
        description=tool_description,
        func=sync_write_file,
        coroutine=async_write_file,
    )


def _edit_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the edit_file tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured edit_file tool that performs string replacements in files using the backend.
    """
    tool_description = custom_description or EDIT_FILE_TOOL_DESCRIPTION

    def sync_edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        """Synchronous wrapper for edit_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        file_path = _validate_path(file_path, allow_native_absolute=allow_native)
        res: EditResult = resolved_backend.edit(file_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    async def async_edit_file(
        file_path: str,
        old_string: str,
        new_string: str,
        runtime: ToolRuntime[None, FilesystemState],
        *,
        replace_all: bool = False,
    ) -> Command | str:
        """Asynchronous wrapper for edit_file tool."""
        resolved_backend = _get_backend(backend, runtime)
        allow_native = _supports_native_paths(resolved_backend)
        file_path = _validate_path(file_path, allow_native_absolute=allow_native)
        res: EditResult = await resolved_backend.aedit(file_path, old_string, new_string, replace_all=replace_all)
        if res.error:
            return res.error
        if res.files_update is not None:
            return Command(
                update={
                    "files": res.files_update,
                    "messages": [
                        ToolMessage(
                            content=f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ],
                }
            )
        return f"Successfully replaced {res.occurrences} instance(s) of the string in '{res.path}'"

    return StructuredTool.from_function(
        name="edit_file",
        description=tool_description,
        func=sync_edit_file,
        coroutine=async_edit_file,
    )


def _glob_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the glob tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured glob tool that finds files by pattern using the backend.
    """
    tool_description = custom_description or GLOB_TOOL_DESCRIPTION

    def sync_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """Synchronous wrapper for glob tool."""
        resolved_backend = _get_backend(backend, runtime)
        infos = resolved_backend.glob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    async def async_glob(pattern: str, runtime: ToolRuntime[None, FilesystemState], path: str = "/") -> str:
        """Asynchronous wrapper for glob tool."""
        resolved_backend = _get_backend(backend, runtime)
        infos = await resolved_backend.aglob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result)

    return StructuredTool.from_function(
        name="glob",
        description=tool_description,
        func=sync_glob,
        coroutine=async_glob,
    )


def _grep_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the grep tool.

    Args:
        backend: Backend to use for file storage, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured grep tool that searches for patterns in files using the backend.
    """
    tool_description = custom_description or GREP_TOOL_DESCRIPTION

    def sync_grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """Synchronous wrapper for grep tool."""
        resolved_backend = _get_backend(backend, runtime)
        raw = resolved_backend.grep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    async def async_grep(
        pattern: str,
        runtime: ToolRuntime[None, FilesystemState],
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
    ) -> str:
        """Asynchronous wrapper for grep tool."""
        resolved_backend = _get_backend(backend, runtime)
        raw = await resolved_backend.agrep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw
        formatted = format_grep_matches(raw, output_mode)
        return truncate_if_too_long(formatted)  # type: ignore[arg-type]

    return StructuredTool.from_function(
        name="grep",
        description=tool_description,
        func=sync_grep,
        coroutine=async_grep,
    )


def _supports_execution(backend: BackendProtocol) -> bool:
    """Check if a backend supports command execution.

    For CompositeBackend, checks if the default backend supports execution.
    For other backends, checks if they implement SandboxBackendProtocol.

    Args:
        backend: The backend to check.

    Returns:
        True if the backend supports execution, False otherwise.
    """
    # Import here to avoid circular dependency
    from deepagents.backends.composite import CompositeBackend

    # For CompositeBackend, check the default backend
    if isinstance(backend, CompositeBackend):
        return isinstance(backend.default, SandboxBackendProtocol)

    # For other backends, use isinstance check
    return isinstance(backend, SandboxBackendProtocol)


def _execute_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the execute tool for sandbox command execution.

    Args:
        backend: Backend to use for execution, or a factory function that takes runtime and returns a backend.
        custom_description: Optional custom description for the tool.

    Returns:
        Configured execute tool that runs commands if backend supports SandboxBackendProtocol.
    """
    tool_description = custom_description or EXECUTE_TOOL_DESCRIPTION

    def sync_execute(
        command: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Synchronous wrapper for execute tool."""
        resolved_backend = _get_backend(backend, runtime)

        # Runtime check - fail gracefully if not supported
        if not _supports_execution(resolved_backend):
            return (
                "Error: Execution not available. This agent's backend "
                "does not support command execution (SandboxBackendProtocol). "
                "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
            )

        try:
            result = resolved_backend.execute(command)
        except NotImplementedError as e:
            # Handle case where execute() exists but raises NotImplementedError
            return f"Error: Execution not available. {e}"

        # Format output for LLM consumption
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)

    async def async_execute(
        command: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Asynchronous wrapper for execute tool."""
        resolved_backend = _get_backend(backend, runtime)

        # Runtime check - fail gracefully if not supported
        if not _supports_execution(resolved_backend):
            return (
                "Error: Execution not available. This agent's backend "
                "does not support command execution (SandboxBackendProtocol). "
                "To use the execute tool, provide a backend that implements SandboxBackendProtocol."
            )

        try:
            result = await resolved_backend.aexecute(command)
        except NotImplementedError as e:
            # Handle case where execute() exists but raises NotImplementedError
            return f"Error: Execution not available. {e}"

        # Format output for LLM consumption
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)

    return StructuredTool.from_function(
        name="execute",
        description=tool_description,
        func=sync_execute,
        coroutine=async_execute,
    )


def _move_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the move_file tool for moving/renaming files and directories."""
    tool_description = custom_description or MOVE_FILE_TOOL_DESCRIPTION

    def sync_move(
        source: str,
        destination: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Synchronous wrapper for move tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = resolved_backend.move(source, destination)
        if result.error:
            return f"Error: {result.error}"
        return f"Moved {result.source} to {result.destination}"

    async def async_move(
        source: str,
        destination: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Asynchronous wrapper for move tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = await resolved_backend.amove(source, destination)
        if result.error:
            return f"Error: {result.error}"
        return f"Moved {result.source} to {result.destination}"

    return StructuredTool.from_function(
        name="move_file",
        description=tool_description,
        func=sync_move,
        coroutine=async_move,
    )


def _copy_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the copy_file tool for copying files and directories."""
    tool_description = custom_description or COPY_FILE_TOOL_DESCRIPTION

    def sync_copy(
        source: str,
        destination: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Synchronous wrapper for copy tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = resolved_backend.copy(source, destination)
        if result.error:
            return f"Error: {result.error}"
        return f"Copied {result.source} to {result.destination}"

    async def async_copy(
        source: str,
        destination: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Asynchronous wrapper for copy tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = await resolved_backend.acopy(source, destination)
        if result.error:
            return f"Error: {result.error}"
        return f"Copied {result.source} to {result.destination}"

    return StructuredTool.from_function(
        name="copy_file",
        description=tool_description,
        func=sync_copy,
        coroutine=async_copy,
    )


def _delete_file_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the delete_file tool for deleting files and directories."""
    tool_description = custom_description or DELETE_FILE_TOOL_DESCRIPTION

    def sync_delete(
        path: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Synchronous wrapper for delete tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = resolved_backend.delete(path)
        if result.error:
            return f"Error: {result.error}"
        return f"Deleted {result.path}"

    async def async_delete(
        path: str,
        runtime: ToolRuntime[None, FilesystemState],
    ) -> str:
        """Asynchronous wrapper for delete tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = await resolved_backend.adelete(path)
        if result.error:
            return f"Error: {result.error}"
        return f"Deleted {result.path}"

    return StructuredTool.from_function(
        name="delete_file",
        description=tool_description,
        func=sync_delete,
        coroutine=async_delete,
    )


def _mkdir_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
) -> BaseTool:
    """Generate the mkdir tool for creating directories."""
    tool_description = custom_description or MKDIR_TOOL_DESCRIPTION

    def sync_mkdir(
        path: str,
        runtime: ToolRuntime[None, FilesystemState],
        parents: bool = True,
    ) -> str:
        """Synchronous wrapper for mkdir tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = resolved_backend.mkdir(path, parents=parents)
        if result.error:
            return f"Error: {result.error}"
        return f"Created directory {result.path}"

    async def async_mkdir(
        path: str,
        runtime: ToolRuntime[None, FilesystemState],
        parents: bool = True,
    ) -> str:
        """Asynchronous wrapper for mkdir tool."""
        resolved_backend = _get_backend(backend, runtime)
        result = await resolved_backend.amkdir(path, parents=parents)
        if result.error:
            return f"Error: {result.error}"
        return f"Created directory {result.path}"

    return StructuredTool.from_function(
        name="mkdir",
        description=tool_description,
        func=sync_mkdir,
        coroutine=async_mkdir,
    )


# Type alias for vision model callable
# Takes (image_bytes, query, context, file_type) -> analysis_text
VisionModelCallable = Callable[[bytes, str, str | None, str], str]
AsyncVisionModelCallable = Callable[[bytes, str, str | None, str], Awaitable[str]]


def _get_file_type_from_extension(path: str) -> str:
    """Get MIME type from file extension."""
    import mimetypes

    mime_type, _ = mimetypes.guess_type(path)
    return mime_type or "application/octet-stream"


TOOL_GENERATORS = {
    "ls": _ls_tool_generator,
    "read_file": _read_file_tool_generator,
    "write_file": _write_file_tool_generator,
    "edit_file": _edit_file_tool_generator,
    "glob": _glob_tool_generator,
    "grep": _grep_tool_generator,
    "execute": _execute_tool_generator,
    "move_file": _move_file_tool_generator,
    "copy_file": _copy_file_tool_generator,
    "delete_file": _delete_file_tool_generator,
    "mkdir": _mkdir_tool_generator,
}


def _get_filesystem_tools(
    backend: BackendProtocol,
    custom_tool_descriptions: dict[str, str] | None = None,
) -> list[BaseTool]:
    """Get filesystem and execution tools.

    Args:
        backend: Backend to use for file storage and optional execution, or a factory function that takes runtime and returns a backend.
        custom_tool_descriptions: Optional custom descriptions for tools.

    Returns:
        List of configured tools: ls, read_file, write_file, edit_file, glob, grep, execute, move_file, copy_file, delete_file, mkdir.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}
    tools = []

    for tool_name, tool_generator in TOOL_GENERATORS.items():
        tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
        tools.append(tool)
    return tools


TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}
You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.
You can do this by specifying an offset and limit in the read_file tool call.
For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here are the first 10 lines of the result:
{content_sample}
"""


class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem and optional execution tools to an agent.

    This middleware adds filesystem tools to the agent: ls, read_file, write_file,
    edit_file, glob, and grep. Files can be stored using any backend that implements
    the BackendProtocol.

    If the backend implements SandboxBackendProtocol, an execute tool is also added
    for running shell commands.

    Args:
        backend: Backend for file storage and optional execution. If not provided, defaults to StateBackend
            (ephemeral storage in agent state). For persistent storage or hybrid setups,
            use CompositeBackend with custom routes. For execution support, use a backend
            that implements SandboxBackendProtocol.
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.

    Note:
        For AI-powered file analysis (images, PDFs), use the `look_at` tool from AdvancedMiddleware instead.

    Example:
        ```python
        from deepagents.middleware.filesystem import FilesystemMiddleware
        from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
        from langchain.agents import create_agent

        # Ephemeral storage only (default, no execution)
        agent = create_agent(middleware=[FilesystemMiddleware()])

        # With hybrid storage (ephemeral + persistent /memories/)
        backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend()})
        agent = create_agent(middleware=[FilesystemMiddleware(backend=backend)])

        # With sandbox backend (supports execution)
        from my_sandbox import DockerSandboxBackend

        sandbox = DockerSandboxBackend(container_id="my-container")
        agent = create_agent(middleware=[FilesystemMiddleware(backend=sandbox)])
        ```
    """

    state_schema = FilesystemState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        tool_token_limit_before_evict: int | None = 20000,
    ) -> None:
        """Initialize the filesystem middleware.

        Args:
            backend: Backend for file storage and optional execution, or a factory callable.
                Defaults to StateBackend if not provided.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions override.
            tool_token_limit_before_evict: Optional token limit before evicting a tool result to the filesystem.
        """
        self.tool_token_limit_before_evict = tool_token_limit_before_evict

        # Use provided backend or default to StateBackend factory
        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))

        # Set system prompt (allow full override or None to generate dynamically)
        self._custom_system_prompt = system_prompt

        self.tools = _get_filesystem_tools(self.backend, custom_tool_descriptions)

    def _get_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Get the resolved backend instance from backend or factory.

        Args:
            runtime: The tool runtime context.

        Returns:
            Resolved backend instance.
        """
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            request = request.override(system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system prompt and filter tools based on backend capabilities.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        # Check if execute tool is present and if backend supports it
        has_execute_tool = any((tool.name if hasattr(tool, "name") else tool.get("name")) == "execute" for tool in request.tools)

        backend_supports_execution = False
        if has_execute_tool:
            # Resolve backend to check execution support
            backend = self._get_backend(request.runtime)
            backend_supports_execution = _supports_execution(backend)

            # If execute tool exists but backend doesn't support it, filter it out
            if not backend_supports_execution:
                filtered_tools = [tool for tool in request.tools if (tool.name if hasattr(tool, "name") else tool.get("name")) != "execute"]
                request = request.override(tools=filtered_tools)
                has_execute_tool = False

        # Use custom system prompt if provided, otherwise generate dynamically
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        else:
            # Build dynamic system prompt based on available tools
            prompt_parts = [FILESYSTEM_SYSTEM_PROMPT]

            # Add execution instructions if execute tool is available
            if has_execute_tool and backend_supports_execution:
                prompt_parts.append(EXECUTION_SYSTEM_PROMPT)

            system_prompt = "\n\n".join(prompt_parts)

        if system_prompt:
            request = request.override(system_prompt=request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt)

        return await handler(request)

    def _process_large_message(
        self,
        message: ToolMessage,
        resolved_backend: BackendProtocol,
    ) -> tuple[ToolMessage, dict[str, FileData] | None]:
        """Process a large ToolMessage by evicting its content to filesystem.

        Args:
            message: The ToolMessage with large content to evict.
            resolved_backend: The filesystem backend to write the content to.

        Returns:
            A tuple of (processed_message, files_update):
            - processed_message: New ToolMessage with truncated content and file reference
            - files_update: Dict of file updates to apply to state, or None if eviction failed

        Note:
            The entire content is converted to string, written to /large_tool_results/{tool_call_id},
            and replaced with a truncated preview plus file reference. The replacement is always
            returned as a plain string for consistency, regardless of original content type.

            ToolMessage supports multimodal content blocks (images, audio, etc.), but these are
            uncommon in tool results. For simplicity, all content is stringified and evicted.
            The model can recover by reading the offloaded file from the backend.
        """
        # Early exit if eviction not configured
        if not self.tool_token_limit_before_evict:
            return message, None

        # Convert content to string once for both size check and eviction
        # Special case: single text block - extract text directly for readability
        if (
            isinstance(message.content, list)
            and len(message.content) == 1
            and isinstance(message.content[0], dict)
            and message.content[0].get("type") == "text"
            and "text" in message.content[0]
        ):
            content_str = str(message.content[0]["text"])
        elif isinstance(message.content, str):
            content_str = message.content
        else:
            # Multiple blocks or non-text content - stringify entire structure
            content_str = str(message.content)

        # Check if content exceeds eviction threshold
        # Using 4 chars per token as a conservative approximation (actual ratio varies by content)
        # This errs on the high side to avoid premature eviction of content that might fit
        if len(content_str) <= 4 * self.tool_token_limit_before_evict:
            return message, None

        # Write content to filesystem
        sanitized_id = sanitize_tool_call_id(message.tool_call_id)
        file_path = f"/large_tool_results/{sanitized_id}"
        result = resolved_backend.write(file_path, content_str)
        if result.error:
            return message, None

        # Create truncated preview for the replacement message
        content_sample = format_content_with_line_numbers([line[:1000] for line in content_str.splitlines()[:10]], start_line=1)
        replacement_text = TOO_LARGE_TOOL_MSG.format(
            tool_call_id=message.tool_call_id,
            file_path=file_path,
            content_sample=content_sample,
        )

        # Always return as plain string after eviction
        processed_message = ToolMessage(
            content=replacement_text,
            tool_call_id=message.tool_call_id,
        )
        return processed_message, result.files_update

    def _intercept_large_tool_result(self, tool_result: ToolMessage | Command, runtime: ToolRuntime) -> ToolMessage | Command:
        """Intercept and process large tool results before they're added to state.

        Args:
            tool_result: The tool result to potentially evict (ToolMessage or Command).
            runtime: The tool runtime providing access to the filesystem backend.

        Returns:
            Either the original result (if small enough) or a Command with evicted
            content written to filesystem and truncated message.

        Note:
            Handles both single ToolMessage results and Command objects containing
            multiple messages. Large content is automatically offloaded to filesystem
            to prevent context window overflow.
        """
        if isinstance(tool_result, ToolMessage):
            resolved_backend = self._get_backend(runtime)
            processed_message, files_update = self._process_large_message(
                tool_result,
                resolved_backend,
            )
            return (
                Command(
                    update={
                        "files": files_update,
                        "messages": [processed_message],
                    }
                )
                if files_update is not None
                else processed_message
            )

        if isinstance(tool_result, Command):
            update = tool_result.update
            if update is None:
                return tool_result
            command_messages = update.get("messages", [])
            accumulated_file_updates = dict(update.get("files", {}))
            resolved_backend = self._get_backend(runtime)
            processed_messages = []
            for message in command_messages:
                if not isinstance(message, ToolMessage):
                    processed_messages.append(message)
                    continue

                processed_message, files_update = self._process_large_message(
                    message,
                    resolved_backend,
                )
                processed_messages.append(processed_message)
                if files_update is not None:
                    accumulated_file_updates.update(files_update)
            return Command(update={**update, "messages": processed_messages, "files": accumulated_file_updates})
        raise AssertionError(f"Unreachable code reached in _intercept_large_tool_result: for tool_result of type {type(tool_result)}")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return handler(request)

        tool_result = handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(async)Check the size of the tool call result and evict to filesystem if too large.

        Args:
            request: The tool call request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The raw ToolMessage, or a pseudo tool message with the ToolResult in state.
        """
        if self.tool_token_limit_before_evict is None or request.tool_call["name"] in TOOL_GENERATORS:
            return await handler(request)

        tool_result = await handler(request)
        return self._intercept_large_tool_result(tool_result, request.runtime)
