"""Middleware for advanced AI-powered tools: finder, look_at, and specialized subagents (librarian, oracle)."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)

# --- Tool Descriptions ---

FINDER_DESCRIPTION = """Semantic codebase search - find code by functionality, not just text.

Unlike grep (exact text match), finder searches for code based on concepts and behavior.

Parameters:
- query: Precise engineering request describing what you're looking for (required)
  Include success criteria for best results.

When to use:
- Finding code that implements specific behavior or functionality
- Locating related code across different areas of the codebase
- Searching for patterns when you don't know the exact text
- Correlating code across multiple files

Examples:
- "Find the authentication logic that validates JWT tokens"
- "Locate all database connection pooling implementations"
- "Find where user permissions are checked before API access"
- "Search for rate limiting implementations across services"

Returns matching code locations with context about why they match.

Note: For exact text matches, use grep instead (faster for known strings)."""

LOOK_AT_DESCRIPTION = """Extract information from images, PDFs, and media files.

Use when the read_file tool can't interpret a file's visual content.

Parameters:
- path: Absolute path to the file (required)
- objective: What specific information to extract (required)
- context: Additional context to help interpretation (optional)
- referenceFiles: Paths to files for comparison (optional)

Supported formats:
- Images: PNG, JPEG, GIF, WebP, SVG
- Documents: PDF
- Other media files with visual content

When to use:
- Extracting text from images (OCR)
- Understanding diagrams, charts, or UI screenshots
- Reading scanned documents
- Analyzing image content for specific information

Returns extracted/analyzed information (not raw file bytes)."""

LIBRARIAN_DESCRIPTION = """Multi-repository codebase expert for deep understanding.

A specialized AI agent for understanding complex, multi-repository codebases.

Parameters:
- question: Detailed question about the codebase (required)
- repos: List of repository paths or URLs to analyze (optional, uses current repo by default)

Capabilities:
- Architecture understanding and component relationships
- Cross-repository dependency analysis
- Code evolution and history analysis
- Commit history and change patterns
- Documentation generation

When to use:
- Understanding unfamiliar codebases
- Analyzing how components interact across repos
- Investigating why code evolved a certain way
- Generating documentation for complex systems

Returns detailed, documentation-quality responses."""

ORACLE_DESCRIPTION = """AI reasoning advisor for complex technical decisions.

A specialized AI agent powered by advanced reasoning for expert guidance.

Parameters:
- task: Description of what you need help with (required)
- files: List of file paths relevant to the task (optional)
- context: Additional context about the situation (optional)

Capabilities:
- Code reviews and quality analysis
- Architecture feedback and design suggestions
- Bug finding and debugging assistance
- Complex planning and decision-making
- Performance optimization recommendations

When to use:
- Getting a second opinion on complex code changes
- Analyzing architecture decisions
- Debugging difficult issues
- Planning major refactors or new features

Has access to: read_file, grep, glob, web_search, read_web_page tools.

Returns expert guidance with reasoning."""


# --- Tool Implementations ---


async def _finder_search(
    query: str,
    backend: BackendProtocol,
    search_engine: Any | None = None,
) -> str:
    """Perform semantic codebase search.

    Uses embedding-based search when available, falls back to keyword search.

    Args:
        query: Search query.
        backend: Backend for file access.
        search_engine: Optional CodeSearchEngine for semantic search.

    Returns:
        Formatted search results.
    """
    # If we have a search engine with embeddings, use semantic search
    if search_engine is not None:
        try:
            # First, index files if needed (lazy indexing on first search)
            if not search_engine._index:
                await _index_codebase(backend, search_engine)

            results = await search_engine.search(query, limit=20)
            if results:
                return _format_semantic_results(query, results)
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

    # Fall back to keyword-based grep search
    return await _keyword_grep_search(query, backend)


async def _index_codebase(
    backend: BackendProtocol,
    search_engine: Any,
    extensions: tuple[str, ...] = (".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h"),
    max_files: int = 500,
) -> None:
    """Index codebase files for semantic search.

    Args:
        backend: Backend for file access.
        search_engine: Search engine to index into.
        extensions: File extensions to index.
        max_files: Maximum files to index.
    """
    logger.info("Indexing codebase for semantic search...")

    # Find files using glob
    try:
        files = await backend.aglob(pattern="**/*", path="/")
    except Exception as e:
        logger.warning(f"Failed to list files for indexing: {e}")
        return

    indexed = 0
    for file_info in files[: max_files * 2]:  # Check more files to find enough matching extensions
        if indexed >= max_files:
            break

        path = file_info.get("path", "")
        if not any(path.endswith(ext) for ext in extensions):
            continue

        # Skip common non-code directories
        if any(skip in path for skip in ["node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build"]):
            continue

        try:
            content = await backend.aread(path)
            if content:
                await search_engine.index_file(path, content)
                indexed += 1
        except Exception as e:
            logger.debug(f"Failed to read {path} for indexing: {e}")

    logger.info(f"Indexed {indexed} files for semantic search")


async def _keyword_grep_search(query: str, backend: BackendProtocol) -> str:
    """Perform keyword-based grep search (fallback).

    Args:
        query: Search query.
        backend: Backend for grep.

    Returns:
        Formatted results.
    """
    import re

    words = re.findall(r"\b\w+\b", query.lower())
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "for",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "from",
        "with",
        "by",
        "that",
        "this",
        "it",
        "its",
        "of",
        "all",
        "find",
        "where",
        "how",
        "what",
        "when",
    }
    keywords = [w for w in words if w not in stopwords and len(w) > 2]

    if not keywords:
        return "Could not extract meaningful keywords from query. Please be more specific."

    results = []
    seen_files: set[str] = set()

    for keyword in keywords[:5]:
        try:
            matches = await backend.agrep(
                pattern=keyword,
                path="/",
                output_mode="content",
            )
            for match in matches[:10]:
                file_path = match.get("path", "")
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    results.append(
                        {
                            "path": file_path,
                            "line": match.get("line", 0),
                            "text": match.get("text", ""),
                            "keyword": keyword,
                        }
                    )
        except Exception as e:
            logger.warning(f"Finder grep failed for keyword '{keyword}': {e}")

    if not results:
        return f"No code found matching query: {query}\n\nTry using grep with specific terms."

    # Group by file
    file_keyword_counts: dict[str, int] = {}
    for r in results:
        path = r["path"]
        file_keyword_counts[path] = file_keyword_counts.get(path, 0) + 1

    sorted_files = sorted(file_keyword_counts.items(), key=lambda x: -x[1])

    output = [f"Keyword search results for: {query}\n"]
    output.append(f"Keywords: {', '.join(keywords[:5])}\n")

    for file_path, count in sorted_files[:15]:
        file_results = [r for r in results if r["path"] == file_path]
        output.append(f"\n**{file_path}** (matches: {count})")
        for r in file_results[:3]:
            output.append(f"  Line {r['line']}: {r['text'][:100]}...")

    return "\n".join(output)


def _format_semantic_results(query: str, results: list[Any]) -> str:
    """Format semantic search results.

    Args:
        query: Original query.
        results: List of SearchResult objects.

    Returns:
        Formatted string.
    """
    output = [f"Semantic search results for: {query}\n"]

    if results and results[0].match_type == "semantic":
        output.append("Search mode: Embedding-based semantic search\n")
    else:
        output.append("Search mode: Keyword matching\n")

    for i, result in enumerate(results[:15], 1):
        score_pct = int(result.score * 100)
        output.append(f"\n**{i}. {result.path}** (relevance: {score_pct}%)")
        output.append(f"   {result.context}")

        # Show preview
        preview = result.text.replace("\n", " ")[:150]
        output.append(f"   Preview: {preview}...")

    return "\n".join(output)


async def _look_at_file(
    path: str,
    objective: str,
    backend: BackendProtocol,
    context: str | None = None,
    referenceFiles: list[str] | None = None,
    model: Any | None = None,
) -> str:
    """Extract information from media files.

    Args:
        path: Path to the file to analyze.
        objective: What to look for in the file.
        backend: Backend for file access.
        context: Optional additional context.
        referenceFiles: Optional reference files for comparison.
        model: Optional vision-capable LLM model for image analysis.

    Returns:
        Analysis result or file info.
    """
    import mimetypes
    import os
    import re

    # Determine file type
    suffix = Path(path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or "application/octet-stream"

    content: bytes | None = None

    # Check if this is an absolute path to an external file (outside workspace)
    # On Windows: C:\... or D:\...
    # On Unix: /home/... /Users/... etc.
    is_absolute_external = (
        (os.name == "nt" and re.match(r"^[A-Za-z]:\\", path))  # Windows absolute
        or (os.name != "nt" and path.startswith("/") and not path.startswith("/workspace"))  # Unix absolute
    )

    if is_absolute_external:
        # Try to read directly from filesystem for external absolute paths
        try:
            file_path = Path(path)
            if file_path.exists() and file_path.is_file():
                content = file_path.read_bytes()
                logger.info(f"Read external file directly: {path}")
            else:
                return f"File not found: {path}"
        except PermissionError:
            return f"Permission denied reading file: {path}"
        except Exception as e:
            return f"Failed to read external file {path}: {e}"
    else:
        # Use backend for workspace-relative paths
        try:
            responses = await backend.adownload_files([path])
            if not responses:
                return f"Failed to read file {path}: No response from backend"
            response = responses[0]
            if response.error:
                return f"Failed to read file {path}: {response.error}"
            if response.content is None:
                return f"Failed to read file {path}: File is empty or could not be read"
            content = response.content
        except Exception as e:
            return f"Failed to read file {path}: {e}"

    if content is None:
        return f"Failed to read file {path}: No content"

    file_size = len(content)

    # Handle different file types
    if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        # If we have a vision model, use it
        if model is not None:
            try:
                from deepagents.vision import ainvoke_vision_model

                return await ainvoke_vision_model(
                    model=model,
                    image_bytes=content,
                    query=objective,
                    context=context,
                    file_type=mime_type,
                )
            except Exception as e:
                logger.warning(f"Vision model failed for {path}: {e}")
                # Fall through to basic info

        # Fallback: return basic file info
        return f"""Image file: {path}
Size: {file_size:,} bytes
Format: {suffix[1:].upper()}

Objective: {objective}
{f"Context: {context}" if context else ""}

Note: Vision model not configured. To enable image analysis, pass a vision-capable
model (Claude 3+, GPT-4V, etc.) to AdvancedMiddleware(model=your_model)."""

    if suffix == ".pdf":
        # If we have a vision model, could render PDF pages and analyze
        # For now, return info about the PDF
        return f"""PDF file: {path}
Size: {file_size:,} bytes

Objective: {objective}
{f"Context: {context}" if context else ""}

Note: PDF analysis requires additional libraries. Install pypdf for text extraction:
  pip install pypdf

For visual PDF analysis, a document processing pipeline is needed."""

    if suffix == ".svg":
        # SVG is XML-based, can be read as text
        svg_text = content.decode("utf-8", errors="replace")
        lines = svg_text.split("\n")[:20]
        return f"""SVG file: {path}
Size: {file_size:,} bytes

First 20 lines of SVG content:
```xml
{chr(10).join(lines)}
```

Objective: {objective}"""

    return f"Unsupported file type: {suffix}. Supported: PNG, JPEG, GIF, WebP, SVG, PDF"


# --- Tool Generators ---


def _finder_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
    search_engine: Any | None = None,
) -> BaseTool:
    """Generate the finder (semantic search) tool.

    Args:
        backend: Backend for file operations.
        custom_description: Optional custom tool description.
        search_engine: Optional CodeSearchEngine for semantic search.

    Returns:
        Configured finder tool.
    """
    tool_description = custom_description or FINDER_DESCRIPTION

    def _get_backend(runtime: ToolRuntime) -> BackendProtocol:
        if callable(backend):
            return backend(runtime)
        return backend

    async def async_finder(query: str, runtime: ToolRuntime) -> str:
        resolved_backend = _get_backend(runtime)
        return await _finder_search(query, resolved_backend, search_engine)

    def sync_finder(query: str, runtime: ToolRuntime) -> str:
        import asyncio

        resolved_backend = _get_backend(runtime)
        return asyncio.get_event_loop().run_until_complete(_finder_search(query, resolved_backend, search_engine))

    return StructuredTool.from_function(
        name="finder",
        description=tool_description,
        func=sync_finder,
        coroutine=async_finder,
    )


def _look_at_tool_generator(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_description: str | None = None,
    model: Any | None = None,
) -> BaseTool:
    """Generate the look_at (media extraction) tool.

    Args:
        backend: Backend for file access.
        custom_description: Optional custom tool description.
        model: Optional vision-capable LLM model for image analysis.

    Returns:
        Configured look_at tool.
    """
    tool_description = custom_description or LOOK_AT_DESCRIPTION

    def _get_backend(runtime: ToolRuntime) -> BackendProtocol:
        if callable(backend):
            return backend(runtime)
        return backend

    async def async_look_at(
        path: str,
        objective: str,
        context: str | None = None,
        referenceFiles: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        resolved_backend = _get_backend(runtime)
        return await _look_at_file(path, objective, resolved_backend, context, referenceFiles, model=model)

    def sync_look_at(
        path: str,
        objective: str,
        context: str | None = None,
        referenceFiles: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        import asyncio

        resolved_backend = _get_backend(runtime)
        return asyncio.get_event_loop().run_until_complete(_look_at_file(path, objective, resolved_backend, context, referenceFiles, model=model))

    return StructuredTool.from_function(
        name="look_at",
        description=tool_description,
        func=sync_look_at,
        coroutine=async_look_at,
    )


# --- Subagent Definitions ---

LIBRARIAN_SYSTEM_PROMPT = """You are a Librarian - an expert at understanding complex codebases.

Your role is to provide deep, documentation-quality insights about code architecture,
patterns, and relationships. You analyze code thoughtfully and explain your findings clearly.

When answering questions:
1. First explore the codebase to understand its structure
2. Identify relevant files and patterns
3. Analyze relationships between components
4. Provide detailed, well-organized explanations
5. Include specific code references where helpful

You have access to file reading and search tools. Use them extensively to gather context
before providing your analysis. Be thorough but focus on what's most relevant to the question."""

ORACLE_SYSTEM_PROMPT = """You are an Oracle - an expert AI advisor for complex technical decisions.

Your role is to provide thoughtful, well-reasoned guidance on technical matters including:
- Code reviews and quality analysis
- Architecture decisions and tradeoffs
- Bug finding and debugging strategies
- Performance optimization
- Refactoring approaches

When providing guidance:
1. Carefully analyze the context and files provided
2. Consider multiple perspectives and tradeoffs
3. Explain your reasoning clearly
4. Provide specific, actionable recommendations
5. Note any caveats or areas of uncertainty

Be direct and helpful. Focus on providing expert-level insights that help make better decisions."""


def get_librarian_subagent() -> dict:
    """Get the librarian subagent specification."""
    return {
        "name": "librarian",
        "description": LIBRARIAN_DESCRIPTION,
        "system_prompt": LIBRARIAN_SYSTEM_PROMPT,
        "tools": [],  # Will use default tools from SubAgentMiddleware
    }


def get_oracle_subagent() -> dict:
    """Get the oracle subagent specification."""
    return {
        "name": "oracle",
        "description": ORACLE_DESCRIPTION,
        "system_prompt": ORACLE_SYSTEM_PROMPT,
        "tools": [],  # Will use default tools from SubAgentMiddleware
    }


# --- Tool Registry ---

ADVANCED_TOOL_GENERATORS = {
    "finder": _finder_tool_generator,
    "look_at": _look_at_tool_generator,
}


def _get_advanced_tools(
    backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol],
    custom_tool_descriptions: dict[str, str] | None = None,
    enabled_tools: list[str] | None = None,
    model: Any | None = None,
    search_engine: Any | None = None,
) -> list[BaseTool]:
    """Get advanced AI-powered tools.

    Args:
        backend: Backend for file operations.
        custom_tool_descriptions: Optional custom tool descriptions.
        enabled_tools: Optional list of tool names to enable.
        model: Optional vision-capable LLM model for look_at tool.
        search_engine: Optional CodeSearchEngine for semantic search.

    Returns:
        List of configured tools.
    """
    if custom_tool_descriptions is None:
        custom_tool_descriptions = {}

    tools_to_generate = enabled_tools or list(ADVANCED_TOOL_GENERATORS.keys())
    tools = []

    for tool_name in tools_to_generate:
        if tool_name in ADVANCED_TOOL_GENERATORS:
            tool_generator = ADVANCED_TOOL_GENERATORS[tool_name]
            # Pass model parameter only to tools that support it (look_at)
            if tool_name == "look_at":
                tool = tool_generator(backend, custom_tool_descriptions.get(tool_name), model=model)
            elif tool_name == "finder":
                tool = tool_generator(backend, custom_tool_descriptions.get(tool_name), search_engine=search_engine)
            else:
                tool = tool_generator(backend, custom_tool_descriptions.get(tool_name))
            tools.append(tool)

    return tools


ADVANCED_SYSTEM_PROMPT = """## Advanced AI Tools

You have access to advanced AI-powered tools:

- finder: Semantic codebase search - find code by functionality/concepts (not just exact text)
- look_at: Extract information from images, PDFs, and media files

For complex analysis, you can also use specialized subagents:
- librarian: Deep codebase understanding and documentation
- oracle: Expert reasoning for technical decisions, code reviews, and architecture
"""


class AdvancedMiddleware(AgentMiddleware):
    """Middleware for advanced AI-powered tools: finder, look_at.

    This middleware adds AI-powered tools for semantic search and media analysis.
    For the specialized subagents (librarian, oracle), use SubAgentMiddleware with
    the subagent specifications from this module.

    Args:
        backend: Backend for file operations.
        system_prompt: Optional custom system prompt override.
        custom_tool_descriptions: Optional custom tool descriptions override.
        enabled_tools: Optional list of tool names to enable. Defaults to all tools.
        model: Optional LLM model for vision-capable tools (e.g., look_at).
            If provided, look_at will use this model to analyze images.
            Use any vision-capable model (Claude 3+, GPT-4V, Gemini, etc.).
        enable_semantic_search: Enable embedding-based semantic search for finder.
            Requires OPENAI_API_KEY environment variable.
        embedding_model: OpenAI embedding model to use. Defaults to "text-embedding-3-small".

    Example:
        ```python
        from deepagents.middleware.advanced import AdvancedMiddleware, get_librarian_subagent
        from deepagents.middleware import SubAgentMiddleware
        from langchain.agents import create_agent

        # Enable advanced tools with vision support
        from deepagents import get_default_model

        agent = create_agent(middleware=[AdvancedMiddleware(model=get_default_model())])

        # Enable semantic search with embeddings
        agent = create_agent(middleware=[AdvancedMiddleware(enable_semantic_search=True)])

        # Also add librarian and oracle subagents
        agent = create_agent(
            middleware=[
                AdvancedMiddleware(model=model),
                SubAgentMiddleware(default_model="claude-sonnet-4-20250514", subagents=[get_librarian_subagent(), get_oracle_subagent()]),
            ]
        )
        ```
    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | Callable[[ToolRuntime], BackendProtocol] | None = None,
        system_prompt: str | None = None,
        custom_tool_descriptions: dict[str, str] | None = None,
        enabled_tools: list[str] | None = None,
        model: Any | None = None,
        enable_semantic_search: bool = False,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        """Initialize the advanced middleware.

        Args:
            backend: Backend for file operations. Defaults to StateBackend.
            system_prompt: Optional custom system prompt override.
            custom_tool_descriptions: Optional custom tool descriptions.
            enabled_tools: Optional list of tool names to enable.
            model: Optional vision-capable LLM model for look_at tool.
            enable_semantic_search: Enable embedding-based semantic search.
            embedding_model: OpenAI embedding model name.
        """
        from deepagents.backends import StateBackend

        self.backend = backend if backend is not None else (lambda rt: StateBackend(rt))
        self._custom_system_prompt = system_prompt
        self.model = model

        # Create search engine if semantic search is enabled
        search_engine = None
        if enable_semantic_search:
            try:
                from deepagents.search import CodeSearchEngine, OpenAIEmbeddingProvider

                embedding_provider = OpenAIEmbeddingProvider(model=embedding_model)
                search_engine = CodeSearchEngine(embedding_provider=embedding_provider)
                logger.info("Semantic search enabled with %s", embedding_model)
            except Exception as e:
                logger.warning("Failed to enable semantic search: %s", e)

        self.search_engine = search_engine
        self.tools = _get_advanced_tools(
            self.backend,
            custom_tool_descriptions,
            enabled_tools,
            model=model,
            search_engine=search_engine,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Add advanced tools system prompt."""
        system_prompt = self._custom_system_prompt or ADVANCED_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Add advanced tools system prompt."""
        system_prompt = self._custom_system_prompt or ADVANCED_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = request.system_prompt + "\n\n" + system_prompt if request.system_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)

        return await handler(request)
