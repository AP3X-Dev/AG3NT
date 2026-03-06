"""DeepAgents runtime for AG3NT.

This module builds and manages the DeepAgents agent graph,
exposing a simple run_turn() interface for the worker.

Supported model providers:
- anthropic: Claude models (requires ANTHROPIC_API_KEY)
- openai: OpenAI models (requires OPENAI_API_KEY)
- openrouter: OpenRouter proxy (requires OPENROUTER_API_KEY)
- kimi: Moonshot AI models (requires KIMI_API_KEY)
- google: Google Gemini models (requires GOOGLE_API_KEY)

Environment variables:
- AG3NT_MODEL_PROVIDER: Model provider override (if unset, provider is auto-detected by available API keys)
- AG3NT_MODEL_NAME: Model name override (if unset, a provider-specific default is used)
- AG3NT_AUTO_APPROVE: Set to "true" to skip approval for risky tools (default: "false")
- AG3NT_MCP_SERVERS: JSON string with MCP server configuration (optional)
- OPENROUTER_API_KEY: Required when using OpenRouter
- KIMI_API_KEY: Required when using Kimi
- TAVILY_API_KEY: Optional, enables web search for research subagent

Tracing (LangSmith):
- LANGSMITH_API_KEY: LangSmith API key to enable tracing (get one at smith.langchain.com)
- LANGCHAIN_PROJECT: Project name in LangSmith dashboard (default: "ag3nt")
- AG3NT_TRACING_ENABLED: Explicit override to enable/disable ("true"/"false")

When tracing is enabled, all agent runs are logged to LangSmith with:
- Full trace of LLM calls, tool executions, and subagent delegations
- Token usage per call
- Latency metrics
- Error information for debugging

Skills System:
AG3NT loads skills from these locations (in priority order, last wins):
1. Bundled: {repo}/skills/ - shipped with AG3NT
2. Global: ~/.ag3nt/skills/ - user's personal skills
3. Workspace: ./skills/ - project-specific skills

Skills are SKILL.md files in folders. See skills/example-skill for the contract.

Memory System:
AG3NT persists memory to ~/.ag3nt/:
- AGENTS.md - Project context and agent identity
- MEMORY.md - Long-term facts about the user
- memory/ - Daily conversation logs (YYYY-MM-DD.md)
- vectors/ - FAISS index for semantic memory search

The `memory_search` tool provides semantic search over memory files.
Requires embeddings API (uses same key as chat model) and faiss-cpu.

Sub-Agent System:
AG3NT can spawn specialized sub-agents for complex tasks:
- Researcher: Web search and information gathering
- Coder: Code writing, analysis, and execution

MCP (Model Context Protocol) Integration:
AG3NT can load tools from external MCP servers. Configure servers via:
1. Environment variable: AG3NT_MCP_SERVERS (JSON string)
2. Config file: ~/.ag3nt/mcp_servers.json

Example config (follows Claude Desktop / MCP standard format):
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_TOKEN": "ghp_..."}
        }
    }
}

Approval System:
AG3NT can pause before executing risky tools for human approval.
Risky tools include: execute, shell, write_file, edit_file, delete_file
Set AG3NT_AUTO_APPROVE=true to skip approval (power user mode).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ag3nt_agent.context_summarization import (
    create_summarization_middleware,
    get_default_summarization_config,
)
from ag3nt_agent.interactive_tools import get_interactive_tools
from ag3nt_agent.identity import IdentityLoader
from ag3nt_agent.memory_search import search_memory
from ag3nt_agent.agent_guard_middleware import AgentGuardMiddleware
from ag3nt_agent.planning_middleware import PlanningMiddleware
from ag3nt_agent.shell_middleware import ShellMiddleware
from ag3nt_agent.skill_trigger_middleware import SkillTriggerMiddleware
from ag3nt_agent.turn_context_middleware import TurnContextMiddleware

# =============================================================================
# LANGCHAIN TRACING CONFIGURATION
# =============================================================================
# LangSmith tracing is enabled automatically if LANGSMITH_API_KEY is set.
# Additional env vars:
# - LANGCHAIN_PROJECT: Project name in LangSmith (default: "ag3nt")
# - LANGCHAIN_TRACING_V2: Set to "true" to enable (auto-enabled if API key present)
# - AG3NT_TRACING_ENABLED: Explicit override ("true"/"false") to enable/disable

def _configure_tracing() -> None:
    """Configure LangChain tracing if API key is available.

    This sets up LangSmith tracing for all agent runs, providing:
    - Detailed trace of all LLM calls and tool executions
    - Token usage tracking
    - Latency metrics
    - Debug information for failed runs
    """
    # Check for explicit override
    tracing_override = os.environ.get("AG3NT_TRACING_ENABLED", "").lower()
    if tracing_override == "false":
        logging.getLogger("ag3nt.tracing").info("Tracing explicitly disabled via AG3NT_TRACING_ENABLED=false")
        return

    # Check for LangSmith API key
    langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")

    if langsmith_api_key or tracing_override == "true":
        # Enable LangChain tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Set project name if not already set
        if not os.environ.get("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "ag3nt"

        project = os.environ.get("LANGCHAIN_PROJECT", "ag3nt")
        logging.getLogger("ag3nt.tracing").info(
            f"LangSmith tracing enabled for project: {project}"
        )
    else:
        logging.getLogger("ag3nt.tracing").debug(
            "LangSmith tracing not configured (set LANGSMITH_API_KEY to enable)"
        )

# Initialize tracing on module load
_configure_tracing()

# Lazy import InterruptOnConfig from langchain middleware
try:
    from langchain.agents.middleware import InterruptOnConfig
except ImportError:
    InterruptOnConfig = dict  # Fallback type hint

# Lazy import to avoid import errors if DeepAgents is not installed
_agent: CompiledStateGraph | None = None

# Stores interrupt IDs per session for multi-interrupt resume
_pending_interrupt_ids: dict[str, list[str]] = {}
_pending_interrupt_ids_lock = threading.Lock()

# Agent pool for pre-warmed instances (optional feature)
_use_agent_pool: bool = os.environ.get("AG3NT_USE_AGENT_POOL", "false").lower() == "true"

# Model fallback chain singleton (lazy-initialized from env)
_fallback_chain: "ModelFallbackChain | None" = None

# Set up logging for approval events
logger = logging.getLogger("ag3nt.approval")

def _emit_turn_completed(*, session_id: str, char_count: int) -> None:
    """Emit turn.completed event to the EventBus for compaction tracking.

    Non-blocking, fire-and-forget. Failures are logged and swallowed.
    """
    try:
        from ag3nt_agent.autonomous.event_bus import get_event_bus, Event, EventPriority

        event = Event(
            event_type="turn.completed",
            source="deepagents_runtime",
            payload={"session_id": session_id, "char_count": char_count},
            priority=EventPriority.LOW,
        )
        bus = get_event_bus()
        bus.emit_sync(event)
    except Exception:
        logger.debug("Failed to emit turn.completed event", exc_info=True)


# =============================================================================
# RISKY TOOL DEFINITIONS
# =============================================================================

# Tools that require human approval before execution (Milestone 5)
RISKY_TOOLS = [
    "execute",        # Execute shell commands
    "shell",          # Run shell commands
    "exec_command",   # Full-featured shell execution
    "write_file",     # Write/create files
    "edit_file",      # Modify existing files
    "delete_file",    # Delete files
    "apply_patch",    # Multi-file structured patching
    "git_commit",     # Create git commits
]

# Tools that are potentially risky but may be allowed in trusted mode
POTENTIALLY_RISKY_TOOLS = [
    "fetch_url",      # Make network requests
    "web_search",     # Search the web
    "task",           # Delegate to subagent
]


def _is_yolo_mode() -> bool:
    """Check if YOLO mode is enabled (full autonomous operation).

    Returns:
        True if AG3NT_YOLO_MODE is set to "true"
    """
    return os.environ.get("AG3NT_YOLO_MODE", "false").lower() == "true"


def _is_auto_approve_enabled() -> bool:
    """Check if auto-approve mode is enabled.

    Returns:
        True if AG3NT_AUTO_APPROVE or AG3NT_YOLO_MODE is set to "true"
    """
    if _is_yolo_mode():
        return True
    return os.environ.get("AG3NT_AUTO_APPROVE", "false").lower() == "true"


def _format_tool_description(tool_call: dict, _state: Any = None, _runtime: Any = None) -> str:
    """Format a tool call for human-readable display.

    This function is used as a callback for langgraph's interrupt mechanism,
    which passes (tool_call, state, runtime). When called directly, only
    tool_call is required.

    Args:
        tool_call: The tool call dict with 'name' and 'args'
        _state: Agent state (unused, for callback compatibility)
        _runtime: Runtime instance (unused, for callback compatibility)

    Returns:
        Formatted description string
    """
    name = tool_call.get("name", "unknown")
    args = tool_call.get("args", {})

    if name == "execute":
        command = args.get("command", "N/A")
        return f"🖥️ Execute Command:\n```\n{command}\n```"
    elif name == "shell":
        command = args.get("command", "N/A")
        return f"🖥️ Shell Command:\n```\n{command}\n```"
    elif name == "exec_command":
        command = args.get("command", "N/A")
        bg = " [background]" if args.get("background") else ""
        return f"⚡ Exec Command{bg}:\n```\n{command}\n```"
    elif name == "process_tool":
        action = args.get("action", "N/A")
        session_id = args.get("session_id", "")
        return f"🔄 Process: {action}" + (f" (session: {session_id})" if session_id else "")
    elif name == "apply_patch":
        patch_text = args.get("patch", "")
        file_count = patch_text.count("*** Add File:") + patch_text.count("*** Update File:") + patch_text.count("*** Delete File:")
        dry = " [dry run]" if args.get("dry_run") else ""
        return f"🩹 Apply Patch{dry}: {file_count} file(s)"
    elif name == "write_file":
        path = args.get("file_path") or args.get("path", "N/A")
        content = args.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content
        return f"📝 Write File: `{path}`\n```\n{preview}\n```"
    elif name == "edit_file":
        path = args.get("file_path") or args.get("path", "N/A")
        return f"✏️ Edit File: `{path}`"
    elif name == "delete_file":
        path = args.get("file_path") or args.get("path", "N/A")
        return f"🗑️ Delete File: `{path}`"
    else:
        return f"🔧 Tool: {name}\nArgs: {args}"


def _get_interrupt_on_config() -> dict[str, bool | dict]:
    """Build interrupt_on configuration for risky tools and interactive tools.

    Returns:
        Dict mapping tool names to interrupt configurations.
        Includes risky tools (if not auto-approved) and interactive tools (always).
    """
    config: dict[str, bool | dict] = {}

    # In YOLO mode, only keep ask_user for agent-initiated questions
    if _is_yolo_mode():
        logger.info("YOLO mode enabled - all approval gates disabled")
    elif not _is_auto_approve_enabled():
        # Add risky tools (if not auto-approved)
        for tool_name in RISKY_TOOLS:
            config[tool_name] = {
                "allowed_decisions": ["approve", "reject"],
                "description": _format_tool_description,
            }
        logger.info(f"Approval required for tools: {', '.join(RISKY_TOOLS)}")
    else:
        logger.info("Auto-approve mode enabled - risky tools will run without approval")

    # Always add ask_user (interactive tool that always needs user input)
    config["ask_user"] = {
        "allowed_decisions": ["answer"],  # Special decision type for user questions
        "description": lambda tool_call, _state=None, _runtime=None: f"Ask user: {tool_call.get('args', {}).get('question', 'N/A')}",
    }

    # Always add request_external_access (requires user approval for external paths)
    try:
        from ag3nt_agent.external_path_tool import (
            EXTERNAL_ACCESS_TOOL,
            format_external_access_request,
        )
        config[EXTERNAL_ACCESS_TOOL] = {
            "allowed_decisions": ["approve", "reject"],
            "description": format_external_access_request,
        }
        logger.debug("External path access approval configured")
    except ImportError:
        pass

    return config


def _get_model_config() -> tuple[str, str]:
    """Get the model provider and name from environment.

    Delegates to ag3nt_agent.model_config.get_model_config().
    """
    from ag3nt_agent.model_config import get_model_config
    return get_model_config()


def _create_model() -> "BaseChatModel | str":
    """Create the appropriate model instance based on provider.

    Delegates to ag3nt_agent.model_config.create_model().
    """
    from ag3nt_agent.model_config import create_model
    return create_model()


def get_fallback_chain():
    """Get or create the singleton ModelFallbackChain.

    The chain is built from environment variables on first call and
    cached for the lifetime of the process.  The primary model (index 0)
    is the same provider/model that ``_create_model()`` would produce;
    additional providers are appended based on available API keys.

    Returns:
        ModelFallbackChain instance.
    """
    global _fallback_chain
    if _fallback_chain is None:
        from ag3nt_agent.model_fallback import ModelFallbackChain
        _fallback_chain = ModelFallbackChain.from_env()
        logger.info(
            "ModelFallbackChain initialised with %d provider(s): %s",
            len(_fallback_chain.providers),
            [p["provider"] for p in _fallback_chain.providers],
        )
    return _fallback_chain


def _get_global_skills_path() -> Path | None:
    """Get the global skills directory path if it exists.

    Returns:
        Path to ~/.ag3nt/skills/ if it exists, else None
    """
    global_skills = Path.home() / ".ag3nt" / "skills"
    if global_skills.exists() and global_skills.is_dir():
        return global_skills
    return None


def _get_user_data_path() -> Path:
    """Get or create the user data directory at ~/.ag3nt/.

    Creates the directory structure if it doesn't exist:
    - ~/.ag3nt/
    - ~/.ag3nt/memory/ (for daily logs)

    Returns:
        Path to ~/.ag3nt/
    """
    user_data = Path.home() / ".ag3nt"
    user_data.mkdir(parents=True, exist_ok=True)
    (user_data / "memory").mkdir(exist_ok=True)
    return user_data


def _get_memory_sources() -> list[str]:
    """Get the memory file sources for MemoryMiddleware.

    Returns paths relative to the CompositeBackend's /user-data/ route.

    Returns:
        List of memory source paths
    """
    # Ensure user data directory exists
    _get_user_data_path()

    return [
        "/user-data/AGENTS.md",  # Project context and identity
        "/user-data/MEMORY.md",  # Long-term facts
    ]


# =============================================================================
# MCP (MODEL CONTEXT PROTOCOL) INTEGRATION
# =============================================================================


def _load_mcp_config() -> dict | None:
    """Load MCP server configuration from config file or environment.

    MCP servers can be configured in two ways (priority order):
    1. Environment variable: AG3NT_MCP_SERVERS (JSON string)
    2. Config file: ~/.ag3nt/mcp_servers.json

    Config format follows the Claude Desktop / MCP standard:
    {
        "mcpServers": {
            "server-name": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path"],
                "env": {"KEY": "value"}  # Optional
            }
        }
    }

    Returns:
        MCP configuration dict or None if not configured
    """
    import json

    # Check environment variable first
    mcp_env = os.environ.get("AG3NT_MCP_SERVERS")
    if mcp_env:
        try:
            config = json.loads(mcp_env)
            logger.info("Loaded MCP config from AG3NT_MCP_SERVERS environment variable")
            return config
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in AG3NT_MCP_SERVERS: {e}")
            return None

    # Check config file
    config_path = Path.home() / ".ag3nt" / "mcp_servers.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded MCP config from {config_path}")
            return config
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load MCP config from {config_path}: {e}")
            return None

    return None


def _sanitize_surrogates(s: str) -> str:
    """Remove unpaired surrogates that break UTF-8 encoding on Windows."""
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _load_mcp_tools() -> tuple[list, dict[str, list[str]]]:
    """Load tools from configured MCP servers.

    Uses langchain-mcp-adapters (v0.2+) to connect to MCP servers and
    convert their tools to LangChain-compatible tools.  Each server is
    loaded independently so one failure doesn't break the rest.

    Returns:
        Tuple of (tools_list, server_tool_map) where server_tool_map maps
        server names to their tool names (e.g. {"context7": ["resolve-library-id", "query-docs"]}).
    """
    import asyncio

    config = _load_mcp_config()
    if not config or "mcpServers" not in config:
        return [], {}

    servers = config["mcpServers"]
    if not servers:
        return [], {}

    async def _async_load_mcp_tools() -> tuple[list, dict[str, list[str]]]:
        """Async implementation of MCP tool loading."""
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            logger.warning(
                "langchain-mcp-adapters failed to import. MCP tools unavailable. "
                "Install with: pip install langchain-mcp-adapters langgraph  "
                "Error: %s", exc
            )
            return [], {}

        # Build params — v0.2+ requires explicit "transport" key.
        server_params = {}
        for name, server_config in servers.items():
            if isinstance(server_config, dict) and server_config.get("enabled") is False:
                logger.info("Skipping disabled MCP server: %s", name)
                continue

            transport = server_config.get("transport", "stdio")
            params: dict = {"transport": transport}

            if transport == "stdio":
                params["command"] = server_config.get("command")
                params["args"] = server_config.get("args", [])
                if server_config.get("env"):
                    params["env"] = server_config["env"]
            else:
                # HTTP / SSE / WebSocket transports
                if server_config.get("url"):
                    params["url"] = server_config["url"]
                if server_config.get("headers"):
                    params["headers"] = server_config["headers"]

            server_params[name] = params

        if not server_params:
            return [], {}

        logger.info(f"Connecting to {len(server_params)} MCP server(s): {list(server_params.keys())}")

        # Load each server independently so one failure doesn't kill the rest.
        all_tools: list = []
        server_tool_map: dict[str, list[str]] = {}
        for srv_name, srv_params in server_params.items():
            try:
                client = MultiServerMCPClient({srv_name: srv_params})
                tools = await client.get_tools()
                logger.info(f"MCP server '{srv_name}': loaded {len(tools)} tool(s)")
                tool_names = []
                for tool in tools:
                    # Sanitize descriptions to strip surrogates from Windows subprocess output
                    if tool.description:
                        tool.description = _sanitize_surrogates(tool.description)
                    tool_names.append(tool.name)
                all_tools.extend(tools)
                server_tool_map[srv_name] = tool_names
            except Exception as e:
                logger.error(f"MCP server '{srv_name}' failed to load: {e}")

        logger.info(f"Total MCP tools loaded: {len(all_tools)}")
        return all_tools, server_tool_map

    # Run async function synchronously
    try:
        try:
            asyncio.get_running_loop()
            # Already in an async context — run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _async_load_mcp_tools())
                return future.result(timeout=60)
        except RuntimeError:
            # No running loop
            return asyncio.run(_async_load_mcp_tools())
    except Exception as e:
        logger.error(f"Error loading MCP tools: {e}")
        return [], {}


# Import the enhanced web search function from web_search module
from ag3nt_agent.web_search import internet_search as _internet_search_impl

# ── Shared HTTP session for connection pooling (fetch_url, etc.) ──
_http_session: "requests.Session | None" = None
_http_session_lock = threading.Lock()


def _get_http_session():
    """Return a shared requests.Session with connection pooling.

    Uses double-checked locking for thread safety with minimal overhead.
    """
    global _http_session
    if _http_session is None:
        with _http_session_lock:
            if _http_session is None:
                import requests as _req
                _http_session = _req.Session()
                _http_session.headers.update({
                    "User-Agent": "Mozilla/5.0 (compatible; AG3NT/1.0; +https://github.com/ag3nt)"
                })
    return _http_session


@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> dict:
    """Search the web for current information.

    Uses Tavily as primary provider with DuckDuckGo fallback.
    Includes caching and rate limiting for efficient API usage.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: "general" for most queries, "news" for current events, "finance" for financial data

    Returns:
        Search results with titles, URLs, content excerpts, and metadata.
    """
    return _internet_search_impl(query, max_results=max_results, topic=topic)


@tool
def fetch_url(
    url: str,
    timeout: int = 30,
) -> dict:
    """Fetch content from a URL and convert HTML to markdown format.

    Use this tool to read web page content. The HTML is automatically converted
    to clean markdown text for easy processing. After receiving the content,
    synthesize the relevant information to answer the user's question.

    Args:
        url: The URL to fetch (must be a valid HTTP/HTTPS URL)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dictionary containing:
        - success: Whether the request succeeded
        - url: The final URL after redirects
        - markdown_content: The page content converted to markdown
        - status_code: HTTP status code
        - content_length: Length of the markdown content
    """
    try:
        from markdownify import markdownify

        session = _get_http_session()
        response = session.get(url, timeout=timeout)
        response.raise_for_status()

        # Convert HTML content to markdown
        markdown_content = markdownify(response.text)

        # Truncate if too long (100KB limit)
        max_length = 100_000
        if len(markdown_content) > max_length:
            markdown_content = markdown_content[:max_length]
            markdown_content += f"\n\n... Content truncated at {max_length} characters."

        return {
            "success": True,
            "url": str(response.url),
            "markdown_content": markdown_content,
            "status_code": response.status_code,
            "content_length": len(markdown_content),
        }
    except ImportError as e:
        return {
            "error": f"Missing dependency: {e}",
            "suggestion": "Install with: pip install markdownify",
        }
    except (OSError, RuntimeError, ValueError) as e:
        return {
            "success": False,
            "error": f"Fetch URL error: {e!s}",
            "url": url,
        }


# Gateway URL for scheduler API
from ag3nt_agent.agent_config import GATEWAY_URL


@tool
def schedule_reminder(
    message: str,
    when: str,
    channel: str | None = None,
) -> dict:
    """Schedule a one-shot reminder to be sent at a specific time.

    Use this tool when the user asks you to remind them about something
    at a specific time or after a duration.

    Args:
        message: The reminder message (what to remind the user about)
        when: When to send the reminder. Can be:
              - Relative time: "in 10 minutes", "in 1 hour", "in 2 days"
              - ISO datetime: "2025-01-27T15:30:00"
        channel: Optional target channel type (e.g., "telegram", "discord")

    Returns:
        Result with job_id if successful, or error message if failed.

    Examples:
        schedule_reminder("Call Alice", "in 30 minutes")
        schedule_reminder("Team meeting", "2025-01-27T14:00:00")
    """
    import requests

    try:
        # Parse relative time to milliseconds if needed
        when_value: str | int = when
        match = re.match(r"^in\s+(\d+)\s+(second|minute|hour|day)s?$", when, re.IGNORECASE)
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()
            multipliers = {"second": 1000, "minute": 60_000, "hour": 3_600_000, "day": 86_400_000}
            when_value = amount * multipliers.get(unit, 60_000)

        response = requests.post(
            f"{GATEWAY_URL}/api/scheduler/reminder",
            json={
                "when": when_value,
                "message": message,
                "channelTarget": channel,
            },
            timeout=10,
        )

        if response.ok:
            data = response.json()
            return {
                "success": True,
                "job_id": data.get("jobId"),
                "message": f"Reminder scheduled: '{message}'",
            }
        else:
            return {
                "success": False,
                "error": f"Gateway returned {response.status_code}: {response.text}",
            }
    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to connect to Gateway: {e}",
            "suggestion": "Make sure the AG3NT Gateway is running on port 18789",
        }
    except Exception as e:
        return {"success": False, "error": f"Failed to schedule reminder: {e}"}


def _create_subagents(mcp_tools: list | None = None) -> list[dict]:
    """Create the sub-agent specifications using SubagentRegistry.

    Args:
        mcp_tools: Optional list of MCP tools to inject into sub-agents.
            These are added to sub-agents that use research/general tools
            (researcher, analyst, etc.) so they can use MCP servers like Context7.

    Returns:
        List of SubAgent dicts for all registered subagents.

    The subagent configurations are managed by SubagentRegistry which supports:
    - Builtin subagents (8 predefined types)
    - Plugin-registered subagents
    - User-defined subagents from config files (~/.ag3nt/subagents/)

    Configurations are converted to the dict format expected by DeepAgents SubAgentMiddleware.
    """
    from ag3nt_agent.subagent_registry import SubagentRegistry

    # Get registry and load user-defined configs from ~/.ag3nt/subagents/
    registry = SubagentRegistry.get_instance()
    user_data_path = _get_user_data_path()
    loaded = registry.load_user_configs(user_data_path)
    if loaded > 0:
        logger.info("Loaded %d user-defined subagents from %s/subagents/", loaded, user_data_path)

    # Map tool names to actual tool functions
    # This maps the string tool names in SubagentConfig.tools to actual callable tools
    # Import browser tools for tool_map
    from ag3nt_agent.browser_tool import (
        browser_start_session,
        browser_navigate,
        browser_screenshot,
        browser_click,
        browser_fill,
        browser_get_content,
        browser_wait_for,
        browser_close,
    )

    tool_map: dict = {
        # Research tools
        "internet_search": internet_search,
        "fetch_url": fetch_url,
        # File tools are provided by DeepAgents backend (use empty list = default tools)
        # These are placeholders that signal we need default tools
        "read_file": None,  # Default tool
        "write_file": None,  # Default tool
        "edit_file": None,  # Default tool
        "shell": None,  # Default tool (shell middleware)
        # Memory tools
        "memory_search": None,
        # Browser tools
        "browser_start_session": browser_start_session,
        "browser_navigate": browser_navigate,
        "browser_screenshot": browser_screenshot,
        "browser_click": browser_click,
        "browser_fill": browser_fill,
        "browser_type": browser_fill,  # alias
        "browser_get_content": browser_get_content,
        "browser_wait_for": browser_wait_for,
        "browser_close": browser_close,
    }

    # Load git tools
    try:
        from ag3nt_agent.git_tool import (
            git_status, git_diff, git_log, git_add, git_commit, git_branch, git_show,
        )
        tool_map["git_status"] = git_status
        tool_map["git_diff"] = git_diff
        tool_map["git_log"] = git_log
        tool_map["git_add"] = git_add
        tool_map["git_commit"] = git_commit
        tool_map["git_branch"] = git_branch
        tool_map["git_show"] = git_show
    except ImportError:
        pass

    # Load planning tools
    try:
        from ag3nt_agent.planning_tools import write_todos, read_todos, update_todo
        tool_map["write_todos"] = write_todos
        tool_map["read_todos"] = read_todos
        tool_map["update_todo"] = update_todo
    except ImportError:
        pass

    # Load session tools
    try:
        from ag3nt_agent.session_tools import sessions_list, sessions_history, sessions_send
        tool_map["sessions_list"] = sessions_list
        tool_map["sessions_history"] = sessions_history
        tool_map["sessions_send"] = sessions_send
    except ImportError:
        pass

    # Try to load additional tools that may be available
    try:
        from ag3nt_agent.memory_search import get_memory_search_tool, get_memory_store_tool
        tool_map["memory_search"] = get_memory_search_tool()
        tool_map["memory_store"] = get_memory_store_tool()
    except ImportError:
        pass

    # Add MCP tools to the tool map so sub-agents can access them
    if mcp_tools:
        for tool in mcp_tools:
            tool_name = getattr(tool, "name", None)
            if tool_name:
                tool_map[tool_name] = tool

    # Sub-agents that should receive MCP tools (research-oriented agents)
    _mcp_subagents = {"researcher", "analyst", "coder", "planner", "writer"}

    subagents = []
    for config in registry.list_all():
        # Convert tool names to actual tool functions
        tools = []
        uses_default_tools = False

        for tool_name in config.tools:
            if tool_name in tool_map:
                tool_func = tool_map[tool_name]
                if tool_func is not None:
                    tools.append(tool_func)
                else:
                    # None means use default tools (filesystem, shell)
                    uses_default_tools = True
            else:
                logger.warning(f"Unknown tool '{tool_name}' in subagent '{config.name}'")

        # If any tool was None (default tool), use empty list to get default tools
        if uses_default_tools and not tools:
            tools = []  # Empty = default tools from DeepAgents

        # Inject MCP tools into research-oriented sub-agents so they can use
        # MCP servers (e.g. Context7 for documentation lookup)
        if mcp_tools and config.name in _mcp_subagents:
            existing_names = {getattr(t, "name", None) for t in tools}
            for mcp_tool in mcp_tools:
                if getattr(mcp_tool, "name", None) not in existing_names:
                    tools.append(mcp_tool)

        subagent_dict = {
            "name": config.name,
            "description": config.description,
            "system_prompt": config.system_prompt,
            "tools": tools,
        }
        subagents.append(subagent_dict)

    logger.info("Created %d subagents from registry", len(subagents))
    return subagents


def _get_skill_sources(root_dir: Path) -> list[str]:
    """Discover skill source paths in priority order (last wins).

    AG3NT skill priority (later sources override earlier):
    1. Bundled: {repo}/skills/ - shipped with AG3NT
    2. Global: ~/.ag3nt/skills/ - user's personal skills (via /global-skills/ route)
    3. Workspace: ./.ag3nt/skills/ - project-specific skills

    Args:
        root_dir: The root directory (repo root or cwd)

    Returns:
        List of POSIX-style skill source paths for SkillsMiddleware.
        Note: /global-skills/ is a virtual path routed via CompositeBackend.
    """
    sources: list[str] = []

    # 1. Bundled skills (lowest priority) - repo's skills/ directory
    bundled = root_dir / "skills"
    if bundled.exists() and bundled.is_dir():
        sources.append("/skills/")

    # 2. Global skills (medium priority) - ~/.ag3nt/skills/
    # Accessed via /global-skills/ virtual route in CompositeBackend
    if _get_global_skills_path() is not None:
        sources.append("/global-skills/")

    # 3. Workspace skills (highest priority) - ./.ag3nt/skills/
    # If the workspace has a separate .ag3nt/skills folder, add it
    ag3nt_skills = root_dir / ".ag3nt" / "skills"
    if ag3nt_skills.exists() and ag3nt_skills.is_dir():
        sources.append("/.ag3nt/skills/")

    return sources


def _get_repo_root() -> Path:
    """Get the repository root directory.

    Returns:
        Path to the repo root (where skills/ directory is located)
    """
    # Start from this file and go up to find the repo root
    # This file is at: apps/agent/ag3nt_agent/deepagents_runtime.py
    # Repo root is 4 levels up
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent

    # Verify we found the right place by checking for skills/ directory
    if (repo_root / "skills").exists():
        return repo_root

    # Fallback to cwd if structure doesn't match
    return Path.cwd()


def _build_backend(repo_root: Path):
    """Build the backend for DeepAgents with multi-root support.

    Uses CompositeBackend to route:
    - /global-skills/ -> ~/.ag3nt/skills/ (user's global skills)
    - /user-data/ -> ~/.ag3nt/ (memory files and user data)

    Args:
        repo_root: The repository root directory

    Returns:
        Backend configured for file operations, skill discovery, and memory
    """
    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.filesystem import FilesystemBackend

    # Create default backend rooted at repo for bundled + workspace skills
    default_backend = FilesystemBackend(root_dir=repo_root, virtual_mode=False)

    # Build routes for CompositeBackend
    routes: dict = {}

    # Route for user data (memory, AGENTS.md, etc.) at ~/.ag3nt/
    user_data_path = _get_user_data_path()
    user_data_backend = FilesystemBackend(root_dir=user_data_path, virtual_mode=True)
    routes["/user-data/"] = user_data_backend

    # Route for workspace at ~/.ag3nt/workspace/ (agent's default working directory)
    # virtual_mode=True: all paths are sandboxed under workspace_path.
    # CompositeBackend strips the /workspace/ prefix before forwarding.
    workspace_path = user_data_path / "workspace"
    workspace_path.mkdir(exist_ok=True)
    workspace_backend = FilesystemBackend(root_dir=workspace_path, virtual_mode=True)
    routes["/workspace/"] = workspace_backend

    # Route for global skills at ~/.ag3nt/skills/ (if exists)
    global_skills_path = _get_global_skills_path()
    if global_skills_path is not None:
        global_backend = FilesystemBackend(root_dir=global_skills_path, virtual_mode=True)
        routes["/global-skills/"] = global_backend

    # Always use CompositeBackend to ensure user-data route is available
    return CompositeBackend(
        default=default_backend,
        routes=routes,
    )


def _get_system_prompt() -> str:
    """Return the AG3NT system prompt.

    Extracted to a standalone function so it can be shared between
    _build_agent() and the UI agent factory.
    """
    return """You are AG3NT (AP3X), a helpful AI assistant with advanced capabilities.

## Virtual File System

Use virtual paths for file operations:
- `/workspace/` — main working directory (default for new files)
- `/skills/` — available skills (read-only)
- `/user-data/` — persistent user data and memory

You also have full filesystem access via absolute paths (e.g., `C:\\Users\\...`). Use absolute paths when the user references files outside the workspace.

## Working with Files

- **Always `read_file` before `edit_file`**. Edits are rejected if the file was modified externally since your last read — re-read and retry.
- `edit_file` uses **fuzzy matching** (exact → line-trimmed → whitespace-normalized → indentation-flexible → block-anchor → context-aware). Include enough surrounding context to ensure a unique match.
- For 2+ edits in one file, prefer **`multi_edit`** over chaining `edit_file` — it's atomic (all-or-nothing) and avoids partial-failure bugs.
- Use `write_file` only for new files or complete rewrites.
- Every write is snapshot-tracked. Use `undo_last()`, `undo_to(id)`, or `unrevert()` to roll back — take risks confidently.

## Approaching Tasks

- For multi-step work, use **`write_todos`** to plan before acting, then mark items done as you go.
- Use **`batch`** to run multiple independent read-only calls concurrently (up to 25).
- When context is getting large, be concise — summarization will eventually compress older messages but keeping responses focused helps.

## Delegating to Subagents

Use the **`task`** tool to delegate to specialized subagents. Each gets a fresh context window, restricted tools, and returns a synthesized report.

| Subagent | When to use |
|----------|-------------|
| `researcher` | Web search, current events, fact-finding. **Use proactively** for any question needing up-to-date info. |
| `coder` | Focused programming — writing, debugging, executing code. |
| `reviewer` | Code review, security audit, quality analysis. |
| `planner` | Task decomposition, project planning, workflow design. |
| `browser` | Web automation, form filling, scraping dynamic sites. |
| `analyst` | Data analysis, statistics, visualization. |
| `writer` | Content creation, documentation, technical writing. |
| `memory` | Knowledge base search and management. |

**Delegate when**: the subtask is self-contained and benefits from a clean context. **Handle directly when**: the task is quick or needs your current context.

## Searching and Navigation

- **`glob_tool`** — find files by pattern (e.g., `**/*.py`). Results sorted by modification time.
- **`grep_tool`** — search file contents with regex. Modes: `files_with_matches` (default), `content`, `count`.
- **`codebase_search_tool`** — semantic natural-language code search (auto-indexed).
- **`lsp_tool`** — go-to-definition, references, hover, symbols, diagnostics. LSP diagnostics also auto-append after every edit.

## Shell Execution

Use `exec_command` with the right mode: **foreground** (default) for quick commands, **`background=True`** for long-running servers, **`yield_ms=N`** to auto-background if still running after N ms. Manage sessions with `process_tool` (list, poll, log, send_keys, kill).

## Browser

Use `browser_start_session(url)` to open pages in the Agent Browser for live viewing. Take `browser_screenshot()` to show the user what a page looks like. Always `browser_close()` when done.

## Git

Review changes with `git_status` and `git_diff` before staging. Use conventional commit format (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). Keep the first line under 72 characters, imperative mood.

## Scheduling

Use `schedule_reminder` for one-shot or recurring (cron) tasks. **Proactively offer** when the user mentions deadlines, periodic tasks, or time-based workflows.

## Communication

Use `ask_user` when you need clarification or a decision to proceed. For longer tasks, provide brief progress updates at reasonable intervals."""


def _wrap_tools_with_cache(all_tools: list) -> list:
    """Wrap cacheable tools with result caching and shell invalidation.

    For read-only tools (read_file, glob_tool, grep_tool, etc.), wraps the tool
    so results are cached on first call and returned from cache on subsequent
    identical calls.

    For exec_command (shell), invalidates the entire cache after execution.
    Write-triggered invalidation for vendor built-ins (write_file, edit_file)
    is handled implicitly via shell/exec_command full cache clear.

    Args:
        all_tools: List of LangChain tool objects.

    Returns:
        List of tools with caching wrappers applied where appropriate.
    """
    try:
        from ag3nt_agent.tool_cache import get_tool_cache, ToolResultCache
    except ImportError:
        logger.debug("tool_cache not available, skipping cache wrappers")
        return all_tools

    from functools import wraps

    cache = get_tool_cache()
    cacheable_names = ToolResultCache.CACHEABLE_TOOLS
    # Only exec_command is a registered tool; "shell" / "shell_tool" don't exist.
    shell_tools = {"exec_command"}

    wrapped: list = []
    for t in all_tools:
        tool_name = getattr(t, "name", None)
        if not tool_name:
            wrapped.append(t)
            continue

        if tool_name in cacheable_names:
            # Wrap read-only tool with cache check/set
            original_func = t.func

            def _make_cached_func(orig, name):
                """Create a closure that captures orig and name."""

                @wraps(orig)
                def cached_func(*args, **kwargs):
                    hit, value = cache.get(name, kwargs)
                    if hit:
                        logger.debug("Cache hit for %s", name)
                        return value
                    result = orig(*args, **kwargs)
                    cache.set(name, kwargs, result)
                    return result

                return cached_func

            t.func = _make_cached_func(original_func, tool_name)
            # Force LangGraph to route async (ainvoke) through the sync wrapper
            t.coroutine = None
            wrapped.append(t)

        elif tool_name in shell_tools:
            # Wrap shell tool with full cache invalidation
            original_func = t.func

            def _make_shell_func(orig):

                @wraps(orig)
                def shell_func(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    cache.invalidate()
                    return result

                return shell_func

            t.func = _make_shell_func(original_func)
            # Force LangGraph to route async (ainvoke) through the sync wrapper
            t.coroutine = None
            wrapped.append(t)

        else:
            wrapped.append(t)

    cached_count = sum(1 for t in wrapped if getattr(t, "name", None) in cacheable_names)
    if cached_count:
        logger.info("Tool cache wrappers applied to %d cacheable tools", cached_count)

    return wrapped


def _build_agent() -> CompiledStateGraph:
    """Build and return the DeepAgents agent graph.

    Returns:
        Configured DeepAgents graph

    Raises:
        ValueError: If required API keys are missing for the selected provider
    """
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import MemorySaver

    model = _create_model()

    # Initialise the fallback chain alongside the primary model so it's
    # ready for error-recovery use without paying extra startup cost later.
    get_fallback_chain()

    # Get repo root for skill discovery and file operations
    repo_root = _get_repo_root()

    # Discover available skill sources
    skill_sources = _get_skill_sources(repo_root)

    # Create backend with multi-root support (skills + user data)
    backend = _build_backend(repo_root)

    # Get memory sources (AGENTS.md, MEMORY.md)
    memory_sources = _get_memory_sources()

    # Load MCP tools early so sub-agents can also use them
    mcp_tools, mcp_server_tool_map = _load_mcp_tools()

    # Create sub-agents (Researcher, Coder) — pass MCP tools so they're available
    subagents = _create_subagents(mcp_tools=mcp_tools)

    # Get interrupt_on configuration for risky tools
    interrupt_on = _get_interrupt_on_config()

    # Persistent checkpoints via AsyncSqliteSaver (survives daemon restarts).
    # Falls back to MemorySaver if aiosqlite is not installed.
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = _get_user_data_path() / "checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # aiosqlite.connect() is sync — returns a lazy Connection proxy.
        # The actual sqlite connection opens on first async use.
        conn = aiosqlite.connect(str(db_path))
        checkpointer = AsyncSqliteSaver(conn)
        logger.info("Using AsyncSqliteSaver checkpointer at %s", db_path)
    except (ImportError, Exception) as exc:
        checkpointer = MemorySaver()
        logger.info("Using MemorySaver checkpointer (AsyncSqliteSaver unavailable: %s)", exc)

    # Create shell middleware for command execution
    # Uses ~/.ag3nt/workspace/ as the working directory
    workspace_path = _get_user_data_path() / "workspace"
    workspace_path.mkdir(exist_ok=True)
    shell_middleware = ShellMiddleware(
        workspace_root=str(workspace_path),
        timeout=60.0,  # 60 second timeout
        max_output_bytes=100_000,  # 100KB output limit
        enable_path_sandbox=False,  # Allow full filesystem access
    )

    # Start file watcher for external change detection
    try:
        from ag3nt_agent.file_watcher import FileWatcher
        from ag3nt_agent.file_tracker import FileTracker
        from ag3nt_agent.agent_config import FILE_WATCHER_DEBOUNCE

        def _on_file_change(file_path: str, event_type: str) -> None:
            """Invalidate FileTracker entries when files change externally."""
            try:
                tracker = FileTracker.get_instance()
                tracker.invalidate_all_sessions(file_path)
            except Exception:
                logger.debug("Failed to invalidate file tracker for %s", file_path)

        watcher = FileWatcher.get_instance()
        watcher.start(str(workspace_path), debounce_seconds=FILE_WATCHER_DEBOUNCE)
        watcher.on_change(_on_file_change)
        logger.info("File watcher started for workspace")
    except ImportError:
        logger.debug("watchdog not installed — file watcher disabled")

    # Initialize path protection (unrestricted — agent can access any path)
    path_protection_middleware = None
    try:
        from ag3nt_agent.tool_policy import PathProtection, PathProtectionMiddleware
        path_protection = PathProtection.get_instance()  # No workspace root = unrestricted
        path_protection_middleware = PathProtectionMiddleware(path_protection)
        logger.info("Path protection initialized (unrestricted filesystem access)")
    except ImportError:
        pass

    # Initialize LSP manager for post-edit diagnostics and code navigation
    try:
        from ag3nt_agent.lsp.manager import LspManager
        LspManager.get_instance(str(workspace_path))
        logger.info("LSP manager initialized for workspace")
    except ImportError:
        logger.debug("LSP manager not available")
    except Exception as e:
        logger.debug(f"LSP manager initialization failed: {e}")

    # Create summarization monitor middleware for observability
    # In DeepAgents 0.4.x, create_deep_agent auto-creates SummarizationMiddleware;
    # we only add a monitor that runs after it to record metrics.
    summarization_config = get_default_summarization_config()
    summarization_monitor_mw = create_summarization_middleware(
        config=summarization_config,
    )
    if summarization_monitor_mw:
        logger.info(
            f"Summarization monitor enabled: trigger={summarization_config.trigger.description}"
        )

    # MCP tools already loaded above (before sub-agent creation)

    # Load tool policy (if available) — used to skip denied tools before importing
    tool_policy = None
    try:
        from ag3nt_agent.tool_policy import ToolPolicyManager
        tool_policy = ToolPolicyManager()
    except ImportError:
        pass

    # Load all registry tools via declarative loader
    from ag3nt_agent.tool_registry import load_tools
    registry_tools = load_tools(tool_policy=tool_policy)

    # Get interactive tools (ask_user, etc.)
    interactive_tools = get_interactive_tools()

    # Import browser tools for main agent
    from ag3nt_agent.browser_tool import get_browser_tools
    browser_tools = get_browser_tools()

    # Deduplicate: if an MCP tool has the same name as a built-in, prefer the MCP version.
    builtin_tools = [internet_search, fetch_url, schedule_reminder] + browser_tools + registry_tools + interactive_tools
    builtin_names = {getattr(t, "name", None) for t in builtin_tools}
    mcp_names = {getattr(t, "name", None) for t in mcp_tools}
    dupes = builtin_names & mcp_names - {None}
    if dupes:
        logger.info(f"MCP tools override {len(dupes)} built-in tool(s): {dupes}")
        builtin_tools = [t for t in builtin_tools if getattr(t, "name", None) not in dupes]
    all_tools = builtin_tools + mcp_tools
    if mcp_tools:
        mcp_tool_names = [getattr(t, "name", "?") for t in mcp_tools]
        logger.info(f"Agent initialized with {len(mcp_tools)} MCP tool(s): {mcp_tool_names}")

    # Apply tool policy filter — exempt MCP tools (user-configured integrations)
    if tool_policy is not None:
        mcp_exempt = {getattr(t, "name", None) for t in mcp_tools} - {None}
        before_count = len(all_tools)
        all_tools = tool_policy.filter_tools(all_tools, exempt=mcp_exempt)
        if len(all_tools) < before_count:
            dropped = before_count - len(all_tools)
            logger.warning(f"Tool policy dropped {dropped} tool(s): {before_count} -> {len(all_tools)}")
        logger.info(f"Tool policy applied: {len(all_tools)} tools available")

    # Wrap cacheable tools with result caching and write-invalidation
    all_tools = _wrap_tools_with_cache(all_tools)

    # Log final tool list for debugging
    all_tool_names = [getattr(t, "name", "?") for t in all_tools]
    logger.info(f"Final tool list ({len(all_tools)}): {all_tool_names}")

    # System prompt — uses shared function for consistency with UI agent
    system_prompt = _get_system_prompt()

    # Append dynamic MCP tools section so the agent knows what MCP tools are available
    if mcp_tools and mcp_server_tool_map:
        mcp_config = _load_mcp_config() or {}
        mcp_servers_cfg = mcp_config.get("mcpServers", {})
        mcp_section = "\n\n## MCP Tools (External Integrations)\n\n"
        mcp_section += "You have tools loaded from external MCP servers. "
        mcp_section += "When the user refers to an MCP server by name, use the corresponding tools:\n\n"
        for srv_name, tool_names in mcp_server_tool_map.items():
            srv_cfg = mcp_servers_cfg.get(srv_name, {})
            display_name = srv_cfg.get("name", srv_name) if isinstance(srv_cfg, dict) else srv_name
            desc = srv_cfg.get("description", "") if isinstance(srv_cfg, dict) else ""
            mcp_section += f"**{display_name}** (server: `{srv_name}`)"
            if desc:
                mcp_section += f" — {desc}"
            mcp_section += "\n"
            for tn in tool_names:
                mcp_section += f"  - `{tn}`\n"
            mcp_section += "\n"
        mcp_section += (
            "IMPORTANT: When the user says 'use context7' or 'use playwright', you MUST call the "
            "corresponding MCP tools DIRECTLY yourself. Do NOT delegate MCP tool calls to sub-agents "
            "via the task tool — call `resolve-library-id` and `query-docs` (or other MCP tools) directly. "
            "MCP tools are available to you as regular tools alongside internet_search, fetch_url, etc.\n"
        )
        system_prompt += mcp_section

    # Strip any unpaired surrogates from the full system prompt (Windows subprocess
    # output, file reads, and MCP tool descriptions can introduce them).
    system_prompt = _sanitize_surrogates(system_prompt)

    # Build middleware list
    # Note: create_deep_agent already adds TodoListMiddleware internally
    # so we only add AG3NT-specific middleware here to avoid duplicates
    planning_middleware = PlanningMiddleware(yolo_mode=_is_yolo_mode())
    skill_trigger_middleware = SkillTriggerMiddleware(planning_middleware=planning_middleware)

    # Context budget tracker — inject report when usage is elevated
    context_budget = None
    try:
        from ag3nt_agent.context_budget import ContextBudgetTracker, BudgetStatus

        context_budget = ContextBudgetTracker()

        def _budget_report_if_elevated() -> str:
            """Return budget report only when status is YELLOW or RED."""
            if context_budget.status() == BudgetStatus.GREEN:
                return ""
            return context_budget.budget_report()
    except Exception:
        logger.debug("ContextBudgetTracker not available", exc_info=True)

    turn_context_middleware = TurnContextMiddleware(
        identity_loader=IdentityLoader(),
        memory_search_fn=search_memory,
        skills_metadata_fn=skill_trigger_middleware.get_skills_metadata,
        context_budget_fn=_budget_report_if_elevated if context_budget else None,
    )
    agent_guard = AgentGuardMiddleware()
    middleware_list = [
        agent_guard,              # Doom loop + steps limit + output truncation
        turn_context_middleware,  # Identity + memory context (runs every turn)
        planning_middleware,  # Plan mode enforcement
        shell_middleware,  # Shell execution capability
        skill_trigger_middleware,  # Skill trigger matching
    ]
    if path_protection_middleware:
        middleware_list.append(path_protection_middleware)

    # Skills hot-reload: invalidate trigger + metadata caches on SKILL.md changes
    try:
        from ag3nt_agent.skills_watcher import SkillsWatcher

        skills_watcher = SkillsWatcher()
        skills_watcher.on_change(skill_trigger_middleware.invalidate_triggers)
        skills_watcher.start()
    except Exception:
        logger.debug("SkillsWatcher not available", exc_info=True)

    # Safety hooks middleware (wraps tool calls with PRE/POST hooks)
    safety_hook_middleware = None
    try:
        from ag3nt_agent.hooks import PhaseHookManager
        from ag3nt_agent.hooks.middleware import HookMiddleware
        from ag3nt_agent.hooks.agent_middleware import SafetyHookAgentMiddleware

        hook_manager = PhaseHookManager()
        hook_manager.start()
        hook_mw = HookMiddleware(hook_manager)
        safety_hook_middleware = SafetyHookAgentMiddleware(hook_mw)

        # Register safety hooks (protect_core, block_danger, compile_check)
        try:
            from ag3nt_agent.hooks.safety import register_safety_hooks
            register_safety_hooks(hook_manager)
        except Exception:
            pass
    except ImportError:
        logger.debug("Safety hooks middleware not available")

    if safety_hook_middleware:
        middleware_list.append(safety_hook_middleware)

    # Add summarization monitor (runs after upstream SummarizationMiddleware)
    if summarization_monitor_mw:
        middleware_list.append(summarization_monitor_mw)

    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        skills=skill_sources if skill_sources else None,
        memory=memory_sources if memory_sources else None,
        subagents=subagents if subagents else None,
        tools=all_tools,  # Custom AG3NT tools + MCP tools
        middleware=middleware_list,
        backend=backend,
        interrupt_on=interrupt_on if interrupt_on else None,
        checkpointer=checkpointer,
    )
    return agent


def get_agent() -> CompiledStateGraph:
    """Get or create the singleton agent instance.

    If AG3NT_USE_AGENT_POOL=true, this uses the agent pool for pre-warmed
    instances. Otherwise, returns a singleton agent.

    For pooled usage, prefer using acquire_agent() and release_agent()
    directly for proper lifecycle management.
    """
    global _agent
    if _use_agent_pool:
        # Use pool but don't track release - for compatibility
        from ag3nt_agent.agent_pool import get_agent_pool
        pool = get_agent_pool()
        if not pool._initialized:
            pool.initialize()
        entry = pool.acquire()
        return entry.agent
    else:
        if _agent is None:
            _agent = _build_agent()
        return _agent


def acquire_agent() -> tuple[CompiledStateGraph, Any]:
    """Acquire an agent from the pool.

    Returns a tuple of (agent, pool_entry). The pool_entry must be
    passed to release_agent() when done to return it to the pool.

    If pooling is disabled, returns (singleton_agent, None).

    Usage:
        agent, entry = acquire_agent()
        try:
            result = await run_turn_with_agent(agent, ...)
        finally:
            release_agent(entry)
    """
    if _use_agent_pool:
        from ag3nt_agent.agent_pool import get_agent_pool
        pool = get_agent_pool()
        if not pool._initialized:
            pool.initialize()
        entry = pool.acquire()
        return entry.agent, entry
    else:
        return get_agent(), None


async def acquire_agent_async() -> tuple[CompiledStateGraph, Any]:
    """Acquire an agent from the pool asynchronously.

    Same as acquire_agent() but uses async pool initialization.
    """
    if _use_agent_pool:
        from ag3nt_agent.agent_pool import get_agent_pool
        pool = get_agent_pool()
        if not pool._initialized:
            await pool.initialize_async()
        entry = await pool.acquire_async()
        return entry.agent, entry
    else:
        return get_agent(), None


def release_agent(entry: Any) -> None:
    """Release an agent back to the pool.

    Args:
        entry: The pool entry returned from acquire_agent().
               If None (non-pooled mode), this is a no-op.
    """
    if entry is not None and _use_agent_pool:
        from ag3nt_agent.agent_pool import get_agent_pool
        pool = get_agent_pool()
        pool.release(entry)


def get_pool_stats() -> dict[str, Any] | None:
    """Get agent pool statistics.

    Returns None if pooling is disabled.
    """
    if not _use_agent_pool:
        return None
    from ag3nt_agent.agent_pool import get_agent_pool
    return get_agent_pool().get_stats().to_dict()


def _extract_interrupt_info(result: dict[str, Any]) -> dict[str, Any] | None:
    """Extract interrupt information from agent result.

    Args:
        result: The result from agent.invoke()

    Returns:
        Dict with interrupt details or None if no interrupt.
        For tool approval: {"interrupt_id", "pending_actions", "action_count"}
        For user question: {"interrupt_id", "type": "user_question", "question", "options", "allow_custom"}
    """
    if "__interrupt__" not in result:
        return None

    interrupts = result["__interrupt__"]
    if not interrupts:
        return None

    # Collect action requests from ALL interrupts
    interrupt_ids: list[str] = []
    all_action_requests: list[dict[str, Any]] = []
    for interrupt in interrupts:
        interrupt_ids.append(str(interrupt.id))
        reqs = interrupt.value.get("action_requests", []) if isinstance(interrupt.value, dict) else []
        all_action_requests.extend(reqs)

    # Use first interrupt's ID as the primary (for backwards compat)
    interrupt_id = interrupt_ids[0] if interrupt_ids else ""
    action_requests = all_action_requests

    # Check if this is a user question (ask_user tool)
    if action_requests and len(action_requests) == 1:
        action = action_requests[0]
        if action.get("name") == "ask_user":
            args = action.get("args", {})
            logger.info(f"User question interrupt: {args.get('question')}")
            return {
                "interrupt_id": interrupt_id,
                "type": "user_question",
                "question": args.get("question", ""),
                "options": args.get("options", []),
                "allow_custom": args.get("allow_custom", True),
            }

    # Otherwise, handle as tool approval interrupt
    # Format pending actions for display
    pending_actions = []
    for action in action_requests:
        tool_name = action.get("name", "unknown")
        tool_args = action.get("args", {})
        description = _format_tool_description({"name": tool_name, "args": tool_args})
        pending_actions.append({
            "tool_name": tool_name,
            "args": tool_args,
            "description": description,
        })

    logger.info(f"Interrupt detected: {len(pending_actions)} actions pending approval")
    for action in pending_actions:
        logger.info(f"  - {action['tool_name']}: {action['args']}")

    return {
        "interrupt_id": interrupt_id,
        "interrupt_ids": interrupt_ids,
        "pending_actions": pending_actions,
        "action_count": len(pending_actions),
    }


def _extract_response(result: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Extract response text and events from agent result.

    Args:
        result: The result from agent.invoke()

    Returns:
        Tuple of (response_text, events)
    """
    response_messages = result.get("messages", [])
    events: list[dict[str, Any]] = []
    response_text = ""

    for msg in reversed(response_messages):
        if isinstance(msg, AIMessage):
            # Extract text content
            if isinstance(msg.content, str):
                response_text = msg.content
            elif isinstance(msg.content, list):
                # Handle content blocks
                text_parts = []
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                response_text = "\n".join(text_parts)

            # Extract tool calls as events
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    events.append({
                        "tool_name": tc.get("name", "unknown"),
                        "input": tc.get("args", {}),
                        "status": "completed",
                    })
            break

    return response_text or "No response generated.", events


def _extract_usage_info(result: dict[str, Any]) -> dict[str, Any]:
    """Extract token usage information from agent result.

    This aggregates usage across all LLM calls in the turn,
    which is then reported to the Gateway for tracking.

    Args:
        result: The agent's result dictionary containing messages

    Returns:
        Dict with usage info: input_tokens, output_tokens, model, provider
    """
    provider, model_name = _get_model_config()
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "model": model_name,
        "provider": provider,
    }

    messages = result.get("messages", [])

    for msg in messages:
        # LangChain messages may have usage_metadata or response_metadata
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            meta = msg.usage_metadata
            usage["input_tokens"] += meta.get("input_tokens", 0)
            usage["output_tokens"] += meta.get("output_tokens", 0)
        elif hasattr(msg, "response_metadata") and msg.response_metadata:
            meta = msg.response_metadata
            if "usage" in meta:
                u = meta["usage"]
                usage["input_tokens"] += u.get("input_tokens", u.get("prompt_tokens", 0))
                usage["output_tokens"] += u.get("output_tokens", u.get("completion_tokens", 0))

    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _maybe_compact_history(
    agent: CompiledStateGraph,
    config: dict[str, Any],
    session_id: str,
) -> None:
    """Run per-turn compaction check on the checkpoint message history.

    If the message count or token count exceeds the configured thresholds,
    the CompactionMiddleware pipeline (masking, flush, pruning, progressive
    summarization) is applied and the checkpoint state is updated so the
    *next* turn starts with a compacted history.

    This is intentionally a no-op when the ``compaction_middleware`` module
    is unavailable or when any step raises an exception — the agent should
    never fail because of compaction bookkeeping.

    NOTE: When SummarizationMiddleware is active (the default), this function
    is skipped to avoid dual compaction that strips too much context.
    Summarization already handles context window management at the
    middleware layer (before_model hook).
    """
    # Skip compaction when summarization middleware is already managing context.
    # Dual compaction causes premature context loss during file-heavy tasks.
    summarization_config = get_default_summarization_config()
    if summarization_config.enabled:
        return

    try:
        from ag3nt_agent.compaction_middleware import get_compaction_middleware

        compaction_mw = get_compaction_middleware()

        # Get current message history from checkpointer state
        checkpoint_state = agent.get_state(config)
        current_messages = checkpoint_state.values.get("messages", [])

        if compaction_mw.should_compact(current_messages):
            compacted_msgs, metrics = compaction_mw.compact(
                current_messages, session_id=session_id
            )
            if metrics.triggered:
                # Update checkpoint state with compacted messages
                agent.update_state(config, {"messages": compacted_msgs})
                logger.info(
                    "Compaction applied: %d->%d messages, %d->%d tokens",
                    metrics.messages_before,
                    metrics.messages_after,
                    metrics.tokens_before,
                    metrics.tokens_after,
                )
    except ImportError:
        pass  # CompactionMiddleware not available
    except Exception as exc:
        logger.debug("Compaction check failed: %s", exc, exc_info=True)


def run_turn(
    session_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single turn of conversation with the agent.

    Args:
        session_id: Unique identifier for the session/conversation.
        text: The user's input text.
        metadata: Optional metadata for the turn.

    Returns:
        A dict containing:
            - session_id: The session ID
            - text: The agent's response text
            - events: List of tool call events (if any)
            - interrupt: Dict with interrupt details (if paused for approval)
    """
    agent = get_agent()

    # Reset guard step counter for this turn
    AgentGuardMiddleware.reset_steps(session_id)

    # Set session context for deep reasoning tool
    try:
        from ag3nt_agent.deep_reasoning import set_current_session_id
        set_current_session_id(session_id)
    except ImportError:
        pass

    # Build the input messages
    messages = [HumanMessage(content=text)]

    # Configure the run with session-specific thread_id for checkpointing
    # Pass metadata through so middleware (PlanningMiddleware, etc.) can read it
    config: dict[str, Any] = {
        "configurable": {
            "thread_id": session_id,
        },
    }
    if metadata:
        config["metadata"] = metadata

    # Invoke the agent
    try:
        result = agent.invoke({"messages": messages}, config=config)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return {
            "session_id": session_id,
            "text": f"Error: {e!s}",
            "events": [],
        }

    # Check for interrupt (approval required)
    interrupt_info = _extract_interrupt_info(result)
    if interrupt_info:
        # Store interrupt IDs for resume (thread-safe)
        with _pending_interrupt_ids_lock:
            _pending_interrupt_ids[session_id] = interrupt_info.get("interrupt_ids", [interrupt_info["interrupt_id"]])
        # Format the pending actions for the user
        action_text = "\n\n".join(
            action["description"] for action in interrupt_info["pending_actions"]
        )
        return {
            "session_id": session_id,
            "text": f"⏸️ **Approval Required**\n\nI need your permission to proceed with the following action(s):\n\n{action_text}\n\nReply with **approve** or **reject**.",
            "events": [],
            "interrupt": interrupt_info,
        }

    # No interrupt — clear any stale pending IDs
    with _pending_interrupt_ids_lock:
        _pending_interrupt_ids.pop(session_id, None)

    # Extract response
    response_text, events = _extract_response(result)

    # Extract usage information from response metadata
    usage = _extract_usage_info(result)

    # Emit turn.completed for compaction tracking
    response_chars = len(response_text) + len(text)
    for ev in events:
        response_chars += len(str(ev.get("output", "")))
    _emit_turn_completed(session_id=session_id, char_count=response_chars)

    # Per-turn compaction: compact history if context is too large
    _maybe_compact_history(agent, config, session_id)

    # Auto-index memory insights from conversation
    try:
        from ag3nt_agent.memory_auto_indexer import MemoryAutoIndexer

        if not hasattr(run_turn, "_memory_indexer"):
            run_turn._memory_indexer = MemoryAutoIndexer()
        checkpoint_state = agent.get_state(config)
        run_turn._memory_indexer.maybe_index(
            checkpoint_state.values.get("messages", []),
            session_id=session_id,
        )
    except Exception:
        logger.debug("Memory auto-indexer failed", exc_info=True)

    return {
        "session_id": session_id,
        "text": response_text,
        "events": events,
        "usage": usage,
    }


def resume_turn(
    session_id: str,
    decisions: list[dict[str, str]],
) -> dict[str, Any]:
    """Resume an interrupted turn after user approval/rejection.

    Args:
        session_id: The session ID of the interrupted turn.
        decisions: List of decisions, each with {"type": "approve"} or {"type": "reject"}

    Returns:
        A dict containing:
            - session_id: The session ID
            - text: The agent's response text
            - events: List of tool call events (if any)
            - interrupt: Dict with interrupt details (if another approval is needed)
    """
    agent = get_agent()

    # Log the decision
    decision_types = [d.get("type", "unknown") for d in decisions]
    logger.info(f"Resuming session {session_id} with decisions: {decision_types}")

    # Configure the run with session-specific thread_id
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    # Build resume Commands — one per pending interrupt, tagged with its ID
    with _pending_interrupt_ids_lock:
        stored_ids = _pending_interrupt_ids.pop(session_id, [])
    if len(stored_ids) > 1:
        # Multiple pending interrupts — one Command per interrupt
        resume_input: Any = [
            Command(resume={"decisions": decisions})
            for _iid in stored_ids
        ]
    elif stored_ids:
        resume_input = Command(resume={"decisions": decisions})
    else:
        resume_input = Command(resume={"decisions": decisions})

    try:
        result = agent.invoke(resume_input, config=config)
    except Exception as e:
        logger.error(f"Resume error: {e}")
        return {
            "session_id": session_id,
            "text": f"Error resuming: {e!s}",
            "events": [],
        }

    # Check for another interrupt
    interrupt_info = _extract_interrupt_info(result)
    if interrupt_info:
        # Store new interrupt IDs for next resume (thread-safe)
        with _pending_interrupt_ids_lock:
            _pending_interrupt_ids[session_id] = interrupt_info.get("interrupt_ids", [interrupt_info["interrupt_id"]])
        action_text = "\n\n".join(
            action["description"] for action in interrupt_info["pending_actions"]
        )
        return {
            "session_id": session_id,
            "text": f"⏸️ **Approval Required**\n\nI need your permission to proceed with the following action(s):\n\n{action_text}\n\nReply with **approve** or **reject**.",
            "events": [],
            "interrupt": interrupt_info,
        }

    # No interrupt — clear any stale pending IDs
    with _pending_interrupt_ids_lock:
        _pending_interrupt_ids.pop(session_id, None)

    # Extract response
    response_text, events = _extract_response(result)

    # Extract usage information from response metadata
    usage = _extract_usage_info(result)

    # Per-turn compaction: compact history if context is too large
    _maybe_compact_history(agent, config, session_id)

    return {
        "session_id": session_id,
        "text": response_text,
        "events": events,
        "usage": usage,
    }


# =============================================================================
# AUTONOMOUS SYSTEM INTEGRATION
# =============================================================================
# The autonomous system provides event-driven goal execution with:
# - Event bus for routing events to handlers
# - Goal manager for YAML-based goal configurations
# - Decision engine for act/ask decisions based on confidence
# - Learning engine for tracking action outcomes

_autonomous_runtime_legacy: "AutonomousRuntime | None" = None


class AutonomousRuntime:
    """Coordinates the autonomous event-driven system.

    Ties together EventBus, GoalManager, DecisionEngine, and LearningEngine
    to provide autonomous goal execution with human-in-the-loop controls.

    Usage:
        runtime = AutonomousRuntime()
        await runtime.start()

        # Publish events from monitors/triggers
        await runtime.publish_event("http_check", "monitor", {"status": 500})

        # Graceful shutdown
        await runtime.stop()
    """

    def __init__(self, goals_dir: Path | None = None):
        """Initialize the autonomous runtime.

        Args:
            goals_dir: Directory containing goal YAML files.
                       Defaults to config/goals/ in workspace.
        """
        from ag3nt_agent.autonomous.event_bus import EventBus
        from ag3nt_agent.autonomous.goal_manager import GoalManager
        from ag3nt_agent.autonomous.decision_engine import DecisionEngine, DecisionConfig
        from ag3nt_agent.autonomous.learning_engine import LearningEngine

        # Determine goals directory
        if goals_dir is None:
            workspace = os.environ.get("AG3NT_WORKSPACE", os.getcwd())
            goals_dir = Path(workspace) / "config" / "goals"

        # Initialize components
        self.event_bus = EventBus()
        self.goal_manager = GoalManager(config_dir=goals_dir if goals_dir.exists() else None)
        self.learning_engine = LearningEngine()
        self.decision_engine = DecisionEngine(
            learning_engine=self.learning_engine,
            config=DecisionConfig()
        )

        # Subscribe to event bus
        self.event_bus.subscribe(self._handle_event)

        self._running = False
        logger.info("Autonomous runtime initialized")

    async def start(self) -> None:
        """Start the autonomous runtime."""
        if self._running:
            return
        await self.event_bus.start()
        self._running = True
        logger.info("Autonomous runtime started")

    async def stop(self) -> None:
        """Stop the autonomous runtime."""
        if not self._running:
            return
        await self.event_bus.stop()
        self._running = False
        logger.info("Autonomous runtime stopped")

    @property
    def is_running(self) -> bool:
        """Check if the runtime is active."""
        return self._running

    async def publish_event(
        self,
        event_type: str,
        source: str,
        payload: dict[str, Any] | None = None,
        priority: str = "MEDIUM"
    ) -> bool:
        """Publish an event to the autonomous system.

        Args:
            event_type: Type of event (e.g., "http_check", "file_change")
            source: Source identifier (e.g., "http_monitor:prod-api")
            payload: Event data
            priority: Event priority (CRITICAL, HIGH, MEDIUM, LOW)

        Returns:
            True if event was accepted, False if deduplicated or queue full
        """
        from ag3nt_agent.autonomous.event_bus import Event, EventPriority

        priority_enum = EventPriority[priority.upper()]
        event = Event(
            event_type=event_type,
            source=source,
            payload=payload or {},
            priority=priority_enum
        )
        return await self.event_bus.publish(event)

    async def _handle_event(self, event) -> None:
        """Handle an event by finding and executing matching goals."""
        # Find matching goals
        matching_goals = self.goal_manager.find_matching_goals(event)

        if not matching_goals:
            logger.debug(f"No goals matched event: {event.event_type}")
            return

        # Process each matching goal
        for goal in matching_goals:
            await self._process_goal(goal, event)

    async def _process_goal(self, goal, event) -> None:
        """Process a single goal for an event."""
        from ag3nt_agent.autonomous.decision_engine import DecisionType

        # Get decision from decision engine
        decision = await self.decision_engine.evaluate(goal, event)

        logger.info(
            f"Decision for goal '{goal.name}': {decision.decision_type.value} "
            f"(confidence: {decision.confidence.score:.0%})"
        )

        if decision.decision_type == DecisionType.ACT:
            # Execute autonomously
            success = await self._execute_goal(goal, event)
            self.decision_engine.record_outcome(goal.id, success)
            await self.learning_engine.record_outcome(
                action_type=goal.action.type.value,
                context=f"Goal: {goal.name}",
                success=success
            )
        elif decision.decision_type == DecisionType.ASK:
            # Queue for human approval (integrate with HITL)
            logger.info(f"Goal '{goal.name}' requires approval: {decision.reason}")
            # TODO: Integrate with Gateway approval queue
        elif decision.decision_type == DecisionType.ESCALATE:
            logger.warning(f"Goal '{goal.name}' escalated: {decision.reason}")
        elif decision.decision_type == DecisionType.REJECT:
            logger.info(f"Goal '{goal.name}' rejected: {decision.reason}")

    async def _execute_goal(self, goal, event) -> bool:
        """Execute a goal's action."""
        import asyncio

        from ag3nt_agent.autonomous.goal_manager import ActionType

        # Render action with event data
        action = goal.action.render(event)
        goal.record_execution()

        try:
            if action.type == ActionType.SHELL:
                # Execute shell command
                process = await asyncio.create_subprocess_shell(
                    action.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=action.timeout_seconds
                )
                success = process.returncode == 0
                logger.info(f"Shell action completed: {action.command[:50]}... (rc={process.returncode})")
                return success

            elif action.type == ActionType.AGENT:
                # Delegate to agent
                result = run_turn(
                    session_id=f"autonomous-{goal.id}-{event.event_id}",
                    text=action.agent_prompt,
                    metadata={"autonomous": True, "goal_id": goal.id}
                )
                return "error" not in result.get("text", "").lower()

            elif action.type == ActionType.NOTIFY:
                # Send notification (placeholder - integrate with channels)
                logger.info(f"Notification: {action.message}")
                return True

            elif action.type == ActionType.HTTP:
                # Make HTTP request
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=action.method,
                        url=action.url,
                        json=action.body,
                        timeout=aiohttp.ClientTimeout(total=action.timeout_seconds)
                    ) as response:
                        return 200 <= response.status < 300

            return False

        except asyncio.TimeoutError:
            logger.error(f"Action timed out for goal '{goal.name}'")
            return False
        except Exception as e:
            logger.error(f"Action failed for goal '{goal.name}': {e}")
            return False

    def get_status(self) -> dict[str, Any]:
        """Get autonomous system status."""
        return {
            "running": self._running,
            "event_bus": self.event_bus.get_metrics(),
            "goals": self.goal_manager.get_status(),
            "learning": self.learning_engine.get_stats() if hasattr(self.learning_engine, "get_stats") else {}
        }

    def add_goal(self, goal_config: dict[str, Any]) -> None:
        """Add a goal programmatically."""
        from ag3nt_agent.autonomous.goal_manager import Goal
        goal = Goal.from_dict(goal_config)
        self.goal_manager.add_goal(goal)


def get_legacy_autonomous_runtime(goals_dir: Path | None = None) -> AutonomousRuntime:
    """Get or create the legacy autonomous runtime singleton.

    Args:
        goals_dir: Optional goals configuration directory

    Returns:
        The AutonomousRuntime instance
    """
    global _autonomous_runtime_legacy
    if _autonomous_runtime_legacy is None:
        _autonomous_runtime_legacy = AutonomousRuntime(goals_dir=goals_dir)
    return _autonomous_runtime_legacy


# =============================================================================
# AUTONOMOUS SYSTEM BOOTSTRAP
# =============================================================================
# Full autonomous system lifecycle management. Wires together all subsystems:
#   EventBus, PhaseHookManager, LoopManager, IntelligenceWorker,
#   ConsolidationLoop, SessionTrustTracker, and the legacy AutonomousRuntime.
#
# Each subsystem is imported lazily and wrapped in try/except so that
# partial startup is possible when optional dependencies are missing.
# =============================================================================

_autonomous_runtime: dict[str, Any] | None = None


async def start_autonomous_system(config: dict | None = None) -> dict:
    """Initialize and start all autonomous subsystems.

    Bootstrap order:
    1. EventBus (central event routing)
    2. PhaseHookManager + safety hooks
    3. LoopManager (all reactive loops)
    4. IntelligenceWorker (daemon thread with MetaLoop)
    5. ConsolidationLoop (end-of-session memory extraction)
    6. SessionTrustTracker (tiered approval)
    7. Legacy AutonomousRuntime (goal-based event processing)

    Args:
        config: The ``autonomous`` section from default-config.yaml.
                If None, sensible defaults are used.

    Returns:
        A dict of ``{"component_name": instance}`` for all started subsystems.
    """
    global _autonomous_runtime

    if _autonomous_runtime is not None:
        logger.debug("Autonomous system already running — returning existing runtime")
        return _autonomous_runtime

    cfg = config or {}
    runtime: dict[str, Any] = {}

    # ── 1. EventBus ──────────────────────────────────────────────────────
    bus = None
    try:
        from ag3nt_agent.autonomous.event_bus import get_event_bus

        bus = get_event_bus()
        await bus.start()
        runtime["event_bus"] = bus
        logger.info("Autonomous bootstrap: EventBus created and started")
    except Exception as exc:
        logger.error("Autonomous bootstrap: failed to create EventBus: %s", exc)

    # ── 2. PhaseHookManager + safety hooks ─────────────────────────────
    hook_manager = None
    try:
        from ag3nt_agent.hooks import PhaseHookManager

        hook_manager = PhaseHookManager()
        hook_manager.start()
        runtime["hook_manager"] = hook_manager
        logger.info("Autonomous bootstrap: PhaseHookManager started")

        # Register safety hooks (protect_core, block_danger, compile_check)
        try:
            from ag3nt_agent.hooks.safety import register_safety_hooks
            register_safety_hooks(hook_manager)
            logger.info("Autonomous bootstrap: safety hooks registered")
        except Exception as exc:
            logger.warning("Autonomous bootstrap: safety hooks unavailable: %s", exc)
    except Exception as exc:
        logger.warning("Autonomous bootstrap: PhaseHookManager unavailable: %s", exc)

    # ── 3. LoopManager ───────────────────────────────────────────────────
    if bus is not None:
        try:
            from ag3nt_agent.loops import LoopManager, LoopManagerConfig

            failover_cfg = cfg.get("provider_failover", {})
            loop_config = LoopManagerConfig(
                quality_gate_enabled=cfg.get("quality_gate", {}).get("enabled", True),
                recovery_enabled=cfg.get("autofix", {}).get("enabled", True),
                failover_enabled=failover_cfg.get("enabled", True),
            )
            loop_manager = LoopManager(bus, config=loop_config)
            await loop_manager.start_all()
            runtime["loop_manager"] = loop_manager
            logger.info("Autonomous bootstrap: LoopManager started (%s)", loop_manager.loop_names)

            # Wire compaction trigger into context tools so check_context_budget
            # and compact_now can query/control the compaction system.
            try:
                from ag3nt_agent.context_tools import set_compaction_trigger

                ct = loop_manager._loops.get("CompactionTrigger")
                if ct is not None:
                    set_compaction_trigger(ct)
            except ImportError:
                pass
        except Exception as exc:
            logger.warning("Autonomous bootstrap: LoopManager unavailable: %s", exc)

    # ── 3b. AutofixEngine ──────────────────────────────────────────────
    if bus is not None:
        try:
            from ag3nt_agent.autofix.engine import AutofixEngine

            autofix_engine = AutofixEngine(bus)
            await autofix_engine.start()
            runtime["autofix_engine"] = autofix_engine
            logger.info("Autonomous bootstrap: AutofixEngine started")
        except Exception as exc:
            logger.warning("Autonomous bootstrap: AutofixEngine unavailable: %s", exc)

    # ── 4. IntelligenceWorker (daemon thread) ────────────────────────────
    intel_cfg = cfg.get("intelligence", {})
    if intel_cfg.get("enabled", True) and bus is not None:
        try:
            from ag3nt_agent.intelligence.metaloop import MetaLoop
            from ag3nt_agent.intelligence.worker import IntelligenceWorker

            metaloop = MetaLoop(
                bus=bus,
                tick_interval_s=float(intel_cfg.get("tick_interval_s", 10)),
            )

            # Register available intelligence loops
            _register_intelligence_loops(metaloop, bus)

            worker = IntelligenceWorker(metaloop)
            worker.start()
            runtime["metaloop"] = metaloop
            runtime["intelligence_worker"] = worker
            logger.info("Autonomous bootstrap: IntelligenceWorker started")
        except Exception as exc:
            logger.warning("Autonomous bootstrap: IntelligenceWorker unavailable: %s", exc)

    # ── 5. ConsolidationLoop ─────────────────────────────────────────────
    if bus is not None:
        try:
            from ag3nt_agent.intelligence.consolidation import ConsolidationLoop

            consolidation = ConsolidationLoop(bus)
            await consolidation.start()
            runtime["consolidation_loop"] = consolidation
            logger.info("Autonomous bootstrap: ConsolidationLoop started")
        except Exception as exc:
            logger.warning("Autonomous bootstrap: ConsolidationLoop unavailable: %s", exc)

    # ── 6. SessionTrustTracker (tiered approval) ─────────────────────────
    approval_cfg = cfg.get("approval", {})
    if approval_cfg.get("tiered", True):
        try:
            from ag3nt_agent.approval.trust import SessionTrustTracker

            threshold = int(approval_cfg.get("trust_escalation_threshold", 3))
            trust_tracker = SessionTrustTracker(threshold=threshold)
            runtime["trust_tracker"] = trust_tracker
            logger.info(
                "Autonomous bootstrap: SessionTrustTracker created (threshold=%d)",
                threshold,
            )
        except Exception as exc:
            logger.warning("Autonomous bootstrap: SessionTrustTracker unavailable: %s", exc)

    # ── 7. Legacy AutonomousRuntime (goal-based event processing) ────────
    try:
        legacy = get_legacy_autonomous_runtime()
        await legacy.start()
        runtime["legacy_runtime"] = legacy
        logger.info("Autonomous bootstrap: legacy AutonomousRuntime started")
    except Exception as exc:
        logger.warning("Autonomous bootstrap: legacy AutonomousRuntime unavailable: %s", exc)


    _autonomous_runtime = runtime
    logger.info(
        "Autonomous system started: %d/%d subsystems active — %s",
        len(runtime),
        7,
        ", ".join(runtime.keys()),
    )
    return runtime


def _register_intelligence_loops(metaloop: Any, bus: Any) -> None:
    """Register available intelligence loops with the MetaLoop.

    Each loop is imported lazily; failures are logged and skipped.
    """
    # Sentinel — codebase knowledge graph
    try:
        from ag3nt_agent.intelligence.sentinel import SentinelLoop

        sentinel = SentinelLoop(bus)
        metaloop.register_loop("sentinel", sentinel, budget=0.15)
    except Exception as exc:
        logger.debug("Intelligence loop 'sentinel' unavailable: %s", exc)

    # Self-Healing — failure pattern detection and remediation
    try:
        from ag3nt_agent.intelligence.healing import SelfHealingIntelligenceLoop

        healing = SelfHealingIntelligenceLoop(bus)
        metaloop.register_loop("healing", healing, budget=0.10)
    except Exception as exc:
        logger.debug("Intelligence loop 'healing' unavailable: %s", exc)

    # Quality — output quality analysis
    try:
        from ag3nt_agent.intelligence.quality import QualityGuardianLoop

        quality_loop = QualityGuardianLoop(bus)
        metaloop.register_loop("quality", quality_loop, budget=0.10)
    except Exception as exc:
        logger.debug("Intelligence loop 'quality' unavailable: %s", exc)

    # Exploration — proactive codebase exploration
    try:
        from ag3nt_agent.intelligence.exploration import ParallelExplorationLoop

        exploration = ParallelExplorationLoop(bus)
        metaloop.register_loop("exploration", exploration, budget=0.05)
    except Exception as exc:
        logger.debug("Intelligence loop 'exploration' unavailable: %s", exc)


async def stop_autonomous_system() -> None:
    """Gracefully shutdown all autonomous subsystems.

    Shutdown order (reverse of startup):
    1. Legacy AutonomousRuntime
    2. SessionTrustTracker (no teardown needed)
    3. ConsolidationLoop (final memory extraction)
    4. IntelligenceWorker
    5. LoopManager
    6. PhaseHookManager
    7. EventBus (no teardown needed)
    """
    global _autonomous_runtime

    if _autonomous_runtime is None:
        logger.debug("Autonomous system not running — nothing to stop")
        return

    runtime = _autonomous_runtime

    # ── 1. Legacy AutonomousRuntime ──────────────────────────────────────
    legacy = runtime.get("legacy_runtime")
    if legacy is not None:
        try:
            await legacy.stop()
            logger.info("Autonomous shutdown: legacy AutonomousRuntime stopped")
        except Exception as exc:
            logger.error("Autonomous shutdown: legacy AutonomousRuntime error: %s", exc)

    # ── 2. ConsolidationLoop ─────────────────────────────────────────────
    consolidation = runtime.get("consolidation_loop")
    if consolidation is not None:
        try:
            await consolidation.stop()
            logger.info("Autonomous shutdown: ConsolidationLoop stopped")
        except Exception as exc:
            logger.error("Autonomous shutdown: ConsolidationLoop error: %s", exc)

    # ── 3. IntelligenceWorker ────────────────────────────────────────────
    worker = runtime.get("intelligence_worker")
    if worker is not None:
        try:
            worker.stop()
            logger.info("Autonomous shutdown: IntelligenceWorker stopped")
        except Exception as exc:
            logger.error("Autonomous shutdown: IntelligenceWorker error: %s", exc)

    # ── 3b. AutofixEngine ───────────────────────────────────────────────
    autofix_engine = runtime.get("autofix_engine")
    if autofix_engine is not None:
        try:
            await autofix_engine.stop()
            logger.info("Autonomous shutdown: AutofixEngine stopped")
        except Exception as exc:
            logger.error("Autonomous shutdown: AutofixEngine error: %s", exc)

    # ── 4. LoopManager ───────────────────────────────────────────────────
    loop_manager = runtime.get("loop_manager")
    if loop_manager is not None:
        try:
            await loop_manager.stop_all()
            logger.info("Autonomous shutdown: LoopManager stopped")
        except Exception as exc:
            logger.error("Autonomous shutdown: LoopManager error: %s", exc)

    # ── 5. PhaseHookManager ──────────────────────────────────────────────
    hook_manager = runtime.get("hook_manager")
    if hook_manager is not None:
        try:
            hook_manager.stop()
            logger.info("Autonomous shutdown: PhaseHookManager stopped")
        except Exception as exc:
            logger.error("Autonomous shutdown: PhaseHookManager error: %s", exc)

    _autonomous_runtime = None
    logger.info("Autonomous system fully stopped")


def get_autonomous_runtime() -> dict[str, Any] | None:
    """Get the current autonomous runtime, or None if not started.

    Returns:
        A dict of ``{"component_name": instance}`` for all active subsystems,
        or ``None`` if :func:`start_autonomous_system` has not been called.
    """
    return _autonomous_runtime
