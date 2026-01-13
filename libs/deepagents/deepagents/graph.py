"""Deepagents come with planning, filesystem, and subagents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents.mcp import MCPConfig

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.agents.structured_output import ResponseFormat
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.cache.base import BaseCache
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.openrouter import get_default_openrouter_model, is_openrouter_configured, load_env

BASE_AGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."


def get_default_model() -> BaseChatModel:
    """Get the default model for deep agents.

    Checks for OpenRouter configuration first (via OPENROUTER_API_KEY environment variable).
    If not configured, falls back to Claude Sonnet 4.5 via Anthropic.

    Returns:
        `BaseChatModel` instance - either OpenRouter or ChatAnthropic depending on configuration.
    """
    # Load environment variables from .env file if it exists
    load_env()

    # Check if OpenRouter is configured
    if is_openrouter_configured():
        return get_default_openrouter_model()

    # Fall back to default Anthropic model
    return ChatAnthropic(
        model_name="claude-sonnet-4-5-20250929",
        max_tokens=20000,
    )


def _create_mcp_middleware(mcp_config: "MCPConfig | dict[str, Any]") -> "AgentMiddleware":
    """Create MCPMiddleware from config.

    Args:
        mcp_config: MCPConfig instance or dict to parse.

    Returns:
        Configured MCPMiddleware instance.

    Raises:
        ImportError: If MCP dependencies are not installed.
    """
    try:
        from deepagents.mcp import MCPConfig
        from deepagents.middleware.mcp import MCPMiddleware
    except ImportError as e:
        msg = (
            "MCP dependencies not installed. "
            "Install with: pip install deepagents[mcp]"
        )
        raise ImportError(msg) from e

    # Parse dict config if needed
    if isinstance(mcp_config, dict):
        mcp_config = MCPConfig(**mcp_config)

    return MCPMiddleware(config=mcp_config)


def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    mcp: "MCPConfig | dict[str, Any] | None" = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph:
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (`write_todos`),
    seven file and execution tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`,
    and a tool to call subagents.

    The `execute` tool allows running shell commands if the backend implements `SandboxBackendProtocol`.
    For non-sandbox backends, the `execute` tool will return an error message.

    Args:
        model: The model to use. Defaults to `claude-sonnet-4-5-20250929`.
        tools: The tools the agent should have access to.
        system_prompt: The additional instructions the agent should have. Will go in
            the system prompt.
        middleware: Additional middleware to apply after standard middleware.
        subagents: The subagents to use.

            Each subagent should be a `dict` with the following keys:

            - `name`
            - `description` (used by the main agent to decide whether to call the sub agent)
            - `prompt` (used as the system prompt in the subagent)
            - (optional) `tools`
            - (optional) `model` (either a `LanguageModelLike` instance or `dict` settings)
            - (optional) `middleware` (list of `AgentMiddleware`)
        skills: Optional list of skill source paths (e.g., `["/skills/user/", "/skills/project/"]`).

            Paths must be specified using POSIX conventions (forward slashes) and are relative
            to the backend's root. When using `StateBackend` (default), provide skill files via
            `invoke(files={...})`. With `FilesystemBackend`, skills are loaded from disk relative
            to the backend's `root_dir`. Later sources override earlier ones for skills with the
            same name (last one wins).
        memory: Optional list of memory file paths (`AGENTS.md` files) to load
            (e.g., `["/memory/AGENTS.md"]`). Display names are automatically derived from paths.
            Memory is loaded at agent startup and added into the system prompt.
        mcp: Optional MCP (Model Context Protocol) configuration for loading tools from
            external MCP servers. Can be an `MCPConfig` instance or a dict that will be
            parsed into one. Requires `pip install deepagents[mcp]`.

            Example::

                mcp=MCPConfig(
                    servers={
                        "math": MCPServerConfig(
                            transport="stdio",
                            command="python",
                            args=["math_server.py"],
                        ),
                    }
                )

        response_format: A structured output response format to use for the agent.
        context_schema: The schema of the deep agent.
        checkpointer: Optional `Checkpointer` for persisting agent state between runs.
        store: Optional store for persistent storage (required if backend uses `StoreBackend`).
        backend: Optional backend for file storage and execution.

            Pass either a `Backend` instance or a callable factory like `lambda rt: StateBackend(rt)`.
            For execution support, use a backend that implements `SandboxBackendProtocol`.
        interrupt_on: Mapping of tool names to interrupt configs.
        debug: Whether to enable debug mode. Passed through to `create_agent`.
        name: The name of the agent. Passed through to `create_agent`.
        cache: The cache to use for the agent. Passed through to `create_agent`.

    Returns:
        A configured deep agent.
    """
    if model is None:
        model = get_default_model()
    elif isinstance(model, str):
        model = init_chat_model(model)

    if (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    ):
        trigger = ("fraction", 0.85)
        keep = ("fraction", 0.10)
    else:
        trigger = ("tokens", 170000)
        keep = ("messages", 6)

    # Build middleware stack for subagents (includes skills if provided)
    subagent_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]

    backend = backend if backend is not None else (lambda rt: StateBackend(rt))

    if skills is not None:
        subagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    subagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            SummarizationMiddleware(
                model=model,
                trigger=trigger,
                keep=keep,
                trim_tokens_to_summarize=None,
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )

    # Build main agent middleware stack
    deepagent_middleware: list[AgentMiddleware] = [
        TodoListMiddleware(),
    ]
    if memory is not None:
        deepagent_middleware.append(MemoryMiddleware(backend=backend, sources=memory))
    if skills is not None:
        deepagent_middleware.append(SkillsMiddleware(backend=backend, sources=skills))
    deepagent_middleware.extend(
        [
            FilesystemMiddleware(backend=backend),
            SubAgentMiddleware(
                default_model=model,
                default_tools=tools,
                subagents=subagents if subagents is not None else [],
                default_middleware=subagent_middleware,
                default_interrupt_on=interrupt_on,
                general_purpose_agent=True,
            ),
            SummarizationMiddleware(
                model=model,
                trigger=trigger,
                keep=keep,
                trim_tokens_to_summarize=None,
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )
    if middleware:
        deepagent_middleware.extend(middleware)

    # Add MCP middleware if configured
    if mcp is not None:
        deepagent_middleware.append(_create_mcp_middleware(mcp))

    if interrupt_on is not None:
        deepagent_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return create_agent(
        model,
        system_prompt=system_prompt + "\n\n" + BASE_AGENT_PROMPT if system_prompt else BASE_AGENT_PROMPT,
        tools=tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
