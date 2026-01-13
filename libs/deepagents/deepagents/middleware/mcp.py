"""MCP Middleware for integrating MCP tools into DeepAgents.

This middleware connects to one or more MCP servers and exposes their tools
to the agent. It handles tool naming, allowlists, and failure modes.

Install with: pip install deepagents[mcp]

Example:
    ```python
    from deepagents import create_deep_agent
    from deepagents.mcp import MCPConfig
    from deepagents.middleware.mcp import MCPMiddleware

    config = MCPConfig(
        servers={
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["math_server.py"],
            }
        }
    )

    agent = create_deep_agent(middleware=[MCPMiddleware(config=config)])
    ```
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.tools import BaseTool, StructuredTool

from deepagents.mcp.config import FailBehavior, MCPConfig, MCPServerConfig
from deepagents.mcp.naming import ToolNameRegistry

logger = logging.getLogger(__name__)


# --- Audit Hook Types ---


@dataclass
class ToolCallAuditInfo:
    """Information about a tool call for audit hooks."""

    server_name: str
    """Name of the MCP server providing the tool."""

    tool_name: str
    """Original tool name from the server."""

    prefixed_name: str
    """Final tool name used by the agent."""

    tenant_id: str
    """Tenant ID for the request."""

    caller_identity: str | None = None
    """Caller identity if available."""

    args: dict[str, Any] = field(default_factory=dict)
    """Tool call arguments."""


@dataclass
class ToolCallResult:
    """Result of a tool call for audit hooks."""

    success: bool
    """Whether the call succeeded."""

    latency_ms: float
    """Call latency in milliseconds."""

    output_size: int
    """Approximate output size in bytes."""

    error: str | None = None
    """Error message if failed."""


# Audit hook type definitions
BeforeToolCallHook = Callable[[ToolCallAuditInfo], Awaitable[None] | None]
AfterToolCallHook = Callable[[ToolCallAuditInfo, ToolCallResult], Awaitable[None] | None]


def _check_mcp_available() -> bool:
    """Check if MCP dependencies are available."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401

        return True
    except ImportError:
        return False


MCP_SYSTEM_PROMPT = """## MCP Tools

You have access to tools from external MCP (Model Context Protocol) servers.
These tools are prefixed with the server name for clarity.
"""


class MCPMiddleware(AgentMiddleware):
    """Middleware for loading and exposing MCP tools to an agent.

    Connects to configured MCP servers at initialization time and loads
    their tools. Handles:
    - Tool name prefixing to avoid collisions
    - Allowlist/blocklist filtering
    - fail_open/fail_closed behavior
    - Audit hooks for observability

    Args:
        config: MCPConfig with server definitions.
        before_tool_call: Optional async callback before each tool call.
        after_tool_call: Optional async callback after each tool call.
        system_prompt: Optional custom system prompt override.

    Raises:
        ImportError: If MCP dependencies are not installed.
    """

    def __init__(
        self,
        config: MCPConfig,
        *,
        before_tool_call: BeforeToolCallHook | None = None,
        after_tool_call: AfterToolCallHook | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize MCP middleware."""
        if not _check_mcp_available():
            msg = "MCP dependencies not installed. Install with: pip install deepagents[mcp]"
            raise ImportError(msg)

        self.config = config
        self._before_tool_call = before_tool_call
        self._after_tool_call = after_tool_call
        self._custom_system_prompt = system_prompt
        self._name_registry = ToolNameRegistry()

        # Tools will be loaded lazily or eagerly depending on usage
        self._tools: list[BaseTool] = []
        self._tools_loaded = False
        self._load_errors: list[str] = []

        # Server metadata for wrapped tools
        self._tool_metadata: dict[str, tuple[str, str]] = {}  # prefixed_name -> (server, original)

        # MCP client for resource/prompt access (set during load_tools_async)
        self._client: Any = None

    @property
    def tools(self) -> list[BaseTool]:
        """Get loaded MCP tools.

        Triggers synchronous tool loading if not already loaded.
        """
        if not self._tools_loaded:
            # Run async loading in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context, can't block
                    logger.warning("MCP tools not loaded yet - call load_tools_async() first in async context")
                else:
                    loop.run_until_complete(self.load_tools_async())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(self.load_tools_async())
        return self._tools

    async def load_tools_async(self) -> list[BaseTool]:
        """Load tools from all configured MCP servers.

        Returns:
            List of LangChain-compatible tools.
        """
        if self._tools_loaded:
            return self._tools

        from langchain_mcp_adapters.client import MultiServerMCPClient

        # Build server config for MultiServerMCPClient
        client_config: dict[str, dict[str, Any]] = {}

        for server_name, server_config in self.config.servers.items():
            try:
                client_config[server_name] = self._build_client_config(server_config)
            except Exception as e:
                self._handle_server_error(server_name, e)

        if not client_config:
            logger.warning("No MCP servers configured successfully")
            self._tools_loaded = True
            return self._tools

        # Connect and load tools
        try:
            client = MultiServerMCPClient(client_config)
            # Store client for resource/prompt access
            self._client = client
            raw_tools = await client.get_tools()

            # Process and wrap tools
            for tool in raw_tools:
                wrapped = self._wrap_tool(tool)
                if wrapped:
                    self._tools.append(wrapped)

            # Add resource and prompt tools for each server
            for server_name, server_config in self.config.servers.items():
                self._tools.extend(self._create_resource_tools(server_name, server_config))
                self._tools.extend(self._create_prompt_tools(server_name, server_config))

        except Exception as e:
            logger.error("Failed to connect to MCP servers: %s", e)
            if self.config.fail_behavior == FailBehavior.FAIL_CLOSED:
                raise

        self._tools_loaded = True
        return self._tools

    def _build_client_config(self, server_config: MCPServerConfig) -> dict[str, Any]:
        """Build config dict for MultiServerMCPClient.

        Args:
            server_config: The server configuration.

        Returns:
            Config dict compatible with MultiServerMCPClient.
        """
        config: dict[str, Any] = {}

        if server_config.transport == "stdio":
            config["transport"] = "stdio"
            config["command"] = server_config.command
            if server_config.args:
                config["args"] = server_config.args
            if server_config.env:
                config["env"] = server_config.env
        else:  # http
            config["transport"] = "streamable_http"
            config["url"] = server_config.url
            if server_config.headers:
                config["headers"] = server_config.headers

        return config

    def _handle_server_error(self, server_name: str, error: Exception) -> None:
        """Handle a server connection error based on fail behavior.

        Args:
            server_name: Name of the failing server.
            error: The exception that occurred.
        """
        fail_behavior = self.config.get_server_fail_behavior(server_name)
        error_msg = f"MCP server '{server_name}' failed: {error}"
        self._load_errors.append(error_msg)

        if fail_behavior == FailBehavior.FAIL_CLOSED:
            logger.error(error_msg)
            raise RuntimeError(error_msg) from error
        logger.warning(error_msg)

    def _wrap_tool(self, tool: BaseTool) -> BaseTool | None:
        """Wrap an MCP tool with naming and audit hooks.

        Args:
            tool: The original tool from MCP.

        Returns:
            Wrapped tool or None if filtered out.
        """
        # Extract server name from tool metadata or name
        # MultiServerMCPClient prefixes tools with server name
        parts = tool.name.split("_", 1)
        if len(parts) == 2:
            server_name, original_name = parts
        else:
            server_name = "unknown"
            original_name = tool.name

        # Check if server config exists and tool is allowed
        server_config = self.config.servers.get(server_name)
        if server_config and not server_config.is_tool_allowed(original_name):
            logger.debug("Tool '%s' from '%s' filtered by allow/block list", original_name, server_name)
            return None

        # Register tool name
        prefix = ""
        if server_config:
            prefix = server_config.get_effective_prefix(server_name)

        try:
            name_info = self._name_registry.register_tool(original_name, server_name, prefix)
        except ValueError as e:
            logger.error("Tool name collision: %s", e)
            if self.config.fail_behavior == FailBehavior.FAIL_CLOSED:
                raise
            return None

        # Store metadata for audit hooks
        self._tool_metadata[name_info.prefixed_name] = (server_name, original_name)

        # Create wrapped tool with audit hooks
        return self._create_audited_tool(tool, name_info.prefixed_name, server_name, original_name)

    def _create_audited_tool(
        self,
        original_tool: BaseTool,
        prefixed_name: str,
        server_name: str,
        original_name: str,
    ) -> BaseTool:
        """Create a tool wrapper with audit hooks.

        Args:
            original_tool: The original MCP tool.
            prefixed_name: The final tool name.
            server_name: Name of the MCP server.
            original_name: Original tool name.

        Returns:
            Wrapped tool with audit hooks.
        """
        before_hook = self._before_tool_call
        after_hook = self._after_tool_call

        async def audited_func(**kwargs: Any) -> Any:
            """Wrapped tool function with audit hooks."""
            audit_info = ToolCallAuditInfo(
                server_name=server_name,
                tool_name=original_name,
                prefixed_name=prefixed_name,
                tenant_id="default",  # TODO: Get from context
                args=kwargs,
            )

            # Before hook
            if before_hook:
                result = before_hook(audit_info)
                if asyncio.iscoroutine(result):
                    await result

            # Execute tool
            start_time = time.perf_counter()
            try:
                if hasattr(original_tool, "ainvoke"):
                    output = await original_tool.ainvoke(kwargs)
                else:
                    output = original_tool.invoke(kwargs)

                latency_ms = (time.perf_counter() - start_time) * 1000

                # After hook
                if after_hook:
                    call_result = ToolCallResult(
                        success=True,
                        latency_ms=latency_ms,
                        output_size=len(str(output)),
                    )
                    result = after_hook(audit_info, call_result)
                    if asyncio.iscoroutine(result):
                        await result

                return output

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000

                if after_hook:
                    call_result = ToolCallResult(
                        success=False,
                        latency_ms=latency_ms,
                        output_size=0,
                        error=str(e),
                    )
                    result = after_hook(audit_info, call_result)
                    if asyncio.iscoroutine(result):
                        await result

                raise

        return StructuredTool.from_function(
            func=None,
            coroutine=audited_func,
            name=prefixed_name,
            description=original_tool.description,
            args_schema=original_tool.args_schema,
        )

    # --- AgentMiddleware Interface ---

    def get_system_prompt(self) -> str | None:
        """Get the system prompt contribution from this middleware.

        Returns:
            System prompt text or None.
        """
        if self._custom_system_prompt:
            return self._custom_system_prompt
        if self._tools:
            return MCP_SYSTEM_PROMPT
        return None

    def get_tools(self) -> list[BaseTool]:
        """Get tools provided by this middleware.

        Returns:
            List of MCP tools.
        """
        return self.tools

    async def before_request(self, request: ModelRequest) -> ModelRequest:
        """Called before each model request.

        Args:
            request: The model request.

        Returns:
            Potentially modified request.
        """
        return request

    async def after_response(self, response: ModelResponse) -> ModelResponse:
        """Called after each model response.

        Args:
            response: The model response.

        Returns:
            Potentially modified response.
        """
        return response

    def get_load_errors(self) -> list[str]:
        """Get any errors that occurred during tool loading.

        Returns:
            List of error messages.
        """
        return list(self._load_errors)

    def _create_resource_tools(
        self,
        server_name: str,
        server_config: MCPServerConfig,
    ) -> list[BaseTool]:
        """Create tools for fetching MCP resources.

        Only created if enable_resources is True in server config.

        Args:
            server_name: Name of the MCP server.
            server_config: Server configuration.

        Returns:
            List of resource tools (may be empty).
        """
        if not server_config.enable_resources:
            return []

        prefix = server_config.get_effective_prefix(server_name)
        tool_name = f"{prefix}_read_resource" if prefix else "read_resource"

        # Capture self for closure
        middleware = self

        async def read_resource(uri: str) -> str:
            """Read a resource from the MCP server by URI.

            Args:
                uri: The resource URI to fetch.

            Returns:
                The resource content as a string.
            """
            if middleware._client is None:
                return f"Error: MCP client not initialized for {server_name}"

            try:
                from langchain_mcp_adapters.resources import load_mcp_resources

                logger.info("Fetching resource %s from %s", uri, server_name)
                async with middleware._client.session(server_name) as session:
                    blobs = await load_mcp_resources(session, uris=[uri])
                    if not blobs:
                        return f"No resource found at URI: {uri}"

                    # Convert blob to string content
                    blob = blobs[0]
                    content_bytes = blob.as_bytes()
                    # Try to decode as text, fallback to base64 for binary
                    try:
                        content = content_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        import base64

                        content = f"[Binary content, base64 encoded]\n{base64.b64encode(content_bytes).decode('ascii')}"

                    return f"Resource: {uri}\nMIME type: {blob.mimetype}\n\n{content}"

            except Exception as e:
                logger.error("Failed to fetch resource %s from %s: %s", uri, server_name, e)
                return f"Error fetching resource {uri}: {e}"

        return [
            StructuredTool.from_function(
                func=None,
                coroutine=read_resource,
                name=tool_name,
                description=f"Read a resource from the {server_name} MCP server by URI. Resources provide access to data like files, database records, or other content managed by the server.",
            )
        ]

    def _create_prompt_tools(
        self,
        server_name: str,
        server_config: MCPServerConfig,
    ) -> list[BaseTool]:
        """Create tools for fetching MCP prompts.

        Only created if enable_prompts is True in server config.

        Args:
            server_name: Name of the MCP server.
            server_config: Server configuration.

        Returns:
            List of prompt tools (may be empty).
        """
        if not server_config.enable_prompts:
            return []

        prefix = server_config.get_effective_prefix(server_name)
        tool_name = f"{prefix}_get_prompt" if prefix else "get_prompt"

        # Capture self for closure
        middleware = self

        async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> str:
            """Get a prompt template from the MCP server.

            Args:
                name: The prompt name to fetch.
                arguments: Optional arguments to fill in the prompt template.

            Returns:
                The rendered prompt content.
            """
            if middleware._client is None:
                return f"Error: MCP client not initialized for {server_name}"

            try:
                from langchain_mcp_adapters.prompts import load_mcp_prompt

                logger.info("Fetching prompt %s from %s", name, server_name)
                async with middleware._client.session(server_name) as session:
                    messages = await load_mcp_prompt(
                        session,
                        name,
                        arguments=arguments or {},
                    )

                    if not messages:
                        return f"No prompt found with name: {name}"

                    # Format messages as readable text
                    output_parts = [f"Prompt: {name}"]
                    if arguments:
                        output_parts.append(f"Arguments: {arguments}")
                    output_parts.append("")

                    for msg in messages:
                        role = getattr(msg, "type", "message")
                        content = getattr(msg, "content", str(msg))
                        output_parts.append(f"[{role}]\n{content}")

                    return "\n".join(output_parts)

            except Exception as e:
                logger.error("Failed to fetch prompt %s from %s: %s", name, server_name, e)
                return f"Error fetching prompt {name}: {e}"

        return [
            StructuredTool.from_function(
                func=None,
                coroutine=get_prompt,
                name=tool_name,
                description=f"Get a prompt template from the {server_name} MCP server. Prompts are reusable templates that can be customized with arguments.",
            )
        ]
