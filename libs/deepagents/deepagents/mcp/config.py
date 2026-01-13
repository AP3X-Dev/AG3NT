"""MCP configuration models with full validation.

This module defines the configuration schema for MCP server connections,
including transport types, tenant modes, and tool filtering.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FailBehavior(str, Enum):
    """Behavior when an MCP server connection fails."""

    FAIL_OPEN = "fail_open"
    """Agent builds successfully, server tools are omitted with a warning."""

    FAIL_CLOSED = "fail_closed"
    """Agent build fails with a clear error message."""


class TenantMode(str, Enum):
    """How tenant identity is resolved for MCP requests."""

    SINGLE = "single"
    """Single tenant mode - uses static tenant_id."""

    HEADER = "header"
    """Tenant ID is extracted from a request header."""

    TOKEN = "token"
    """Tenant ID is extracted from an auth token."""


class ServerInstanceScope(str, Enum):
    """How MCP server connections are scoped."""

    SHARED = "shared"
    """Single shared connection for all tenants."""

    PER_TENANT = "per_tenant"
    """Separate connection per tenant (for stateful servers)."""


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server connection.

    Supports both stdio (local process) and HTTP transports.
    """

    # Transport configuration
    transport: Literal["stdio", "http"] = Field(
        description="Transport type: 'stdio' for local process, 'http' for remote server"
    )

    # Stdio transport fields
    command: str | None = Field(
        default=None,
        description="Command to run for stdio transport (e.g., 'python', 'node')",
    )
    args: list[str] | None = Field(
        default=None,
        description="Arguments for the command",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Environment variables for the subprocess",
    )

    # HTTP transport fields
    url: str | None = Field(
        default=None,
        description="URL for HTTP transport (e.g., 'http://localhost:8000/mcp')",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="HTTP headers to include in requests",
    )

    # Common fields
    timeout_s: float | None = Field(
        default=30.0,
        description="Connection/request timeout in seconds",
    )

    # Tool naming and filtering
    tool_name_prefix: bool = Field(
        default=True,
        description="Whether to prefix tool names with server name",
    )
    name_prefix: str | None = Field(
        default=None,
        description="Custom prefix for tool names (defaults to server name)",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Allowlist of tool names to load (None = all)",
    )
    blocked_tools: list[str] | None = Field(
        default=None,
        description="Blocklist of tool names to exclude",
    )

    # Resource and prompt support (gated features)
    enable_resources: bool = Field(
        default=False,
        description="Enable resource fetching tools",
    )
    enable_prompts: bool = Field(
        default=False,
        description="Enable prompt fetching tools",
    )

    # Tenant readiness fields
    tenant_mode: TenantMode = Field(
        default=TenantMode.SINGLE,
        description="How tenant identity is resolved",
    )
    tenant_id: str | None = Field(
        default="default",
        description="Static tenant ID for single tenant mode",
    )
    tenant_header: str | None = Field(
        default=None,
        description="Header name for header-based tenant mode",
    )
    server_instance_scope: ServerInstanceScope = Field(
        default=ServerInstanceScope.SHARED,
        description="Whether connections are shared or per-tenant",
    )

    # Server-specific fail behavior override
    fail_behavior: FailBehavior | None = Field(
        default=None,
        description="Override global fail behavior for this server",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerConfig":
        """Validate that required fields are set for each transport type."""
        if self.transport == "stdio":
            if not self.command:
                msg = "stdio transport requires 'command' to be set"
                raise ValueError(msg)
        elif self.transport == "http":
            if not self.url:
                msg = "http transport requires 'url' to be set"
                raise ValueError(msg)

        # Validate tenant mode fields
        if self.tenant_mode == TenantMode.HEADER and not self.tenant_header:
            msg = "header tenant mode requires 'tenant_header' to be set"
            raise ValueError(msg)

        return self

    def get_effective_prefix(self, server_name: str) -> str:
        """Get the effective prefix to use for tool names.

        Args:
            server_name: The server name from the config dict key.

        Returns:
            The prefix to use, or empty string if prefixing is disabled.
        """
        if not self.tool_name_prefix:
            return ""
        return self.name_prefix or server_name

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool should be loaded based on allow/block lists.

        Precedence: blocked_tools takes priority over allowed_tools.

        Args:
            tool_name: The name of the tool to check.

        Returns:
            True if the tool should be loaded, False otherwise.
        """
        # Blocklist takes precedence
        if self.blocked_tools and tool_name in self.blocked_tools:
            return False

        # If allowlist is set, tool must be in it
        if self.allowed_tools is not None:
            return tool_name in self.allowed_tools

        # No restrictions, allow all
        return True


class MCPConfig(BaseModel):
    """Top-level MCP configuration for DeepAgents.

    Defines one or more MCP server connections with global settings.

    Example:
        ```python
        config = MCPConfig(
            servers={
                "math": MCPServerConfig(
                    transport="stdio",
                    command="python",
                    args=["math_server.py"],
                ),
                "api": MCPServerConfig(
                    transport="http",
                    url="http://localhost:8000/mcp",
                ),
            },
            fail_behavior=FailBehavior.FAIL_OPEN,
        )
        ```
    """

    servers: dict[str, MCPServerConfig] = Field(
        description="Named MCP server configurations"
    )
    fail_behavior: FailBehavior = Field(
        default=FailBehavior.FAIL_OPEN,
        description="Default behavior when server connections fail",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_server_names(self) -> "MCPConfig":
        """Validate that server names are valid identifiers."""
        name_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        for name in self.servers:
            if not name_pattern.match(name):
                msg = f"Server name '{name}' must be a valid identifier (letters, numbers, underscores, not starting with a number)"
                raise ValueError(msg)
        return self

    def get_server_fail_behavior(self, server_name: str) -> FailBehavior:
        """Get the effective fail behavior for a server.

        Args:
            server_name: Name of the server.

        Returns:
            Server-specific fail behavior if set, otherwise global default.
        """
        server = self.servers.get(server_name)
        if server and server.fail_behavior is not None:
            return server.fail_behavior
        return self.fail_behavior

