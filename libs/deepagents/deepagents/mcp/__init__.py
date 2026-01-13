"""MCP (Model Context Protocol) integration for DeepAgents.

This module provides optional MCP client connectivity, allowing DeepAgents
to load tools from one or more MCP servers.

Install with: pip install deepagents[mcp]

Example:
    ```python
    from deepagents import create_deep_agent
    from deepagents.mcp import MCPConfig

    config = MCPConfig(
        servers={
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["math_server.py"],
            },
            "weather": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
            },
        }
    )

    agent = create_deep_agent(mcp=config)
    ```
"""

from deepagents.mcp.config import (
    FailBehavior,
    MCPConfig,
    MCPServerConfig,
    ServerInstanceScope,
    TenantMode,
)
from deepagents.mcp.naming import ToolNameInfo, ToolNameRegistry
from deepagents.mcp.tenant import ClientPool, RequestContext, TenantResolver

__all__ = [
    "ClientPool",
    "FailBehavior",
    "MCPConfig",
    "MCPServerConfig",
    "RequestContext",
    "ServerInstanceScope",
    "TenantMode",
    "TenantResolver",
    "ToolNameInfo",
    "ToolNameRegistry",
]

# Optional middleware (requires langchain-mcp-adapters)
try:
    from deepagents.middleware.mcp import (
        MCPMiddleware,
        ToolCallAuditInfo,
        ToolCallResult,
    )

    __all__.extend(
        [
            "MCPMiddleware",
            "ToolCallAuditInfo",
            "ToolCallResult",
        ]
    )
except ImportError:
    # MCP adapter dependencies not installed
    pass
