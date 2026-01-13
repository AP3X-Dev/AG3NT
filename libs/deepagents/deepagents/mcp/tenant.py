"""Tenant resolution and client pooling for MCP multi-tenancy.

This module provides primitives for multi-tenant MCP deployments,
allowing per-tenant connection isolation when needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from deepagents.mcp.config import MCPServerConfig, ServerInstanceScope, TenantMode

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    """Context for an incoming request, used for tenant resolution.

    This is a placeholder that will be extended when integrating
    with specific request handlers (e.g., HTTP, WebSocket).
    """

    headers: dict[str, str] = field(default_factory=dict)
    """Request headers."""

    auth_token: str | None = None
    """Authentication token if available."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for tenant resolution."""


class TenantResolver:
    """Resolves tenant identity from request context.

    Supports multiple tenant modes:
    - single: Static tenant ID from config
    - header: Extract from request header
    - token: Extract from authentication token (placeholder)
    """

    def resolve_tenant_id(
        self,
        request_context: RequestContext | None,
        server_config: MCPServerConfig,
    ) -> str:
        """Resolve the tenant ID for an MCP request.

        Args:
            request_context: The request context (may be None for single tenant).
            server_config: The server configuration with tenant mode.

        Returns:
            The resolved tenant ID string.

        Raises:
            ValueError: If tenant cannot be resolved in header/token mode.
        """
        if server_config.tenant_mode == TenantMode.SINGLE:
            return server_config.tenant_id or "default"

        if request_context is None:
            logger.warning(
                "No request context provided for tenant mode %s, falling back to default",
                server_config.tenant_mode,
            )
            return "default"

        if server_config.tenant_mode == TenantMode.HEADER:
            if not server_config.tenant_header:
                msg = "tenant_header must be set for header tenant mode"
                raise ValueError(msg)

            tenant_id = request_context.headers.get(server_config.tenant_header)
            if not tenant_id:
                msg = f"Missing tenant header: {server_config.tenant_header}"
                raise ValueError(msg)

            # Validate tenant ID format (prevent injection)
            if not self._is_valid_tenant_id(tenant_id):
                msg = f"Invalid tenant ID format: {tenant_id}"
                raise ValueError(msg)

            return tenant_id

        if server_config.tenant_mode == TenantMode.TOKEN:
            # Placeholder for token-based tenant resolution
            # This would integrate with your auth system
            if not request_context.auth_token:
                msg = "auth_token required for token tenant mode"
                raise ValueError(msg)

            # TODO: Implement token parsing/validation
            # For now, return a placeholder
            logger.warning("Token tenant mode not fully implemented")
            return "token_tenant"

        msg = f"Unknown tenant mode: {server_config.tenant_mode}"
        raise ValueError(msg)

    def _is_valid_tenant_id(self, tenant_id: str) -> bool:
        """Validate tenant ID format to prevent injection attacks.

        Args:
            tenant_id: The tenant ID to validate.

        Returns:
            True if valid, False otherwise.
        """
        import re

        # Allow alphanumeric, hyphens, underscores, max 128 chars
        pattern = r"^[a-zA-Z0-9_-]{1,128}$"
        return bool(re.match(pattern, tenant_id))


class MCPClientProtocol(Protocol):
    """Protocol for MCP client instances (for type hinting)."""

    async def get_tools(self) -> list[Any]:
        """Get tools from the MCP server."""
        ...

    async def close(self) -> None:
        """Close the client connection."""
        ...


class ClientPool:
    """Connection pool for MCP clients with optional per-tenant isolation.

    Manages MCP client connections, supporting both shared (single connection)
    and per-tenant (isolated connections) modes.
    """

    def __init__(self) -> None:
        """Initialize the client pool."""
        # Keyed by (server_name, tenant_id) for per-tenant scope
        # Keyed by (server_name, "shared") for shared scope
        self._clients: dict[tuple[str, str], Any] = {}
        self._tenant_resolver = TenantResolver()

    def get_pool_key(
        self,
        server_name: str,
        server_config: MCPServerConfig,
        request_context: RequestContext | None = None,
    ) -> tuple[str, str]:
        """Get the pool key for a server/tenant combination.

        Args:
            server_name: Name of the MCP server.
            server_config: Server configuration.
            request_context: Request context for tenant resolution.

        Returns:
            Tuple of (server_name, tenant_key) for pool lookup.
        """
        if server_config.server_instance_scope == ServerInstanceScope.SHARED:
            return (server_name, "shared")

        # Per-tenant scope
        tenant_id = self._tenant_resolver.resolve_tenant_id(request_context, server_config)
        return (server_name, tenant_id)
