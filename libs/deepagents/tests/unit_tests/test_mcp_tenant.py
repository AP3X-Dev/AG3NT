"""Tests for MCP tenant resolution."""

import pytest

from deepagents.mcp.config import MCPServerConfig, ServerInstanceScope, TenantMode
from deepagents.mcp.tenant import ClientPool, RequestContext, TenantResolver


class TestTenantResolver:
    """Tests for tenant resolution."""

    def test_single_tenant_mode(self):
        """Single tenant mode should return static tenant_id."""
        resolver = TenantResolver()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            tenant_mode=TenantMode.SINGLE,
            tenant_id="my-tenant",
        )

        tenant_id = resolver.resolve_tenant_id(None, config)
        assert tenant_id == "my-tenant"

    def test_single_tenant_default(self):
        """Single tenant mode should default to 'default'."""
        resolver = TenantResolver()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            tenant_mode=TenantMode.SINGLE,
            tenant_id=None,
        )

        tenant_id = resolver.resolve_tenant_id(None, config)
        assert tenant_id == "default"

    def test_header_tenant_mode(self):
        """Header tenant mode should extract from headers."""
        resolver = TenantResolver()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            tenant_mode=TenantMode.HEADER,
            tenant_header="X-Tenant-ID",
        )
        context = RequestContext(headers={"X-Tenant-ID": "tenant-123"})

        tenant_id = resolver.resolve_tenant_id(context, config)
        assert tenant_id == "tenant-123"

    def test_header_tenant_missing_header(self):
        """Missing tenant header should raise error."""
        resolver = TenantResolver()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            tenant_mode=TenantMode.HEADER,
            tenant_header="X-Tenant-ID",
        )
        context = RequestContext(headers={})

        with pytest.raises(ValueError, match="Missing tenant header"):
            resolver.resolve_tenant_id(context, config)

    def test_header_tenant_invalid_format(self):
        """Invalid tenant ID format should raise error."""
        resolver = TenantResolver()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            tenant_mode=TenantMode.HEADER,
            tenant_header="X-Tenant-ID",
        )
        # Tenant ID with invalid characters
        context = RequestContext(headers={"X-Tenant-ID": "tenant; DROP TABLE users;"})

        with pytest.raises(ValueError, match="Invalid tenant ID format"):
            resolver.resolve_tenant_id(context, config)

    def test_no_context_fallback(self):
        """Missing context in non-single mode should fallback to default."""
        resolver = TenantResolver()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            tenant_mode=TenantMode.HEADER,
            tenant_header="X-Tenant-ID",
        )

        # Should log warning and return default
        tenant_id = resolver.resolve_tenant_id(None, config)
        assert tenant_id == "default"


class TestClientPool:
    """Tests for client pooling."""

    def test_shared_scope_key(self):
        """Shared scope should use 'shared' as tenant key."""
        pool = ClientPool()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            server_instance_scope=ServerInstanceScope.SHARED,
        )

        key = pool.get_pool_key("math", config)
        assert key == ("math", "shared")

    def test_per_tenant_scope_key(self):
        """Per-tenant scope should use tenant ID as key."""
        pool = ClientPool()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            server_instance_scope=ServerInstanceScope.PER_TENANT,
            tenant_mode=TenantMode.HEADER,
            tenant_header="X-Tenant-ID",
        )
        context = RequestContext(headers={"X-Tenant-ID": "tenant-abc"})

        key = pool.get_pool_key("math", config, context)
        assert key == ("math", "tenant-abc")

    def test_per_tenant_no_context(self):
        """Per-tenant scope without context should use default."""
        pool = ClientPool()
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            server_instance_scope=ServerInstanceScope.PER_TENANT,
            tenant_mode=TenantMode.SINGLE,
            tenant_id="my-tenant",
        )

        key = pool.get_pool_key("math", config)
        assert key == ("math", "my-tenant")

