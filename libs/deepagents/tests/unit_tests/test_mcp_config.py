"""Tests for MCP configuration models."""

import pytest
from pydantic import ValidationError

from deepagents.mcp.config import (
    FailBehavior,
    MCPConfig,
    MCPServerConfig,
    TenantMode,
)


class TestMCPServerConfig:
    """Tests for MCPServerConfig validation."""

    def test_stdio_transport_requires_command(self):
        """Stdio transport must have command set."""
        with pytest.raises(ValidationError, match="stdio transport requires 'command'"):
            MCPServerConfig(transport="stdio")

    def test_stdio_transport_valid(self):
        """Valid stdio config should work."""
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            args=["server.py"],
        )
        assert config.command == "python"
        assert config.args == ["server.py"]

    def test_http_transport_requires_url(self):
        """HTTP transport must have url set."""
        with pytest.raises(ValidationError, match="http transport requires 'url'"):
            MCPServerConfig(transport="http")

    def test_http_transport_valid(self):
        """Valid HTTP config should work."""
        config = MCPServerConfig(
            transport="http",
            url="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert config.url == "http://localhost:8000/mcp"
        assert config.headers == {"Authorization": "Bearer token"}

    def test_header_tenant_mode_requires_header(self):
        """Header tenant mode requires tenant_header."""
        with pytest.raises(ValidationError, match="tenant_header"):
            MCPServerConfig(
                transport="stdio",
                command="python",
                tenant_mode=TenantMode.HEADER,
            )

    def test_header_tenant_mode_valid(self):
        """Valid header tenant mode config."""
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            tenant_mode=TenantMode.HEADER,
            tenant_header="X-Tenant-ID",
        )
        assert config.tenant_header == "X-Tenant-ID"

    def test_tool_filtering_allowed(self):
        """Test allowed_tools filtering."""
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            allowed_tools=["add", "multiply"],
        )
        assert config.is_tool_allowed("add") is True
        assert config.is_tool_allowed("subtract") is False

    def test_tool_filtering_blocked(self):
        """Test blocked_tools filtering."""
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            blocked_tools=["dangerous_tool"],
        )
        assert config.is_tool_allowed("safe_tool") is True
        assert config.is_tool_allowed("dangerous_tool") is False

    def test_blocked_takes_precedence(self):
        """Blocked tools take precedence over allowed."""
        config = MCPServerConfig(
            transport="stdio",
            command="python",
            allowed_tools=["tool1", "tool2"],
            blocked_tools=["tool1"],
        )
        assert config.is_tool_allowed("tool1") is False
        assert config.is_tool_allowed("tool2") is True

    def test_get_effective_prefix(self):
        """Test prefix generation."""
        config = MCPServerConfig(transport="stdio", command="python")
        assert config.get_effective_prefix("math") == "math"

        config_custom = MCPServerConfig(
            transport="stdio",
            command="python",
            name_prefix="custom",
        )
        assert config_custom.get_effective_prefix("math") == "custom"

        config_disabled = MCPServerConfig(
            transport="stdio",
            command="python",
            tool_name_prefix=False,
        )
        assert config_disabled.get_effective_prefix("math") == ""


class TestMCPConfig:
    """Tests for MCPConfig validation."""

    def test_valid_server_names(self):
        """Valid server names should work."""
        config = MCPConfig(
            servers={
                "math": MCPServerConfig(transport="stdio", command="python"),
                "api_v2": MCPServerConfig(transport="http", url="http://localhost:8000"),
            }
        )
        assert "math" in config.servers
        assert "api_v2" in config.servers

    def test_invalid_server_name(self):
        """Invalid server names should fail."""
        with pytest.raises(ValidationError, match="must be a valid identifier"):
            MCPConfig(
                servers={
                    "123invalid": MCPServerConfig(transport="stdio", command="python"),
                }
            )

    def test_get_server_fail_behavior(self):
        """Test fail behavior resolution."""
        config = MCPConfig(
            servers={
                "math": MCPServerConfig(
                    transport="stdio",
                    command="python",
                    fail_behavior=FailBehavior.FAIL_CLOSED,
                ),
                "api": MCPServerConfig(transport="http", url="http://localhost:8000"),
            },
            fail_behavior=FailBehavior.FAIL_OPEN,
        )
        # Server-specific override
        assert config.get_server_fail_behavior("math") == FailBehavior.FAIL_CLOSED
        # Global default
        assert config.get_server_fail_behavior("api") == FailBehavior.FAIL_OPEN
