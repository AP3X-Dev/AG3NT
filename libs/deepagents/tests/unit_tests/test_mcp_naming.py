"""Tests for MCP tool naming utilities."""

import pytest

from deepagents.mcp.naming import (
    ToolNameRegistry,
    create_prefixed_name,
    normalize_tool_name,
)


class TestNormalizeToolName:
    """Tests for tool name normalization."""

    def test_lowercase(self):
        """Names should be lowercased."""
        assert normalize_tool_name("GetWeather") == "getweather"

    def test_replace_invalid_chars(self):
        """Invalid characters should be replaced with underscores."""
        assert normalize_tool_name("get-weather") == "get_weather"
        assert normalize_tool_name("get.weather") == "get_weather"
        assert normalize_tool_name("get weather") == "get_weather"

    def test_remove_consecutive_underscores(self):
        """Consecutive underscores should be collapsed."""
        assert normalize_tool_name("get__weather") == "get_weather"
        assert normalize_tool_name("get---weather") == "get_weather"

    def test_strip_underscores(self):
        """Leading/trailing underscores should be removed."""
        assert normalize_tool_name("_weather_") == "weather"
        assert normalize_tool_name("__weather__") == "weather"


class TestCreatePrefixedName:
    """Tests for prefixed name creation."""

    def test_basic_prefix(self):
        """Basic prefix should work."""
        assert create_prefixed_name("math", "add") == "math_add"

    def test_normalizes_both(self):
        """Both server and tool names should be normalized."""
        assert create_prefixed_name("Math-Server", "Get-Result") == "math_server_get_result"

    def test_custom_separator(self):
        """Custom separator should work."""
        assert create_prefixed_name("math", "add", separator="__") == "math__add"


class TestToolNameRegistry:
    """Tests for tool name registry."""

    def test_register_tool(self):
        """Basic tool registration should work."""
        registry = ToolNameRegistry()
        info = registry.register_tool("add", "math", "math")

        assert info.original_name == "add"
        assert info.server_name == "math"
        assert info.prefixed_name == "math_add"
        assert info.prefix == "math"

    def test_register_without_prefix(self):
        """Registration without prefix should work."""
        registry = ToolNameRegistry()
        info = registry.register_tool("add", "math", "")

        assert info.prefixed_name == "add"
        assert info.prefix == ""

    def test_collision_detection(self):
        """Collisions should be detected."""
        registry = ToolNameRegistry()
        registry.register_tool("add", "math1", "")

        with pytest.raises(ValueError, match="Tool name collision"):
            registry.register_tool("add", "math2", "")

    def test_no_collision_with_prefix(self):
        """Different prefixes should not collide."""
        registry = ToolNameRegistry()
        registry.register_tool("add", "math1", "math1")
        info = registry.register_tool("add", "math2", "math2")

        assert info.prefixed_name == "math2_add"

    def test_get_all_tools(self):
        """Should return all registered tools."""
        registry = ToolNameRegistry()
        registry.register_tool("add", "math", "math")
        registry.register_tool("multiply", "math", "math")

        tools = registry.get_all_tools()
        assert len(tools) == 2
        assert "math_add" in tools
        assert "math_multiply" in tools

    def test_has_tool(self):
        """Should check if tool exists."""
        registry = ToolNameRegistry()
        registry.register_tool("add", "math", "math")

        assert registry.has_tool("math_add") is True
        assert registry.has_tool("math_subtract") is False

    def test_clear(self):
        """Should clear all tools."""
        registry = ToolNameRegistry()
        registry.register_tool("add", "math", "math")
        registry.clear()

        assert registry.has_tool("math_add") is False
        assert len(registry.get_all_tools()) == 0
