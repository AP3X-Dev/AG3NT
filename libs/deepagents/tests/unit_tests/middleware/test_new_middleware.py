"""Tests for new middleware modules: UtilitiesMiddleware, WebMiddleware, AdvancedMiddleware."""


from deepagents.middleware import AdvancedMiddleware, UtilitiesMiddleware, WebMiddleware
from deepagents.middleware.advanced import get_librarian_subagent, get_oracle_subagent


class TestUtilitiesMiddleware:
    """Test UtilitiesMiddleware initialization and tools."""

    def test_init_default(self):
        """Test default initialization."""
        middleware = UtilitiesMiddleware()
        assert middleware is not None
        assert len(middleware.tools) == 4  # undo_edit, format_file, get_diagnostics, mermaid

    def test_init_with_enabled_tools(self):
        """Test initialization with specific tools enabled."""
        middleware = UtilitiesMiddleware(enabled_tools=["undo_edit", "mermaid"])
        assert len(middleware.tools) == 2

    def test_tool_names(self):
        """Test that all expected tools are present."""
        middleware = UtilitiesMiddleware()
        tool_names = {tool.name for tool in middleware.tools}
        assert tool_names == {"undo_edit", "format_file", "get_diagnostics", "mermaid"}


class TestWebMiddleware:
    """Test WebMiddleware initialization and tools."""

    def test_init_default(self):
        """Test default initialization."""
        middleware = WebMiddleware()
        assert middleware is not None
        assert len(middleware.tools) == 2  # web_search, read_web_page

    def test_init_with_enabled_tools(self):
        """Test initialization with specific tools enabled."""
        middleware = WebMiddleware(enabled_tools=["web_search"])
        assert len(middleware.tools) == 1

    def test_tool_names(self):
        """Test that all expected tools are present."""
        middleware = WebMiddleware()
        tool_names = {tool.name for tool in middleware.tools}
        assert tool_names == {"web_search", "read_web_page"}


class TestAdvancedMiddleware:
    """Test AdvancedMiddleware initialization and tools."""

    def test_init_default(self):
        """Test default initialization."""
        middleware = AdvancedMiddleware()
        assert middleware is not None
        assert len(middleware.tools) == 2  # finder, look_at

    def test_init_with_enabled_tools(self):
        """Test initialization with specific tools enabled."""
        middleware = AdvancedMiddleware(enabled_tools=["finder"])
        assert len(middleware.tools) == 1

    def test_tool_names(self):
        """Test that all expected tools are present."""
        middleware = AdvancedMiddleware()
        tool_names = {tool.name for tool in middleware.tools}
        assert tool_names == {"finder", "look_at"}


class TestSubagentSpecs:
    """Test subagent specification helpers."""

    def test_librarian_subagent_spec(self):
        """Test librarian subagent specification."""
        spec = get_librarian_subagent()
        assert spec["name"] == "librarian"
        assert "description" in spec
        assert "system_prompt" in spec
        assert "tools" in spec

    def test_oracle_subagent_spec(self):
        """Test oracle subagent specification."""
        spec = get_oracle_subagent()
        assert spec["name"] == "oracle"
        assert "description" in spec
        assert "system_prompt" in spec
        assert "tools" in spec
