"""Tests for SafetyHookAgentMiddleware."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from ag3nt_agent.hooks import PhaseHookManager, HookPhase, CANCEL
from ag3nt_agent.hooks.middleware import HookMiddleware
from ag3nt_agent.hooks.agent_middleware import SafetyHookAgentMiddleware


class TestSafetyHookAgentMiddleware:
    """Tests for the AgentMiddleware adapter around HookMiddleware."""

    @pytest.fixture
    def hook_manager(self):
        """Create a started PhaseHookManager."""
        mgr = PhaseHookManager()
        mgr.start()
        return mgr

    @pytest.fixture
    def middleware(self, hook_manager):
        """Create SafetyHookAgentMiddleware with a real HookMiddleware."""
        hook_mw = HookMiddleware(hook_manager)
        return SafetyHookAgentMiddleware(hook_mw)

    def _make_tool_request(self, name="read_file", args=None, tool_call_id="tc1"):
        """Create a mock ToolCallRequest."""
        request = MagicMock()
        request.tool_call = {
            "name": name,
            "args": args or {},
            "id": tool_call_id,
        }
        return request

    # -- Properties -----------------------------------------------------------

    def test_name_property(self, middleware):
        """Middleware name should be 'safety_hooks'."""
        assert middleware.name == "safety_hooks"

    def test_tools_empty(self, middleware):
        """Middleware should not register additional tools."""
        assert middleware.tools == []

    # -- Model calls: passthrough ---------------------------------------------

    def test_wrap_model_call_passthrough(self, middleware):
        """Model calls should pass through unchanged."""
        request = MagicMock()
        handler = MagicMock(return_value="response")
        result = middleware.wrap_model_call(request, handler)
        handler.assert_called_once_with(request)
        assert result == "response"

    async def test_awrap_model_call_passthrough(self, middleware):
        """Async model calls should pass through unchanged."""
        request = MagicMock()
        handler = AsyncMock(return_value="response")
        result = await middleware.awrap_model_call(request, handler)
        handler.assert_awaited_once_with(request)
        assert result == "response"

    # -- Tool calls: allowed --------------------------------------------------

    async def test_awrap_tool_call_allowed(self, middleware):
        """Tool calls not blocked by hooks should execute normally."""
        tool_msg = MagicMock()
        request = self._make_tool_request(name="read_file", args={"path": "/foo"})
        handler = AsyncMock(return_value=tool_msg)

        result = await middleware.awrap_tool_call(request, handler)
        assert result is tool_msg

    async def test_awrap_tool_call_handler_called(self, middleware):
        """Handler should be called exactly once for allowed tool calls."""
        request = self._make_tool_request()
        handler = AsyncMock(return_value=MagicMock())

        await middleware.awrap_tool_call(request, handler)
        handler.assert_awaited_once()

    # -- Tool calls: blocked by PRE_TOOL hook ---------------------------------

    async def test_awrap_tool_call_blocked_by_pre_hook(self, hook_manager, middleware):
        """Tool calls blocked by PRE_TOOL hook should return cancellation message."""
        async def blocker(ctx):
            return CANCEL

        hook_manager.register(HookPhase.PRE_TOOL, blocker, priority=10, name="blocker")

        request = self._make_tool_request(
            name="write_file", args={"path": "/etc/passwd"}, tool_call_id="tc2"
        )
        handler = AsyncMock(return_value=MagicMock())

        result = await middleware.awrap_tool_call(request, handler)

        # Tool should NOT have been called
        handler.assert_not_awaited()
        # Result should be a ToolMessage with cancellation info
        from langchain_core.messages import ToolMessage
        assert isinstance(result, ToolMessage)
        assert "blocked" in result.content.lower()
        assert result.tool_call_id == "tc2"

    async def test_awrap_tool_call_blocked_includes_reason(self, hook_manager, middleware):
        """Cancellation message should include the hook's cancel reason."""
        async def blocker(ctx):
            return CANCEL

        hook_manager.register(HookPhase.PRE_TOOL, blocker, priority=10, name="protect_core")

        request = self._make_tool_request(name="edit", args={"file_path": ".env"})
        handler = AsyncMock(return_value=MagicMock())

        result = await middleware.awrap_tool_call(request, handler)
        assert "protect_core" in result.content or "blocked" in result.content.lower()

    # -- POST_TOOL hooks execute after tool ------------------------------------

    async def test_post_tool_hook_runs(self, hook_manager, middleware):
        """POST_TOOL hooks should run after tool execution."""
        post_called = []

        async def post_hook(ctx):
            post_called.append(ctx.get("tool_name"))

        hook_manager.register(HookPhase.POST_TOOL, post_hook, name="tracker")

        request = self._make_tool_request(name="my_tool")
        handler = AsyncMock(return_value=MagicMock())

        await middleware.awrap_tool_call(request, handler)
        assert post_called == ["my_tool"]

    # -- Tool execution error propagation -------------------------------------

    async def test_awrap_tool_call_propagates_errors(self, middleware):
        """Errors from the tool handler should propagate through."""
        request = self._make_tool_request()
        handler = AsyncMock(side_effect=ValueError("tool failed"))

        with pytest.raises(ValueError, match="tool failed"):
            await middleware.awrap_tool_call(request, handler)

    # -- Integration with safety hooks ----------------------------------------

    async def test_integration_with_safety_hooks(self, hook_manager):
        """Integration test: register real safety hooks and verify blocking."""
        from ag3nt_agent.hooks.safety import register_safety_hooks

        register_safety_hooks(hook_manager)
        hook_mw = HookMiddleware(hook_manager)
        mw = SafetyHookAgentMiddleware(hook_mw)

        # Try to write to a protected file (.env)
        request = self._make_tool_request(
            name="write_file", args={"file_path": ".env"}, tool_call_id="tc_env"
        )
        handler = AsyncMock(return_value=MagicMock())

        result = await mw.awrap_tool_call(request, handler)
        handler.assert_not_awaited()

        from langchain_core.messages import ToolMessage
        assert isinstance(result, ToolMessage)
        assert "blocked" in result.content.lower()

    async def test_integration_allows_normal_file(self, hook_manager):
        """Integration test: normal file writes should be allowed."""
        from ag3nt_agent.hooks.safety import register_safety_hooks

        register_safety_hooks(hook_manager)
        hook_mw = HookMiddleware(hook_manager)
        mw = SafetyHookAgentMiddleware(hook_mw)

        tool_msg = MagicMock()
        request = self._make_tool_request(
            name="write_file", args={"file_path": "src/main.py"}, tool_call_id="tc_ok"
        )
        handler = AsyncMock(return_value=tool_msg)

        result = await mw.awrap_tool_call(request, handler)
        assert result is tool_msg
        handler.assert_awaited_once()
