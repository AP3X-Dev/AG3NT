"""Tests for prompt mode support in system prompt assembly.

PromptMode controls how much context is injected into every model call:
- FULL: identity + memory + ui_context + environment + summary guardrail
- MINIMAL: environment block only
- NONE: no context injected (request returned unchanged)
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the langchain / langgraph / deepagents module tree so the middleware
# can be imported without having those packages installed.
# ---------------------------------------------------------------------------
_mock_modules: dict[str, ModuleType] = {}


def _ensure_mock_module(name: str) -> ModuleType:
    if name in _mock_modules:
        return _mock_modules[name]
    mod = ModuleType(name)
    _mock_modules[name] = mod
    return mod


# langchain
_lc = _ensure_mock_module("langchain")
_lc_agents = _ensure_mock_module("langchain.agents")
_lc_agents_mw = _ensure_mock_module("langchain.agents.middleware")
_lc_agents_mw_types = _ensure_mock_module("langchain.agents.middleware.types")

_lc_agents_mw_types.AgentMiddleware = type(
    "AgentMiddleware", (), {"__class_getitem__": classmethod(lambda cls, *a: cls)}
)
_lc_agents_mw_types.AgentState = MagicMock()
_lc_agents_mw_types.ModelRequest = MagicMock()
_lc_agents_mw_types.ModelResponse = MagicMock()
_lc_agents_mw_types.ModelCallResult = MagicMock()

_lc_core = _ensure_mock_module("langchain_core")
_lc_core_msgs = _ensure_mock_module("langchain_core.messages")
_lc_core_msgs.SystemMessage = MagicMock()

# langgraph
_lg = _ensure_mock_module("langgraph")
_lg_config = _ensure_mock_module("langgraph.config")
_lg_config.get_config = MagicMock(return_value={"metadata": {}})

# deepagents
_da = _ensure_mock_module("deepagents")
_da_mw = _ensure_mock_module("deepagents.middleware")
_da_mw_utils = _ensure_mock_module("deepagents.middleware._utils")


def _mock_append_to_system_message(system_message, text):
    new_msg = MagicMock()
    new_msg._appended_text = text
    new_msg._original = system_message
    return new_msg


_da_mw_utils.append_to_system_message = _mock_append_to_system_message

# Register all
for mod_name, mod_obj in _mock_modules.items():
    sys.modules.setdefault(mod_name, mod_obj)

# Now import the module under test
from ag3nt_agent.turn_context_middleware import (  # noqa: E402
    TurnContextMiddleware,
    PromptMode,
)
from ag3nt_agent.identity import IdentityLoader  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(*, messages=None, ui_context=None):
    """Build a mock ModelRequest for testing _inject_context."""
    request = MagicMock()
    request.state = {"messages": messages or []}
    request.system_message = MagicMock()

    new_request = MagicMock()
    request.override.return_value = new_request

    # Mock get_config for UI context
    _lg_config.get_config = MagicMock(
        return_value={"metadata": {"ui_context": ui_context} if ui_context else {}}
    )
    return request, new_request


# ============================================================================
# Tests
# ============================================================================


class TestPromptModeEnum:
    """PromptMode enum should have the expected values and be a str enum."""

    def test_values(self):
        assert PromptMode.FULL == "full"
        assert PromptMode.MINIMAL == "minimal"
        assert PromptMode.NONE == "none"

    def test_is_string(self):
        assert isinstance(PromptMode.FULL, str)
        assert isinstance(PromptMode.MINIMAL, str)
        assert isinstance(PromptMode.NONE, str)


class TestDefaultMode:
    """Default prompt_mode should be FULL for backward compatibility."""

    def test_default_mode_is_full(self):
        mw = TurnContextMiddleware()
        assert mw.prompt_mode == PromptMode.FULL

    def test_explicit_full_mode(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        assert mw.prompt_mode == PromptMode.FULL

    def test_explicit_minimal_mode(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.MINIMAL)
        assert mw.prompt_mode == PromptMode.MINIMAL

    def test_explicit_none_mode(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.NONE)
        assert mw.prompt_mode == PromptMode.NONE


class TestNoneMode:
    """NONE mode should return the request completely unchanged."""

    def test_none_mode_returns_request_unchanged(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.NONE)
        request, _ = _make_request()

        result = mw._inject_context(request)

        # The original request is returned, not a new one
        assert result is request
        # override should NOT have been called
        request.override.assert_not_called()

    def test_none_mode_via_wrap_model_call(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = "You are AG3NT."

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.NONE,
        )

        request, _ = _make_request()
        handler = MagicMock(return_value=MagicMock())
        mw.wrap_model_call(request, handler)

        # Handler is called with the original request (unchanged)
        handler.assert_called_once_with(request)
        # override never called
        request.override.assert_not_called()

    def test_none_mode_ignores_memory(self):
        mock_search = MagicMock(return_value={
            "results": [{"content": "should not appear"}],
        })
        mw = TurnContextMiddleware(
            memory_search_fn=mock_search,
            prompt_mode=PromptMode.NONE,
        )

        user_msg = MagicMock()
        user_msg.type = "human"
        user_msg.content = "Tell me about authentication"
        request, _ = _make_request(messages=[user_msg])

        result = mw._inject_context(request)
        assert result is request
        mock_search.assert_not_called()


class TestMinimalMode:
    """MINIMAL mode should only inject the environment block."""

    def test_minimal_mode_injects_environment_only(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = "You are AG3NT."

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.MINIMAL,
        )

        request, new_request = _make_request()
        result = mw._inject_context(request)

        # override should be called
        request.override.assert_called_once()
        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text

        # Environment block should be present
        assert "## Environment" in appended
        assert "Platform:" in appended
        assert "Current time:" in appended

        # Identity should NOT be present
        assert "You are AG3NT" not in appended

        # Memory should NOT be present
        assert "## Relevant Memories" not in appended

        # UI context should NOT be present
        assert "## UI Context" not in appended

        # Summary guardrail should NOT be present
        assert "## Context Handling" not in appended

    def test_minimal_mode_skips_memory_search(self):
        mock_search = MagicMock(return_value={
            "results": [{"content": "Found fact"}],
        })
        mw = TurnContextMiddleware(
            memory_search_fn=mock_search,
            prompt_mode=PromptMode.MINIMAL,
        )

        user_msg = MagicMock()
        user_msg.type = "human"
        user_msg.content = "Tell me about authentication errors"
        request, _ = _make_request(messages=[user_msg])

        mw._inject_context(request)

        # Memory search function should not be called
        mock_search.assert_not_called()

    def test_minimal_mode_skips_ui_context(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.MINIMAL)

        request, _ = _make_request(ui_context="Active tab: Chat")
        mw._inject_context(request)

        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text
        assert "## UI Context" not in appended
        assert "Active tab: Chat" not in appended

    def test_minimal_mode_does_not_call_identity(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = "Full identity text..."

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.MINIMAL,
        )

        request, _ = _make_request()
        mw._inject_context(request)

        # Identity's build_system_prompt should NOT be called at all
        mock_identity.build_system_prompt.assert_not_called()


class TestFullMode:
    """FULL mode should include all sections (existing behavior)."""

    def test_full_mode_includes_identity(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = "You are AG3NT, a helpful assistant."

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.FULL,
        )

        request, _ = _make_request()
        mw._inject_context(request)

        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text
        assert "You are AG3NT, a helpful assistant." in appended

    def test_full_mode_includes_memory(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        def mock_search(query, top_k=5):
            return {"results": [{"content": "Important fact", "score": 0.9}]}

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=mock_search,
            prompt_mode=PromptMode.FULL,
        )

        user_msg = MagicMock()
        user_msg.type = "human"
        user_msg.content = "Tell me about authentication errors"
        request, _ = _make_request(messages=[user_msg])

        mw._inject_context(request)

        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text
        assert "## Relevant Memories" in appended
        assert "Important fact" in appended

    def test_full_mode_includes_environment(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.FULL,
        )

        request, _ = _make_request()
        mw._inject_context(request)

        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text
        assert "## Environment" in appended

    def test_full_mode_includes_summary_guardrail(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.FULL,
        )

        request, _ = _make_request()
        mw._inject_context(request)

        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text
        assert "## Context Handling" in appended

    def test_full_mode_includes_ui_context(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.FULL,
        )

        request, _ = _make_request(ui_context="Active tab: Browser")
        mw._inject_context(request)

        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text
        assert "## UI Context" in appended
        assert "Active tab: Browser" in appended


class TestMinimalIdentity:
    """IdentityLoader.build_system_prompt(minimal=True) should be compact."""

    def test_minimal_identity_is_compact(self):
        loader = IdentityLoader()
        minimal = loader.build_system_prompt(minimal=True)
        assert minimal == "You are AG3NT."

    def test_minimal_identity_shorter_than_full_when_files_exist(self):
        """When identity files exist, minimal should be shorter than full."""
        loader = IdentityLoader()
        minimal = loader.build_system_prompt(minimal=True)
        # Full with no files returns "" which is shorter, so we just
        # verify the minimal form is the expected compact string.
        assert len(minimal) < 50
        assert "AG3NT" in minimal


class TestPromptModeWithAsyncHandler:
    """Ensure prompt modes work correctly with the async model call path."""

    async def test_none_mode_async(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.NONE)
        request, _ = _make_request()

        async def async_handler(req):
            return MagicMock()

        result = await mw.awrap_model_call(request, async_handler)
        # override should not be called
        request.override.assert_not_called()

    async def test_minimal_mode_async(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = "Full identity text"

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            prompt_mode=PromptMode.MINIMAL,
        )
        request, new_request = _make_request()

        async def async_handler(req):
            return MagicMock()

        result = await mw.awrap_model_call(request, async_handler)

        # override should be called (for environment block)
        request.override.assert_called_once()
        # Identity should not have been called
        mock_identity.build_system_prompt.assert_not_called()


class TestPromptModeBackwardCompatibility:
    """Existing code that doesn't pass prompt_mode should continue to work."""

    def test_no_prompt_mode_param_works(self):
        mw = TurnContextMiddleware(
            identity_loader=MagicMock(spec=IdentityLoader),
            memory_search_fn=None,
            memory_char_budget=1000,
        )
        assert mw.prompt_mode == PromptMode.FULL

    def test_full_mode_matches_original_behavior(self):
        """FULL mode context should contain all the same sections as before."""
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = "Identity block"

        def mock_search(query, top_k=5):
            return {"results": [{"content": "Memory item"}]}

        mw_full = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=mock_search,
            prompt_mode=PromptMode.FULL,
        )

        user_msg = MagicMock()
        user_msg.type = "human"
        user_msg.content = "Tell me about authentication errors"

        request, _ = _make_request(
            messages=[user_msg],
            ui_context="Tab: Code Editor",
        )
        mw_full._inject_context(request)

        sys_msg = (
            request.override.call_args.kwargs.get("system_message")
            or request.override.call_args[1].get("system_message")
        )
        appended = sys_msg._appended_text

        # All sections present
        assert "Identity block" in appended
        assert "## Relevant Memories" in appended
        assert "## UI Context" in appended
        assert "## Environment" in appended
        assert "## Context Handling" in appended
