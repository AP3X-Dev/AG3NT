"""Tests for TurnContextMiddleware.

Since langchain is not installed in the test environment, we mock the
langchain module hierarchy at sys.modules level before importing the
middleware under test.
"""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the langchain module tree so the middleware can be imported without
# having langchain installed.
# ---------------------------------------------------------------------------
_mock_modules: dict[str, ModuleType] = {}


def _ensure_mock_module(name: str) -> ModuleType:
    """Get existing mock module from sys.modules or create a new one."""
    if name in sys.modules:
        return sys.modules[name]
    if name in _mock_modules:
        return _mock_modules[name]
    mod = ModuleType(name)
    _mock_modules[name] = mod
    sys.modules[name] = mod
    return mod


# Build the module tree
_lc = _ensure_mock_module("langchain")
_lc_agents = _ensure_mock_module("langchain.agents")
_lc_agents_mw = _ensure_mock_module("langchain.agents.middleware")
_lc_agents_mw_types = _ensure_mock_module("langchain.agents.middleware.types")

# Provide the types the middleware imports
_lc_agents_mw_types.AgentMiddleware = type("AgentMiddleware", (), {"__class_getitem__": classmethod(lambda cls, *a: cls)})
_lc_agents_mw_types.AgentState = MagicMock()
_lc_agents_mw_types.ModelRequest = MagicMock()
_lc_agents_mw_types.ModelResponse = MagicMock()
_lc_agents_mw_types.ModelCallResult = MagicMock()

_lc_core = _ensure_mock_module("langchain_core")
_lc_core_msgs = _ensure_mock_module("langchain_core.messages")
_lc_core_msgs.SystemMessage = MagicMock()

# langgraph config mock (get_config used for UI context injection)
_lg = _ensure_mock_module("langgraph")
_lg_config = _ensure_mock_module("langgraph.config")
_lg_config.get_config = MagicMock(return_value={"metadata": {}})

# deepagents middleware mock
_da = _ensure_mock_module("deepagents")
_da_mw = _ensure_mock_module("deepagents.middleware")
_da_mw_utils = _ensure_mock_module("deepagents.middleware._utils")


def _mock_append_to_system_message(system_message, text):
    """Fake append — returns a new mock with the text attached."""
    new_msg = MagicMock()
    new_msg._appended_text = text
    new_msg._original = system_message
    return new_msg


_da_mw_utils.append_to_system_message = _mock_append_to_system_message

# Register all mock modules
for mod_name, mod_obj in _mock_modules.items():
    sys.modules.setdefault(mod_name, mod_obj)

# Now we can safely import the module under test
from ag3nt_agent.turn_context_middleware import TurnContextMiddleware  # noqa: E402
from ag3nt_agent.identity import IdentityLoader  # noqa: E402


# ============================================================================
# Tests
# ============================================================================


class TestExtractLatestUserMessage:
    def test_extracts_last_human_message(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        msg1 = MagicMock()
        msg1.type = "human"
        msg1.content = "first message"
        msg2 = MagicMock()
        msg2.type = "human"
        msg2.content = "second message"
        msg3 = MagicMock()
        msg3.type = "ai"
        msg3.content = "response"
        messages = [msg1, msg3, msg2]
        result = mw._extract_latest_user_message(messages)
        assert result == "second message"

    def test_returns_empty_when_no_human(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        msg = MagicMock()
        msg.type = "ai"
        msg.content = "response"
        result = mw._extract_latest_user_message([msg])
        assert result == ""

    def test_handles_list_content_blocks(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        msg = MagicMock()
        msg.type = "human"
        msg.content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
        ]
        result = mw._extract_latest_user_message([msg])
        assert result == "Hello world"

    def test_handles_empty_list(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        result = mw._extract_latest_user_message([])
        assert result == ""


class TestExtractTopics:
    def test_filters_short_words(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        topics = mw._extract_topics("How do I fix the error?")
        assert "error" in topics

    def test_filters_stopwords(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        topics = mw._extract_topics("Please help with authentication system")
        assert "please" not in topics
        assert "help" not in topics
        assert "authentication" in topics
        assert "system" in topics

    def test_strips_punctuation(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        topics = mw._extract_topics("Check the error! And the warning.")
        assert "error" in topics
        assert "warning" in topics

    def test_empty_text(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        topics = mw._extract_topics("")
        assert topics == []


class TestFormatMemoryContext:
    def test_formats_results_with_budget(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        results = [
            {"content": "Memory chunk 1", "score": 0.9},
            {"content": "Memory chunk 2", "score": 0.8},
        ]
        text = mw._format_memory_context(results, max_chars=2000)
        assert "## Relevant Memories" in text
        assert "Memory chunk 1" in text

    def test_truncates_at_budget(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        results = [
            {"content": "x" * 1500, "score": 0.9},
            {"content": "y" * 1500, "score": 0.8},
        ]
        text = mw._format_memory_context(results, max_chars=2000)
        assert "x" * 100 in text
        assert "y" * 100 not in text

    def test_returns_empty_for_no_results(self):
        mw = TurnContextMiddleware.__new__(TurnContextMiddleware)
        text = mw._format_memory_context([], max_chars=2000)
        assert text == ""


class TestRecallMemory:
    def test_calls_memory_search_with_topics(self):
        mock_search = MagicMock(return_value={
            "results": [{"content": "Found fact", "score": 0.9}],
        })
        mw = TurnContextMiddleware(
            identity_loader=MagicMock(),
            memory_search_fn=mock_search,
        )
        result = mw._recall_memory("Tell me about authentication errors")
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        # Should extract topics and pass them as query
        assert "authentication" in call_args[0][0]
        assert "errors" in call_args[0][0]
        assert "## Relevant Memories" in result

    def test_returns_empty_on_exception(self):
        mock_search = MagicMock(side_effect=RuntimeError("search broke"))
        mw = TurnContextMiddleware(
            identity_loader=MagicMock(),
            memory_search_fn=mock_search,
        )
        result = mw._recall_memory("anything")
        assert result == ""

    def test_falls_back_to_full_text_if_no_topics(self):
        """If no topics are extracted (all short/stopwords), use the raw text."""
        mock_search = MagicMock(return_value={"results": []})
        mw = TurnContextMiddleware(
            identity_loader=MagicMock(),
            memory_search_fn=mock_search,
        )
        mw._recall_memory("hi do it")
        call_args = mock_search.call_args
        assert call_args[0][0] == "hi do it"


class TestEnvironmentBlock:
    def test_contains_platform(self):
        import platform as plat
        text = TurnContextMiddleware._environment_block()
        assert "## Environment" in text
        assert plat.system() in text

    def test_contains_timestamp(self):
        text = TurnContextMiddleware._environment_block()
        assert "Current time:" in text


class TestWrapModelCall:
    def test_prepends_identity_and_memory(self):
        from ag3nt_agent.identity import IdentityLoader

        mock_identity = MagicMock(spec=IdentityLoader)
        mock_identity.build_system_prompt.return_value = "You are AG3NT."

        def mock_search(query, top_k=5):
            return {"results": [{"content": "Remembered fact", "score": 0.9}]}

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=mock_search,
        )

        # Build a mock request with dict-like .state
        request = MagicMock()
        user_msg = MagicMock()
        user_msg.type = "human"
        user_msg.content = "Tell me about authentication"
        request.state = {"messages": [user_msg]}
        request.system_message = MagicMock()

        # request.override should return a new object
        new_request = MagicMock()
        request.override.return_value = new_request

        handler = MagicMock(return_value=MagicMock())

        mw.wrap_model_call(request, handler)

        # Verify handler was called
        handler.assert_called_once()

        # Verify request.override was called (meaning context was injected)
        request.override.assert_called_once()

        # Verify handler received the new (overridden) request
        call_args = handler.call_args[0][0]
        assert call_args is new_request

    def test_calls_handler_without_override_when_identity_empty(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=None,
        )

        request = MagicMock()
        request.state = {"messages": []}
        request.system_message = MagicMock()

        new_request = MagicMock()
        request.override.return_value = new_request

        handler = MagicMock(return_value=MagicMock())

        mw.wrap_model_call(request, handler)

        handler.assert_called_once()
        # Even with empty identity, the environment block is still injected
        request.override.assert_called_once()

    def test_identity_failure_is_graceful(self):
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.side_effect = RuntimeError("identity broke")

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=None,
        )

        request = MagicMock()
        request.state = {"messages": []}
        request.system_message = MagicMock()
        request.override.return_value = MagicMock()

        handler = MagicMock(return_value=MagicMock())

        # Should not raise
        mw.wrap_model_call(request, handler)
        handler.assert_called_once()


class TestAwrapModelCall:
    async def test_async_handler_called_with_modified_request(self):
        from ag3nt_agent.identity import IdentityLoader

        mock_identity = MagicMock(spec=IdentityLoader)
        mock_identity.build_system_prompt.return_value = "You are AG3NT."

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=None,
        )

        request = MagicMock()
        request.state = {"messages": []}
        request.system_message = MagicMock()

        new_request = MagicMock()
        request.override.return_value = new_request

        async def async_handler(req):
            return MagicMock()

        result = await mw.awrap_model_call(request, async_handler)
        # override was called to inject environment block at minimum
        request.override.assert_called_once()


class TestUIContextInjection:
    def test_ui_context_injected_from_config_metadata(self):
        """UI context from config metadata should be injected into system prompt."""
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=None,
        )

        request = MagicMock()
        request.state = {"messages": []}
        request.system_message = MagicMock()
        new_request = MagicMock()
        request.override.return_value = new_request

        # Mock get_config to return ui_context in metadata
        _lg_config.get_config = MagicMock(return_value={
            "metadata": {"ui_context": "Active tab: Chat - working on auth"}
        })

        handler = MagicMock(return_value=MagicMock())
        mw.wrap_model_call(request, handler)

        # The override should have been called with a system message
        request.override.assert_called_once()
        # The appended text should contain UI Context
        call_kwargs = request.override.call_args
        sys_msg = call_kwargs.kwargs.get("system_message") or call_kwargs[1].get("system_message")
        assert sys_msg is not None
        # The mock append_to_system_message stores the appended text
        assert "UI Context" in sys_msg._appended_text
        assert "Active tab: Chat" in sys_msg._appended_text

    def test_ui_context_not_injected_when_absent(self):
        """When no ui_context in metadata, nothing extra should be injected."""
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=None,
        )

        request = MagicMock()
        request.state = {"messages": []}
        request.system_message = MagicMock()
        new_request = MagicMock()
        request.override.return_value = new_request

        # Mock get_config with empty metadata
        _lg_config.get_config = MagicMock(return_value={"metadata": {}})

        handler = MagicMock(return_value=MagicMock())
        mw.wrap_model_call(request, handler)

        request.override.assert_called_once()
        sys_msg = request.override.call_args.kwargs.get("system_message") or request.override.call_args[1].get("system_message")
        assert "UI Context" not in sys_msg._appended_text

    def test_ui_context_graceful_on_get_config_failure(self):
        """If get_config raises, UI context injection should be skipped silently."""
        mock_identity = MagicMock()
        mock_identity.build_system_prompt.return_value = ""

        mw = TurnContextMiddleware(
            identity_loader=mock_identity,
            memory_search_fn=None,
        )

        request = MagicMock()
        request.state = {"messages": []}
        request.system_message = MagicMock()
        request.override.return_value = MagicMock()

        # Make get_config raise
        _lg_config.get_config = MagicMock(side_effect=RuntimeError("no config"))

        handler = MagicMock(return_value=MagicMock())
        # Should not raise
        mw.wrap_model_call(request, handler)
        handler.assert_called_once()


class TestInit:
    def test_default_identity_loader(self):
        mw = TurnContextMiddleware()
        assert isinstance(mw._identity, IdentityLoader)
        assert mw._memory_search is None
        assert mw._memory_budget == 2000
        assert mw.tools == []

    def test_custom_params(self):
        mock_id = MagicMock()
        mock_search = MagicMock()
        mw = TurnContextMiddleware(
            identity_loader=mock_id,
            memory_search_fn=mock_search,
            memory_char_budget=500,
        )
        assert mw._identity is mock_id
        assert mw._memory_search is mock_search
        assert mw._memory_budget == 500
