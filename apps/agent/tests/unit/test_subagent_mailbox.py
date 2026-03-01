"""Tests for SubagentMailbox inter-agent messaging.

The test environment may not have langchain installed.  We use importlib
to load ``subagents.py`` directly from its file path, bypassing the
``deepagents.__init__`` import chain that pulls in langchain.

SubagentMailbox is a pure-Python data structure with zero external
dependencies, so direct-loading the module is safe.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Strategy: load subagents.py directly from disk so we never trigger
# deepagents/__init__.py (which imports langchain).  We stub only the
# external modules that subagents.py itself imports.
# ---------------------------------------------------------------------------

_mock_modules: dict[str, ModuleType] = {}


def _mock(name: str, **attrs: object) -> ModuleType:
    """Create or retrieve a stub module and set attributes on it."""
    if name not in _mock_modules:
        mod = ModuleType(name)
        mod.__dict__.setdefault("__path__", [])
        _mock_modules[name] = mod
    m = _mock_modules[name]
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


# Check whether we can import subagents normally
try:
    from deepagents.middleware.subagents import SubagentMailbox as _probe  # noqa: F401
    _NEED_MOCKS = False
except Exception:
    _NEED_MOCKS = True


def _load_subagents_module() -> ModuleType:
    """Load subagents.py directly, mocking its dependencies."""
    # -- langchain ---------------------------------------------------------
    _mock("langchain")
    _mock("langchain.agents", create_agent=MagicMock())
    _mock(
        "langchain.agents.middleware",
        HumanInTheLoopMiddleware=MagicMock(),
        InterruptOnConfig=MagicMock(),
    )
    _mock(
        "langchain.agents.middleware.types",
        AgentMiddleware=MagicMock(),
        ContextT=MagicMock(),
        ModelRequest=MagicMock(),
        ModelResponse=MagicMock(),
        ResponseT=MagicMock(),
    )
    _mock("langchain.chat_models", init_chat_model=MagicMock())
    _mock("langchain.tools", BaseTool=MagicMock(), ToolRuntime=MagicMock())

    # -- langchain_core ----------------------------------------------------
    _mock("langchain_core")
    _mock("langchain_core.language_models", BaseChatModel=MagicMock())
    _mock(
        "langchain_core.messages",
        HumanMessage=MagicMock(),
        ToolMessage=MagicMock(),
        SystemMessage=MagicMock(),
    )
    _mock("langchain_core.runnables", Runnable=MagicMock())
    _mock("langchain_core.tools", BaseTool=MagicMock(), StructuredTool=MagicMock())

    # -- langgraph ---------------------------------------------------------
    _mock("langgraph")
    _mock("langgraph.types", Command=MagicMock())

    # -- deepagents internal deps of subagents.py --------------------------
    _mock("deepagents")
    _mock("deepagents.backends")
    _mock(
        "deepagents.backends.protocol",
        BackendFactory=MagicMock(),
        BackendProtocol=MagicMock(),
    )
    _mock("deepagents.middleware")
    _mock("deepagents.middleware._utils", append_to_system_message=MagicMock())

    # Temporarily inject mocks into sys.modules, saving originals
    saved: dict[str, ModuleType | None] = {}
    for name, mod in _mock_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        # Locate subagents.py on disk
        base = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..",
            "vendor", "deepagents", "libs", "deepagents", "deepagents",
            "middleware", "subagents.py",
        )
        base = os.path.normpath(base)

        spec = importlib.util.spec_from_file_location(
            "deepagents.middleware.subagents", base
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules["deepagents.middleware.subagents"] = loaded
        spec.loader.exec_module(loaded)
    finally:
        # Restore original sys.modules to avoid polluting other tests.
        # Keep "deepagents.middleware.subagents" so our imported names work.
        for name, orig in saved.items():
            if name == "deepagents.middleware.subagents":
                continue
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

    return loaded


if _NEED_MOCKS:
    _subagents_mod = _load_subagents_module()
    SubagentMailbox = _subagents_mod.SubagentMailbox
    SubAgent = _subagents_mod.SubAgent
    _EXCLUDED_STATE_KEYS = _subagents_mod._EXCLUDED_STATE_KEYS
else:
    from deepagents.middleware.subagents import (  # type: ignore[assignment]
        SubagentMailbox,
        SubAgent,
        _EXCLUDED_STATE_KEYS,
    )


class TestSubagentMailbox:
    """Tests for the SubagentMailbox FIFO queue."""

    def test_create_mailbox(self) -> None:
        """A freshly created mailbox should be empty."""
        mb = SubagentMailbox()
        assert mb.empty() is True

    def test_send_and_receive(self) -> None:
        """send() enqueues a message; receive() dequeues it."""
        mb = SubagentMailbox()
        mb.send("parent", "hello child")
        assert not mb.empty()

        msg = mb.receive()
        assert msg == {"sender": "parent", "content": "hello child"}
        assert mb.empty() is True

    def test_fifo_ordering(self) -> None:
        """Messages are received in FIFO order."""
        mb = SubagentMailbox()
        mb.send("parent", "first")
        mb.send("child", "second")

        assert mb.receive() == {"sender": "parent", "content": "first"}
        assert mb.receive() == {"sender": "child", "content": "second"}

    def test_receive_empty_returns_none(self) -> None:
        """receive() on an empty mailbox returns None."""
        mb = SubagentMailbox()
        assert mb.receive() is None

    def test_drain_returns_all(self) -> None:
        """drain() returns all messages and empties the mailbox."""
        mb = SubagentMailbox()
        mb.send("a", "msg1")
        mb.send("b", "msg2")

        msgs = mb.drain()
        assert len(msgs) == 2
        assert msgs[0] == {"sender": "a", "content": "msg1"}
        assert msgs[1] == {"sender": "b", "content": "msg2"}
        assert mb.empty() is True


class TestMailboxInjection:
    """Tests for mailbox integration with SubAgent TypedDict and state passing."""

    def test_mailbox_in_subagent_typedef(self) -> None:
        """SubAgent TypedDict accepts an optional mailbox field."""
        mb = SubagentMailbox()
        spec: SubAgent = {
            "name": "test",
            "description": "test agent",
            "system_prompt": "you are a test agent",
            "mailbox": mb,
        }
        assert spec["mailbox"] is mb

    def test_mailbox_passed_through_state(self) -> None:
        """'mailbox' should NOT be in _EXCLUDED_STATE_KEYS so it passes to subagents."""
        assert "mailbox" not in _EXCLUDED_STATE_KEYS
