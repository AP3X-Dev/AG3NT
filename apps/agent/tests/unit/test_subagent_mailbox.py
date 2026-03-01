"""Tests for SubagentMailbox and its integration with the SubAgent TypedDict."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType


def _load_subagents_module():
    """Load ``deepagents.middleware.subagents`` directly from its file path,
    bypassing the ``deepagents/__init__.py`` which pulls in heavy langchain
    dependencies that are not installed in the test environment.
    """
    # Resolve the actual source file.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    subagents_path = os.path.join(
        repo_root, "vendor", "deepagents", "libs", "deepagents",
        "deepagents", "middleware", "subagents.py",
    )

    # Stub external dependencies that subagents.py imports at module level.
    _ext_stubs = {
        "langchain": {},
        "langchain.agents": {"create_agent": lambda *a, **kw: None},
        "langchain.agents.middleware": {
            "HumanInTheLoopMiddleware": type("HumanInTheLoopMiddleware", (), {}),
            "InterruptOnConfig": type("InterruptOnConfig", (), {}),
        },
        "langchain.agents.middleware.types": {
            "AgentMiddleware": type("AgentMiddleware", (), {"__init_subclass__": classmethod(lambda cls, **kw: None)}),
            "ModelRequest": type("ModelRequest", (), {}),
            "ModelResponse": type("ModelResponse", (), {}),
        },
        "langchain.tools": {
            "BaseTool": type("BaseTool", (), {}),
            "ToolRuntime": type("ToolRuntime", (), {}),
        },
        "langchain_core": {},
        "langchain_core.language_models": {
            "BaseChatModel": type("BaseChatModel", (), {}),
        },
        "langchain_core.messages": {
            "HumanMessage": type("HumanMessage", (), {}),
            "ToolMessage": type("ToolMessage", (), {}),
        },
        "langchain_core.runnables": {
            "Runnable": type("Runnable", (), {}),
        },
        "langchain_core.tools": {
            "StructuredTool": type("StructuredTool", (), {}),
        },
        "langgraph": {},
        "langgraph.types": {
            "Command": type("Command", (), {}),
        },
    }
    for name, attrs in _ext_stubs.items():
        if name not in sys.modules:
            mod = ModuleType(name)
            for attr_name, attr_val in attrs.items():
                setattr(mod, attr_name, attr_val)
            sys.modules[name] = mod

    # Stub internal deepagents helper.
    if "deepagents.middleware._utils" not in sys.modules:
        _utils = ModuleType("deepagents.middleware._utils")
        _utils.append_to_system_message = lambda msg, prompt: msg  # type: ignore[attr-defined]
        sys.modules["deepagents.middleware._utils"] = _utils

    # Register parent packages so Python considers the module as part of a package.
    deepagents_dir = os.path.join(repo_root, "vendor", "deepagents", "libs", "deepagents", "deepagents")
    middleware_dir = os.path.join(deepagents_dir, "middleware")

    for pkg_name, pkg_path in [("deepagents", deepagents_dir), ("deepagents.middleware", middleware_dir)]:
        if pkg_name not in sys.modules:
            pkg_mod = ModuleType(pkg_name)
            pkg_mod.__path__ = [pkg_path]  # type: ignore[attr-defined]
            pkg_mod.__package__ = pkg_name  # type: ignore[attr-defined]
            sys.modules[pkg_name] = pkg_mod

    # Load the module from the source file.
    spec = importlib.util.spec_from_file_location("deepagents.middleware.subagents", subagents_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deepagents.middleware.subagents"] = mod
    spec.loader.exec_module(mod)
    return mod


_subagents_mod = _load_subagents_module()

SubagentMailbox = _subagents_mod.SubagentMailbox
_EXCLUDED_STATE_KEYS = _subagents_mod._EXCLUDED_STATE_KEYS


# ---------------------------------------------------------------------------
# TestSubagentMailbox
# ---------------------------------------------------------------------------

class TestSubagentMailbox:
    """Unit tests for the SubagentMailbox data structure."""

    def test_create_mailbox(self):
        mb = SubagentMailbox()
        assert mb.empty() is True

    def test_send_and_receive(self):
        mb = SubagentMailbox()
        mb.send("parent", "hello child")
        assert mb.empty() is False
        msg = mb.receive()
        assert msg == {"sender": "parent", "content": "hello child"}
        assert mb.empty() is True

    def test_fifo_ordering(self):
        mb = SubagentMailbox()
        mb.send("parent", "first")
        mb.send("child", "second")
        assert mb.receive() == {"sender": "parent", "content": "first"}
        assert mb.receive() == {"sender": "child", "content": "second"}

    def test_receive_empty_returns_none(self):
        mb = SubagentMailbox()
        assert mb.receive() is None

    def test_drain_returns_all(self):
        mb = SubagentMailbox()
        mb.send("a", "msg1")
        mb.send("b", "msg2")
        msgs = mb.drain()
        assert len(msgs) == 2
        assert msgs[0] == {"sender": "a", "content": "msg1"}
        assert msgs[1] == {"sender": "b", "content": "msg2"}
        assert mb.empty() is True


# ---------------------------------------------------------------------------
# TestMailboxInjection
# ---------------------------------------------------------------------------

class TestMailboxInjection:
    """Tests that SubagentMailbox integrates with the SubAgent TypedDict."""

    def test_mailbox_in_subagent_typedef(self):
        """SubAgent TypedDict accepts an optional mailbox field."""
        SubAgent = _subagents_mod.SubAgent

        # TypedDict accepts the mailbox key without error.
        spec: SubAgent = {  # type: ignore[typeddict-item]
            "name": "test",
            "description": "test agent",
            "system_prompt": "do stuff",
            "tools": [],
            "mailbox": SubagentMailbox(),
        }
        assert isinstance(spec["mailbox"], SubagentMailbox)

    def test_mailbox_passed_through_state(self):
        """'mailbox' is NOT in _EXCLUDED_STATE_KEYS, so it passes through."""
        assert "mailbox" not in _EXCLUDED_STATE_KEYS
