"""Tests for append_cached_text and prompt caching utilities.

These tests need the real langchain_core.messages.SystemMessage and the real
deepagents.middleware._utils module.  Other test files (e.g. test_turn_context)
may mock these at sys.modules level, so we load _utils.py directly from its
file path using importlib.util to avoid any mock interference.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# 1. Guarantee real langchain_core is available
# ---------------------------------------------------------------------------
_lc_msgs = sys.modules.get("langchain_core.messages")
if _lc_msgs is not None and not hasattr(_lc_msgs, "__file__"):
    # A mock is installed — remove it (and the parent) so the real one loads.
    _saved_lc: dict[str, object] = {}
    for _name in list(sys.modules):
        if _name == "langchain_core" or _name.startswith("langchain_core."):
            _saved_lc[_name] = sys.modules.pop(_name)
    from langchain_core.messages import SystemMessage  # noqa: E402
    # Restore the mocks for other test modules that still need them
    for _name, _mod in _saved_lc.items():
        sys.modules.setdefault(_name, _mod)
else:
    from langchain_core.messages import SystemMessage  # noqa: E402

# ---------------------------------------------------------------------------
# 2. Load _utils.py directly from its file path (avoids deepagents.__init__)
# ---------------------------------------------------------------------------
_UTILS_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..",
        "vendor", "deepagents", "libs", "deepagents",
        "deepagents", "middleware", "_utils.py",
    )
)
_spec = importlib.util.spec_from_file_location("deepagents.middleware._utils", _UTILS_PATH)
_utils_mod = importlib.util.module_from_spec(_spec)
# Ensure SystemMessage is resolvable inside _utils when it does
# `from langchain_core.messages import SystemMessage`
_spec.loader.exec_module(_utils_mod)

append_to_system_message = _utils_mod.append_to_system_message
append_cached_text = _utils_mod.append_cached_text


class TestAppendCachedTextCreatesBlock:
    """test_append_cached_text_creates_cached_block"""

    def test_creates_block_with_cache_control(self):
        result = append_cached_text(None, "Identity block")
        assert isinstance(result, SystemMessage)
        blocks = result.content
        assert len(blocks) == 1
        block = blocks[0]
        assert block["type"] == "text"
        assert block["text"] == "Identity block"
        assert block["cache_control"] == {"type": "ephemeral"}


class TestAppendCachedTextNoCache:
    """test_append_cached_text_no_cache"""

    def test_no_cache_control_when_false(self):
        result = append_cached_text(None, "Volatile block", cache=False)
        blocks = result.content
        assert len(blocks) == 1
        block = blocks[0]
        assert block["type"] == "text"
        assert block["text"] == "Volatile block"
        assert "cache_control" not in block


class TestAppendCachedTextAppendsToExisting:
    """test_append_cached_text_appends_to_existing"""

    def test_appends_to_existing_system_message(self):
        existing = SystemMessage(content=[{"type": "text", "text": "Existing block"}])
        result = append_cached_text(existing, "New cached block", cache=True)
        blocks = result.content
        assert len(blocks) == 2
        # First block is the original
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Existing block"
        # Second block is the new cached one
        assert blocks[1]["type"] == "text"
        assert blocks[1]["text"].endswith("New cached block")
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_prepends_newlines_when_existing_blocks(self):
        existing = SystemMessage(content=[{"type": "text", "text": "First"}])
        result = append_cached_text(existing, "Second", cache=True)
        blocks = result.content
        # The new block text should start with \n\n
        assert blocks[1]["text"] == "\n\nSecond"


class TestAppendCachedTextNoneCreatesNew:
    """test_append_cached_text_none_creates_new"""

    def test_creates_new_system_message_from_none(self):
        result = append_cached_text(None, "Brand new", cache=True)
        assert isinstance(result, SystemMessage)
        blocks = result.content
        assert len(blocks) == 1
        assert blocks[0]["text"] == "Brand new"
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    def test_creates_new_without_cache(self):
        result = append_cached_text(None, "No cache", cache=False)
        assert isinstance(result, SystemMessage)
        blocks = result.content
        assert len(blocks) == 1
        assert blocks[0]["text"] == "No cache"
        assert "cache_control" not in blocks[0]


class TestExistingAppendUnchanged:
    """test_existing_append_unchanged — original append_to_system_message still works."""

    def test_no_cache_control_in_original(self):
        result = append_to_system_message(None, "Hello")
        blocks = result.content
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello"
        assert "cache_control" not in blocks[0]

    def test_appends_to_existing(self):
        existing = SystemMessage(content=[{"type": "text", "text": "First"}])
        result = append_to_system_message(existing, "Second")
        blocks = result.content
        assert len(blocks) == 2
        assert blocks[1]["text"] == "\n\nSecond"
        assert "cache_control" not in blocks[1]
