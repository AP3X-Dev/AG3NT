"""Tests for CompactionMiddleware wiring in run_turn / resume_turn.

The test environment may not have langchain installed.  The
CompactionMiddleware tests (TestCompactionMiddlewareBasics) mock the
langchain module tree so the middleware can be tested directly.

The _maybe_compact_history helper lives in deepagents_runtime which
has heavy langchain imports, so those tests are skipped when langchain
is unavailable (same pattern as test_autonomous_bootstrap.py).
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the langchain_core module tree so compaction_middleware can be imported
# without langchain installed.
# ---------------------------------------------------------------------------
_mock_modules: dict[str, ModuleType] = {}


def _ensure_mock_module(name: str) -> ModuleType:
    """Create or return a mock module and register it in sys.modules."""
    if name in _mock_modules:
        return _mock_modules[name]
    mod = ModuleType(name)
    _mock_modules[name] = mod
    return mod


# langchain_core.messages + utils (needed by compaction_middleware.py)
_lc_core = _ensure_mock_module("langchain_core")
_lc_core_msgs = _ensure_mock_module("langchain_core.messages")
_lc_core_msgs.AnyMessage = MagicMock()
_lc_core_msgs.HumanMessage = MagicMock()
_lc_core_msgs.AIMessage = MagicMock()
_lc_core_msgs.SystemMessage = MagicMock()

_lc_core_msgs_utils = _ensure_mock_module("langchain_core.messages.utils")


def _fake_count_tokens(msgs):
    """Approximate token count: 100 tokens per message."""
    return len(msgs) * 100


_lc_core_msgs_utils.count_tokens_approximately = _fake_count_tokens

# Register all mock modules
for _mod_name, _mod_obj in _mock_modules.items():
    sys.modules.setdefault(_mod_name, _mod_obj)

# Now we can safely import CompactionMiddleware
from ag3nt_agent.compaction_middleware import (  # noqa: E402
    CompactionConfig,
    CompactionMetrics,
    CompactionMiddleware,
    get_compaction_middleware,
    reset_compaction_middleware,
)

# Try importing the runtime helper — may fail if full langchain is absent
try:
    from ag3nt_agent.deepagents_runtime import _maybe_compact_history

    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False


# ============================================================================
# CompactionMiddleware unit tests (always run — langchain_core is mocked)
# ============================================================================


class TestCompactionMiddlewareBasics:
    """Verify CompactionMiddleware singleton and threshold logic."""

    def test_compaction_middleware_importable(self):
        """get_compaction_middleware() returns a usable singleton."""
        reset_compaction_middleware()
        mw = get_compaction_middleware()
        assert mw is not None
        assert hasattr(mw, "should_compact")
        assert hasattr(mw, "compact")

    def test_should_compact_below_threshold(self):
        """Below threshold, should_compact returns False."""
        mw = CompactionMiddleware(
            CompactionConfig(token_threshold=100_000, message_threshold=200)
        )
        assert mw.should_compact([], token_count=0) is False

    def test_should_compact_above_message_threshold(self):
        """Above message threshold, should_compact returns True."""
        mw = CompactionMiddleware(CompactionConfig(message_threshold=5))
        msgs = [MagicMock() for _ in range(10)]
        assert mw.should_compact(msgs, token_count=0) is True

    def test_should_compact_above_token_threshold(self):
        """Above token threshold, should_compact returns True."""
        mw = CompactionMiddleware(CompactionConfig(token_threshold=100))
        msgs = [MagicMock()]
        assert mw.should_compact(msgs, token_count=200) is True

    def test_should_compact_disabled(self):
        """When disabled, should_compact always returns False."""
        mw = CompactionMiddleware(CompactionConfig(enabled=False))
        msgs = [MagicMock() for _ in range(1000)]
        assert mw.should_compact(msgs, token_count=999_999) is False


# ============================================================================
# _maybe_compact_history wiring tests (skip if langchain not installed)
# ============================================================================


@pytest.mark.skipif(
    not _HAS_RUNTIME,
    reason="deepagents_runtime requires langchain (not installed)",
)
class TestMaybeCompactHistory:
    """Verify _maybe_compact_history helper used by run_turn / resume_turn."""

    def test_calls_compact_when_threshold_exceeded(self):
        """When should_compact is True, compact() is called and state updated."""
        fake_metrics = CompactionMetrics(
            triggered=True,
            messages_before=120,
            messages_after=30,
            tokens_before=90_000,
            tokens_after=20_000,
        )

        fake_mw = MagicMock()
        fake_mw.should_compact.return_value = True
        fake_mw.compact.return_value = (["compacted"], fake_metrics)

        fake_state = MagicMock()
        fake_state.values = {"messages": ["m1", "m2"] * 60}

        fake_agent = MagicMock()
        fake_agent.get_state.return_value = fake_state

        config = {"configurable": {"thread_id": "sess-1"}}

        with patch(
            "ag3nt_agent.compaction_middleware.get_compaction_middleware",
            return_value=fake_mw,
        ):
            _maybe_compact_history(fake_agent, config, "sess-1")

        fake_mw.should_compact.assert_called_once()
        fake_mw.compact.assert_called_once()
        fake_agent.update_state.assert_called_once_with(
            config, {"messages": ["compacted"]}
        )

    def test_noop_when_below_threshold(self):
        """When should_compact is False, no compaction happens."""
        fake_mw = MagicMock()
        fake_mw.should_compact.return_value = False

        fake_state = MagicMock()
        fake_state.values = {"messages": ["m1"]}

        fake_agent = MagicMock()
        fake_agent.get_state.return_value = fake_state

        config = {"configurable": {"thread_id": "sess-2"}}

        with patch(
            "ag3nt_agent.compaction_middleware.get_compaction_middleware",
            return_value=fake_mw,
        ):
            _maybe_compact_history(fake_agent, config, "sess-2")

        fake_mw.compact.assert_not_called()
        fake_agent.update_state.assert_not_called()

    def test_graceful_on_import_error(self):
        """If get_compaction_middleware raises ImportError, no-op."""
        fake_agent = MagicMock()
        config = {"configurable": {"thread_id": "sess-3"}}

        with patch(
            "ag3nt_agent.compaction_middleware.get_compaction_middleware",
            side_effect=ImportError("no module"),
        ):
            # Should not raise
            _maybe_compact_history(fake_agent, config, "sess-3")

        fake_agent.get_state.assert_not_called()

    def test_graceful_on_runtime_error(self):
        """If compact() raises, _maybe_compact_history catches it silently."""
        fake_mw = MagicMock()
        fake_mw.should_compact.return_value = True
        fake_mw.compact.side_effect = RuntimeError("boom")

        fake_state = MagicMock()
        fake_state.values = {"messages": ["m1"] * 200}

        fake_agent = MagicMock()
        fake_agent.get_state.return_value = fake_state

        config = {"configurable": {"thread_id": "sess-4"}}

        with patch(
            "ag3nt_agent.compaction_middleware.get_compaction_middleware",
            return_value=fake_mw,
        ):
            # Should not raise
            _maybe_compact_history(fake_agent, config, "sess-4")

        fake_agent.update_state.assert_not_called()

    def test_no_update_when_not_triggered(self):
        """If compact() returns triggered=False, state is NOT updated."""
        fake_metrics = CompactionMetrics(triggered=False)

        fake_mw = MagicMock()
        fake_mw.should_compact.return_value = True
        fake_mw.compact.return_value = (["same"], fake_metrics)

        fake_state = MagicMock()
        fake_state.values = {"messages": ["m1"] * 200}

        fake_agent = MagicMock()
        fake_agent.get_state.return_value = fake_state

        config = {"configurable": {"thread_id": "sess-5"}}

        with patch(
            "ag3nt_agent.compaction_middleware.get_compaction_middleware",
            return_value=fake_mw,
        ):
            _maybe_compact_history(fake_agent, config, "sess-5")

        fake_agent.update_state.assert_not_called()
