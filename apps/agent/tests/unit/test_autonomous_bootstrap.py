"""Integration tests for autonomous system bootstrap.

These tests exercise start_autonomous_system / stop_autonomous_system.
Since deepagents_runtime imports langchain (not always installed in test env),
tests are skipped if the import fails.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

try:
    from ag3nt_agent.deepagents_runtime import (
        start_autonomous_system,
        stop_autonomous_system,
        get_autonomous_runtime,
    )
    import ag3nt_agent.deepagents_runtime as rt
    _HAS_RUNTIME = True
except ImportError:
    _HAS_RUNTIME = False

pytestmark = pytest.mark.skipif(
    not _HAS_RUNTIME,
    reason="deepagents_runtime requires langchain (not installed)",
)


class TestAutonomousBootstrap:

    async def test_start_creates_runtime(self):
        rt._autonomous_runtime = None
        runtime = await start_autonomous_system()
        assert runtime is not None
        assert isinstance(runtime, dict)
        assert "event_bus" in runtime
        assert "hook_manager" in runtime
        await stop_autonomous_system()

    async def test_stop_clears_runtime(self):
        rt._autonomous_runtime = None
        await start_autonomous_system()
        assert get_autonomous_runtime() is not None
        await stop_autonomous_system()
        assert get_autonomous_runtime() is None

    async def test_double_start_returns_existing(self):
        rt._autonomous_runtime = None
        r1 = await start_autonomous_system()
        r2 = await start_autonomous_system()
        assert r1 is r2
        await stop_autonomous_system()

    async def test_stop_when_not_started(self):
        rt._autonomous_runtime = None
        await stop_autonomous_system()  # should not raise

    async def test_safety_hooks_registered(self):
        rt._autonomous_runtime = None
        runtime = await start_autonomous_system()
        hook_manager = runtime.get("hook_manager")
        if hook_manager:
            from ag3nt_agent.hooks import HookPhase
            pre_hooks = hook_manager.get_hooks(HookPhase.PRE_TOOL)
            names = {h["name"] for h in pre_hooks}
            assert "protect_core" in names
            assert "block_danger" in names
        await stop_autonomous_system()


class TestBootstrapEventBus:
    """Verify start_autonomous_system() starts the shared EventBus."""

    @pytest.mark.asyncio
    async def test_bootstrap_starts_event_bus(self):
        """start_autonomous_system should call get_event_bus() and await bus.start()."""
        if not _HAS_RUNTIME:
            pytest.skip("deepagents_runtime requires langchain (not installed)")

        rt._autonomous_runtime = None

        mock_bus = MagicMock()
        mock_bus.start = AsyncMock()
        mock_bus.stop = AsyncMock()
        mock_bus.is_running = True

        with patch("ag3nt_agent.autonomous.event_bus.get_event_bus", return_value=mock_bus) as mock_get:
            try:
                runtime = await start_autonomous_system()
            except Exception:
                pass  # Other subsystems may fail, that's fine

            # The key assertion: get_event_bus was called and bus.start was awaited
            mock_get.assert_called_once()
            mock_bus.start.assert_awaited_once()

        # Clean up
        rt._autonomous_runtime = None
