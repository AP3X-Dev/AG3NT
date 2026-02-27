"""Concurrency tests for module-level singletons."""

import threading
import pytest
from unittest.mock import patch, MagicMock


class TestMemoryStoreSingleton:
    """Test memory_search singleton thread safety."""

    def test_memory_store_singleton_thread_safety(self):
        """10 threads via Barrier, verify same id()."""
        import ag3nt_agent.memory_search as mod

        # Reset
        with mod._memory_store_lock:
            mod._memory_store = None

        barrier = threading.Barrier(10)
        results = []

        def get_store():
            barrier.wait()
            store = mod._get_memory_store()
            results.append(id(store))

        threads = [threading.Thread(target=get_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results)) == 1  # All same instance


class TestPlanningSingleton:
    """Test planning_tools singleton thread safety."""

    def test_planning_singleton_thread_safety(self, tmp_path, monkeypatch):
        """10 threads call _get_planning(), verify same instance."""
        import ag3nt_agent.planning_tools as mod

        # Reset
        mod._planning = None
        monkeypatch.setattr(mod, "get_default_storage_path", lambda: tmp_path / "todos.json")

        barrier = threading.Barrier(10)
        results = []

        def get_planning():
            barrier.wait()
            planner = mod._get_planning()
            results.append(id(planner))

        threads = [threading.Thread(target=get_planning) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results)) == 1  # All same instance


class TestSkillTriggerLazyInit:
    """Test skill_trigger_middleware lazy init thread safety."""

    def test_skill_trigger_lazy_init_thread_safety(self):
        """10 threads on same instance, verify load_skill_triggers called once."""
        from ag3nt_agent.skill_trigger_middleware import SkillTriggerMiddleware

        call_count = 0
        original_triggers = {"test_skill": ["hello"]}

        def mock_load():
            nonlocal call_count
            call_count += 1
            return original_triggers

        middleware = SkillTriggerMiddleware()

        with patch(
            "ag3nt_agent.skill_trigger_middleware.load_skill_triggers",
            side_effect=mock_load,
        ):
            barrier = threading.Barrier(10)
            results = []

            def load():
                barrier.wait()
                triggers = middleware._load_triggers()
                results.append(id(triggers))

            threads = [threading.Thread(target=load) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(results) == 10
        # All should get the same dict
        assert len(set(results)) == 1
        # load_skill_triggers should have been called exactly once
        assert call_count == 1
