"""Concurrency tests for SubagentMonitor."""

import threading
import pytest

from ag3nt_agent.subagent_monitor import SubagentMonitor, SubagentEventType


class TestSubagentMonitorConcurrency:
    """Tests proving SubagentMonitor is thread-safe."""

    @pytest.fixture
    def monitor(self, tmp_path):
        return SubagentMonitor(
            max_history=200,
            persistence_path=tmp_path / "runs.json",
            auto_persist=False,
        )

    def test_concurrent_start_execution(self, monitor):
        """10 threads start simultaneously, verify all entries present."""
        barrier = threading.Barrier(10)
        results = []

        def start(i):
            barrier.wait()
            ex = monitor.start_execution(
                parent_id="parent",
                subagent_type="worker",
                task=f"task-{i}",
            )
            results.append(ex.id)

        threads = [threading.Thread(target=start, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert monitor.get_active_count() == 10

    def test_concurrent_end_execution(self, monitor):
        """Start N, end all concurrently, verify executions list."""
        ids = []
        for i in range(10):
            ex = monitor.start_execution("parent", "worker", f"task-{i}")
            ids.append(ex.id)

        barrier = threading.Barrier(10)

        def end(exec_id):
            barrier.wait()
            monitor.end_execution(exec_id, result="done")

        threads = [threading.Thread(target=end, args=(eid,)) for eid in ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert monitor.get_active_count() == 0
        assert len(monitor.executions) == 10

    def test_concurrent_record_turn(self, monitor):
        """10 threads increment turns, verify count == 10."""
        ex = monitor.start_execution("parent", "worker", "task")
        barrier = threading.Barrier(10)

        def record(exec_id):
            barrier.wait()
            monitor.record_turn(exec_id)

        threads = [threading.Thread(target=record, args=(ex.id,)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ex.turns == 10

    def test_concurrent_record_tool_call(self, monitor):
        """10 threads append tool calls, verify all present."""
        ex = monitor.start_execution("parent", "worker", "task")
        barrier = threading.Barrier(10)

        def record(i):
            barrier.wait()
            monitor.record_tool_call(ex.id, f"tool-{i}", {"arg": i}, f"result-{i}")

        threads = [threading.Thread(target=record, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ex.tool_calls) == 10

    def test_emit_during_callback_modification(self, monitor):
        """Add callbacks while emitting, no iteration error."""
        events_received = []

        def callback1(event):
            events_received.append(("cb1", event.execution_id))

        def callback2(event):
            events_received.append(("cb2", event.execution_id))

        monitor.on_event(SubagentEventType.STARTED, callback1)

        barrier = threading.Barrier(2)

        def add_callback():
            barrier.wait()
            monitor.on_event(SubagentEventType.STARTED, callback2)

        def start_exec():
            barrier.wait()
            monitor.start_execution("parent", "worker", "task")

        t1 = threading.Thread(target=add_callback)
        t2 = threading.Thread(target=start_exec)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No crash — success

    def test_start_and_end_interleaved(self, monitor):
        """Interleaved start/end for different IDs, verify consistency."""
        barrier = threading.Barrier(20)
        errors = []

        def worker(i):
            try:
                barrier.wait()
                ex = monitor.start_execution("parent", "worker", f"task-{i}")
                monitor.record_turn(ex.id)
                monitor.end_execution(ex.id, result=f"done-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert monitor.get_active_count() == 0
        assert len(monitor.executions) == 20

    def test_concurrent_get_statistics(self, monitor):
        """Read stats during mutations, no crash."""
        for i in range(5):
            monitor.start_execution("parent", "worker", f"task-{i}")

        barrier = threading.Barrier(10)
        errors = []

        def read_stats():
            try:
                barrier.wait()
                monitor.get_statistics()
            except Exception as e:
                errors.append(e)

        def end_exec(i):
            try:
                barrier.wait()
                exec_id = list(monitor.active_subagents.keys())[0] if monitor.active_subagents else None
                if exec_id:
                    monitor.end_execution(exec_id, result="done")
            except (IndexError, KeyError):
                pass  # Race is expected, but should not crash
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=read_stats))
            threads.append(threading.Thread(target=end_exec, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_clear_history(self, monitor):
        """Clear while appending, no crash."""
        # Pre-populate some history
        for i in range(5):
            ex = monitor.start_execution("parent", "worker", f"task-{i}")
            monitor.end_execution(ex.id, result="done")

        barrier = threading.Barrier(10)
        errors = []

        def clear():
            try:
                barrier.wait()
                monitor.clear_history()
            except Exception as e:
                errors.append(e)

        def add():
            try:
                barrier.wait()
                ex = monitor.start_execution("parent", "worker", "new-task")
                monitor.end_execution(ex.id, result="done")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=clear))
            threads.append(threading.Thread(target=add))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
