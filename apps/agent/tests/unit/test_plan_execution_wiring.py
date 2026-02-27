"""Tests for PlanExecutor/Learning/Validation wiring in PlanningMiddleware."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from ag3nt_agent.planning_middleware import PlanningMiddleware, PlanningState


# ------------------------------------------------------------------
# PlanExecutor wiring on confirm_plan
# ------------------------------------------------------------------


@pytest.mark.unit
class TestPlanExecutorWiring:
    """Verify PlanExecutor is created on plan confirmation."""

    def test_confirm_plan_creates_executor_when_plan_exists(self, tmp_path):
        """confirm_plan should create PlanExecutor when plan file exists."""
        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(enabled=True, planning_phase=True)

        # Create a fake plan file
        plan_file = tmp_path / "plan.md"
        plan_file.write_text(
            "# Plan\n"
            "- [ ] Step 1: Do something\n"
            "- [ ] Step 2: Do another thing\n",
            encoding="utf-8",
        )

        # Mock PlanState to return the plan path
        mock_session_state = MagicMock()
        mock_session_state.active_plan = plan_file

        with patch("ag3nt_agent.plan_tools.PlanState") as MockPlanState:
            MockPlanState.get.return_value = mock_session_state
            mw.confirm_plan("test", ["Task 1", "Task 2"])

        state = mw.get_state("test")
        assert state.plan_confirmed is True
        assert state.planning_phase is False
        assert len(state.plan_tasks) == 2
        # PlanExecutor should have been created
        assert state.plan_executor is not None
        assert len(state.plan_executor.steps) == 2

    def test_confirm_plan_no_plan_file(self):
        """confirm_plan should work when no plan file exists (executor stays None)."""
        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(enabled=True, planning_phase=True)

        mock_session_state = MagicMock()
        mock_session_state.active_plan = None  # No plan file

        with patch("ag3nt_agent.plan_tools.PlanState") as MockPlanState:
            MockPlanState.get.return_value = mock_session_state
            mw.confirm_plan("test", ["Task 1"])

        state = mw.get_state("test")
        assert state.plan_confirmed is True
        assert state.plan_executor is None  # No plan file, no executor

    def test_confirm_plan_graceful_without_plan_module(self):
        """confirm_plan should work gracefully when plan_tools import fails."""
        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(enabled=True, planning_phase=True)

        with patch.dict("sys.modules", {"ag3nt_agent.plan_tools": None}):
            mw.confirm_plan("test", ["Task 1"])

        state = mw.get_state("test")
        assert state.plan_confirmed is True
        assert state.plan_executor is None  # Graceful degradation

    def test_confirm_plan_creates_recorder(self):
        """confirm_plan should create PlanLearningRecorder."""
        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(enabled=True, planning_phase=True)

        mock_session_state = MagicMock()
        mock_session_state.active_plan = None

        with patch("ag3nt_agent.plan_tools.PlanState") as MockPlanState:
            MockPlanState.get.return_value = mock_session_state
            mw.confirm_plan("test", ["Task 1"])

        state = mw.get_state("test")
        # PlanLearningRecorder should have been created
        assert state.plan_recorder is not None


# ------------------------------------------------------------------
# advance_task with PlanExecutor
# ------------------------------------------------------------------


@pytest.mark.unit
class TestAdvanceTaskWiring:
    """Verify advance_task uses PlanExecutor when available."""

    def test_advance_task_marks_step_complete(self):
        """advance_task should call executor.mark_step_complete if available."""
        mock_step_1 = MagicMock(is_checkpoint=False, number=1)
        mock_step_2 = MagicMock(is_checkpoint=False, number=2)

        mock_executor = MagicMock()
        mock_executor.steps = [mock_step_1, mock_step_2]

        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(
            enabled=True, planning_phase=False, plan_confirmed=True,
            plan_tasks=["Task 1", "Task 2"],
            plan_executor=mock_executor,
        )

        mw.advance_task("test")
        mock_executor.mark_step_complete.assert_called_once_with(mock_step_1)
        mock_executor.emit_step_completed.assert_called_once_with(mock_step_1, success=True)

    def test_advance_task_creates_checkpoint_on_checkpoint_step(self):
        """advance_task should create git checkpoint if step has [CHECKPOINT]."""
        mock_step = MagicMock(is_checkpoint=True, number=1)

        mock_executor = MagicMock()
        mock_executor.steps = [mock_step, MagicMock(is_checkpoint=False, number=2)]

        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(
            enabled=True, planning_phase=False, plan_confirmed=True,
            plan_tasks=["Task 1", "Task 2"],
            plan_executor=mock_executor,
        )

        mw.advance_task("test")
        mock_executor.create_git_checkpoint.assert_any_call(1)

    def test_advance_task_emits_next_step_started(self):
        """advance_task should emit step_started for the next step."""
        mock_step_1 = MagicMock(is_checkpoint=False, number=1)
        mock_step_2 = MagicMock(is_checkpoint=False, number=2)
        mock_step_3 = MagicMock(is_checkpoint=False, number=3)

        mock_executor = MagicMock()
        mock_executor.steps = [mock_step_1, mock_step_2, mock_step_3]

        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(
            enabled=True, planning_phase=False, plan_confirmed=True,
            plan_tasks=["Task 1", "Task 2", "Task 3"],
            plan_executor=mock_executor,
        )

        mw.advance_task("test")
        mock_executor.emit_step_started.assert_called_once_with(mock_step_2)

    def test_advance_task_emits_execution_completed_on_last_task(self):
        """advance_task should emit execution_completed when all tasks done."""
        mock_step = MagicMock(is_checkpoint=False, number=1)

        mock_executor = MagicMock()
        mock_executor.steps = [mock_step]

        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(
            enabled=True, planning_phase=False, plan_confirmed=True,
            plan_tasks=["Task 1"],
            plan_executor=mock_executor,
        )

        mw.advance_task("test")
        mock_executor.emit_execution_completed.assert_called_once_with(success=True)

    def test_advance_task_without_executor(self):
        """advance_task should work without executor (backward compat)."""
        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(
            enabled=True, planning_phase=False, plan_confirmed=True,
            plan_tasks=["Task 1", "Task 2"],
        )

        mw.advance_task("test")
        state = mw.get_state("test")
        assert state.current_task_index == 1

    def test_advance_task_unknown_session(self):
        """advance_task should not raise for unknown session."""
        mw = PlanningMiddleware()
        mw.advance_task("nonexistent")  # should not raise


# ------------------------------------------------------------------
# Validation gate wiring
# ------------------------------------------------------------------


@pytest.mark.unit
class TestValidationGateWiring:
    """Verify validation gates run between tasks."""

    def test_validation_gate_disabled(self):
        """When gates disabled, run_validation_gate returns None."""
        mw = PlanningMiddleware()
        mw.sessions["test"] = PlanningState(
            enabled=True, validation_gates_enabled=False,
        )

        result = mw.run_validation_gate("test", files_modified=["test.py"])
        assert result is None

    def test_validation_gate_no_session(self):
        """When no session exists, returns None."""
        mw = PlanningMiddleware()
        result = mw.run_validation_gate("nonexistent")
        assert result is None

    def test_validation_gate_no_plan_state(self):
        """When plan state is None, returns None."""
        mw = PlanningMiddleware()
        result = mw.run_validation_gate("ghost", files_modified=["a.py"])
        assert result is None


# ------------------------------------------------------------------
# PlanningState new fields
# ------------------------------------------------------------------


@pytest.mark.unit
class TestPlanningStateNewFields:
    """Verify new fields have backward-compatible defaults."""

    def test_default_executor_is_none(self):
        s = PlanningState()
        assert s.plan_executor is None

    def test_default_recorder_is_none(self):
        s = PlanningState()
        assert s.plan_recorder is None

    def test_existing_fields_unchanged(self):
        """Existing fields should still have their defaults."""
        s = PlanningState()
        assert s.enabled is False
        assert s.planning_phase is True
        assert s.plan_confirmed is False
        assert s.plan_tasks == []
        assert s.current_task_index == 0
        assert s.context_engineering is False
        assert s.validation_gates_enabled is False

    def test_custom_executor_value(self):
        """Should be able to set plan_executor at construction."""
        mock_exec = MagicMock()
        s = PlanningState(plan_executor=mock_exec)
        assert s.plan_executor is mock_exec


# ------------------------------------------------------------------
# _get_execution_prompt with PlanExecutor
# ------------------------------------------------------------------


@pytest.mark.unit
class TestExecutionPromptWithExecutor:
    """Verify _get_execution_prompt uses executor step notes when available."""

    def test_uses_executor_step_note(self):
        """When executor has steps, should use get_step_system_note."""
        mock_step = MagicMock()
        mock_executor = MagicMock()
        mock_executor.steps = [mock_step]
        mock_executor.get_step_system_note.return_value = "Execute step 1 now."

        plan_state = PlanningState(
            enabled=True, plan_confirmed=True, planning_phase=False,
            plan_tasks=["Task 1"],
            plan_executor=mock_executor,
        )

        mw = PlanningMiddleware()
        result = mw._get_execution_prompt(plan_state)
        assert result == "Execute step 1 now."
        mock_executor.get_step_system_note.assert_called_once_with(mock_step)

    def test_falls_back_to_generic_prompt(self):
        """When no executor, should use the generic execution prompt."""
        plan_state = PlanningState(
            enabled=True, plan_confirmed=True, planning_phase=False,
            plan_tasks=["Task 1", "Task 2"],
        )

        mw = PlanningMiddleware()
        result = mw._get_execution_prompt(plan_state)
        assert "PLAN APPROVED - EXECUTION MODE" in result
        assert "Task 1/2" in result

    def test_falls_back_when_index_exceeds_steps(self):
        """When task index exceeds executor steps, falls back to generic."""
        mock_executor = MagicMock()
        mock_executor.steps = []  # No steps parsed

        plan_state = PlanningState(
            enabled=True, plan_confirmed=True, planning_phase=False,
            plan_tasks=["Task 1"],
            plan_executor=mock_executor,
        )

        mw = PlanningMiddleware()
        result = mw._get_execution_prompt(plan_state)
        assert "PLAN APPROVED - EXECUTION MODE" in result


# ------------------------------------------------------------------
# _record_learning_outcome
# ------------------------------------------------------------------


@pytest.mark.unit
class TestRecordLearningOutcome:
    """Verify _record_learning_outcome is fire-and-forget."""

    def test_no_recorder_does_nothing(self):
        """When recorder is None, should return without error."""
        plan_state = PlanningState(
            plan_tasks=["Task 1"], plan_recorder=None,
        )
        mw = PlanningMiddleware()
        # Should not raise
        mw._record_learning_outcome(plan_state, "test", success=True)

    def test_recorder_exception_is_swallowed(self):
        """Recorder errors should be caught and logged, not raised."""
        mock_recorder = MagicMock()
        mock_recorder.record_blueprint_outcome = MagicMock(side_effect=Exception("boom"))

        plan_state = PlanningState(
            plan_tasks=["Task 1"],
            plan_recorder=mock_recorder,
        )

        mw = PlanningMiddleware()
        # Should not raise even if recorder throws
        mw._record_learning_outcome(plan_state, "test", success=True)
