"""Tests for plan_revise tool and revision workflow."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from ag3nt_agent.plan_tools import (
    PlanState,
    get_plan_tools,
    plan_enter,
    plan_exit,
    plan_revise,
)


class TestPlanRevise:
    def setup_method(self):
        PlanState.reset()

    def test_revise_no_active_plan(self):
        result = plan_revise.invoke({"reason": "need changes"})
        assert "no active plan" in result.lower()

    def test_revise_sets_revision_mode(self, tmp_path):
        # Create a plan first
        import ag3nt_agent.plan_tools as pt
        orig_fn = pt._get_plans_dir
        pt._get_plans_dir = lambda: tmp_path

        try:
            plan_enter.invoke({"title": "Test Plan", "mode": "feature"})

            thread_id = PlanState._current_thread_id or "default"
            state = PlanState.get(thread_id)
            plan_path = state.active_plan
            assert plan_path is not None

            # Exit plan mode (simulate approval)
            plan_exit.invoke({"summary": "Ready", "ready": True})

            # Now revise
            state.active_plan = plan_path  # restore path after exit
            result = plan_revise.invoke({"reason": "Need to add auth step"})

            assert "revision mode activated" in result.lower()
            assert "Need to add auth step" in result

            # Verify file was updated
            content = plan_path.read_text(encoding="utf-8")
            assert "REVISING" in content
            assert "Need to add auth step" in content

            # Verify state
            assert state.plan_mode is True
        finally:
            pt._get_plans_dir = orig_fn
            PlanState.reset()

    def test_revise_preserves_completed_steps(self, tmp_path):
        import ag3nt_agent.plan_tools as pt
        orig_fn = pt._get_plans_dir
        pt._get_plans_dir = lambda: tmp_path

        try:
            plan_enter.invoke({"title": "Multi Step", "mode": "feature"})
            thread_id = PlanState._current_thread_id or "default"
            state = PlanState.get(thread_id)
            plan_path = state.active_plan

            # Simulate some completed steps
            content = plan_path.read_text(encoding="utf-8")
            content = content.replace("- [ ] Step 1:", "- [x] Step 1: Done")
            plan_path.write_text(content, encoding="utf-8")

            # Exit and revise
            plan_exit.invoke({"summary": "v1", "ready": True})
            state.active_plan = plan_path
            plan_revise.invoke({"reason": "Adding step 4"})

            # Completed steps should still be there
            revised = plan_path.read_text(encoding="utf-8")
            assert "- [x] Step 1: Done" in revised
        finally:
            pt._get_plans_dir = orig_fn
            PlanState.reset()

    def test_revise_missing_plan_file(self, tmp_path):
        # plan_revise without runtime resolves to thread_id "default"
        PlanState.set_current("default")
        state = PlanState.get("default")
        state.active_plan = tmp_path / "nonexistent.md"

        result = plan_revise.invoke({"reason": "test"})
        assert "missing" in result.lower()

    def test_plan_revise_in_plan_mode_tools(self):
        assert "plan_revise" in PlanState.PLAN_MODE_TOOLS

    def test_get_plan_tools_includes_revise(self):
        tools = get_plan_tools()
        tool_names = [t.name for t in tools]
        assert "plan_revise" in tool_names


class TestRevisionWorkflow:
    """Integration-style tests for the full revision workflow."""

    def setup_method(self):
        PlanState.reset()

    def test_full_workflow(self, tmp_path):
        """PLANNING → EXIT → REVISING → EXIT (re-approval)."""
        import ag3nt_agent.plan_tools as pt
        orig_fn = pt._get_plans_dir
        pt._get_plans_dir = lambda: tmp_path

        try:
            # 1. Create plan
            result = plan_enter.invoke({"title": "Auth Feature", "mode": "feature"})
            assert "plan created" in result.lower()

            tid = PlanState._current_thread_id or "default"
            state = PlanState.get(tid)

            # 2. Exit plan (ready for review)
            result = plan_exit.invoke({"summary": "Initial plan", "ready": True})
            assert "READY FOR REVIEW" in result

            # 3. Revise
            state.active_plan = state.active_plan or list(tmp_path.glob("*.md"))[0]
            result = plan_revise.invoke({"reason": "Forgot error handling"})
            assert "revision mode" in result.lower()
            assert state.plan_mode is True

            # 4. Re-exit (re-approval)
            result = plan_exit.invoke({"summary": "Added error handling", "ready": True})
            assert "READY FOR REVIEW" in result
        finally:
            pt._get_plans_dir = orig_fn
            PlanState.reset()
