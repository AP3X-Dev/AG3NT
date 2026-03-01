"""Tests for proactive context budget tracker."""
from ag3nt_agent.context_budget import ContextBudgetTracker, BudgetStatus


class TestContextBudgetTracker:
    def test_initial_state_is_green(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        assert tracker.status() == BudgetStatus.GREEN

    def test_tracks_token_usage(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("system_prompt", 5000)
        tracker.record("messages", 20000)
        assert tracker.total_tokens() == 25000

    def test_yellow_at_60_percent(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 65000)
        assert tracker.status() == BudgetStatus.YELLOW

    def test_red_at_85_percent(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 90000)
        assert tracker.status() == BudgetStatus.RED

    def test_remaining_tokens(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 40000)
        assert tracker.remaining() == 60000

    def test_suggested_max_output_scales_with_remaining(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 40000)
        suggested = tracker.suggested_max_tool_output()
        assert 0 < suggested <= 60000

    def test_budget_report_as_string(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("system_prompt", 5000)
        tracker.record("messages", 30000)
        report = tracker.budget_report()
        assert "35,000" in report or "35000" in report
        assert "100,000" in report or "100000" in report

    def test_reset_clears_usage(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 50000)
        tracker.reset()
        assert tracker.total_tokens() == 0

    def test_record_accumulates_same_component(self):
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 10000)
        tracker.record("messages", 5000)
        assert tracker.total_tokens() == 15000

    def test_remaining_never_negative(self):
        tracker = ContextBudgetTracker(max_tokens=100)
        tracker.record("messages", 200)
        assert tracker.remaining() == 0
