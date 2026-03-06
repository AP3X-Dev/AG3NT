"""Tests for ag3nt_agent.cost_tracker."""
from __future__ import annotations

from ag3nt_agent.cost_tracker import (
    CostTracker,
    TurnCost,
    ToolTiming,
    MODEL_PRICING,
    get_cost_tracker,
)


class TestTurnCostDefaults:
    def test_defaults(self):
        tc = TurnCost()
        assert tc.input_tokens == 0
        assert tc.output_tokens == 0
        assert tc.model == ""
        assert tc.estimated_cost_usd == 0.0


class TestRecordTurn:
    def test_records_and_computes_cost(self):
        tracker = CostTracker()
        turn = tracker.record_turn(1000, 500, "claude-sonnet-4-6")
        assert turn.input_tokens == 1000
        assert turn.output_tokens == 500
        assert turn.model == "claude-sonnet-4-6"
        # 1000 * 3.0/1M + 500 * 15.0/1M = 0.003 + 0.0075 = 0.0105
        assert abs(turn.estimated_cost_usd - 0.0105) < 1e-6
        assert turn.timestamp != ""

    def test_unknown_model_uses_default(self):
        tracker = CostTracker()
        turn = tracker.record_turn(1_000_000, 0, "unknown-model-v99")
        # Default input price is 3.0 per 1M → 1M * 3.0/1M = 3.0
        assert abs(turn.estimated_cost_usd - 3.0) < 1e-4

    def test_opus_pricing(self):
        tracker = CostTracker()
        turn = tracker.record_turn(1_000_000, 1_000_000, "claude-opus-4-6")
        # 1M * 15.0/1M + 1M * 75.0/1M = 15 + 75 = 90
        assert abs(turn.estimated_cost_usd - 90.0) < 1e-4

    def test_haiku_pricing(self):
        tracker = CostTracker()
        turn = tracker.record_turn(1_000_000, 1_000_000, "claude-haiku-4-5")
        # 1M * 0.80/1M + 1M * 4.0/1M = 0.80 + 4.0 = 4.80
        assert abs(turn.estimated_cost_usd - 4.80) < 1e-4


class TestSessionTotal:
    def test_empty_tracker(self):
        tracker = CostTracker()
        total = tracker.get_session_total()
        assert total.input_tokens == 0
        assert total.output_tokens == 0
        assert total.estimated_cost_usd == 0.0
        assert total.model == ""

    def test_single_turn(self):
        tracker = CostTracker()
        tracker.record_turn(1000, 500, "claude-sonnet-4-6")
        total = tracker.get_session_total()
        assert total.input_tokens == 1000
        assert total.output_tokens == 500
        assert total.model == "claude-sonnet-4-6"

    def test_multiple_turns_same_model(self):
        tracker = CostTracker()
        tracker.record_turn(1000, 500, "claude-sonnet-4-6")
        tracker.record_turn(2000, 1000, "claude-sonnet-4-6")
        total = tracker.get_session_total()
        assert total.input_tokens == 3000
        assert total.output_tokens == 1500
        assert total.model == "claude-sonnet-4-6"

    def test_mixed_models(self):
        tracker = CostTracker()
        tracker.record_turn(1000, 500, "claude-sonnet-4-6")
        tracker.record_turn(1000, 500, "claude-haiku-4-5")
        total = tracker.get_session_total()
        assert total.model == "mixed"

    def test_cumulative_cost(self):
        tracker = CostTracker()
        t1 = tracker.record_turn(1000, 500, "claude-sonnet-4-6")
        t2 = tracker.record_turn(2000, 1000, "claude-sonnet-4-6")
        total = tracker.get_session_total()
        assert abs(total.estimated_cost_usd - (t1.estimated_cost_usd + t2.estimated_cost_usd)) < 1e-6


class TestPerTurnBreakdown:
    def test_empty(self):
        tracker = CostTracker()
        assert tracker.get_per_turn_breakdown() == []

    def test_returns_all_turns(self):
        tracker = CostTracker()
        tracker.record_turn(100, 50, "claude-sonnet-4-6")
        tracker.record_turn(200, 100, "claude-sonnet-4-6")
        breakdown = tracker.get_per_turn_breakdown()
        assert len(breakdown) == 2
        assert breakdown[0].input_tokens == 100
        assert breakdown[1].input_tokens == 200

    def test_returns_copy(self):
        tracker = CostTracker()
        tracker.record_turn(100, 50, "claude-sonnet-4-6")
        b1 = tracker.get_per_turn_breakdown()
        b2 = tracker.get_per_turn_breakdown()
        assert b1 is not b2


class TestToolTiming:
    def test_record_and_retrieve(self):
        tracker = CostTracker()
        tracker.record_tool_timing("exec_command", 150.5)
        timings = tracker.get_tool_timings()
        assert len(timings) == 1
        assert timings[0].tool_name == "exec_command"
        assert timings[0].duration_ms == 150.5

    def test_slowest_tools(self):
        tracker = CostTracker()
        tracker.record_tool_timing("fast_tool", 10.0)
        tracker.record_tool_timing("slow_tool", 5000.0)
        tracker.record_tool_timing("medium_tool", 500.0)
        slowest = tracker.get_slowest_tools(top_n=2)
        assert len(slowest) == 2
        assert slowest[0] == ("slow_tool", 5000.0)
        assert slowest[1] == ("medium_tool", 500.0)

    def test_avg_tool_latency(self):
        tracker = CostTracker()
        tracker.record_tool_timing("read_file", 10.0)
        tracker.record_tool_timing("read_file", 20.0)
        tracker.record_tool_timing("exec_command", 100.0)
        avg = tracker.get_avg_tool_latency()
        assert avg["read_file"] == 15.0
        assert avg["exec_command"] == 100.0


class TestTurnCount:
    def test_starts_at_zero(self):
        assert CostTracker().turn_count == 0

    def test_increments(self):
        tracker = CostTracker()
        tracker.record_turn(100, 50, "m")
        tracker.record_turn(100, 50, "m")
        assert tracker.turn_count == 2


class TestFormatSummary:
    def test_empty(self):
        tracker = CostTracker()
        assert tracker.format_summary() == "$0.00 | 0 tokens"

    def test_thousands(self):
        tracker = CostTracker()
        tracker.record_turn(10_000, 2_500, "claude-sonnet-4-6")
        summary = tracker.format_summary()
        assert "12.5K tokens" in summary
        assert "$" in summary

    def test_millions(self):
        tracker = CostTracker()
        tracker.record_turn(900_000, 200_000, "claude-sonnet-4-6")
        summary = tracker.format_summary()
        assert "1.1M tokens" in summary


class TestReset:
    def test_clears_everything(self):
        tracker = CostTracker()
        tracker.record_turn(1000, 500, "claude-sonnet-4-6")
        tracker.record_tool_timing("tool", 50.0)
        tracker.reset()
        assert tracker.turn_count == 0
        assert tracker.get_tool_timings() == []
        assert tracker.get_session_total().estimated_cost_usd == 0.0


class TestSingleton:
    def test_get_cost_tracker_returns_same_instance(self):
        import ag3nt_agent.cost_tracker as ct
        ct._instance = None
        t1 = get_cost_tracker()
        t2 = get_cost_tracker()
        assert t1 is t2
        ct._instance = None


class TestModelPricing:
    def test_all_models_have_input_and_output(self):
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing, f"{model} missing input price"
            assert "output" in pricing, f"{model} missing output price"
            assert pricing["input"] > 0
            assert pricing["output"] > 0
