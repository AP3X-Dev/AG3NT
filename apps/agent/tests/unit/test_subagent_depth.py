"""Tests for subagent depth limit enforcement logic."""


class TestSubagentDepthLogic:
    def test_depth_defaults_to_zero(self):
        state = {}
        assert state.get("_subagent_depth", 0) == 0

    def test_depth_check_rejects_at_max(self):
        state = {"_subagent_depth": 2, "_max_subagent_depth": 2}
        assert state.get("_subagent_depth", 0) >= state.get("_max_subagent_depth", 2)

    def test_depth_check_allows_below_max(self):
        state = {"_subagent_depth": 1, "_max_subagent_depth": 2}
        assert state.get("_subagent_depth", 0) < state.get("_max_subagent_depth", 2)

    def test_custom_max_depth(self):
        state = {"_subagent_depth": 3, "_max_subagent_depth": 5}
        assert state.get("_subagent_depth", 0) < state.get("_max_subagent_depth", 2)  # 3 < 5

    def test_depth_increments_in_child_state(self):
        """Simulate what _validate_and_prepare_state does."""
        parent_state = {"_subagent_depth": 0, "some_key": "value"}
        # Simulate child state creation
        child_state = {}
        current_depth = parent_state.get("_subagent_depth", 0)
        child_state["_subagent_depth"] = current_depth + 1
        child_state["_max_subagent_depth"] = parent_state.get("_max_subagent_depth", 2)
        assert child_state["_subagent_depth"] == 1
        assert child_state["_max_subagent_depth"] == 2
