"""Tests for bootstrap file budget manager."""
import pytest
from ag3nt_agent.bootstrap_budget import BootstrapBudgetManager, BudgetConfig


class TestBootstrapBudgetManager:
    def test_short_file_passes_through_unchanged(self):
        mgr = BootstrapBudgetManager(BudgetConfig(per_file_chars=20000, total_chars=150000))
        content = "Short content"
        result = mgr.apply_budget("test.md", content)
        assert result == content

    def test_long_file_is_truncated_with_head_tail(self):
        mgr = BootstrapBudgetManager(BudgetConfig(per_file_chars=100, total_chars=150000))
        content = "A" * 200
        result = mgr.apply_budget("test.md", content)
        assert len(result) <= 120  # 100 + truncation notice
        assert result.startswith("A")
        assert result.endswith("A")
        assert "[truncated]" in result.lower() or "..." in result

    def test_head_tail_ratio_70_20(self):
        mgr = BootstrapBudgetManager(BudgetConfig(per_file_chars=100, total_chars=150000))
        lines = [f"line-{i}" for i in range(200)]
        content = "\n".join(lines)
        result = mgr.apply_budget("test.md", content)
        # Head portion should contain early lines
        assert "line-0" in result
        # Tail portion should contain late lines
        assert "line-199" in result

    def test_total_budget_limits_across_files(self):
        mgr = BootstrapBudgetManager(BudgetConfig(per_file_chars=200, total_chars=300))
        r1 = mgr.apply_budget("file1.md", "A" * 200)
        r2 = mgr.apply_budget("file2.md", "B" * 200)
        total = len(r1) + len(r2)
        assert total <= 350  # 300 + truncation notice allowance

    def test_empty_file_returns_empty(self):
        mgr = BootstrapBudgetManager(BudgetConfig(per_file_chars=20000, total_chars=150000))
        assert mgr.apply_budget("test.md", "") == ""

    def test_get_stats_tracks_usage(self):
        mgr = BootstrapBudgetManager(BudgetConfig(per_file_chars=20000, total_chars=150000))
        mgr.apply_budget("f1.md", "Hello")
        mgr.apply_budget("f2.md", "World")
        stats = mgr.get_stats()
        assert stats["files_loaded"] == 2
        assert stats["total_chars_used"] == 10
        assert stats["files_truncated"] == 0

    def test_reset_clears_total_budget(self):
        mgr = BootstrapBudgetManager(BudgetConfig(per_file_chars=100, total_chars=150))
        mgr.apply_budget("f1.md", "A" * 100)
        mgr.reset()
        stats = mgr.get_stats()
        assert stats["total_chars_used"] == 0
