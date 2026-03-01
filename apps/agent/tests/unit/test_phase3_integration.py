"""Phase 3 Integration Tests.

Verifies all Phase 3 components work together:
1. Bootstrap budget + identity truncation
2. Prompt modes + context budget status
3. Skills hot-reload + create skill tool
4. Context budget + tool output scaling
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock the langchain / langgraph / deepagents module tree.
# ---------------------------------------------------------------------------
_mock_modules: dict[str, ModuleType] = {}


def _ensure_mock_module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    if name in _mock_modules:
        return _mock_modules[name]
    mod = ModuleType(name)
    _mock_modules[name] = mod
    sys.modules[name] = mod
    return mod


_lc = _ensure_mock_module("langchain")
_lc_agents = _ensure_mock_module("langchain.agents")
_lc_agents_mw = _ensure_mock_module("langchain.agents.middleware")
_lc_agents_mw_types = _ensure_mock_module("langchain.agents.middleware.types")

if not hasattr(_lc_agents_mw_types, "AgentMiddleware") or not isinstance(
    _lc_agents_mw_types.AgentMiddleware, type
):
    _lc_agents_mw_types.AgentMiddleware = type(
        "AgentMiddleware", (), {"__class_getitem__": classmethod(lambda cls, *a: cls)}
    )
for _attr in ("AgentState", "ModelRequest", "ModelResponse", "ModelCallResult"):
    if not hasattr(_lc_agents_mw_types, _attr):
        setattr(_lc_agents_mw_types, _attr, MagicMock())

_lc_core = _ensure_mock_module("langchain_core")
_lc_core_msgs = _ensure_mock_module("langchain_core.messages")
if not hasattr(_lc_core_msgs, "SystemMessage"):
    _lc_core_msgs.SystemMessage = MagicMock()

# langchain_core.tools — needed by create_skill_tool.py
_lc_core_tools = _ensure_mock_module("langchain_core.tools")
if not hasattr(_lc_core_tools, "tool"):
    # Real @tool decorator: just return the function as-is with .invoke added
    def _mock_tool_decorator(fn):
        fn.invoke = lambda args: fn(**args)
        return fn

    _lc_core_tools.tool = _mock_tool_decorator

_da = _ensure_mock_module("deepagents")
_da_mw = _ensure_mock_module("deepagents.middleware")
_da_mw_utils = _ensure_mock_module("deepagents.middleware._utils")
if not hasattr(_da_mw_utils, "append_to_system_message"):

    def _mock_append(sys_msg, text):
        result = MagicMock()
        result._appended_text = text
        return result

    _da_mw_utils.append_to_system_message = _mock_append


# Now safe to import our modules
from ag3nt_agent.bootstrap_budget import BootstrapBudgetManager, BudgetConfig
from ag3nt_agent.context_budget import ContextBudgetTracker, BudgetStatus
from ag3nt_agent.create_skill_tool import create_skill, _build_skill_content
from ag3nt_agent.skills_watcher import SkillsWatcher
from ag3nt_agent.turn_context_middleware import TurnContextMiddleware, PromptMode


class TestBootstrapBudgetWithIdentity:
    """Verify bootstrap budget + identity file loading integration."""

    def test_oversized_identity_file_is_truncated(self, tmp_path):
        """An oversized identity file should be truncated with head+tail."""
        identity_file = tmp_path / "IDENTITY.md"
        # Write a large identity file (30KB > 20KB default budget)
        lines = [f"# Identity Line {i}\nContent for line {i}." for i in range(500)]
        identity_file.write_text("\n".join(lines))

        from ag3nt_agent.identity import IdentityLoader

        loader = IdentityLoader(base_dir=tmp_path)
        prompt = loader.build_system_prompt()

        # Should be truncated (original is ~30KB, budget is 20KB)
        assert len(prompt) < 25000
        # Should contain early content (head)
        assert "Line 0" in prompt
        # Should contain late content (tail) — line 499
        assert "Line 499" in prompt or "truncated" in prompt.lower()

    def test_small_identity_file_passes_through(self, tmp_path):
        """A small identity file should pass through unchanged."""
        identity_file = tmp_path / "IDENTITY.md"
        identity_file.write_text("You are a helpful assistant.")

        from ag3nt_agent.identity import IdentityLoader

        loader = IdentityLoader(base_dir=tmp_path)
        prompt = loader.build_system_prompt()
        assert "helpful assistant" in prompt


class TestPromptModesWithContextBudget:
    """Verify prompt modes + context budget integration."""

    def test_full_mode_can_use_budget_report(self):
        """FULL mode should be able to include budget status."""
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 70000)  # YELLOW
        assert tracker.status() == BudgetStatus.YELLOW

        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        # The budget report can be included in environment
        report = tracker.budget_report()
        assert "70,000" in report or "70000" in report
        assert "yellow" in report

    def test_minimal_mode_skips_budget_report(self):
        """MINIMAL mode skips all extra context including budget."""
        mw = TurnContextMiddleware(prompt_mode=PromptMode.MINIMAL)
        # MINIMAL only injects environment — no budget or skills
        manifest = mw._build_skills_manifest(
            {"test": {"name": "test", "description": "test"}}
        )
        assert manifest == ""


class TestSkillsHotReloadWithCreate:
    """Verify skills hot-reload + create skill integration."""

    def test_create_skill_triggers_watcher(self, tmp_path):
        """Creating a skill should be detectable by the watcher."""
        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        callback = MagicMock()
        watcher.on_change(callback)

        # Create a skill
        with patch("ag3nt_agent.create_skill_tool.SKILLS_DIR", tmp_path):
            result = create_skill.invoke({
                "name": "auto-test",
                "description": "Auto-created skill",
                "content": "## Instructions\nDo testing.",
                "triggers": "run auto-test",
            })

        assert "Created" in result

        # Simulate watcher detecting the new file
        watcher._handle_change(str(tmp_path / "auto-test" / "SKILL.md"))
        callback.assert_called_once()

    def test_created_skill_appears_in_manifest(self, tmp_path):
        """A created skill's metadata should work with the manifest builder."""
        with patch("ag3nt_agent.create_skill_tool.SKILLS_DIR", tmp_path):
            create_skill.invoke({
                "name": "new-workflow",
                "description": "A new workflow",
                "content": "Do stuff.",
            })

        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        manifest = mw._build_skills_manifest({
            "new-workflow": {"name": "new-workflow", "description": "A new workflow"},
        })
        assert "new-workflow" in manifest
        assert "A new workflow" in manifest


class TestContextBudgetWithToolOutput:
    """Verify context budget + tool output scaling."""

    def test_suggested_output_decreases_as_context_fills(self):
        """As context fills, suggested max tool output should decrease."""
        tracker = ContextBudgetTracker(max_tokens=100000)

        # At start — plenty of room
        suggested_start = tracker.suggested_max_tool_output()

        # After filling 50%
        tracker.record("messages", 50000)
        suggested_half = tracker.suggested_max_tool_output()
        assert suggested_half < suggested_start

        # After filling 90%
        tracker.record("tool_output", 40000)
        suggested_red = tracker.suggested_max_tool_output()
        assert suggested_red < suggested_half
        assert tracker.status() == BudgetStatus.RED

    def test_budget_tracker_resets_for_new_session(self):
        """Reset should clear all usage for a fresh session."""
        tracker = ContextBudgetTracker(max_tokens=100000)
        tracker.record("messages", 90000)
        assert tracker.status() == BudgetStatus.RED

        tracker.reset()
        assert tracker.status() == BudgetStatus.GREEN
        assert tracker.total_tokens() == 0
        assert tracker.remaining() == 100000


class TestEndToEndWorkflow:
    """End-to-end workflow: budget + modes + skills + context tracking."""

    def test_full_pipeline(self, tmp_path):
        """Simulate a full agent session with all Phase 3 components."""
        # 1. Bootstrap budget limits identity files
        budget = BootstrapBudgetManager(BudgetConfig(per_file_chars=100, total_chars=500))
        large_content = "X" * 200
        truncated = budget.apply_budget("IDENTITY.md", large_content)
        assert len(truncated) < 200
        stats = budget.get_stats()
        assert stats["files_truncated"] == 1

        # 2. Context budget tracks usage
        ctx = ContextBudgetTracker(max_tokens=100000)
        ctx.record("system_prompt", len(truncated))
        assert ctx.status() == BudgetStatus.GREEN

        # 3. Prompt mode controls what's injected
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        manifest = mw._build_skills_manifest({
            "helper": {"name": "helper", "description": "A helper skill"},
        })
        assert "helper" in manifest

        # 4. Skills can be created at runtime
        with patch("ag3nt_agent.create_skill_tool.SKILLS_DIR", tmp_path):
            result = create_skill.invoke({
                "name": "runtime-skill",
                "description": "Created during session",
                "content": "Dynamic skill content.",
            })
        assert "Created" in result

        # 5. Watcher detects the new skill
        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        detected = MagicMock()
        watcher.on_change(detected)
        watcher._handle_change(str(tmp_path / "runtime-skill" / "SKILL.md"))
        detected.assert_called_once()
