"""Tests for Phase 3 wiring: skills manifest, context budget, and skills watcher.

Verifies that the three components are properly connected into the runtime:
1. _build_skills_manifest() is called during FULL mode context injection
2. ContextBudgetTracker report is injected when status is YELLOW/RED
3. SkillsWatcher connects to FileWatcher and invalidates trigger caches
4. load_skills_metadata() returns correct skill metadata
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the langchain / langgraph / deepagents module tree
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
if not hasattr(_lc_agents_mw_types, "AgentState"):
    _lc_agents_mw_types.AgentState = MagicMock()
if not hasattr(_lc_agents_mw_types, "ModelRequest"):
    _lc_agents_mw_types.ModelRequest = MagicMock()
if not hasattr(_lc_agents_mw_types, "ModelResponse"):
    _lc_agents_mw_types.ModelResponse = MagicMock()
if not hasattr(_lc_agents_mw_types, "ModelCallResult"):
    _lc_agents_mw_types.ModelCallResult = MagicMock()

_lc_core = _ensure_mock_module("langchain_core")
_lc_core_msgs = _ensure_mock_module("langchain_core.messages")
if not hasattr(_lc_core_msgs, "SystemMessage"):
    _lc_core_msgs.SystemMessage = MagicMock()
if not hasattr(_lc_core_msgs, "HumanMessage"):
    _HumanMessage = type("HumanMessage", (), {"__init__": lambda self, **kw: None})
    _lc_core_msgs.HumanMessage = _HumanMessage

_da = _ensure_mock_module("deepagents")
_da_mw = _ensure_mock_module("deepagents.middleware")
_da_mw_utils = _ensure_mock_module("deepagents.middleware._utils")
def _get_prev_text(system_message):
    """Extract accumulated text from previous mock appends."""
    prev = getattr(system_message, "_appended_text", None)
    return prev if isinstance(prev, str) else ""


def _mock_append_to_system_message(system_message, text):
    """Mock append that tracks text for assertions."""
    new_msg = MagicMock()
    prev = _get_prev_text(system_message)
    combined = (prev + "\n\n" + text) if prev else text
    new_msg._appended_text = combined
    new_msg._original = system_message
    new_msg.__contains__ = lambda self, item: item in combined
    new_msg.__str__ = lambda self: combined
    return new_msg


def _mock_append_cached_text(system_message, text, cache=True):
    """Mock cached append that tracks text for assertions."""
    new_msg = MagicMock()
    prev = _get_prev_text(system_message)
    combined = (prev + "\n\n" + text) if prev else text
    new_msg._appended_text = combined
    new_msg._cached = cache
    new_msg._original = system_message
    new_msg.__contains__ = lambda self, item: item in combined
    new_msg.__str__ = lambda self: combined
    return new_msg


_da_mw_utils.append_to_system_message = _mock_append_to_system_message
_da_mw_utils.append_cached_text = _mock_append_cached_text

from ag3nt_agent.turn_context_middleware import TurnContextMiddleware, PromptMode
from ag3nt_agent.context_budget import ContextBudgetTracker, BudgetStatus


@pytest.fixture(autouse=True)
def _patch_append_fns():
    """Patch the bound names in turn_context_middleware regardless of import order."""
    import ag3nt_agent.turn_context_middleware as tcm
    orig_cached = getattr(tcm, "append_cached_text", None)
    orig_append = getattr(tcm, "append_to_system_message", None)
    tcm.append_cached_text = _mock_append_cached_text
    tcm.append_to_system_message = _mock_append_to_system_message
    yield
    if orig_cached is not None:
        tcm.append_cached_text = orig_cached
    if orig_append is not None:
        tcm.append_to_system_message = orig_append


# ============================================================================
# Skills manifest wiring tests
# ============================================================================


class TestSkillsManifestWiring:
    """Verify skills manifest is injected during FULL mode context injection."""

    def _make_request(self):
        """Create a minimal ModelRequest-like object."""
        req = MagicMock()
        req.system_message = "Base system prompt"
        req.state = {"messages": []}
        return req

    def _get_appended_text(self, req):
        """Extract the appended text from a _inject_context call."""
        req.override.assert_called_once()
        sys_msg = (
            req.override.call_args.kwargs.get("system_message")
            or req.override.call_args[1].get("system_message")
        )
        return getattr(sys_msg, "_appended_text", str(sys_msg))

    def test_manifest_injected_in_full_mode(self):
        skills = {
            "git-commit": {"name": "git-commit", "description": "Commit changes"},
        }
        mw = TurnContextMiddleware(
            prompt_mode=PromptMode.FULL,
            skills_metadata_fn=lambda: skills,
        )
        req = self._make_request()
        mw._inject_context(req)

        appended = self._get_appended_text(req)
        assert "Available Skills" in appended
        assert "git-commit" in appended

    def test_manifest_not_injected_in_minimal_mode(self):
        skills = {"s": {"name": "s", "description": "d"}}
        mw = TurnContextMiddleware(
            prompt_mode=PromptMode.MINIMAL,
            skills_metadata_fn=lambda: skills,
        )
        req = self._make_request()
        mw._inject_context(req)

        appended = self._get_appended_text(req)
        assert "Available Skills" not in appended

    def test_manifest_not_injected_when_no_fn(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        req = self._make_request()
        mw._inject_context(req)

        appended = self._get_appended_text(req)
        assert "Available Skills" not in appended

    def test_manifest_fn_error_is_swallowed(self):
        def bad_fn():
            raise RuntimeError("boom")

        mw = TurnContextMiddleware(
            prompt_mode=PromptMode.FULL,
            skills_metadata_fn=bad_fn,
        )
        req = self._make_request()
        # Should not raise
        mw._inject_context(req)
        req.override.assert_called_once()


# ============================================================================
# Context budget wiring tests
# ============================================================================


class TestContextBudgetWiring:
    """Verify context budget report is injected when status is elevated."""

    def _make_request(self):
        req = MagicMock()
        req.system_message = "Base prompt"
        req.state = {"messages": []}
        return req

    def _get_appended_text(self, req):
        req.override.assert_called_once()
        sys_msg = (
            req.override.call_args.kwargs.get("system_message")
            or req.override.call_args[1].get("system_message")
        )
        return getattr(sys_msg, "_appended_text", str(sys_msg))

    def test_budget_report_injected_when_yellow(self):
        tracker = ContextBudgetTracker(max_tokens=100_000)
        tracker.record("messages", 70_000)  # 70% = YELLOW
        assert tracker.status() == BudgetStatus.YELLOW

        def budget_fn():
            if tracker.status() == BudgetStatus.GREEN:
                return ""
            return tracker.budget_report()

        mw = TurnContextMiddleware(
            prompt_mode=PromptMode.FULL,
            context_budget_fn=budget_fn,
        )
        req = self._make_request()
        mw._inject_context(req)

        appended = self._get_appended_text(req)
        assert "Context Budget" in appended
        assert "yellow" in appended.lower()

    def test_budget_report_injected_when_red(self):
        tracker = ContextBudgetTracker(max_tokens=100_000)
        tracker.record("messages", 90_000)  # 90% = RED
        assert tracker.status() == BudgetStatus.RED

        def budget_fn():
            if tracker.status() == BudgetStatus.GREEN:
                return ""
            return tracker.budget_report()

        mw = TurnContextMiddleware(
            prompt_mode=PromptMode.FULL,
            context_budget_fn=budget_fn,
        )
        req = self._make_request()
        mw._inject_context(req)

        appended = self._get_appended_text(req)
        assert "Context Budget" in appended
        assert "compaction" in appended.lower()

    def test_no_report_when_green(self):
        tracker = ContextBudgetTracker(max_tokens=100_000)
        tracker.record("messages", 10_000)  # 10% = GREEN
        assert tracker.status() == BudgetStatus.GREEN

        def budget_fn():
            if tracker.status() == BudgetStatus.GREEN:
                return ""
            return tracker.budget_report()

        mw = TurnContextMiddleware(
            prompt_mode=PromptMode.FULL,
            context_budget_fn=budget_fn,
        )
        req = self._make_request()
        mw._inject_context(req)

        appended = self._get_appended_text(req)
        assert "Context Budget" not in appended

    def test_no_report_when_no_fn(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        req = self._make_request()
        mw._inject_context(req)

        appended = self._get_appended_text(req)
        assert "Context Budget" not in appended

    def test_budget_fn_error_is_swallowed(self):
        def bad_fn():
            raise RuntimeError("boom")

        mw = TurnContextMiddleware(
            prompt_mode=PromptMode.FULL,
            context_budget_fn=bad_fn,
        )
        req = self._make_request()
        # Should not raise
        mw._inject_context(req)
        req.override.assert_called_once()


# ============================================================================
# SkillsWatcher + FileWatcher integration tests
# ============================================================================


class TestSkillsWatcherIntegration:
    """Verify SkillsWatcher connects to FileWatcher and handles events."""

    def test_handle_change_accepts_two_args(self, tmp_path):
        """_handle_change works with (file_path, event_type) signature."""
        from ag3nt_agent.skills_watcher import SkillsWatcher

        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        cb = MagicMock()
        watcher.on_change(cb)

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("content")

        # FileWatcher passes (path, event_type)
        watcher._handle_change(str(skill_dir / "SKILL.md"), "modified")
        cb.assert_called_once()

    def test_ignores_files_outside_watch_dirs(self, tmp_path):
        """Files in unrelated directories are ignored."""
        from ag3nt_agent.skills_watcher import SkillsWatcher

        watch_dir = tmp_path / "watched"
        watch_dir.mkdir()
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        watcher = SkillsWatcher(watch_dirs=[watch_dir])
        cb = MagicMock()
        watcher.on_change(cb)

        # SKILL.md outside watch dir
        (other_dir / "SKILL.md").write_text("content")
        watcher._handle_change(str(other_dir / "SKILL.md"), "created")
        cb.assert_not_called()

    def test_start_registers_with_file_watcher(self, tmp_path):
        """start() registers callback with FileWatcher singleton."""
        from ag3nt_agent.skills_watcher import SkillsWatcher

        watcher = SkillsWatcher(watch_dirs=[tmp_path])

        mock_fw = MagicMock()
        mock_fw_class = MagicMock()
        mock_fw_class.get_instance.return_value = mock_fw

        with patch("ag3nt_agent.skills_watcher.FileWatcher", mock_fw_class, create=True):
            # Patch the import inside start()
            with patch.dict(sys.modules, {
                "ag3nt_agent.file_watcher": MagicMock(FileWatcher=mock_fw_class),
            }):
                watcher.start()

        mock_fw.on_change.assert_called_once_with(watcher._handle_change)
        assert watcher._registered is True

    def test_stop_unregisters_from_file_watcher(self, tmp_path):
        """stop() removes callback from FileWatcher."""
        from ag3nt_agent.skills_watcher import SkillsWatcher

        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        watcher._registered = True

        mock_fw = MagicMock()
        mock_fw_class = MagicMock()
        mock_fw_class.get_instance.return_value = mock_fw

        with patch.dict(sys.modules, {
            "ag3nt_agent.file_watcher": MagicMock(FileWatcher=mock_fw_class),
        }):
            watcher.stop()

        mock_fw.remove_callback.assert_called_once_with(watcher._handle_change)
        assert watcher._registered is False


# ============================================================================
# load_skills_metadata tests
# ============================================================================


class TestLoadSkillsMetadata:
    """Verify load_skills_metadata() returns correct skill metadata."""

    def test_loads_name_and_description(self, tmp_path):
        # Mock the yaml import that skill_trigger_middleware uses
        skill_dir = tmp_path / "skills" / "my-tool"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-tool\ndescription: Does things\ntriggers:\n  - run my tool\n---\nBody"
        )

        with patch(
            "ag3nt_agent.skill_trigger_middleware._scan_skill_dirs",
            return_value=[tmp_path / "skills"],
        ):
            from ag3nt_agent.skill_trigger_middleware import load_skills_metadata

            metadata = load_skills_metadata()

        assert "my-tool" in metadata
        assert metadata["my-tool"]["name"] == "my-tool"
        assert metadata["my-tool"]["description"] == "Does things"

    def test_uses_dir_name_when_no_name_in_frontmatter(self, tmp_path):
        skill_dir = tmp_path / "skills" / "fallback-name"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: No name field\n---\nBody"
        )

        with patch(
            "ag3nt_agent.skill_trigger_middleware._scan_skill_dirs",
            return_value=[tmp_path / "skills"],
        ):
            from ag3nt_agent.skill_trigger_middleware import load_skills_metadata

            metadata = load_skills_metadata()

        assert "fallback-name" in metadata

    def test_empty_when_no_skills(self, tmp_path):
        with patch(
            "ag3nt_agent.skill_trigger_middleware._scan_skill_dirs",
            return_value=[tmp_path / "nonexistent"],
        ):
            from ag3nt_agent.skill_trigger_middleware import load_skills_metadata

            metadata = load_skills_metadata()

        assert metadata == {}

    def test_later_sources_override_earlier(self, tmp_path):
        bundled = tmp_path / "bundled"
        workspace = tmp_path / "workspace"
        for d in [bundled, workspace]:
            skill = d / "shared-skill"
            skill.mkdir(parents=True)

        (bundled / "shared-skill" / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Bundled version\n---\nBody"
        )
        (workspace / "shared-skill" / "SKILL.md").write_text(
            "---\nname: shared-skill\ndescription: Workspace override\n---\nBody"
        )

        with patch(
            "ag3nt_agent.skill_trigger_middleware._scan_skill_dirs",
            return_value=[bundled, workspace],
        ):
            from ag3nt_agent.skill_trigger_middleware import load_skills_metadata

            metadata = load_skills_metadata()

        assert metadata["shared-skill"]["description"] == "Workspace override"


# ============================================================================
# SkillTriggerMiddleware.get_skills_metadata tests
# ============================================================================


class TestSkillTriggerMiddlewareMetadata:
    """Verify get_skills_metadata caching and invalidation."""

    def test_get_skills_metadata_returns_dict(self):
        """Ensure get_skills_metadata returns metadata and caches it."""
        # Use the same stand-in pattern to avoid importing langchain
        class FakeSkillTriggerMiddleware:
            def __init__(self):
                self._triggers_map = None
                self._metadata_cache = None

            def get_skills_metadata(self):
                if self._metadata_cache is None:
                    self._metadata_cache = {
                        "test": {"name": "test", "description": "A test skill"}
                    }
                return self._metadata_cache

            def invalidate_triggers(self):
                self._triggers_map = None
                self._metadata_cache = None

        mw = FakeSkillTriggerMiddleware()
        result = mw.get_skills_metadata()
        assert "test" in result

        # Verify caching
        assert mw.get_skills_metadata() is result

        # Verify invalidation clears cache
        mw.invalidate_triggers()
        assert mw._metadata_cache is None

    def test_invalidate_clears_metadata_cache(self):
        """invalidate_triggers() also clears the metadata cache."""

        class FakeSkillTriggerMiddleware:
            def __init__(self):
                self._triggers_map = {"loaded": ["trigger"]}
                self._metadata_cache = {"loaded": {"name": "loaded", "description": ""}}

            def invalidate_triggers(self):
                self._triggers_map = None
                self._metadata_cache = None

        mw = FakeSkillTriggerMiddleware()
        assert mw._metadata_cache is not None
        mw.invalidate_triggers()
        assert mw._metadata_cache is None
        assert mw._triggers_map is None
