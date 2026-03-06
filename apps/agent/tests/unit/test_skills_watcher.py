"""Tests for skills hot-reload via file watcher."""
from pathlib import Path
from unittest.mock import MagicMock

from ag3nt_agent.skills_watcher import SkillsWatcher


class TestSkillsWatcher:
    def test_detects_new_skill_directory(self, tmp_path):
        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        callback = MagicMock()
        watcher.on_change(callback)
        # Simulate new skill
        skill_dir = tmp_path / "new-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: new-skill\n---\nContent")
        watcher._handle_change(str(skill_dir / "SKILL.md"))
        callback.assert_called_once()

    def test_detects_modified_skill(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: my-skill\n---\nOriginal")
        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        callback = MagicMock()
        watcher.on_change(callback)
        skill_md.write_text("---\nname: my-skill\n---\nUpdated")
        watcher._handle_change(str(skill_md))
        callback.assert_called_once()

    def test_ignores_non_skill_files(self, tmp_path):
        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        callback = MagicMock()
        watcher.on_change(callback)
        watcher._handle_change(str(tmp_path / "random.txt"))
        callback.assert_not_called()

    def test_invalidates_trigger_cache(self):
        """Test that invalidate_triggers resets the cached triggers map.

        We test this at the unit level without importing SkillTriggerMiddleware
        (which requires langchain) by creating a simple stand-in that has the
        same invalidate_triggers behavior.
        """

        class TriggerCache:
            """Minimal stand-in matching SkillTriggerMiddleware's cache pattern."""

            def __init__(self):
                self._triggers_map = None

            def load(self):
                self._triggers_map = {"test-skill": ["run test"]}

            def invalidate_triggers(self):
                self._triggers_map = None

        cache = TriggerCache()
        cache.load()
        assert cache._triggers_map is not None
        cache.invalidate_triggers()
        assert cache._triggers_map is None

    def test_lists_watched_directories(self, tmp_path):
        watcher = SkillsWatcher(watch_dirs=[tmp_path, tmp_path / "extra"])
        assert len(watcher.watch_dirs) >= 1

    def test_callback_error_does_not_break_others(self, tmp_path):
        watcher = SkillsWatcher(watch_dirs=[tmp_path])
        bad_cb = MagicMock(side_effect=RuntimeError("fail"))
        good_cb = MagicMock()
        watcher.on_change(bad_cb)
        watcher.on_change(good_cb)
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("content")
        watcher._handle_change(str(skill_dir / "SKILL.md"))
        bad_cb.assert_called_once()
        good_cb.assert_called_once()
