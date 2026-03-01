"""Tests for the self-building skills tool."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


def _ensure_mock_module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = ModuleType(name)
    sys.modules[name] = mod
    return mod


# Ensure langchain_core.tools is available (may be mocked as ModuleType by other tests)
_lc_core = _ensure_mock_module("langchain_core")
_lc_core_tools = _ensure_mock_module("langchain_core.tools")
if not hasattr(_lc_core_tools, "tool"):

    def _mock_tool_decorator(fn):
        fn.invoke = lambda args: fn(**args)
        return fn

    _lc_core_tools.tool = _mock_tool_decorator

from ag3nt_agent.create_skill_tool import create_skill, _build_skill_content


class TestBuildSkillContent:
    def test_generates_valid_yaml_frontmatter(self):
        content = _build_skill_content(
            name="my-tool",
            description="A test tool",
            triggers=["run my-tool", "use my-tool"],
            body="## Usage\nRun it like this.",
        )
        assert "---" in content
        assert "name: my-tool" in content
        assert 'description: "A test tool"' in content
        assert "triggers:" in content
        assert '  - "run my-tool"' in content
        assert "## Usage" in content

    def test_without_triggers(self):
        content = _build_skill_content(name="simple", description="Simple skill")
        assert "name: simple" in content
        assert "triggers:" not in content

    def test_without_body(self):
        content = _build_skill_content(name="empty", description="No body")
        assert content.endswith("---\n")


class TestCreateSkillTool:
    def test_creates_skill_in_workspace(self, tmp_path):
        with patch("ag3nt_agent.create_skill_tool.SKILLS_DIR", tmp_path):
            result = create_skill.invoke({
                "name": "test-skill",
                "description": "A test skill",
                "content": "## Instructions\nDo the thing.",
                "triggers": "do the thing, run test",
            })
        assert "Created" in result or "created" in result
        skill_path = tmp_path / "test-skill" / "SKILL.md"
        assert skill_path.exists()
        text = skill_path.read_text()
        assert "name: test-skill" in text
        assert "## Instructions" in text
        assert '  - "do the thing"' in text
        assert '  - "run test"' in text

    def test_rejects_invalid_skill_name(self):
        result = create_skill.invoke({
            "name": "../escape",
            "description": "bad",
            "content": "test",
        })
        assert "invalid" in result.lower() or "error" in result.lower()

    def test_rejects_name_with_uppercase(self):
        result = create_skill.invoke({
            "name": "BadName",
            "description": "bad",
            "content": "test",
        })
        assert "invalid" in result.lower() or "error" in result.lower()

    def test_rejects_overwrite_of_existing_skill(self, tmp_path):
        with patch("ag3nt_agent.create_skill_tool.SKILLS_DIR", tmp_path):
            (tmp_path / "existing-skill").mkdir()
            (tmp_path / "existing-skill" / "SKILL.md").write_text("existing")
            result = create_skill.invoke({
                "name": "existing-skill",
                "description": "overwrite attempt",
                "content": "new content",
            })
        assert "exists" in result.lower() or "already" in result.lower()

    def test_creates_without_triggers(self, tmp_path):
        with patch("ag3nt_agent.create_skill_tool.SKILLS_DIR", tmp_path):
            result = create_skill.invoke({
                "name": "no-triggers",
                "description": "A simple skill",
                "content": "Just do it.",
            })
        assert "Created" in result or "created" in result
        text = (tmp_path / "no-triggers" / "SKILL.md").read_text()
        assert "triggers:" not in text
