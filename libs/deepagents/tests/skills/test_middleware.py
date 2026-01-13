"""Tests for skills middleware."""

import tempfile
from pathlib import Path

import pytest

from deepagents.skills.config import SkillsConfig
from deepagents.skills.middleware import SkillsToolkitMiddleware


@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory with test skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create prompt-only skill
        prompt_dir = Path(tmpdir) / "prompt-skill"
        prompt_dir.mkdir()
        (prompt_dir / "SKILL.md").write_text("""---
id: prompt-skill
name: Prompt Skill
description: A prompt-only skill
version: "1.0.0"
mode: prompt
tags: ["prompt"]
tools: ["view"]
inputs: Input
outputs: Output
---

## Purpose
Prompt skill purpose.

## When to Use
When prompting.

## Operating Procedure
1. Do prompt things.

## Tool Usage Rules
Use view only.

## Output Format
Text.

## Failure Modes and Recovery
Handle errors.
""")

        # Create subagent-only skill
        subagent_dir = Path(tmpdir) / "subagent-skill"
        subagent_dir.mkdir()
        (subagent_dir / "SKILL.md").write_text("""---
id: subagent-skill
name: Subagent Skill
description: A subagent-only skill
version: "1.0.0"
mode: subagent
tags: ["subagent"]
tools: ["*"]
inputs: Input
outputs: Output
---

## Purpose
Subagent skill purpose.

## When to Use
When spawning.

## Operating Procedure
1. Do subagent things.

## Tool Usage Rules
All tools allowed.

## Output Format
JSON.

## Failure Modes and Recovery
Handle errors.
""")
        yield tmpdir


class TestSkillsToolkitMiddleware:
    """Tests for SkillsToolkitMiddleware."""

    def test_init_and_scan(self, temp_skills_dir):
        """Test middleware initialization and scanning."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        middleware = SkillsToolkitMiddleware(config=config)
        assert len(middleware.registry.list_all()) == 2

    def test_get_tools(self, temp_skills_dir):
        """Test getting tool definitions."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        middleware = SkillsToolkitMiddleware(config=config)
        tools = middleware.get_tools()
        tool_names = [t["name"] for t in tools]
        assert "list_skills" in tool_names
        assert "get_skill" in tool_names
        assert "apply_skill" in tool_names
        assert "spawn_skill_agent" in tool_names
        assert "build_skill" in tool_names

    def test_handle_list_skills(self, temp_skills_dir):
        """Test list_skills tool handler."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        middleware = SkillsToolkitMiddleware(config=config)
        result = middleware.handle_tool_call("list_skills", {})
        assert "skills" in result
        assert result["total"] == 2

    def test_handle_get_skill(self, temp_skills_dir):
        """Test get_skill tool handler."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        middleware = SkillsToolkitMiddleware(config=config)
        result = middleware.handle_tool_call("get_skill", {"skill_id": "prompt-skill"})
        assert result["id"] == "prompt-skill"
        assert "body" in result

    def test_handle_apply_skill(self, temp_skills_dir):
        """Test apply_skill tool handler."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        middleware = SkillsToolkitMiddleware(config=config)
        result = middleware.handle_tool_call("apply_skill", {"skill_id": "prompt-skill"})
        assert result["applied"] is True
        assert "context_block" in result

    def test_apply_subagent_only_skill_fails(self, temp_skills_dir):
        """Test that applying a subagent-only skill fails."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        middleware = SkillsToolkitMiddleware(config=config)
        result = middleware.handle_tool_call("apply_skill", {"skill_id": "subagent-skill"})
        assert "error" in result

    def test_handle_spawn_skill_agent(self, temp_skills_dir):
        """Test spawn_skill_agent tool handler."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        middleware = SkillsToolkitMiddleware(config=config)
        result = middleware.handle_tool_call(
            "spawn_skill_agent",
            {"skill_id": "subagent-skill", "task": "Do something"},
        )
        assert result["spawned"] is True
        assert "agent_id" in result

    def test_check_tool_allowed(self, temp_skills_dir):
        """Test tool allowlist enforcement."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir], enforce_tool_allowlist=True)
        middleware = SkillsToolkitMiddleware(config=config)

        # Apply skill with restricted tools
        middleware.handle_tool_call("apply_skill", {"skill_id": "prompt-skill"})

        # Check allowed tool
        allowed, reason = middleware.check_tool_allowed("view")
        assert allowed is True

        # Check disallowed tool
        allowed, reason = middleware.check_tool_allowed("delete")
        assert allowed is False
