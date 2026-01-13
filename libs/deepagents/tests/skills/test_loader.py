"""Tests for skills loader."""

import tempfile
from pathlib import Path

import pytest

from deepagents.skills.config import SkillsConfig
from deepagents.skills.loader import SkillLoader
from deepagents.skills.registry import SkillRegistry


@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory with a test skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
id: test-skill
name: Test Skill
description: A test skill for unit testing
version: "1.0.0"
mode: both
tags: ["test"]
tools: ["view"]
inputs: Test input
outputs: Test output
---

## Purpose

This skill is for testing the loader.

## When to Use

Use when running unit tests.

## Operating Procedure

1. Load the skill
2. Verify it works

## Tool Usage Rules

Only use the view tool.

## Output Format

Plain text output.

## Failure Modes and Recovery

If loading fails, check the SKILL.md format.
""")
        yield tmpdir


class TestSkillLoader:
    """Tests for SkillLoader."""

    def test_load_skill(self, temp_skills_dir):
        """Test loading a full skill."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        skill = loader.load("test-skill")
        assert skill.meta.id == "test-skill"
        assert skill.body.purpose != ""
        assert "testing" in skill.body.purpose.lower()

    def test_load_nonexistent_skill(self, temp_skills_dir):
        """Test loading a nonexistent skill."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        with pytest.raises(ValueError, match="not found"):
            loader.load("nonexistent")

    def test_load_caching(self, temp_skills_dir):
        """Test that skills are cached."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        skill1 = loader.load("test-skill")
        skill2 = loader.load("test-skill")
        assert skill1 is skill2  # Same object from cache

    def test_load_no_cache(self, temp_skills_dir):
        """Test loading without cache."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        skill1 = loader.load("test-skill", use_cache=False)
        skill2 = loader.load("test-skill", use_cache=False)
        assert skill1 is not skill2  # Different objects

    def test_validate_skill(self, temp_skills_dir):
        """Test skill validation."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        skill = loader.load("test-skill")
        errors = loader.validate(skill)
        assert len(errors) == 0  # Valid skill

    def test_clear_cache(self, temp_skills_dir):
        """Test clearing the cache."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        skill1 = loader.load("test-skill")
        loader.clear_cache()
        skill2 = loader.load("test-skill")
        assert skill1 is not skill2  # Different after cache clear

    def test_get_system_prompt(self, temp_skills_dir):
        """Test generating system prompt from skill."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        skill = loader.load("test-skill")
        prompt = skill.get_system_prompt()
        assert "Test Skill" in prompt
        assert "Purpose" in prompt

    def test_get_context_block(self, temp_skills_dir):
        """Test generating context block from skill."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()
        loader = SkillLoader(config, registry)

        skill = loader.load("test-skill")
        block = skill.get_context_block()
        assert "Active Skill" in block
        assert "Test Skill" in block
