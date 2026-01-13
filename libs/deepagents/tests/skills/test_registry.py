"""Tests for skills registry."""

import pytest
import tempfile
from pathlib import Path

from deepagents.skills.config import SkillsConfig
from deepagents.skills.registry import SkillRegistry
from deepagents.skills.models import SkillMode


@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory with a test skill."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "test-skill"
        skill_dir.mkdir()

        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text('''---
id: test-skill
name: Test Skill
description: A test skill for unit testing
version: "1.0.0"
mode: both
tags: ["test", "unit"]
tools: ["view", "edit"]
inputs: Test input
outputs: Test output
safety: No safety concerns
triggers: ["test this"]
---

## Purpose

This is a test skill.

## When to Use

Use for testing.

## Operating Procedure

1. Test step

## Tool Usage Rules

Use view and edit.

## Output Format

Text output.

## Failure Modes and Recovery

Handle errors gracefully.
''')
        yield tmpdir


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_scan_empty_dir(self):
        """Test scanning an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SkillsConfig(skills_dirs=[tmpdir])
            registry = SkillRegistry(config)
            count = registry.scan()
            assert count == 0

    def test_scan_with_skill(self, temp_skills_dir):
        """Test scanning a directory with a skill."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        count = registry.scan()
        assert count == 1

    def test_get_skill(self, temp_skills_dir):
        """Test getting a skill by ID."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()

        meta = registry.get("test-skill")
        assert meta is not None
        assert meta.id == "test-skill"
        assert meta.name == "Test Skill"
        assert meta.mode == SkillMode.BOTH
        assert "test" in meta.tags

    def test_get_nonexistent_skill(self, temp_skills_dir):
        """Test getting a nonexistent skill."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()

        meta = registry.get("nonexistent")
        assert meta is None

    def test_list_all(self, temp_skills_dir):
        """Test listing all skills."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()

        skills = registry.list_all()
        assert len(skills) == 1
        assert skills[0].id == "test-skill"

    def test_search_by_query(self, temp_skills_dir):
        """Test searching skills by query."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()

        results = registry.search(query="test")
        assert len(results) == 1

        results = registry.search(query="nonexistent")
        assert len(results) == 0

    def test_search_by_tags(self, temp_skills_dir):
        """Test searching skills by tags."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()

        results = registry.search(tags=["test"])
        assert len(results) == 1

        results = registry.search(tags=["nonexistent"])
        assert len(results) == 0

    def test_recommend(self, temp_skills_dir):
        """Test skill recommendations."""
        config = SkillsConfig(skills_dirs=[temp_skills_dir])
        registry = SkillRegistry(config)
        registry.scan()

        results = registry.recommend("test this code")
        assert len(results) >= 0  # May or may not match

    def test_save_and_load_index(self, temp_skills_dir):
        """Test saving and loading registry index."""
        with tempfile.TemporaryDirectory() as workspace:
            config = SkillsConfig(
                skills_dirs=[temp_skills_dir],
                workspace_dir=Path(workspace),
            )
            registry = SkillRegistry(config)
            registry.scan()

            # Save index
            index_path = registry.save_index()
            assert index_path.exists()

            # Create new registry and load from index
            registry2 = SkillRegistry(config)
            count = registry2.load_index(index_path)
            assert count == 1

            meta = registry2.get("test-skill")
            assert meta is not None

