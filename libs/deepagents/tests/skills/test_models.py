"""Tests for skills models."""

import pytest

from deepagents.skills.models import (
    SkillBody,
    SkillMeta,
    SkillMode,
    SkillSpec,
    SkillUsageRecord,
)


class TestSkillMeta:
    """Tests for SkillMeta model."""

    def test_create_valid_meta(self):
        """Test creating a valid SkillMeta."""
        meta = SkillMeta(
            id="test-skill",
            name="Test Skill",
            description="A test skill",
            version="1.0.0",
            mode=SkillMode.BOTH,
            tags=["test"],
            tools=["view", "edit"],
            inputs="Test input",
            outputs="Test output",
            path="/path/to/skill",
        )
        assert meta.id == "test-skill"
        assert meta.name == "Test Skill"
        assert meta.mode == SkillMode.BOTH

    def test_invalid_id_format(self):
        """Test that invalid ID format raises error."""
        with pytest.raises(ValueError, match="lowercase alphanumeric"):
            SkillMeta(
                id="Invalid_ID",
                name="Test",
                description="Test",
                version="1.0.0",
                mode=SkillMode.BOTH,
                tools=["*"],
                inputs="",
                outputs="",
                path="/path",
            )

    def test_invalid_version_format(self):
        """Test that invalid version format raises error."""
        with pytest.raises(ValueError, match="semver"):
            SkillMeta(
                id="test-skill",
                name="Test",
                description="Test",
                version="invalid",
                mode=SkillMode.BOTH,
                tools=["*"],
                inputs="",
                outputs="",
                path="/path",
            )

    def test_allows_tool_wildcard(self):
        """Test that wildcard allows all tools."""
        meta = SkillMeta(
            id="test-skill",
            name="Test",
            description="Test",
            version="1.0.0",
            mode=SkillMode.BOTH,
            tools=["*"],
            inputs="",
            outputs="",
            path="/path",
        )
        assert meta.allows_tool("any_tool") is True

    def test_allows_tool_specific(self):
        """Test specific tool allowlist."""
        meta = SkillMeta(
            id="test-skill",
            name="Test",
            description="Test",
            version="1.0.0",
            mode=SkillMode.BOTH,
            tools=["view", "edit"],
            inputs="",
            outputs="",
            path="/path",
        )
        assert meta.allows_tool("view") is True
        assert meta.allows_tool("delete") is False

    def test_matches_query(self):
        """Test query matching."""
        meta = SkillMeta(
            id="code-review",
            name="Code Review",
            description="Reviews code for bugs",
            version="1.0.0",
            mode=SkillMode.BOTH,
            tags=["code", "quality"],
            tools=["*"],
            inputs="",
            outputs="",
            path="/path",
            triggers=["review my code"],
        )
        assert meta.matches_query("code") > 0
        assert meta.matches_query("review") > 0
        assert meta.matches_query("xyz") == 0


class TestSkillBody:
    """Tests for SkillBody model."""

    def test_get_section(self):
        """Test getting sections by name."""
        body = SkillBody(
            purpose="Test purpose",
            when_to_use="When testing",
            operating_procedure="1. Do this",
            tool_usage_rules="Use tools",
            output_format="Text",
            failure_modes="Handle errors",
            raw_markdown="# Full content",
        )
        assert body.get_section("purpose") == "Test purpose"
        assert body.get_section("when_to_use") == "When testing"
        assert body.get_section("unknown") is None


class TestSkillSpec:
    """Tests for SkillSpec model."""

    def test_create_spec(self):
        """Test creating a skill spec."""
        spec = SkillSpec(
            id="new-skill",
            name="New Skill",
            description="A new skill",
            mode=SkillMode.PROMPT,
            tags=["new"],
            tools=["view"],
        )
        assert spec.id == "new-skill"
        assert spec.mode == SkillMode.PROMPT


class TestSkillUsageRecord:
    """Tests for SkillUsageRecord model."""

    def test_create_record(self):
        """Test creating a usage record."""
        record = SkillUsageRecord(
            skill_id="test-skill",
            action="applied",
            calling_agent="main",
        )
        assert record.skill_id == "test-skill"
        assert record.action == "applied"
        assert record.timestamp is not None
