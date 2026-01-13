"""Data models for the Skills Toolkit.

All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Get current UTC time in a timezone-aware manner."""
    return datetime.now(UTC)


class SkillMode(str, Enum):
    """Mode in which a skill can operate."""

    PROMPT = "prompt"  # Merge skill into agent context
    SUBAGENT = "subagent"  # Spawn dedicated subagent
    BOTH = "both"  # Supports both modes


class SkillValidationError(Exception):
    """Raised when skill validation fails."""

    def __init__(self, skill_id: str, errors: list[str]):
        self.skill_id = skill_id
        self.errors = errors
        super().__init__(f"Skill '{skill_id}' validation failed: {'; '.join(errors)}")


class SkillMeta(BaseModel):
    """Compact metadata for a skill (loaded without full body).

    This is the lightweight representation used for discovery and listing.
    Only loaded from YAML frontmatter, not the full markdown body.
    """

    # Required fields
    id: str = Field(..., description="Skill identifier (must match folder name)")
    name: str = Field(..., description="Human-readable skill name")
    description: str = Field(..., max_length=500, description="One paragraph description")
    version: str = Field(..., description="Semantic version (e.g., 1.0.0)")
    mode: SkillMode = Field(..., description="Operating mode")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    tools: list[str] = Field(..., description="Allowed tools (* for all)")
    inputs: str = Field(..., description="Description of expected inputs")
    outputs: str = Field(..., description="Description of expected outputs")
    safety: str = Field("", description="Safety constraints and gating notes")

    # Optional fields
    model_hint: str | None = Field(None, description="Preferred model class")
    budget_hint: int | None = Field(None, description="Token budget hint")
    triggers: list[str] = Field(default_factory=list, description="Phrases suggesting this skill")
    examples_artifact_ids: list[str] = Field(default_factory=list, description="Artifact pointers")
    schema_path: str | None = Field(None, description="Path to output schema")
    requires_browser: bool = Field(False, description="Whether skill needs browser")
    requires_network: bool = Field(False, description="Whether skill needs network")
    deprecates: list[str] = Field(default_factory=list, description="Skill IDs this replaces")

    # Metadata
    path: str = Field(..., description="Path to SKILL.md file")
    source_dir: str = Field("", description="Source directory name")
    loaded_at: datetime = Field(default_factory=_utcnow)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate skill ID format."""
        import re

        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError("ID must be lowercase alphanumeric with hyphens")
        if len(v) > 64:
            raise ValueError("ID must be 64 characters or less")
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        import re

        if not re.match(r"^\d+\.\d+\.\d+(-[\w.]+)?(\+[\w.]+)?$", v):
            raise ValueError("Version must be semver format (e.g., 1.0.0)")
        return v

    def allows_tool(self, tool_name: str) -> bool:
        """Check if this skill allows a specific tool."""
        if "*" in self.tools:
            return True
        return tool_name in self.tools

    def matches_query(self, query: str) -> float:
        """Calculate relevance score for a search query (0-1)."""
        query_lower = query.lower()
        score = 0.0

        # Check name match
        if query_lower in self.name.lower():
            score += 0.4

        # Check description match
        if query_lower in self.description.lower():
            score += 0.3

        # Check tags match
        for tag in self.tags:
            if query_lower in tag.lower():
                score += 0.2
                break

        # Check triggers match
        for trigger in self.triggers:
            if query_lower in trigger.lower():
                score += 0.1
                break

        return min(score, 1.0)


class SkillBody(BaseModel):
    """Parsed markdown body sections of a skill."""

    purpose: str = Field("", description="Purpose section content")
    when_to_use: str = Field("", description="When to use section content")
    operating_procedure: str = Field("", description="Operating procedure steps")
    tool_usage_rules: str = Field("", description="Tool usage rules and restrictions")
    output_format: str = Field("", description="Expected output format")
    failure_modes: str = Field("", description="Failure modes and recovery")
    raw_markdown: str = Field("", description="Full raw markdown body")

    def get_full_prompt(self) -> str:
        """Get the full skill body formatted as a prompt."""
        return self.raw_markdown

    def get_section(self, section_name: str) -> str | None:
        """Get a specific section by name."""
        mapping = {
            "purpose": self.purpose,
            "when_to_use": self.when_to_use,
            "operating_procedure": self.operating_procedure,
            "tool_usage_rules": self.tool_usage_rules,
            "output_format": self.output_format,
            "failure_modes": self.failure_modes,
        }
        return mapping.get(section_name.lower().replace(" ", "_").replace("-", "_"))


class Skill(BaseModel):
    """Full skill with metadata and body (loaded on demand)."""

    meta: SkillMeta = Field(..., description="Skill metadata")
    body: SkillBody = Field(..., description="Parsed skill body")
    raw_content: str = Field("", description="Raw SKILL.md content")

    def get_system_prompt(self) -> str:
        """Generate system prompt from skill for subagent mode."""
        return f"""# Skill: {self.meta.name}

## Purpose
{self.body.purpose}

## Operating Procedure
{self.body.operating_procedure}

## Tool Usage Rules
{self.body.tool_usage_rules}

## Output Format
{self.body.output_format}

## Failure Modes and Recovery
{self.body.failure_modes}
"""

    def get_context_block(self) -> str:
        """Generate context block for prompt module mode."""
        return f"""## Active Skill: {self.meta.name}

{self.body.raw_markdown}
"""


class SkillSpec(BaseModel):
    """Specification for building a new skill (used by SkillBuilder)."""

    # Basic info
    id: str = Field(..., description="Suggested skill ID")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="One paragraph description")
    mode: SkillMode = Field(SkillMode.BOTH, description="Operating mode")
    tags: list[str] = Field(default_factory=list, description="Tags")
    tools: list[str] = Field(default_factory=lambda: ["*"], description="Allowed tools")

    # Content hints
    inputs: str = Field("", description="Expected inputs")
    outputs: str = Field("", description="Expected outputs")
    output_format: str = Field("text", description="Output format (text, json, etc.)")
    output_schema: dict[str, Any] | None = Field(None, description="JSON schema for output")

    # Safety
    safety: str = Field("", description="Safety constraints")
    destructive_actions: list[str] = Field(default_factory=list, description="Forbidden actions")

    # Seeds
    seed_examples: list[str] = Field(default_factory=list, description="Example artifact IDs")
    seed_transcripts: list[str] = Field(default_factory=list, description="Transcript paths")

    # Generation hints
    operating_procedure_hints: list[str] = Field(default_factory=list, description="Procedure hints")
    triggers: list[str] = Field(default_factory=list, description="Trigger phrases")


class SkillUsageRecord(BaseModel):
    """Record of a skill usage event."""

    skill_id: str = Field(..., description="ID of the skill used")
    action: str = Field(..., description="Action: loaded, applied, spawned, blocked")
    timestamp: datetime = Field(default_factory=_utcnow)
    calling_agent: str = Field("main", description="Agent that used the skill")
    outcome: str = Field("", description="Outcome description")
    scope: str = Field("", description="Scope: current_step, current_task, session")
    blocked_tools: list[str] = Field(default_factory=list, description="Tools blocked by policy")
    duration_ms: int | None = Field(None, description="Duration in milliseconds")
    error: str | None = Field(None, description="Error message if failed")
