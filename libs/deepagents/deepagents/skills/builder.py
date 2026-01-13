"""Skill Builder for creating new skills.

The builder provides a pipeline for generating new skills
from specifications or transcripts.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from deepagents.skills.models import Skill, SkillBody, SkillMeta, SkillMode, SkillSpec

if TYPE_CHECKING:
    from deepagents.skills.config import SkillsConfig

logger = logging.getLogger(__name__)


# Template for SKILL.md
SKILL_TEMPLATE = '''---
id: {id}
name: {name}
description: {description}
version: "1.0.0"
mode: {mode}
tags: {tags}
tools: {tools}
inputs: {inputs}
outputs: {outputs}
safety: {safety}
triggers: {triggers}
---

## Purpose

{purpose}

## When to Use

{when_to_use}

## Operating Procedure

{operating_procedure}

## Tool Usage Rules

{tool_usage_rules}

## Output Format

{output_format}

## Failure Modes and Recovery

{failure_modes}
'''


class SkillBuilder:
    """Builder for creating new skills.

    Provides scaffolding, authoring, and validation for
    generating new skills from specifications.
    """

    def __init__(self, config: "SkillsConfig"):
        """Initialize the builder.

        Args:
            config: Skills configuration.
        """
        self.config = config

    def scaffold(self, spec: SkillSpec) -> Path:
        """Create skill directory structure from spec.

        Args:
            spec: Skill specification.

        Returns:
            Path to created skill directory.
        """
        output_dir = Path(self.config.builder_output_dir).expanduser()
        skill_dir = output_dir / spec.id

        # Create directories
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "schemas").mkdir(exist_ok=True)
        (skill_dir / "templates").mkdir(exist_ok=True)
        (skill_dir / "fixtures").mkdir(exist_ok=True)

        # Generate SKILL.md content
        content = self._generate_skill_md(spec)

        # Write SKILL.md
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(content, encoding="utf-8")

        # Write output schema if provided
        if spec.output_schema:
            import json
            schema_path = skill_dir / "schemas" / "output.json"
            schema_path.write_text(
                json.dumps(spec.output_schema, indent=2),
                encoding="utf-8",
            )

        logger.info(f"Scaffolded skill '{spec.id}' at {skill_dir}")
        return skill_dir

    def _generate_skill_md(self, spec: SkillSpec) -> str:
        """Generate SKILL.md content from spec.

        Args:
            spec: Skill specification.

        Returns:
            SKILL.md content string.
        """
        # Format lists for YAML
        def format_list(items: list[str]) -> str:
            if not items:
                return "[]"
            return "[" + ", ".join(f'"{item}"' for item in items) + "]"

        # Generate section content
        purpose = self._generate_purpose(spec)
        when_to_use = self._generate_when_to_use(spec)
        operating_procedure = self._generate_operating_procedure(spec)
        tool_usage_rules = self._generate_tool_usage_rules(spec)
        output_format = self._generate_output_format(spec)
        failure_modes = self._generate_failure_modes(spec)

        return SKILL_TEMPLATE.format(
            id=spec.id,
            name=spec.name,
            description=spec.description,
            mode=spec.mode.value,
            tags=format_list(spec.tags),
            tools=format_list(spec.tools),
            inputs=spec.inputs,
            outputs=spec.outputs,
            safety=spec.safety,
            triggers=format_list(spec.triggers),
            purpose=purpose,
            when_to_use=when_to_use,
            operating_procedure=operating_procedure,
            tool_usage_rules=tool_usage_rules,
            output_format=output_format,
            failure_modes=failure_modes,
        )

    def _generate_purpose(self, spec: SkillSpec) -> str:
        """Generate Purpose section."""
        return spec.description or f"This skill provides {spec.name} capabilities."

    def _generate_when_to_use(self, spec: SkillSpec) -> str:
        """Generate When to Use section."""
        triggers = spec.triggers or ["when the user requests this capability"]
        return "Use this skill when:\n" + "\n".join(f"- {t}" for t in triggers)

    def _generate_operating_procedure(self, spec: SkillSpec) -> str:
        """Generate Operating Procedure section."""
        if spec.operating_procedure_hints:
            steps = spec.operating_procedure_hints
        else:
            steps = [
                "Analyze the input and understand the requirements",
                "Plan the approach based on the task",
                "Execute the necessary steps using allowed tools",
                "Validate the output matches expected format",
                "Return the result",
            ]
        return "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

    def _generate_tool_usage_rules(self, spec: SkillSpec) -> str:
        """Generate Tool Usage Rules section."""
        rules = []

        if "*" in spec.tools:
            rules.append("- All tools are available for this skill")
        else:
            rules.append(f"- Allowed tools: {', '.join(spec.tools)}")

        if spec.destructive_actions:
            rules.append("- FORBIDDEN actions:")
            for action in spec.destructive_actions:
                rules.append(f"  - {action}")

        if spec.safety:
            rules.append(f"- Safety constraints: {spec.safety}")

        return "\n".join(rules) if rules else "No specific tool restrictions."

    def _generate_output_format(self, spec: SkillSpec) -> str:
        """Generate Output Format section."""
        if spec.output_schema:
            import json
            return f"Output must conform to the following JSON schema:\n\n```json\n{json.dumps(spec.output_schema, indent=2)}\n```"
        return f"Output format: {spec.output_format or 'text'}"

    def _generate_failure_modes(self, spec: SkillSpec) -> str:
        """Generate Failure Modes section."""
        return """Common failure modes and recovery strategies:

1. **Invalid input**: Validate input before processing. Request clarification if needed.
2. **Tool errors**: Retry with exponential backoff. Report persistent failures.
3. **Timeout**: Break large tasks into smaller chunks. Report progress.
4. **Unexpected output**: Validate against schema. Retry if validation fails."""

    def validate_spec(self, spec: SkillSpec) -> list[str]:
        """Validate a skill specification.

        Args:
            spec: Specification to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        # Validate ID format
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", spec.id):
            errors.append("ID must be lowercase alphanumeric with hyphens")

        if len(spec.id) > 64:
            errors.append("ID must be 64 characters or less")

        if not spec.name:
            errors.append("Name is required")

        if not spec.description:
            errors.append("Description is required")

        if len(spec.description) > 500:
            errors.append("Description must be 500 characters or less")

        if not spec.tools:
            errors.append("At least one tool must be specified")

        return errors

    def build_from_transcript(
        self,
        transcript: str,
        skill_id: str,
        skill_name: str,
    ) -> SkillSpec:
        """Build a skill spec from a conversation transcript.

        This is a simplified version - in production, this would
        use an LLM to analyze the transcript.

        Args:
            transcript: Conversation transcript.
            skill_id: Desired skill ID.
            skill_name: Desired skill name.

        Returns:
            Generated SkillSpec.
        """
        # Extract tools mentioned in transcript
        tools = self._extract_tools_from_transcript(transcript)

        # Extract potential triggers
        triggers = self._extract_triggers_from_transcript(transcript)

        return SkillSpec(
            id=skill_id,
            name=skill_name,
            description=f"Skill generated from transcript for {skill_name}",
            mode=SkillMode.BOTH,
            tools=tools or ["*"],
            triggers=triggers,
        )

    def _extract_tools_from_transcript(self, transcript: str) -> list[str]:
        """Extract tool names from transcript."""
        # Simple pattern matching for tool calls
        pattern = r"(?:using|calling|invoke|tool)\s+['\"]?(\w+)['\"]?"
        matches = re.findall(pattern, transcript, re.IGNORECASE)
        return list(set(matches)) if matches else []

    def _extract_triggers_from_transcript(self, transcript: str) -> list[str]:
        """Extract potential trigger phrases from transcript."""
        # Look for user requests
        pattern = r"(?:please|can you|I need|help me)\s+(.+?)(?:\.|$)"
        matches = re.findall(pattern, transcript, re.IGNORECASE)
        return [m.strip() for m in matches[:5]] if matches else []

