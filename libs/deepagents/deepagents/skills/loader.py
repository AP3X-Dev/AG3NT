"""Skill Loader for loading full skill content.

The loader handles parsing the full SKILL.md content including
the markdown body sections.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from deepagents.skills.models import (
    Skill,
    SkillBody,
    SkillMeta,
    SkillMode,
    SkillValidationError,
)

if TYPE_CHECKING:
    from deepagents.skills.config import SkillsConfig
    from deepagents.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


# Required sections in skill body
REQUIRED_SECTIONS = [
    "purpose",
    "when to use",
    "operating procedure",
    "tool usage rules",
    "output format",
    "failure modes",
]


class SkillLoader:
    """Loader for full skill content.

    Loads and parses the complete SKILL.md file including
    the markdown body with all sections.
    """

    def __init__(self, config: "SkillsConfig", registry: "SkillRegistry"):
        """Initialize the loader.

        Args:
            config: Skills configuration.
            registry: Skill registry for metadata lookup.
        """
        self.config = config
        self.registry = registry
        self._cache: dict[str, Skill] = {}

    def load(self, skill_id: str, use_cache: bool = True) -> Skill:
        """Load a full skill by ID.

        Args:
            skill_id: Skill identifier.
            use_cache: Whether to use cached skill.

        Returns:
            Full Skill object.

        Raises:
            ValueError: If skill not found.
            SkillValidationError: If skill validation fails.
        """
        if use_cache and skill_id in self._cache:
            return self._cache[skill_id]

        meta = self.registry.get(skill_id)
        if not meta:
            raise ValueError(f"Skill not found: {skill_id}")

        skill = self._load_from_path(Path(meta.path), meta)
        self._cache[skill_id] = skill
        return skill

    def _load_from_path(self, skill_md: Path, meta: SkillMeta) -> Skill:
        """Load skill from SKILL.md path.

        Args:
            skill_md: Path to SKILL.md file.
            meta: Pre-loaded metadata.

        Returns:
            Full Skill object.
        """
        content = skill_md.read_text(encoding="utf-8")

        # Truncate if needed
        if len(content) > self.config.max_skill_body_chars:
            content = content[: self.config.max_skill_body_chars]
            logger.warning(f"Skill {meta.id} body truncated to {self.config.max_skill_body_chars} chars")

        # Parse frontmatter and body
        _, body_md = self._parse_frontmatter(content)

        # Parse sections
        body = self._parse_body(body_md, meta.id)

        return Skill(meta=meta, body=body, raw_content=content)

    def _parse_frontmatter(self, content: str) -> tuple[dict | None, str]:
        """Parse YAML frontmatter from markdown content."""
        if not content.startswith("---"):
            return None, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None, content

        try:
            frontmatter = yaml.safe_load(parts[1])
            body = parts[2].strip()
            return frontmatter, body
        except yaml.YAMLError:
            return None, content

    def _parse_body(self, body_md: str, skill_id: str) -> SkillBody:
        """Parse markdown body into sections.

        Args:
            body_md: Markdown body content.
            skill_id: Skill ID for error messages.

        Returns:
            Parsed SkillBody.
        """
        sections = self._extract_sections(body_md)

        # Validate required sections
        missing = []
        for required in REQUIRED_SECTIONS:
            normalized = required.lower().replace(" ", "_").replace("-", "_")
            if normalized not in sections:
                missing.append(required)

        if missing:
            logger.warning(f"Skill {skill_id} missing sections: {missing}")

        return SkillBody(
            purpose=sections.get("purpose", ""),
            when_to_use=sections.get("when_to_use", ""),
            operating_procedure=sections.get("operating_procedure", ""),
            tool_usage_rules=sections.get("tool_usage_rules", ""),
            output_format=sections.get("output_format", ""),
            failure_modes=sections.get("failure_modes", sections.get("failure_modes_and_recovery", "")),
            raw_markdown=body_md,
        )

    def _extract_sections(self, body_md: str) -> dict[str, str]:
        """Extract sections from markdown body.

        Sections are identified by ## headers.

        Args:
            body_md: Markdown body content.

        Returns:
            Dict mapping normalized section names to content.
        """
        sections: dict[str, str] = {}

        # Split by ## headers
        pattern = r"^##\s+(.+?)$"
        parts = re.split(pattern, body_md, flags=re.MULTILINE)

        # First part is content before any header
        if len(parts) > 1:
            # parts[0] is before first header, then alternating header/content
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    header = parts[i].strip()
                    content = parts[i + 1].strip()
                    # Normalize header name
                    normalized = header.lower().replace(" ", "_").replace("-", "_")
                    sections[normalized] = content

        return sections

    def validate(self, skill: Skill) -> list[str]:
        """Validate a loaded skill.

        Args:
            skill: Skill to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        # Check required metadata
        if not skill.meta.id:
            errors.append("Missing skill ID")
        if not skill.meta.name:
            errors.append("Missing skill name")
        if not skill.meta.description:
            errors.append("Missing skill description")

        # Check required sections
        if not skill.body.purpose:
            errors.append("Missing 'Purpose' section")
        if not skill.body.operating_procedure:
            errors.append("Missing 'Operating Procedure' section")
        if not skill.body.output_format:
            errors.append("Missing 'Output Format' section")

        # Check tools list
        if not skill.meta.tools:
            errors.append("Missing tools list")

        return errors

    def clear_cache(self) -> None:
        """Clear the skill cache."""
        self._cache.clear()

    def preload(self, skill_ids: list[str]) -> dict[str, Skill | Exception]:
        """Preload multiple skills.

        Args:
            skill_ids: List of skill IDs to preload.

        Returns:
            Dict mapping skill IDs to Skill or Exception.
        """
        results: dict[str, Skill | Exception] = {}
        for skill_id in skill_ids:
            try:
                results[skill_id] = self.load(skill_id)
            except Exception as e:
                results[skill_id] = e
        return results

