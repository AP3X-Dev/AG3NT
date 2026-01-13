"""Skill Registry for discovering and indexing skills.

The registry scans configured directories for SKILL.md files,
parses their YAML frontmatter, and maintains an in-memory index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from deepagents.skills.models import SkillMeta, SkillMode, SkillValidationError

if TYPE_CHECKING:
    from deepagents.skills.config import SkillsConfig

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Registry for discovering and indexing skills.

    The registry maintains an in-memory index of skill metadata,
    loaded from SKILL.md files in configured directories.
    """

    def __init__(self, config: SkillsConfig):
        """Initialize the registry.

        Args:
            config: Skills configuration.
        """
        self.config = config
        self._skills: dict[str, SkillMeta] = {}
        self._loaded = False

    def scan(self, force: bool = False) -> int:
        """Scan all configured directories for skills.

        Args:
            force: If True, rescan even if already loaded.

        Returns:
            Number of skills discovered.
        """
        if self._loaded and not force:
            return len(self._skills)

        self._skills.clear()
        errors: list[str] = []

        for skills_dir in self.config.resolve_skills_dirs():
            if not skills_dir.exists():
                logger.debug(f"Skills directory does not exist: {skills_dir}")
                continue

            for skill_dir in skills_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue

                try:
                    meta = self._load_meta(skill_md, skill_dir.name)
                    if meta.id in self._skills:
                        logger.info(f"Skill '{meta.id}' overridden by {skill_md}")
                    self._skills[meta.id] = meta
                except Exception as e:
                    errors.append(f"{skill_dir.name}: {e}")
                    logger.warning(f"Failed to load skill from {skill_md}: {e}")

        self._loaded = True
        if errors:
            logger.warning(f"Skill loading errors: {len(errors)}")

        return len(self._skills)

    def _load_meta(self, skill_md: Path, expected_id: str) -> SkillMeta:
        """Load skill metadata from SKILL.md frontmatter.

        Args:
            skill_md: Path to SKILL.md file.
            expected_id: Expected skill ID (folder name).

        Returns:
            Parsed SkillMeta.

        Raises:
            SkillValidationError: If validation fails.
        """
        content = skill_md.read_text(encoding="utf-8")
        frontmatter, _ = self._parse_frontmatter(content)

        if not frontmatter:
            raise SkillValidationError(expected_id, ["Missing YAML frontmatter"])

        # Validate ID matches folder name
        skill_id = frontmatter.get("id", "")
        if skill_id != expected_id:
            raise SkillValidationError(
                expected_id,
                [f"Skill ID '{skill_id}' does not match folder name '{expected_id}'"],
            )

        # Parse mode enum
        mode_str = frontmatter.get("mode", "both")
        try:
            mode = SkillMode(mode_str.lower())
        except ValueError:
            raise SkillValidationError(expected_id, [f"Invalid mode: {mode_str}"])

        # Build SkillMeta
        try:
            meta = SkillMeta(
                id=skill_id,
                name=frontmatter.get("name", skill_id),
                description=frontmatter.get("description", ""),
                version=frontmatter.get("version", "0.0.1"),
                mode=mode,
                tags=frontmatter.get("tags", []),
                tools=frontmatter.get("tools", ["*"]),
                inputs=frontmatter.get("inputs", ""),
                outputs=frontmatter.get("outputs", ""),
                safety=frontmatter.get("safety", ""),
                model_hint=frontmatter.get("model_hint"),
                budget_hint=frontmatter.get("budget_hint"),
                triggers=frontmatter.get("triggers", []),
                examples_artifact_ids=frontmatter.get("examples_artifact_ids", []),
                schema_path=frontmatter.get("schema_path"),
                requires_browser=frontmatter.get("requires_browser", False),
                requires_network=frontmatter.get("requires_network", False),
                deprecates=frontmatter.get("deprecates", []),
                path=str(skill_md),
                source_dir=str(skill_md.parent),
            )
        except Exception as e:
            raise SkillValidationError(expected_id, [str(e)])

        return meta

    def _parse_frontmatter(self, content: str) -> tuple[dict | None, str]:
        """Parse YAML frontmatter from markdown content.

        Args:
            content: Full markdown content.

        Returns:
            Tuple of (frontmatter dict, body markdown).
        """
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

    def get(self, skill_id: str) -> SkillMeta | None:
        """Get skill metadata by ID.

        Args:
            skill_id: Skill identifier.

        Returns:
            SkillMeta if found, None otherwise.
        """
        if not self._loaded:
            self.scan()
        return self._skills.get(skill_id)

    def list_all(self) -> list[SkillMeta]:
        """List all registered skills.

        Returns:
            List of all skill metadata.
        """
        if not self._loaded:
            self.scan()
        return list(self._skills.values())

    def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        mode: SkillMode | None = None,
        limit: int | None = None,
    ) -> list[SkillMeta]:
        """Search for skills matching criteria.

        Args:
            query: Text query to match against name, description, tags.
            tags: Filter by tags (any match).
            mode: Filter by mode.
            limit: Maximum results to return.

        Returns:
            List of matching skills, sorted by relevance.
        """
        if not self._loaded:
            self.scan()

        results: list[tuple[float, SkillMeta]] = []

        for meta in self._skills.values():
            # Filter by mode
            if mode and meta.mode not in (mode, SkillMode.BOTH):
                continue

            # Filter by tags
            if tags and not any(t in meta.tags for t in tags):
                continue

            # Calculate relevance score
            score = 1.0
            if query:
                score = meta.matches_query(query)
                if score == 0:
                    continue

            results.append((score, meta))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        # Apply limit
        limit = limit or self.config.max_skills_list
        return [meta for _, meta in results[:limit]]

    def recommend(self, context: str, limit: int = 5) -> list[SkillMeta]:
        """Recommend skills based on context.

        Uses trigger phrases and description matching.

        Args:
            context: Current task or conversation context.
            limit: Maximum recommendations.

        Returns:
            List of recommended skills.
        """
        if not self._loaded:
            self.scan()

        scored: list[tuple[float, SkillMeta]] = []
        context_lower = context.lower()

        for meta in self._skills.values():
            score = 0.0

            # Check triggers
            for trigger in meta.triggers:
                if trigger.lower() in context_lower:
                    score += 0.5

            # Check description match
            score += meta.matches_query(context) * 0.5

            if score > 0:
                scored.append((score, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in scored[:limit]]

    def save_index(self) -> Path:
        """Save registry index to artifact file.

        Returns:
            Path to saved index file.
        """
        if not self._loaded:
            self.scan()

        index_path = self.config.get_registry_index_path()
        index_data = {
            "version": "1.0",
            "skills": [meta.model_dump(mode="json") for meta in self._skills.values()],
        }

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, default=str)

        return index_path

    def load_index(self, index_path: Path | None = None) -> int:
        """Load registry from index file.

        Args:
            index_path: Path to index file. Uses default if None.

        Returns:
            Number of skills loaded.
        """
        index_path = index_path or self.config.get_registry_index_path()
        if not index_path.exists():
            return 0

        with open(index_path, encoding="utf-8") as f:
            index_data = json.load(f)

        self._skills.clear()
        for skill_data in index_data.get("skills", []):
            try:
                meta = SkillMeta.model_validate(skill_data)
                self._skills[meta.id] = meta
            except Exception as e:
                logger.warning(f"Failed to load skill from index: {e}")

        self._loaded = True
        return len(self._skills)
