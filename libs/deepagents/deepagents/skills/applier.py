"""Skill Applier for prompt module mode.

The applier merges skill instructions into the agent's context,
enabling the skill to guide the agent's behavior.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from deepagents.skills.models import Skill, SkillMode, SkillUsageRecord

if TYPE_CHECKING:
    from deepagents.skills.config import SkillsConfig
    from deepagents.skills.ledger import SkillUsageLedger
    from deepagents.skills.loader import SkillLoader

logger = logging.getLogger(__name__)


# Scope types for skill application
ScopeType = Literal["current_step", "current_task", "session"]


class SkillApplier:
    """Applies skills in prompt module mode.

    Merges skill instructions into the agent's context and
    enforces tool restrictions.
    """

    def __init__(
        self,
        config: "SkillsConfig",
        loader: "SkillLoader",
        ledger: "SkillUsageLedger",
    ):
        """Initialize the applier.

        Args:
            config: Skills configuration.
            loader: Skill loader.
            ledger: Usage ledger for tracking.
        """
        self.config = config
        self.loader = loader
        self.ledger = ledger
        self._active_skills: dict[str, tuple[Skill, ScopeType]] = {}

    def apply(
        self,
        skill_id: str,
        scope: ScopeType = "current_task",
        calling_agent: str = "main",
    ) -> str:
        """Apply a skill in prompt module mode.

        Args:
            skill_id: Skill to apply.
            scope: How long the skill stays active.
            calling_agent: Agent applying the skill.

        Returns:
            Context block to merge into agent prompt.

        Raises:
            ValueError: If skill not found or wrong mode.
        """
        skill = self.loader.load(skill_id)

        # Validate mode
        if skill.meta.mode == SkillMode.SUBAGENT:
            raise ValueError(f"Skill '{skill_id}' is subagent-only, use spawn_skill_agent")

        # Record usage
        self.ledger.record(
            SkillUsageRecord(
                skill_id=skill_id,
                action="applied",
                calling_agent=calling_agent,
                scope=scope,
            )
        )

        # Track active skill
        self._active_skills[skill_id] = (skill, scope)

        logger.info(f"Applied skill '{skill_id}' with scope '{scope}'")

        return skill.get_context_block()

    def get_active_skills(self) -> list[tuple[str, Skill, ScopeType]]:
        """Get currently active skills.

        Returns:
            List of (skill_id, skill, scope) tuples.
        """
        return [(sid, skill, scope) for sid, (skill, scope) in self._active_skills.items()]

    def get_allowed_tools(self) -> set[str] | None:
        """Get intersection of allowed tools from active skills.

        Returns:
            Set of allowed tool names, or None if no restrictions.
        """
        if not self._active_skills:
            return None

        if not self.config.enforce_tool_allowlist:
            return None

        allowed: set[str] | None = None

        for skill, _ in self._active_skills.values():
            if "*" in skill.meta.tools:
                continue

            skill_tools = set(skill.meta.tools)
            if allowed is None:
                allowed = skill_tools
            else:
                allowed = allowed.intersection(skill_tools)

        return allowed

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed by active skills.

        Args:
            tool_name: Tool to check.

        Returns:
            True if allowed, False otherwise.
        """
        allowed = self.get_allowed_tools()
        if allowed is None:
            return True
        return tool_name in allowed

    def clear_scope(self, scope: ScopeType) -> int:
        """Clear skills with the given scope.

        Args:
            scope: Scope to clear.

        Returns:
            Number of skills cleared.
        """
        to_remove = [sid for sid, (_, s) in self._active_skills.items() if s == scope]
        for sid in to_remove:
            del self._active_skills[sid]
        return len(to_remove)

    def clear_all(self) -> int:
        """Clear all active skills.

        Returns:
            Number of skills cleared.
        """
        count = len(self._active_skills)
        self._active_skills.clear()
        return count

