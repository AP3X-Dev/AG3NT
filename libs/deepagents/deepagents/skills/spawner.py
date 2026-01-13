"""Skill Spawner for subagent mode.

The spawner creates dedicated subagents powered by skills,
with their own context and tool restrictions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from deepagents.skills.models import Skill, SkillMode, SkillUsageRecord

if TYPE_CHECKING:
    from deepagents.skills.config import SkillsConfig
    from deepagents.skills.ledger import SkillUsageLedger
    from deepagents.skills.loader import SkillLoader

logger = logging.getLogger(__name__)


@dataclass
class SkillSubagent:
    """Represents a spawned skill subagent."""

    skill_id: str
    skill: Skill
    agent_id: str
    system_prompt: str
    allowed_tools: list[str]
    created_at: float = field(default_factory=time.time)
    completed: bool = False
    result: Any = None
    error: str | None = None


class SkillSpawner:
    """Spawns dedicated subagents for skills.

    Creates subagents with skill-specific system prompts
    and tool restrictions.
    """

    def __init__(
        self,
        config: "SkillsConfig",
        loader: "SkillLoader",
        ledger: "SkillUsageLedger",
        agent_factory: Callable[[str, str, list[str]], Any] | None = None,
    ):
        """Initialize the spawner.

        Args:
            config: Skills configuration.
            loader: Skill loader.
            ledger: Usage ledger for tracking.
            agent_factory: Factory function to create agents.
                Signature: (agent_id, system_prompt, allowed_tools) -> agent
        """
        self.config = config
        self.loader = loader
        self.ledger = ledger
        self.agent_factory = agent_factory
        self._subagents: dict[str, SkillSubagent] = {}
        self._counter = 0

    def spawn(
        self,
        skill_id: str,
        task: str,
        calling_agent: str = "main",
    ) -> SkillSubagent:
        """Spawn a subagent for a skill.

        Args:
            skill_id: Skill to spawn.
            task: Task description for the subagent.
            calling_agent: Agent spawning the subagent.

        Returns:
            SkillSubagent instance.

        Raises:
            ValueError: If skill not found or wrong mode.
        """
        skill = self.loader.load(skill_id)

        # Validate mode
        if skill.meta.mode == SkillMode.PROMPT:
            raise ValueError(f"Skill '{skill_id}' is prompt-only, use apply_skill")

        # Generate agent ID
        self._counter += 1
        agent_id = f"skill_{skill_id}_{self._counter}"

        # Build system prompt
        system_prompt = skill.get_system_prompt()
        system_prompt += f"\n\n## Current Task\n{task}"

        # Get allowed tools
        allowed_tools = skill.meta.tools.copy()

        # Record usage
        self.ledger.record(
            SkillUsageRecord(
                skill_id=skill_id,
                action="spawned",
                calling_agent=calling_agent,
                outcome=f"agent_id={agent_id}",
            )
        )

        # Create subagent
        subagent = SkillSubagent(
            skill_id=skill_id,
            skill=skill,
            agent_id=agent_id,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
        )

        self._subagents[agent_id] = subagent

        logger.info(f"Spawned subagent '{agent_id}' for skill '{skill_id}'")

        return subagent

    def get_subagent(self, agent_id: str) -> SkillSubagent | None:
        """Get a subagent by ID.

        Args:
            agent_id: Subagent identifier.

        Returns:
            SkillSubagent if found, None otherwise.
        """
        return self._subagents.get(agent_id)

    def complete_subagent(
        self,
        agent_id: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """Mark a subagent as completed.

        Args:
            agent_id: Subagent identifier.
            result: Result from the subagent.
            error: Error message if failed.
        """
        subagent = self._subagents.get(agent_id)
        if subagent:
            subagent.completed = True
            subagent.result = result
            subagent.error = error

            duration_ms = int((time.time() - subagent.created_at) * 1000)
            self.ledger.record(
                SkillUsageRecord(
                    skill_id=subagent.skill_id,
                    action="completed" if not error else "failed",
                    outcome=str(result)[:200] if result else "",
                    duration_ms=duration_ms,
                    error=error,
                )
            )

    def list_active(self) -> list[SkillSubagent]:
        """List active (non-completed) subagents."""
        return [s for s in self._subagents.values() if not s.completed]

