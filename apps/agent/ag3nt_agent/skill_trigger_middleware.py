"""
Skill Trigger Matching Middleware for AG3NT.

This middleware analyzes user messages and suggests relevant skills based on
trigger keywords defined in SKILL.md files.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


def parse_skill_frontmatter(content: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter from SKILL.md content."""
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if not match:
        return None
    
    try:
        frontmatter = yaml.safe_load(match.group(1))
        return frontmatter if isinstance(frontmatter, dict) else None
    except yaml.YAMLError:
        return None


def find_repo_root() -> Path:
    """Find the repository root by looking for the skills/ directory."""
    repo_root = Path.cwd()
    while repo_root != repo_root.parent:
        if (repo_root / "skills").exists():
            return repo_root
        repo_root = repo_root.parent
    return Path.cwd()


def _scan_skill_dirs() -> list[Path]:
    """Return skill directories in priority order (bundled, global, workspace)."""
    repo_root = find_repo_root()
    return [
        repo_root / "skills",  # Bundled
        Path.home() / ".ag3nt" / "skills",  # Global
        repo_root / ".ag3nt" / "skills",  # Workspace
    ]


def load_skill_triggers() -> dict[str, list[str]]:
    """Load triggers from all skills.

    Returns:
        Dictionary mapping skill names to their trigger phrases
    """
    triggers_map: dict[str, list[str]] = {}

    for skill_dir in _scan_skill_dirs():
        if not skill_dir.exists():
            continue

        for skill_path in skill_dir.iterdir():
            if not skill_path.is_dir():
                continue

            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter = parse_skill_frontmatter(content)

                if frontmatter and "triggers" in frontmatter:
                    skill_name = frontmatter.get("name", skill_path.name)
                    triggers = frontmatter["triggers"]

                    if isinstance(triggers, list):
                        # Later sources override earlier ones
                        triggers_map[skill_name] = triggers
            except Exception as e:
                logger.warning(f"Failed to load triggers from {skill_md}: {e}")

    return triggers_map


def load_skills_metadata() -> dict[str, dict[str, Any]]:
    """Load name and description metadata from all skills.

    Returns:
        Dictionary mapping skill name to ``{"name": ..., "description": ...}``.
        Later sources (global, workspace) override earlier ones (bundled).
    """
    metadata: dict[str, dict[str, Any]] = {}

    for skill_dir in _scan_skill_dirs():
        if not skill_dir.exists():
            continue

        for skill_path in skill_dir.iterdir():
            if not skill_path.is_dir():
                continue

            skill_md = skill_path / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter = parse_skill_frontmatter(content)
                if frontmatter:
                    skill_name = frontmatter.get("name", skill_path.name)
                    metadata[skill_name] = {
                        "name": skill_name,
                        "description": frontmatter.get("description", ""),
                    }
            except Exception as e:
                logger.warning(f"Failed to load metadata from {skill_md}: {e}")

    return metadata


def match_triggers(user_message: str, triggers_map: dict[str, list[str]]) -> list[str]:
    """Match user message against skill triggers.

    Uses a multi-strategy approach:
    1. Exact substring match (highest confidence)
    2. All trigger words present in message (word-level match)
    3. Skill name appears in message (e.g., "run the git-commit skill")

    Args:
        user_message: The user's message
        triggers_map: Dictionary mapping skill names to trigger phrases

    Returns:
        List of skill names that match the user message
    """
    user_message_lower = user_message.lower()
    user_words = set(re.split(r'\W+', user_message_lower))
    matched_skills = []

    for skill_name, triggers in triggers_map.items():
        matched = False

        # Strategy 1 & 2: Check each trigger phrase
        for trigger in triggers:
            trigger_lower = trigger.lower()
            # Exact substring match
            if trigger_lower in user_message_lower:
                matched = True
                break
            # Word-level match: all words in trigger appear in message
            trigger_words = set(re.split(r'\W+', trigger_lower))
            if trigger_words and trigger_words.issubset(user_words):
                matched = True
                break

        # Strategy 3: Skill name appears in message (e.g., "git-commit")
        if not matched and skill_name.lower() in user_message_lower:
            matched = True

        if matched:
            matched_skills.append(skill_name)

    return matched_skills


class SkillTriggerMiddleware(AgentMiddleware):
    """Middleware that suggests skills based on trigger matching."""
    
    def __init__(self, planning_middleware=None):
        super().__init__()
        self._triggers_map: dict[str, list[str]] | None = None
        self._metadata_cache: dict[str, dict[str, Any]] | None = None
        self._planning_middleware = planning_middleware
        self._triggers_lock = threading.Lock()

    def _load_triggers(self) -> dict[str, list[str]]:
        """Load triggers lazily with double-checked locking."""
        if self._triggers_map is None:
            with self._triggers_lock:
                if self._triggers_map is None:
                    self._triggers_map = load_skill_triggers()
        return self._triggers_map

    def invalidate_triggers(self) -> None:
        """Invalidate cached triggers, forcing reload on next access."""
        self._triggers_map = None
        self._metadata_cache = None

    def get_skills_metadata(self) -> dict[str, dict[str, Any]]:
        """Return cached skills metadata (name + description per skill)."""
        if getattr(self, "_metadata_cache", None) is None:
            self._metadata_cache = load_skills_metadata()
        return self._metadata_cache
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject skill suggestions based on trigger matching."""
        # Skip skill suggestions during planning phase
        if self._planning_middleware is not None:
            config = getattr(request, 'config', {}) or {}
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "default")
            plan_state = self._planning_middleware.get_state(thread_id)
            if plan_state and plan_state.enabled and plan_state.planning_phase:
                return handler(request)

        # Get the last user message
        last_user_message = None
        for msg in reversed(request.messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message or not isinstance(last_user_message, str):
            return handler(request)
        
        # Load triggers and match
        triggers_map = self._load_triggers()
        matched_skills = match_triggers(last_user_message, triggers_map)
        
        if not matched_skills:
            return handler(request)
        
        # Inject skill suggestions into system prompt
        suggestion_text = "\n\n**💡 Skill Suggestions:**\n\n"
        suggestion_text += f"Based on the user's message, these skills may be relevant:\n"
        for skill_name in matched_skills:
            suggestion_text += f"- **{skill_name}**: Consider reading its SKILL.md for specialized workflows\n"
        suggestion_text += "\nRemember to use the `run_skill` tool to execute skill entrypoints when appropriate."
        
        # Append to system message
        from deepagents.middleware._utils import append_to_system_message
        new_system_message = append_to_system_message(request.system_message, suggestion_text)
        modified_request = request.override(system_message=new_system_message)
        
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject skill suggestions based on trigger matching (async version)."""
        # Skip skill suggestions during planning phase
        if self._planning_middleware is not None:
            config = getattr(request, 'config', {}) or {}
            configurable = config.get("configurable", {})
            thread_id = configurable.get("thread_id", "default")
            plan_state = self._planning_middleware.get_state(thread_id)
            if plan_state and plan_state.enabled and plan_state.planning_phase:
                return await handler(request)

        # Get the last user message
        last_user_message = None
        for msg in reversed(request.messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message or not isinstance(last_user_message, str):
            return await handler(request)

        # Load triggers and match
        triggers_map = self._load_triggers()
        matched_skills = match_triggers(last_user_message, triggers_map)

        if not matched_skills:
            return await handler(request)

        # Inject skill suggestions into system prompt
        suggestion_text = "\n\n**💡 Skill Suggestions:**\n\n"
        suggestion_text += "Based on the user's message, these skills may be relevant:\n"
        for skill_name in matched_skills:
            suggestion_text += (
                f"- **{skill_name}**: Consider reading its SKILL.md for specialized workflows\n"
            )
        suggestion_text += (
            "\nRemember to use the `run_skill` tool to execute skill entrypoints when appropriate."
        )

        # Append to system message
        from deepagents.middleware._utils import append_to_system_message

        new_system_message = append_to_system_message(request.system_message, suggestion_text)
        modified_request = request.override(system_message=new_system_message)

        return await handler(modified_request)
