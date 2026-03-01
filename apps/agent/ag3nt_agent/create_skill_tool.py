"""Self-building skills tool — lets the agent create new skills at runtime.

The agent can write SKILL.md files with proper YAML frontmatter,
and the skill becomes available immediately via hot-reload.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

SKILLS_DIR = Path.home() / ".ag3nt" / "skills"
VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _build_skill_content(
    name: str,
    description: str,
    triggers: list[str] | None = None,
    body: str = "",
) -> str:
    """Build a SKILL.md file with YAML frontmatter."""
    lines = ["---"]
    lines.append(f"name: {name}")
    # Quote description to prevent YAML injection
    safe_desc = description.replace('"', '\\"')
    lines.append(f'description: "{safe_desc}"')
    if triggers:
        lines.append("triggers:")
        for t in triggers:
            safe_trigger = t.strip().replace('"', '\\"')
            lines.append(f'  - "{safe_trigger}"')
    lines.append("---")
    lines.append("")
    if body:
        lines.append(body)
    return "\n".join(lines)


@tool
def create_skill(
    name: str,
    description: str,
    content: str,
    triggers: str = "",
) -> str:
    """Create a new skill that will be available immediately.

    Args:
        name: Skill name (lowercase, hyphens/underscores allowed, e.g. 'my-tool')
        description: Short description of what the skill does
        content: The skill's markdown content (instructions, examples, etc.)
        triggers: Comma-separated trigger phrases (e.g. 'run my-tool, use my-tool')
    """
    # Validate name
    if not VALID_NAME.match(name):
        return f"Error: Invalid skill name '{name}'. Use lowercase letters, numbers, hyphens."

    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        return f"Error: Skill '{name}' already exists at {skill_dir}. Use a different name."

    trigger_list = (
        [t.strip() for t in triggers.split(",") if t.strip()] if triggers else None
    )

    skill_content = _build_skill_content(
        name=name,
        description=description,
        triggers=trigger_list,
        body=content,
    )

    try:
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
        logger.info("[CreateSkill] Created skill '%s' at %s", name, skill_dir)
        return f"Created skill '{name}' at {skill_dir}/SKILL.md. It is now available for use."
    except Exception as e:
        return f"Error creating skill: {e}"


def get_create_skill_tools() -> list:
    """Factory function for tool registry."""
    return [create_skill]
