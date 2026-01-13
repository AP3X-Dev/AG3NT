"""Skills Toolkit for DeepAgents.

This module provides a comprehensive skills system that matches the Claude Code style workflow:
- Skills live as folders containing SKILL.md with YAML frontmatter plus markdown body
- Progressive disclosure: lightweight metadata first, full body on demand
- Prompt module mode: merge skill instructions into agent context
- Subagent mode: spawn dedicated subagent powered by skill
- Skill Builder: generate new skills from specs or transcripts

## Architecture

Skills are organized in directories:
```
skills/
  <skill_id>/
    SKILL.md        # YAML frontmatter + markdown body
    README.md       # Optional documentation
    schemas/        # Optional JSON schemas
    templates/      # Optional templates
    fixtures/       # Optional test fixtures
```

## SKILL.md Format

Required YAML frontmatter fields:
- id: string (must match folder name)
- name: string
- description: string (one paragraph max)
- version: string (semver)
- mode: enum (prompt, subagent, both)
- tags: list of strings
- tools: list of tool names or "*" for all
- inputs: description of expected inputs
- outputs: description of expected outputs
- safety: constraints and gating notes

Required markdown sections:
- Purpose
- When to use
- Operating procedure
- Tool usage rules
- Output format
- Failure modes and recovery

## Usage

```python
from deepagents.skills import SkillsToolkitMiddleware, SkillsConfig

middleware = SkillsToolkitMiddleware(
    config=SkillsConfig(
        skills_dirs=["./skills/", "~/.deepagents/skills/"],
    ),
)
agent = create_deep_agent(middleware=[middleware])
```
"""

from deepagents.skills.applier import SkillApplier
from deepagents.skills.builder import SkillBuilder
from deepagents.skills.config import SkillsConfig
from deepagents.skills.ledger import SkillUsageLedger
from deepagents.skills.loader import SkillLoader
from deepagents.skills.middleware import SkillsToolkitMiddleware
from deepagents.skills.models import (
    Skill,
    SkillMeta,
    SkillMode,
    SkillSpec,
    SkillValidationError,
)
from deepagents.skills.registry import SkillRegistry
from deepagents.skills.spawner import SkillSpawner

__all__ = [
    # Config
    "SkillsConfig",
    # Models
    "Skill",
    "SkillMeta",
    "SkillMode",
    "SkillSpec",
    "SkillValidationError",
    # Core components
    "SkillRegistry",
    "SkillLoader",
    "SkillApplier",
    "SkillSpawner",
    "SkillUsageLedger",
    "SkillBuilder",
    # Middleware
    "SkillsToolkitMiddleware",
]
