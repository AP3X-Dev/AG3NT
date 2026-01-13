"""Skills Toolkit Middleware for DeepAgents.

This middleware integrates the skills system with the agent,
providing tools for skill discovery, loading, and execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from deepagents.skills.applier import SkillApplier, ScopeType
from deepagents.skills.builder import SkillBuilder
from deepagents.skills.config import SkillsConfig
from deepagents.skills.ledger import SkillUsageLedger
from deepagents.skills.loader import SkillLoader
from deepagents.skills.models import SkillMode, SkillSpec, SkillUsageRecord
from deepagents.skills.registry import SkillRegistry
from deepagents.skills.spawner import SkillSpawner

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SkillsToolkitMiddleware:
    """Middleware that provides skills toolkit to agents.

    Exposes tools for:
    - list_skills: List available skills
    - get_skill: Get full skill details
    - recommend_skills: Get skill recommendations
    - apply_skill: Apply skill in prompt mode
    - spawn_skill_agent: Spawn skill subagent
    - build_skill: Create new skill (if enabled)
    """

    def __init__(
        self,
        config: SkillsConfig | None = None,
        agent_factory: Callable[[str, str, list[str]], Any] | None = None,
    ):
        """Initialize the middleware.

        Args:
            config: Skills configuration.
            agent_factory: Factory for creating subagents.
        """
        self.config = config or SkillsConfig()

        # Initialize components
        self.registry = SkillRegistry(self.config)
        self.ledger = SkillUsageLedger(self.config)
        self.loader = SkillLoader(self.config, self.registry)
        self.applier = SkillApplier(self.config, self.loader, self.ledger)
        self.spawner = SkillSpawner(self.config, self.loader, self.ledger, agent_factory)
        self.builder = SkillBuilder(self.config)

        # Scan for skills on init
        self.registry.scan()

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for the agent.

        Returns:
            List of tool definitions.
        """
        tools = [
            self._list_skills_tool(),
            self._get_skill_tool(),
            self._recommend_skills_tool(),
            self._apply_skill_tool(),
            self._spawn_skill_agent_tool(),
        ]

        if self.config.enable_builder:
            tools.append(self._build_skill_tool())

        return tools

    def _list_skills_tool(self) -> dict[str, Any]:
        """Define list_skills tool."""
        return {
            "name": "list_skills",
            "description": "List available skills with optional filtering by tags or mode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to filter skills",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["prompt", "subagent", "both"],
                        "description": "Filter by mode",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20,
                    },
                },
            },
        }

    def _get_skill_tool(self) -> dict[str, Any]:
        """Define get_skill tool."""
        return {
            "name": "get_skill",
            "description": "Get full details of a skill including its body sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {
                        "type": "string",
                        "description": "Skill identifier",
                    },
                },
                "required": ["skill_id"],
            },
        }

    def _recommend_skills_tool(self) -> dict[str, Any]:
        """Define recommend_skills tool."""
        return {
            "name": "recommend_skills",
            "description": "Get skill recommendations based on current context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Current task or conversation context",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum recommendations",
                        "default": 5,
                    },
                },
                "required": ["context"],
            },
        }

    def _apply_skill_tool(self) -> dict[str, Any]:
        """Define apply_skill tool."""
        return {
            "name": "apply_skill",
            "description": "Apply a skill in prompt module mode, merging its instructions into context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "Skill to apply"},
                    "scope": {
                        "type": "string",
                        "enum": ["current_step", "current_task", "session"],
                        "description": "How long skill stays active",
                        "default": "current_task",
                    },
                },
                "required": ["skill_id"],
            },
        }

    def _spawn_skill_agent_tool(self) -> dict[str, Any]:
        """Define spawn_skill_agent tool."""
        return {
            "name": "spawn_skill_agent",
            "description": "Spawn a dedicated subagent powered by a skill.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "Skill to spawn"},
                    "task": {"type": "string", "description": "Task for the subagent"},
                },
                "required": ["skill_id", "task"],
            },
        }

    def _build_skill_tool(self) -> dict[str, Any]:
        """Define build_skill tool."""
        return {
            "name": "build_skill",
            "description": "Create a new skill from a specification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Skill ID (lowercase, hyphens)"},
                    "name": {"type": "string", "description": "Human-readable name"},
                    "description": {"type": "string", "description": "One paragraph description"},
                    "mode": {
                        "type": "string",
                        "enum": ["prompt", "subagent", "both"],
                        "default": "both",
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "tools": {"type": "array", "items": {"type": "string"}, "default": ["*"]},
                    "inputs": {"type": "string"},
                    "outputs": {"type": "string"},
                    "safety": {"type": "string"},
                    "triggers": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "name", "description"],
            },
        }

    def handle_tool_call(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Handle a tool call from the agent.

        Args:
            tool_name: Name of the tool.
            args: Tool arguments.

        Returns:
            Tool result.
        """
        handlers = {
            "list_skills": self._handle_list_skills,
            "get_skill": self._handle_get_skill,
            "recommend_skills": self._handle_recommend_skills,
            "apply_skill": self._handle_apply_skill,
            "spawn_skill_agent": self._handle_spawn_skill_agent,
            "build_skill": self._handle_build_skill,
        }

        handler = handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return handler(args)
        except Exception as e:
            logger.exception(f"Error handling {tool_name}")
            return {"error": str(e)}

    def _handle_list_skills(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle list_skills tool call."""
        mode = None
        if args.get("mode"):
            mode = SkillMode(args["mode"])

        skills = self.registry.search(
            query=args.get("query"),
            tags=args.get("tags"),
            mode=mode,
            limit=args.get("limit", 20),
        )

        return {
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "mode": s.mode.value,
                    "tags": s.tags,
                }
                for s in skills
            ],
            "total": len(skills),
        }

    def _handle_get_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle get_skill tool call."""
        skill = self.loader.load(args["skill_id"])
        return {
            "id": skill.meta.id,
            "name": skill.meta.name,
            "description": skill.meta.description,
            "version": skill.meta.version,
            "mode": skill.meta.mode.value,
            "tags": skill.meta.tags,
            "tools": skill.meta.tools,
            "inputs": skill.meta.inputs,
            "outputs": skill.meta.outputs,
            "safety": skill.meta.safety,
            "body": {
                "purpose": skill.body.purpose,
                "when_to_use": skill.body.when_to_use,
                "operating_procedure": skill.body.operating_procedure,
                "tool_usage_rules": skill.body.tool_usage_rules,
                "output_format": skill.body.output_format,
                "failure_modes": skill.body.failure_modes,
            },
        }

    def _handle_recommend_skills(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle recommend_skills tool call."""
        skills = self.registry.recommend(
            context=args["context"],
            limit=args.get("limit", 5),
        )
        return {
            "recommendations": [
                {"id": s.id, "name": s.name, "description": s.description}
                for s in skills
            ],
        }

    def _handle_apply_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle apply_skill tool call."""
        scope: ScopeType = args.get("scope", "current_task")  # type: ignore
        context_block = self.applier.apply(args["skill_id"], scope=scope)
        return {
            "applied": True,
            "skill_id": args["skill_id"],
            "scope": scope,
            "context_block": context_block,
        }

    def _handle_spawn_skill_agent(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle spawn_skill_agent tool call."""
        subagent = self.spawner.spawn(args["skill_id"], args["task"])
        return {
            "spawned": True,
            "agent_id": subagent.agent_id,
            "skill_id": subagent.skill_id,
            "allowed_tools": subagent.allowed_tools,
        }

    def _handle_build_skill(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle build_skill tool call."""
        if not self.config.enable_builder:
            return {"error": "Skill builder is disabled"}

        spec = SkillSpec(
            id=args["id"],
            name=args["name"],
            description=args["description"],
            mode=SkillMode(args.get("mode", "both")),
            tags=args.get("tags", []),
            tools=args.get("tools", ["*"]),
            inputs=args.get("inputs", ""),
            outputs=args.get("outputs", ""),
            safety=args.get("safety", ""),
            triggers=args.get("triggers", []),
        )

        errors = self.builder.validate_spec(spec)
        if errors:
            return {"error": "Validation failed", "errors": errors}

        skill_dir = self.builder.scaffold(spec)

        # Rescan registry to pick up new skill
        self.registry.scan(force=True)

        return {
            "created": True,
            "skill_id": spec.id,
            "path": str(skill_dir),
        }

    def check_tool_allowed(self, tool_name: str) -> tuple[bool, str | None]:
        """Check if a tool is allowed by active skills.

        Args:
            tool_name: Tool to check.

        Returns:
            Tuple of (allowed, reason if blocked).
        """
        if not self.config.enforce_tool_allowlist:
            return True, None

        if self.config.dev_mode:
            return True, None

        if self.applier.is_tool_allowed(tool_name):
            return True, None

        # Record blocked tool
        for skill_id, skill, _ in self.applier.get_active_skills():
            if not skill.meta.allows_tool(tool_name):
                self.ledger.record(
                    SkillUsageRecord(
                        skill_id=skill_id,
                        action="blocked",
                        outcome=f"Tool '{tool_name}' blocked",
                        blocked_tools=[tool_name],
                    )
                )

        return False, f"Tool '{tool_name}' not allowed by active skills"

    def get_active_context(self) -> str:
        """Get combined context from all active skills.

        Returns:
            Combined context block.
        """
        blocks = []
        for _, skill, _ in self.applier.get_active_skills():
            blocks.append(skill.get_context_block())
        return "\n\n".join(blocks)

