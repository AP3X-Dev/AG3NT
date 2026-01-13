"""Middleware contracts and ordering enforcement for DeepAgents.

This module defines:
1. MiddlewareContract - metadata about what each middleware does
2. MiddlewarePhase - canonical ordering phases
3. validate_middleware_stack - validates ordering and detects conflicts
4. PromptBudget - token budget allocation per middleware

The goal is to make middleware responsibilities explicit and enforce
proper ordering to prevent conflicts and token bloat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentMiddleware

logger = logging.getLogger(__name__)


class MiddlewarePhase(IntEnum):
    """Canonical ordering phases for middleware.
    
    Middleware should be ordered by phase. Within a phase, order is flexible.
    Lower phase numbers run first (closer to user input).
    Higher phase numbers run last (closer to LLM).
    """
    
    # Phase 1: Context Loading (memory, skills metadata)
    CONTEXT_LOADING = 10
    
    # Phase 2: Tool Registration (filesystem, shell, web, utilities)
    TOOL_REGISTRATION = 20
    
    # Phase 3: Orchestration (subagents, task delegation)
    ORCHESTRATION = 30
    
    # Phase 4: Context Management (compaction, summarization)
    CONTEXT_MANAGEMENT = 40
    
    # Phase 5: Model Optimization (caching, patching)
    MODEL_OPTIMIZATION = 50
    
    # Phase 6: Approval Gating (HITL, must be last before LLM)
    APPROVAL_GATING = 60


@dataclass
class PromptBudget:
    """Token budget allocation for a middleware's prompt injection."""
    
    max_tokens: int = 500
    """Maximum tokens this middleware can inject into system prompt."""
    
    priority: int = 50
    """Priority for budget allocation (higher = more important, 0-100)."""
    
    can_be_trimmed: bool = True
    """Whether this content can be trimmed if over budget."""


@dataclass
class MiddlewareContract:
    """Contract defining what a middleware does and its constraints.
    
    This makes middleware responsibilities explicit and enables
    validation of the middleware stack.
    """
    
    name: str
    """Unique identifier for this middleware type."""
    
    phase: MiddlewarePhase
    """Which phase this middleware belongs to."""
    
    # What this middleware does
    injects_prompt: bool = False
    """Whether this middleware injects content into system prompt."""
    
    registers_tools: bool = False
    """Whether this middleware registers tools."""
    
    tool_names: list[str] = field(default_factory=list)
    """Names of tools this middleware registers."""
    
    modifies_tool_output: bool = False
    """Whether this middleware modifies tool outputs."""
    
    can_interrupt: bool = False
    """Whether this middleware can interrupt execution."""
    
    # Budget constraints
    prompt_budget: PromptBudget | None = None
    """Token budget for prompt injection."""
    
    # Dependencies
    requires: list[str] = field(default_factory=list)
    """Middleware names that must come before this one."""
    
    conflicts_with: list[str] = field(default_factory=list)
    """Middleware names that cannot be used with this one."""
    
    # Metadata
    description: str = ""
    """Human-readable description of what this middleware does."""


# Registry of known middleware contracts
MIDDLEWARE_CONTRACTS: dict[str, MiddlewareContract] = {
    "MemoryMiddleware": MiddlewareContract(
        name="MemoryMiddleware",
        phase=MiddlewarePhase.CONTEXT_LOADING,
        injects_prompt=True,
        prompt_budget=PromptBudget(max_tokens=2000, priority=80),
        description="Loads AGENTS.md memory files into system prompt",
    ),
    "SkillsMiddleware": MiddlewareContract(
        name="SkillsMiddleware",
        phase=MiddlewarePhase.CONTEXT_LOADING,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["list_skills", "apply_skill", "spawn_skill_agent"],
        prompt_budget=PromptBudget(max_tokens=1000, priority=70),
        description="Loads skills metadata and provides skill tools",
    ),
    "TodoListMiddleware": MiddlewareContract(
        name="TodoListMiddleware",
        phase=MiddlewarePhase.CONTEXT_LOADING,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["write_todos", "read_todos"],
        prompt_budget=PromptBudget(max_tokens=500, priority=60),
        description="Provides todo list management",
    ),
    "FilesystemMiddleware": MiddlewareContract(
        name="FilesystemMiddleware",
        phase=MiddlewarePhase.TOOL_REGISTRATION,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"],
        prompt_budget=PromptBudget(max_tokens=800, priority=90),
        description="Provides filesystem and execution tools",
    ),
    "ShellMiddleware": MiddlewareContract(
        name="ShellMiddleware",
        phase=MiddlewarePhase.TOOL_REGISTRATION,
        registers_tools=True,
        tool_names=["shell"],
        conflicts_with=["FilesystemMiddleware"],  # Both provide execution
        description="Provides shell command execution (local mode only)",
    ),
    "WebMiddleware": MiddlewareContract(
        name="WebMiddleware",
        phase=MiddlewarePhase.TOOL_REGISTRATION,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["web_search", "read_web_page"],
        prompt_budget=PromptBudget(max_tokens=300, priority=50),
        description="Provides web search and page reading",
    ),
    "UtilitiesMiddleware": MiddlewareContract(
        name="UtilitiesMiddleware",
        phase=MiddlewarePhase.TOOL_REGISTRATION,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["undo_edit", "mermaid", "format_file", "get_diagnostics"],
        prompt_budget=PromptBudget(max_tokens=300, priority=40),
        description="Provides utility tools",
    ),
    "ImageGenerationMiddleware": MiddlewareContract(
        name="ImageGenerationMiddleware",
        phase=MiddlewarePhase.TOOL_REGISTRATION,
        registers_tools=True,
        tool_names=["generate_image"],
        description="Provides image generation",
    ),
    "SubAgentMiddleware": MiddlewareContract(
        name="SubAgentMiddleware",
        phase=MiddlewarePhase.ORCHESTRATION,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["task"],
        prompt_budget=PromptBudget(max_tokens=500, priority=60),
        requires=["FilesystemMiddleware"],  # Subagents need filesystem
        description="Provides subagent task delegation",
    ),
    "CompactionMiddleware": MiddlewareContract(
        name="CompactionMiddleware",
        phase=MiddlewarePhase.CONTEXT_MANAGEMENT,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["save_artifact", "read_artifact", "search_artifacts", "retrieve_snippets"],
        modifies_tool_output=True,
        prompt_budget=PromptBudget(max_tokens=400, priority=70),
        description="Masks large outputs and provides artifact tools (legacy)",
    ),
    "ContextEngineeringMiddleware": MiddlewareContract(
        name="ContextEngineeringMiddleware",
        phase=MiddlewarePhase.CONTEXT_MANAGEMENT,
        injects_prompt=True,
        registers_tools=True,
        tool_names=["read_artifact", "search_artifacts"],
        modifies_tool_output=True,
        prompt_budget=PromptBudget(max_tokens=500, priority=70),
        conflicts_with=["CompactionMiddleware", "SummarizationMiddleware"],
        description="Unified context engineering: artifact masking + budget tracking + summarization coordination",
    ),
    "SummarizationMiddleware": MiddlewareContract(
        name="SummarizationMiddleware",
        phase=MiddlewarePhase.CONTEXT_MANAGEMENT,
        modifies_tool_output=True,
        description="Auto-summarizes conversation when over threshold (legacy)",
    ),
    "PromptCachingMiddleware": MiddlewareContract(
        name="PromptCachingMiddleware",
        phase=MiddlewarePhase.MODEL_OPTIMIZATION,
        description="Caches system prompts to reduce costs and latency",
    ),
    "PatchToolCallsMiddleware": MiddlewareContract(
        name="PatchToolCallsMiddleware",
        phase=MiddlewarePhase.MODEL_OPTIMIZATION,
        modifies_tool_output=True,
        description="Patches tool call format issues",
    ),
    "HumanInTheLoopMiddleware": MiddlewareContract(
        name="HumanInTheLoopMiddleware",
        phase=MiddlewarePhase.APPROVAL_GATING,
        can_interrupt=True,
        description="Interrupts for human approval on configured tools",
    ),
    "MCPMiddleware": MiddlewareContract(
        name="MCPMiddleware",
        phase=MiddlewarePhase.TOOL_REGISTRATION,
        injects_prompt=True,
        registers_tools=True,
        prompt_budget=PromptBudget(max_tokens=500, priority=50, can_be_trimmed=True),
        description="Loads tools from MCP servers",
    ),
}


@dataclass
class ValidationResult:
    """Result of middleware stack validation."""

    valid: bool
    """Whether the stack is valid."""

    errors: list[str] = field(default_factory=list)
    """Critical errors that must be fixed."""

    warnings: list[str] = field(default_factory=list)
    """Non-critical issues that should be reviewed."""

    total_prompt_budget: int = 0
    """Total estimated prompt tokens from all middleware."""

    tool_count: int = 0
    """Total number of tools registered."""


def get_middleware_name(middleware: "AgentMiddleware") -> str:
    """Get the contract name for a middleware instance."""
    return type(middleware).__name__


def validate_middleware_stack(
    middleware_list: list["AgentMiddleware"],
    max_prompt_budget: int = 6000,
) -> ValidationResult:
    """Validate a middleware stack for ordering and conflicts.

    Args:
        middleware_list: List of middleware instances in order.
        max_prompt_budget: Maximum total prompt tokens allowed.

    Returns:
        ValidationResult with errors, warnings, and metrics.
    """
    result = ValidationResult(valid=True)
    seen_names: list[str] = []
    seen_tools: set[str] = set()
    last_phase = MiddlewarePhase.CONTEXT_LOADING

    for i, mw in enumerate(middleware_list):
        name = get_middleware_name(mw)
        contract = MIDDLEWARE_CONTRACTS.get(name)

        if contract is None:
            result.warnings.append(f"Unknown middleware at position {i}: {name}")
            seen_names.append(name)
            continue

        # Check phase ordering
        if contract.phase < last_phase:
            result.errors.append(
                f"Middleware '{name}' (phase {contract.phase.name}) "
                f"is out of order - should come before phase {last_phase.name}"
            )
            result.valid = False
        last_phase = max(last_phase, contract.phase)

        # Check dependencies
        for req in contract.requires:
            if req not in seen_names:
                result.errors.append(
                    f"Middleware '{name}' requires '{req}' but it was not found before"
                )
                result.valid = False

        # Check conflicts
        for conflict in contract.conflicts_with:
            if conflict in seen_names:
                result.warnings.append(
                    f"Middleware '{name}' conflicts with '{conflict}' - review usage"
                )

        # Check tool conflicts
        for tool in contract.tool_names:
            if tool in seen_tools:
                result.errors.append(
                    f"Tool '{tool}' registered by '{name}' conflicts with earlier middleware"
                )
                result.valid = False
            seen_tools.add(tool)

        # Accumulate budget
        if contract.prompt_budget:
            result.total_prompt_budget += contract.prompt_budget.max_tokens

        seen_names.append(name)

    result.tool_count = len(seen_tools)

    # Check total budget
    if result.total_prompt_budget > max_prompt_budget:
        result.warnings.append(
            f"Total prompt budget ({result.total_prompt_budget}) exceeds max ({max_prompt_budget})"
        )

    return result

