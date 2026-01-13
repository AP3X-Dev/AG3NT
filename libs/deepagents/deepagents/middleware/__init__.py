"""Middleware for the DeepAgent."""

from deepagents.middleware.advanced import (
    AdvancedMiddleware,
    get_librarian_subagent,
    get_oracle_subagent,
)
from deepagents.middleware.contracts import (
    MIDDLEWARE_CONTRACTS,
    MiddlewareContract,
    MiddlewarePhase,
    PromptBudget,
    ValidationResult,
    validate_middleware_stack,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.image_generation import ImageGenerationMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.prompt_caching import PromptCachingMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.middleware.utilities import UtilitiesMiddleware
from deepagents.middleware.web import WebMiddleware

__all__ = [
    "MIDDLEWARE_CONTRACTS",
    "AdvancedMiddleware",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "ImageGenerationMiddleware",
    "MemoryMiddleware",
    "MiddlewareContract",
    "MiddlewarePhase",
    "PatchToolCallsMiddleware",
    "PromptBudget",
    "PromptCachingMiddleware",
    "SkillsMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "UtilitiesMiddleware",
    "ValidationResult",
    "WebMiddleware",
    "get_librarian_subagent",
    "get_oracle_subagent",
    "validate_middleware_stack",
]

# Optional MCP middleware (requires deepagents[mcp])
try:
    from deepagents.middleware.mcp import MCPMiddleware

    __all__ += ["MCPMiddleware"]
except ImportError:
    # MCP dependencies not installed
    pass
