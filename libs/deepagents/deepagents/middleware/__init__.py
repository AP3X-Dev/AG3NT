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
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.middleware.utilities import UtilitiesMiddleware
from deepagents.middleware.web import WebMiddleware

__all__ = [
    "AdvancedMiddleware",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "ImageGenerationMiddleware",
    "MIDDLEWARE_CONTRACTS",
    "MemoryMiddleware",
    "MiddlewareContract",
    "MiddlewarePhase",
    "PromptBudget",
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

    __all__.append("MCPMiddleware")
except ImportError:
    # MCP dependencies not installed
    pass
