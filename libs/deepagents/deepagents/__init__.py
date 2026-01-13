"""DeepAgents package.

A comprehensive framework for building context-engineered AI agents with:
- Compaction: Intelligent context management and summarization
- Approval: Tool risk classification and human-in-the-loop policies
- Subagent: Contained subagent execution with return contracts
- Context Engineering: Token budget management and prompt assembly
"""

from deepagents.agent_factory import (
    AG3NTConfig,
    WorkspaceLayout,
    build_ag3nt_middleware_stack,
    get_default_workspace_layout,
)
from deepagents.graph import create_deep_agent
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware
from deepagents.openrouter import get_default_openrouter_model, get_openrouter_model, is_openrouter_configured

# Context Engineering exports
from deepagents.context_engineering import (
    ContextEngineeringConfig,
    ContextEngineeringMiddleware,
    TokenBudgetTracker,
)

# Approval Policy exports
from deepagents.approval import (
    ApprovalLedger,
    ApprovalPolicy,
    ApprovalRecord,
    ToolRiskLevel,
)

# Subagent Containment exports
from deepagents.subagent import (
    ContainedSubAgentMiddleware,
    DistilledReturnContract,
    SubagentConfig,
)

__all__ = [
    # Core
    "AG3NTConfig",
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "MemoryMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "WorkspaceLayout",
    "build_ag3nt_middleware_stack",
    "create_deep_agent",
    "get_default_openrouter_model",
    "get_default_workspace_layout",
    "get_openrouter_model",
    "is_openrouter_configured",
    # Context Engineering
    "ContextEngineeringConfig",
    "ContextEngineeringMiddleware",
    "TokenBudgetTracker",
    # Approval Policy
    "ApprovalLedger",
    "ApprovalPolicy",
    "ApprovalRecord",
    "ToolRiskLevel",
    # Subagent Containment
    "ContainedSubAgentMiddleware",
    "DistilledReturnContract",
    "SubagentConfig",
]

# Optional MCP exports (requires deepagents[mcp])
try:
    from deepagents.mcp import MCPConfig, MCPServerConfig, FailBehavior

    __all__.extend(["MCPConfig", "MCPServerConfig", "FailBehavior"])
except ImportError:
    # MCP dependencies not installed
    pass
