"""Context Engineering - Unified context window management.

This module provides a unified layer that coordinates all context management:
1. CompactionMiddleware: Artifact masking and evidence tracking
2. SummarizationMiddleware: Message summarization (from upstream)
3. Token budget tracking and reporting

The ContextEngineeringMiddleware wraps both to prevent conflicts and ensure
coordinated behavior.

Usage:
    from deepagents.context_engineering import ContextEngineeringMiddleware

    # This replaces both CompactionMiddleware and SummarizationMiddleware
    middleware = ContextEngineeringMiddleware(
        model=model,
        workspace_dir=workspace_dir,
    )
    agent = create_deep_agent(middleware=[middleware])
"""

from deepagents.context_engineering.budget import TokenBudgetTracker
from deepagents.context_engineering.cache import CacheStats, PromptAssemblyCache
from deepagents.context_engineering.config import ContextEngineeringConfig
from deepagents.context_engineering.middleware import ContextEngineeringMiddleware

__all__ = [
    "CacheStats",
    "ContextEngineeringConfig",
    "ContextEngineeringMiddleware",
    "PromptAssemblyCache",
    "TokenBudgetTracker",
]
