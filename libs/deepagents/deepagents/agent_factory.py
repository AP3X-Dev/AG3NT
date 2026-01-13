"""AG3NT Agent Factory - Canonical stack builder for all AG3NT agents.

This module provides the standard AG3NT middleware stack that should be used
by ALL callers (CLI, API, eval harness, embedded). The CLI is just a thin
UX wrapper around this factory.

## Architecture Principles

1. **AG3NT Standard Stack**: Memory, Skills, Web, Utilities, Compaction, ImageGen
   are baseline capabilities for every AG3NT instance.

2. **Environment-Conditional Middleware**:
   - ShellMiddleware: Local backend only (not "CLI only")
   - execute tool: Sandbox backend only

3. **UI Layer is Pure UX**: CLI/API/embedded just select backend and call factory.

## Usage

```python
from deepagents.agent_factory import create_ag3nt_agent, AG3NTConfig

# Local mode
agent = create_ag3nt_agent(
    config=AG3NTConfig(
        backend_type="local",
        workspace_root=Path.cwd(),
    )
)

# Sandbox mode
agent = create_ag3nt_agent(
    config=AG3NTConfig(
        backend_type="sandbox",
        sandbox=my_sandbox_backend,
    )
)
```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Checkpointer

    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceLayout:
    """Canonical workspace layout for AG3NT agents.
    
    All paths are relative to workspace_root.
    """
    
    workspace_root: Path
    """Root directory for all agent data."""
    
    # Standard subdirectories
    compaction_dir: str = "compaction"
    """Directory for compaction artifacts."""
    
    skills_dir: str = "skills"
    """Directory for user skills."""
    
    memory_file: str = "AGENTS.md"
    """Memory file name."""
    
    research_dir: str = "research_sessions"
    """Directory for research session data."""
    
    def get_compaction_path(self) -> Path:
        """Get full path to compaction directory."""
        return self.workspace_root / self.compaction_dir
    
    def get_skills_path(self) -> Path:
        """Get full path to skills directory."""
        return self.workspace_root / self.skills_dir
    
    def get_memory_path(self) -> Path:
        """Get full path to memory file."""
        return self.workspace_root / self.memory_file
    
    def get_research_path(self) -> Path:
        """Get full path to research directory."""
        return self.workspace_root / self.research_dir
    
    def ensure_directories(self) -> None:
        """Create all required directories."""
        self.get_compaction_path().mkdir(parents=True, exist_ok=True)
        self.get_skills_path().mkdir(parents=True, exist_ok=True)
        self.get_research_path().mkdir(parents=True, exist_ok=True)


@dataclass
class AG3NTConfig:
    """Configuration for creating an AG3NT agent.

    This captures all the configuration needed to build the standard
    AG3NT middleware stack. The stack is NOT optional - all AG3NT agents
    have the same core capabilities. This config only controls:

    1. Environment-specific behavior (local vs sandbox)
    2. Paths to user-provided content (memory files, skills dirs)
    3. Tuning parameters (compaction thresholds, etc.)

    If you don't want the full AG3NT stack, use create_deep_agent() directly.
    """

    # Backend configuration
    backend_type: Literal["local", "sandbox"] = "local"
    """Whether running locally or in a sandbox."""

    workspace_root: Path | None = None
    """Root directory for agent workspace. Defaults to ~/.deepagents/{agent_id}/"""

    agent_id: str = "default"
    """Unique identifier for this agent instance."""

    # User-provided content paths
    memory_sources: list[str] = field(default_factory=list)
    """Paths to AGENTS.md files to load. Empty = no memory injection."""

    skills_sources: list[str] = field(default_factory=list)
    """Paths to skills directories. Empty = no skills injection."""

    # Compaction tuning (compaction itself is NOT optional)
    mask_tool_output_if_chars_gt: int = 8000
    """Mask tool outputs larger than this."""

    keep_last_unmasked_tool_outputs: int = 5
    """Keep last N tool outputs unmasked."""

    # Shell environment (local mode only)
    shell_env: dict[str, str] | None = None
    """Environment variables for shell commands."""


def build_ag3nt_middleware_stack(
    config: AG3NTConfig,
    backend: "BackendProtocol",
    workspace_layout: WorkspaceLayout,
) -> list["AgentMiddleware"]:
    """Build the canonical AG3NT middleware stack.

    ALL AG3NT agents get the same core capabilities:
    - Memory (if sources provided)
    - Skills (if sources provided)
    - Shell (local mode) / Execute (sandbox mode)
    - Image generation
    - Web search + page reading
    - Utilities (undo, format, diagnostics)
    - Compaction (ALWAYS - this is core infrastructure)

    The stack is NOT configurable beyond paths and tuning params.
    If you want a different stack, use create_deep_agent() directly.

    Args:
        config: AG3NT configuration (paths and tuning only).
        backend: The backend to use for file operations.
        workspace_layout: Workspace layout for paths.

    Returns:
        List of middleware in canonical order (phases 1→2→4).
    """
    from deepagents.backends import FilesystemBackend
    from deepagents.context_engineering import (
        ContextEngineeringConfig,
        ContextEngineeringMiddleware,
    )
    from deepagents.middleware.image_generation import ImageGenerationMiddleware
    from deepagents.middleware.memory import MemoryMiddleware
    from deepagents.middleware.skills import SkillsMiddleware
    from deepagents.middleware.utilities import UtilitiesMiddleware
    from deepagents.middleware.web import WebMiddleware

    middleware: list[AgentMiddleware] = []

    # =========================================================================
    # PHASE 1: CONTEXT LOADING
    # Inject user-provided context into system prompt BEFORE agent runs.
    # These only fire if user provided sources - but the CAPABILITY is always there.
    # =========================================================================

    if config.memory_sources:
        middleware.append(
            MemoryMiddleware(
                backend=FilesystemBackend(),
                sources=config.memory_sources,
            )
        )

    if config.skills_sources:
        middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(),
                sources=config.skills_sources,
            )
        )

    # =========================================================================
    # PHASE 2: TOOL REGISTRATION
    # Core AG3NT tools. Environment-conditional = based on backend, not caller.
    # =========================================================================

    # Shell: local mode only (sandbox gets 'execute' via FilesystemMiddleware)
    if config.backend_type == "local":
        try:
            from deepagents_cli.shell import ShellMiddleware
            middleware.append(
                ShellMiddleware(
                    workspace_root=str(workspace_layout.workspace_root),
                    env=config.shell_env,
                )
            )
        except ImportError:
            logger.warning(
                "ShellMiddleware not available - install deepagents-cli for local shell"
            )

    # Core tools - ALWAYS present, no feature flags
    middleware.append(ImageGenerationMiddleware(backend=backend))
    middleware.append(WebMiddleware())
    middleware.append(UtilitiesMiddleware(backend=backend))

    # =========================================================================
    # PHASE 4: CONTEXT MANAGEMENT
    # Context Engineering is CORE INFRASTRUCTURE - not optional.
    # Unifies artifact masking + summarization coordination + budget tracking.
    # Without it, long conversations blow up the context window.
    # =========================================================================

    context_path = workspace_layout.get_compaction_path()
    context_path.mkdir(exist_ok=True)
    middleware.append(
        ContextEngineeringMiddleware(
            config=ContextEngineeringConfig(
                workspace_dir=context_path,
                mask_tool_output_if_chars_gt=config.mask_tool_output_if_chars_gt,
                keep_last_unmasked_tool_outputs=config.keep_last_unmasked_tool_outputs,
            ),
        )
    )

    return middleware


def get_default_workspace_layout(agent_id: str = "default") -> WorkspaceLayout:
    """Get the default workspace layout for an agent.

    Uses ~/.deepagents/{agent_id}/ as the root.

    Args:
        agent_id: Unique identifier for the agent.

    Returns:
        WorkspaceLayout with default paths.
    """
    home = Path.home()
    workspace_root = home / ".deepagents" / agent_id
    return WorkspaceLayout(workspace_root=workspace_root)

