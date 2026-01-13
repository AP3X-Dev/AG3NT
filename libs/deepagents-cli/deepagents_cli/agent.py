"""Agent management and creation for the CLI."""

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.sandbox import SandboxBackendProtocol
from deepagents.compaction import CompactionConfig, CompactionMiddleware
from deepagents.middleware import (
    AdvancedMiddleware,
    ImageGenerationMiddleware,
    MemoryMiddleware,
    SkillsMiddleware,
    UtilitiesMiddleware,
    WebMiddleware,
)
from deepagents.middleware.mcp import MCPMiddleware
from deepagents.mcp import MCPConfig, MCPServerConfig
from deepagents.research import (
    ResearchConfig,
    ResearchOrchestrator,
    ResearchSession,
)
from langchain.agents.middleware import (
    InterruptOnConfig,
)
from langchain.agents.middleware.types import AgentState
from langchain.messages import ToolCall
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from deepagents_cli.config import COLORS, config, console, get_default_coding_instructions, settings
from deepagents_cli.integrations.sandbox_factory import get_default_working_dir
from deepagents_cli.shell import ShellMiddleware


def list_agents() -> None:
    """List all available agents."""
    agents_dir = settings.user_deepagents_dir

    if not agents_dir.exists() or not any(agents_dir.iterdir()):
        console.print("[yellow]No agents found.[/yellow]")
        console.print(
            "[dim]Agents will be created in ~/.deepagents/ when you first use them.[/dim]",
            style=COLORS["dim"],
        )
        return

    console.print("\n[bold]Available Agents:[/bold]\n", style=COLORS["primary"])

    for agent_path in sorted(agents_dir.iterdir()):
        if agent_path.is_dir():
            agent_name = agent_path.name
            agent_md = agent_path / "AGENTS.md"

            if agent_md.exists():
                console.print(f"  • [bold]{agent_name}[/bold]", style=COLORS["primary"])
                console.print(f"    {agent_path}", style=COLORS["dim"])
            else:
                console.print(
                    f"  • [bold]{agent_name}[/bold] [dim](incomplete)[/dim]", style=COLORS["tool"]
                )
                console.print(f"    {agent_path}", style=COLORS["dim"])

    console.print()


def reset_agent(agent_name: str, source_agent: str | None = None) -> None:
    """Reset an agent to default or copy from another agent."""
    agents_dir = settings.user_deepagents_dir
    agent_dir = agents_dir / agent_name

    if source_agent:
        source_dir = agents_dir / source_agent
        source_md = source_dir / "AGENTS.md"

        if not source_md.exists():
            console.print(
                f"[bold red]Error:[/bold red] Source agent '{source_agent}' not found "
                "or has no AGENTS.md"
            )
            return

        source_content = source_md.read_text()
        action_desc = f"contents of agent '{source_agent}'"
    else:
        source_content = get_default_coding_instructions()
        action_desc = "default"

    if agent_dir.exists():
        shutil.rmtree(agent_dir)
        console.print(f"Removed existing agent directory: {agent_dir}", style=COLORS["tool"])

    agent_dir.mkdir(parents=True, exist_ok=True)
    agent_md = agent_dir / "AGENTS.md"
    agent_md.write_text(source_content)

    console.print(f"✓ Agent '{agent_name}' reset to {action_desc}", style=COLORS["primary"])
    console.print(f"Location: {agent_dir}\n", style=COLORS["dim"])


def get_system_prompt(assistant_id: str, sandbox_type: str | None = None) -> str:
    """Get the base system prompt for the agent.

    Args:
        assistant_id: The agent identifier for path references
        sandbox_type: Type of sandbox provider ("modal", "runloop", "daytona").
                     If None, agent is operating in local mode.

    Returns:
        The system prompt string (without AGENTS.md content)
    """
    agent_dir_path = f"~/.deepagents/{assistant_id}"

    if sandbox_type:
        # Get provider-specific working directory

        working_dir = get_default_working_dir(sandbox_type)

        working_dir_section = f"""### Current Working Directory

You are operating in a **remote Linux sandbox** at `{working_dir}`.

All code execution and file operations happen in this sandbox environment.

**Important:**
- The CLI is running locally on the user's machine, but you execute code remotely
- Use `{working_dir}` as your working directory for all operations

"""
    else:
        cwd = Path.cwd()
        working_dir_section = f"""<env>
Working directory: {cwd}
</env>

### Current Working Directory

The filesystem backend is currently operating in: `{cwd}`

### File System and Paths

**IMPORTANT - Path Handling:**
- All file paths must be absolute paths (e.g., `{cwd}/file.txt`)
- Use the working directory from <env> to construct absolute paths
- Example: To create a file in your working directory, use `{cwd}/research_project/file.md`
- Never use relative paths - always construct full absolute paths

"""

    return (
        working_dir_section
        + f"""### Skills Directory

Your skills are stored at: `{agent_dir_path}/skills/`
Skills may contain scripts or supporting files. When executing skill scripts with bash, use the real filesystem path:
Example: `bash python {agent_dir_path}/skills/web-research/script.py`

### Human-in-the-Loop Tool Approval

Some tool calls require user approval before execution. When a tool call is rejected by the user:
1. Accept their decision immediately - do NOT retry the same command
2. Explain that you understand they rejected the action
3. Suggest an alternative approach or ask for clarification
4. Never attempt the exact same rejected command again

Respect the user's decisions and work with them collaboratively.

### Web Search Tool Usage

When you use the web_search tool:
1. The tool will return search results with titles, URLs, and content excerpts
2. You MUST read and process these results, then respond naturally to the user
3. NEVER show raw JSON or tool results directly to the user
4. Synthesize the information from multiple sources into a coherent answer
5. Cite your sources by mentioning page titles or URLs when relevant
6. If the search doesn't find what you need, explain what you found and ask clarifying questions

The user only sees your text responses - not tool results. Always provide a complete, natural language answer after using web_search.

### Todo List Management

When using the write_todos tool:
1. Keep the todo list MINIMAL - aim for 3-6 items maximum
2. Only create todos for complex, multi-step tasks that truly need tracking
3. Break down work into clear, actionable items without over-fragmenting
4. For simple tasks (1-2 steps), just do them directly without creating todos
5. When first creating a todo list for a task, ALWAYS ask the user if the plan looks good before starting work
   - Create the todos, let them render, then ask: "Does this plan look good?" or similar
   - Wait for the user's response before marking the first todo as in_progress
   - If they want changes, adjust the plan accordingly
6. Update todo status promptly as you complete each item

The todo list is a planning tool - use it judiciously to avoid overwhelming the user with excessive task tracking."""
    )


def _format_write_file_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format write_file tool call for approval prompt."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    content = args.get("content", "")

    action = "Overwrite" if Path(file_path).exists() else "Create"
    line_count = len(content.splitlines())

    return f"File: {file_path}\nAction: {action} file\nLines: {line_count}"


def _format_edit_file_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format edit_file tool call for approval prompt."""
    args = tool_call["args"]
    file_path = args.get("file_path", "unknown")
    replace_all = bool(args.get("replace_all", False))

    return (
        f"File: {file_path}\n"
        f"Action: Replace text ({'all occurrences' if replace_all else 'single occurrence'})"
    )


def _format_web_search_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format web_search tool call for approval prompt."""
    args = tool_call["args"]
    query = args.get("query", "unknown")
    max_results = args.get("max_results", 5)

    return f"Query: {query}\nMax results: {max_results}\n\n⚠️  This will use Tavily API credits"


def _format_fetch_url_description(
    tool_call: ToolCall, _state: AgentState, _runtime: Runtime
) -> str:
    """Format fetch_url tool call for approval prompt."""
    args = tool_call["args"]
    url = args.get("url", "unknown")
    timeout = args.get("timeout", 30)

    return f"URL: {url}\nTimeout: {timeout}s\n\n⚠️  Will fetch and convert web content to markdown"


def _format_task_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format task (subagent) tool call for approval prompt.

    The task tool signature is: task(description: str, subagent_type: str)
    The description contains all instructions that will be sent to the subagent.
    """
    args = tool_call["args"]
    description = args.get("description", "unknown")
    subagent_type = args.get("subagent_type", "unknown")

    # Truncate description if too long for display
    description_preview = description
    if len(description) > 500:
        description_preview = description[:500] + "..."

    return (
        f"Subagent Type: {subagent_type}\n\n"
        f"Task Instructions:\n"
        f"{'─' * 40}\n"
        f"{description_preview}\n"
        f"{'─' * 40}\n\n"
        f"⚠️  Subagent will have access to file operations and shell commands"
    )


def _format_shell_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format shell tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "N/A")
    return f"Shell Command: {command}\nWorking Directory: {Path.cwd()}"


def _format_execute_description(tool_call: ToolCall, _state: AgentState, _runtime: Runtime) -> str:
    """Format execute tool call for approval prompt."""
    args = tool_call["args"]
    command = args.get("command", "N/A")
    return f"Execute Command: {command}\nLocation: Remote Sandbox"


def _add_interrupt_on() -> dict[str, InterruptOnConfig]:
    """Configure human-in-the-loop interrupt_on settings for destructive tools."""
    shell_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_shell_description,
    }

    execute_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_execute_description,
    }

    write_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_write_file_description,
    }

    edit_file_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_edit_file_description,
    }

    web_search_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_web_search_description,
    }

    fetch_url_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_fetch_url_description,
    }

    task_interrupt_config: InterruptOnConfig = {
        "allowed_decisions": ["approve", "reject"],
        "description": _format_task_description,
    }
    return {
        "shell": shell_interrupt_config,
        "execute": execute_interrupt_config,
        "write_file": write_file_interrupt_config,
        "edit_file": edit_file_interrupt_config,
        "web_search": web_search_interrupt_config,
        "fetch_url": fetch_url_interrupt_config,
        "task": task_interrupt_config,
    }


def create_research_tool(workspace_dir: Path) -> BaseTool:
    """Create a deep research tool that runs multi-step web research.

    This tool runs the ResearchOrchestrator to:
    - Search for sources on a topic
    - Read and analyze web pages
    - Extract evidence and findings
    - Return a structured research bundle

    Args:
        workspace_dir: Directory for research session artifacts.

    Returns:
        A StructuredTool for deep research.
    """
    RESEARCH_DESCRIPTION = """Run deep multi-step research on a topic.

This tool performs comprehensive web research by:
1. Generating search queries for the goal
2. Finding and ranking relevant sources
3. Reading and extracting information from web pages
4. Building an evidence-backed research bundle with findings

Parameters:
- goal: The research question or topic (be specific)
- constraints: Optional dict with constraints like {"recency": "last_30_days"}
- required_outputs: Optional list of specific items to find

Use this tool for:
- Researching market trends, pricing, or competitive analysis
- Finding technical documentation or specifications
- Gathering information that requires multiple sources
- Questions that need current, up-to-date information

Returns:
- Executive summary of findings
- List of findings with confidence levels
- Evidence sources with URLs
- Open questions for follow-up

Note: This runs a multi-step research process and may take 30-60 seconds."""

    async def async_deep_research(
        goal: str,
        constraints: dict[str, Any] | None = None,
        required_outputs: list[str] | None = None,
    ) -> str:
        """Run deep research on a topic."""
        # Create a research session
        session = ResearchSession.create(
            workspace_dir=workspace_dir / "research_sessions",
            config=ResearchConfig(
                max_sources=10,
                max_steps=15,
            ),
        )

        # Create orchestrator and run research
        orchestrator = ResearchOrchestrator(session=session)
        bundle = await orchestrator.research(
            goal=goal,
            constraints=constraints,
            required_outputs=required_outputs,
        )

        # Format the bundle for response
        lines = [
            "## Research Results",
            "",
            f"**Summary:** {bundle.executive_summary}",
            "",
        ]

        if bundle.findings:
            lines.append("### Findings")
            for i, finding in enumerate(bundle.findings[:10], 1):
                conf_emoji = {"high": "✓", "medium": "○", "low": "?"}
                emoji = conf_emoji.get(finding.confidence.value, "○")
                lines.append(f"{i}. [{emoji}] {finding.claim}")
            lines.append("")

        if bundle.evidence:
            lines.append("### Sources Consulted")
            for ev in bundle.evidence[:5]:
                title = ev.title or ev.url
                lines.append(f"- [{title}]({ev.url})")
            if len(bundle.evidence) > 5:
                lines.append(f"- ... and {len(bundle.evidence) - 5} more sources")
            lines.append("")

        if bundle.open_questions:
            lines.append("### Open Questions")
            for q in bundle.open_questions[:3]:
                lines.append(f"- {q}")

        return "\n".join(lines)

    def sync_deep_research(
        goal: str,
        constraints: dict[str, Any] | None = None,
        required_outputs: list[str] | None = None,
    ) -> str:
        """Sync wrapper for deep research."""
        return asyncio.get_event_loop().run_until_complete(
            async_deep_research(goal, constraints, required_outputs)
        )

    return StructuredTool.from_function(
        name="deep_research",
        description=RESEARCH_DESCRIPTION,
        func=sync_deep_research,
        coroutine=async_deep_research,
    )


def create_cli_agent(
    model: str | BaseChatModel,
    assistant_id: str,
    *,
    tools: list[BaseTool] | None = None,
    sandbox: SandboxBackendProtocol | None = None,
    sandbox_type: str | None = None,
    system_prompt: str | None = None,
    auto_approve: bool = False,
    enable_memory: bool = True,
    enable_skills: bool = True,
    enable_shell: bool = True,
    enable_web: bool = True,
    enable_utilities: bool = True,
    enable_compaction: bool = True,
    enable_research: bool = True,
    enable_browser: bool = True,
    checkpointer: BaseCheckpointSaver | None = None,
) -> tuple[Pregel, CompositeBackend]:
    """Create a CLI-configured agent with flexible options.

    This is the main entry point for creating a deepagents CLI agent, usable both
    internally and from external code (e.g., benchmarking frameworks, Harbor).

    Args:
        model: LLM model to use (e.g., "anthropic:claude-sonnet-4-5-20250929")
        assistant_id: Agent identifier for memory/state storage
        tools: Additional tools to provide to agent
        sandbox: Optional sandbox backend for remote execution (e.g., ModalBackend).
                 If None, uses local filesystem + shell.
        sandbox_type: Type of sandbox provider ("modal", "runloop", "daytona").
                     Used for system prompt generation.
        system_prompt: Override the default system prompt. If None, generates one
                      based on sandbox_type and assistant_id.
        auto_approve: If True, automatically approves all tool calls without human
                     confirmation. Useful for automated workflows.
        enable_memory: Enable MemoryMiddleware for persistent memory
        enable_skills: Enable SkillsMiddleware for custom agent skills
        enable_shell: Enable ShellMiddleware for local shell execution (only in local mode)
        enable_web: Enable WebMiddleware for web_search and read_web_page tools
        enable_utilities: Enable UtilitiesMiddleware for undo_edit, mermaid, etc.
        enable_compaction: Enable CompactionMiddleware for context window management
        enable_research: Enable deep_research tool for multi-step web research
        enable_browser: Enable browser automation via Playwright MCP (requires @playwright/mcp)
        checkpointer: Optional checkpointer for session persistence. If None, uses
                     InMemorySaver (no persistence across CLI invocations).

    Returns:
        2-tuple of (agent_graph, backend)
        - agent_graph: Configured LangGraph Pregel instance ready for execution
        - composite_backend: CompositeBackend for file operations
    """
    tools = list(tools) if tools else []

    # Add deep_research tool if enabled
    if enable_research:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        research_tool = create_research_tool(workspace_dir=agent_dir)
        tools.append(research_tool)

    # Setup agent directory for persistent memory (if enabled)
    if enable_memory or enable_skills:
        agent_dir = settings.ensure_agent_dir(assistant_id)
        agent_md = agent_dir / "AGENTS.md"
        if not agent_md.exists():
            source_content = get_default_coding_instructions()
            agent_md.write_text(source_content)

    # Skills directories (if enabled)
    skills_dir = None
    project_skills_dir = None
    if enable_skills:
        skills_dir = settings.ensure_user_skills_dir(assistant_id)
        project_skills_dir = settings.get_project_skills_dir()

    # Build middleware stack based on enabled features
    agent_middleware = []

    # Add memory middleware
    if enable_memory:
        memory_sources = [str(settings.get_user_agent_md_path(assistant_id))]
        project_agent_md = settings.get_project_agent_md_path()
        if project_agent_md:
            memory_sources.append(str(project_agent_md))

        agent_middleware.append(
            MemoryMiddleware(
                backend=FilesystemBackend(),
                sources=memory_sources,
            )
        )

    # Add skills middleware
    if enable_skills:
        sources = [str(skills_dir)]
        if project_skills_dir:
            sources.append(str(project_skills_dir))

        agent_middleware.append(
            SkillsMiddleware(
                backend=FilesystemBackend(),
                sources=sources,
            )
        )

    # CONDITIONAL SETUP: Local vs Remote Sandbox
    if sandbox is None:
        # ========== LOCAL MODE ==========
        backend = FilesystemBackend()  # Current working directory

        # Add shell middleware (only in local mode)
        if enable_shell:
            # Create environment for shell commands
            # Restore user's original LANGSMITH_PROJECT so their code traces separately
            shell_env = os.environ.copy()
            if settings.user_langchain_project:
                shell_env["LANGSMITH_PROJECT"] = settings.user_langchain_project

            agent_middleware.append(
                ShellMiddleware(
                    workspace_root=str(Path.cwd()),
                    env=shell_env,
                )
            )
    else:
        # ========== REMOTE SANDBOX MODE ==========
        backend = sandbox  # Remote sandbox (ModalBackend, etc.)
        # Note: Shell middleware not used in sandbox mode
        # File operations and execute tool are provided by the sandbox backend

    # Add image generation middleware (uses OPENROUTER_API_KEY from env)
    agent_middleware.append(
        ImageGenerationMiddleware(backend=backend)
    )

    # Add advanced middleware for look_at (vision analysis) and finder (semantic search)
    # Pass the model for vision capabilities
    agent_middleware.append(
        AdvancedMiddleware(backend=backend, model=model)
    )

    # Add web middleware for web_search and read_web_page tools
    if enable_web:
        agent_middleware.append(WebMiddleware())

    # Add utilities middleware for undo_edit, mermaid, format_file, get_diagnostics
    if enable_utilities:
        agent_middleware.append(UtilitiesMiddleware(backend=backend))

    # Add compaction middleware for context window management and artifact storage
    if enable_compaction:
        # Create workspace for artifacts in agent directory
        compaction_workspace = settings.ensure_agent_dir(assistant_id) / "compaction"
        compaction_workspace.mkdir(exist_ok=True)
        agent_middleware.append(
            CompactionMiddleware(
                config=CompactionConfig(
                    workspace_dir=compaction_workspace,
                    mask_tool_output_if_chars_gt=8000,  # Mask outputs > 8KB
                    keep_last_unmasked_tool_outputs=5,  # Keep last 5 unmasked
                ),
            )
        )

    # Add browser automation middleware via Playwright MCP
    if enable_browser:
        try:
            # Configure Playwright MCP server (requires: npx @anthropic-ai/claude-code mcp add @playwright/mcp)
            browser_mcp_config = MCPConfig(
                servers={
                    "browser": MCPServerConfig(
                        transport="stdio",
                        command="npx",
                        args=["@playwright/mcp@latest"],
                        tool_name_prefix=False,  # Don't prefix with "browser_"
                    ),
                },
            )
            agent_middleware.append(MCPMiddleware(config=browser_mcp_config))
        except ImportError:
            # MCP dependencies not installed, skip browser tools
            pass

    # Get or use custom system prompt
    if system_prompt is None:
        system_prompt = get_system_prompt(assistant_id=assistant_id, sandbox_type=sandbox_type)

    # Configure interrupt_on based on auto_approve setting
    if auto_approve:
        # No interrupts - all tools run automatically
        interrupt_on = {}
    else:
        # Full HITL for destructive operations
        interrupt_on = _add_interrupt_on()

    composite_backend = CompositeBackend(
        default=backend,
        routes={},
    )

    # Create the agent
    # Use provided checkpointer or fallback to InMemorySaver
    final_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    agent = create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        backend=composite_backend,
        middleware=agent_middleware,
        interrupt_on=interrupt_on,
        checkpointer=final_checkpointer,
    ).with_config(config)
    return agent, composite_backend
