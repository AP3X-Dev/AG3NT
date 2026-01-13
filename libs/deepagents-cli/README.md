# 🚀 AG3NT CLI

**Production-grade AI coding assistant for your terminal.**

Advanced capabilities:

- **19+ Built-in Tools**: Complete filesystem ops, shell execution, web research, code analysis, and more
- **Progressive Disclosure Skills**: Domain expertise loaded on-demand to minimize token usage
- **Persistent Memory**: Learns your coding style, project patterns, and preferences across sessions
- **Intelligent Subagents**: Delegate complex tasks to specialized agents (Librarian, Oracle, custom)
- **Context Compaction**: Handle massive codebases without hitting token limits
- **Multi-Provider LLM Support**: OpenRouter, Anthropic, OpenAI, Google - use any model
- **Remote Sandbox Execution**: Run code safely in Modal, Runloop, or Daytona environments
- **Human-in-the-Loop**: Configurable approval workflows for sensitive operations

<img src="cli-banner.jpg" alt="AG3NT CLI" width="100%"/>

## 🚀 Quick Start

### Installation

```bash
# Install via pip
pip install deepagents-cli

# Or using uv (recommended for faster installs)
uv pip install deepagents-cli
```

### Basic Usage

```bash
# Run with default settings (uses OpenRouter if configured)
ag3nt

# Or use the full command
deepagents
```

### Advanced Usage

```bash
# Use specific model (auto-detects provider)
ag3nt --model anthropic/claude-sonnet-4.5
ag3nt --model openai/gpt-5.2
ag3nt --model google/gemini-3-pro

# Use specific agent configuration
ag3nt --agent mybot

# Auto-approve all operations (for automation)
ag3nt --auto-approve

# Execute in remote sandbox
ag3nt --sandbox modal        # or runloop, daytona
ag3nt --sandbox-id dbx_123   # reuse existing sandbox

# Get help
ag3nt help
```

### First Run

On first run, AG3NT will:
1. Create configuration directory at `~/.deepagents/`
2. Load default skills and memory
3. Start an interactive session

Type naturally as you would in a chat. The agent will autonomously use its tools, skills, and memory to accomplish your goals.

## 🧠 Model Configuration

### OpenRouter (Recommended)

Access **100+ models** through a single API with automatic failover:

```bash
# Set up OpenRouter
export OPENROUTER_API_KEY=sk-or-v1-your-key
export OPENROUTER_MODEL=anthropic/claude-sonnet-4.5  # optional

# Run with OpenRouter
ag3nt
```

**Why OpenRouter?**
- Access to all major providers (Anthropic, OpenAI, Google, X.AI, Meta, etc.)
- Automatic fallback if a model is unavailable
- Cost tracking and optimization
- Single API key for everything

### Direct Provider Access

You can also use providers directly:

```bash
# Anthropic
export ANTHROPIC_API_KEY=your-key
ag3nt --model claude-sonnet-4-5-20250929

# OpenAI
export OPENAI_API_KEY=your-key
ag3nt --model gpt-5.2

# Google
export GOOGLE_API_KEY=your-key
ag3nt --model gemini-3-pro-preview
```

### Model Selection

```bash
# Use cutting-edge models via OpenRouter
ag3nt --model anthropic/claude-sonnet-4.5
ag3nt --model openai/gpt-5.2
ag3nt --model google/gemini-3-pro
ag3nt --model x-ai/grok-4.1-fast
ag3nt --model meta-llama/llama-4-405b

# Or direct provider models
ag3nt --model claude-sonnet-4-5-20250929  # Anthropic
ag3nt --model gpt-4o                       # OpenAI
ag3nt --model gemini-2.5-pro               # Google
```

The active model and provider are displayed on startup.

## 🛠️ Built-in Tools

AG3NT CLI comes with **19+ built-in tools** organized by category:

### Filesystem Operations
| Tool | Description |
|------|-------------|
| `ls` | List directory contents |
| `read_file` | Read file contents with pagination |
| `write_file` | Create or overwrite files |
| `edit_file` | Targeted string replacements |
| `glob` | Pattern-based file search (`**/*.py`) |
| `grep` | Text search across files |

### Code Development
| Tool | Description |
|------|-------------|
| `undo_edit` | Revert last file edit |
| `format_file` | Auto-format code (IDE integration) |
| `get_diagnostics` | Get IDE errors and warnings |
| `mermaid` | Generate diagrams |

### Execution
| Tool | Description |
|------|-------------|
| `shell` | Execute shell commands (local) |
| `execute` | Execute in sandbox (Modal/Runloop/Daytona) |

### Web & Research
| Tool | Description |
|------|-------------|
| `web_search` | Search the web (Tavily) |
| `read_web_page` | Fetch and parse URLs to markdown |
| `deep_research` | Multi-step research with synthesis |

### AI-Powered Analysis
| Tool | Description |
|------|-------------|
| `finder` | Semantic codebase search |
| `look_at` | AI-powered code analysis |
| `librarian` | Deep codebase understanding subagent |
| `oracle` | Expert technical guidance subagent |

### Task Management
| Tool | Description |
|------|-------------|
| `write_todos` | Create and manage task lists |
| `task` | Delegate to specialized subagents |

### Browser Automation (Optional MCP)
When Playwright MCP is configured, AG3NT can:
- Navigate and interact with web pages
- Fill forms and click elements
- Take screenshots and monitor network
- Extract data from dynamic sites

> [!WARNING]
> **Human-in-the-Loop (HITL) Approval**
>
> Sensitive operations require approval before execution:
> - **File writes**: `write_file`, `edit_file`
> - **Command execution**: `shell`, `execute`
> - **External requests**: `web_search`, `read_web_page`
> - **Delegation**: `task` (subagents)
>
> Use `--auto-approve` to skip prompts for automation:
> ```bash
> ag3nt --auto-approve
> ```

## 🎯 Key Features

### 1. Persistent Memory System
AG3NT learns and remembers across sessions:
- **Coding preferences**: Indentation, naming conventions, frameworks
- **Project patterns**: Architecture decisions, common workflows
- **User feedback**: Improves based on corrections and guidance

Memory is stored in `~/.deepagents/<agent>/AGENTS.md` and automatically loaded.

### 2. Progressive Disclosure Skills
Domain expertise loaded on-demand to minimize token usage:
- **Built-in skills**: `web-research`, `langgraph-docs`, `security-audit`
- **Custom skills**: Create your own in `~/.deepagents/<agent>/skills/`
- **Project skills**: Project-specific expertise in `.deepagents/skills/`

Skills can be applied as prompt modules or spawned as specialized subagents.

### 3. Intelligent Subagent Delegation
Spawn specialized agents for complex tasks:
- **Librarian**: Deep codebase understanding and analysis
- **Oracle**: Expert technical guidance and architecture advice
- **Custom subagents**: Define your own specialized agents

Subagents run in isolated contexts to prevent pollution of the main thread.

### 4. Context Compaction
Handle massive codebases without hitting token limits:
- Automatic artifact storage for large outputs
- Conversation history compaction
- On-demand detail retrieval
- Maintains working set in active context

### 5. Remote Sandbox Execution
Run code safely in isolated environments:
- **Modal**: Serverless Python execution
- **Runloop**: General-purpose sandboxes
- **Daytona**: Development environments

Prevents local system pollution and enables secure untrusted code execution.

## ⚙️ Agent Configuration

Each agent has its own configuration directory at `~/.deepagents/<agent_name>/`, with default `agent`.

```bash
# List all configured agents
ag3nt list

# Create a new agent
ag3nt create <agent_name>

# Use specific agent
ag3nt --agent <agent_name>
```

### Environment Variables

#### LangSmith Tracing

The CLI supports separate LangSmith project configuration for agent tracing vs user code tracing:

**Agent Tracing** - Traces deepagents operations (tool calls, agent decisions):

```bash
export DEEPAGENTS_LANGSMITH_PROJECT="my-agent-project"
```

**User Code Tracing** - Traces code executed via shell commands:

```bash
export LANGSMITH_PROJECT="my-user-code-project"
```

**Complete Setup Example:**

```bash
# Enable LangSmith tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"

# Configure separate projects
export DEEPAGENTS_LANGSMITH_PROJECT="agent-traces"
export LANGSMITH_PROJECT="user-code-traces"

# Run deepagents
deepagents
```

When both are configured, the CLI displays:

```
✓ LangSmith tracing enabled: Deepagents → 'agent-traces'
  User code (shell) → 'user-code-traces'
```

**Why separate projects?**

- Keep agent operations separate from your application code traces
- Easier debugging by isolating agent vs user code behavior
- Different retention policies or access controls per project

**Backwards Compatibility:**
If `DEEPAGENTS_LANGSMITH_PROJECT` is not set, both agent and user code trace to the same project specified by `LANGSMITH_PROJECT`.

## Customization

There are two primary ways to customize any agent: **memory** and **skills**.

Each agent has its own global configuration directory at `~/.deepagents/<agent_name>/`:

```
~/.deepagents/<agent_name>/
  ├── AGENTS.md              # Auto-loaded global personality/style
  └── skills/               # Auto-loaded agent-specific skills
      ├── web-research/
      │   └── SKILL.md
      └── langgraph-docs/
          └── SKILL.md
```

Projects can extend the global configuration with project-specific instructions and skills:

```
my-project/
  ├── .git/
  └── .deepagents/
      ├── AGENTS.md          # Project-specific instructions
      └── skills/           # Project-specific skills
          └── custom-tool/
              └── SKILL.md
```

The CLI automatically detects project roots (via `.git`) and loads:

- Project-specific `AGENTS.md` from `[project-root]/.deepagents/AGENTS.md`
- Project-specific skills from `[project-root]/.deepagents/skills/`

Both global and project configurations are loaded together, allowing you to:

- Keep general coding style/preferences in global AGENTS.md
- Add project-specific context, conventions, or guidelines in project AGENTS.md
- Share project-specific skills with your team (committed to version control)
- Override global skills with project-specific versions (when skill names match)

### AGENTS.md files

`AGENTS.md` files provide persistent memory that is always loaded at session start. Both global and project-level `AGENTS.md` files are loaded together and injected into the system prompt.

**Global `AGENTS.md`** (`~/.deepagents/agent/AGENTS.md`)

- Your personality, style, and universal coding preferences
- General tone and communication style
- Universal coding preferences (formatting, type hints, etc.)
- Tool usage patterns that apply everywhere
- Workflows and methodologies that don't change per-project

**Project `AGENTS.md`** (`.deepagents/AGENTS.md` in project root)

- Project-specific context and conventions
- Project architecture and design patterns
- Coding conventions specific to this codebase
- Testing strategies and deployment processes
- Team guidelines and project structure

**How it works:**

- Loads memory files at startup and injects into system prompt as `<agent_memory>`
- Includes guidelines on when/how to update memory files via `edit_file`

**When the agent updates memory:**

- IMMEDIATELY when you describe how it should behave
- IMMEDIATELY when you give feedback on its work
- When you explicitly ask it to remember something
- When patterns or preferences emerge from your interactions

The agent uses `edit_file` to update memories when learning preferences or receiving feedback.

### Project memory files

Beyond `AGENTS.md`, you can create additional memory files in `.deepagents/` for structured project knowledge. These work similarly to [Anthropic's Memory Tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool). The agent receives instructions on when to read and update these files.

**How it works:**

1. Create markdown files in `[project-root]/.deepagents/` (e.g., `api-design.md`, `architecture.md`, `deployment.md`)
2. The agent checks these files when relevant to a task (not auto-loaded into every prompt)
3. The agent uses `write_file` or `edit_file` to create/update memory files when learning project patterns

**Example workflow:**

```bash
# Agent discovers deployment pattern and saves it
.deepagents/
├── AGENTS.md           # Always loaded (personality + conventions)
├── architecture.md    # Loaded on-demand (system design)
└── deployment.md      # Loaded on-demand (deploy procedures)
```

**When the agent reads memory files:**

- At the start of new sessions (checks what files exist)
- Before answering questions about project-specific topics
- When you reference past work or patterns
- When performing tasks that match saved knowledge domains

**Benefits:**

- **Persistent learning**: Agent remembers project patterns across sessions
- **Team collaboration**: Share project knowledge through version control
- **Contextual retrieval**: Load only relevant memory when needed (reduces token usage)
- **Structured knowledge**: Organize information by domain (APIs, architecture, deployment, etc.)

### Skills

Skills are reusable agent capabilities that provide specialized workflows and domain knowledge. Example skills are provided in the `examples/skills/` directory:

- **web-research** - Structured web research workflow with planning, parallel delegation, and synthesis
- **langgraph-docs** - LangGraph documentation lookup and guidance

To use an example skill globally with the default agent, just copy them to the agent's skills global or project-level skills directory:

```bash
mkdir -p ~/.deepagents/agent/skills
cp -r examples/skills/web-research ~/.deepagents/agent/skills/
```

To manage skills:

```bash
# List all skills (global + project)
deepagents skills list

# List only project skills
deepagents skills list --project

# Create a new global skill from template
deepagents skills create my-skill

# Create a new project skill
deepagents skills create my-tool --project

# View detailed information about a skill
deepagents skills info web-research

# View info for a project skill only
deepagents skills info my-tool --project
```

To use skills (e.g., the langgraph-docs skill), just type a request relevant to a skill and the skill will be used automatically.

```bash
deepagents 
"create a agent.py script that implements a LangGraph agent" 
```

Skills follow Anthropic's [progressive disclosure pattern](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) - the agent knows skills exist but only reads full instructions when needed.

1. **At startup** - SkillsMiddleware scans `~/.deepagents/agent/skills/` and `.deepagents/skills/` directories
2. **Parse metadata** - Extracts YAML frontmatter (name + description) from each `SKILL.md` file
3. **Inject into prompt** - Adds skill list with descriptions to system prompt: "Available Skills: web-research - Use for web research tasks..."
4. **Progressive loading** - Agent reads full `SKILL.md` content with `read_file` only when a task matches the skill's description
5. **Execute workflow** - Agent follows the step-by-step instructions in the skill file

## Development

### Running Tests

To run the test suite:

```bash
uv sync --all-groups

make test
```

### Running During Development

```bash
# From libs/deepagents-cli directory
uv run deepagents

# Or install in editable mode
uv pip install -e .
deepagents
```

### Modifying the CLI

- **UI changes** → Edit `ui.py` or `input.py`
- **Add new tools** → Edit `tools.py`
- **Change execution flow** → Edit `execution.py`
- **Add commands** → Edit `commands.py`
- **Agent configuration** → Edit `agent.py`
- **Skills system** → Edit `skills/` modules
- **Constants/colors** → Edit `config.py`

## 📖 Documentation

- **[Main README](../../README.md)** - AG3NT framework overview
- **[API Reference](https://reference.langchain.com/python/deepagents/)** - Complete API documentation
- **[LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview)** - Underlying framework

## 🏗️ Built With

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agent orchestration framework
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM integration layer
- **[OpenRouter](https://openrouter.ai/)** - Multi-provider LLM access
- **[Rich](https://github.com/Textualize/rich)** - Terminal UI framework

## 📜 License

MIT License - see [LICENSE](../../LICENSE) for details
