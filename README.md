<img src=".github/images/AG3NT_header.png" alt="AG3NT" width="100%"/>


# 🚀 AG3NT

**Production-grade AI agent framework for complex, long-horizon tasks.**

A sophisticated agent harness that combines:
- **Advanced middleware architecture** for extensible tool ecosystems
- **Persistent memory** that learns and evolves across sessions
- **Progressive disclosure skills system** for domain expertise
- **Intelligent subagent delegation** for parallel, specialized work
- **Context compaction** for handling massive codebases and long conversations
- **Multi-provider LLM support** via OpenRouter, Anthropic, OpenAI, and Google

<img src=".github/images/AG3NT_header.png" alt="AG3NT" width="100%"/>

## 🎯 What Makes AG3NT Different

| Feature | Basic Agents | AG3NT |
|---------|-------------|-------|
| **Tool System** | Fixed set of tools | Extensible middleware with 19+ built-in tools |
| **Memory** | Stateless or simple RAG | Persistent, structured memory with auto-learning |
| **Skills** | Hardcoded prompts | Progressive disclosure skill library |
| **Delegation** | Single agent only | Specialized subagents (Librarian, Oracle, custom) |
| **Context Management** | Token limits crash | Automatic compaction and artifact storage |
| **Code Understanding** | Basic file reading | Semantic search + AI-powered codebase analysis |
| **Research** | Single web search | Multi-step deep research with synthesis |
| **Execution** | Local only | Local + remote sandboxes (Modal, Runloop, Daytona) |

## 📚 Resources

- **[CLI Documentation](libs/deepagents-cli/)** - Interactive terminal interface with full feature set
- **[API Reference](https://reference.langchain.com/python/deepagents/)** - Complete API documentation

## 🚀 Quick Start

### Installation

```bash
pip install deepagents

# Or with uv (recommended)
uv add deepagents
```

### Basic Agent (OpenRouter Recommended)

Set up your `.env` file:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=anthropic/claude-sonnet-4.5  # or any OpenRouter model
```

Create an agent with full capabilities:

```python
from deepagents import create_deep_agent
from deepagents.middleware import (
    UtilitiesMiddleware,
    WebMiddleware,
    AdvancedMiddleware,
)

# Full-featured agent with extended tools
agent = create_deep_agent(
    middleware=[
        UtilitiesMiddleware(),  # undo_edit, format_file, get_diagnostics, mermaid
        WebMiddleware(),        # web_search, read_web_page
        AdvancedMiddleware(),   # finder, look_at, librarian, oracle
    ],
    system_prompt="You are an expert software engineer and researcher.",
)

# Use it
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this codebase and suggest improvements"}]
})
```

### CLI Mode (Recommended for Interactive Use)

```bash
# Install CLI
pip install deepagents-cli

# Run interactively
ag3nt

# With specific model
ag3nt --model anthropic/claude-sonnet-4.5

# With auto-approve for automation
ag3nt --auto-approve
```

The CLI provides:
- **Human-in-the-loop approvals** for sensitive operations
- **Persistent memory** across sessions
- **Skills library** for domain expertise
- **Session management** with resume capability

## 🧠 Advanced Capabilities

### 1. OpenRouter Integration (Recommended)

Access **any LLM** through a single API with automatic failover and cost optimization:

```python
from deepagents import create_deep_agent, get_openrouter_model

# Use cutting-edge models
model = get_openrouter_model("anthropic/claude-sonnet-4.5")
# Or: "openai/gpt-5.2", "google/gemini-3-pro", "x-ai/grok-4.1-fast"

agent = create_deep_agent(model=model)
```

**Why OpenRouter?**
- Access to 100+ models from multiple providers
- Automatic fallback if a model is unavailable
- Cost tracking and optimization
- No need to manage multiple API keys

### 2. Extended Middleware System

AG3NT's power comes from its middleware architecture. Each middleware adds specialized capabilities:

```python
from deepagents import create_deep_agent
from deepagents.middleware import (
    UtilitiesMiddleware,    # Dev tools
    WebMiddleware,          # Web access
    AdvancedMiddleware,     # AI-powered analysis
    MemoryMiddleware,       # Persistent learning
    SkillsMiddleware,       # Domain expertise
)

agent = create_deep_agent(
    middleware=[
        UtilitiesMiddleware(),  # undo_edit, format_file, get_diagnostics, mermaid
        WebMiddleware(),        # web_search, read_web_page
        AdvancedMiddleware(),   # finder, look_at, librarian, oracle
        MemoryMiddleware(       # Persistent memory across sessions
            sources=["/memories/preferences.md", "/memories/patterns.md"]
        ),
        SkillsMiddleware(       # Progressive disclosure skills
            sources=["/skills/user/", "/skills/project/"]
        ),
    ]
)
```

### 3. Intelligent Subagent Delegation

Spawn specialized subagents for complex, isolated tasks:

```python
from deepagents import create_deep_agent
from deepagents.middleware import SubAgentMiddleware
from deepagents.middleware.advanced import get_librarian_subagent, get_oracle_subagent

agent = create_deep_agent(
    middleware=[
        SubAgentMiddleware(
            subagents=[
                get_librarian_subagent(),  # Deep codebase understanding
                get_oracle_subagent(),     # Expert technical guidance
                # Add custom subagents...
            ]
        ),
    ]
)

# The agent can now delegate:
# - Code analysis to Librarian
# - Architecture questions to Oracle
# - Research tasks to custom research subagent
```

**When to use subagents:**
- Parallel execution of independent tasks
- Isolating context for specialized work
- Preventing context pollution in main thread
- Applying domain expertise (research, security audit, code review)

### 4. Persistent Memory System

AG3NT learns and remembers across sessions:

```python
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),  # Ephemeral working files
        routes={
            "/memories/": StoreBackend(store=InMemoryStore()),  # Persistent
        },
    ),
)

# Agent can now:
# - Remember your coding preferences
# - Learn project patterns
# - Build knowledge bases across conversations
# - Self-improve based on feedback
```

### 5. Progressive Disclosure Skills System

Skills are reusable instruction modules that provide domain expertise:

```python
from deepagents.middleware import SkillsMiddleware

agent = create_deep_agent(
    middleware=[
        SkillsMiddleware(
            sources=[
                "/skills/base/",      # Built-in skills
                "/skills/user/",      # Your custom skills
                "/skills/project/",   # Project-specific skills
            ]
        ),
    ]
)

# Skills can be:
# 1. Applied as prompt modules (lightweight)
# 2. Spawned as specialized subagents (isolated context)
```

**Example skills:**
- `web-research` - Multi-step research with synthesis
- `security-audit` - Code security analysis
- `api-design` - RESTful API design patterns
- `langgraph-docs` - Framework-specific guidance

**Create custom skills:**
```yaml
---
id: my-skill
name: Custom Skill
description: Domain-specific expertise
mode: both  # prompt, subagent, or both
tools: ["web_search", "read_file"]
---

# Purpose
Detailed skill instructions...
```

### 6. Context Compaction & Artifact Storage

Handle massive codebases and long conversations without hitting token limits:

```python
from deepagents.middleware import CompactionMiddleware

agent = create_deep_agent(
    middleware=[
        CompactionMiddleware(),  # Automatic context management
    ]
)

# Agent automatically:
# - Stores large outputs as artifacts
# - Compacts conversation history
# - Retrieves details on-demand
# - Maintains working set in active context
```

### 7. Remote Sandbox Execution

Execute code in isolated, secure environments:

```python
from deepagents_cli.backends.modal import ModalBackend

agent = create_deep_agent(
    sandbox=ModalBackend(),  # or RunloopBackend(), DaytonaBackend()
)

# Agent can now:
# - Run untrusted code safely
# - Execute in clean environments
# - Scale compute on-demand
# - Avoid polluting local system
```

### 8. Custom Tools & MCP Integration

Add your own tools or connect to MCP servers:

```python
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

def custom_tool(query: str) -> str:
    """Your custom tool"""
    return "result"

async def main():
    # MCP tools (e.g., Playwright browser automation)
    mcp_client = MultiServerMCPClient(...)
    mcp_tools = await mcp_client.get_tools()

    agent = create_deep_agent(
        tools=[custom_tool] + mcp_tools
    )
```

### `interrupt_on`

Some tools may be sensitive and require human approval before execution. Deepagents supports human-in-the-loop workflows through LangGraph’s interrupt capabilities. You can configure which tools require approval using a checkpointer.

These tool configs are passed to our prebuilt [HITL middleware](https://docs.langchain.com/oss/python/langchain/middleware#human-in-the-loop) so that the agent pauses execution and waits for feedback from the user before executing configured tools.

```python
from langchain_core.tools import tool
from deepagents import create_deep_agent

@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."

agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[get_weather],
    interrupt_on={
        "get_weather": {
            "allowed_decisions": ["approve", "edit", "reject"]
        },
    }
)
```

See the [human-in-the-loop documentation](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop) for more details.

### `backend`

Deep agents use pluggable backends to control how filesystem operations work. By default, files are stored in the agent's ephemeral state. You can configure different backends for local disk access, persistent cross-conversation storage, or hybrid routing.

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

agent = create_deep_agent(
    backend=FilesystemBackend(root_dir="/path/to/project"),
)
```

Available backends include:

- **`StateBackend`** (default): Ephemeral files stored in agent state
- **`FilesystemBackend`**: Real disk operations under a root directory
- **`StoreBackend`**: Persistent storage using LangGraph Store
- **`CompositeBackend`**: Route different paths to different backends

See the [backends documentation](https://docs.langchain.com/oss/python/deepagents/backends) for more details.

### Long-term Memory

Deep agents can maintain persistent memory across conversations using a `CompositeBackend` that routes specific paths to durable storage.

This enables hybrid memory where working files remain ephemeral while important data (like user preferences or knowledge bases) persists across threads.

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

agent = create_deep_agent(
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(store=InMemoryStore())},
    ),
)
```

Files under `/memories/` will persist across all conversations, while other paths remain temporary. Use cases include:

- Preserving user preferences across sessions
- Building knowledge bases from multiple conversations
- Self-improving instructions based on feedback
- Maintaining research progress across sessions

See the [long-term memory documentation](https://docs.langchain.com/oss/python/deepagents/long-term-memory) for more details.

## 🛠️ Complete Tool Ecosystem

AG3NT provides **19+ built-in tools** across multiple categories:

### Core Tools (Always Available)

| Tool | Description | Category |
|------|-------------|----------|
| `ls` | List directory contents | Filesystem |
| `read_file` | Read file contents | Filesystem |
| `write_file` | Create/overwrite files | Filesystem |
| `edit_file` | Targeted file edits | Filesystem |
| `glob` | Pattern-based file search | Filesystem |
| `grep` | Text search across files | Filesystem |
| `shell` | Execute shell commands (local) | Execution |
| `execute` | Execute in sandbox (remote) | Execution |
| `write_todos` | Task planning & tracking | Planning |
| `task` | Delegate to subagents | Delegation |

### Extended Tools (Via Middleware)

| Tool | Description | Middleware |
|------|-------------|------------|
| `undo_edit` | Revert file changes | UtilitiesMiddleware |
| `format_file` | Auto-format code | UtilitiesMiddleware |
| `get_diagnostics` | IDE error checking | UtilitiesMiddleware |
| `mermaid` | Generate diagrams | UtilitiesMiddleware |
| `web_search` | Search the web | WebMiddleware |
| `read_web_page` | Fetch & parse URLs | WebMiddleware |
| `finder` | Semantic code search | AdvancedMiddleware |
| `look_at` | AI-powered code analysis | AdvancedMiddleware |
| `librarian` | Deep codebase understanding | AdvancedMiddleware |
| `oracle` | Expert technical guidance | AdvancedMiddleware |

### Browser Automation (MCP)

When Playwright MCP is configured:
- Full browser control (navigate, click, type, screenshot)
- Form filling and interaction
- Network request inspection
- Console log monitoring

## 🎨 Middleware Architecture

AG3NT's power comes from its composable middleware system:

### Core Middleware (Included by Default)

| Middleware | Purpose |
|------------|---------|
| **`TodoListMiddleware`** | Task planning and progress tracking |
| **`FilesystemMiddleware`** | File operations and context offloading |
| **`SubAgentMiddleware`** | Delegate tasks to isolated sub-agents |
| **`SummarizationMiddleware`** | Auto-summarizes when context exceeds 170k tokens |
| **`AnthropicPromptCachingMiddleware`** | Caches system prompts to reduce costs |
| **`HumanInTheLoopMiddleware`** | Pauses execution for human approval |

### Extended Middleware (Optional)

| Middleware | Purpose |
|------------|---------|
| **`UtilitiesMiddleware`** | Development helper tools |
| **`WebMiddleware`** | Web content access |
| **`AdvancedMiddleware`** | AI-powered analysis tools |
| **`MemoryMiddleware`** | Persistent learning across sessions |
| **`SkillsMiddleware`** | Progressive disclosure skills system |
| **`CompactionMiddleware`** | Context window management |

**Enable extended middleware:**

```python
from deepagents import create_deep_agent
from deepagents.middleware import (
    UtilitiesMiddleware,
    WebMiddleware,
    AdvancedMiddleware,
    MemoryMiddleware,
    SkillsMiddleware,
)

agent = create_deep_agent(
    middleware=[
        UtilitiesMiddleware(),
        WebMiddleware(),
        AdvancedMiddleware(),
        MemoryMiddleware(sources=["/memories/"]),
        SkillsMiddleware(sources=["/skills/"]),
    ]
)
```

## 💡 Real-World Use Cases

### Software Engineering
```python
# Full-stack development agent with all capabilities
agent = create_deep_agent(
    middleware=[
        UtilitiesMiddleware(),  # Code formatting, diagnostics
        WebMiddleware(),        # Documentation lookup
        AdvancedMiddleware(),   # Codebase analysis
        MemoryMiddleware(sources=["/memories/coding-style.md"]),
    ],
    system_prompt="You are an expert full-stack engineer."
)
```

### Research & Analysis
```python
# Deep research agent with web access and synthesis
agent = create_deep_agent(
    middleware=[
        WebMiddleware(),        # Web search and scraping
        SkillsMiddleware(sources=["/skills/research/"]),
    ],
    system_prompt="Conduct thorough research and synthesize findings."
)
```

### DevOps & Automation
```python
# Infrastructure agent with sandbox execution
from deepagents_cli.backends.modal import ModalBackend

agent = create_deep_agent(
    sandbox=ModalBackend(),
    middleware=[WebMiddleware()],
    system_prompt="Automate infrastructure tasks safely."
)
```

## 🔒 Security & Safety

AG3NT follows a **"trust but verify"** model:

- **Human-in-the-Loop**: Sensitive operations require approval by default
- **Sandbox Execution**: Run untrusted code in isolated environments
- **Audit Logging**: All tool calls are logged for review
- **Configurable Permissions**: Fine-grained control over tool access

```python
# Configure HITL for specific tools
agent = create_deep_agent(
    interrupt_on={
        "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        "shell": {"allowed_decisions": ["approve", "reject"]},
    }
)
```

## 🚀 Getting Started

1. **Install AG3NT**
   ```bash
   pip install deepagents deepagents-cli
   ```

2. **Set up OpenRouter** (recommended)
   ```bash
   export OPENROUTER_API_KEY=sk-or-v1-your-key
   ```

3. **Run the CLI**
   ```bash
   ag3nt
   ```

4. **Or use programmatically**
   ```python
   from deepagents import create_deep_agent

   agent = create_deep_agent()
   result = agent.invoke({"messages": [{"role": "user", "content": "Your task"}]})
   ```

## 📖 Documentation

- **[CLI Guide](libs/deepagents-cli/)** - Complete CLI documentation
- **[API Reference](https://reference.langchain.com/python/deepagents/)** - Full API docs
- **[LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/overview)** - Underlying framework

## 🏗️ Built With

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agent orchestration framework
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM integration layer
- **[OpenRouter](https://openrouter.ai/)** - Multi-provider LLM access

## 📜 License

MIT License - see [LICENSE](LICENSE) for details
