# DeepAgents Developer Quick Start

## Installation

```bash
cd libs/deepagents
pip install -e .
```

## Basic Agent Setup

```python
from deepagents import create_deep_agent
from deepagents.middleware import (
    FilesystemMiddleware,
    UtilitiesMiddleware,
    WebMiddleware,
    AdvancedMiddleware,
    SubAgentMiddleware,
)
from deepagents.middleware.advanced import get_librarian_subagent, get_oracle_subagent

# Full-featured agent with all tools
agent = create_deep_agent(
    model="claude-3-5-sonnet-20241022",
    middleware=[
        FilesystemMiddleware(),  # File operations
        UtilitiesMiddleware(),   # Dev utilities
        WebMiddleware(),         # Web access
        AdvancedMiddleware(),    # AI-powered tools
        SubAgentMiddleware(
            subagents=[
                get_librarian_subagent(),
                get_oracle_subagent(),
            ]
        ),
    ]
)

# Run the agent
response = agent.run("Your task here")
```

## Common Use Cases

### 1. Code Development Agent

```python
agent = create_deep_agent(
    middleware=[
        FilesystemMiddleware(),
        UtilitiesMiddleware(enabled_tools=["undo_edit", "format_file", "get_diagnostics"]),
        AdvancedMiddleware(enabled_tools=["finder"]),
    ]
)

# Agent can now:
# - Read/write/edit files
# - Undo mistakes
# - Format code
# - Find code by concept
# - Get IDE diagnostics
```

### 2. Research Agent

```python
agent = create_deep_agent(
    middleware=[
        WebMiddleware(),
        UtilitiesMiddleware(enabled_tools=["mermaid"]),
    ]
)

# Agent can now:
# - Search the web
# - Read web pages
# - Create diagrams
```

### 3. Documentation Agent

```python
agent = create_deep_agent(
    middleware=[
        FilesystemMiddleware(),
        AdvancedMiddleware(),
        UtilitiesMiddleware(enabled_tools=["mermaid"]),
        SubAgentMiddleware(subagents=[get_librarian_subagent()]),
    ]
)

# Agent can now:
# - Analyze codebase deeply
# - Generate documentation
# - Create architecture diagrams
# - Search by concept
```

## Tool Reference

### Filesystem Tools (FilesystemMiddleware)

```python
# List directory
ls(path="src/")

# Read file
read_file(path="main.py")

# Write file
write_file(path="config.json", content="...")

# Edit file
edit_file(path="app.py", old_str="...", new_str="...")

# Search files
glob(pattern="**/*.py")

# Search content
grep(pattern="TODO", include="*.py")

# Execute command
execute(command="npm test")
```

### Utilities Tools (UtilitiesMiddleware)

```python
# Undo last edit
undo_edit(path="app.py")

# Format file
format_file(path="main.py")

# Get diagnostics
get_diagnostics(paths=["src/"])

# Render diagram
mermaid(diagram_definition="graph TD; A-->B;")
```

### Web Tools (WebMiddleware)

```python
# Search web
web_search(query="Python async best practices", num_results=5)

# Read web page
read_web_page(url="https://example.com/article")
```

### Advanced Tools (AdvancedMiddleware)

```python
# Find code by concept
finder(query="authentication logic", path="src/")

# Analyze image/PDF
look_at(path="diagram.png")

# Use librarian subagent
task(
    subagent="librarian",
    task="Explain how the authentication system works"
)

# Use oracle subagent
task(
    subagent="oracle",
    task="What's the best way to implement caching?"
)
```

## Selective Tool Enabling

```python
# Enable only specific tools
agent = create_deep_agent(
    middleware=[
        FilesystemMiddleware(enabled_tools=["read_file", "write_file"]),
        UtilitiesMiddleware(enabled_tools=["undo_edit"]),
        WebMiddleware(enabled_tools=["web_search"]),
    ]
)
```

## Custom System Prompts

```python
agent = create_deep_agent(
    middleware=[
        FilesystemMiddleware(
            system_prompt="You are a Python expert. Always follow PEP 8."
        ),
    ]
)
```

## Testing

```bash
# Run all tests
python -m pytest tests/unit_tests/ -v

# Run specific middleware tests
python -m pytest tests/unit_tests/middleware/test_new_middleware.py -v
```

## Tool Availability Matrix

| Tool | Requires Backend | Requires API Key | Notes |
|------|-----------------|------------------|-------|
| All Filesystem | ✅ | ❌ | Standard backend methods |
| undo_edit | ✅ | ❌ | Uses read/write |
| format_file | ⚠️ | ❌ | Needs backend.aformat_file() |
| get_diagnostics | ⚠️ | ❌ | Needs backend.aget_diagnostics() |
| mermaid | ❌ | ❌ | Pure Python |
| web_search | ❌ | ❌ | Uses DuckDuckGo |
| read_web_page | ❌ | ❌ | Uses httpx |
| finder | ✅ | ❌ | Uses grep backend |
| look_at | ✅ | ❌ | Basic file info only |
| librarian | ✅ | ❌ | Subagent |
| oracle | ❌ | ❌ | Subagent |

Legend:
- ✅ = Required
- ⚠️ = Optional (graceful degradation)
- ❌ = Not required

## Next Steps

1. Check `DEEPAGENTS_TOOLS_REFERENCE.md` for detailed tool documentation
2. See `TOOLS_QUICK_REFERENCE.md` for tool examples
3. Review `IMPLEMENTATION_SUMMARY.md` for implementation details

