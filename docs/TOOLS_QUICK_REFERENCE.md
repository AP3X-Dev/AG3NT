# DeepAgents Tools - Quick Reference

One-page cheat sheet for all built-in tools.

## 📋 Planning (1 tool)

```python
# Create/update task list
write_todos(todos=[
    {"content": "Task 1", "status": "completed"},
    {"content": "Task 2", "status": "in_progress"},
    {"content": "Task 3", "status": "pending"}
])
```

## 📁 Filesystem (7 tools)

```python
# List files in directory
ls("/workspace")

# Read file (with pagination)
read_file("/src/app.py", offset=0, limit=100)

# Write new file (also overwrites existing)
write_file("/config.json", content='{"key": "value"}')

# Edit existing file (shows git-style diff)
edit_file(
    file_path="/src/app.py",
    old_string="old code",
    new_string="new code",
    replace_all=False
)

# Find files by pattern (sorted by modification time)
glob("**/*.py")              # All Python files
glob("/src/**/*.{js,ts}")    # JS and TS files in /src

# Search text in files (ripgrep-based)
grep(pattern="TODO")                           # Find all TODOs
grep(pattern="import", glob="*.py")            # In Python files only
grep(pattern="error", output_mode="content")   # Show matching lines

# Execute shell commands
execute("pytest /workspace/tests")
```

## 🔧 Utility Tools (4 tools)

```python
# Undo last file edit
undo_edit("/src/app.py")

# Format file using IDE formatter
format_file("/src/app.py")

# Get IDE diagnostics (errors, warnings)
get_diagnostics("/src")

# Render Mermaid diagram
mermaid(code="graph TD; A-->B; B-->C;")
```

## 🌐 Web Tools (2 tools)

```python
# Search the web
web_search(objective="Python async best practices")

# Read web page content
read_web_page(url="https://docs.python.org/3/library/asyncio.html")
read_web_page(url="...", objective="error handling patterns")  # Filter content
```

## 🧠 Advanced AI Tools (2 tools)

```python
# Semantic codebase search (by concept, not just text)
finder(query="Find authentication logic that validates JWT tokens")

# Extract info from images/PDFs
look_at(path="/docs/architecture.png", objective="Identify main components")
```

## 🔄 Subagents (3 tools)

```python
# Delegate to general-purpose subagent
task(description="Research OAuth2 implementation", subagent_type="general")

# Use librarian for deep codebase understanding
task(description="Explain the data flow architecture", subagent_type="librarian")

# Use oracle for expert technical guidance
task(description="Review this authentication code for security issues", subagent_type="oracle")
```

## ⚙️ Custom Tools (unlimited)

```python
# Add your own tools
from deepagents import create_deep_agent

def my_tool(query: str) -> str:
    """Your custom tool."""
    return results

agent = create_deep_agent(tools=[my_tool])
```

---

## Common Patterns

### Explore Codebase
```python
ls("/")                      # See top-level structure
glob("**/*.py")              # Find all Python files
grep(pattern="class", glob="*.py", output_mode="content")
```

### Read & Edit File
```python
ls("/src")                   # Find the file
read_file("/src/app.py", limit=100)  # Read first 100 lines
edit_file(
    file_path="/src/app.py",
    old_string="old_function()",
    new_string="new_function()"
)
```

### Complex Task
```python
write_todos(todos=[
    {"content": "Analyze requirements", "status": "in_progress"},
    {"content": "Design solution", "status": "pending"},
    {"content": "Implement code", "status": "pending"},
    {"content": "Write tests", "status": "pending"}
])
```

### Run Tests
```python
execute("pytest /workspace/tests -v")
execute("npm test")
execute("python -m unittest discover")
```

---

## Best Practices

✅ **DO:**
- Use `ls` before reading/editing to explore structure
- Read files before editing them
- Use pagination for large files
- Use `glob` and `grep` tools instead of shell commands
- Keep todo lists minimal (3-6 items)
- Use absolute paths (starting with `/`)

❌ **DON'T:**
- Use `cat`, `find`, `grep` shell commands (use tools instead)
- Edit files without reading them first
- Create new files when you can edit existing ones
- Use relative paths
- Over-fragment simple tasks with todos

---

## Tool Categories

| Category | Tools | Count |
|----------|-------|-------|
| Planning | `write_todos` | 1 |
| Filesystem | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute` | 7 |
| Subagents | `task` | 1 |
| Custom | User-defined | ∞ |
| **TOTAL** | **Built-in** | **9** |

---

## Quick Syntax Reference

```python
# Planning
write_todos(todos: list[dict])

# Filesystem
ls(path: str)
read_file(file_path: str, offset: int = 0, limit: int = 500)
write_file(file_path: str, content: str)
edit_file(file_path: str, old_string: str, new_string: str, replace_all: bool = False)
glob(pattern: str, path: str = "/")
grep(pattern: str, path: str | None = None, glob: str | None = None, 
     output_mode: "files_with_matches" | "content" | "count" = "files_with_matches")
execute(command: str)

# Subagents
task(description: str, subagent_type: str)
```

---

## See Full Documentation

📖 [DEEPAGENTS_TOOLS_REFERENCE.md](DEEPAGENTS_TOOLS_REFERENCE.md) - Complete reference with examples and best practices

