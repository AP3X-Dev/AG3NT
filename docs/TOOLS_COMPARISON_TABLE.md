# DeepAgents Tools - Detailed Comparison Table

Complete comparison of all built-in tools with use cases and examples.

## Planning Tools

| Tool | Purpose | When to Use | When NOT to Use | Example |
|------|---------|-------------|-----------------|---------|
| `write_todos` | Task management | Complex multi-step tasks (3+ steps) | Simple 1-2 step tasks | Breaking down "Build a web app" into research, design, implement, test |

## Filesystem Tools

| Tool | Purpose | When to Use | When NOT to Use | Example |
|------|---------|-------------|-----------------|---------|
| `ls` | List directory contents | Exploring file structure, finding files | When you already know exact file path | `ls("/src")` to see what's in src folder |
| `read_file` | Read file contents | Reading code, configs, data files | For binary files | `read_file("/app.py", limit=100)` to scan first 100 lines |
| `write_file` | Create new file | Creating new files from scratch | When file already exists (use edit instead) | `write_file("/config.json", '{"port": 3000}')` |
| `edit_file` | Modify existing file | Changing code, fixing bugs, updates | Creating new files | `edit_file("/app.py", "old_code", "new_code")` |
| `glob` | Find files by pattern | Finding all files of a type | When you know exact path | `glob("**/*.py")` to find all Python files |
| `grep` | Search text in files | Finding where code/text appears | When you know which file to read | `grep("TODO", glob="*.py")` to find TODOs in Python files |
| `execute` | Run shell commands | Running tests, builds, scripts | For file operations (use specific tools) | `execute("pytest tests/")` to run tests |

## Utility Tools

| Tool | Purpose | When to Use | When NOT to Use | Example |
|------|---------|-------------|-----------------|---------|
| `undo_edit` | Revert last file edit | Made incorrect edit, need to try different approach | When you want to keep the edit | `undo_edit("/src/app.py")` to revert last change |
| `format_file` | Format code files | After making edits to ensure consistent style | When formatting is disabled for file | `format_file("/src/app.py")` |
| `get_diagnostics` | Get IDE errors/warnings | After edits to check for issues | When no IDE is connected | `get_diagnostics("/src")` for directory |
| `mermaid` | Render diagrams | Visualizing architecture, workflows, algorithms | For simple lists or text | `mermaid(code="graph TD; A-->B")` |

## Web Tools

| Tool | Purpose | When to Use | When NOT to Use | Example |
|------|---------|-------------|-----------------|---------|
| `web_search` | Search the web | Finding documentation, up-to-date info | For internal/private docs | `web_search(objective="Python asyncio patterns")` |
| `read_web_page` | Read web page content | Following up on search results, reading docs | For localhost URLs (use curl) | `read_web_page(url="https://docs.python.org")` |

## Advanced AI Tools

| Tool | Purpose | When to Use | When NOT to Use | Example |
|------|---------|-------------|-----------------|---------|
| `finder` | Semantic codebase search | Finding code by functionality/concept | For exact text matches (use grep) | `finder(query="JWT token validation logic")` |
| `look_at` | Extract info from media files | Images, PDFs, diagrams | For text files (use read_file) | `look_at(path="/docs/diagram.png", objective="components")` |

## Subagent Tools

| Tool | Purpose | When to Use | When NOT to Use | Example |
|------|---------|-------------|-----------------|---------|
| `task` | Spawn specialized subagent | Complex subtasks needing isolation | Simple tasks main agent can handle | Delegating "Research OAuth2" to keep main context clean |
| `librarian` | Deep codebase understanding | Architecture analysis, cross-repo relationships | Simple file reading | `task(description="Explain data flow", subagent_type="librarian")` |
| `oracle` | Expert technical guidance | Code reviews, architecture decisions, debugging | Simple code changes | `task(description="Review auth code", subagent_type="oracle")` |

## Tool Selection Guide

### For File Operations

```
Need to...                          → Use this tool
─────────────────────────────────────────────────────
See what files exist                → ls
Find files by type/pattern          → glob
Search for text across files        → grep
Read a specific file                → read_file
Create a new file                   → write_file
Modify an existing file             → edit_file
```

### For Code Exploration

```
Task                                → Recommended Workflow
─────────────────────────────────────────────────────
Explore unknown codebase            → 1. ls("/")
                                      2. glob("**/*.py")
                                      3. grep("main", glob="*.py")
                                      4. read_file(found_files)

Find where function is used         → grep("function_name", output_mode="content")

Understand file structure           → read_file(path, limit=50)  # First 50 lines

Read large file                     → read_file(path, offset=0, limit=100)
                                      read_file(path, offset=100, limit=100)
                                      ...
```

### For Task Management

```
Task Complexity                     → Approach
─────────────────────────────────────────────────────
Simple (1-2 steps)                  → Just do it, no todos needed
Medium (3-6 steps)                  → Use write_todos to track
Complex (7+ steps)                  → Use write_todos + subagents
Very Complex                        → Break into subtasks with task tool
```

## Tool Combinations (Common Patterns)

### Pattern 1: Explore → Read → Edit

```python
# 1. Explore structure
ls("/src")

# 2. Find relevant files
glob("/src/**/*.py")

# 3. Search for specific code
grep(pattern="config", glob="*.py", output_mode="content")

# 4. Read the file
read_file("/src/config.py", limit=100)

# 5. Edit the file
edit_file("/src/config.py", old_string="old", new_string="new")
```

### Pattern 2: Plan → Execute → Verify

```python
# 1. Plan the work
write_todos(todos=[
    {"content": "Write tests", "status": "in_progress"},
    {"content": "Run tests", "status": "pending"},
    {"content": "Fix failures", "status": "pending"}
])

# 2. Execute
execute("pytest tests/ -v")

# 3. Update plan based on results
write_todos(todos=[
    {"content": "Write tests", "status": "completed"},
    {"content": "Run tests", "status": "completed"},
    {"content": "Fix failures", "status": "in_progress"}
])
```

### Pattern 3: Delegate Complex Work

```python
# Main agent delegates research to subagent
task(
    description="Research best practices for OAuth2 implementation in Python",
    subagent_type="general"
)

# Subagent returns concise summary
# Main agent continues with clean context
```

## Performance Tips

### Efficient File Reading

| Scenario | Inefficient ❌ | Efficient ✅ |
|----------|---------------|-------------|
| Large file | `read_file("/big.py")` (reads all) | `read_file("/big.py", limit=100)` (paginate) |
| Find function | Read entire file | `grep("function_name", output_mode="content")` |
| Explore codebase | Read every file | `ls` + `glob` + selective `read_file` |

### Efficient Searching

| Goal | Inefficient ❌ | Efficient ✅ |
|------|---------------|-------------|
| Find Python files | `execute("find . -name '*.py'")` | `glob("**/*.py")` |
| Search for text | `execute("grep -r 'pattern' .")` | `grep(pattern="pattern")` |
| List files | `execute("ls /path")` | `ls("/path")` |

### Efficient Editing

| Scenario | Inefficient ❌ | Efficient ✅ |
|----------|---------------|-------------|
| Update file | `write_file` (overwrites) | `edit_file` (precise change) |
| Rename variable | Multiple `edit_file` calls | `edit_file(..., replace_all=True)` |
| Edit without reading | Guess the content | `read_file` first, then `edit_file` |

## Tool Limitations

| Tool | Limitation | Workaround |
|------|-----------|------------|
| `read_file` | Lines > 2000 chars truncated | Read in smaller chunks or use `grep` |
| `edit_file` | Requires unique old_string | Provide more context or use `replace_all` |
| `execute` | Output truncated at 50000 chars | Use grep/head/tail, or save to file |
| `grep` | 100 matches max (10 per file) | Narrow search with path/glob params |
| `glob` | No regex support | Use standard glob patterns with `{a,b}` alternatives |
| `undo_edit` | Only last edit can be undone | Make backup copy before complex changes |
| `format_file` | Requires IDE/backend support | Use execute with formatter CLI |
| `get_diagnostics` | Requires IDE connection | Run linter via execute |
| `web_search` | May not access all sites | Use read_web_page for direct access |
| `read_web_page` | No localhost support | Use execute with curl |
| `finder` | Keyword-based (not true semantic) | Combine with grep for precision |
| `look_at` | Limited without vision model | Use external OCR/PDF tools |

## Advanced Usage

### Conditional Tool Selection

```python
# Use glob when pattern is known
if need_to_find_by_pattern:
    glob("**/*.test.js")

# Use grep when searching for content
if need_to_find_by_content:
    grep(pattern="test", glob="*.js")

# Use ls when exploring directory
if need_to_see_structure:
    ls("/src")
```

### Combining Tools for Complex Tasks

```python
# Find all test files, read them, run tests
test_files = glob("**/*.test.py")
for file in test_files:
    read_file(file, limit=50)  # Understand what's tested
execute("pytest tests/ -v")     # Run all tests
```

## See Also

- 📖 [DEEPAGENTS_TOOLS_REFERENCE.md](DEEPAGENTS_TOOLS_REFERENCE.md) - Complete reference
- 📋 [TOOLS_QUICK_REFERENCE.md](TOOLS_QUICK_REFERENCE.md) - Quick cheat sheet
- 🏗️ [DeepAgents README](libs/deepagents/README.md) - Main documentation

