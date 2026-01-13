# DeepAgents Tools Documentation - Index

Complete guide to all documentation about DeepAgents tools.

## 📚 Documentation Overview

We've created comprehensive documentation for all DeepAgents tools across multiple formats:

### 1. **Complete Reference** 📖
**File:** [DEEPAGENTS_TOOLS_REFERENCE.md](DEEPAGENTS_TOOLS_REFERENCE.md)

**What's Inside:**
- Detailed description of all 19 built-in tools
- Parameters and return types
- Code examples for each tool
- Best practices and common pitfalls
- When to use (and when NOT to use) each tool
- Middleware architecture diagram and explanation

**Best For:** Learning about tools in depth, understanding tool capabilities

---

### 2. **Quick Reference** 📋
**File:** [TOOLS_QUICK_REFERENCE.md](TOOLS_QUICK_REFERENCE.md)

**What's Inside:**
- One-page cheat sheet
- Quick syntax reference
- Common patterns and workflows
- Do's and don'ts
- Tool categories summary

**Best For:** Quick lookups, copy-paste examples, daily reference

---

### 3. **Comparison Table** 📊
**File:** [TOOLS_COMPARISON_TABLE.md](TOOLS_COMPARISON_TABLE.md)

**What's Inside:**
- Side-by-side tool comparisons
- Tool selection guide
- Performance tips
- Common patterns and combinations
- Limitations and workarounds
- Advanced usage examples

**Best For:** Choosing the right tool, optimizing tool usage, understanding trade-offs

---

## 🎯 Quick Navigation

### By Use Case

**I want to...**

- **Learn all available tools** → [DEEPAGENTS_TOOLS_REFERENCE.md](DEEPAGENTS_TOOLS_REFERENCE.md)
- **Get quick syntax** → [TOOLS_QUICK_REFERENCE.md](TOOLS_QUICK_REFERENCE.md)
- **Choose the right tool** → [TOOLS_COMPARISON_TABLE.md](TOOLS_COMPARISON_TABLE.md)
- **See visual architecture** → [DEEPAGENTS_TOOLS_REFERENCE.md](DEEPAGENTS_TOOLS_REFERENCE.md) (includes diagram)

### By Tool Category

**Planning Tools (1):**
- `write_todos` - Task management and planning

**Filesystem Tools (7):**
- `ls` - List files
- `read_file` - Read file contents
- `write_file` - Create new files
- `edit_file` - Modify existing files
- `glob` - Find files by pattern
- `grep` - Search text in files
- `execute` - Run shell commands

**Utility Tools (4):**
- `undo_edit` - Revert last file edit
- `format_file` - Format code using IDE
- `get_diagnostics` - Get IDE errors/warnings
- `mermaid` - Render diagrams

**Web Tools (2):**
- `web_search` - Search the web
- `read_web_page` - Fetch web content

**Advanced AI Tools (2):**
- `finder` - Semantic codebase search
- `look_at` - Extract info from images/PDFs

**Subagent Tools (3):**
- `task` - Spawn specialized subagents
- `librarian` - Deep codebase understanding
- `oracle` - Expert technical guidance

**Custom Tools:**
- Add your own unlimited tools

---

## 📖 Tool Count Summary

| Category | Tools | Description |
|----------|-------|-------------|
| **Planning** | 1 | Task breakdown and tracking |
| **Filesystem** | 7 | File operations and execution |
| **Subagents** | 1 | Delegate to specialized agents |
| **Custom** | ∞ | User-defined tools |
| **TOTAL Built-in** | **9** | Ready to use out of the box |

---

## 🚀 Getting Started

### For Beginners

1. Start with [TOOLS_QUICK_REFERENCE.md](TOOLS_QUICK_REFERENCE.md) to see what's available
2. Read [DEEPAGENTS_TOOLS_REFERENCE.md](DEEPAGENTS_TOOLS_REFERENCE.md) for detailed understanding
3. Use [TOOLS_COMPARISON_TABLE.md](TOOLS_COMPARISON_TABLE.md) when choosing tools

### For Experienced Users

1. Keep [TOOLS_QUICK_REFERENCE.md](TOOLS_QUICK_REFERENCE.md) open for syntax
2. Refer to [TOOLS_COMPARISON_TABLE.md](TOOLS_COMPARISON_TABLE.md) for optimization
3. Check [DEEPAGENTS_TOOLS_REFERENCE.md](DEEPAGENTS_TOOLS_REFERENCE.md) for edge cases

---

## 💡 Common Workflows

### Explore Codebase
```
1. ls("/") → See structure
2. glob("**/*.py") → Find Python files
3. grep("class", glob="*.py") → Find classes
4. read_file(path, limit=100) → Read relevant files
```

### Implement Feature
```
1. write_todos([...]) → Plan the work
2. read_file(...) → Understand existing code
3. edit_file(...) → Make changes
4. execute("pytest") → Test changes
5. Update todos → Track progress
```

### Debug Issue
```
1. grep("error", output_mode="content") → Find error locations
2. read_file(...) → Read problematic code
3. edit_file(...) → Fix the issue
4. execute("pytest") → Verify fix
```

---

## 🔗 Related Documentation

- [DeepAgents README](libs/deepagents/README.md) - Main project documentation
- [OpenRouter Integration](OPENROUTER_INTEGRATION.md) - Using OpenRouter with DeepAgents
- [Examples](examples/) - Example agents and use cases
- [LangChain Middleware Docs](https://docs.langchain.com/oss/python/deepagents/middleware) - Official middleware documentation

---

## 📝 Document Maintenance

**Last Updated:** January 2026

**Documents:**
- ✅ DEEPAGENTS_TOOLS_REFERENCE.md - Complete reference
- ✅ TOOLS_QUICK_REFERENCE.md - Quick cheat sheet
- ✅ TOOLS_COMPARISON_TABLE.md - Comparison and selection guide
- ✅ TOOLS_DOCUMENTATION_INDEX.md - This index
- ✅ DEVELOPER_QUICK_START.md - Developer quick start guide
- ✅ IMPLEMENTATION_SUMMARY.md - Implementation details

**Coverage:**
- ✅ All 19 built-in tools documented
- ✅ Examples for each tool
- ✅ Best practices included
- ✅ Common patterns covered
- ✅ Visual architecture diagrams included
- ✅ Middleware architecture explained

---

## 🤝 Contributing

Found an issue or want to improve the documentation?

1. Check existing documentation for accuracy
2. Add examples for common use cases
3. Update best practices based on experience
4. Keep examples simple and clear

---

## 📞 Support

- **Issues:** Check the comparison table for tool selection help
- **Examples:** See the quick reference for code snippets
- **Deep Dive:** Read the complete reference for full details
- **Community:** LangChain Discord and GitHub discussions

---

**Happy Building with DeepAgents! 🚀**

