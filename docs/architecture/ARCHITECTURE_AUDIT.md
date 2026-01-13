# AG3NT Architecture Audit - Phase 0

## Executive Summary

This audit documents the current state of the AG3NT/DeepAgents architecture, identifying overlap, gaps, and optimization opportunities per the FRP requirements.

**Key Insight**: The middleware stack is the **AG3NT product**, not CLI-specific. CLI is pure UX (TUI, approval rendering, session display). The standard stack should be usable by CLI, API, eval harness, or embedded callers.

---

## 1. End-to-End Control Flow (Target)

```
User Message → UI Layer (CLI/API/Embedded) → AG3NT Agent Factory
  ↓
AG3NT Standard Middleware Stack wraps model call, augments prompt + tool registry
  ↓
LLM returns AIMessage with tool_calls
  ↓
If tool requires approval → UI layer interrupts → User approves/rejects/edits
  ↓
Approved tools execute via Backend routing
  ↓
Tool results return to agent
  ↓
Middleware post-processes (compaction masks, summarization if needed)
  ↓
Agent returns final response to UI Layer → User
```

---

## 2. Middleware Inventory

### AG3NT Standard Middleware Stack (Baseline for ALL Agents)

| Order | Middleware | Prompt Injection | Tools Registered | Post-Processing |
|-------|-----------|------------------|------------------|-----------------|
| 1 | MemoryMiddleware | ✅ AGENTS.md content | - | - |
| 2 | SkillsMiddleware | ✅ Skills metadata | list_skills, apply_skill, spawn_skill_agent | - |
| 3 | ImageGenerationMiddleware | - | generate_image | - |
| 4 | WebMiddleware | ✅ Web guidance | web_search, read_web_page | - |
| 5 | UtilitiesMiddleware | ✅ Utils guidance | undo_edit, mermaid, format_file, get_diagnostics | - |
| 6 | CompactionMiddleware | ✅ Artifact guidance | save_artifact, read_artifact, search_artifacts, retrieve_snippets | ✅ Masks large outputs |

### Environment-Conditional Middleware

| Middleware | Condition | Tools | Notes |
|-----------|-----------|-------|-------|
| ShellMiddleware | Local backend only | shell | NOT "CLI only" - tied to backend type |
| execute tool | Sandbox backend only | execute | Provided by FilesystemMiddleware when backend supports it |

### Core Middleware (from `create_deep_agent()`)

| Order | Middleware | Prompt Injection | Tools Registered | Post-Processing |
|-------|-----------|------------------|------------------|-----------------|
| 1 | TodoListMiddleware | ✅ Todo guidance | write_todos, read_todos | - |
| 2 | FilesystemMiddleware | ✅ Filesystem guidance | ls, read_file, write_file, edit_file, glob, grep, execute | - |
| 3 | SubAgentMiddleware | ✅ Task guidance | task | - |
| 4 | SummarizationMiddleware | - | - | ✅ Auto-summarizes on threshold |
| 5 | PromptCachingMiddleware | - | - | - |
| 6 | PatchToolCallsMiddleware | - | - | ✅ Patches tool call formats |
| 7 | MCPMiddleware (optional) | ✅ MCP tool docs | MCP server tools | - |
| 8 | HumanInTheLoopMiddleware | - | - | ✅ Interrupts for approval |

---

## 3. Identified Problems

### Problem 1: Middleware Overlap and Unclear Ownership

**Evidence:**
- Memory, Skills, Web, Utilities, Compaction, TodoList, Filesystem, SubAgent ALL inject prompt content
- No contract defines ordering or budget limits per slice
- Total prompt injection is unbounded and cumulative

**Impact:**
- Duplicated instructions across slices
- Token bloat from uncoordinated growth
- Inconsistent behavior in local vs sandbox mode

### Problem 2: Approval Gating Inconsistency

**Evidence:**
- `_add_interrupt_on()` in CLI defines: shell, execute, write_file, edit_file, web_search, fetch_url, task
- `deep_research` tool internally calls `web_search` and `read_web_page` - NOT gated separately
- Subagents inherit `default_interrupt_on` but may bypass if not properly passed

**Impact:**
- Research can run web operations without per-call approval
- Subagent tool calls may not trigger approval prompts

### Problem 3: Backend Routing and Workspace Inconsistency

**Evidence:**
- CLI creates `compaction_workspace` at `~/.deepagents/{agent}/compaction/`
- Research sessions write to `workspace_dir / "research_sessions"`
- Skills at `~/.deepagents/{agent}/skills/`
- Memory at `~/.deepagents/{agent}/AGENTS.md`
- No unified workspace layout constant or router

**Impact:**
- Artifact paths are scattered
- No single artifact writer service
- Sandbox mode may have different layout

### Problem 4: Tool Output Handling Inconsistency

**Evidence:**
- CompactionMiddleware masks outputs > 8KB (configurable)
- But masking only applies to CompactionMiddleware's `wrap_tool_call`
- Web, Shell, Research outputs may bypass if not processed by compaction
- No uniform normalization to (summary + artifact pointer)

**Impact:**
- Large outputs can still land in context
- Inconsistent handling across tool types

### Problem 5: Subagent Output Discipline

**Evidence:**
- SubAgentMiddleware creates subagent via `create_agent()` with middleware
- Subagent runs to completion, full transcript processed
- No explicit distillation contract - relies on summarization

**Impact:**
- Subagent can inflate token usage
- No guarantee of distilled output format

### Problem 6: CLI Owns Capability Decisions (Architecture Violation)

**Evidence:**
- `create_cli_agent()` in deepagents_cli builds the AG3NT middleware stack
- Memory, Skills, Web, Utilities, Compaction middleware are defined in CLI layer
- No agent factory in core deepagents lib for standard AG3NT stack
- API, eval harness, embedded callers cannot easily get same capabilities

**Impact:**
- AG3NT product capabilities tied to CLI implementation
- Code duplication if other callers want same stack
- Unclear what "AG3NT agent" means without CLI

**Solution:**
- Create `agent_factory.py` in deepagents lib with canonical AG3NT stack
- CLI becomes thin wrapper: select backend → call factory → render UI
- Environment-conditional middleware tied to backend type, not caller

---

## 4. Baseline Metrics (Estimated)

| Metric | Current Estimate | Target |
|--------|-----------------|--------|
| System prompt tokens (full stack) | ~4,000-8,000 | <3,000 |
| Middleware count (CLI) | 7 + 8 = 15 | <10 |
| Prompt injection points | 10+ | 3-4 |
| Tool count (full stack) | ~25-30 | ~20 |
| Approval-gated tools | 7 | All destructive |

---

## 5. Recommendations Summary

### R1: Unified Prompt Budget System
- Define token budget per middleware slice
- Implement prompt assembler with priority ordering
- Add runtime budget enforcement

### R2: Approval Gating Audit
- Ensure all destructive operations are gated
- Add approval for research sub-operations
- Document approval inheritance for subagents

### R3: Workspace Layout Standardization
- Define canonical workspace layout
- Single artifact writer service
- Consistent paths for local and sandbox modes

### R4: Tool Output Normalization
- All tools return (summary, artifact_pointer) tuple
- Compaction middleware as universal post-processor
- Enforce output size limits at tool level

### R5: Subagent Output Contract
- Define distillation interface
- Enforce output format for subagent returns
- Add token budget for subagent runs

---

## 6. Files Audited

| File | Purpose |
|------|---------|
| `libs/deepagents-cli/deepagents_cli/agent.py` | CLI agent creation, middleware stack |
| `libs/deepagents/deepagents/graph.py` | Core agent creation, middleware ordering |
| `libs/deepagents/deepagents/middleware/memory.py` | Memory/AGENTS.md loading |
| `libs/deepagents/deepagents/middleware/skills.py` | Skills loading and tools |
| `libs/deepagents/deepagents/compaction/middleware.py` | Compaction, artifact tools |
| `libs/deepagents/deepagents/middleware/subagents.py` | Subagent middleware |
| `libs/deepagents/deepagents/research/orchestrator.py` | Research orchestration |

---

## 7. Next Steps

1. **Phase 1**: Define canonical middleware ordering and prompt budget
2. **Phase 2**: Implement unified artifact writer
3. **Phase 3**: Standardize tool output format
4. **Phase 4**: Add approval gating for all destructive operations
5. **Phase 5**: Define subagent output contract

---

*Audit completed: Phase 0 of FRP implementation*

