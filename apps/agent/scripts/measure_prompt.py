#!/usr/bin/env python3
"""Measure the system prompt token count and structure.

Usage:
    cd apps/agent
    .venv/Scripts/python.exe scripts/measure_prompt.py

Outputs a breakdown of the system prompt blocks:
- Base prompt (from deepagents base_prompt.md)
- Identity block (AGENTS.md / identity files)
- Skills manifest
- Environment block
- Memory recall (typical size)
- Context budget
- Summary guardrail
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the agent source to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _count_chars(text: str) -> int:
    return len(text)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def _load_base_prompt() -> str:
    """Load the base prompt from deepagents."""
    base_paths = [
        Path(__file__).resolve().parent.parent.parent.parent
        / "vendor" / "deepagents" / "libs" / "deepagents" / "deepagents" / "base_prompt.md",
    ]
    for p in base_paths:
        if p.exists():
            return p.read_text(encoding="utf-8")
    return ""


def _load_identity() -> str:
    """Load identity files from ~/.ag3nt/."""
    identity_dir = Path.home() / ".ag3nt"
    parts = []
    for name in ["AGENTS.md", "MEMORY.md"]:
        path = identity_dir / name
        if path.exists():
            parts.append(f"--- {name} ---\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def _build_tool_list() -> str:
    """List all registered tools."""
    try:
        from ag3nt_agent.tool_registry import TOOL_REGISTRY
        lines = [f"  {i+1}. {name} ({module}.{func})"
                 for i, (name, module, func) in enumerate(TOOL_REGISTRY)]
        return f"Registered tools ({len(TOOL_REGISTRY)}):\n" + "\n".join(lines)
    except Exception as e:
        return f"Could not load tool registry: {e}"


def _environment_block() -> str:
    import platform
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"## Environment\nPlatform: {platform.system()}\nCurrent time: {now}"


def _summary_guardrail() -> str:
    return (
        "## Context Handling\n"
        "If a conversation summary appears in the message history, "
        "use it silently for context continuity. Never repeat, describe, "
        "or reference the summary content in your response."
    )


def main():
    print("=" * 60)
    print("AG3NT System Prompt Measurement")
    print("=" * 60)

    blocks = {}

    # Base prompt
    base = _load_base_prompt()
    blocks["Base prompt (base_prompt.md)"] = base

    # Identity
    identity = _load_identity()
    blocks["Identity (AGENTS.md + MEMORY.md)"] = identity

    # Environment
    env = _environment_block()
    blocks["Environment block"] = env

    # Summary guardrail
    guardrail = _summary_guardrail()
    blocks["Summary guardrail"] = guardrail

    # Tool list
    tool_list = _build_tool_list()
    blocks["Tool registry summary"] = tool_list

    # Print breakdown
    total_chars = 0
    total_tokens = 0
    print()
    for name, content in blocks.items():
        chars = _count_chars(content)
        tokens = _estimate_tokens(content)
        total_chars += chars
        total_tokens += tokens
        print(f"  {name}:")
        print(f"    {chars:,} chars / ~{tokens:,} tokens")
        if chars > 0:
            lines = content.count("\n") + 1
            print(f"    {lines} lines")
        print()

    # Typical memory recall (estimate)
    memory_estimate_chars = 2000
    memory_estimate_tokens = 500
    total_chars += memory_estimate_chars
    total_tokens += memory_estimate_tokens
    print(f"  Memory recall (estimated typical):")
    print(f"    ~{memory_estimate_chars:,} chars / ~{memory_estimate_tokens:,} tokens")
    print()

    # Skills manifest (estimate based on typical skill count)
    skills_estimate_chars = 1500
    skills_estimate_tokens = 375
    total_chars += skills_estimate_chars
    total_tokens += skills_estimate_tokens
    print(f"  Skills manifest (estimated typical):")
    print(f"    ~{skills_estimate_chars:,} chars / ~{skills_estimate_tokens:,} tokens")
    print()

    print("-" * 60)
    print(f"  TOTAL: ~{total_chars:,} chars / ~{total_tokens:,} tokens")
    print()

    # Caching analysis
    print("Prompt Caching Analysis:")
    cached_chars = _count_chars(base) + _count_chars(identity)
    cached_tokens = _estimate_tokens(base) + _estimate_tokens(identity) + skills_estimate_tokens
    volatile_tokens = total_tokens - cached_tokens
    print(f"  Cached (identity + skills):     ~{cached_tokens:,} tokens")
    print(f"  Volatile (memory + env + etc):   ~{volatile_tokens:,} tokens")
    if total_tokens > 0:
        cache_pct = cached_tokens / total_tokens * 100
        print(f"  Cache hit ratio (estimated):     {cache_pct:.0f}%")
    print()

    # Tool descriptions audit
    print("Tool Registry:")
    print(tool_list)


if __name__ == "__main__":
    main()
