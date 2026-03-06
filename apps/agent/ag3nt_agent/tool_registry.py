"""Declarative tool registry for AG3NT.

Replaces repetitive try/except ImportError blocks with a single
data-driven loader.

Usage:
    from ag3nt_agent.tool_registry import TOOL_REGISTRY, load_tools

    tools = load_tools(TOOL_REGISTRY)
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger("ag3nt.tools")

# Each entry: (friendly_name, module_path, callable_name)
TOOL_REGISTRY: list[tuple[str, str, str]] = [
    ("memory_search", "ag3nt_agent.memory_search", "get_memory_search_tool"),
    ("memory_store", "ag3nt_agent.memory_search", "get_memory_store_tool"),
    ("memory_summarizer", "ag3nt_agent.memory_summarizer", "get_summarize_memory_tool"),
    ("node_action", "ag3nt_agent.node_action_tool", "get_node_action_tool"),
    ("skill_executor", "ag3nt_agent.skill_executor", "run_skill"),
    ("browser", "ag3nt_agent.browser_tool", "get_browser_tools"),
    ("deep_reasoning", "ag3nt_agent.deep_reasoning", "get_deep_reasoning_tool"),
    ("glob", "ag3nt_agent.glob_tool", "get_glob_tool"),
    ("grep", "ag3nt_agent.grep_tool", "get_grep_tool"),
    ("notebook", "ag3nt_agent.notebook_tool", "get_notebook_tool"),
    ("pdf", "ag3nt_agent.pdf_tool", "get_pdf_tool"),
    ("codebase_search", "ag3nt_agent.codebase_search", "get_codebase_search_tool"),
    ("exec_command", "ag3nt_agent.exec_tool", "get_exec_tool"),
    ("process", "ag3nt_agent.process_tool", "get_process_tool"),
    ("apply_patch", "ag3nt_agent.apply_patch_tool", "get_apply_patch_tool"),
    ("git", "ag3nt_agent.git_tool", "get_git_tools"),
    ("planning", "ag3nt_agent.planning_tools", "get_planning_tools"),
    ("plan_lifecycle", "ag3nt_agent.plan_tools", "get_plan_tools"),
    ("context_blueprint", "ag3nt_agent.context_blueprint", "get_blueprint_tools"),
    ("sessions", "ag3nt_agent.session_tools", "get_session_tools"),
    ("lsp", "ag3nt_agent.lsp.tool", "get_lsp_tool"),
    ("revert", "ag3nt_agent.revert_tools", "get_revert_tools"),
    ("multi_edit", "ag3nt_agent.multi_edit_tool", "get_multi_edit_tool"),
    ("batch", "ag3nt_agent.batch_tool", "get_batch_tool"),
    ("external_path", "ag3nt_agent.external_path_tool", "get_external_access_tools"),
    ("session_search", "ag3nt_agent.session_search_tool", "get_session_search_tool"),
    ("image_tools", "ag3nt_agent.generate_image_tool", "get_image_tools"),
    ("create_skill", "ag3nt_agent.create_skill_tool", "get_create_skill_tools"),
    ("context_tools", "ag3nt_agent.context_tools", "get_context_tools"),
]


def _is_allowed(tool_policy: Any, name: str) -> bool:
    """Check if a tool is allowed by the policy (safe helper)."""
    try:
        return tool_policy.is_allowed(name)
    except AttributeError:
        return True  # policy has no is_allowed — allow anyway


def load_tools(
    registry: list[tuple[str, str, str]] | None = None,
    tool_policy: Any | None = None,
) -> list:
    """Load tools from the registry.

    Args:
        registry: List of (name, module_path, func_name) tuples.
                  Defaults to ``TOOL_REGISTRY``.
        tool_policy: Optional ``ToolPolicyManager`` instance.  When supplied,
                     tools denied by policy are skipped *before* importing
                     their module, avoiding unnecessary work.

    Returns:
        Flat list of loaded tool objects.
    """
    from concurrent.futures import ThreadPoolExecutor

    if registry is None:
        registry = TOOL_REGISTRY

    # Filter by policy first
    allowed = [
        (name, mod_path, func_name)
        for name, mod_path, func_name in registry
        if tool_policy is None or _is_allowed(tool_policy, name)
    ]
    if tool_policy is not None:
        denied = set(name for name, _, _ in registry) - set(name for name, _, _ in allowed)
        for name in denied:
            logger.debug("Tool '%s' denied by policy, skipping import", name)

    def _load_one(entry: tuple[str, str, str]) -> tuple[str, Any]:
        name, module_path, func_name = entry
        try:
            mod = importlib.import_module(module_path)
            getter = getattr(mod, func_name)
            result = getter() if callable(getter) and not hasattr(getter, "name") else getter
            return name, result
        except ImportError as exc:
            logger.warning("%s tool not available: %s", name, exc)
            return name, None
        except Exception as exc:
            logger.warning("%s tool failed to load: %s", name, exc)
            return name, None

    tools: list = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = pool.map(_load_one, allowed)
        for name, result in results:
            if result is None:
                continue
            if isinstance(result, list):
                tools.extend(result)
                logger.info("%s tools loaded (%d tool(s))", name, len(result))
            else:
                tools.append(result)
                logger.info("%s tool loaded", name)

    # NOTE: Tool event wrapping is handled by start_autonomous_system()
    # in deepagents_runtime.py AFTER the autonomous system boots. This avoids
    # the race condition where _autonomous_runtime is None at tool-load time.

    return tools
