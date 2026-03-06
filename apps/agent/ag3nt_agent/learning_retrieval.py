"""Lightweight learning retrieval for the planning phase.

Searches past blueprints and plans for relevant learnings to inject
into the planning prompt. Uses two strategies:

1. **Context Engine** (when available): Vector similarity search via
   ``ContextEngineClient`` for semantic matching.
2. **Keyword fallback**: Simple keyword search over ``~/.ag3nt/blueprints/``
   and ``~/.ag3nt/plans/`` JSON/Markdown files. No external dependencies.

Usage:
    from ag3nt_agent.learning_retrieval import retrieve_learnings

    learnings = retrieve_learnings("add user authentication")
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("ag3nt.learning_retrieval")

BLUEPRINTS_DIR = Path.home() / ".ag3nt" / "blueprints"
PLANS_DIR = Path.home() / ".ag3nt" / "plans"


def _keyword_search_blueprints(query: str, limit: int = 5) -> list[str]:
    """Search blueprint JSON files by keyword matching.

    Each blueprint is a JSON file with fields like ``goal``, ``tasks``,
    ``learnings``, ``status``, etc.  We match against ``goal`` and
    ``learnings`` fields.

    Returns:
        List of formatted learning strings.
    """
    if not BLUEPRINTS_DIR.exists():
        return []

    query_terms = set(query.lower().split())
    scored: list[tuple[int, str]] = []

    for path in BLUEPRINTS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        goal = str(data.get("goal", "")).lower()
        learnings = data.get("learnings", [])
        status = data.get("status", "unknown")

        # Score by keyword overlap with goal
        goal_terms = set(goal.split())
        overlap = len(query_terms & goal_terms)
        if overlap == 0:
            continue

        for learning in learnings:
            if isinstance(learning, str) and learning.strip():
                scored.append((overlap, f"[{status}] {goal}: {learning}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:limit]]


def _keyword_search_plans(query: str, limit: int = 5) -> list[str]:
    """Search plan markdown files for relevant learnings/outcomes.

    Looks for status lines, outcome summaries, and lesson-learned sections.

    Returns:
        List of formatted learning strings.
    """
    if not PLANS_DIR.exists():
        return []

    query_lower = query.lower()
    results: list[str] = []

    for path in sorted(PLANS_DIR.glob("*.md"), reverse=True):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        if query_lower not in content.lower():
            continue

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem

        # Extract status
        status_match = re.search(r"(?:\*\*)?[Ss]tatus(?:\*\*)?[:\s]+(\w+)", content)
        status = status_match.group(1) if status_match else "unknown"

        # Count progress
        total = len(re.findall(r"- \[[ x]\]", content, re.IGNORECASE))
        done = len(re.findall(r"- \[x\]", content, re.IGNORECASE))

        results.append(f"[{status}] {title} ({done}/{total} tasks)")
        if len(results) >= limit:
            break

    return results


def _context_engine_search(query: str, limit: int = 5) -> list[str]:
    """Search via ContextEngineClient for semantic learning retrieval.

    Returns empty list if the context engine is unavailable.
    """
    try:
        from ag3nt_agent.context_engine_client import (
            ContextEngineClient,
            get_context_engine,
        )

        ce = get_context_engine()
        if ce is None:
            return []

        # Search the learning collection
        import asyncio

        async def _search():
            results = await ce.find_memories(
                query=query,
                limit=limit,
                collection=ContextEngineClient.COLLECTION_LEARNING,
            )
            learnings = []
            for r in results:
                success = r.metadata.get("success", True)
                prefix = "✓" if success else "✗"
                learnings.append(f"{prefix} {r.content}")
            return learnings

        try:
            loop = asyncio.get_running_loop()
            # Can't await in sync context — skip
            return []
        except RuntimeError:
            return asyncio.run(_search())

    except (ImportError, Exception) as exc:
        logger.debug("Context engine search unavailable: %s", exc)
        return []


def retrieve_learnings(query: str, limit: int = 5) -> list[str]:
    """Retrieve relevant learnings for a planning query.

    Combines results from the context engine (if available) and keyword
    search over local blueprint/plan files.

    Args:
        query: The planning goal or topic to search for.
        limit: Maximum number of learnings to return.

    Returns:
        List of formatted learning strings.
    """
    learnings: list[str] = []

    # 1. Try context engine (semantic search)
    ce_results = _context_engine_search(query, limit=limit)
    learnings.extend(ce_results)

    # 2. Keyword search blueprints
    bp_results = _keyword_search_blueprints(query, limit=limit)
    learnings.extend(bp_results)

    # 3. Keyword search plans
    plan_results = _keyword_search_plans(query, limit=limit)
    learnings.extend(plan_results)

    # Deduplicate and limit
    seen: set[str] = set()
    unique: list[str] = []
    for l in learnings:
        if l not in seen:
            seen.add(l)
            unique.append(l)

    return unique[:limit]


def format_learnings_for_prompt(learnings: list[str]) -> str:
    """Format learnings as a system prompt section.

    Returns empty string if no learnings found.
    """
    if not learnings:
        return ""

    lines = ["## Past Learnings", "From past experience with similar tasks:"]
    for l in learnings:
        lines.append(f"- {l}")
    return "\n".join(lines)
