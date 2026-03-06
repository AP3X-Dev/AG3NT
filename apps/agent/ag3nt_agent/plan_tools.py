"""Structured plan tools for AG3NT planning mode.

This module provides plan-file-based planning that replaces the simpler
``write_todos`` approach with structured Markdown plan documents stored
in ``~/.ag3nt/plans/``.  Three @tool functions are exposed:

- **plan_enter** -- create a new plan file (feature or project template)
- **plan_exit**  -- finalize the plan and mark it ready for review
- **checkpoint** -- create a lightweight git tag as a safety-net marker

A ``PlanState`` class holds class-level state so the rest of the agent
(middleware, guard, executor) can query whether a plan is active and
which tools are allowed during planning mode.

These tools coexist with the existing ``planning_tools.py``
(write_todos / read_todos / update_todo) which remains for backward
compatibility.

Usage:
    from ag3nt_agent.plan_tools import get_plan_tools, PlanState

    tools = get_plan_tools()
"""

from __future__ import annotations

import logging
import re
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool

try:
    from langchain.tools import ToolRuntime
except ImportError:
    ToolRuntime = None  # type: ignore[assignment,misc]

logger = logging.getLogger("ag3nt.tools.plan")


# ---------------------------------------------------------------------------
# Plan state (thread-safe, keyed by thread_id)
# ---------------------------------------------------------------------------

@dataclass
class _SessionPlanState:
    """Per-session plan state."""

    active_plan: Path | None = None
    plan_mode: bool = False
    plan_mode_type: Literal["feature", "project"] | None = None
    original_request: str = ""


class PlanState:
    """Thread-safe plan state manager keyed by thread_id.

    Provides both a per-session API (``get(thread_id)``) and backward-
    compatible class-level accessors for code that doesn't know the
    thread_id.  The class-level properties delegate to a ``_current``
    context variable set by the tools at call time.

    Attributes:
        PLAN_MODE_TOOLS: ``frozenset`` of tool names the agent is allowed to
            invoke while in plan mode.
    """

    # Tools the agent may use during planning (everything else is blocked
    # by the plan-mode guard middleware).
    PLAN_MODE_TOOLS: frozenset[str] = frozenset({
        "plan_enter",
        "plan_exit",
        "plan_revise",
        "checkpoint",
        "list_plans",
        "resume_plan",
        "read_todos",
        "write_todos",
        "update_todo",
        "read_file",
        "grep",
        "glob",
        "codebase_search",
        "deep_reasoning",
        "memory_search",
        "memory_recall",
        "web_search",
        "webfetch",
        "question",
    })

    _lock = threading.Lock()
    _sessions: dict[str, _SessionPlanState] = {}
    # Fallback for code that doesn't pass thread_id
    _current_thread_id: str | None = None

    @classmethod
    def get(cls, thread_id: str) -> _SessionPlanState:
        """Get or create plan state for a session."""
        with cls._lock:
            if thread_id not in cls._sessions:
                cls._sessions[thread_id] = _SessionPlanState()
            return cls._sessions[thread_id]

    @classmethod
    def set_current(cls, thread_id: str) -> None:
        """Set the current thread context (called by tools on entry)."""
        cls._current_thread_id = thread_id

    @classmethod
    def _current(cls) -> _SessionPlanState:
        """Get the current session's state (fallback: first active or empty)."""
        tid = cls._current_thread_id
        if tid and tid in cls._sessions:
            return cls._sessions[tid]
        # Fallback: find any active session
        with cls._lock:
            for s in cls._sessions.values():
                if s.plan_mode:
                    return s
        return _SessionPlanState()

    # Backward-compatible class-level accessors
    @classmethod
    @property
    def active_plan(cls) -> Path | None:  # type: ignore[override]
        return cls._current().active_plan

    @classmethod
    @property
    def plan_mode(cls) -> bool:  # type: ignore[override]
        return cls._current().plan_mode

    @classmethod
    @property
    def plan_mode_type(cls) -> Literal["feature", "project"] | None:  # type: ignore[override]
        return cls._current().plan_mode_type

    @classmethod
    @property
    def original_request(cls) -> str:  # type: ignore[override]
        return cls._current().original_request

    @classmethod
    def reset(cls, thread_id: str | None = None) -> None:
        """Reset state for a session, or all sessions if thread_id is None."""
        with cls._lock:
            if thread_id:
                cls._sessions.pop(thread_id, None)
            else:
                cls._sessions.clear()
            cls._current_thread_id = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _emit_plan_event(
    event_type: str,
    data: dict,
    runtime: Any = None,
) -> None:
    """Emit a plan lifecycle event via stream_writer (custom stream mode).

    Falls back silently if no runtime/stream_writer is available (e.g. in tests).
    """
    payload = {"type": event_type, **data}
    writer = getattr(runtime, "stream_writer", None) if runtime else None
    if writer is not None:
        try:
            writer(payload)
        except Exception as exc:
            logger.debug("stream_writer failed for %s: %s", event_type, exc)
    else:
        logger.debug("No stream_writer available; %s event not emitted", event_type)


def _get_plans_dir() -> Path:
    """Return ``~/.ag3nt/plans/``, creating it if necessary."""
    plans_dir = Path.home() / ".ag3nt" / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    return plans_dir


def _slugify(text: str, max_length: int = 48) -> str:
    """Convert *text* to a filesystem-safe slug.

    - Lowercases the string
    - Replaces non-alphanumeric characters with hyphens
    - Collapses consecutive hyphens
    - Strips leading/trailing hyphens
    - Truncates to *max_length* characters

    Args:
        text: The raw string to slugify.
        max_length: Maximum character length for the slug.

    Returns:
        A URL/filename-safe slug string.
    """
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug[:max_length]


# ---------------------------------------------------------------------------
# Plan templates
# ---------------------------------------------------------------------------

_FEATURE_TEMPLATE = """\
# Plan: {title}

**Created:** {created}
**Mode:** feature
**Status:** IN PROGRESS

## Objective
<!-- What are we trying to accomplish? -->


## Analysis
<!-- Key observations, constraints, and dependencies -->


## Steps
- [ ] Step 1:
- [ ] Step 2:
- [ ] Step 3:

## Files to Modify
<!-- List of files that will be created or changed -->


## Risks
<!-- What could go wrong?  Edge cases to watch for -->

"""

_PROJECT_TEMPLATE = """\
# Plan: {title}

**Created:** {created}
**Mode:** project (PRP)
**Status:** IN PROGRESS

## Goal & Context
<!-- What are we building and why?  Link to requirements if available. -->


## Blueprint
<!-- High-level architecture and design decisions -->


## Step 1: Setup & Scaffolding
[CHECKPOINT]
- [ ] Step 1:

## Step 2: Core Implementation
[CHECKPOINT]
- [ ] Step 2:

## Step 3: Integration & Wiring
[CHECKPOINT]
- [ ] Step 3:

## Step 4: Validation & Polish
[CHECKPOINT]
- [ ] Step 4:

## Validation
<!-- How will we know this is done?  Acceptance criteria. -->


## Anti-Patterns
<!-- Things to avoid based on past experience -->


## Notes
<!-- Scratch space, open questions, follow-ups -->

"""


def _get_thread_id(runtime: Any) -> str:
    """Extract thread_id from a ToolRuntime, falling back to 'default'."""
    if runtime is None:
        return "default"
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    return configurable.get("thread_id", "default")


# ---------------------------------------------------------------------------
# @tool functions
# ---------------------------------------------------------------------------

@tool
def plan_enter(
    title: str,
    mode: str = "feature",
    runtime: Any = None,
) -> str:
    """Create a new structured plan and enter planning mode.

    Creates a Markdown plan file in ``~/.ag3nt/plans/`` using either a
    lightweight *feature* template or a full *project* (PRP-style)
    template.  The agent switches into planning mode, restricting
    available tools to those relevant for planning.

    Args:
        title: Short descriptive title for the plan.
        mode: Template to use -- ``"feature"`` (default, lightweight)
              or ``"project"`` (full PRP with checkpoints).
        runtime: Injected by LangChain -- provides ``stream_writer``
                 for emitting custom events to the LangGraph stream.

    Returns:
        Confirmation message with the path to the newly created plan file.
    """
    thread_id = _get_thread_id(runtime)
    PlanState.set_current(thread_id)
    session = PlanState.get(thread_id)

    if session.plan_mode:
        return (
            f"Already in plan mode.  Active plan: {session.active_plan}\n"
            "Use plan_exit() first to close the current plan before starting a new one."
        )

    if mode not in ("feature", "project"):
        return f"Invalid mode '{mode}'.  Use 'feature' or 'project'."

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    slug = _slugify(title)
    filename = f"{timestamp}-{slug}.md"

    plans_dir = _get_plans_dir()
    plan_path = plans_dir / filename

    # Select template and fill placeholders
    template = _PROJECT_TEMPLATE if mode == "project" else _FEATURE_TEMPLATE
    content = template.format(
        title=title,
        created=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    plan_path.write_text(content, encoding="utf-8")

    # Update per-session state
    session.active_plan = plan_path
    session.plan_mode = True
    session.plan_mode_type = mode
    # original_request is set by the caller (middleware) before invoking the tool

    logger.info("Plan created: %s (mode=%s)", plan_path, mode)

    # Emit custom event for LangGraph streaming consumers
    _emit_plan_event(
        "plan_created",
        {
            "plan_path": str(plan_path),
            "title": title,
            "mode": mode,
            "created": now.isoformat(),
        },
        runtime=runtime,
    )

    return (
        f"Plan created: {plan_path}\n"
        f"Mode: {mode}\n"
        f"Title: {title}\n\n"
        "You are now in **planning mode**.  "
        "Fill in the plan details, then call plan_exit() when ready."
    )


@tool
def plan_exit(
    summary: str,
    ready: bool = True,
    runtime: Any = None,
) -> str:
    """Finalize the active plan and exit planning mode.

    Updates the plan file status to ``READY FOR REVIEW``, appends the
    provided summary, counts total steps and checkpoints, and emits a
    ``plan_ready_for_approval`` event so the UI can show an approval
    card.

    Args:
        summary: Brief summary of the completed plan (what it will do).
        ready: If ``True`` (default), marks the plan as ready for review.
               If ``False``, marks it as a draft (not yet finalized).
        runtime: Injected by LangChain -- provides ``stream_writer``
                 for emitting custom events to the LangGraph stream.

    Returns:
        Confirmation message with step/checkpoint counts.
    """
    thread_id = _get_thread_id(runtime)
    PlanState.set_current(thread_id)
    session = PlanState.get(thread_id)

    if not session.plan_mode or session.active_plan is None:
        return "Not in plan mode.  Call plan_enter() first to create a plan."

    plan_path = session.active_plan

    if not plan_path.exists():
        PlanState.reset(thread_id)
        return f"Plan file missing: {plan_path}.  State has been reset."

    # Read current plan content
    content = plan_path.read_text(encoding="utf-8")

    # Count steps  (two patterns to be resilient)
    #   Pattern 1:  "- [ ] Step N:"  (checkbox style)
    #   Pattern 2:  "## Step N"      (heading style, fallback)
    step_pattern_checkbox = re.compile(r"^- \[ \] Step \d+:", re.MULTILINE)
    step_pattern_heading = re.compile(r"^##\s+Step\s+\d+", re.MULTILINE)
    step_count = len(step_pattern_checkbox.findall(content))
    if step_count == 0:
        step_count = len(step_pattern_heading.findall(content))

    # Count checkpoints
    checkpoint_count = len(re.findall(r"\[CHECKPOINT\]", content))

    # Update status in the plan file
    new_status = "READY FOR REVIEW" if ready else "DRAFT"
    content = re.sub(
        r"\*\*Status:\*\*\s*.*",
        f"**Status:** {new_status}",
        content,
        count=1,
    )

    # Append summary section
    summary_section = (
        f"\n---\n\n"
        f"## Summary\n\n"
        f"{summary}\n\n"
        f"**Steps:** {step_count}  \n"
        f"**Checkpoints:** {checkpoint_count}  \n"
        f"**Finalized:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
    )
    content += summary_section

    plan_path.write_text(content, encoding="utf-8")

    logger.info(
        "Plan finalized: %s (steps=%d, checkpoints=%d, ready=%s)",
        plan_path, step_count, checkpoint_count, ready,
    )

    # Only emit approval event and exit plan mode when ready
    if ready:
        _emit_plan_event(
            "plan_ready_for_approval",
            {
                "plan_path": str(plan_path),
                "plan_content": content,
                "summary": summary,
                "step_count": step_count,
                "checkpoint_count": checkpoint_count,
                "ready": ready,
            },
            runtime=runtime,
        )
        # Exit plan mode — agent awaits user approval
        session.plan_mode = False
    # When ready=False, stay in plan mode so agent can continue editing

    return (
        f"Plan finalized: {plan_path}\n"
        f"Status: {new_status}\n"
        f"Steps: {step_count}\n"
        f"Checkpoints: {checkpoint_count}\n\n"
        f"Summary: {summary}"
    )


@tool
def checkpoint(
    label: str = "",
) -> str:
    """Create a lightweight git tag as a plan checkpoint safety net.

    Tags the current HEAD with ``plan-checkpoint-{label}`` so the user
    can easily roll back if needed.  This is best-effort: if git is
    unavailable or the working directory is not a repository the tool
    returns a warning instead of failing.

    Args:
        label: Optional short label for the checkpoint (e.g. ``"step-2"``).
               If empty, a timestamp is used.

    Returns:
        Confirmation message with the tag name, or a warning if tagging failed.
    """
    if not label:
        label = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    tag_name = f"plan-checkpoint-{_slugify(label)}"

    try:
        result = subprocess.run(
            ["git", "tag", tag_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("Checkpoint created: %s", tag_name)
            return f"Checkpoint created: git tag `{tag_name}`"
        else:
            error = result.stderr.strip()
            logger.warning("Checkpoint tag failed: %s", error)
            return f"Checkpoint tag failed (non-fatal): {error}"
    except FileNotFoundError:
        logger.warning("git not found; checkpoint skipped")
        return "Checkpoint skipped: git is not available on this system."
    except subprocess.TimeoutExpired:
        logger.warning("git tag timed out; checkpoint skipped")
        return "Checkpoint skipped: git tag command timed out."
    except Exception as e:
        logger.warning("Checkpoint error: %s", e)
        return f"Checkpoint skipped (non-fatal): {e}"


# ---------------------------------------------------------------------------
# Cross-session plan tools
# ---------------------------------------------------------------------------

@tool
def list_plans(status: str = "all", limit: int = 10) -> str:
    """List your plans across all sessions.

    Args:
        status: Filter by status -- "active", "completed", "failed", or "all"
        limit: Maximum number of plans to return
    """
    from ag3nt_agent.plan_index import get_plan_index

    index = get_plan_index()
    plans = index.list_plans(status=status if status != "all" else None, limit=limit)
    if not plans:
        return "No plans found."

    lines = []
    for p in plans:
        date_str = p.created.strftime("%Y-%m-%d") if p.created else "unknown"
        progress = f"{p.completed_count}/{p.task_count}" if p.task_count > 0 else "—"
        lines.append(
            f"- [{p.status}] {p.title} ({date_str}, {progress} tasks) ID: {p.plan_id}"
        )
    return "\n".join(lines)


@tool
def resume_plan(plan_id: str) -> str:
    """Resume a plan from a previous session.

    Loads the plan file, shows current progress, and sets it as
    the active plan for this session.

    Args:
        plan_id: The plan ID (filename stem) from list_plans
    """
    from ag3nt_agent.plan_index import get_plan_index

    index = get_plan_index()
    content = index.get_plan_content(plan_id)
    if content is None:
        return f"Plan '{plan_id}' not found. Use list_plans() to see available plans."

    # Set as active plan in PlanState
    plans_dir = _get_plans_dir()
    plan_path = plans_dir / f"{plan_id}.md"
    if not plan_path.exists():
        # Try fuzzy
        for p in plans_dir.glob(f"*{plan_id}*.md"):
            plan_path = p
            break

    # Find the thread_id from runtime context
    thread_id = PlanState._current_thread_id or "default"
    state = PlanState.get(thread_id)
    state.active_plan = plan_path
    state.plan_mode = True

    # Count progress
    total = len(re.findall(r"- \[[ x]\]", content, re.IGNORECASE))
    done = len(re.findall(r"- \[x\]", content, re.IGNORECASE))

    preview = content[:2000]
    if len(content) > 2000:
        preview += "\n\n[... truncated, use read_file to see full plan]"

    return (
        f"Plan resumed: {plan_path.name}\n"
        f"Progress: {done}/{total} tasks completed\n\n"
        f"{preview}"
    )


# ---------------------------------------------------------------------------
# Plan revision
# ---------------------------------------------------------------------------

@tool
def plan_revise(
    reason: str,
    runtime: Any = None,
) -> str:
    """Pause the current plan and enter revision mode.

    Use this when you discover the plan needs changes during execution —
    new steps needed, steps should be reordered, or the approach has changed.
    After revising, the plan must be re-approved before execution resumes.

    Args:
        reason: Why the plan needs revision.
        runtime: Injected by LangChain.
    """
    thread_id = _get_thread_id(runtime)
    PlanState.set_current(thread_id)
    session = PlanState.get(thread_id)

    if session.active_plan is None:
        return "No active plan to revise. Use plan_enter() to create one."

    plan_path = session.active_plan
    if not plan_path.exists():
        return f"Plan file missing: {plan_path}"

    # Switch to revision mode
    session.plan_mode = True
    session.plan_mode_type = session.plan_mode_type or "feature"

    # Update status in the plan file
    content = plan_path.read_text(encoding="utf-8")
    content = re.sub(
        r"\*\*Status:\*\*\s*.*",
        "**Status:** REVISING",
        content,
        count=1,
    )
    # Append revision note
    revision_note = (
        f"\n---\n\n"
        f"## Revision ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})\n\n"
        f"**Reason:** {reason}\n\n"
    )
    content += revision_note
    plan_path.write_text(content, encoding="utf-8")

    logger.info("Plan revision started: %s (reason: %s)", plan_path, reason)

    _emit_plan_event(
        "plan_revision_started",
        {
            "plan_path": str(plan_path),
            "reason": reason,
        },
        runtime=runtime,
    )

    return (
        f"Plan revision mode activated for: {plan_path.name}\n"
        f"Reason: {reason}\n\n"
        "You can now edit the plan. Completed steps remain marked.\n"
        "When done revising, call plan_exit() to submit for re-approval."
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_plan_tools() -> list:
    """Get all plan-mode LangChain tools for the agent.

    Returns:
        List of @tool decorated plan functions.
    """
    return [plan_enter, plan_exit, checkpoint, list_plans, resume_plan, plan_revise]
