"""
Standardized Event Definitions for AG3NT Autonomous System.

This module defines all event types used across the AG3NT platform.
Each event is an ``EventDef`` instance that carries a unique name, a human-
readable description, and an optional payload schema used for runtime
validation.

Categories
----------
- Session        session lifecycle
- Message        chat / streaming messages
- Tool           tool invocation lifecycle
- File           filesystem mutations & external changes
- Provider       LLM provider health
- Quality        quality-gate verdicts
- Approval       human-in-the-loop approval flow
- Plan           planning lifecycle
- MCP            Model Context Protocol connections
- LSP            Language Server Protocol diagnostics
- Sentinel       codebase knowledge-graph indexing
- Autofix        automated repair pipeline
- Recovery       self-healing cascade
- Meta           orchestrator ticks & status
- Hook           phase-hook lifecycle
- Heartbeat/Cron periodic scheduling
- Agent          agent-level errors
- Security       anomaly detection & escalation
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# EventDef dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EventDef:
    """Immutable definition of an event in the AG3NT event bus.

    Parameters
    ----------
    name:
        Unique, dot-free, UPPER_SNAKE_CASE identifier (e.g. ``TOOL_CALLED``).
    description:
        Short human-readable explanation of when this event fires.
    schema:
        Optional mapping of *required* payload keys to their expected Python
        types.  Stored as a :class:`types.MappingProxyType` to preserve
        hashability of the frozen dataclass.  Used by :meth:`validate` for
        lightweight runtime checks.
    """

    name: str
    description: str = ""
    schema: types.MappingProxyType | None = None

    def validate(self, payload: dict[str, Any]) -> bool:
        """Return ``True`` if *payload* satisfies this event's schema.

        Validation rules:
        - If ``schema`` is ``None`` any payload is accepted.
        - Every key declared in ``schema`` must be present in *payload*.
        - Each value must be an instance of the declared type.
        """
        if self.schema is None:
            return True
        for key, expected_type in self.schema.items():
            if key not in payload:
                return False
            if not isinstance(payload[key], expected_type):
                return False
        return True

    def __hash__(self) -> int:
        """Hash by *name* only.

        ``MappingProxyType`` is not inherently hashable (it delegates to
        the underlying dict), so the auto-generated frozen-dataclass hash
        would fail.  Since event names are unique module-level constants,
        hashing by name alone is both correct and sufficient.
        """
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventDef):
            return NotImplemented
        return self.name == other.name

    def __repr__(self) -> str:  # pragma: no cover
        return f"EventDef({self.name!r})"


# =========================================================================
# Session events
# =========================================================================

SESSION_CREATED = EventDef(
    name="SESSION_CREATED",
    description="A new agent session has been created.",
    schema=types.MappingProxyType({"session_id": str}),
)
SESSION_UPDATED = EventDef(
    name="SESSION_UPDATED",
    description="An existing session's metadata was updated.",
    schema=types.MappingProxyType({"session_id": str}),
)
SESSION_ARCHIVED = EventDef(
    name="SESSION_ARCHIVED",
    description="A session was archived (soft-deleted).",
    schema=types.MappingProxyType({"session_id": str}),
)
SESSION_RESTORED = EventDef(
    name="SESSION_RESTORED",
    description="An archived session was restored.",
    schema=types.MappingProxyType({"session_id": str}),
)

# =========================================================================
# Message events
# =========================================================================

MESSAGE_SENT = EventDef(
    name="MESSAGE_SENT",
    description="A complete message was sent (user or assistant).",
    schema=types.MappingProxyType({"session_id": str, "role": str}),
)
MESSAGE_PART_UPDATED = EventDef(
    name="MESSAGE_PART_UPDATED",
    description="A streaming message part was updated.",
    schema=types.MappingProxyType({"session_id": str}),
)

# =========================================================================
# Tool events
# =========================================================================

TOOL_CALLED = EventDef(
    name="TOOL_CALLED",
    description="A tool invocation was initiated.",
    schema=types.MappingProxyType({"tool_name": str, "arguments": dict}),
)
TOOL_COMPLETED = EventDef(
    name="TOOL_COMPLETED",
    description="A tool invocation finished (success or failure).",
    schema=types.MappingProxyType({"tool_name": str, "success": bool, "duration_ms": int}),
)

# =========================================================================
# File events
# =========================================================================

FILE_WRITTEN = EventDef(
    name="FILE_WRITTEN",
    description="A file was written by the agent.",
    schema=types.MappingProxyType({"path": str}),
)
FILE_EDITED = EventDef(
    name="FILE_EDITED",
    description="An existing file was edited by the agent.",
    schema=types.MappingProxyType({"path": str}),
)
FILE_DELETED = EventDef(
    name="FILE_DELETED",
    description="A file was deleted by the agent.",
    schema=types.MappingProxyType({"path": str}),
)
FILE_EXTERNAL_CREATED = EventDef(
    name="FILE_EXTERNAL_CREATED",
    description="A file was created outside the agent (detected by watcher).",
    schema=types.MappingProxyType({"path": str}),
)
FILE_EXTERNAL_MODIFIED = EventDef(
    name="FILE_EXTERNAL_MODIFIED",
    description="A file was modified outside the agent (detected by watcher).",
    schema=types.MappingProxyType({"path": str}),
)
FILE_EXTERNAL_DELETED = EventDef(
    name="FILE_EXTERNAL_DELETED",
    description="A file was deleted outside the agent (detected by watcher).",
    schema=types.MappingProxyType({"path": str}),
)

# =========================================================================
# Provider events
# =========================================================================

PROVIDER_ERROR = EventDef(
    name="PROVIDER_ERROR",
    description="An LLM provider returned an error.",
    schema=types.MappingProxyType({"provider": str, "error": str}),
)
PROVIDER_RESPONSE = EventDef(
    name="PROVIDER_RESPONSE",
    description="An LLM provider returned a successful response.",
    schema=types.MappingProxyType({"provider": str, "latency_ms": int}),
)
PROVIDER_FAILOVER = EventDef(
    name="PROVIDER_FAILOVER",
    description="Traffic was failed over to a backup LLM provider.",
    schema=types.MappingProxyType({"from_provider": str, "to_provider": str}),
)
PROVIDER_RATE_LIMITED = EventDef(
    name="PROVIDER_RATE_LIMITED",
    description="An LLM provider rate-limited the request.",
    schema=types.MappingProxyType({"provider": str}),
)

# =========================================================================
# Quality events
# =========================================================================

QUALITY_CHECK_STARTED = EventDef(
    name="QUALITY_CHECK_STARTED",
    description="A quality check (lint, type-check, test) was started.",
    schema=types.MappingProxyType({"check_type": str}),
)
QUALITY_CHECK_PASSED = EventDef(
    name="QUALITY_CHECK_PASSED",
    description="A quality check passed.",
    schema=types.MappingProxyType({"check_type": str}),
)
QUALITY_CHECK_FAILED = EventDef(
    name="QUALITY_CHECK_FAILED",
    description="A quality check failed.",
    schema=types.MappingProxyType({"check_type": str, "error_count": int}),
)
QUALITY_FIX_APPLIED = EventDef(
    name="QUALITY_FIX_APPLIED",
    description="An automatic quality fix was applied.",
    schema=types.MappingProxyType({"check_type": str, "fix_description": str}),
)
QUALITY_HEALTH_UPDATED = EventDef(
    name="QUALITY_HEALTH_UPDATED",
    description="The overall quality health score was recalculated.",
    schema=types.MappingProxyType({"score": float}),
)
QUALITY_DEGRADATION_ALERT = EventDef(
    name="QUALITY_DEGRADATION_ALERT",
    description="Quality score dropped below threshold.",
    schema=types.MappingProxyType({"score": float, "threshold": float}),
)

# =========================================================================
# Approval events
# =========================================================================

APPROVAL_REQUESTED = EventDef(
    name="APPROVAL_REQUESTED",
    description="Human approval was requested for an action.",
    schema=types.MappingProxyType({"action": str}),
)
APPROVAL_GRANTED = EventDef(
    name="APPROVAL_GRANTED",
    description="Human granted approval.",
    schema=types.MappingProxyType({"action": str}),
)
APPROVAL_DENIED = EventDef(
    name="APPROVAL_DENIED",
    description="Human denied approval.",
    schema=types.MappingProxyType({"action": str}),
)

# =========================================================================
# Plan events
# =========================================================================

PLAN_CREATED = EventDef(
    name="PLAN_CREATED",
    description="A new execution plan was created.",
    schema=types.MappingProxyType({"plan_id": str}),
)
PLAN_APPROVAL_REQUESTED = EventDef(
    name="PLAN_APPROVAL_REQUESTED",
    description="A plan requires human approval before execution.",
    schema=types.MappingProxyType({"plan_id": str}),
)
PLAN_APPROVED = EventDef(
    name="PLAN_APPROVED",
    description="A plan was approved for execution.",
    schema=types.MappingProxyType({"plan_id": str}),
)
PLAN_REJECTED = EventDef(
    name="PLAN_REJECTED",
    description="A plan was rejected by the user.",
    schema=types.MappingProxyType({"plan_id": str}),
)
PLAN_STEP_STARTED = EventDef(
    name="PLAN_STEP_STARTED",
    description="A plan step began executing.",
    schema=types.MappingProxyType({"plan_id": str, "step_index": int}),
)
PLAN_STEP_COMPLETED = EventDef(
    name="PLAN_STEP_COMPLETED",
    description="A plan step finished executing.",
    schema=types.MappingProxyType({"plan_id": str, "step_index": int, "success": bool}),
)
PLAN_COMPLETED = EventDef(
    name="PLAN_COMPLETED",
    description="All plan steps have been executed.",
    schema=types.MappingProxyType({"plan_id": str, "success": bool}),
)

# =========================================================================
# MCP events (Model Context Protocol)
# =========================================================================

MCP_CONNECTED = EventDef(
    name="MCP_CONNECTED",
    description="An MCP server connection was established.",
    schema=types.MappingProxyType({"server_name": str}),
)
MCP_DISCONNECTED = EventDef(
    name="MCP_DISCONNECTED",
    description="An MCP server connection was lost.",
    schema=types.MappingProxyType({"server_name": str}),
)
MCP_HEALTH_CHECK = EventDef(
    name="MCP_HEALTH_CHECK",
    description="An MCP server health check completed.",
    schema=types.MappingProxyType({"server_name": str, "healthy": bool}),
)
MCP_TOOL_DISCOVERED = EventDef(
    name="MCP_TOOL_DISCOVERED",
    description="A new tool was discovered from an MCP server.",
    schema=types.MappingProxyType({"server_name": str, "tool_name": str}),
)

# =========================================================================
# LSP events (Language Server Protocol)
# =========================================================================

LSP_DIAGNOSTICS = EventDef(
    name="LSP_DIAGNOSTICS",
    description="LSP diagnostics (errors/warnings) were received.",
    schema=types.MappingProxyType({"file_path": str, "diagnostic_count": int}),
)
LSP_SERVER_STARTED = EventDef(
    name="LSP_SERVER_STARTED",
    description="An LSP server process was started.",
    schema=types.MappingProxyType({"language": str}),
)
LSP_SERVER_CRASHED = EventDef(
    name="LSP_SERVER_CRASHED",
    description="An LSP server process crashed unexpectedly.",
    schema=types.MappingProxyType({"language": str, "exit_code": int}),
)

# =========================================================================
# Sentinel events (codebase knowledge graph)
# =========================================================================

SENTINEL_INDEXED = EventDef(
    name="SENTINEL_INDEXED",
    description="A file was indexed by the Sentinel knowledge graph.",
    schema=types.MappingProxyType({"path": str}),
)
SENTINEL_SYMBOLS_CHANGED = EventDef(
    name="SENTINEL_SYMBOLS_CHANGED",
    description="Symbol definitions changed in the knowledge graph.",
    schema=types.MappingProxyType({"path": str, "symbols_added": int, "symbols_removed": int}),
)
SENTINEL_REINDEX_REQUESTED = EventDef(
    name="SENTINEL_REINDEX_REQUESTED",
    description="A full or partial reindex of the knowledge graph was requested.",
)

# =========================================================================
# Autofix events
# =========================================================================

AUTOFIX_STARTED = EventDef(
    name="AUTOFIX_STARTED",
    description="The autofix pipeline started processing an issue.",
    schema=types.MappingProxyType({"issue_id": str}),
)
AUTOFIX_COMPLETED = EventDef(
    name="AUTOFIX_COMPLETED",
    description="The autofix pipeline successfully fixed an issue.",
    schema=types.MappingProxyType({"issue_id": str, "fix_description": str}),
)
AUTOFIX_FAILED = EventDef(
    name="AUTOFIX_FAILED",
    description="The autofix pipeline failed to fix an issue.",
    schema=types.MappingProxyType({"issue_id": str, "reason": str}),
)
AUTOFIX_STAGE_ENTERED = EventDef(
    name="AUTOFIX_STAGE_ENTERED",
    description="The autofix pipeline entered a new stage.",
    schema=types.MappingProxyType({"issue_id": str, "stage": str}),
)

# =========================================================================
# Recovery events (self-healing cascade)
# =========================================================================

RECOVERY_STARTED = EventDef(
    name="RECOVERY_STARTED",
    description="A self-healing recovery sequence was initiated.",
    schema=types.MappingProxyType({"level": int}),
)
RECOVERY_SUCCEEDED = EventDef(
    name="RECOVERY_SUCCEEDED",
    description="A recovery action succeeded.",
    schema=types.MappingProxyType({"level": int}),
)
RECOVERY_FAILED = EventDef(
    name="RECOVERY_FAILED",
    description="A recovery action at the current level failed.",
    schema=types.MappingProxyType({"level": int, "error": str}),
)
RECOVERY_ESCALATED = EventDef(
    name="RECOVERY_ESCALATED",
    description="Recovery was escalated to a higher level.",
    schema=types.MappingProxyType({"from_level": int, "to_level": int}),
)

# =========================================================================
# Meta events (orchestrator / meta-loop)
# =========================================================================

META_TICK = EventDef(
    name="META_TICK",
    description="The meta-loop orchestrator produced a periodic tick.",
)
META_STATUS_HINT = EventDef(
    name="META_STATUS_HINT",
    description="The meta-loop emitted a status hint for the UI.",
    schema=types.MappingProxyType({"hint": str}),
)
META_RUNAWAY = EventDef(
    name="META_RUNAWAY",
    description="A runaway loop was detected and halted.",
    schema=types.MappingProxyType({"loop_name": str, "iterations": int}),
)

# =========================================================================
# Hook events (phase-hook lifecycle)
# =========================================================================

HOOK_REGISTERED = EventDef(
    name="HOOK_REGISTERED",
    description="A new phase hook was registered.",
    schema=types.MappingProxyType({"hook_id": str, "phase": str}),
)
HOOK_CANCELLED = EventDef(
    name="HOOK_CANCELLED",
    description="A phase hook was cancelled.",
    schema=types.MappingProxyType({"hook_id": str}),
)
HOOK_FIRED = EventDef(
    name="HOOK_FIRED",
    description="A phase hook was invoked.",
    schema=types.MappingProxyType({"hook_id": str, "phase": str}),
)

# =========================================================================
# Heartbeat / Cron events
# =========================================================================

HEARTBEAT_TICK = EventDef(
    name="HEARTBEAT_TICK",
    description="A periodic heartbeat tick was emitted.",
)
CRON_JOB_FIRED = EventDef(
    name="CRON_JOB_FIRED",
    description="A scheduled cron job was triggered.",
    schema=types.MappingProxyType({"job_name": str}),
)

# =========================================================================
# Agent events
# =========================================================================

AGENT_ERROR = EventDef(
    name="AGENT_ERROR",
    description="The agent encountered an unhandled error.",
    schema=types.MappingProxyType({"error": str}),
)
AGENT_EMPTY_RESPONSE = EventDef(
    name="AGENT_EMPTY_RESPONSE",
    description="The agent produced an empty response.",
)
AGENT_STARTED = EventDef(
    name="AGENT_STARTED",
    description="The agent worker process started.",
)
AGENT_STOPPED = EventDef(
    name="AGENT_STOPPED",
    description="The agent worker process stopped.",
)

# =========================================================================
# Security events
# =========================================================================

SECURITY_ANOMALY = EventDef(
    name="SECURITY_ANOMALY",
    description="A security anomaly was detected (e.g. suspicious file access).",
    schema=types.MappingProxyType({"anomaly_type": str, "details": str}),
)
SECURITY_ESCALATION = EventDef(
    name="SECURITY_ESCALATION",
    description="A security issue was escalated for human review.",
    schema=types.MappingProxyType({"anomaly_type": str, "severity": str}),
)
SECURITY_POLICY_VIOLATION = EventDef(
    name="SECURITY_POLICY_VIOLATION",
    description="A security policy violation was detected.",
    schema=types.MappingProxyType({"policy": str, "action": str}),
)

# =========================================================================
# Snapshot events
# =========================================================================

SNAPSHOT_CREATED = EventDef(
    name="SNAPSHOT_CREATED",
    description="A filesystem snapshot was created.",
    schema=types.MappingProxyType({"snapshot_id": str}),
)
SNAPSHOT_RESTORED = EventDef(
    name="SNAPSHOT_RESTORED",
    description="A filesystem snapshot was restored.",
    schema=types.MappingProxyType({"snapshot_id": str}),
)

# =========================================================================
# Subagent events
# =========================================================================

SUBAGENT_SPAWNED = EventDef(
    name="SUBAGENT_SPAWNED",
    description="A sub-agent was spawned for parallel work.",
    schema=types.MappingProxyType({"subagent_id": str, "task": str}),
)
SUBAGENT_COMPLETED = EventDef(
    name="SUBAGENT_COMPLETED",
    description="A sub-agent completed its task.",
    schema=types.MappingProxyType({"subagent_id": str, "success": bool}),
)
SUBAGENT_FAILED = EventDef(
    name="SUBAGENT_FAILED",
    description="A sub-agent failed its task.",
    schema=types.MappingProxyType({"subagent_id": str, "error": str}),
)

# =========================================================================
# Context events
# =========================================================================

CONTEXT_COMPACTED = EventDef(
    name="CONTEXT_COMPACTED",
    description="The conversation context was compacted to reduce token usage.",
    schema=types.MappingProxyType({"tokens_before": int, "tokens_after": int}),
)
CONTEXT_MEMORY_FLUSHED = EventDef(
    name="CONTEXT_MEMORY_FLUSHED",
    description="Memory was flushed to long-term storage.",
)

# =========================================================================
# ALL_EVENTS registry
# =========================================================================

# WARNING: ALL_EVENTS must remain the LAST statement in this module.
# Any EventDef defined after this line will not be registered.
ALL_EVENTS: dict[str, EventDef] = {
    v.name: v
    for k, v in globals().items()
    if isinstance(v, EventDef)
}
