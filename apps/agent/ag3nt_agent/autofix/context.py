"""FixContext — single data object flowing through the 7 autofix pipeline stages."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class StackFrame:
    file: str = ""
    line: int = 0
    function: str = ""
    source_snippet: str = ""
    local_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class GitContext:
    diffs: str = ""
    recent_logs: list[str] = field(default_factory=list)
    blame: str = ""


@dataclass
class AttemptRecord:
    attempt_number: int = 0
    diff_summary: str = ""
    verification_stage_failed: str = ""
    failure_detail: str = ""
    files_modified: list[str] = field(default_factory=list)


@dataclass
class FixContext:
    # Identity
    id: str = field(default_factory=lambda: str(uuid4()))
    trigger: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Error info (input)
    errors: list[dict[str, Any]] = field(default_factory=list)
    health_metrics: dict[str, Any] = field(default_factory=dict)

    # Classify stage
    category: str = ""
    severity: str = "medium"
    affected_modules: list[str] = field(default_factory=list)

    # Enrich stage
    stack_frames: list[StackFrame] = field(default_factory=list)
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    git_context: GitContext = field(default_factory=GitContext)
    event_timeline: list[dict[str, Any]] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    fix_hints: list[str] = field(default_factory=list)

    # Score stage
    confidence: float = 0.0
    risk_level: str = "medium"
    verification_depth: str = "standard"

    # Fix stage
    attempt_number: int = 0
    fix_response: str = ""
    prior_attempts: list[AttemptRecord] = field(default_factory=list)
    session_id: str = ""

    # Verify stage
    validation: dict[str, Any] = field(default_factory=dict)
    reproduced_before: bool = False
    reproduced_after: bool = False

    # Learn stage
    outcome: str = ""
    fix_duration_ms: int = 0
    context_types_used: list[str] = field(default_factory=list)
