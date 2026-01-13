"""Core data models for the Universal Work System.

These models define the persistent entities for work management:
- WorkItem: A unit of work across any domain
- PlanStep: A step in a WorkItem plan (maps to DeepAgents todos)
- Link: Relationships between WorkItems
- AgentSession: Visibility into agent work
- AgentActivity: Audit trail for agent actions
- FeedbackEvent: User corrections for ranking improvement
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def generate_id() -> str:
    """Generate a unique ID for entities."""
    return str(uuid4())


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class WorkItemStatus(str, Enum):
    """Status of a WorkItem."""
    INBOX = "inbox"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELED = "canceled"


class PlanStepStatus(str, Enum):
    """Status of a PlanStep (maps to DeepAgents todo status)."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class OwnerType(str, Enum):
    """Type of owner for a WorkItem."""
    HUMAN = "human"
    AGENT = "agent"
    UNASSIGNED = "unassigned"


class LinkType(str, Enum):
    """Type of relationship between WorkItems."""
    DUPLICATE_OF = "duplicate_of"
    RELATED_TO = "related_to"
    BLOCKS = "blocks"
    BLOCKED_BY = "blocked_by"
    SAME_ENTITY = "same_entity"


class Origin(BaseModel):
    """Source metadata for external intake."""
    source: str = Field(description="Source system (e.g., 'github', 'email', 'slack')")
    external_id: str | None = Field(default=None, description="ID in source system")
    url: str | None = Field(default=None, description="URL in source system")
    raw_data: dict[str, Any] | None = Field(default=None, description="Raw source data")


class PlanStep(BaseModel):
    """A step in a WorkItem plan.
    
    Maps directly to DeepAgents todo items:
    - content -> PlanStep.content
    - status -> PlanStep.status  
    - activeForm -> PlanStep.active_form
    """
    id: str = Field(default_factory=generate_id)
    work_item_id: str = Field(description="Parent WorkItem ID")
    content: str = Field(description="Task description")
    status: PlanStepStatus = Field(default=PlanStepStatus.PENDING)
    active_form: str | None = Field(default=None, description="What is currently being done")
    position: int = Field(default=0, description="Order in the plan")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def to_todo_dict(self) -> dict[str, Any]:
        """Convert to DeepAgents todo format."""
        result = {
            "content": self.content,
            "status": self.status.value,
        }
        if self.active_form:
            result["activeForm"] = self.active_form
        return result

    @classmethod
    def from_todo_dict(cls, todo: dict[str, Any], work_item_id: str, position: int = 0) -> PlanStep:
        """Create from DeepAgents todo format."""
        return cls(
            work_item_id=work_item_id,
            content=todo.get("content", ""),
            status=PlanStepStatus(todo.get("status", "pending")),
            active_form=todo.get("activeForm"),
            position=position,
        )


class WorkItem(BaseModel):
    """A persistent unit of work across any domain."""
    id: str = Field(default_factory=generate_id)
    title: str = Field(description="Short title for the work item")
    body: str = Field(default="", description="Detailed description")
    status: WorkItemStatus = Field(default=WorkItemStatus.INBOX)
    priority: int = Field(default=2, ge=0, le=4, description="Priority 0 (highest) to 4 (lowest)")
    due_at: datetime | None = Field(default=None, description="Optional due date")
    domain: str = Field(default="general", description="Domain category")
    labels: list[str] = Field(default_factory=list, description="Tags/labels")
    owner_type: OwnerType = Field(default=OwnerType.UNASSIGNED)
    owner_id: str | None = Field(default=None, description="ID of owner (human or agent)")
    origin: Origin | None = Field(default=None, description="External source metadata")
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    # Denormalized for convenience - plan steps are stored separately
    plan_step_ids: list[str] = Field(default_factory=list, description="Ordered list of PlanStep IDs")


class Link(BaseModel):
    """Relationship between WorkItems."""
    id: str = Field(default_factory=generate_id)
    from_id: str = Field(description="Source WorkItem ID")
    to_id: str = Field(description="Target WorkItem ID")
    link_type: LinkType = Field(description="Type of relationship")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    created_at: datetime = Field(default_factory=utc_now)
    created_by: str | None = Field(default=None, description="Who created the link")


class SessionState(str, Enum):
    """State of an agent session."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentSession(BaseModel):
    """Visibility into agent work on a WorkItem."""
    id: str = Field(default_factory=generate_id)
    agent_id: str = Field(description="Agent identifier")
    work_item_id: str = Field(description="WorkItem being worked on")
    state: SessionState = Field(default=SessionState.ACTIVE)
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = Field(default=None)
    summary: str | None = Field(default=None, description="Summary of work done")


class ActivityType(str, Enum):
    """Type of agent activity."""
    STARTED = "started"
    STEP_COMPLETED = "step_completed"
    STEP_STARTED = "step_started"
    TOOL_CALLED = "tool_called"
    ERROR = "error"
    PAUSED = "paused"
    COMPLETED = "completed"
    MESSAGE = "message"


class AgentActivity(BaseModel):
    """Audit trail for agent actions."""
    id: str = Field(default_factory=generate_id)
    session_id: str = Field(description="Parent AgentSession ID")
    activity_type: ActivityType = Field(description="Type of activity")
    summary: str = Field(description="Short description of the activity")
    artifacts: list[str] = Field(default_factory=list, description="References to related artifacts")
    timestamp: datetime = Field(default_factory=utc_now)


class SuggestionType(str, Enum):
    """Type of triage suggestion."""
    DUPLICATE = "duplicate"
    RELATED = "related"
    ASSIGNEE = "assignee"
    PRIORITY = "priority"
    NEXT_ACTION = "next_action"


class FeedbackEvent(BaseModel):
    """Captures user corrections to improve ranking."""
    id: str = Field(default_factory=generate_id)
    work_item_id: str = Field(description="WorkItem the feedback is about")
    suggestion_type: SuggestionType = Field(description="Type of suggestion being corrected")
    suggested_value: Any = Field(description="What the system suggested")
    final_value: Any = Field(description="What the user chose")
    accepted: bool = Field(description="Whether suggestion was accepted")
    timestamp: datetime = Field(default_factory=utc_now)


class TriageSuggestion(BaseModel):
    """A single triage suggestion with explanation."""
    suggestion_type: SuggestionType
    suggested_value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(max_length=3, description="Up to 3 short reasons")
    evidence: list[str] = Field(default_factory=list, description="Evidence pointers (e.g., item IDs)")


class TriageSuggestionBundle(BaseModel):
    """Bundle of triage suggestions for a WorkItem."""
    work_item_id: str
    duplicates: list[TriageSuggestion] = Field(default_factory=list)
    related: list[TriageSuggestion] = Field(default_factory=list)
    assignee: TriageSuggestion | None = None
    priority: TriageSuggestion | None = None
    next_actions: list[TriageSuggestion] = Field(default_factory=list)

