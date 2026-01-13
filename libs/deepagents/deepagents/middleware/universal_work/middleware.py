"""Universal Work Middleware - Drop-in replacement for TodoListMiddleware.

Provides backward-compatible write_todos and read_todos tools backed by
persistent WorkItem and PlanStep models, plus additional universal work tools.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command

from deepagents.middleware.universal_work.models import (
    ActivityType,
    AgentActivity,
    AgentSession,
    FeedbackEvent,
    Link,
    LinkType,
    OwnerType,
    PlanStep,
    PlanStepStatus,
    SuggestionType,
    TriageSuggestion,
    TriageSuggestionBundle,
    WorkItem,
    WorkItemStatus,
)
from deepagents.middleware.universal_work.storage import (
    FileBackendStorage,
    WorkStorageProtocol,
)
from deepagents.middleware.universal_work.retrieval import (
    RetrievalBackend,
    SimpleKeywordRetrieval,
    TriageEngine,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# --- Tool Descriptions ---

WRITE_TODOS_DESCRIPTION = """Update the plan for the current work item.

Replaces the current todo list with the provided items. Each todo represents 
a step in the plan for the active WorkItem.

Parameters:
- todos: List of todo items, each with:
  - content (str): Task description
  - status (str): "pending", "in_progress", or "completed"
  - activeForm (str, optional): What you're currently doing for this task

Best practices:
- Keep 3-6 items maximum
- Only use for complex, multi-step tasks
- Update status as you complete each item
- Do NOT call multiple times in parallel

Note: Todos are plan steps for the currently active WorkItem. If no WorkItem 
is active, one will be created automatically based on the current objective."""

READ_TODOS_DESCRIPTION = """Read the current todo list (plan steps) for the active WorkItem.

Returns the list of todos with their current status.

Use this to:
- Check what tasks remain
- Review the current plan
- Understand progress"""


# --- Backward-Compatible Todo Tools ---

def _create_write_todos_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the write_todos tool backed by persistent storage."""

    async def async_write_todos(
        todos: list[dict[str, Any]],
        runtime: ToolRuntime,
    ) -> str:
        """Write todos to the current WorkItem's plan steps."""
        # Get or create current WorkItem
        current_id = storage.get_current_work_item_id()
        
        if current_id is None:
            # Auto-create WorkItem from current objective
            item = WorkItem(
                title="Current Task",
                body="Auto-created from agent objective",
                status=WorkItemStatus.IN_PROGRESS,
                owner_type=OwnerType.AGENT,
            )
            item = storage.create_work_item(item)
            storage.set_current_work_item_id(item.id)
            current_id = item.id
            logger.info(f"Auto-created WorkItem {item.id} for todos")

        # Convert todos to PlanSteps
        plan_steps = []
        for i, todo in enumerate(todos):
            step = PlanStep.from_todo_dict(todo, current_id, position=i)
            plan_steps.append(step)

        # Replace all plan steps for this work item
        storage.replace_plan_steps(current_id, plan_steps)

        # Update WorkItem status based on plan step statuses
        item = storage.get_work_item(current_id)
        if item:
            all_completed = all(s.status == PlanStepStatus.COMPLETED for s in plan_steps)
            any_in_progress = any(s.status == PlanStepStatus.IN_PROGRESS for s in plan_steps)
            
            if all_completed and plan_steps:
                item.status = WorkItemStatus.DONE
            elif any_in_progress:
                item.status = WorkItemStatus.IN_PROGRESS
            
            storage.update_work_item(item)

        return f"Updated {len(plan_steps)} plan steps for WorkItem {current_id}"

    def sync_write_todos(todos: list[dict[str, Any]], runtime: ToolRuntime) -> str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            async_write_todos(todos, runtime)
        )

    return StructuredTool.from_function(
        name="write_todos",
        description=WRITE_TODOS_DESCRIPTION,
        func=sync_write_todos,
        coroutine=async_write_todos,
    )


def _create_read_todos_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the read_todos tool backed by persistent storage."""

    async def async_read_todos(runtime: ToolRuntime) -> list[dict[str, Any]]:
        """Read todos from the current WorkItem's plan steps."""
        current_id = storage.get_current_work_item_id()
        
        if current_id is None:
            return []
        
        steps = storage.get_plan_steps(current_id)
        return [step.to_todo_dict() for step in steps]

    def sync_read_todos(runtime: ToolRuntime) -> list[dict[str, Any]]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_read_todos(runtime))

    return StructuredTool.from_function(
        name="read_todos",
        description=READ_TODOS_DESCRIPTION,
        func=sync_read_todos,
        coroutine=async_read_todos,
    )


# --- WorkItem Tools ---

WORK_ITEM_CREATE_DESCRIPTION = """Create a new work item for tracking.

Parameters:
- title (str): Short title for the work item
- body (str, optional): Detailed description
- domain (str, optional): Domain category (default: "general")
- labels (list[str], optional): Tags/labels
- priority (int, optional): 0 (highest) to 4 (lowest), default 2

Returns the created WorkItem ID."""

WORK_ITEM_GET_DESCRIPTION = """Get details of a specific work item.

Parameters:
- item_id (str): The WorkItem ID to retrieve

Returns the full WorkItem details including status, plan steps, and links."""

INBOX_LIST_DESCRIPTION = """List work items in the inbox or with filters.

Parameters:
- status (str, optional): Filter by status (inbox, accepted, in_progress, etc.)
- domain (str, optional): Filter by domain
- labels (list[str], optional): Filter by labels (any match)
- limit (int, optional): Max results (default 20)

Returns a list of WorkItem summaries."""


def _create_work_item_create_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the work_item_create tool."""

    async def async_create(
        title: str,
        body: str = "",
        domain: str = "general",
        labels: list[str] | None = None,
        priority: int = 2,
        runtime: ToolRuntime = None,
    ) -> dict[str, Any]:
        item = WorkItem(
            title=title,
            body=body,
            domain=domain,
            labels=labels or [],
            priority=priority,
            status=WorkItemStatus.INBOX,
        )
        item = storage.create_work_item(item)
        return {"id": item.id, "title": item.title, "status": item.status.value}

    def sync_create(**kwargs) -> dict[str, Any]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_create(**kwargs))

    return StructuredTool.from_function(
        name="work_item_create",
        description=WORK_ITEM_CREATE_DESCRIPTION,
        func=sync_create,
        coroutine=async_create,
    )


def _create_work_item_get_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the work_item_get tool."""

    async def async_get(item_id: str, runtime: ToolRuntime = None) -> dict[str, Any] | str:
        item = storage.get_work_item(item_id)
        if item is None:
            return f"WorkItem {item_id} not found"

        steps = storage.get_plan_steps(item_id)
        links = storage.get_links(item_id)

        return {
            "id": item.id,
            "title": item.title,
            "body": item.body,
            "status": item.status.value,
            "priority": item.priority,
            "domain": item.domain,
            "labels": item.labels,
            "owner_type": item.owner_type.value,
            "owner_id": item.owner_id,
            "created_at": item.created_at.isoformat(),
            "plan_steps": [s.to_todo_dict() for s in steps],
            "links": [{"type": l.link_type.value, "to": l.to_id, "confidence": l.confidence} for l in links],
        }

    def sync_get(item_id: str, runtime: ToolRuntime = None) -> dict[str, Any] | str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_get(item_id, runtime))

    return StructuredTool.from_function(
        name="work_item_get",
        description=WORK_ITEM_GET_DESCRIPTION,
        func=sync_get,
        coroutine=async_get,
    )


def _create_inbox_list_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the inbox_list tool."""

    async def async_list(
        status: str | None = None,
        domain: str | None = None,
        labels: list[str] | None = None,
        limit: int = 20,
        runtime: ToolRuntime = None,
    ) -> list[dict[str, Any]]:
        status_filter = None
        if status:
            try:
                status_filter = WorkItemStatus(status)
            except ValueError:
                pass

        items = storage.list_work_items(
            status=status_filter,
            domain=domain,
            labels=labels,
            limit=limit,
        )

        return [
            {
                "id": item.id,
                "title": item.title,
                "status": item.status.value,
                "priority": item.priority,
                "domain": item.domain,
                "created_at": item.created_at.isoformat(),
            }
            for item in items
        ]

    def sync_list(**kwargs) -> list[dict[str, Any]]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_list(**kwargs))

    return StructuredTool.from_function(
        name="inbox_list",
        description=INBOX_LIST_DESCRIPTION,
        func=sync_list,
        coroutine=async_list,
    )


# --- Linking Tools ---

LINK_CREATE_DESCRIPTION = """Create a relationship between two work items.

Parameters:
- from_id (str): Source WorkItem ID
- to_id (str): Target WorkItem ID
- link_type (str): Type of relationship:
  - "duplicate_of": This item duplicates another
  - "related_to": Items are related
  - "blocks": This item blocks another
  - "blocked_by": This item is blocked by another
- confidence (float, optional): Confidence score 0-1 (default 1.0)

Returns the created Link."""


def _create_link_create_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the link_create tool."""

    async def async_create(
        from_id: str,
        to_id: str,
        link_type: str,
        confidence: float = 1.0,
        runtime: ToolRuntime = None,
    ) -> dict[str, Any] | str:
        try:
            lt = LinkType(link_type)
        except ValueError:
            return f"Invalid link type: {link_type}. Use: duplicate_of, related_to, blocks, blocked_by"

        link = Link(
            from_id=from_id,
            to_id=to_id,
            link_type=lt,
            confidence=confidence,
        )
        link = storage.create_link(link)
        return {"id": link.id, "from": from_id, "to": to_id, "type": link_type}

    def sync_create(**kwargs) -> dict[str, Any] | str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_create(**kwargs))

    return StructuredTool.from_function(
        name="link_create",
        description=LINK_CREATE_DESCRIPTION,
        func=sync_create,
        coroutine=async_create,
    )


LINK_LIST_DESCRIPTION = """List all links for a work item.

Parameters:
- item_id (str): WorkItem ID to get links for

Returns links where this item is either source or target."""


def _create_link_list_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the link_list tool."""

    async def async_list(item_id: str, runtime: ToolRuntime = None) -> list[dict[str, Any]]:
        links = storage.get_links(item_id)
        return [
            {
                "id": l.id,
                "from": l.from_id,
                "to": l.to_id,
                "type": l.link_type.value,
                "confidence": l.confidence,
            }
            for l in links
        ]

    def sync_list(item_id: str, runtime: ToolRuntime = None) -> list[dict[str, Any]]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_list(item_id, runtime))

    return StructuredTool.from_function(
        name="link_list",
        description=LINK_LIST_DESCRIPTION,
        func=sync_list,
        coroutine=async_list,
    )


# --- Agent Session Tools ---

SESSION_START_DESCRIPTION = """Start a new agent session for working on a work item.

Parameters:
- agent_id (str): Identifier for this agent
- work_item_id (str): WorkItem to work on

Returns the session ID for logging activities."""


def _create_session_start_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the agent_session_start tool."""

    async def async_start(
        agent_id: str,
        work_item_id: str,
        runtime: ToolRuntime = None,
    ) -> dict[str, Any] | str:
        # Verify work item exists
        item = storage.get_work_item(work_item_id)
        if item is None:
            return f"WorkItem {work_item_id} not found"

        session = AgentSession(
            agent_id=agent_id,
            work_item_id=work_item_id,
        )
        session = storage.create_session(session)

        # Set as current work item
        storage.set_current_work_item_id(work_item_id)

        # Update work item ownership
        item.owner_type = OwnerType.AGENT
        item.owner_id = agent_id
        item.status = WorkItemStatus.IN_PROGRESS
        storage.update_work_item(item)

        return {"session_id": session.id, "work_item_id": work_item_id}

    def sync_start(**kwargs) -> dict[str, Any] | str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_start(**kwargs))

    return StructuredTool.from_function(
        name="agent_session_start",
        description=SESSION_START_DESCRIPTION,
        func=sync_start,
        coroutine=async_start,
    )


ACTIVITY_LOG_DESCRIPTION = """Log an activity in the current agent session.

Parameters:
- session_id (str): Session ID from agent_session_start
- activity_type (str): Type: started, step_completed, step_started, tool_called, error, completed
- summary (str): Short description of the activity
- artifacts (list[str], optional): References to related files/outputs"""


def _create_activity_log_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the agent_activity_log tool."""

    async def async_log(
        session_id: str,
        activity_type: str,
        summary: str,
        artifacts: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> dict[str, Any] | str:
        session = storage.get_session(session_id)
        if session is None:
            return f"Session {session_id} not found"

        try:
            at = ActivityType(activity_type)
        except ValueError:
            return f"Invalid activity type: {activity_type}"

        activity = AgentActivity(
            session_id=session_id,
            activity_type=at,
            summary=summary,
            artifacts=artifacts or [],
        )
        activity = storage.log_activity(activity)
        return {"activity_id": activity.id}

    def sync_log(**kwargs) -> dict[str, Any] | str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_log(**kwargs))

    return StructuredTool.from_function(
        name="agent_activity_log",
        description=ACTIVITY_LOG_DESCRIPTION,
        func=sync_log,
        coroutine=async_log,
    )


# --- Feedback Tool ---

FEEDBACK_RECORD_DESCRIPTION = """Record feedback on a triage suggestion.

Use this to improve future suggestions by recording whether a suggestion
was accepted or corrected.

Parameters:
- work_item_id (str): The WorkItem the feedback is about
- suggestion_type (str): Type: duplicate, related, assignee, priority, next_action
- suggested_value: What the system suggested
- final_value: What was actually used
- accepted (bool): Whether the suggestion was accepted as-is"""


def _create_feedback_record_tool(storage: WorkStorageProtocol) -> BaseTool:
    """Create the feedback_record tool."""

    async def async_record(
        work_item_id: str,
        suggestion_type: str,
        suggested_value: Any,
        final_value: Any,
        accepted: bool,
        runtime: ToolRuntime = None,
    ) -> dict[str, Any] | str:
        try:
            st = SuggestionType(suggestion_type)
        except ValueError:
            return f"Invalid suggestion type: {suggestion_type}"

        feedback = FeedbackEvent(
            work_item_id=work_item_id,
            suggestion_type=st,
            suggested_value=suggested_value,
            final_value=final_value,
            accepted=accepted,
        )
        feedback = storage.record_feedback(feedback)
        return {"feedback_id": feedback.id, "accepted": accepted}

    def sync_record(**kwargs) -> dict[str, Any] | str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_record(**kwargs))

    return StructuredTool.from_function(
        name="feedback_record",
        description=FEEDBACK_RECORD_DESCRIPTION,
        func=sync_record,
        coroutine=async_record,
    )


# --- Triage Tool ---

TRIAGE_SUGGEST_DESCRIPTION = """Get AI-powered triage suggestions for a work item.

Analyzes the work item and suggests:
- Potential duplicates (similar existing items)
- Related items (items that may be connected)
- Priority recommendation

Parameters:
- item_id (str): WorkItem ID to analyze
- modes (list[str], optional): Types of suggestions to generate
  - "duplicates": Find potential duplicate items
  - "related": Find related items
  - "priority": Suggest priority level
  Default: all modes

Returns suggestions with confidence scores and explanations."""


def _create_triage_suggest_tool(
    storage: WorkStorageProtocol,
    triage_engine: TriageEngine,
) -> BaseTool:
    """Create the triage_suggest tool."""

    async def async_suggest(
        item_id: str,
        modes: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> dict[str, Any] | str:
        result = triage_engine.generate_suggestions(item_id, modes)

        if isinstance(result, str):
            return result

        # Convert to dict for tool output
        return {
            "work_item_id": result.work_item_id,
            "duplicates": [
                {
                    "item_id": s.suggested_value,
                    "confidence": s.confidence,
                    "reasons": s.reasons,
                }
                for s in result.duplicates
            ],
            "related": [
                {
                    "item_id": s.suggested_value,
                    "confidence": s.confidence,
                    "reasons": s.reasons,
                }
                for s in result.related
            ],
            "priority": {
                "suggested": result.priority.suggested_value,
                "confidence": result.priority.confidence,
                "reasons": result.priority.reasons,
            } if result.priority else None,
        }

    def sync_suggest(**kwargs) -> dict[str, Any] | str:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(async_suggest(**kwargs))

    return StructuredTool.from_function(
        name="triage_suggest",
        description=TRIAGE_SUGGEST_DESCRIPTION,
        func=sync_suggest,
        coroutine=async_suggest,
    )


# --- System Prompt ---

UNIVERSAL_WORK_SYSTEM_PROMPT = """## Universal Work System

You have access to a persistent work management system. Your todos represent plan steps
for the currently active WorkItem.

### Planning Tools (backward compatible)
- **write_todos**: Update plan steps for the current work item
- **read_todos**: Read current plan steps

### Work Management Tools
- **work_item_create**: Create a new work item
- **work_item_get**: Get full details of a work item
- **inbox_list**: List work items with filters

### Linking Tools
- **link_create**: Create relationships between work items
- **link_list**: List links for a work item

### Agent Session Tools
- **agent_session_start**: Start working on a work item (sets it as current)
- **agent_activity_log**: Log activities during work

### Triage
- **triage_suggest**: Get AI-powered suggestions for duplicates, related items, priority

### Feedback
- **feedback_record**: Record feedback on suggestions

Best practices:
- Use write_todos for planning complex tasks (3-6 steps max)
- Use agent_session_start to claim and work on specific items
- Log important activities for audit trail
- Create links to mark duplicates or related work
"""


# --- Middleware Class ---

class UniversalWorkMiddleware(AgentMiddleware):
    """Universal Work System Middleware - Drop-in replacement for TodoListMiddleware.

    Provides persistent work management with backward-compatible write_todos and
    read_todos tools, plus additional tools for work intake, triage, linking,
    agent sessions, and feedback.

    Args:
        storage: Storage backend for persistence. Defaults to FileBackendStorage.
        storage_path: Path for file storage (used if storage not provided).
        system_prompt: Optional custom system prompt override.
        enabled_tools: Optional list of tool names to enable. Defaults to all.

    Example:
        ```python
        from deepagents.middleware.universal_work import UniversalWorkMiddleware
        from langchain.agents import create_agent

        # Drop-in replacement for TodoListMiddleware
        agent = create_agent(
            model="anthropic:claude-sonnet-4-20250514",
            middleware=[UniversalWorkMiddleware()],
        )

        # With custom storage path
        agent = create_agent(
            middleware=[UniversalWorkMiddleware(storage_path="./my_work_data")],
        )
        ```
    """

    # All available tools
    ALL_TOOLS = [
        "write_todos",
        "read_todos",
        "work_item_create",
        "work_item_get",
        "inbox_list",
        "link_create",
        "link_list",
        "agent_session_start",
        "agent_activity_log",
        "triage_suggest",
        "feedback_record",
    ]

    def __init__(
        self,
        *,
        storage: WorkStorageProtocol | None = None,
        storage_path: str | Path = ".universal_work",
        system_prompt: str | None = None,
        enabled_tools: list[str] | None = None,
        retrieval_backend: RetrievalBackend | None = None,
    ) -> None:
        """Initialize the Universal Work Middleware."""
        super().__init__()

        # Initialize storage
        self.storage = storage or FileBackendStorage(storage_path)
        self._custom_system_prompt = system_prompt

        # Initialize triage engine
        self.triage_engine = TriageEngine(
            storage=self.storage,
            retrieval=retrieval_backend,
        )

        # Build tool generators
        self._tool_generators = {
            "write_todos": lambda: _create_write_todos_tool(self.storage),
            "read_todos": lambda: _create_read_todos_tool(self.storage),
            "work_item_create": lambda: _create_work_item_create_tool(self.storage),
            "work_item_get": lambda: _create_work_item_get_tool(self.storage),
            "inbox_list": lambda: _create_inbox_list_tool(self.storage),
            "link_create": lambda: _create_link_create_tool(self.storage),
            "link_list": lambda: _create_link_list_tool(self.storage),
            "agent_session_start": lambda: _create_session_start_tool(self.storage),
            "agent_activity_log": lambda: _create_activity_log_tool(self.storage),
            "triage_suggest": lambda: _create_triage_suggest_tool(self.storage, self.triage_engine),
            "feedback_record": lambda: _create_feedback_record_tool(self.storage),
        }

        # Generate enabled tools
        tools_to_enable = enabled_tools or self.ALL_TOOLS
        self.tools = [
            self._tool_generators[name]()
            for name in tools_to_enable
            if name in self._tool_generators
        ]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Add Universal Work system prompt."""
        system_prompt = self._custom_system_prompt or UNIVERSAL_WORK_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = (
                request.system_prompt + "\n\n" + system_prompt
                if request.system_prompt
                else system_prompt
            )
            request = request.override(system_prompt=new_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Add Universal Work system prompt."""
        system_prompt = self._custom_system_prompt or UNIVERSAL_WORK_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = (
                request.system_prompt + "\n\n" + system_prompt
                if request.system_prompt
                else system_prompt
            )
            request = request.override(system_prompt=new_prompt)

        return await handler(request)

