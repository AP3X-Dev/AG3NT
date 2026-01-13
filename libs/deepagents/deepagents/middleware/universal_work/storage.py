"""Storage interface and implementations for the Universal Work System.

Provides:
- WorkStorageProtocol: Abstract interface for storage backends
- FileBackendStorage: JSON file-backed storage for development/testing
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepagents.middleware.universal_work.models import (
    AgentActivity,
    AgentSession,
    FeedbackEvent,
    Link,
    LinkType,
    PlanStep,
    WorkItem,
    WorkItemStatus,
)

logger = logging.getLogger(__name__)


class WorkStorageProtocol(abc.ABC):
    """Abstract protocol for Universal Work System storage.
    
    All implementations must be storage-agnostic and support:
    - CRUD operations for all entity types
    - Filtering and querying
    - Async variants for all operations
    """

    # --- WorkItem Operations ---
    
    @abc.abstractmethod
    def create_work_item(self, item: WorkItem) -> WorkItem:
        """Create a new WorkItem."""
        ...

    @abc.abstractmethod
    def get_work_item(self, item_id: str) -> WorkItem | None:
        """Get a WorkItem by ID."""
        ...

    @abc.abstractmethod
    def update_work_item(self, item: WorkItem) -> WorkItem:
        """Update an existing WorkItem."""
        ...

    @abc.abstractmethod
    def list_work_items(
        self,
        status: WorkItemStatus | list[WorkItemStatus] | None = None,
        owner_id: str | None = None,
        domain: str | None = None,
        labels: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkItem]:
        """List WorkItems with optional filters."""
        ...

    # --- PlanStep Operations ---

    @abc.abstractmethod
    def create_plan_step(self, step: PlanStep) -> PlanStep:
        """Create a new PlanStep."""
        ...

    @abc.abstractmethod
    def get_plan_steps(self, work_item_id: str) -> list[PlanStep]:
        """Get all PlanSteps for a WorkItem, ordered by position."""
        ...

    @abc.abstractmethod
    def update_plan_step(self, step: PlanStep) -> PlanStep:
        """Update an existing PlanStep."""
        ...

    @abc.abstractmethod
    def replace_plan_steps(self, work_item_id: str, steps: list[PlanStep]) -> list[PlanStep]:
        """Replace all PlanSteps for a WorkItem (for write_todos compatibility)."""
        ...

    # --- Link Operations ---

    @abc.abstractmethod
    def create_link(self, link: Link) -> Link:
        """Create a new Link."""
        ...

    @abc.abstractmethod
    def get_links(self, work_item_id: str) -> list[Link]:
        """Get all Links for a WorkItem (both from and to)."""
        ...

    # --- AgentSession Operations ---

    @abc.abstractmethod
    def create_session(self, session: AgentSession) -> AgentSession:
        """Create a new AgentSession."""
        ...

    @abc.abstractmethod
    def get_session(self, session_id: str) -> AgentSession | None:
        """Get an AgentSession by ID."""
        ...

    @abc.abstractmethod
    def update_session(self, session: AgentSession) -> AgentSession:
        """Update an existing AgentSession."""
        ...

    # --- AgentActivity Operations ---

    @abc.abstractmethod
    def log_activity(self, activity: AgentActivity) -> AgentActivity:
        """Log an agent activity."""
        ...

    @abc.abstractmethod
    def get_activities(self, session_id: str) -> list[AgentActivity]:
        """Get all activities for a session."""
        ...

    # --- Feedback Operations ---

    @abc.abstractmethod
    def record_feedback(self, feedback: FeedbackEvent) -> FeedbackEvent:
        """Record a feedback event."""
        ...

    # --- Context Management ---

    @abc.abstractmethod
    def get_current_work_item_id(self) -> str | None:
        """Get the currently active WorkItem ID for this context."""
        ...

    @abc.abstractmethod
    def set_current_work_item_id(self, item_id: str | None) -> None:
        """Set the currently active WorkItem ID for this context."""
        ...

    # --- Async variants (default implementations use asyncio.to_thread) ---

    async def acreate_work_item(self, item: WorkItem) -> WorkItem:
        return await asyncio.to_thread(self.create_work_item, item)

    async def aget_work_item(self, item_id: str) -> WorkItem | None:
        return await asyncio.to_thread(self.get_work_item, item_id)

    async def aupdate_work_item(self, item: WorkItem) -> WorkItem:
        return await asyncio.to_thread(self.update_work_item, item)

    async def alist_work_items(self, **kwargs) -> list[WorkItem]:
        return await asyncio.to_thread(lambda: self.list_work_items(**kwargs))

    async def aget_plan_steps(self, work_item_id: str) -> list[PlanStep]:
        return await asyncio.to_thread(self.get_plan_steps, work_item_id)

    async def areplace_plan_steps(self, work_item_id: str, steps: list[PlanStep]) -> list[PlanStep]:
        return await asyncio.to_thread(self.replace_plan_steps, work_item_id, steps)


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def datetime_decoder(dct: dict) -> dict:
    """Decode datetime strings in JSON dicts."""
    for key, value in dct.items():
        if isinstance(value, str):
            # Try to parse as datetime
            for fmt in ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    dct[key] = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
    return dct


class FileBackendStorage(WorkStorageProtocol):
    """File-backed JSON storage for Universal Work System.

    Stores all data in JSON files within a specified directory.
    Suitable for development, testing, and single-user deployments.

    Directory structure:
        {base_path}/
            work_items.json
            plan_steps.json
            links.json
            sessions.json
            activities.json
            feedback.json
            context.json
    """

    def __init__(self, base_path: str | Path = ".universal_work"):
        """Initialize file backend storage.

        Args:
            base_path: Directory to store JSON files. Created if doesn't exist.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize empty files if they don't exist
        self._ensure_file("work_items.json", {})
        self._ensure_file("plan_steps.json", {})
        self._ensure_file("links.json", {})
        self._ensure_file("sessions.json", {})
        self._ensure_file("activities.json", {})
        self._ensure_file("feedback.json", [])
        self._ensure_file("context.json", {"current_work_item_id": None})

    def _ensure_file(self, filename: str, default_content: Any) -> None:
        """Ensure a JSON file exists with default content."""
        filepath = self.base_path / filename
        if not filepath.exists():
            with open(filepath, "w") as f:
                json.dump(default_content, f, cls=DateTimeEncoder)

    def _read_json(self, filename: str) -> Any:
        """Read and parse a JSON file."""
        filepath = self.base_path / filename
        with open(filepath, "r") as f:
            return json.load(f, object_hook=datetime_decoder)

    def _write_json(self, filename: str, data: Any) -> None:
        """Write data to a JSON file."""
        filepath = self.base_path / filename
        with open(filepath, "w") as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)

    # --- WorkItem Operations ---

    def create_work_item(self, item: WorkItem) -> WorkItem:
        items = self._read_json("work_items.json")
        items[item.id] = item.model_dump()
        self._write_json("work_items.json", items)
        return item

    def get_work_item(self, item_id: str) -> WorkItem | None:
        items = self._read_json("work_items.json")
        if item_id in items:
            return WorkItem(**items[item_id])
        return None

    def update_work_item(self, item: WorkItem) -> WorkItem:
        items = self._read_json("work_items.json")
        item.updated_at = datetime.now(timezone.utc)
        items[item.id] = item.model_dump()
        self._write_json("work_items.json", items)
        return item

    def list_work_items(
        self,
        status: WorkItemStatus | list[WorkItemStatus] | None = None,
        owner_id: str | None = None,
        domain: str | None = None,
        labels: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[WorkItem]:
        items = self._read_json("work_items.json")
        result = []

        for item_data in items.values():
            item = WorkItem(**item_data)

            # Apply filters
            if status is not None:
                statuses = [status] if isinstance(status, WorkItemStatus) else status
                if item.status not in statuses:
                    continue
            if owner_id is not None and item.owner_id != owner_id:
                continue
            if domain is not None and item.domain != domain:
                continue
            if labels is not None and not any(l in item.labels for l in labels):
                continue

            result.append(item)

        # Sort by created_at descending, then apply pagination
        result.sort(key=lambda x: x.created_at, reverse=True)
        return result[offset:offset + limit]

    # --- PlanStep Operations ---

    def create_plan_step(self, step: PlanStep) -> PlanStep:
        steps = self._read_json("plan_steps.json")
        steps[step.id] = step.model_dump()
        self._write_json("plan_steps.json", steps)
        return step

    def get_plan_steps(self, work_item_id: str) -> list[PlanStep]:
        steps = self._read_json("plan_steps.json")
        result = [
            PlanStep(**s) for s in steps.values()
            if s.get("work_item_id") == work_item_id
        ]
        result.sort(key=lambda x: x.position)
        return result

    def update_plan_step(self, step: PlanStep) -> PlanStep:
        steps = self._read_json("plan_steps.json")
        step.updated_at = datetime.now(timezone.utc)
        steps[step.id] = step.model_dump()
        self._write_json("plan_steps.json", steps)
        return step

    def replace_plan_steps(self, work_item_id: str, new_steps: list[PlanStep]) -> list[PlanStep]:
        """Replace all PlanSteps for a WorkItem (write_todos compatibility)."""
        steps = self._read_json("plan_steps.json")

        # Remove existing steps for this work item
        steps = {k: v for k, v in steps.items() if v.get("work_item_id") != work_item_id}

        # Add new steps
        for i, step in enumerate(new_steps):
            step.work_item_id = work_item_id
            step.position = i
            steps[step.id] = step.model_dump()

        self._write_json("plan_steps.json", steps)

        # Update WorkItem's plan_step_ids
        item = self.get_work_item(work_item_id)
        if item:
            item.plan_step_ids = [s.id for s in new_steps]
            self.update_work_item(item)

        return new_steps

    # --- Link Operations ---

    def create_link(self, link: Link) -> Link:
        links = self._read_json("links.json")
        links[link.id] = link.model_dump()
        self._write_json("links.json", links)
        return link

    def get_links(self, work_item_id: str) -> list[Link]:
        links = self._read_json("links.json")
        return [
            Link(**l) for l in links.values()
            if l.get("from_id") == work_item_id or l.get("to_id") == work_item_id
        ]

    # --- AgentSession Operations ---

    def create_session(self, session: AgentSession) -> AgentSession:
        sessions = self._read_json("sessions.json")
        sessions[session.id] = session.model_dump()
        self._write_json("sessions.json", sessions)
        return session

    def get_session(self, session_id: str) -> AgentSession | None:
        sessions = self._read_json("sessions.json")
        if session_id in sessions:
            return AgentSession(**sessions[session_id])
        return None

    def update_session(self, session: AgentSession) -> AgentSession:
        sessions = self._read_json("sessions.json")
        sessions[session.id] = session.model_dump()
        self._write_json("sessions.json", sessions)
        return session

    # --- AgentActivity Operations ---

    def log_activity(self, activity: AgentActivity) -> AgentActivity:
        activities = self._read_json("activities.json")
        activities[activity.id] = activity.model_dump()
        self._write_json("activities.json", activities)
        return activity

    def get_activities(self, session_id: str) -> list[AgentActivity]:
        activities = self._read_json("activities.json")
        result = [
            AgentActivity(**a) for a in activities.values()
            if a.get("session_id") == session_id
        ]
        result.sort(key=lambda x: x.timestamp)
        return result

    # --- Feedback Operations ---

    def record_feedback(self, feedback: FeedbackEvent) -> FeedbackEvent:
        feedbacks = self._read_json("feedback.json")
        feedbacks.append(feedback.model_dump())
        self._write_json("feedback.json", feedbacks)
        return feedback

    # --- Context Management ---

    def get_current_work_item_id(self) -> str | None:
        context = self._read_json("context.json")
        return context.get("current_work_item_id")

    def set_current_work_item_id(self, item_id: str | None) -> None:
        context = self._read_json("context.json")
        context["current_work_item_id"] = item_id
        self._write_json("context.json", context)

