"""Tests for Universal Work System middleware.

Tests backward compatibility with TodoListMiddleware and new functionality.
"""

import tempfile
from pathlib import Path

import pytest

from deepagents.middleware.universal_work import (
    ActivityType,
    AgentActivity,
    AgentSession,
    FileBackendStorage,
    Link,
    LinkType,
    OwnerType,
    PlanStep,
    PlanStepStatus,
    SessionState,
    SimpleKeywordRetrieval,
    TriageEngine,
    UniversalWorkMiddleware,
    WorkItem,
    WorkItemStatus,
)


class TestWorkItemModel:
    """Tests for WorkItem model."""

    def test_work_item_creation(self) -> None:
        """Test basic WorkItem creation."""
        item = WorkItem(
            title="Test Task",
            body="This is a test task",
            domain="testing",
            labels=["test", "unit"],
        )
        assert item.title == "Test Task"
        assert item.status == WorkItemStatus.INBOX
        assert item.owner_type == OwnerType.UNASSIGNED
        assert len(item.id) > 0

    def test_work_item_status_transitions(self) -> None:
        """Test WorkItem status values."""
        item = WorkItem(title="Test")
        assert item.status == WorkItemStatus.INBOX
        
        item.status = WorkItemStatus.ACCEPTED
        assert item.status == WorkItemStatus.ACCEPTED
        
        item.status = WorkItemStatus.IN_PROGRESS
        assert item.status == WorkItemStatus.IN_PROGRESS


class TestPlanStepModel:
    """Tests for PlanStep model."""

    def test_plan_step_creation(self) -> None:
        """Test basic PlanStep creation."""
        step = PlanStep(
            work_item_id="test-item-id",
            content="Complete the task",
            position=0,
        )
        assert step.content == "Complete the task"
        assert step.status == PlanStepStatus.PENDING
        assert step.position == 0

    def test_plan_step_from_todo_dict(self) -> None:
        """Test creating PlanStep from todo dict (backward compatibility)."""
        todo = {
            "content": "Do something",
            "status": "in_progress",
            "activeForm": "Working on it",
        }
        step = PlanStep.from_todo_dict(todo, "work-item-123", position=1)
        
        assert step.content == "Do something"
        assert step.status == PlanStepStatus.IN_PROGRESS
        assert step.active_form == "Working on it"
        assert step.work_item_id == "work-item-123"
        assert step.position == 1

    def test_plan_step_to_todo_dict(self) -> None:
        """Test converting PlanStep to todo dict (backward compatibility)."""
        step = PlanStep(
            work_item_id="test-id",
            content="Test task",
            status=PlanStepStatus.COMPLETED,
            active_form="Done",
            position=0,
        )
        todo = step.to_todo_dict()
        
        assert todo["content"] == "Test task"
        assert todo["status"] == "completed"
        assert todo["activeForm"] == "Done"


class TestFileBackendStorage:
    """Tests for FileBackendStorage."""

    def test_storage_initialization(self) -> None:
        """Test storage creates directory and files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)
            
            assert (Path(tmpdir) / "work_items.json").exists()
            assert (Path(tmpdir) / "plan_steps.json").exists()
            assert (Path(tmpdir) / "context.json").exists()

    def test_work_item_crud(self) -> None:
        """Test WorkItem CRUD operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)
            
            # Create
            item = WorkItem(title="Test Item", body="Test body")
            created = storage.create_work_item(item)
            assert created.id == item.id
            
            # Read
            retrieved = storage.get_work_item(item.id)
            assert retrieved is not None
            assert retrieved.title == "Test Item"
            
            # Update
            retrieved.status = WorkItemStatus.IN_PROGRESS
            updated = storage.update_work_item(retrieved)
            assert updated.status == WorkItemStatus.IN_PROGRESS
            
            # List
            items = storage.list_work_items()
            assert len(items) == 1
            assert items[0].id == item.id

    def test_plan_step_operations(self) -> None:
        """Test PlanStep operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)
            
            # Create work item first
            item = WorkItem(title="Test")
            storage.create_work_item(item)
            
            # Create plan steps
            steps = [
                PlanStep(work_item_id=item.id, content="Step 1", position=0),
                PlanStep(work_item_id=item.id, content="Step 2", position=1),
            ]
            storage.replace_plan_steps(item.id, steps)
            
            # Retrieve
            retrieved = storage.get_plan_steps(item.id)
            assert len(retrieved) == 2
            assert retrieved[0].content == "Step 1"
            assert retrieved[1].content == "Step 2"

    def test_context_management(self) -> None:
        """Test current work item context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)

            assert storage.get_current_work_item_id() is None

            storage.set_current_work_item_id("test-id")
            assert storage.get_current_work_item_id() == "test-id"

            storage.set_current_work_item_id(None)
            assert storage.get_current_work_item_id() is None


class TestLinkOperations:
    """Tests for Link operations."""

    def test_link_creation(self) -> None:
        """Test creating links between work items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)

            # Create two work items
            item1 = WorkItem(title="Item 1")
            item2 = WorkItem(title="Item 2")
            storage.create_work_item(item1)
            storage.create_work_item(item2)

            # Create link
            link = Link(
                from_id=item1.id,
                to_id=item2.id,
                link_type=LinkType.RELATED_TO,
                confidence=0.8,
            )
            created = storage.create_link(link)
            assert created.id == link.id

            # Retrieve links
            links1 = storage.get_links(item1.id)
            links2 = storage.get_links(item2.id)

            assert len(links1) == 1
            assert len(links2) == 1
            assert links1[0].link_type == LinkType.RELATED_TO


class TestAgentSessionOperations:
    """Tests for AgentSession operations."""

    def test_session_lifecycle(self) -> None:
        """Test agent session creation and updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)

            # Create work item
            item = WorkItem(title="Test Task")
            storage.create_work_item(item)

            # Create session
            session = AgentSession(
                agent_id="test-agent",
                work_item_id=item.id,
            )
            created = storage.create_session(session)
            assert created.state == SessionState.ACTIVE

            # Retrieve
            retrieved = storage.get_session(session.id)
            assert retrieved is not None
            assert retrieved.agent_id == "test-agent"

            # Update
            retrieved.state = SessionState.COMPLETED
            updated = storage.update_session(retrieved)
            assert updated.state == SessionState.COMPLETED

    def test_activity_logging(self) -> None:
        """Test logging activities in a session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)

            # Create session
            session = AgentSession(agent_id="test", work_item_id="item-1")
            storage.create_session(session)

            # Log activities
            activity1 = AgentActivity(
                session_id=session.id,
                activity_type=ActivityType.STARTED,
                summary="Started working",
            )
            activity2 = AgentActivity(
                session_id=session.id,
                activity_type=ActivityType.STEP_COMPLETED,
                summary="Completed step 1",
            )
            storage.log_activity(activity1)
            storage.log_activity(activity2)

            # Retrieve
            activities = storage.get_activities(session.id)
            assert len(activities) == 2
            assert activities[0].activity_type == ActivityType.STARTED


class TestTriageEngine:
    """Tests for triage and retrieval."""

    def test_keyword_retrieval(self) -> None:
        """Test keyword-based retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)
            retrieval = SimpleKeywordRetrieval(storage)

            # Create items
            item1 = WorkItem(title="Fix login bug", body="Users cannot login")
            item2 = WorkItem(title="Add logout feature", body="Need logout button")
            item3 = WorkItem(title="Update database", body="Migrate to new schema")

            storage.create_work_item(item1)
            storage.create_work_item(item2)
            storage.create_work_item(item3)

            # Index items
            retrieval.rebuild_index()

            # Search
            results = retrieval.search("login authentication", limit=5)

            # Should find login-related items
            assert len(results) > 0
            item_ids = [r.item.id for r in results]
            assert item1.id in item_ids

    def test_triage_suggestions(self) -> None:
        """Test triage suggestion generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileBackendStorage(tmpdir)
            engine = TriageEngine(storage)

            # Create similar items
            item1 = WorkItem(
                title="Fix login page bug",
                body="Users cannot login to the application",
                domain="auth",
            )
            item2 = WorkItem(
                title="Login page not working",
                body="Authentication fails for users",
                domain="auth",
            )
            item3 = WorkItem(
                title="Update database schema",
                body="Need to migrate tables",
                domain="database",
            )

            storage.create_work_item(item1)
            storage.create_work_item(item2)
            storage.create_work_item(item3)

            # Generate suggestions for item2
            bundle = engine.generate_suggestions(item2.id, modes=["duplicates", "related"])

            assert bundle.work_item_id == item2.id
            # item1 should be suggested as duplicate or related
            all_suggestions = bundle.duplicates + bundle.related
            suggested_ids = [s.suggested_value for s in all_suggestions]
            assert item1.id in suggested_ids


class TestUniversalWorkMiddleware:
    """Tests for UniversalWorkMiddleware."""

    def test_middleware_initialization(self) -> None:
        """Test middleware initializes with all tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            middleware = UniversalWorkMiddleware(storage_path=tmpdir)

            tool_names = [t.name for t in middleware.tools]

            # Check backward-compatible tools
            assert "write_todos" in tool_names
            assert "read_todos" in tool_names

            # Check new tools
            assert "work_item_create" in tool_names
            assert "work_item_get" in tool_names
            assert "inbox_list" in tool_names
            assert "link_create" in tool_names
            assert "triage_suggest" in tool_names

    def test_middleware_enabled_tools_filter(self) -> None:
        """Test middleware respects enabled_tools filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            middleware = UniversalWorkMiddleware(
                storage_path=tmpdir,
                enabled_tools=["write_todos", "read_todos"],
            )

            tool_names = [t.name for t in middleware.tools]

            assert len(tool_names) == 2
            assert "write_todos" in tool_names
            assert "read_todos" in tool_names
            assert "work_item_create" not in tool_names

