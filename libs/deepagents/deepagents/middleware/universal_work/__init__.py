"""Universal Work System - Persistent work management middleware.

A drop-in replacement for TodoListMiddleware that provides:
- Persistent WorkItem and PlanStep storage
- Backward-compatible write_todos and read_todos tools
- Work intake, triage, and linking capabilities
- Agent session tracking and activity logging
- Feedback collection for continuous improvement

Example:
    ```python
    from deepagents.middleware.universal_work import UniversalWorkMiddleware
    from langchain.agents import create_agent

    # Drop-in replacement for TodoListMiddleware
    agent = create_agent(
        model="anthropic:claude-sonnet-4-20250514",
        middleware=[UniversalWorkMiddleware()],
    )
    ```
"""

from deepagents.middleware.universal_work.middleware import (
    UniversalWorkMiddleware,
)
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
    SessionState,
    SuggestionType,
    TriageSuggestion,
    TriageSuggestionBundle,
    WorkItem,
    WorkItemStatus,
)
from deepagents.middleware.universal_work.retrieval import (
    DuplicateReranker,
    RelatedReranker,
    Reranker,
    RetrievalBackend,
    RetrievalCandidate,
    SimpleKeywordRetrieval,
    TriageEngine,
)
from deepagents.middleware.universal_work.storage import (
    FileBackendStorage,
    WorkStorageProtocol,
)

__all__ = [
    # Main middleware
    "UniversalWorkMiddleware",
    # Models
    "WorkItem",
    "WorkItemStatus",
    "OwnerType",
    "PlanStep",
    "PlanStepStatus",
    "Link",
    "LinkType",
    "AgentSession",
    "SessionState",
    "AgentActivity",
    "ActivityType",
    "FeedbackEvent",
    "SuggestionType",
    "TriageSuggestion",
    "TriageSuggestionBundle",
    # Storage
    "WorkStorageProtocol",
    "FileBackendStorage",
    # Retrieval
    "RetrievalBackend",
    "RetrievalCandidate",
    "SimpleKeywordRetrieval",
    "Reranker",
    "DuplicateReranker",
    "RelatedReranker",
    "TriageEngine",
]

