"""Research Agents v2: Deep Research with Compaction-Native Workflows.

This module provides a next-generation research capability that runs long,
multi-step investigations in a dedicated research context while returning
only distilled, evidence-backed results to the main agent.

Key principles:
- Tool outputs do not live in the main context by default
- Evidence is stored as artifacts with stable pointers
- The main agent consumes ResearchBundles, not browsing transcripts
- All operations are compaction-native and artifact-first

Core components:
- ResearchSession: Resumable workspace with artifacts, evidence, and state
- ResearchOrchestrator: Plans queries, routes steps, enforces budgets
- SourceCollector: Generates queries, deduplicates and ranks sources
- PageReader: Fetches and normalizes content via HTTP
- BrowserOperator: Handles JS-heavy sites via browser automation
- Distiller: Produces structured findings from sources
- Reviewer: Validates evidence coverage and triggers follow-ups

Usage:
    from deepagents.research import ResearchSession, ResearchOrchestrator

    # Create a new research session
    session = ResearchSession.create(workspace_dir=Path("./research"))

    # Run research with an orchestrator
    orchestrator = ResearchOrchestrator(session=session)
    bundle = await orchestrator.research(
        goal="Find the latest pricing for AWS Lambda",
        constraints={"recency": "last_30_days"},
    )

    # Use the bundle in main agent context
    print(bundle.executive_summary)
"""

from deepagents.research.browser_operator import (
    BrowserAction,
    BrowserActionType,
    BrowserDriver,
    BrowserOperator,
    BrowserState,
    BrowserTask,
    BrowserTaskResult,
    MockBrowserDriver,
)
from deepagents.research.config import ResearchConfig
from deepagents.research.distiller import (
    Distiller,
    ExtractionResult,
    Extractor,
)
from deepagents.research.evidence_ledger import EvidenceLedger
from deepagents.research.models import (
    ResearchBrief,
    ResearchMode,
    SourceQueueItem,
    SourceReasonCode,
    SourceStatus,
)
from deepagents.research.orchestrator import ResearchOrchestrator
from deepagents.research.page_reader import PageContent, PageReader
from deepagents.research.reviewer import (
    FollowUpTask,
    Gap,
    GapType,
    Reviewer,
    ReviewResult,
    ReviewStatus,
)
from deepagents.research.session import ResearchSession
from deepagents.research.source_collector import (
    MockSearchProvider,
    SearchProvider,
    SearchResult,
    SourceCollector,
)

__all__ = [
    # Config
    "ResearchConfig",
    # Models
    "ResearchBrief",
    "ResearchMode",
    "SourceQueueItem",
    "SourceStatus",
    "SourceReasonCode",
    # Session
    "ResearchSession",
    "EvidenceLedger",
    # Orchestrator
    "ResearchOrchestrator",
    # Source Collection
    "SourceCollector",
    "SearchProvider",
    "SearchResult",
    "MockSearchProvider",
    # Page Reading
    "PageReader",
    "PageContent",
    # Browser Mode
    "BrowserOperator",
    "BrowserDriver",
    "BrowserTask",
    "BrowserTaskResult",
    "BrowserAction",
    "BrowserActionType",
    "BrowserState",
    "MockBrowserDriver",
    # Distillation
    "Distiller",
    "Extractor",
    "ExtractionResult",
    # Review
    "Reviewer",
    "ReviewResult",
    "ReviewStatus",
    "Gap",
    "GapType",
    "FollowUpTask",
]
