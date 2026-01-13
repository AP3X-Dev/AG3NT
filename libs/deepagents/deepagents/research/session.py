"""Research Session for managing research workspace state.

A ResearchSession is a resumable workspace that contains:
- ArtifactStore directory and metadata ledger
- EvidenceLedger for tracking sources
- ReasoningState timeline
- Query plan and source queue
- Final ResearchBundle output
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from deepagents.compaction.artifact_store import ArtifactStore
from deepagents.compaction.config import CompactionConfig
from deepagents.compaction.models import ReasoningState, ResearchBundle
from deepagents.research.config import ResearchConfig
from deepagents.research.evidence_ledger import EvidenceLedger
from deepagents.research.models import ResearchBrief, SourceQueueItem

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SessionState(BaseModel):
    """Serializable state of a research session."""

    session_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    # Research brief
    brief: ResearchBrief | None = None

    # Progress tracking
    current_step: int = 0
    status: str = "created"  # created, running, paused, completed, failed

    # Source queue
    source_queue: list[SourceQueueItem] = Field(default_factory=list)

    # Reasoning state timeline
    reasoning_states: list[ReasoningState] = Field(default_factory=list)

    # Final output
    result_bundle: ResearchBundle | None = None

    # Metrics
    total_artifacts: int = 0
    total_bytes_persisted: int = 0
    browser_escalations: int = 0
    errors: list[str] = Field(default_factory=list)


class ResearchSession:
    """Resumable research workspace.

    A ResearchSession manages the complete state of a research task,
    including artifacts, evidence, source queue, and final results.
    Sessions are addressable by session_id and can be resumed across runs.

    Args:
        session_id: Unique identifier for the session.
        workspace_dir: Directory for session workspace.
        config: Research configuration.
        compaction_config: Optional compaction configuration override.
    """

    def __init__(
        self,
        session_id: str,
        workspace_dir: Path,
        config: ResearchConfig,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.config = config

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Initialize compaction config for this session
        self.compaction_config = compaction_config or CompactionConfig(
            workspace_dir=self.workspace_dir,
        )

        # Initialize components
        self.artifact_store = ArtifactStore(self.compaction_config)
        self.evidence_ledger = EvidenceLedger(self.workspace_dir, config)

        # Load or create state
        self._state_path = self.workspace_dir / "session_state.json"
        self._state = self._load_state()

    def _load_state(self) -> SessionState:
        """Load session state from disk or create new."""
        if self._state_path.exists():
            try:
                with open(self._state_path, encoding="utf-8") as f:
                    data = json.load(f)
                return SessionState.model_validate(data)
            except Exception as e:
                logger.warning(f"Failed to load session state: {e}")

        return SessionState(session_id=self.session_id)

    def _save_state(self) -> None:
        """Save session state to disk."""
        self._state.updated_at = _utcnow()
        with open(self._state_path, "w", encoding="utf-8") as f:
            f.write(self._state.model_dump_json(indent=2))

    @classmethod
    def create(
        cls,
        workspace_dir: Path | None = None,
        config: ResearchConfig | None = None,
        session_id: str | None = None,
    ) -> ResearchSession:
        """Create a new research session.

        Args:
            workspace_dir: Base directory for the session.
            config: Research configuration.
            session_id: Optional session ID (generated if not provided).

        Returns:
            A new ResearchSession instance.
        """
        config = config or ResearchConfig()
        session_id = session_id or f"rs_{uuid.uuid4().hex[:12]}"
        workspace_dir = workspace_dir or config.get_session_dir(session_id)

        session = cls(
            session_id=session_id,
            workspace_dir=workspace_dir,
            config=config,
        )
        session._save_state()
        logger.info(f"Created research session: {session_id}")
        return session

    @classmethod
    def load(
        cls,
        session_id: str,
        config: ResearchConfig | None = None,
    ) -> ResearchSession:
        """Load an existing research session.

        Args:
            session_id: The session ID to load.
            config: Research configuration.

        Returns:
            The loaded ResearchSession.

        Raises:
            FileNotFoundError: If the session doesn't exist.
        """
        config = config or ResearchConfig()
        workspace_dir = config.get_session_dir(session_id)

        if not workspace_dir.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        return cls(
            session_id=session_id,
            workspace_dir=workspace_dir,
            config=config,
        )

    # === Brief Management ===

    def set_brief(self, brief: ResearchBrief) -> None:
        """Set the research brief for this session."""
        self._state.brief = brief
        self._save_state()

    def get_brief(self) -> ResearchBrief | None:
        """Get the current research brief."""
        return self._state.brief

    # === Status Management ===

    @property
    def status(self) -> str:
        """Get current session status."""
        return self._state.status

    def set_status(self, status: str) -> None:
        """Set session status."""
        self._state.status = status
        self._save_state()

    @property
    def current_step(self) -> int:
        """Get current step number."""
        return self._state.current_step

    def increment_step(self) -> int:
        """Increment and return the step counter."""
        self._state.current_step += 1
        self._save_state()
        return self._state.current_step

    # === Source Queue Management ===

    def add_source(self, source: SourceQueueItem) -> None:
        """Add a source to the queue."""
        self._state.source_queue.append(source)
        self._save_state()

    def add_sources(self, sources: list[SourceQueueItem]) -> None:
        """Add multiple sources to the queue."""
        self._state.source_queue.extend(sources)
        self._save_state()

    def get_source_queue(self) -> list[SourceQueueItem]:
        """Get the current source queue."""
        return self._state.source_queue.copy()

    def get_pending_sources(self) -> list[SourceQueueItem]:
        """Get sources that haven't been processed yet."""
        from deepagents.research.models import SourceStatus

        pending = [SourceStatus.QUEUED, SourceStatus.BROWSER_NEEDED]
        return [s for s in self._state.source_queue if s.status in pending]

    def update_source(self, url: str, **updates: Any) -> bool:
        """Update a source in the queue by URL."""
        for source in self._state.source_queue:
            if source.url == url:
                for key, value in updates.items():
                    if hasattr(source, key):
                        setattr(source, key, value)
                self._save_state()
                return True
        return False

    # === Reasoning State Management ===

    def add_reasoning_state(self, state: ReasoningState) -> None:
        """Add a reasoning state snapshot."""
        self._state.reasoning_states.append(state)
        self._save_state()

    def get_latest_reasoning_state(self) -> ReasoningState | None:
        """Get the most recent reasoning state."""
        if self._state.reasoning_states:
            return self._state.reasoning_states[-1]
        return None

    def get_reasoning_timeline(self) -> list[ReasoningState]:
        """Get the full reasoning state timeline."""
        return self._state.reasoning_states.copy()

    # === Result Management ===

    def set_result(self, bundle: ResearchBundle) -> None:
        """Set the final result bundle."""
        self._state.result_bundle = bundle
        self._state.status = "completed"
        self._save_state()

    def get_result(self) -> ResearchBundle | None:
        """Get the result bundle if available."""
        return self._state.result_bundle

    # === Metrics ===

    def record_error(self, error: str) -> None:
        """Record an error."""
        self._state.errors.append(error)
        self._save_state()

    def record_browser_escalation(self) -> None:
        """Record a browser mode escalation."""
        self._state.browser_escalations += 1
        self._save_state()

    def update_artifact_metrics(self) -> None:
        """Update artifact metrics from the store."""
        self._state.total_artifacts = self.artifact_store.get_artifact_count()
        self._state.total_bytes_persisted = self.artifact_store.get_total_bytes()
        self._save_state()

    def get_metrics(self) -> dict[str, Any]:
        """Get session metrics."""
        return {
            "session_id": self.session_id,
            "status": self._state.status,
            "current_step": self._state.current_step,
            "total_sources": len(self._state.source_queue),
            "sources_processed": len([s for s in self._state.source_queue if s.status.value in ("read", "browsed", "errored", "rejected")]),
            "evidence_count": self.evidence_ledger.count(),
            "unique_domains": len(self.evidence_ledger.get_unique_domains()),
            "total_artifacts": self._state.total_artifacts,
            "total_bytes_persisted": self._state.total_bytes_persisted,
            "browser_escalations": self._state.browser_escalations,
            "error_count": len(self._state.errors),
        }
