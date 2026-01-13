"""Configuration for Context Engineering layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ContextEngineeringConfig:
    """Unified configuration for context engineering.

    Combines settings from CompactionConfig and SummarizationMiddleware
    to provide a single configuration point.

    Attributes:
        workspace_dir: Root directory for storing artifacts and metadata.
        model_context_limit: Maximum context tokens for the model.
        summarization_trigger_ratio: Trigger summarization at this % of context limit.
        summarization_keep_messages: Number of recent messages to keep after summarization.
        mask_tool_output_if_chars_gt: Mask tool outputs larger than this.
        keep_last_unmasked_tool_outputs: Keep last N outputs unmasked.
        retrieval_top_k: Number of results to retrieve from index.
        enable_metrics: Whether to collect metrics.
    """

    # Workspace
    workspace_dir: Path | None = None

    # Model limits
    model_context_limit: int = 128_000
    """Maximum context window for the model."""

    # Summarization settings
    summarization_trigger_ratio: float = 0.85
    """Trigger summarization when context exceeds this ratio of limit."""

    summarization_keep_messages: int = 10
    """Number of recent messages to keep after summarization."""

    # Compaction settings
    mask_tool_output_if_chars_gt: int = 8000
    """Mask tool outputs larger than this character count."""

    keep_last_unmasked_tool_outputs: int = 5
    """Keep last N tool outputs unmasked in conversation."""

    # Retrieval
    retrieval_top_k: int = 8
    """Number of top results to retrieve from artifact index."""

    retrieval_backend: Literal["sqlite_fts", "bm25"] = "sqlite_fts"
    """Which retrieval backend to use."""

    # Metrics
    enable_metrics: bool = True
    """Whether to collect and log metrics."""

    # Token budgets per block type
    compressed_snippets_token_budget: int = 800
    reasoning_state_token_budget: int = 600
    decision_ledger_token_budget: int = 400

    def get_workspace_dir(self) -> Path:
        """Get workspace directory, creating temp if needed."""
        if self.workspace_dir is None:
            import tempfile
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="deepagents_context_"))
        return self.workspace_dir

    def get_artifacts_dir(self) -> Path:
        """Get artifacts subdirectory."""
        artifacts_dir = self.get_workspace_dir() / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir

    def get_summarization_trigger_tokens(self) -> int:
        """Get token count that triggers summarization."""
        return int(self.model_context_limit * self.summarization_trigger_ratio)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4

