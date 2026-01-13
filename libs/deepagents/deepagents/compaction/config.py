"""Configuration for the Context Window Compaction System."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class CompactionConfig:
    """Configuration for the compaction system.

    All thresholds are configurable to allow tuning per agent and use case.

    Attributes:
        workspace_dir: Root directory for storing artifacts and metadata.
            If None, uses a temporary directory.
        mask_tool_output_if_chars_gt: Tool outputs exceeding this character
            count are persisted as artifacts and replaced with placeholders.
        keep_last_unmasked_tool_outputs: Number of recent tool outputs to
            keep unmasked in the conversation for short-term grounding.
        summarize_every_steps: Create a reasoning state summary every N steps.
        summarize_if_estimated_context_tokens_gt: Trigger summarization if
            estimated context tokens exceed this fraction of model limit.
        model_context_limit: Maximum context tokens for the model.
        retrieval_top_k: Number of top results to retrieve from index.
        compressed_snippets_token_budget: Token budget for compressed snippets.
        reasoning_state_token_budget: Token budget for reasoning state block.
        decision_ledger_token_budget: Token budget for decision ledger block.
        working_memory_token_budget: Token budget for working memory block.
        plan_state_token_budget: Token budget for plan state block.
        recent_observations_token_budget: Token budget for recent observations.
        retrieval_backend: Which retrieval backend to use.
        enable_metrics: Whether to collect and log metrics.
        metrics_file: File to write metrics to (jsonl format).
        redact_secrets: Whether to redact obvious secrets from artifacts.
    """

    # Workspace configuration
    workspace_dir: Path | None = None

    # Masking thresholds
    mask_tool_output_if_chars_gt: int = 6000
    keep_last_unmasked_tool_outputs: int = 3

    # Summarization triggers
    summarize_every_steps: int = 8
    summarize_if_estimated_context_tokens_gt: float = 0.7
    model_context_limit: int = 128000

    # Retrieval configuration
    retrieval_top_k: int = 8
    retrieval_backend: Literal["sqlite_fts", "bm25"] = "sqlite_fts"

    # Token budgets per block type
    compressed_snippets_token_budget: int = 800
    reasoning_state_token_budget: int = 600
    decision_ledger_token_budget: int = 400
    working_memory_token_budget: int = 400
    plan_state_token_budget: int = 300
    recent_observations_token_budget: int = 1000

    # Metrics and logging
    enable_metrics: bool = True
    metrics_file: str = "compaction_metrics.jsonl"

    # Security
    redact_secrets: bool = True

    # Block priorities (lower = higher priority, included first)
    block_priorities: dict[str, int] = field(default_factory=lambda: {
        "working_memory": 1,
        "plan_state": 2,
        "decision_ledger": 3,
        "reasoning_state": 4,
        "recent_observations": 5,
        "masked_placeholders": 6,
        "retrieved_snippets": 7,
    })

    def get_workspace_dir(self) -> Path:
        """Get the workspace directory, creating a temp dir if needed."""
        if self.workspace_dir is None:
            import tempfile
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="deepagents_compaction_"))
        return self.workspace_dir

    def get_artifacts_dir(self) -> Path:
        """Get the artifacts subdirectory."""
        artifacts_dir = self.get_workspace_dir() / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir

    def get_metadata_path(self) -> Path:
        """Get the path to the metadata ledger."""
        return self.get_workspace_dir() / "artifact_metadata.jsonl"

    def get_index_path(self) -> Path:
        """Get the path to the retrieval index database."""
        return self.get_workspace_dir() / "retrieval_index.db"

    def get_metrics_path(self) -> Path:
        """Get the path to the metrics file."""
        return self.get_workspace_dir() / self.metrics_file

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Rough estimate: ~4 characters per token for English text
        return len(text) // 4

