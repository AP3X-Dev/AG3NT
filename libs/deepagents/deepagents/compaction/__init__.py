"""Context Window Compaction System for DeepAgents.

This module provides a production-ready context window compaction system that:
- Prevents context blowups during web browsing and multi-step tasks
- Preserves faithfulness by never losing access to raw evidence
- Provides deterministic compaction behaviors that are testable
- Makes compaction configurable per agent and per tool
- Integrates cleanly with DeepAgents patterns

Core components:
- ArtifactStore: Persists and retrieves artifacts (HTML, PDF, text, etc.)
- ObservationMasker: Replaces large tool outputs with compact placeholders
- ReasoningStateSummarizer: Periodic structured summarization
- RetrievalIndex: On-demand retrieval of relevant snippets
- ContextAssembler: Builds final prompt context with budgets

Usage:
    from deepagents.compaction import CompactionMiddleware, CompactionConfig

    middleware = CompactionMiddleware(
        config=CompactionConfig(
            mask_tool_output_if_chars_gt=6000,
            keep_last_unmasked_tool_outputs=3,
        )
    )
    agent = create_deep_agent(middleware=[middleware])
"""

from deepagents.compaction.artifact_store import ArtifactStore
from deepagents.compaction.assembler import AssembledContext, ContextAssembler, ContextBlock
from deepagents.compaction.config import CompactionConfig
from deepagents.compaction.middleware import CompactionMiddleware
from deepagents.compaction.models import (
    ArtifactMeta,
    EvidenceRecord,
    Finding,
    MaskedObservationPlaceholder,
    ReasoningState,
    ResearchBundle,
)
from deepagents.compaction.observation_masker import ObservationMasker
from deepagents.compaction.research_subagent import ResearchSubagentRunner
from deepagents.compaction.retrieval import RetrievalIndex, RetrievalResult
from deepagents.compaction.summarizer import ReasoningStateSummarizer

__all__ = [
    # Config
    "CompactionConfig",
    # Models
    "ArtifactMeta",
    "EvidenceRecord",
    "Finding",
    "MaskedObservationPlaceholder",
    "ReasoningState",
    "ResearchBundle",
    # Core components
    "ArtifactStore",
    "ObservationMasker",
    "ReasoningStateSummarizer",
    "RetrievalIndex",
    "RetrievalResult",
    "ContextAssembler",
    "ContextBlock",
    "AssembledContext",
    "ResearchSubagentRunner",
    "CompactionMiddleware",
]
