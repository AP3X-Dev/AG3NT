"""ContextEngineeringMiddleware - Unified context management.

This middleware replaces both CompactionMiddleware and SummarizationMiddleware
with a coordinated layer that:
1. Masks large tool outputs as artifacts
2. Tracks token budgets across components
3. Triggers summarization when context exceeds thresholds
4. Provides artifact retrieval tools
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from deepagents.context_engineering.budget import TokenBudgetTracker
from deepagents.context_engineering.config import ContextEngineeringConfig

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# System prompt for context engineering
CONTEXT_ENGINEERING_PROMPT = """## Context Management

You have access to artifact management tools for handling large outputs:
- Large tool outputs are automatically saved as artifacts with placeholders
- Use `read_artifact` to retrieve full content when needed
- Use `search_artifacts` to find relevant artifacts by keyword

Token budget is actively tracked. When the assistant indicates context is getting
full, consider summarizing your findings before proceeding."""


class ContextEngineeringMiddleware(AgentMiddleware):
    """Unified middleware for context engineering.

    Combines artifact masking (from CompactionMiddleware) with coordinated
    summarization triggers. This replaces both CompactionMiddleware and
    the need for SummarizationMiddleware.
    """

    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        config: ContextEngineeringConfig | None = None,
        workspace_dir: Any | None = None,
    ) -> None:
        """Initialize context engineering middleware.

        Args:
            model: Model for summarization (optional, for future use).
            config: Configuration. If None, uses defaults.
            workspace_dir: Workspace directory override.
        """
        self.model = model
        self.config = config or ContextEngineeringConfig()
        if workspace_dir is not None:
            from pathlib import Path

            self.config.workspace_dir = Path(workspace_dir)

        self.budget_tracker = TokenBudgetTracker(self.config)
        self._masked_outputs: dict[str, str] = {}  # artifact_id -> content

        # Initialize compaction components
        self._init_compaction()

        # Build tools
        self.tools = self._build_tools()

    def _init_compaction(self) -> None:
        """Initialize compaction components."""
        from deepagents.compaction.artifact_store import ArtifactStore
        from deepagents.compaction.config import CompactionConfig
        from deepagents.compaction.observation_masker import ObservationMasker
        from deepagents.compaction.retrieval import RetrievalIndex

        # Create CompactionConfig from our unified config
        compaction_config = CompactionConfig(
            workspace_dir=self.config.workspace_dir,
            mask_tool_output_if_chars_gt=self.config.mask_tool_output_if_chars_gt,
            keep_last_unmasked_tool_outputs=self.config.keep_last_unmasked_tool_outputs,
            retrieval_top_k=self.config.retrieval_top_k,
            retrieval_backend=self.config.retrieval_backend,
            enable_metrics=self.config.enable_metrics,
        )

        self.store = ArtifactStore(compaction_config)
        self.masker = ObservationMasker(store=self.store, config=compaction_config)
        self.retrieval_index = RetrievalIndex(config=compaction_config, store=self.store)

    def _build_tools(self) -> list[BaseTool]:
        """Build artifact management tools."""
        from langchain_core.tools import StructuredTool

        def read_artifact(artifact_id: str) -> str:
            """Read the full content of an artifact by ID."""
            if artifact_id in self._masked_outputs:
                return self._masked_outputs[artifact_id]
            content = self.store.read_artifact(artifact_id)
            if content is None:
                return f"Artifact '{artifact_id}' not found."
            return content.decode("utf-8", errors="replace")

        def search_artifacts(query: str, top_k: int = 5) -> str:
            """Search artifacts by keyword."""
            results = self.retrieval_index.search(query, top_k=top_k)
            if not results:
                return "No matching artifacts found."
            lines = ["Found artifacts:"]
            for r in results:
                lines.append(f"- {r.artifact_id}: {r.snippet[:100]}...")
            return "\n".join(lines)

        return [
            StructuredTool.from_function(
                func=read_artifact,
                name="read_artifact",
                description="Read full content of a masked artifact by ID.",
            ),
            StructuredTool.from_function(
                func=search_artifacts,
                name="search_artifacts",
                description="Search artifacts by keyword query.",
            ),
        ]

    def _estimate_message_tokens(self, messages: list[BaseMessage]) -> int:
        """Estimate total tokens in messages."""
        total = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total += self.config.estimate_tokens(msg.content)
        return total

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Track context and inject system prompt."""
        self.budget_tracker.increment_step()

        # Update token tracking
        msg_tokens = self._estimate_message_tokens(request.messages)
        self.budget_tracker.update_component("messages", msg_tokens)

        if request.system_prompt:
            sys_tokens = self.config.estimate_tokens(request.system_prompt)
            self.budget_tracker.update_component("system_prompt", sys_tokens)

        # Log budget if approaching limit
        if self.budget_tracker.should_summarize():
            self.budget_tracker.log_report()
            logger.warning("Context approaching limit - summarization recommended")

        # Inject system prompt
        new_prompt = request.system_prompt or ""
        new_prompt = new_prompt + "\n\n" + CONTEXT_ENGINEERING_PROMPT
        request = request.override(system_prompt=new_prompt)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        self.budget_tracker.increment_step()

        msg_tokens = self._estimate_message_tokens(request.messages)
        self.budget_tracker.update_component("messages", msg_tokens)

        if request.system_prompt:
            sys_tokens = self.config.estimate_tokens(request.system_prompt)
            self.budget_tracker.update_component("system_prompt", sys_tokens)

        if self.budget_tracker.should_summarize():
            self.budget_tracker.log_report()

        new_prompt = (request.system_prompt or "") + "\n\n" + CONTEXT_ENGINEERING_PROMPT
        request = request.override(system_prompt=new_prompt)

        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool outputs and mask large ones."""
        result = handler(request)
        return self._process_tool_result(result, request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of wrap_tool_call."""
        result = await handler(request)
        return self._process_tool_result(result, request)

    def _process_tool_result(
        self,
        result: ToolMessage | Command,
        request: ToolCallRequest,
    ) -> ToolMessage | Command:
        """Process tool result, masking if too large."""
        if isinstance(result, Command):
            return result

        if not isinstance(result, ToolMessage):
            return result

        content = result.content
        if not isinstance(content, str):
            return result

        # Check if content exceeds threshold
        if len(content) <= self.config.mask_tool_output_if_chars_gt:
            return result

        # Mask the output
        masked = self.masker.mask_if_large(
            content=content,
            tool_name=request.name,
            tool_call_id=result.tool_call_id,
        )

        if masked is None:
            return result

        # Store for later retrieval
        from deepagents.compaction.models import MaskedObservationPlaceholder

        if isinstance(masked, MaskedObservationPlaceholder):
            self._masked_outputs[masked.artifact_id] = content
            return ToolMessage(
                content=masked.to_placeholder_text(),
                tool_call_id=result.tool_call_id,
                name=result.name,
            )

        return result

    def get_budget_report(self) -> dict[str, Any]:
        """Get current token budget report as dict."""
        report = self.budget_tracker.get_report()
        return {
            "total_budget": report.total_budget,
            "used_tokens": report.used_tokens,
            "available_tokens": report.available_tokens,
            "usage_ratio": report.usage_ratio,
            "breakdown": report.breakdown,
            "summarization_recommended": report.summarization_recommended,
            "step_count": self.budget_tracker.step_count,
        }
