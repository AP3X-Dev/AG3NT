"""CompactionMiddleware for integrating context window compaction with DeepAgents.

This middleware:
1. Intercepts tool outputs and masks large ones as artifacts
2. Provides tools for artifact management and retrieval
3. Tracks evidence and maintains the evidence ledger
4. Integrates with the agent loop for periodic summarization
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from deepagents.compaction.artifact_store import ArtifactStore
from deepagents.compaction.assembler import ContextAssembler
from deepagents.compaction.config import CompactionConfig
from deepagents.compaction.models import MaskedObservationPlaceholder
from deepagents.compaction.observation_masker import ObservationMasker
from deepagents.compaction.retrieval import RetrievalIndex
from deepagents.compaction.summarizer import ReasoningStateSummarizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

COMPACTION_SYSTEM_PROMPT = """## Artifact Management

You have access to an artifact storage system for managing large content:

- **save_artifact**: Save important content (research findings, extracted data, quotes) for later retrieval
- **read_artifact**: Read content from a stored artifact by ID
- **search_artifacts**: Search stored artifacts by tool name, tags, or URL
- **retrieve_snippets**: Find specific information within a large artifact using a query

When you see a [MASKED OUTPUT] placeholder, the full content has been stored as an artifact.
Use retrieve_snippets to find specific information without loading the entire content.
"""


class CompactionMiddleware(AgentMiddleware):
    """Middleware for context window compaction.

    This middleware automatically:
    - Masks large tool outputs as artifacts with placeholders
    - Provides tools for artifact retrieval and search
    - Tracks evidence sources for factual claims
    - Collects metrics on compaction behavior

    Args:
        config: Compaction configuration. If None, uses defaults.
        workspace_dir: Optional workspace directory override.

    Example:
        ```python
        from deepagents.compaction import CompactionMiddleware, CompactionConfig

        middleware = CompactionMiddleware(
            config=CompactionConfig(
                mask_tool_output_if_chars_gt=6000,
                keep_last_unmasked_tool_outputs=3,
            )
        )
        agent = create_deep_agent(middleware=[middleware])
        ```
    """

    def __init__(
        self,
        config: CompactionConfig | None = None,
        workspace_dir: Path | str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.config = config or CompactionConfig()
        if workspace_dir:
            self.config.workspace_dir = Path(workspace_dir)

        self._custom_system_prompt = system_prompt
        self.store = ArtifactStore(self.config)
        self.masker = ObservationMasker(store=self.store, config=self.config)
        self.summarizer = ReasoningStateSummarizer(config=self.config, masker=self.masker)
        self.retrieval_index = RetrievalIndex(config=self.config, store=self.store)
        self.assembler = ContextAssembler(config=self.config)

        # Metrics tracking
        self._step_count = 0
        self._metrics: list[dict[str, Any]] = []
        self._step_start_time: float | None = None

        # Build tools
        self.tools = self._build_tools()

    def _build_tools(self) -> list[BaseTool]:
        """Build the compaction-related tools."""
        return [
            self._build_save_artifact_tool(),
            self._build_read_artifact_tool(),
            self._build_search_artifacts_tool(),
            self._build_retrieve_snippets_tool(),
        ]

    def _build_save_artifact_tool(self) -> BaseTool:
        """Build the save_artifact tool."""
        def save_artifact(
            content: str,
            runtime: ToolRuntime,
            *,
            title: str | None = None,
            source_url: str | None = None,
            tags: str | None = None,
        ) -> str:
            """Save content as a persistent artifact.

            Use this to save important content that may be needed later,
            such as research findings, extracted data, or important quotes.

            Args:
                content: The content to save.
                title: Optional title for the artifact.
                source_url: Optional source URL.
                tags: Optional comma-separated tags.

            Returns:
                Confirmation with artifact ID.
            """
            tag_list = [t.strip() for t in tags.split(",")] if tags else []
            artifact_id, path = self.store.write_artifact(
                content,
                tool_name="save_artifact",
                source_url=source_url,
                title=title,
                tags=tag_list,
            )
            return f"Saved artifact {artifact_id} ({len(content):,} chars)"

        return StructuredTool.from_function(
            name="save_artifact",
            description="Save content as a persistent artifact for later retrieval.",
            func=save_artifact,
        )

    def _build_read_artifact_tool(self) -> BaseTool:
        """Build the read_artifact tool."""
        def read_artifact(
            artifact_id: str,
            runtime: ToolRuntime,
            *,
            offset: int = 0,
            limit: int = 500,
        ) -> str:
            """Read content from a stored artifact.

            Args:
                artifact_id: The artifact ID to read.
                offset: Line offset to start reading from.
                limit: Maximum number of lines to return.

            Returns:
                The artifact content (or portion thereof).
            """
            content = self.store.read_artifact(artifact_id)
            if content is None:
                return f"Error: Artifact {artifact_id} not found"

            if isinstance(content, bytes):
                return f"Error: Artifact {artifact_id} is binary, cannot display as text"

            # Apply offset and limit
            lines = content.split("\n")
            selected = lines[offset:offset + limit]
            result = "\n".join(selected)

            if offset > 0 or offset + limit < len(lines):
                result += f"\n\n[Showing lines {offset+1}-{min(offset+limit, len(lines))} of {len(lines)}]"

            return result

        return StructuredTool.from_function(
            name="read_artifact",
            description="Read content from a stored artifact by ID.",
            func=read_artifact,
        )

    def _build_search_artifacts_tool(self) -> BaseTool:
        """Build the search_artifacts tool."""
        def search_artifacts(
            runtime: ToolRuntime,
            *,
            tool_name: str | None = None,
            tags: str | None = None,
            url_contains: str | None = None,
            limit: int = 20,
        ) -> str:
            """Search stored artifacts by filters.

            Args:
                tool_name: Filter by the tool that created the artifact.
                tags: Comma-separated tags to filter by.
                url_contains: Filter by source URL substring.
                limit: Maximum number of results.

            Returns:
                List of matching artifacts with their IDs and metadata.
            """
            tag_list = [t.strip() for t in tags.split(",")] if tags else None
            results = self.store.list_artifacts(
                tool_name=tool_name,
                tags=tag_list,
                source_url_contains=url_contains,
                limit=limit,
            )

            if not results:
                return "No artifacts found matching the criteria."

            lines = [f"Found {len(results)} artifact(s):\n"]
            for meta in results:
                title = meta.title or "(untitled)"
                url = f" | URL: {meta.source_url}" if meta.source_url else ""
                lines.append(
                    f"- {meta.artifact_id}: {title} | {meta.tool_name} | "
                    f"{meta.size_bytes:,} bytes{url}"
                )
            return "\n".join(lines)

        return StructuredTool.from_function(
            name="search_artifacts",
            description="Search stored artifacts by tool name, tags, or URL.",
            func=search_artifacts,
        )

    def _build_retrieve_snippets_tool(self) -> BaseTool:
        """Build the retrieve_snippets tool."""
        def retrieve_snippets(
            query: str,
            runtime: ToolRuntime,
            *,
            artifact_id: str | None = None,
            max_snippets: int = 5,
        ) -> str:
            """Retrieve relevant snippets from artifacts based on a query.

            Use this to find specific information within stored artifacts
            without loading the entire content. Uses full-text search.

            Args:
                query: The search query to find relevant content.
                artifact_id: Optional artifact ID to limit search to.
                max_snippets: Maximum number of snippets to return.

            Returns:
                Relevant snippets from matching artifacts.
            """
            # Ensure artifact is indexed if specified
            if artifact_id:
                self.retrieval_index.index_artifact(artifact_id)
            else:
                # Index any new artifacts
                self.retrieval_index.index_all_artifacts()

            # Search using FTS
            results = self.retrieval_index.search_with_context(
                query,
                artifact_id=artifact_id,
                top_k=max_snippets,
            )

            if not results:
                target = f"artifact {artifact_id}" if artifact_id else "any artifacts"
                return f"No snippets matching '{query}' found in {target}"

            # Build result
            result_lines = [f"Found {len(results)} snippet(s) matching '{query}':\n"]
            for result in results:
                preview = result.snippet[:400] + "..." if len(result.snippet) > 400 else result.snippet
                result_lines.append(
                    f"[{result.artifact_id} | Line {result.line_number} | Score: {result.score:.2f}]"
                )
                result_lines.append(preview)
                result_lines.append("")

            return "\n".join(result_lines)

        return StructuredTool.from_function(
            name="retrieve_snippets",
            description="Retrieve relevant snippets from an artifact based on a query.",
            func=retrieve_snippets,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool outputs and mask large ones.

        Args:
            request: The tool call request.
            handler: The handler to call.

        Returns:
            The tool result, possibly with masked content.
        """
        result = handler(request)
        return self._process_tool_result(result, request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """(Async) Intercept tool outputs and mask large ones."""
        result = await handler(request)
        return self._process_tool_result(result, request)

    def _process_tool_result(
        self,
        result: ToolMessage | Command,
        request: ToolCallRequest,
    ) -> ToolMessage | Command:
        """Process a tool result and mask if necessary."""
        # Skip our own tools
        tool_name = request.tool_call.get("name", "")
        if tool_name in {"save_artifact", "read_artifact", "search_artifacts", "retrieve_snippets"}:
            return result

        # Handle Command objects
        if isinstance(result, Command):
            return result

        # Check if content should be masked
        content = result.content
        if not isinstance(content, str):
            return result

        masked = self.masker.mask_observation(
            tool_call_id=result.tool_call_id,
            tool_name=tool_name,
            content=content,
        )

        if isinstance(masked, MaskedObservationPlaceholder):
            # Return modified tool message with placeholder
            return ToolMessage(
                content=masked.to_placeholder_text(),
                tool_call_id=result.tool_call_id,
                name=result.name,
            )

        return result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Track model calls for metrics and step counting."""
        self._step_count += 1
        self._step_start_time = time.time()

        # Add tools and system prompt to the request
        request = self._add_tools_to_request(request)
        request = self._add_system_prompt(request)

        result = handler(request)

        # Record metrics
        self._record_step_metrics()

        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(Async) Track model calls for metrics and step counting."""
        self._step_count += 1
        self._step_start_time = time.time()

        request = self._add_tools_to_request(request)
        request = self._add_system_prompt(request)

        result = await handler(request)

        self._record_step_metrics()

        return result

    def _add_system_prompt(self, request: ModelRequest) -> ModelRequest:
        """Add compaction system prompt to the request."""
        system_prompt = self._custom_system_prompt or COMPACTION_SYSTEM_PROMPT

        if system_prompt:
            new_prompt = (
                request.system_prompt + "\n\n" + system_prompt
                if request.system_prompt
                else system_prompt
            )
            return request.override(system_prompt=new_prompt)

        return request

    def _add_tools_to_request(self, request: ModelRequest) -> ModelRequest:
        """Add compaction tools to the request."""
        existing_tools = list(request.tools) if request.tools else []
        existing_names = {t.name for t in existing_tools if hasattr(t, "name")}

        for tool in self.tools:
            if tool.name not in existing_names:
                existing_tools.append(tool)

        return request.override(tools=existing_tools)

    def _record_step_metrics(self) -> None:
        """Record metrics for the current step."""
        if not self.config.enable_metrics:
            return

        elapsed = time.time() - self._step_start_time if self._step_start_time else 0

        metrics = {
            "step": self._step_count,
            "timestamp": time.time(),
            "artifacts_count": self.store.get_artifact_count(),
            "bytes_persisted": self.store.get_total_bytes(),
            "masked_count": self.masker.get_masked_count(),
            "compaction_time_ms": int(elapsed * 1000),
        }
        self._metrics.append(metrics)

        # Write to file
        try:
            with open(self.config.get_metrics_path(), "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write metrics: {e}")

    def get_metrics(self) -> list[dict[str, Any]]:
        """Get all collected metrics."""
        return self._metrics.copy()

    def get_step_count(self) -> int:
        """Get the current step count."""
        return self._step_count

