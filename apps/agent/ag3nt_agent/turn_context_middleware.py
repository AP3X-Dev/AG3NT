"""TurnContextMiddleware — injects identity and memory context every turn.

On every model call:
1. Loads identity from ~/.ag3nt/ identity files
2. Searches memory with the latest user message
3. Prepends both to the system prompt
"""
from __future__ import annotations

import logging
import platform
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Awaitable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ModelCallResult,
)
from langchain_core.messages import SystemMessage

from deepagents.middleware._utils import append_to_system_message, append_cached_text

from ag3nt_agent.identity import IdentityLoader

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "this", "that", "are", "was",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "not", "no", "so", "if",
    "then", "than", "when", "what", "which", "who", "how", "where", "why",
    "all", "each", "every", "both", "few", "more", "most", "some", "any",
    "just", "about", "into", "over", "after", "before", "between",
    "through", "during", "without", "also", "very", "too", "only",
    "please", "help", "need", "want", "like", "know", "think", "make",
    "get", "got", "use", "using", "used",
})


class PromptMode(str, Enum):
    """Controls how much context is injected into the system prompt.

    FULL: identity + memory + ui_context + environment + summary guardrail
    MINIMAL: environment block only (agent ID, platform, time)
    NONE: no context injected — request returned unchanged
    """

    FULL = "full"
    MINIMAL = "minimal"
    NONE = "none"


class TurnContextMiddleware(AgentMiddleware[AgentState, Any]):
    """Middleware that injects identity and memory context on every turn."""

    def __init__(
        self,
        identity_loader: IdentityLoader | None = None,
        memory_search_fn: Callable | None = None,
        memory_char_budget: int = 2000,
        prompt_mode: PromptMode = PromptMode.FULL,
        skills_metadata_fn: Callable[[], dict[str, dict]] | None = None,
        context_budget_fn: Callable[[], str] | None = None,
    ) -> None:
        self.tools = []  # Required by AgentMiddleware
        self._identity = identity_loader or IdentityLoader()
        self._memory_search = memory_search_fn
        self._memory_budget = memory_char_budget
        self.prompt_mode = prompt_mode
        self._skills_metadata_fn = skills_metadata_fn
        self._context_budget_fn = context_budget_fn

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        request = self._inject_context(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request = self._inject_context(request)
        return await handler(request)

    def _inject_context(self, request: ModelRequest) -> ModelRequest:
        # NONE mode: return request completely unchanged
        if self.prompt_mode == PromptMode.NONE:
            return request

        # MINIMAL mode: environment block only
        if self.prompt_mode == PromptMode.MINIMAL:
            context = self._environment_block()
            context = context.encode("utf-8", errors="replace").decode("utf-8")
            new_sys = append_to_system_message(request.system_message, context)
            return request.override(system_message=new_sys)

        # FULL mode: identity + memory + ui_context + environment + summary guardrail
        #
        # Stable blocks (identity, skills) use append_cached_text so that
        # Anthropic prompt caching (via PromptCachingMiddleware) can re-use
        # them across turns.  Volatile blocks (memory, environment, etc.)
        # use the regular append_to_system_message without cache_control.

        sys_msg = request.system_message

        # 1. Identity — stable across turns → cached
        try:
            identity_text = self._identity.build_system_prompt()
            if identity_text:
                identity_text = identity_text.encode("utf-8", errors="replace").decode("utf-8")
                sys_msg = append_cached_text(sys_msg, identity_text, cache=True)
        except Exception:
            logger.debug("Identity loading failed", exc_info=True)

        # 2. Skills manifest — stable across turns → cached
        if self._skills_metadata_fn:
            try:
                manifest = self._build_skills_manifest(self._skills_metadata_fn())
                if manifest:
                    manifest = manifest.encode("utf-8", errors="replace").decode("utf-8")
                    sys_msg = append_cached_text(sys_msg, manifest, cache=True)
            except Exception:
                logger.debug("Skills manifest loading failed", exc_info=True)

        # 3. Memory recall — changes per turn → NOT cached
        volatile_parts: list[str] = []
        user_text = self._extract_latest_user_message(
            request.state.get("messages", [])
        )
        if user_text and self._memory_search:
            memory_text = self._recall_memory(user_text)
            if memory_text:
                volatile_parts.append(memory_text)

        # 4. UI context (passed via config metadata by the daemon) → NOT cached
        try:
            from langgraph.config import get_config
            cfg = get_config()
            ui_ctx = (cfg.get("metadata") or {}).get("ui_context")
            if ui_ctx:
                volatile_parts.append(f"## UI Context\n{ui_ctx}")
        except Exception:
            logger.debug("UI context not available", exc_info=True)

        # 5. Environment block — has timestamp, changes per turn → NOT cached
        volatile_parts.append(self._environment_block())

        # 6. Context budget report (only when usage is elevated) → NOT cached
        if self._context_budget_fn:
            try:
                report = self._context_budget_fn()
                if report:
                    volatile_parts.append(f"## Context Budget\n{report}")
            except Exception:
                logger.debug("Context budget report failed", exc_info=True)

        # 7. Summary guardrail → NOT cached (small, volatile-adjacent)
        volatile_parts.append(
            "## Context Handling\n"
            "If a conversation summary appears in the message history, "
            "use it silently for context continuity. Never repeat, describe, "
            "or reference the summary content in your response."
        )

        if volatile_parts:
            volatile_text = "\n\n".join(volatile_parts)
            # Strip unpaired surrogates that break UTF-8 encoding in the LLM API call
            volatile_text = volatile_text.encode("utf-8", errors="replace").decode("utf-8")
            sys_msg = append_to_system_message(sys_msg, volatile_text)

        if sys_msg is request.system_message:
            return request

        return request.override(system_message=sys_msg)

    def _recall_memory(self, user_text: str) -> str:
        topics = self._extract_topics(user_text)
        query = " ".join(topics) if topics else user_text
        try:
            result = self._memory_search(query, top_k=5)
            results = result.get("results", []) if isinstance(result, dict) else []
            return self._format_memory_context(results, self._memory_budget)
        except Exception:
            logger.debug("Memory search failed", exc_info=True)
            return ""

    @staticmethod
    def _extract_latest_user_message(messages: list) -> str:
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                content = msg.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
        return ""

    @staticmethod
    def _extract_topics(text: str) -> list[str]:
        words = text.lower().split()
        topics = []
        for w in words:
            cleaned = w.strip(".,;:!?\"'()[]{}—-")
            if len(cleaned) > 4 and cleaned not in _STOPWORDS:
                topics.append(cleaned)
        return topics

    @staticmethod
    def _format_memory_context(results: list[dict], max_chars: int) -> str:
        if not results:
            return ""
        lines: list[str] = []
        chars_used = 0
        for item in results:
            content = item.get("content", "")
            line = f"- {content}"
            if chars_used + len(line) > max_chars:
                break
            lines.append(line)
            chars_used += len(line)
        if not lines:
            return ""
        return "## Relevant Memories\n" + "\n".join(lines)

    @staticmethod
    def _environment_block() -> str:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return f"## Environment\nPlatform: {platform.system()}\nCurrent time: {now}"

    def _build_skills_manifest(self, skills_metadata: dict[str, dict]) -> str:
        """Build a compact skills manifest for the system prompt.

        Args:
            skills_metadata: Dict mapping skill name to metadata dict
                             (must have 'name' and 'description' keys).

        Returns:
            Formatted manifest string, or empty string if no skills or
            prompt mode is not FULL.
        """
        if self.prompt_mode != PromptMode.FULL:
            return ""
        if not skills_metadata:
            return ""
        lines = ["## Available Skills"]
        for meta in skills_metadata.values():
            name = meta.get("name", "unknown")
            desc = meta.get("description", "")
            lines.append(f"- **{name}**: {desc}")
        return "\n".join(lines)
