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
from typing import Any, Callable, Awaitable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ModelCallResult,
)
from langchain_core.messages import SystemMessage

from deepagents.middleware._utils import append_to_system_message

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


class TurnContextMiddleware(AgentMiddleware[AgentState, Any]):
    """Middleware that injects identity and memory context on every turn."""

    def __init__(
        self,
        identity_loader: IdentityLoader | None = None,
        memory_search_fn: Callable | None = None,
        memory_char_budget: int = 2000,
    ) -> None:
        self.tools = []  # Required by AgentMiddleware
        self._identity = identity_loader or IdentityLoader()
        self._memory_search = memory_search_fn
        self._memory_budget = memory_char_budget

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
        parts: list[str] = []

        # 1. Identity
        try:
            identity_text = self._identity.build_system_prompt()
            if identity_text:
                parts.append(identity_text)
        except Exception:
            logger.debug("Identity loading failed", exc_info=True)

        # 2. Memory recall
        user_text = self._extract_latest_user_message(
            request.state.get("messages", [])
        )
        if user_text and self._memory_search:
            memory_text = self._recall_memory(user_text)
            if memory_text:
                parts.append(memory_text)

        # 3. Environment block
        parts.append(self._environment_block())

        if not parts:
            return request

        context = "\n\n".join(parts)
        new_sys = append_to_system_message(request.system_message, context)
        return request.override(system_message=new_sys)

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
