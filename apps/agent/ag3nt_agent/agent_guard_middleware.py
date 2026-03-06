"""AgentGuardMiddleware — doom loop detection, steps limit, output truncation.

Combines three agent reliability features in a single middleware:

1. **Doom Loop Detection** (wrap_model_call): Scans message history for
   repeated identical tool calls. After DOOM_LOOP_THRESHOLD consecutive
   identical calls, injects a system prompt telling the model to stop.

2. **Steps Limit** (wrap_model_call): Counts model calls per session.
   After MAX_AGENT_STEPS, injects a system prompt disabling tool use
   and requesting a summary.

3. **Universal Output Truncation** (wrap_tool_call): Applies
   output_truncation.maybe_truncate() to every tool result, catching
   large outputs that slip past FilesystemMiddleware's eviction.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger("ag3nt.guard")

DOOM_LOOP_THRESHOLD = 3

# Module-level step counters keyed by session_id
_step_counts: dict[str, int] = {}


def _get_max_steps() -> int:
    """Load MAX_AGENT_STEPS from agent_config (lazy to avoid circular imports)."""
    try:
        from ag3nt_agent.agent_config import MAX_AGENT_STEPS
        return MAX_AGENT_STEPS
    except ImportError:
        return 75


def _detect_doom_loop(messages: list) -> tuple[bool, str | None]:
    """Check if the last N tool calls are identical.

    Scans AIMessages in reverse for tool_calls. If the last
    DOOM_LOOP_THRESHOLD calls share the same tool name AND identical
    JSON-serialized args, returns (True, tool_name).

    Returns:
        (is_looping, tool_name) — tool_name is None when not looping.
    """
    recent_calls: list[tuple[str, str]] = []  # (name, serialized_args)

    for msg in reversed(messages):
        if len(recent_calls) >= DOOM_LOOP_THRESHOLD:
            break
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in reversed(msg.tool_calls):
                if len(recent_calls) >= DOOM_LOOP_THRESHOLD:
                    break
                name = tc.get("name", "")
                args = tc.get("args", {})
                try:
                    serialized = json.dumps(args, sort_keys=True, default=str)
                except (TypeError, ValueError):
                    serialized = str(args)
                recent_calls.append((name, serialized))

    if len(recent_calls) < DOOM_LOOP_THRESHOLD:
        return False, None

    # Check if all collected calls are identical
    first = recent_calls[0]
    if all(call == first for call in recent_calls[1:]):
        return True, first[0]

    return False, None


class AgentGuardMiddleware(AgentMiddleware[AgentState, Any]):
    """Middleware for doom loop detection, steps limit, and output truncation."""

    def __init__(self) -> None:
        self.tools: list = []  # Required by AgentMiddleware interface

    @property
    def name(self) -> str:
        return "agent_guard"

    # -- Step counter management -----------------------------------------------

    @staticmethod
    def reset_steps(session_id: str) -> None:
        """Reset the step counter for a session. Call at start of run_turn()."""
        _step_counts[session_id] = 0

    @staticmethod
    def get_steps(session_id: str) -> int:
        """Get current step count for a session."""
        return _step_counts.get(session_id, 0)

    # -- Model call hooks (doom loop + steps limit) ----------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        request = self._guard_model_call(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        request = self._guard_model_call(request)
        return await handler(request)

    def _guard_model_call(self, request: ModelRequest) -> ModelRequest:
        """Apply doom loop detection and steps limit to model request."""
        messages = request.state.get("messages", [])
        session_id = self._extract_session_id(request)
        injections: list[str] = []

        # 1. Doom loop detection
        is_looping, tool_name = _detect_doom_loop(messages)
        if is_looping:
            logger.warning("Doom loop detected: tool=%s", tool_name)
            injections.append(
                f"DOOM LOOP DETECTED: You have called `{tool_name}` with identical "
                f"arguments {DOOM_LOOP_THRESHOLD} times.\n"
                "STOP calling this tool with the same arguments. Try a different approach:\n"
                "1. If the tool keeps failing, skip it and move to the next task\n"
                "2. Try a different tool or different arguments\n"
                "3. If stuck, summarize what you've accomplished and ask the user for guidance"
            )

        # 2. Steps limit
        max_steps = _get_max_steps()
        step = _step_counts.get(session_id, 0)
        _step_counts[session_id] = step + 1

        if step >= max_steps:
            logger.warning(
                "Max steps reached: session=%s steps=%d", session_id, step
            )
            injections.append(
                "MAXIMUM STEPS REACHED — Tools are disabled until next user input.\n"
                "Respond with text only. Summarize what has been accomplished "
                "and list remaining tasks.\n"
                "Do NOT make any tool calls."
            )

        # 3. Cost guardrail
        cost_injection = self._check_cost_guardrail()
        if cost_injection:
            injections.append(cost_injection)

        if not injections:
            return request

        text = "\n\n".join(injections)
        new_sys = append_to_system_message(request.system_message, text)
        return request.override(system_message=new_sys)

    # -- Tool call hooks (output truncation + timing) --------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        start = time.monotonic()
        result = handler(request)
        self._record_tool_timing(request, start)
        return self._maybe_truncate_result(result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        start = time.monotonic()
        result = await handler(request)
        self._record_tool_timing(request, start)
        return self._maybe_truncate_result(result)

    # -- Internals -------------------------------------------------------------

    _AUTO_DUMP_THRESHOLD = int(os.environ.get("AG3NT_AUTO_DUMP_THRESHOLD", "4000"))
    _PREVIEW_CHARS = 500

    @staticmethod
    def _maybe_truncate_result(
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Apply output truncation to tool results.

        Handles ToolMessage (truncate .content) and passes through
        Command results (handled by FilesystemMiddleware's eviction).

        Outputs exceeding ~1K tokens (4000 chars) are automatically stored
        as artifacts; the agent receives a preview + artifact reference.
        """
        if isinstance(result, Command):
            return result

        if not isinstance(result, ToolMessage):
            return result

        content = result.content
        if not isinstance(content, str) or not content:
            return result

        # Auto-dump large outputs to artifact store
        if len(content) > AgentGuardMiddleware._AUTO_DUMP_THRESHOLD:
            try:
                from ag3nt_agent.artifact_store import get_artifact_store
                store = get_artifact_store()
                tool_name = getattr(result, "name", None) or "unknown_tool"
                meta = store.write_artifact(content, tool_name=tool_name)
                preview = content[:AgentGuardMiddleware._PREVIEW_CHARS]
                result.content = (
                    f"{preview}\n\n"
                    f"[Full output ({meta.size_bytes} bytes) stored as artifact "
                    f"{meta.artifact_id}. Use read_artifact('{meta.artifact_id}') "
                    f"to see the complete content.]"
                )
                return result
            except Exception:
                logger.debug("Auto-dump to artifact failed, falling back to truncation", exc_info=True)

        # Fallback: existing truncation
        try:
            from ag3nt_agent.output_truncation import maybe_truncate
            truncated, was_truncated, _path = maybe_truncate(content)
            if was_truncated:
                result.content = truncated
        except Exception:
            logger.debug("Output truncation failed", exc_info=True)

        return result

    @staticmethod
    def _record_tool_timing(request: ToolCallRequest, start: float) -> None:
        """Record tool call duration to cost tracker and trace logger."""
        duration_ms = (time.monotonic() - start) * 1000
        tool_name = getattr(request, "tool_name", None) or getattr(request, "name", "unknown")
        try:
            from ag3nt_agent.cost_tracker import get_cost_tracker
            get_cost_tracker().record_tool_timing(tool_name, duration_ms)
        except Exception:
            pass
        try:
            from ag3nt_agent.trace_logger import get_trace_logger, is_langsmith_configured
            if not is_langsmith_configured():
                get_trace_logger().log_tool_call(tool_name, duration_ms)
        except Exception:
            pass

    @staticmethod
    def _check_cost_guardrail() -> str | None:
        """Check session cost against limit. Returns injection text or None."""
        try:
            from ag3nt_agent.agent_config import MAX_SESSION_COST_USD
            from ag3nt_agent.cost_tracker import get_cost_tracker

            tracker = get_cost_tracker()
            total = tracker.get_session_total()
            cost = total.estimated_cost_usd
            limit = MAX_SESSION_COST_USD

            if cost >= limit * 2:
                logger.warning("Hard cost limit reached: $%.2f (limit $%.2f)", cost, limit)
                return (
                    f"SESSION COST LIMIT EXCEEDED (${{cost:.2f}} / ${{limit:.2f}}).\n"
                    "You MUST stop all tool calls immediately.\n"
                    "Respond with a summary of what was accomplished."
                ).format(cost=cost, limit=limit)
            elif cost >= limit:
                logger.warning("Soft cost limit reached: $%.2f (limit $%.2f)", cost, limit)
                return (
                    f"COST WARNING: Session cost (${{cost:.2f}}) has reached the "
                    f"limit (${{limit:.2f}}). Avoid tool calls unless essential. "
                    "Wrap up your current task efficiently."
                ).format(cost=cost, limit=limit)
            elif cost >= limit * 0.8:
                return (
                    f"Cost notice: ${{cost:.2f}} of ${{limit:.2f}} session budget used ({{pct:.0f}}%). "
                    "Be mindful of token usage."
                ).format(cost=cost, limit=limit, pct=(cost / limit) * 100)
        except Exception:
            logger.debug("Cost guardrail check failed", exc_info=True)
        return None

    @staticmethod
    def _extract_session_id(request: ModelRequest) -> str:
        """Extract session_id from request config, falling back to 'default'."""
        try:
            configurable = request.config.get("configurable", {})
            return configurable.get("thread_id", "default")
        except (AttributeError, TypeError):
            return "default"
