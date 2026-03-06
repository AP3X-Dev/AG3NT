"""AgentMiddleware adapter for HookMiddleware.

Bridges the HookMiddleware (PRE_TOOL/POST_TOOL hooks) to the
LangChain AgentMiddleware interface used by create_deep_agent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from ag3nt_agent.hooks import HookPhase
from ag3nt_agent.hooks.middleware import HookMiddleware

logger = logging.getLogger("ag3nt.hooks.agent_middleware")


class SafetyHookAgentMiddleware(AgentMiddleware[AgentState, Any]):
    """Wraps tool calls with PRE_TOOL / POST_TOOL safety hooks.

    Delegates to :class:`HookMiddleware` for actual hook execution.
    This adapter bridges the HookMiddleware's custom signature to the
    LangChain ``AgentMiddleware`` interface expected by ``create_deep_agent``.
    """

    def __init__(self, hook_middleware: HookMiddleware) -> None:
        self._hook_mw = hook_middleware
        self.tools: list = []  # Required by AgentMiddleware interface

    @property
    def name(self) -> str:
        """Return middleware name."""
        return "safety_hooks"

    # -- Model calls: pass through (hooks only apply to tools) ----------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Pass through model calls unchanged."""
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Pass through async model calls unchanged."""
        return await handler(request)

    # -- Tool calls: delegate to HookMiddleware -------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Sync tool call wrapping -- runs hooks via event loop.

        Falls back to direct execution if the event loop is unavailable.
        """
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        # Run PRE_TOOL hooks synchronously (best-effort)
        try:
            loop = asyncio.get_event_loop()
            pre_result = loop.run_until_complete(
                self._hook_mw._hook_manager.run(
                    HookPhase.PRE_TOOL, {"tool_name": tool_name, "arguments": args}
                )
            )
            if pre_result.cancelled:
                logger.warning(
                    "Tool %s blocked by hook: %s", tool_name, pre_result.cancel_reason
                )
                return ToolMessage(
                    content=f"Tool blocked by safety hook: {pre_result.cancel_reason}",
                    tool_call_id=tool_call_id,
                )
        except Exception as exc:
            logger.debug("PRE_TOOL hook failed (allowing tool to proceed): %s", exc)

        # Execute the tool
        result = handler(request)

        # POST_TOOL hooks (best-effort, don't block result)
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(
                self._hook_mw._hook_manager.run(
                    HookPhase.POST_TOOL,
                    {
                        "tool_name": tool_name,
                        "arguments": args,
                        "result": str(result),
                    },
                )
            )
        except Exception:
            pass

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async tool call wrapping -- delegates to HookMiddleware.

        Uses ``HookMiddleware.wrap_tool_call`` which handles PRE_TOOL and
        POST_TOOL hook execution, including cancellation logic.
        """
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        # Define async execute_fn that calls the original handler
        async def execute_fn():
            return await handler(request)

        # Delegate to HookMiddleware which handles PRE/POST hooks
        result = await self._hook_mw.wrap_tool_call(
            tool_name=tool_name,
            arguments=args,
            execute_fn=execute_fn,
        )

        # HookMiddleware returns a dict on cancellation
        if isinstance(result, dict) and result.get("cancelled"):
            tool_call_id = tool_call.get("id", "")
            return ToolMessage(
                content=f"Tool blocked by safety hook: {result.get('reason', 'blocked')}",
                tool_call_id=tool_call_id,
            )

        # If result is already a ToolMessage or Command, return as-is
        if isinstance(result, (ToolMessage, Command)):
            return result

        # Shouldn't normally happen, but handle gracefully
        return result
