"""DeepAgents JSON-RPC daemon for the web UI.

This process is intended to be spawned by the Next.js server (node runtime) and
communicates over stdio using newline-delimited JSON.

Requests:
  {"id": "1", "method": "chat", "params": {"thread_id": "abcd1234", "assistant_id": "agent", "message": "..."}}
  {"id": "2", "method": "resume", "params": {"thread_id": "abcd1234", "assistant_id": "agent", "interrupt_id": "...", "decision": "approve"}}

Responses:
  {"id": "1", "ok": true, "result": {"thread_id": "...", "events": [...], "assistant": "...", "auto_approve": false}}
  {"id": "X", "ok": false, "error": {"message": "...", "type": "..."}}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add ag3nt_agent to path before importing
_ag3nt_path = Path(__file__).parent.parent.parent.parent / "apps" / "agent"
if str(_ag3nt_path) not in sys.path:
    sys.path.insert(0, str(_ag3nt_path))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt
from pydantic import TypeAdapter

from deepagents_cli.config import settings
# Use ag3nt_agent's model creation which supports OpenRouter and Kimi
from ag3nt_agent.deepagents_runtime import _create_model as _ag3nt_create_model


def create_model(model_name_override: str | None = None):
    """Create model with optional name override.

    Wraps ag3nt_agent's model creation to support UI model selection.
    If model_name_override is provided, sets AG3NT_MODEL_NAME env var.

    Model name formats supported:
    - OpenRouter: "provider/model" (e.g., "moonshotai/kimi-k2-thinking", "deepseek/deepseek-chat")
    - Anthropic: "claude-*"
    - OpenAI: "gpt-*", "o1-*"
    - Google: "gemini-*"
    - Kimi direct: "kimi-*", "moonshot-*"
    """
    import os
    logging.info(f"[create_model] model_name_override={model_name_override!r}")
    if model_name_override:
        # Detect provider from model name format
        if "/" in model_name_override:
            # OpenRouter format: "provider/model" (e.g., "moonshotai/kimi-k2-thinking")
            os.environ["AG3NT_MODEL_PROVIDER"] = "openrouter"
            os.environ["AG3NT_MODEL_NAME"] = model_name_override
        elif model_name_override.startswith("claude-"):
            os.environ["AG3NT_MODEL_PROVIDER"] = "anthropic"
            os.environ["AG3NT_MODEL_NAME"] = model_name_override
        elif model_name_override.startswith(("gpt-", "o1-", "o3-")):
            os.environ["AG3NT_MODEL_PROVIDER"] = "openai"
            os.environ["AG3NT_MODEL_NAME"] = model_name_override
        elif model_name_override.startswith("gemini-"):
            os.environ["AG3NT_MODEL_PROVIDER"] = "google"
            os.environ["AG3NT_MODEL_NAME"] = model_name_override
        elif model_name_override.startswith(("kimi-", "moonshot-")):
            os.environ["AG3NT_MODEL_PROVIDER"] = "kimi"
            os.environ["AG3NT_MODEL_NAME"] = model_name_override
        elif model_name_override == "kimi-for-coding":
            os.environ["AG3NT_MODEL_PROVIDER"] = "kimi"
            os.environ["AG3NT_MODEL_NAME"] = "kimi-for-coding"
        else:
            # Unknown format, just set the model name and let ag3nt detect
            os.environ["AG3NT_MODEL_NAME"] = model_name_override
        logging.info(
            f"[create_model] Set env: AG3NT_MODEL_PROVIDER={os.environ.get('AG3NT_MODEL_PROVIDER')}, "
            f"AG3NT_MODEL_NAME={os.environ.get('AG3NT_MODEL_NAME')}"
        )
    else:
        logging.warning(
            "[create_model] No model override provided, using env defaults: "
            f"AG3NT_MODEL_PROVIDER={os.environ.get('AG3NT_MODEL_PROVIDER', '(unset)')}, "
            f"AG3NT_MODEL_NAME={os.environ.get('AG3NT_MODEL_NAME', '(unset)')}"
        )

    return _ag3nt_create_model()
from deepagents_cli.file_ops import FileOpTracker
from deepagents_cli.sessions import (
    generate_thread_id,
    get_checkpointer,
    list_threads,
    delete_thread,
    get_thread_messages,
)
from deepagents_cli.tools import http_request, web_search
from deepagents_cli.e2b_tools import get_e2b_tools, has_e2b

# Use UI-specific agent factory with UI-optimized defaults
from ui_agent_factory import create_ui_agent


_HITL_REQUEST_ADAPTER = TypeAdapter(dict)  # HITLRequest is a TypedDict at runtime


def sanitize_error(error: Exception) -> dict[str, Any]:
    """Convert exception to client-safe error without exposing internal details.

    Full tracebacks are logged to stderr for debugging but not sent to client.
    """
    error_type = type(error).__name__

    # Map common errors to user-friendly messages
    ERROR_MESSAGES = {
        "ConnectionRefusedError": "Unable to connect to agent service",
        "ConnectionError": "Connection error occurred",
        "ConnectionResetError": "Connection was reset, please retry",
        "APIConnectionError": "Failed to connect to AI provider — retrying may help",
        "APITimeoutError": "AI provider request timed out — retrying may help",
        "TimeoutError": "Request timed out",
        "asyncio.TimeoutError": "Request timed out",
        "ValueError": "Invalid input provided",
        "KeyError": "Missing required field",
        "FileNotFoundError": "File not found",
        "PermissionError": "Permission denied",
        "JSONDecodeError": "Invalid JSON format",
        "ModuleNotFoundError": "Required module not available",
        "ImportError": "Failed to load required component",
    }

    # Get user-friendly message or use generic
    message = ERROR_MESSAGES.get(error_type)
    if message is None:
        # For unknown errors, include the error message but not the traceback
        message = str(error) if str(error) else "An error occurred"

    # Log full trace to stderr for debugging (not sent to client)
    import traceback
    print(f"[daemon-error] {error_type}: {error}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

    return {
        "message": message,
        "code": error_type,
        # No traceback in response - security improvement
    }


def _format_status_message(tool_name: str, args: dict[str, Any] | None) -> str:
    """Format a user-friendly status message for tool execution.

    For subagent tasks, shows the agent type (e.g., "Deep Research Agent working...")
    For regular tools, shows a user-friendly message.
    """
    if tool_name == "task" and args:
        subagent_type = args.get("subagent_type", "")
        if subagent_type:
            # Format: "deep-research" -> "Deep Research Agent"
            formatted = " ".join(
                word.capitalize()
                for word in subagent_type.replace("-", " ").replace("_", " ").split()
            )
            return f"{formatted} Agent working..."
        # No subagent_type but we have a description
        description = args.get("description", "")
        if description:
            # Show first 40 chars of description
            preview = description[:40] + "..." if len(description) > 40 else description
            return f"Agent working: {preview}"
        return "Subagent working..."

    # User-friendly names for common tools
    friendly_names = {
        "web_search": "Searching the web...",
        "read_file": "Reading file...",
        "write_file": "Writing file...",
        "edit_file": "Editing file...",
        "list_directory": "Listing directory...",
        "run_shell_command": "Running command...",
        "shell": "Running command...",
        "execute": "Executing code...",
        "deep_research": "Deep Research Agent working...",
    }

    if tool_name in friendly_names:
        return friendly_names[tool_name]

    # Fallback: format the tool name nicely
    formatted_name = " ".join(
        word.capitalize()
        for word in tool_name.replace("-", " ").replace("_", " ").split()
    )
    return f"{formatted_name}..."


@dataclass
class SessionState:
    thread_id: str
    auto_approve: bool = False


class AgentRuntime:
    def __init__(self, checkpointer: Any) -> None:
        self._checkpointer = checkpointer
        # Cache agents per model name (not assistant_id) for dynamic model switching
        self._agents: dict[str, tuple[Any, Any]] = {}
        self._sessions: dict[str, SessionState] = {}
        self._pending_interrupts: dict[str, dict[str, Any]] = {}
        self._pending_plan_approvals: dict[str, dict[str, Any]] = {}
        # Track cancelled stream request IDs
        self._cancelled_streams: set[str] = set()

    def cancel_stream(self, stream_id: str) -> dict[str, Any]:
        """Mark a stream as cancelled. The streaming loop will check this and stop."""
        self._cancelled_streams.add(stream_id)
        logging.info(f"Stream {stream_id} marked for cancellation")
        return {"cancelled": True, "stream_id": stream_id}

    def is_stream_cancelled(self, stream_id: str) -> bool:
        """Check if a stream has been cancelled."""
        return stream_id in self._cancelled_streams

    def clear_cancelled_stream(self, stream_id: str) -> None:
        """Remove a stream from the cancelled set after processing."""
        self._cancelled_streams.discard(stream_id)

    def clear_caches(self) -> dict[str, Any]:
        """Clear all agent and session caches. Call this to force fresh model creation."""
        agent_count = len(self._agents)
        session_count = len(self._sessions)
        interrupt_count = len(self._pending_interrupts)

        self._agents.clear()
        self._sessions.clear()
        self._pending_interrupts.clear()

        logging.info(
            f"Cleared caches: {agent_count} agents, {session_count} sessions, {interrupt_count} interrupts"
        )
        return {
            "cleared_agents": agent_count,
            "cleared_sessions": session_count,
            "cleared_interrupts": interrupt_count,
        }

    def _get_session(self, thread_id: str) -> SessionState:
        state = self._sessions.get(thread_id)
        if state is None:
            state = SessionState(thread_id=thread_id)
            self._sessions[thread_id] = state
        return state

    def _get_agent(
        self, assistant_id: str, model_name: str | None = None
    ) -> tuple[Any, Any]:
        """Get or create an agent, keyed by model name for dynamic model switching."""
        # DISABLED CACHING: Always create fresh agents to pick up code changes
        # This ensures MCP tools and middleware changes take effect immediately
        cache_key = model_name or "default"

        # Always create fresh agent (caching disabled for development)
        # Create model - use override if provided, otherwise use env default
        model = create_model(model_name_override=model_name)
        actual_model = getattr(model, "model_name", getattr(model, "model", "unknown"))
        logging.info(f"Created new agent with model: {actual_model}")

        tools = [http_request]
        if settings.has_tavily:
            tools.append(web_search)

        # Add E2B sandbox tools if available
        if has_e2b():
            e2b_tools = get_e2b_tools()
            tools.extend(e2b_tools)
            logging.info(f"E2B sandbox tools enabled: {len(e2b_tools)} tools")
        else:
            logging.info(
                "E2B sandbox tools disabled (no API key or library not installed)"
            )

        # Debug: Log project root and skills directory
        logging.info(f"Project root: {settings.project_root}")
        project_skills_dir = settings.get_project_skills_dir()
        logging.info(f"Project skills dir: {project_skills_dir}")
        if project_skills_dir and project_skills_dir.exists():
            logging.info(
                f"Project skills dir exists: {list(project_skills_dir.iterdir())}"
            )
        else:
            logging.info("Project skills dir does not exist")

        # IMPORTANT: always create with auto_approve=False so interrupts are emitted.
        # Use UI agent factory which enables UI-optimized features by default:
        # - deep-research subagent for comprehensive research
        # - deep-web subagent for browser automation, scraping, crawling
        # - memory and skills middleware
        agent, backend = create_ui_agent(
            model=model,
            assistant_id=assistant_id,
            tools=tools,
            auto_approve=False,
            # Enable deep-web subagent for browser automation (scraping, crawling, live browser)
            enable_deep_web=True,
            enable_deep_research=True,
            checkpointer=self._checkpointer,
        )
        # Don't cache - always create fresh agents
        # self._agents[cache_key] = (agent, backend)
        return agent, backend

    def _build_config(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        plan_mode: bool = False,
    ) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "metadata": {
                "assistant_id": assistant_id,
                "agent_name": assistant_id,
                "updated_at": datetime.now(UTC).isoformat(),
                "plan_mode": plan_mode,
            },
        }

    async def chat(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        message: str,
        auto_approve: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        session = self._get_session(thread_id)
        # Update session auto_approve from the UI toggle
        session.auto_approve = auto_approve
        agent, backend = self._get_agent(assistant_id, model_name=model)
        config = self._build_config(thread_id=thread_id, assistant_id=assistant_id)

        stream_input: dict[str, Any] | Command = {
            "messages": [{"role": "user", "content": message}],
        }
        return await self._run_until_done_or_approval(
            agent=agent,
            backend=backend,
            config=config,
            session=session,
            stream_input=stream_input,
        )

    async def resume(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        interrupt_id: str,
        decision: str,
    ) -> dict[str, Any]:
        session = self._get_session(thread_id)
        agent, backend = self._get_agent(assistant_id)
        config = self._build_config(thread_id=thread_id, assistant_id=assistant_id)

        pending = self._pending_interrupts.get(thread_id, {})
        hitl_request = pending.get(interrupt_id)
        action_reqs = []
        if isinstance(hitl_request, dict):
            action_reqs = hitl_request.get("action_requests", []) or []

        if decision == "auto_approve_all":
            session.auto_approve = True
            decision_type = "approve"
        else:
            decision_type = decision

        decisions = [{"type": decision_type} for _ in action_reqs] or [
            {"type": decision_type}
        ]
        hitl_response = {interrupt_id: {"decisions": decisions}}
        stream_input = Command(resume=hitl_response)
        return await self._run_until_done_or_approval(
            agent=agent,
            backend=backend,
            config=config,
            session=session,
            stream_input=stream_input,
        )

    async def new_thread(self) -> dict[str, Any]:
        thread_id = generate_thread_id()
        self._sessions[thread_id] = SessionState(thread_id=thread_id)
        return {"thread_id": thread_id, "auto_approve": False}

    async def chat_stream(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        message: str,
        auto_approve: bool = False,
        model: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        ui_context: str | None = None,
        plan_mode: bool = False,
    ):
        """Streaming version of chat that yields events as they happen."""
        session = self._get_session(thread_id)
        session.auto_approve = auto_approve
        agent, backend = self._get_agent(assistant_id, model_name=model)
        config = self._build_config(
            thread_id=thread_id,
            assistant_id=assistant_id,
            plan_mode=plan_mode,
        )

        # Clear any pending interrupts for this thread - new message takes priority
        if thread_id in self._pending_interrupts:
            logging.info(
                f"[chat_stream] Clearing pending interrupts for thread {thread_id}"
            )
            del self._pending_interrupts[thread_id]

        # Build message content - support multimodal with attachments
        message_content = self._build_message_content(message, attachments)

        messages: list[dict[str, Any]] = []

        if ui_context:
            # Inject UI context as a preamble in the user message, NOT as a
            # separate system message.  A system message here would conflict
            # with the agent's own system_prompt and middleware overrides,
            # producing "Received multiple non-consecutive system messages."
            if isinstance(message_content, str):
                message_content = f"[UI Context]\n{ui_context}\n[/UI Context]\n\n{message_content}"
            elif isinstance(message_content, list):
                message_content = [{"type": "text", "text": f"[UI Context]\n{ui_context}\n[/UI Context]\n\n"}] + message_content

        messages.append({"role": "user", "content": message_content})

        stream_input: dict[str, Any] | Command = {
            "messages": messages,
        }
        async for event in self._stream_until_done_or_approval(
            agent=agent,
            backend=backend,
            config=config,
            session=session,
            stream_input=stream_input,
        ):
            yield event

    def _build_message_content(
        self, message: str, attachments: list[dict[str, Any]] | None
    ) -> str | list[dict[str, Any]]:
        """Build message content, handling file attachments.

        For text-only messages, returns a plain string.
        For messages with attachments, returns multimodal content blocks.

        For images: saves to temp file and includes path reference (to avoid context overflow).
        For text/data files: embeds content directly with file markers.
        """
        import base64
        import tempfile
        import os

        if not attachments:
            return message

        content_blocks: list[dict[str, Any]] = []

        # Add text block if there's a message
        if message.strip():
            content_blocks.append({"type": "text", "text": message})

        # Process each attachment
        for attachment in attachments:
            file_type = attachment.get("type", "")
            file_name = attachment.get("name", "unknown")
            content = attachment.get("content", "")
            data_url = attachment.get("data_url", "")

            if file_type.startswith("image/"):
                # For images: save to temp file to avoid context overflow
                # The agent can use the file path with image tools
                try:
                    # Extract base64 content
                    image_data = None
                    if data_url and "base64," in data_url:
                        image_data = base64.b64decode(data_url.split("base64,")[1])
                    elif content:
                        image_data = base64.b64decode(content)

                    if image_data:
                        # Get file extension from mime type
                        ext = file_type.split("/")[-1].split(";")[0]
                        if ext == "jpeg":
                            ext = "jpg"

                        # Save to temp directory with descriptive name
                        temp_dir = tempfile.gettempdir()
                        safe_name = "".join(
                            c if c.isalnum() or c in "._-" else "_" for c in file_name
                        )
                        temp_path = os.path.join(temp_dir, f"attachment_{safe_name}")
                        if not temp_path.lower().endswith(f".{ext}"):
                            temp_path = f"{temp_path}.{ext}"

                        with open(temp_path, "wb") as f:
                            f.write(image_data)

                        logging.info(f"Saved image attachment to: {temp_path}")

                        # Send image as vision content block so the LLM can see it
                        b64_str = base64.b64encode(image_data).decode("utf-8")
                        mime = file_type.split(";")[0]  # e.g. "image/png"
                        content_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{b64_str}"
                                },
                            }
                        )

                        # Also tell the agent where the file is saved
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": f"\n[User attached image: {file_name} — saved to: {temp_path}]\n",
                            }
                        )
                    else:
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": f"\n[Attachment: {file_name} - could not decode image data]\n",
                            }
                        )
                except Exception as e:
                    logging.warning(f"Failed to save image attachment {file_name}: {e}")
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": f"\n[Attachment: {file_name} - failed to process: {e}]\n",
                        }
                    )
            else:
                # Text/code/data files: include as text with clear file markers
                try:
                    if content:
                        # Content is base64 encoded for non-images
                        try:
                            decoded = base64.b64decode(content).decode("utf-8")
                        except Exception:
                            decoded = content  # May already be plain text
                    else:
                        decoded = "[No content]"

                    # Format as a clearly marked file block
                    file_block = (
                        f"\n\n--- FILE: {file_name} ---\n{decoded}\n--- END FILE ---\n"
                    )
                    content_blocks.append({"type": "text", "text": file_block})
                except Exception as e:
                    logging.warning(f"Failed to decode attachment {file_name}: {e}")
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": f"\n[Attachment: {file_name} - failed to decode]\n",
                        }
                    )

        return content_blocks if content_blocks else message

    async def resume_stream(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        interrupt_id: str,
        decision: str,
    ):
        """Streaming version of resume that yields events as they happen."""
        session = self._get_session(thread_id)
        agent, backend = self._get_agent(assistant_id)
        config = self._build_config(thread_id=thread_id, assistant_id=assistant_id)

        pending = self._pending_interrupts.get(thread_id, {})
        hitl_request = pending.get(interrupt_id)
        action_reqs = []
        if isinstance(hitl_request, dict):
            action_reqs = hitl_request.get("action_requests", []) or []

        if decision == "auto_approve_all":
            session.auto_approve = True
            decision_type = "approve"
        else:
            decision_type = decision

        decisions = [{"type": decision_type} for _ in action_reqs] or [
            {"type": decision_type}
        ]
        hitl_response = {interrupt_id: {"decisions": decisions}}
        stream_input = Command(resume=hitl_response)
        async for event in self._stream_until_done_or_approval(
            agent=agent,
            backend=backend,
            config=config,
            session=session,
            stream_input=stream_input,
        ):
            yield event

    async def resume_plan_decision(
        self,
        *,
        thread_id: str,
        assistant_id: str,
        decision: str,
        reason: str = "",
    ):
        """Handle user's plan approval or rejection decision.

        Yields SSE events as the agent continues execution after
        the plan decision.
        """
        session = self._get_session(thread_id)
        agent, backend = self._get_agent(assistant_id)

        pending = self._pending_plan_approvals.pop(thread_id, None)
        if not pending:
            yield {"type": "error", "message": "No pending plan approval for this thread"}
            return

        if decision == "approve":
            yield {"type": "plan_approved", "plan_path": pending["plan_path"]}

            config = self._build_config(
                thread_id=thread_id,
                assistant_id=assistant_id,
                plan_mode=True,
            )
            stream_input = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Plan approved. Execute the plan step by step. "
                            "For each step: announce it, execute it, verify it works, "
                            "then move to the next. Create git checkpoints before each "
                            "step using the checkpoint tool."
                        ),
                    }
                ],
            }
            async for event in self._stream_until_done_or_approval(
                agent=agent,
                backend=backend,
                config=config,
                session=session,
                stream_input=stream_input,
            ):
                yield event
        else:
            yield {
                "type": "plan_rejected",
                "plan_path": pending["plan_path"],
                "reason": reason,
            }

            config = self._build_config(
                thread_id=thread_id,
                assistant_id=assistant_id,
                plan_mode=True,
            )
            rejection_msg = "Plan rejected."
            if reason:
                rejection_msg += f" Reason: {reason}"
            rejection_msg += " Please revise the plan or ask for clarification."

            stream_input = {
                "messages": [{"role": "user", "content": rejection_msg}],
            }
            async for event in self._stream_until_done_or_approval(
                agent=agent,
                backend=backend,
                config=config,
                session=session,
                stream_input=stream_input,
            ):
                yield event

    async def _stream_until_done_or_approval(
        self,
        *,
        agent: Any,
        backend: Any,
        config: dict[str, Any],
        session: SessionState,
        stream_input: dict[str, Any] | Command,
    ):
        """Stream events as they happen until completion or approval required."""
        # Emit initial status
        yield {
            "type": "status",
            "status": "thinking",
            "message": "Agent is thinking...",
        }
        yield {"type": "thread_id", "thread_id": session.thread_id}

        tool_call_args: dict[str, dict[str, Any]] = {}
        tool_call_names: dict[str, str] = {}
        file_op_tracker = FileOpTracker(
            assistant_id=config["metadata"]["assistant_id"], backend=backend
        )
        tool_call_buffers: dict[str | int, dict[str, Any]] = {}
        displayed_tool_ids: set[str] = set()
        assistant_text_parts: list[str] = []

        while True:
            interrupt_occurred = False
            pending_interrupts: dict[str, Any] = {}

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
                config=config,
                durability="exit",
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue
                namespace, mode, data = chunk
                ns_key = tuple(namespace) if namespace else ()
                is_main = ns_key == ()

                if mode == "custom":
                    if isinstance(data, dict):
                        event_type = data.get("type", "")

                        # Forward plan lifecycle events directly to UI
                        if event_type.startswith("plan_"):
                            yield {
                                "type": event_type,
                                **{k: v for k, v in data.items() if k != "type"},
                            }

                            # plan_ready_for_approval triggers an approval gate
                            if event_type == "plan_ready_for_approval":
                                self._pending_plan_approvals[session.thread_id] = {
                                    "plan_path": data.get("plan_path", ""),
                                    "plan_content": data.get("plan_content", ""),
                                    "summary": data.get("summary", ""),
                                    "step_count": data.get("step_count", 0),
                                    "checkpoint_count": data.get("checkpoint_count", 0),
                                }

                                if not session.auto_approve:
                                    yield {
                                        "type": "plan_approval_requested",
                                        "plan_path": data.get("plan_path", ""),
                                        "plan_content": data.get("plan_content", ""),
                                        "summary": data.get("summary", ""),
                                        "step_count": data.get("step_count", 0),
                                        "checkpoint_count": data.get("checkpoint_count", 0),
                                    }
                                    yield {"type": "done", "plan_approval_required": True}
                                    return

                            continue

                        # Existing: forward parallel_tasks_progress
                        if event_type == "parallel_tasks_progress":
                            yield {
                                "type": "tool_progress",
                                "tool_name": "parallel_tasks",
                                "status": data.get("status", ""),
                                "message": data.get("message", ""),
                                "completed": data.get("completed"),
                                "total": data.get("total"),
                            }
                    continue

                if mode == "updates":
                    if not isinstance(data, dict):
                        continue
                    if "__interrupt__" in data:
                        interrupts: list[Interrupt] = data.get("__interrupt__") or []
                        for it in interrupts:
                            interrupt_occurred = True
                            try:
                                validated = _HITL_REQUEST_ADAPTER.validate_python(
                                    it.value
                                )
                            except Exception:
                                validated = it.value
                            pending_interrupts[it.id] = validated

                elif mode == "messages":
                    if not is_main:
                        continue
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    message, _meta = data

                    if isinstance(message, HumanMessage):
                        continue

                    if isinstance(message, ToolMessage):
                        tool_call_id = getattr(message, "tool_call_id", None)
                        status = getattr(message, "status", "success")
                        name = getattr(message, "name", "") or tool_call_names.get(
                            tool_call_id or "", ""
                        )
                        output = message.content
                        output_str = _stringify_tool_content(output)

                        record = await file_op_tracker.complete_with_message(message)
                        if record is not None:
                            name = record.tool_name

                        args = tool_call_args.get(tool_call_id or "", {})

                        ev: dict[str, Any] = {
                            "type": "tool_result",
                            "tool_name": name,
                            "tool_call_id": tool_call_id,
                            "status": status,
                            "output": output_str,
                            "args": args,
                        }
                        if record is not None:
                            if record.display_path:
                                ev["path"] = record.display_path
                            if record.diff:
                                ev["diff"] = record.diff
                            if record.read_output:
                                ev["read_output"] = record.read_output
                            if record.error:
                                ev["error"] = record.error
                        yield ev
                        yield {
                            "type": "status",
                            "status": "thinking",
                            "message": "Agent is thinking...",
                        }
                        continue

                    # AI message chunks / blocks
                    blocks = getattr(message, "content_blocks", None)
                    if blocks is None:
                        # Some models may yield a completed AIMessage with tool_calls
                        if isinstance(message, AIMessage) and getattr(
                            message, "tool_calls", None
                        ):
                            for tc in message.tool_calls:
                                tc_id = tc.get("id")
                                tc_name = tc.get("name")
                                tc_args = tc.get("args")
                                if (
                                    tc_id
                                    and tc_id not in displayed_tool_ids
                                    and tc_name
                                ):
                                    displayed_tool_ids.add(tc_id)
                                    if isinstance(tc_args, dict):
                                        tool_call_args[tc_id] = tc_args
                                    tool_call_names[tc_id] = tc_name
                                    await file_op_tracker.start_operation(
                                        tc_name, tc_args or {}, tc_id
                                    )
                                    status_msg = _format_status_message(
                                        tc_name,
                                        tc_args if isinstance(tc_args, dict) else None,
                                    )
                                    yield {
                                        "type": "status",
                                        "status": "executing",
                                        "message": status_msg,
                                    }
                                    yield {
                                        "type": "tool_call",
                                        "tool_name": tc_name,
                                        "tool_call_id": tc_id,
                                        "args": tc_args or {},
                                    }
                        continue

                    for block in blocks:
                        btype = block.get("type")
                        if btype == "text":
                            text = block.get("text", "")
                            if text:
                                assistant_text_parts.append(text)
                                yield {"type": "text_delta", "text": text}
                        elif btype == "reasoning_details":
                            # Stream reasoning/thinking process from models like DeepSeek, Claude, o1
                            reasoning_text = block.get("text", "")
                            if reasoning_text:
                                yield {
                                    "type": "reasoning_delta",
                                    "text": reasoning_text,
                                }
                        elif btype in ("tool_call_chunk", "tool_call"):
                            chunk_name = block.get("name")
                            chunk_args = block.get("args")
                            chunk_id = block.get("id")
                            chunk_index = block.get("index")

                            key: str | int
                            if chunk_index is not None:
                                key = chunk_index
                            elif chunk_id is not None:
                                key = chunk_id
                            else:
                                key = f"unknown-{len(tool_call_buffers)}"

                            buffer = tool_call_buffers.setdefault(
                                key,
                                {
                                    "name": None,
                                    "id": None,
                                    "args": None,
                                    "args_parts": [],
                                },
                            )
                            if chunk_name:
                                buffer["name"] = chunk_name
                            if chunk_id:
                                buffer["id"] = chunk_id

                            parsed_args: Any = None
                            if isinstance(chunk_args, dict):
                                parsed_args = chunk_args
                            elif isinstance(chunk_args, str):
                                parts: list[str] = buffer.setdefault("args_parts", [])
                                if chunk_args and (
                                    not parts or chunk_args != parts[-1]
                                ):
                                    parts.append(chunk_args)
                                try:
                                    parsed_args = json.loads("".join(parts))
                                except json.JSONDecodeError:
                                    parsed_args = None

                            if (
                                buffer.get("name")
                                and buffer.get("id")
                                and isinstance(parsed_args, dict)
                            ):
                                tc_name = buffer["name"]
                                tc_id = buffer["id"]
                                if tc_id not in displayed_tool_ids:
                                    displayed_tool_ids.add(tc_id)
                                    tool_call_args[tc_id] = parsed_args
                                    tool_call_names[tc_id] = tc_name
                                    await file_op_tracker.start_operation(
                                        tc_name, parsed_args, tc_id
                                    )
                                    status_msg = _format_status_message(
                                        tc_name, parsed_args
                                    )
                                    yield {
                                        "type": "status",
                                        "status": "executing",
                                        "message": status_msg,
                                    }
                                    yield {
                                        "type": "tool_call",
                                        "tool_name": tc_name,
                                        "tool_call_id": tc_id,
                                        "args": parsed_args,
                                    }
                                tool_call_buffers.pop(key, None)

            if interrupt_occurred:
                self._pending_interrupts[session.thread_id] = pending_interrupts

                if session.auto_approve:
                    hitl_response: dict[str, Any] = {}
                    for iid, hitl_req in pending_interrupts.items():
                        action_reqs = []
                        if isinstance(hitl_req, dict):
                            action_reqs = hitl_req.get("action_requests", []) or []
                        hitl_response[iid] = {
                            "decisions": [{"type": "approve"} for _ in action_reqs]
                        }
                    stream_input = Command(resume=hitl_response)
                    continue

                approval_payload = []
                for iid, hitl_req in pending_interrupts.items():
                    action_reqs = []
                    if isinstance(hitl_req, dict):
                        action_reqs = hitl_req.get("action_requests", []) or []
                    approval_payload.append(
                        {"interrupt_id": iid, "action_requests": action_reqs}
                    )

                yield {
                    "type": "approval_required",
                    "approvals": approval_payload,
                    "auto_approve": session.auto_approve,
                }
                yield {"type": "done", "approval_required": True}
                return

            # No interrupt, done.
            yield {"type": "status", "status": "done", "message": ""}
            yield {
                "type": "done",
                "approval_required": False,
                "auto_approve": session.auto_approve,
            }
            return

    async def _run_until_done_or_approval(
        self,
        *,
        agent: Any,
        backend: Any,
        config: dict[str, Any],
        session: SessionState,
        stream_input: dict[str, Any] | Command,
    ) -> dict[str, Any]:
        """Run the agent until completion, or return approval_required when HITL triggers."""

        events: list[dict[str, Any]] = []
        assistant_text_parts: list[str] = []

        # Track tool call args for mapping tool outputs to UI components
        tool_call_args: dict[str, dict[str, Any]] = {}
        tool_call_names: dict[str, str] = {}

        file_op_tracker = FileOpTracker(
            assistant_id=config["metadata"]["assistant_id"], backend=backend
        )
        tool_call_buffers: dict[str | int, dict[str, Any]] = {}
        displayed_tool_ids: set[str] = set()

        while True:
            interrupt_occurred = False
            pending_interrupts: dict[str, Any] = {}

            async for chunk in agent.astream(
                stream_input,
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
                config=config,
                durability="exit",
            ):
                if not isinstance(chunk, tuple) or len(chunk) != 3:
                    continue
                namespace, mode, data = chunk
                ns_key = tuple(namespace) if namespace else ()
                is_main = ns_key == ()

                if mode == "custom":
                    # Forward custom progress events (e.g. parallel_tasks)
                    if isinstance(data, dict) and data.get("type") == "parallel_tasks_progress":
                        events.append({
                            "type": "tool_progress",
                            "tool_name": "parallel_tasks",
                            "status": data.get("status", ""),
                            "message": data.get("message", ""),
                            "completed": data.get("completed"),
                            "total": data.get("total"),
                        })
                    continue

                if mode == "updates":
                    if not isinstance(data, dict):
                        continue
                    if "__interrupt__" in data:
                        interrupts: list[Interrupt] = data.get("__interrupt__") or []
                        for it in interrupts:
                            interrupt_occurred = True
                            try:
                                validated = _HITL_REQUEST_ADAPTER.validate_python(
                                    it.value
                                )
                            except Exception:
                                validated = it.value
                            pending_interrupts[it.id] = validated

                elif mode == "messages":
                    if not is_main:
                        continue
                    if not isinstance(data, tuple) or len(data) != 2:
                        continue
                    message, _meta = data

                    if isinstance(message, HumanMessage):
                        continue

                    if isinstance(message, ToolMessage):
                        tool_call_id = getattr(message, "tool_call_id", None)
                        status = getattr(message, "status", "success")
                        name = getattr(message, "name", "") or tool_call_names.get(
                            tool_call_id or "", ""
                        )
                        output = message.content
                        output_str = _stringify_tool_content(output)

                        record = await file_op_tracker.complete_with_message(message)
                        if record is not None:
                            name = record.tool_name

                        args = tool_call_args.get(tool_call_id or "", {})

                        ev: dict[str, Any] = {
                            "type": "tool_result",
                            "tool_name": name,
                            "tool_call_id": tool_call_id,
                            "status": status,
                            "output": output_str,
                            "args": args,
                        }
                        if record is not None:
                            if record.display_path:
                                ev["path"] = record.display_path
                            if record.diff:
                                ev["diff"] = record.diff
                            if record.read_output:
                                ev["read_output"] = record.read_output
                            if record.error:
                                ev["error"] = record.error
                        events.append(ev)
                        continue

                    # AI message chunks / blocks
                    blocks = getattr(message, "content_blocks", None)
                    if blocks is None:
                        # Some models may yield a completed AIMessage with tool_calls
                        if isinstance(message, AIMessage) and getattr(
                            message, "tool_calls", None
                        ):
                            for tc in message.tool_calls:
                                tc_id = tc.get("id")
                                tc_name = tc.get("name")
                                tc_args = tc.get("args")
                                if (
                                    tc_id
                                    and tc_id not in displayed_tool_ids
                                    and tc_name
                                ):
                                    displayed_tool_ids.add(tc_id)
                                    if isinstance(tc_args, dict):
                                        tool_call_args[tc_id] = tc_args
                                    tool_call_names[tc_id] = tc_name
                                    await file_op_tracker.start_operation(
                                        tc_name, tc_args or {}, tc_id
                                    )
                                    events.append(
                                        {
                                            "type": "tool_call",
                                            "tool_name": tc_name,
                                            "tool_call_id": tc_id,
                                            "args": tc_args or {},
                                        }
                                    )
                        continue

                    for block in blocks:
                        btype = block.get("type")
                        if btype == "text":
                            text = block.get("text", "")
                            if text:
                                assistant_text_parts.append(text)
                        elif btype in ("tool_call_chunk", "tool_call"):
                            chunk_name = block.get("name")
                            chunk_args = block.get("args")
                            chunk_id = block.get("id")
                            chunk_index = block.get("index")

                            key: str | int
                            if chunk_index is not None:
                                key = chunk_index
                            elif chunk_id is not None:
                                key = chunk_id
                            else:
                                key = f"unknown-{len(tool_call_buffers)}"

                            buffer = tool_call_buffers.setdefault(
                                key,
                                {
                                    "name": None,
                                    "id": None,
                                    "args": None,
                                    "args_parts": [],
                                },
                            )
                            if chunk_name:
                                buffer["name"] = chunk_name
                            if chunk_id:
                                buffer["id"] = chunk_id

                            parsed_args: Any = None
                            if isinstance(chunk_args, dict):
                                parsed_args = chunk_args
                            elif isinstance(chunk_args, str):
                                parts: list[str] = buffer.setdefault("args_parts", [])
                                if chunk_args and (
                                    not parts or chunk_args != parts[-1]
                                ):
                                    parts.append(chunk_args)
                                try:
                                    parsed_args = json.loads("".join(parts))
                                except json.JSONDecodeError:
                                    parsed_args = None

                            if (
                                buffer.get("name")
                                and buffer.get("id")
                                and isinstance(parsed_args, dict)
                            ):
                                tc_name = buffer["name"]
                                tc_id = buffer["id"]
                                if tc_id not in displayed_tool_ids:
                                    displayed_tool_ids.add(tc_id)
                                    tool_call_args[tc_id] = parsed_args
                                    tool_call_names[tc_id] = tc_name
                                    await file_op_tracker.start_operation(
                                        tc_name, parsed_args, tc_id
                                    )
                                    events.append(
                                        {
                                            "type": "tool_call",
                                            "tool_name": tc_name,
                                            "tool_call_id": tc_id,
                                            "args": parsed_args,
                                        }
                                    )
                                tool_call_buffers.pop(key, None)

            if interrupt_occurred:
                # Persist pending interrupts for resume
                self._pending_interrupts[session.thread_id] = pending_interrupts

                # Auto-approve in-session if enabled
                if session.auto_approve:
                    hitl_response: dict[str, Any] = {}
                    for iid, hitl_req in pending_interrupts.items():
                        action_reqs = []
                        if isinstance(hitl_req, dict):
                            action_reqs = hitl_req.get("action_requests", []) or []
                        hitl_response[iid] = {
                            "decisions": [{"type": "approve"} for _ in action_reqs]
                        }
                    stream_input = Command(resume=hitl_response)
                    continue

                # Otherwise return approval request to the UI
                approval_payload = []
                for iid, hitl_req in pending_interrupts.items():
                    action_reqs = []
                    if isinstance(hitl_req, dict):
                        action_reqs = hitl_req.get("action_requests", []) or []
                    approval_payload.append(
                        {"interrupt_id": iid, "action_requests": action_reqs}
                    )

                return {
                    "thread_id": session.thread_id,
                    "assistant": "".join(assistant_text_parts).strip(),
                    "events": events,
                    "approval_required": True,
                    "approvals": approval_payload,
                    "auto_approve": session.auto_approve,
                }

            # No interrupt, done.
            return {
                "thread_id": session.thread_id,
                "assistant": "".join(assistant_text_parts).strip(),
                "events": events,
                "approval_required": False,
                "auto_approve": session.auto_approve,
            }


def _stringify_tool_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
            else:
                try:
                    out.append(json.dumps(item))
                except Exception:
                    out.append(str(item))
        return "\n".join(out)
    return str(content)


async def _readline() -> str:
    return await asyncio.to_thread(sys.stdin.readline)


async def main() -> None:
    # Configure logging to stderr so we can see model info
    logging.basicConfig(
        level=logging.INFO, format="[deepagents-daemon] %(message)s", stream=sys.stderr
    )
    logging.info(
        f"Starting daemon with OPENROUTER_MODEL={os.environ.get('OPENROUTER_MODEL', 'NOT SET')}"
    )

    # Keep checkpointer open for the lifetime of the daemon
    async with get_checkpointer() as checkpointer:
        runtime = AgentRuntime(checkpointer)

        while True:
            line = await _readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                req_id = req.get("id")
                method = req.get("method")
                params = req.get("params") or {}

                if method == "health":
                    result = {"status": "ok"}
                elif method == "clear_caches":
                    result = runtime.clear_caches()
                elif method == "new_thread":
                    result = await runtime.new_thread()
                elif method == "chat":
                    thread_id = params.get("thread_id") or generate_thread_id()
                    assistant_id = params.get("assistant_id") or "agent"
                    message = params.get("message") or ""
                    auto_approve = bool(params.get("auto_approve", False))
                    result = await runtime.chat(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                        message=message,
                        auto_approve=auto_approve,
                    )
                elif method == "resume":
                    thread_id = params.get("thread_id")
                    assistant_id = params.get("assistant_id") or "agent"
                    interrupt_id = params.get("interrupt_id")
                    decision = params.get("decision")
                    if not thread_id or not interrupt_id or not decision:
                        raise ValueError(
                            "resume requires thread_id, interrupt_id, decision"
                        )
                    result = await runtime.resume(
                        thread_id=thread_id,
                        assistant_id=assistant_id,
                        interrupt_id=interrupt_id,
                        decision=decision,
                    )
                elif method == "list_threads":
                    # List all threads with optional agent filter
                    agent_name = params.get("agent_name")
                    limit = params.get("limit", 50)
                    threads = await list_threads(agent_name, limit=limit)
                    # Enhance with previews
                    from deepagents_cli.sessions import get_thread_preview

                    for t in threads:
                        t["preview"] = await get_thread_preview(t["thread_id"])
                    result = {"threads": threads}
                elif method == "delete_thread":
                    # Delete a thread
                    thread_id = params.get("thread_id")
                    if not thread_id:
                        raise ValueError("delete_thread requires thread_id")
                    deleted = await delete_thread(thread_id)
                    result = {"deleted": deleted, "thread_id": thread_id}
                elif method == "get_thread_messages":
                    # Get messages from a thread
                    from deepagents_cli.sessions import get_thread_messages

                    thread_id = params.get("thread_id")
                    limit = params.get("limit", 50)
                    if not thread_id:
                        raise ValueError("get_thread_messages requires thread_id")
                    messages = await get_thread_messages(thread_id, limit=limit)
                    result = {"messages": messages, "thread_id": thread_id}
                elif method == "chat_stream":
                    # Streaming chat - emit events line by line
                    thread_id = params.get("thread_id") or generate_thread_id()
                    assistant_id = params.get("assistant_id") or "agent"
                    message = params.get("message") or ""
                    auto_approve = bool(params.get("auto_approve", False))
                    model = params.get("model")  # Optional: use UI-selected model
                    attachments = params.get(
                        "attachments"
                    )  # Optional: file attachments
                    plan_mode = bool(params.get("plan_mode", False))
                    logging.info(
                        f"chat_stream: model={model!r}, thread={thread_id}, "
                        f"attachments={len(attachments) if attachments else 0}, plan_mode={plan_mode}"
                    )
                    try:
                        async for event in runtime.chat_stream(
                            thread_id=thread_id,
                            assistant_id=assistant_id,
                            message=message,
                            auto_approve=auto_approve,
                            model=model,
                            attachments=attachments,
                            plan_mode=plan_mode,
                        ):
                            # Check if stream was cancelled
                            if runtime.is_stream_cancelled(req_id):
                                logging.info(f"Stream {req_id} cancelled, stopping")
                                runtime.clear_cancelled_stream(req_id)
                                # Emit done event so client knows stream ended
                                done_event = {"id": req_id, "ok": True, "event": {"type": "done", "cancelled": True}}
                                sys.stdout.write(json.dumps(done_event) + "\n")
                                sys.stdout.flush()
                                break
                            event_line = {"id": req_id, "ok": True, "event": event}
                            sys.stdout.write(json.dumps(event_line) + "\n")
                            sys.stdout.flush()
                    except Exception as e:
                        err_resp = {
                            "id": req_id,
                            "ok": False,
                            "error": sanitize_error(e),
                        }
                        sys.stdout.write(json.dumps(err_resp) + "\n")
                        sys.stdout.flush()
                    continue  # Skip the normal response write
                elif method == "cancel_stream":
                    # Cancel a streaming request
                    stream_id = params.get("stream_id")
                    if stream_id:
                        result = runtime.cancel_stream(stream_id)
                    else:
                        result = {"cancelled": False, "error": "No stream_id provided"}
                elif method == "resume_stream":
                    # Streaming resume - emit events line by line
                    thread_id = params.get("thread_id")
                    assistant_id = params.get("assistant_id") or "agent"
                    interrupt_id = params.get("interrupt_id")
                    decision = params.get("decision")
                    if not thread_id or not interrupt_id or not decision:
                        raise ValueError(
                            "resume_stream requires thread_id, interrupt_id, decision"
                        )
                    try:
                        async for event in runtime.resume_stream(
                            thread_id=thread_id,
                            assistant_id=assistant_id,
                            interrupt_id=interrupt_id,
                            decision=decision,
                        ):
                            # Check if stream was cancelled
                            if runtime.is_stream_cancelled(req_id):
                                logging.info(f"Stream {req_id} cancelled, stopping")
                                runtime.clear_cancelled_stream(req_id)
                                # Emit done event so client knows stream ended
                                done_event = {"id": req_id, "ok": True, "event": {"type": "done", "cancelled": True}}
                                sys.stdout.write(json.dumps(done_event) + "\n")
                                sys.stdout.flush()
                                break
                            event_line = {"id": req_id, "ok": True, "event": event}
                            sys.stdout.write(json.dumps(event_line) + "\n")
                            sys.stdout.flush()
                    except Exception as e:
                        err_resp = {
                            "id": req_id,
                            "ok": False,
                            "error": sanitize_error(e),
                        }
                        sys.stdout.write(json.dumps(err_resp) + "\n")
                        sys.stdout.flush()
                    continue  # Skip the normal response write
                else:
                    raise ValueError(f"Unknown method: {method}")

                resp = {"id": req_id, "ok": True, "result": result}
            except Exception as e:
                resp = {
                    "id": req.get("id") if isinstance(req, dict) else None,
                    "ok": False,
                    "error": sanitize_error(e),
                }

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
