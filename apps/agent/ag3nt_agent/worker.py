"""AG3NT agent worker.

FastAPI RPC server that hosts the DeepAgents runtime.
The Gateway calls this worker to run session turns.

Endpoints:
- POST /turn: Run a conversation turn
- POST /resume: Resume an interrupted turn after approval/rejection
- GET /health: Health check
- WS /ws: Persistent WebSocket connection for low-latency communication
- GET /subagents: List all registered subagents
- GET /subagents/{name}: Get a specific subagent
- POST /subagents: Register a new custom subagent
- DELETE /subagents/{name}: Unregister a custom subagent
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from ag3nt_agent.agent_config import GATEWAY_TOKEN
from pydantic import BaseModel

# WebSocket connection logger
ws_logger = logging.getLogger("ag3nt.websocket")


def _create_tracked_task(coro, *, name: str | None = None) -> asyncio.Task:
    """Create an asyncio Task with exception logging via done_callback.

    Prevents 'Task exception was never retrieved' warnings by attaching
    a callback that logs any unhandled exceptions.
    """
    task = asyncio.create_task(coro, name=name)

    def _done_cb(t: asyncio.Task) -> None:
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            ws_logger.error("Unhandled exception in background task %s: %s", t.get_name(), exc, exc_info=exc)

    task.add_done_callback(_done_cb)
    return task

from ag3nt_agent.deepagents_runtime import (
    run_turn as deepagents_run_turn,
    resume_turn as deepagents_resume_turn,
)
from ag3nt_agent.errors import get_error_registry
from ag3nt_agent.subagent_registry import SubagentRegistry
from ag3nt_agent.subagent_configs import SubagentConfig

app = FastAPI(title="ag3nt-agent")


# =============================================================================
# Agent Pre-Warming (Latency Optimization)
# =============================================================================

# Check if agent pool is enabled
_use_agent_pool = os.environ.get("AG3NT_USE_AGENT_POOL", "false").lower() == "true"
_use_autonomous = os.environ.get("AG3NT_AUTONOMOUS_ENABLED", "false").lower() == "true"


@app.on_event("startup")
async def prewarm_agent():
    """Pre-warm the agent on startup.

    If AG3NT_USE_AGENT_POOL=true, initializes the agent pool with pre-warmed instances.
    Otherwise, creates a singleton agent.

    This eliminates cold-start latency (50-200ms) on the first request.
    """
    import asyncio
    import logging
    import os

    logger = logging.getLogger("ag3nt.worker")

    if _use_agent_pool:
        # Use agent pool for pre-warming
        from ag3nt_agent.agent_pool import initialize_pool_async
        logger.info("Pre-warming agent pool...")
        pool_size = int(os.environ.get("AG3NT_POOL_SIZE", "3"))
        await initialize_pool_async(pool_size=pool_size)
        logger.info(f"Agent pool pre-warm complete ({pool_size} agents)")
    else:
        # Use singleton pre-warming
        def _prewarm():
            from ag3nt_agent.deepagents_runtime import get_agent
            logger.info("Pre-warming agent singleton...")
            get_agent()
            logger.info("Agent pre-warm complete")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _prewarm)


@app.on_event("startup")
async def start_autonomous_system():
    """Start the autonomous event-driven system if enabled.

    Enable with AG3NT_AUTONOMOUS_ENABLED=true.
    """
    import logging
    logger = logging.getLogger("ag3nt.worker")

    if not _use_autonomous:
        return

    try:
        from ag3nt_agent.deepagents_runtime import start_autonomous_system
        await start_autonomous_system()
        logger.info("Autonomous system started")
    except Exception as e:
        logger.error(f"Failed to start autonomous system: {e}")


@app.on_event("shutdown")
async def shutdown_systems():
    """Gracefully shutdown agent pool and autonomous system."""
    import logging
    logger = logging.getLogger("ag3nt.worker")

    # Shutdown autonomous system
    if _use_autonomous:
        try:
            from ag3nt_agent.deepagents_runtime import stop_autonomous_system
            await stop_autonomous_system()
            logger.info("Autonomous system stopped")
        except Exception as e:
            logger.error(f"Error stopping autonomous system: {e}")

    # Shutdown agent pool
    if _use_agent_pool:
        try:
            from ag3nt_agent.agent_pool import shutdown_pool
            shutdown_pool()
            logger.info("Agent pool shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down agent pool: {e}")


@app.middleware("http")
async def verify_gateway_token(request: Request, call_next):
    """Verify X-Gateway-Token header on /turn and /resume endpoints."""
    if GATEWAY_TOKEN and request.url.path in ("/turn", "/resume"):
        token = request.headers.get("X-Gateway-Token", "")
        if token != GATEWAY_TOKEN:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing gateway token"},
            )
    return await call_next(request)


# =============================================================================
# Request/Response Models
# =============================================================================


class TurnRequest(BaseModel):
    session_id: str
    text: str
    metadata: dict | None = None


class UsageInfo(BaseModel):
    """Token usage information for tracking and billing."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = "unknown"
    provider: str = "unknown"


class InterruptInfo(BaseModel):
    """Information about an interrupt requiring approval or user input.

    For tool approval interrupts:
        - pending_actions: List of actions needing approval
        - action_count: Number of actions

    For user question interrupts:
        - type: "user_question"
        - question: The question to ask
        - options: Optional list of choices
        - allow_custom: Whether custom answers are allowed
    """

    interrupt_id: str
    # Tool approval fields (optional)
    pending_actions: list[dict] | None = None
    action_count: int | None = None
    # User question fields (optional)
    type: str | None = None
    question: str | None = None
    options: list[str] | None = None
    allow_custom: bool | None = None


class TurnResponse(BaseModel):
    """Response from a turn, may include interrupt for approval."""

    session_id: str
    text: str
    events: list[dict] = []
    interrupt: InterruptInfo | None = None
    usage: UsageInfo | None = None


class ResumeRequest(BaseModel):
    """Request to resume an interrupted turn."""

    session_id: str
    decisions: list[dict]  # Each with {"type": "approve"} or {"type": "reject"}


class ResumeResponse(BaseModel):
    """Response from resuming a turn."""

    session_id: str
    text: str
    events: list[dict] = []
    interrupt: InterruptInfo | None = None
    usage: UsageInfo | None = None


class SubagentConfigRequest(BaseModel):
    """Request to create a new subagent."""

    name: str
    description: str
    system_prompt: str
    tools: list[str] = []
    max_tokens: int = 8000
    max_turns: int = 3
    model_override: str | None = None
    thinking_mode: str = "disabled"
    priority: int = 100


class SubagentResponse(BaseModel):
    """Response for a subagent."""

    name: str
    description: str
    source: str
    system_prompt: str
    tools: list[str]
    max_tokens: int
    max_turns: int
    model_override: str | None
    thinking_mode: str
    priority: int


class AutonomousEventRequest(BaseModel):
    """Request to publish an event to the autonomous system."""

    event_type: str
    source: str
    payload: dict = {}
    priority: str = "MEDIUM"


class AutonomousStatusResponse(BaseModel):
    """Response with autonomous system status."""

    enabled: bool
    running: bool
    event_bus: dict | None = None
    goals: dict | None = None


class PoolStatsResponse(BaseModel):
    """Response with agent pool statistics."""

    enabled: bool
    stats: dict | None = None


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"ok": True, "name": "ag3nt-agent"}


@app.get("/pool/stats", response_model=PoolStatsResponse)
def pool_stats():
    """Get agent pool statistics.

    Returns pool hit rate, current size, and usage metrics.
    Only available when AG3NT_USE_AGENT_POOL=true.
    """
    if not _use_agent_pool:
        return PoolStatsResponse(enabled=False, stats=None)

    from ag3nt_agent.deepagents_runtime import get_pool_stats
    stats = get_pool_stats()
    return PoolStatsResponse(enabled=True, stats=stats)


@app.get("/autonomous/status", response_model=AutonomousStatusResponse)
async def autonomous_status():
    """Get autonomous system status.

    Returns event bus metrics, goal status, and running state.
    Only available when AG3NT_AUTONOMOUS_ENABLED=true.
    """
    if not _use_autonomous:
        return AutonomousStatusResponse(enabled=False, running=False)

    try:
        from ag3nt_agent.deepagents_runtime import get_autonomous_runtime
        rt = get_autonomous_runtime()
        if rt is None:
            return AutonomousStatusResponse(enabled=True, running=False)
        legacy = rt.get("legacy_runtime")
        if legacy is not None:
            status = legacy.get_status()
            return AutonomousStatusResponse(
                enabled=True,
                running=status.get("running", False),
                event_bus=status.get("event_bus"),
                goals=status.get("goals"),
            )
        # No legacy runtime — report subsystem names
        return AutonomousStatusResponse(
            enabled=True,
            running=len(rt) > 0,
            event_bus={"subsystems": list(rt.keys())},
        )
    except Exception as e:
        return AutonomousStatusResponse(
            enabled=True,
            running=False,
            event_bus={"error": str(e)},
        )


@app.post("/autonomous/event")
async def publish_autonomous_event(req: AutonomousEventRequest):
    """Publish an event to the autonomous system.

    Events are processed by matching goals and may trigger autonomous
    actions or require human approval based on confidence scores.

    Only available when AG3NT_AUTONOMOUS_ENABLED=true.
    """
    if not _use_autonomous:
        raise HTTPException(
            status_code=503,
            detail="Autonomous system is not enabled. Set AG3NT_AUTONOMOUS_ENABLED=true"
        )

    try:
        from ag3nt_agent.deepagents_runtime import get_autonomous_runtime
        rt = get_autonomous_runtime()
        if rt is None:
            raise HTTPException(
                status_code=503,
                detail="Autonomous system is not running"
            )

        legacy = rt.get("legacy_runtime")
        if legacy is None or not getattr(legacy, "is_running", False):
            raise HTTPException(
                status_code=503,
                detail="Autonomous system is not running"
            )

        accepted = await legacy.publish_event(
            event_type=req.event_type,
            source=req.source,
            payload=req.payload,
            priority=req.priority,
        )

        return {
            "accepted": accepted,
            "event_type": req.event_type,
            "source": req.source,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/turn", response_model=TurnResponse)
async def turn(req: TurnRequest):
    """Run a turn through the DeepAgents runtime.

    If the agent attempts to use a risky tool, the response will include
    an `interrupt` field with details about the pending action(s).
    The client should display these to the user and call /resume with
    the user's decision.
    """
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: deepagents_run_turn(
            session_id=req.session_id,
            text=req.text,
            metadata=req.metadata,
        ),
    )

    # Build interrupt info if present
    interrupt = None
    if "interrupt" in result and result["interrupt"]:
        interrupt_data = result["interrupt"]
        interrupt = InterruptInfo(
            interrupt_id=interrupt_data["interrupt_id"],
            pending_actions=interrupt_data.get("pending_actions"),
            action_count=interrupt_data.get("action_count"),
            type=interrupt_data.get("type"),
            question=interrupt_data.get("question"),
            options=interrupt_data.get("options"),
            allow_custom=interrupt_data.get("allow_custom"),
        )

    # Build usage info if present
    usage = None
    if "usage" in result and result["usage"]:
        usage = UsageInfo(**result["usage"])

    return TurnResponse(
        session_id=result["session_id"],
        text=result["text"],
        events=result.get("events", []),
        interrupt=interrupt,
        usage=usage,
    )


@app.post("/resume", response_model=ResumeResponse)
async def resume(req: ResumeRequest):
    """Resume an interrupted turn after user approval/rejection.

    The `decisions` field should contain one decision per pending action,
    in order. Each decision is a dict with {"type": "approve"} or {"type": "reject"}.

    If the resumed execution triggers another risky tool, the response
    will again contain an `interrupt` field.
    """
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: deepagents_resume_turn(
            session_id=req.session_id,
            decisions=req.decisions,
        ),
    )

    # Build interrupt info if present
    interrupt = None
    if "interrupt" in result and result["interrupt"]:
        interrupt_data = result["interrupt"]
        interrupt = InterruptInfo(
            interrupt_id=interrupt_data["interrupt_id"],
            pending_actions=interrupt_data.get("pending_actions"),
            action_count=interrupt_data.get("action_count"),
            type=interrupt_data.get("type"),
            question=interrupt_data.get("question"),
            options=interrupt_data.get("options"),
            allow_custom=interrupt_data.get("allow_custom"),
        )

    # Build usage info if present
    usage = None
    if "usage" in result and result["usage"]:
        usage = UsageInfo(**result["usage"])

    return ResumeResponse(
        session_id=result["session_id"],
        text=result["text"],
        events=result.get("events", []),
        interrupt=interrupt,
        usage=usage,
    )


@app.get("/errors")
def get_errors():
    """Get all standardized error definitions.

    Returns a dictionary of error codes to their definitions,
    useful for clients to understand error responses.
    """
    registry = get_error_registry()
    definitions = registry.get_all_definitions()
    return {
        "ok": True,
        "errors": {
            code: {
                "code": defn.code,
                "message": defn.message,
                "http_status": defn.http_status,
                "retryable": defn.retryable,
            }
            for code, defn in definitions.items()
        }
    }


# =============================================================================
# WebSocket Endpoint (Low-Latency Gateway Connection)
# =============================================================================

# Active WebSocket connections from Gateway
_gateway_connections: dict[str, WebSocket] = {}

# Session ID -> WebSocket mapping for streaming
_session_websockets: dict[str, WebSocket] = {}

# Lock for thread-safe access to WebSocket dicts
_ws_lock = threading.Lock()

# Bounded concurrency for WebSocket task processing
_WS_MAX_CONCURRENT = int(os.environ.get("AG3NT_WS_MAX_CONCURRENT", "10"))
_ws_semaphore: asyncio.Semaphore | None = None


def _get_ws_semaphore() -> asyncio.Semaphore:
    """Get or create the WebSocket concurrency semaphore."""
    global _ws_semaphore
    if _ws_semaphore is None:
        _ws_semaphore = asyncio.Semaphore(_WS_MAX_CONCURRENT)
    return _ws_semaphore


def _build_interrupt_info(interrupt_data: dict) -> dict:
    """Build interrupt info dict from raw interrupt data."""
    return {
        "interrupt_id": interrupt_data["interrupt_id"],
        "pending_actions": interrupt_data.get("pending_actions"),
        "action_count": interrupt_data.get("action_count"),
        "type": interrupt_data.get("type"),
        "question": interrupt_data.get("question"),
        "options": interrupt_data.get("options"),
        "allow_custom": interrupt_data.get("allow_custom"),
    }


def _build_usage_info(usage_data: dict) -> dict:
    """Build usage info dict from raw usage data."""
    return {
        "input_tokens": usage_data.get("input_tokens", 0),
        "output_tokens": usage_data.get("output_tokens", 0),
        "total_tokens": usage_data.get("total_tokens", 0),
        "model": usage_data.get("model", "unknown"),
        "provider": usage_data.get("provider", "unknown"),
    }


async def _process_turn_ws(
    ws: WebSocket,
    request_id: str,
    session_id: str,
    text: str,
    metadata: dict | None,
) -> None:
    """Process a turn request over WebSocket.

    Runs the turn in a thread pool and streams tool events back in real-time.
    """
    from ag3nt_agent.streaming import get_stream_manager, ToolEvent, EventType

    start_time = time.time()
    unsubscribe = None
    _progress_token = None

    try:
        # Set up streaming: forward tool events to WebSocket
        stream_manager = get_stream_manager()

        # Set up progress bridge: subagent progress events -> StreamManager
        from deepagents.middleware.subagents import subagent_progress_callback

        def _on_subagent_progress(data: dict) -> None:
            """Bridge subagent progress events to StreamManager."""
            stream_manager.emit(ToolEvent(
                event_type=EventType.TOOL_PROGRESS,
                session_id=session_id,
                tool_name="parallel_tasks",
                tool_call_id=data.get("task_index", "parallel"),
                data=data,
            ))

        _progress_token = subagent_progress_callback.set(_on_subagent_progress)

        # Capture the running loop for thread-safe scheduling from sync callbacks
        _loop = asyncio.get_running_loop()

        def on_tool_event(event: ToolEvent) -> None:
            """Forward tool events to WebSocket.

            Pre-serializes to JSON in the worker thread to avoid blocking
            the event loop with serialization work.
            """
            try:
                payload = json.dumps({
                    "type": "stream",
                    "request_id": request_id,
                    "event": event.to_dict(),
                })
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(payload),
                    _loop,
                )
            except Exception as e:
                ws_logger.debug(f"Failed to send stream event: {e}")

        unsubscribe = stream_manager.subscribe(session_id, on_tool_event)

        # Track session -> websocket for this turn
        with _ws_lock:
            _session_websockets[session_id] = ws

        # Run the synchronous turn in a thread pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: deepagents_run_turn(
                session_id=session_id,
                text=text,
                metadata=metadata,
            ),
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Build response
        response: dict[str, Any] = {
            "type": "response",
            "id": request_id,
            "data": {
                "session_id": result["session_id"],
                "text": result["text"],
                "events": result.get("events", []),
                "latency_ms": latency_ms,
            },
        }

        # Add interrupt info if present
        if result.get("interrupt"):
            response["data"]["interrupt"] = _build_interrupt_info(result["interrupt"])

        # Add usage info if present
        if result.get("usage"):
            response["data"]["usage"] = _build_usage_info(result["usage"])

        await ws.send_json(response)
        ws_logger.debug(f"Turn completed in {latency_ms}ms for session {session_id[:16]}...")

    except Exception as e:
        ws_logger.error(f"Turn error for {session_id}: {e}")
        try:
            await ws.send_json({
                "type": "error",
                "id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
            })
        except Exception:
            ws_logger.debug(f"Failed to send error response to WS for {session_id}")
    finally:
        # Clean up progress bridge contextvar
        if _progress_token is not None:
            subagent_progress_callback.reset(_progress_token)
        # Clean up streaming subscription
        if unsubscribe:
            unsubscribe()
        with _ws_lock:
            _session_websockets.pop(session_id, None)


async def _process_resume_ws(
    ws: WebSocket,
    request_id: str,
    session_id: str,
    decisions: list[dict],
) -> None:
    """Process a resume request over WebSocket."""
    start_time = time.time()

    try:
        # Run the synchronous resume in a thread pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: deepagents_resume_turn(
                session_id=session_id,
                decisions=decisions,
            ),
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Build response
        response: dict[str, Any] = {
            "type": "response",
            "id": request_id,
            "data": {
                "session_id": result["session_id"],
                "text": result["text"],
                "events": result.get("events", []),
                "latency_ms": latency_ms,
            },
        }

        # Add interrupt info if present
        if result.get("interrupt"):
            response["data"]["interrupt"] = _build_interrupt_info(result["interrupt"])

        # Add usage info if present
        if result.get("usage"):
            response["data"]["usage"] = _build_usage_info(result["usage"])

        await ws.send_json(response)
        ws_logger.debug(f"Resume completed in {latency_ms}ms for session {session_id[:16]}...")

    except Exception as e:
        ws_logger.error(f"Resume error for {session_id}: {e}")
        try:
            await ws.send_json({
                "type": "error",
                "id": request_id,
                "error": str(e),
                "error_type": type(e).__name__,
            })
        except Exception:
            ws_logger.debug(f"Failed to send error response to WS for {session_id}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Persistent WebSocket connection for Gateway communication.

    This provides lower latency than HTTP by:
    1. Eliminating connection setup overhead
    2. Allowing request pipelining
    3. Enabling streaming responses (future)

    Message protocol:
    - Client sends: {"type": "turn"|"resume"|"ping", "id": "uuid", ...data}
    - Server sends: {"type": "response"|"error"|"pong", "id": "uuid", ...data}
    """
    # Verify gateway token from headers or query param
    token = websocket.headers.get("X-Gateway-Token", "")
    if not token:
        # Try query param as fallback (for browser testing)
        token = websocket.query_params.get("token", "")

    if GATEWAY_TOKEN and token != GATEWAY_TOKEN:
        ws_logger.warning("WebSocket connection rejected: invalid token")
        await websocket.close(code=4001, reason="Invalid or missing gateway token")
        return

    await websocket.accept()

    connection_id = str(uuid.uuid4())
    with _ws_lock:
        _gateway_connections[connection_id] = websocket
    ws_logger.info(f"Gateway WebSocket connected: {connection_id[:8]}...")

    try:
        while True:
            # Receive message
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError as e:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Invalid JSON: {e}",
                })
                continue

            msg_type = data.get("type")
            request_id = data.get("id", str(uuid.uuid4()))

            if msg_type == "ping":
                # Simple ping/pong for keepalive
                await websocket.send_json({
                    "type": "pong",
                    "id": request_id,
                    "timestamp": time.time(),
                })

            elif msg_type == "turn":
                # Process turn asynchronously (don't block other messages)
                session_id = data.get("session_id")
                text = data.get("text", "")
                metadata = data.get("metadata")

                if not session_id:
                    await websocket.send_json({
                        "type": "error",
                        "id": request_id,
                        "error": "Missing session_id",
                    })
                    continue

                # Fire and forget - process in background with bounded concurrency
                async def _guarded_turn(ws, rid, sid, txt, meta):
                    async with _get_ws_semaphore():
                        await _process_turn_ws(ws, rid, sid, txt, meta)

                _create_tracked_task(
                    _guarded_turn(websocket, request_id, session_id, text, metadata),
                    name=f"ws-turn-{session_id[:8]}",
                )

            elif msg_type == "resume":
                # Process resume asynchronously
                session_id = data.get("session_id")
                decisions = data.get("decisions", [])

                if not session_id:
                    await websocket.send_json({
                        "type": "error",
                        "id": request_id,
                        "error": "Missing session_id",
                    })
                    continue

                async def _guarded_resume(ws, rid, sid, decs):
                    async with _get_ws_semaphore():
                        await _process_resume_ws(ws, rid, sid, decs)

                _create_tracked_task(
                    _guarded_resume(websocket, request_id, session_id, decisions),
                    name=f"ws-resume-{session_id[:8]}",
                )

            else:
                await websocket.send_json({
                    "type": "error",
                    "id": request_id,
                    "error": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        ws_logger.info(f"Gateway WebSocket disconnected: {connection_id[:8]}...")
    except Exception as e:
        ws_logger.error(f"WebSocket error: {e}")
    finally:
        with _ws_lock:
            _gateway_connections.pop(connection_id, None)


@app.get("/ws/status")
def websocket_status():
    """Get WebSocket connection status."""
    with _ws_lock:
        return {
            "active_connections": len(_gateway_connections),
            "connection_ids": list(_gateway_connections.keys()),
        }


# =============================================================================
# Subagent Management Endpoints
# =============================================================================


@app.get("/subagents")
def list_subagents():
    """List all registered subagents (builtin + plugin + user-defined)."""
    registry = SubagentRegistry.get_instance()
    subagents = []
    for config in registry.list_all():
        source = registry.get_source(config.name.lower())
        subagents.append({
            "name": config.name,
            "description": config.description,
            "source": source or "unknown",
            "tools": config.tools,
            "max_tokens": config.max_tokens,
            "priority": config.priority,
        })
    return {"subagents": subagents, "count": len(subagents)}


@app.get("/subagents/{name}")
def get_subagent(name: str):
    """Get a specific subagent by name."""
    registry = SubagentRegistry.get_instance()
    config = registry.get(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Subagent '{name}' not found")

    source = registry.get_source(name.lower())
    return SubagentResponse(
        name=config.name,
        description=config.description,
        source=source or "unknown",
        system_prompt=config.system_prompt,
        tools=config.tools,
        max_tokens=config.max_tokens,
        max_turns=config.max_turns,
        model_override=config.model_override,
        thinking_mode=config.thinking_mode,
        priority=config.priority,
    )


@app.post("/subagents", status_code=201)
def create_subagent(req: SubagentConfigRequest):
    """Register a new custom subagent (user-defined)."""
    registry = SubagentRegistry.get_instance()

    # Check if it already exists
    if registry.get(req.name) is not None:
        raise HTTPException(
            status_code=409, detail=f"Subagent '{req.name}' already exists"
        )

    # Create the SubagentConfig
    config = SubagentConfig(
        name=req.name,
        description=req.description,
        system_prompt=req.system_prompt,
        tools=req.tools,
        max_tokens=req.max_tokens,
        max_turns=req.max_turns,
        model_override=req.model_override,
        thinking_mode=req.thinking_mode,
        priority=req.priority,
    )

    # Register as user-defined
    success = registry.register(config, source="user")
    if not success:
        raise HTTPException(status_code=500, detail="Failed to register subagent")

    # Persist to user data directory
    from pathlib import Path
    user_data_path = Path.home() / ".ag3nt"
    registry.save_single_config(config, user_data_path)

    return {"message": f"Subagent '{req.name}' registered successfully", "name": req.name}


@app.delete("/subagents/{name}")
def delete_subagent(name: str):
    """Unregister a custom subagent (only user-defined subagents can be deleted)."""
    registry = SubagentRegistry.get_instance()

    # Check if it exists
    config = registry.get(name)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Subagent '{name}' not found")

    # Check if it's builtin (cannot be deleted)
    source = registry.get_source(name.lower())
    if source == "builtin":
        raise HTTPException(
            status_code=403, detail="Cannot delete builtin subagents"
        )

    # Unregister
    success = registry.unregister(name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to unregister subagent")

    # Delete the config file if it exists
    from pathlib import Path
    user_data_path = Path.home() / ".ag3nt" / "subagents"
    config_file = user_data_path / f"{name.lower()}.yaml"
    if config_file.exists():
        config_file.unlink()
    # Also check for JSON
    json_file = user_data_path / f"{name.lower()}.json"
    if json_file.exists():
        json_file.unlink()

    return {"message": f"Subagent '{name}' unregistered successfully"}


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=18790)


if __name__ == "__main__":
    main()
