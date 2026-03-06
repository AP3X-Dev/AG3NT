"""Planning middleware for structured planning before execution.

This middleware enforces a planning-first approach where the agent must:
1. Analyze the user's request
2. Create a detailed plan using write_todos
3. Present the plan for user approval
4. Execute the plan step-by-step

The middleware tracks planning state across turns and modifies the system
prompt to guide the agent through planning and execution phases.
"""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
import logging
import threading

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from deepagents.middleware._utils import append_to_system_message
from ag3nt_agent.plan_mode_guard import PlanModeGuard

logger = logging.getLogger(__name__)


@dataclass
class PlanningState:
    """Tracks planning mode state for a session."""

    enabled: bool = False  # Whether planning mode is active
    planning_phase: bool = True  # True = planning, False = executing
    plan_confirmed: bool = False  # Whether user confirmed the plan
    plan_tasks: list[str] = field(default_factory=list)  # List of task descriptions
    current_task_index: int = 0  # Current task being executed

    # Context engineering extensions (all backward-compatible defaults)
    context_engineering: bool = False  # Whether PRP-style planning is active
    context_package: str = ""  # Serialized context for prompt injection
    active_blueprint_id: str | None = None  # Active blueprint ID
    validation_gates_enabled: bool = False  # Run validation between tasks

    # Plan execution tracking (set on confirm, None when not available)
    plan_executor: Any = None  # PlanExecutor instance
    plan_recorder: Any = None  # PlanLearningRecorder instance


class PlanningMiddleware(AgentMiddleware[AgentState, Any]):
    """Middleware that enforces planning before execution.

    When enabled, this middleware:
    - Forces the agent to create a plan before taking any actions
    - Prevents file writes/edits during planning phase
    - Tracks execution progress through planned tasks
    - Provides progress updates as tasks complete
    """

    # Tools blocked during planning phase
    BLOCKED_TOOLS = frozenset({
        "shell", "exec_command", "write_file", "edit_file",
        "delete_file", "apply_patch", "multi_edit", "git_commit",
    })

    def __init__(self, yolo_mode: bool = False):
        self.sessions: dict[str, PlanningState] = {}
        self.tools = []  # Required by AgentMiddleware
        self._lock = threading.Lock()
        self.yolo_mode = yolo_mode
        logger.info(f"PlanningMiddleware initialized (yolo_mode={yolo_mode})")

    def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Intercept before model call to enforce planning."""
        # Get thread_id from runtime config
        config = getattr(runtime, 'config', {}) or {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id", "default")

        with self._lock:
            # Get or create planning state for this session
            if thread_id not in self.sessions:
                # Check if plan_mode enabled in metadata
                metadata = config.get("metadata", {})
                plan_mode = metadata.get("plan_mode", False)
                context_engineering = metadata.get("context_engineering", False)
                self.sessions[thread_id] = PlanningState(
                    enabled=plan_mode,
                    context_engineering=context_engineering,
                    validation_gates_enabled=context_engineering,
                )
                if plan_mode:
                    logger.info(f"Planning mode activated for session {thread_id}")
                if context_engineering:
                    logger.info(f"Context engineering activated for session {thread_id}")

            plan_state = self.sessions[thread_id]

        # If planning mode not enabled, pass through
        if not plan_state.enabled:
            return None

        # Trigger context gathering on first call when context_engineering is on.
        # This populates plan_state.context_package so it's ready by the time
        # wrap_model_call injects it into the system prompt.
        if (
            plan_state.context_engineering
            and plan_state.planning_phase
            and not plan_state.context_package
        ):
            plan_state.context_package = self._gather_context_sync(
                state, config, thread_id,
            )

        # System message injection is done in wrap_model_call / awrap_model_call
        # using request.override(system_message=...) — NOT via state updates.
        return None

    def _get_planning_prompt(self) -> str:
        """Get planning instructions to add to system prompt."""
        return """

🎯 **PLANNING MODE ACTIVE**

You are in planning mode. Before executing any actions, you MUST create a detailed plan.

## Planning Process

1. **Enter Planning**
   Call `plan_enter(title="...", mode="feature")` to create a plan file.
   - Use mode="feature" for focused tasks
   - Use mode="project" for complex multi-component work

2. **Analyze & Plan**
   - Read files, search the codebase, explore
   - Edit the plan file with your findings and steps
   - Structure steps as: `- [ ] Step N: Description [CHECKPOINT]`
   - Add [CHECKPOINT] to steps that need validation

3. **Submit for Approval**
   Call `plan_exit(summary="...")` when your plan is ready.
   The user will review and accept or reject.

## During Planning Phase

✅ **You CAN:** Read files, search, explore, edit the plan file
❌ **You CANNOT:** Write/edit other files, run shell commands, make code changes

---
**Remember:** Call `plan_enter` first, explore, then `plan_exit` when ready.
"""

    def _get_execution_prompt(self, plan_state: PlanningState) -> str:
        """Get execution guidance with progress tracking."""
        total_tasks = len(plan_state.plan_tasks)
        current = plan_state.current_task_index + 1

        # Use PlanExecutor's step note if available
        executor = plan_state.plan_executor
        if executor and plan_state.current_task_index < len(executor.steps):
            step = executor.steps[plan_state.current_task_index]
            return executor.get_step_system_note(step)

        # Fallback to generic execution prompt
        return f"""

✅ **PLAN APPROVED - EXECUTION MODE**

Execute the approved plan step-by-step.

**Progress:** Task {current}/{total_tasks}

## Execution Guidelines

For each task:
1. **Announce:** State which task you're starting
2. **Execute:** Perform the task carefully
3. **Verify:** Check that changes work as expected
4. **Update:** Mark task complete with `update_todo`
5. **Report:** Briefly summarize what was done

## Current Status

Tasks completed: {plan_state.current_task_index}
Tasks remaining: {total_tasks - plan_state.current_task_index}

After completing each task, provide a brief progress update.
"""

    def _get_context_planning_prompt(self, context_text: str) -> str:
        """Get enhanced planning prompt with gathered context."""
        return f"""

🎯 **CONTEXT-ENGINEERED PLANNING MODE**

You are in planning mode with context engineering enabled.
Rich context has been gathered from the codebase, past learnings,
and semantic memory to help you create a better plan.

## Gathered Context

{context_text}

## Planning Process

1. **Review the Context Above**
   - Examine the relevant code references
   - Note any anti-patterns to avoid
   - Consider past learnings and gotchas

2. **Create a Structured Blueprint**
   Use `write_blueprint` to create a PRP-style plan with:
   - **goal**: What needs to be achieved
   - **why**: Business/technical reason
   - **what**: Detailed description of changes
   - **tasks**: Ordered list with pseudocode, files, dependencies, and complexity
   - **success_criteria**: How to verify the work is correct
   - **anti_patterns**: Things to avoid (from context above)
   - **gotchas**: Potential issues to watch for

3. **Present for Approval**
   - Summarize the blueprint
   - Highlight any risks or assumptions
   - Ask user to confirm before proceeding

## During Planning Phase

✅ **You CAN:**
- Read files to understand context
- Search for information
- Use memory_search, codebase_search
- Create a blueprint with `write_blueprint`

❌ **You CANNOT:**
- Write, edit, or delete files
- Execute shell commands
- Make any changes to code

---

**Remember:** Use `write_blueprint` (not `write_todos`) for structured planning.
"""

    def _get_execution_with_validation_prompt(self, plan_state: PlanningState) -> str:
        """Get execution prompt with validation gate instructions."""
        total_tasks = len(plan_state.plan_tasks)
        current = plan_state.current_task_index + 1

        return f"""

✅ **BLUEPRINT APPROVED - EXECUTION WITH VALIDATION**

Execute the approved blueprint step-by-step with validation gates.

**Progress:** Task {current}/{total_tasks}

## Execution Guidelines

For each task:
1. **Announce:** State which task you're starting
2. **Execute:** Perform the task carefully
3. **Validate:** After completing a task, run the appropriate validation:
   - Level 1 (Syntax): Lint/format check on modified files
   - Level 2 (Unit Test): Run relevant unit tests
   - Level 3 (Integration): Run integration tests if applicable
4. **Update:** Mark task complete with `update_blueprint_task`
5. **Report:** Summarize what was done and validation results

## Current Status

Tasks completed: {plan_state.current_task_index}
Tasks remaining: {total_tasks - plan_state.current_task_index}

If a validation gate fails, fix the issues before moving to the next task.
"""

    def _gather_context_sync(
        self, state: Any, config: dict, thread_id: str,
    ) -> str:
        """Synchronously gather context for planning.

        Runs the async ``ContextGatherer`` in a thread pool to avoid
        blocking the middleware's synchronous ``before_model`` path.
        """
        try:
            import asyncio
            from ag3nt_agent.context_gatherer import ContextGatherer

            # Extract user message from state for the query
            user_request = ""
            messages = getattr(state, "messages", None)
            if messages:
                for msg in reversed(messages):
                    content = getattr(msg, "content", "")
                    role = getattr(msg, "type", "") or getattr(msg, "role", "")
                    if role in ("human", "user") and content:
                        user_request = content if isinstance(content, str) else str(content)
                        break

            if not user_request:
                return ""

            gatherer = ContextGatherer()

            async def _do_gather() -> str:
                package = await gatherer.gather_context(user_request, session_id=thread_id)
                return package.to_prompt_text()

            # Try to run in existing loop, fall back to new loop in thread
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _do_gather())
                    return future.result(timeout=15)
            except RuntimeError:
                return asyncio.run(_do_gather())

        except Exception as exc:
            logger.warning("Context gathering failed: %s", exc)
            return ""

    def confirm_plan(self, thread_id: str, tasks: list[str]):
        """User confirmed the plan - switch to execution mode."""
        with self._lock:
            if thread_id not in self.sessions:
                return
            plan_state = self.sessions[thread_id]
            plan_state.plan_confirmed = True
            plan_state.planning_phase = False
            plan_state.plan_tasks = tasks

        # Create PlanExecutor and PlanLearningRecorder outside the lock
        # to avoid holding the lock during I/O (file reads, imports).

        # Create PlanExecutor if plan file exists
        try:
            from ag3nt_agent.plan_tools import PlanState as ToolPlanState
            tool_state = ToolPlanState.get(thread_id)
            plan_path = tool_state.active_plan
            if plan_path and plan_path.exists():
                from ag3nt_agent.plan_executor import PlanExecutor
                plan_content = plan_path.read_text(encoding="utf-8")
                plan_state.plan_executor = PlanExecutor(
                    plan_path=str(plan_path),
                    plan_content=plan_content,
                )
                logger.info(
                    "PlanExecutor created for session %s (%d steps)",
                    thread_id, len(plan_state.plan_executor.steps),
                )
        except Exception as exc:
            logger.debug("PlanExecutor not available: %s", exc)

        # Create PlanLearningRecorder
        try:
            from ag3nt_agent.plan_learning import PlanLearningRecorder
            plan_state.plan_recorder = PlanLearningRecorder()
        except Exception as exc:
            logger.debug("PlanLearningRecorder not available: %s", exc)

        logger.info(f"Plan confirmed for session {thread_id}: {len(tasks)} tasks")

    def advance_task(self, thread_id: str):
        """Move to next task in plan, with optional executor tracking."""
        deferred_learning: tuple | None = None

        with self._lock:
            if thread_id not in self.sessions:
                return
            plan_state = self.sessions[thread_id]

            # Use PlanExecutor to mark step complete and emit events
            executor = plan_state.plan_executor
            if executor and plan_state.current_task_index < len(executor.steps):
                step = executor.steps[plan_state.current_task_index]
                try:
                    executor.mark_step_complete(step)
                    executor.emit_step_completed(step, success=True)

                    # Create git checkpoint if step has [CHECKPOINT]
                    if step.is_checkpoint:
                        executor.create_git_checkpoint(step.number)
                except Exception as exc:
                    logger.debug("PlanExecutor step completion error: %s", exc)

            plan_state.current_task_index += 1

            # Check if all tasks done
            if plan_state.current_task_index >= len(plan_state.plan_tasks):
                # Emit execution completed
                if executor:
                    try:
                        executor.emit_execution_completed(success=True)
                    except Exception as exc:
                        logger.debug("PlanExecutor completion emit error: %s", exc)

                # Defer learning recording to outside the lock to avoid
                # deadlock when _record_learning_outcome calls asyncio.run()
                deferred_learning = (plan_state, thread_id, True)
            else:
                # Emit next step started
                if executor and plan_state.current_task_index < len(executor.steps):
                    try:
                        next_step = executor.steps[plan_state.current_task_index]
                        executor.emit_step_started(next_step)
                        executor.create_git_checkpoint(next_step.number)
                    except Exception as exc:
                        logger.debug("PlanExecutor next step emit error: %s", exc)

                logger.info(f"Advanced to task {plan_state.current_task_index + 1}")

        # Fire-and-forget learning recording outside the lock
        if deferred_learning:
            self._record_learning_outcome(*deferred_learning)

    def run_validation_gate(
        self, thread_id: str, files_modified: list[str] | None = None,
    ) -> dict | None:
        """Run validation gate for current task if enabled.

        Returns validation result dict or None if gates not enabled.
        """
        with self._lock:
            plan_state = self.sessions.get(thread_id)

        if not plan_state or not plan_state.validation_gates_enabled:
            return None

        try:
            import asyncio
            from ag3nt_agent.validation_gates import ValidationGateRunner, ValidationLevel

            runner = ValidationGateRunner()

            # Run syntax check on modified files
            async def _run():
                result = await runner.run_gate(
                    ValidationLevel.SYNTAX, files_modified=files_modified,
                )
                return {
                    "passed": result.passed,
                    "details": result.details,
                    "level": "SYNTAX",
                }

            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _run())
                    return future.result(timeout=30)
            except RuntimeError:
                return asyncio.run(_run())

        except Exception as exc:
            logger.debug("Validation gate failed: %s", exc)
            return None

    def _record_learning_outcome(
        self, plan_state: PlanningState, thread_id: str, success: bool,
    ):
        """Record blueprint outcome for future learning (fire-and-forget)."""
        recorder = plan_state.plan_recorder
        if not recorder:
            return

        try:
            import asyncio

            # Create a minimal blueprint-like object for the recorder
            class _MiniBP:
                def __init__(self, tasks, goal):
                    self.id = thread_id
                    self.session_id = thread_id
                    self.goal = goal
                    self.why = ""
                    self.status = "completed" if success else "failed"
                    self.learnings = []
                    self.tasks = tasks

            class _MiniTask:
                def __init__(self, title, completed):
                    self.title = title
                    self.status = "completed" if completed else "pending"
                    self.complexity = "medium"
                    self.files_involved = []
                    self.validation_result = None
                    self.notes = ""

            mini_tasks = [
                _MiniTask(t, i < plan_state.current_task_index)
                for i, t in enumerate(plan_state.plan_tasks)
            ]
            bp = _MiniBP(
                mini_tasks,
                plan_state.plan_tasks[0] if plan_state.plan_tasks else "plan",
            )

            async def _record():
                await recorder.record_blueprint_outcome(bp, success=success)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_record())
            except RuntimeError:
                asyncio.run(_record())
        except Exception as exc:
            logger.debug("Learning recording failed: %s", exc)

    def disable_plan_mode(self, thread_id: str):
        """Disable planning mode for a session."""
        with self._lock:
            if thread_id in self.sessions:
                self.sessions[thread_id].enabled = False
                logger.info(f"Planning mode disabled for session {thread_id}")

    def get_state(self, thread_id: str) -> PlanningState | None:
        """Get planning state for a session."""
        with self._lock:
            return self.sessions.get(thread_id)

    def handle_plan_decision(self, thread_id: str, decision: str) -> bool:
        """Handle a plan confirmation decision.

        Args:
            thread_id: Session ID
            decision: "approve" or "reject"

        Returns:
            True if decision was handled, False if no plan pending
        """
        with self._lock:
            plan_state = self.sessions.get(thread_id)
            if not plan_state or not plan_state.enabled:
                return False
            if plan_state.plan_confirmed:
                return False

            if decision == "approve":
                plan_state.plan_confirmed = True
                plan_state.planning_phase = False
                logger.info(f"Plan approved for session {thread_id}")
                return True
            elif decision == "reject":
                # Reset planning phase to let agent create a new plan
                plan_state.plan_tasks = []
                logger.info(f"Plan rejected for session {thread_id}, resetting")
                return True

        return False

    def _get_thread_id_from_request(self, request: ModelRequest) -> str:
        """Extract thread_id from a model request's config."""
        config = getattr(request, 'config', {}) or {}
        configurable = config.get("configurable", {})
        return configurable.get("thread_id", "default")

    def _compute_planning_prompt(self, thread_id: str) -> str | None:
        """Return the planning/execution prompt text for this session, or None."""
        with self._lock:
            plan_state = self.sessions.get(thread_id)

        if not plan_state or not plan_state.enabled:
            return None

        if plan_state.planning_phase and not plan_state.plan_confirmed:
            if plan_state.context_engineering and plan_state.context_package:
                base = self._get_context_planning_prompt(plan_state.context_package)
            else:
                base = self._get_planning_prompt()
            # Inject past learnings (default-on, no opt-in required)
            learnings_text = self._retrieve_learnings(plan_state)
            if learnings_text:
                base = f"{base}\n\n{learnings_text}"
            return base
        elif plan_state.plan_confirmed:
            if plan_state.validation_gates_enabled:
                return self._get_execution_with_validation_prompt(plan_state)
            return self._get_execution_prompt(plan_state)

        return None

    @staticmethod
    def _retrieve_learnings(plan_state: PlanningState) -> str:
        """Retrieve past learnings relevant to the current planning topic.

        Uses both the context engine (vector search) and keyword search
        over local blueprint/plan files. Returns formatted prompt text
        or empty string if no learnings found.
        """
        try:
            from ag3nt_agent.learning_retrieval import (
                format_learnings_for_prompt,
                retrieve_learnings,
            )

            query = plan_state.original_request or ""
            if not query:
                return ""
            learnings = retrieve_learnings(query, limit=5)
            return format_learnings_for_prompt(learnings)
        except Exception:
            logger.debug("Learning retrieval failed", exc_info=True)
            return ""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Intercept model calls to inject planning prompt and block write tools."""
        thread_id = self._get_thread_id_from_request(request)
        prompt_text = self._compute_planning_prompt(thread_id)
        if prompt_text:
            new_sys = append_to_system_message(request.system_message, prompt_text)
            request = request.override(system_message=new_sys)
        response = handler(request)
        return self._filter_blocked_tools(request, response)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        thread_id = self._get_thread_id_from_request(request)
        prompt_text = self._compute_planning_prompt(thread_id)
        if prompt_text:
            new_sys = append_to_system_message(request.system_message, prompt_text)
            request = request.override(system_message=new_sys)
        response = await handler(request)
        return self._filter_blocked_tools(request, response)

    def _filter_blocked_tools(
        self, request: ModelRequest, response: ModelResponse
    ) -> ModelResponse:
        """Check response tool calls and block write tools during planning.

        Also detects write_todos calls to trigger plan confirmation interrupt.
        """
        thread_id = self._get_thread_id_from_request(request)

        with self._lock:
            plan_state = self.sessions.get(thread_id)

        # If not in planning mode or plan is confirmed, allow everything
        if not plan_state or not plan_state.enabled:
            return response
        if plan_state.plan_confirmed:
            return response
        if not plan_state.planning_phase:
            return response

        # YOLO mode: auto-confirm plans without interrupt
        if self.yolo_mode:
            # Check for write_todos or write_blueprint call and auto-confirm
            for tc in response.tool_calls if hasattr(response, 'tool_calls') and response.tool_calls else []:
                tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                if tool_name == "write_todos":
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    tasks = args.get("todos", args.get("tasks", []))
                    task_descs = []
                    if isinstance(tasks, list):
                        for t in tasks:
                            if isinstance(t, str):
                                task_descs.append(t)
                            elif isinstance(t, dict):
                                task_descs.append(t.get("description", t.get("title", str(t))))
                    self.confirm_plan(thread_id, task_descs)
                    logger.info(f"YOLO mode: auto-confirmed plan for {thread_id}")
                elif tool_name == "write_blueprint":
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    bp_tasks = args.get("tasks", [])
                    task_descs = [
                        t.get("title", str(t)) if isinstance(t, dict) else str(t)
                        for t in bp_tasks
                    ]
                    self.confirm_plan(thread_id, task_descs)
                    logger.info(f"YOLO mode: auto-confirmed blueprint for {thread_id}")
            return response

        # Check if response has tool calls
        if not hasattr(response, 'tool_calls') or not response.tool_calls:
            return response

        # Check for write_todos or write_blueprint call — signals a plan was created
        for tc in response.tool_calls:
            tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            if tool_name == "write_todos":
                # Extract plan tasks from the tool call args
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                tasks = args.get("todos", args.get("tasks", []))
                if isinstance(tasks, list):
                    task_descriptions = []
                    for t in tasks:
                        if isinstance(t, str):
                            task_descriptions.append(t)
                        elif isinstance(t, dict):
                            task_descriptions.append(t.get("description", t.get("title", str(t))))
                    with self._lock:
                        if thread_id in self.sessions:
                            self.sessions[thread_id].plan_tasks = task_descriptions
                    logger.info(
                        f"Plan created for session {thread_id}: {len(task_descriptions)} tasks, awaiting confirmation"
                    )
            elif tool_name == "write_blueprint":
                args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                bp_tasks = args.get("tasks", [])
                task_descriptions = [
                    t.get("title", str(t)) if isinstance(t, dict) else str(t)
                    for t in bp_tasks
                ]
                with self._lock:
                    if thread_id in self.sessions:
                        self.sessions[thread_id].plan_tasks = task_descriptions
                        # Store blueprint ID if available from tool result later
                logger.info(
                    f"Blueprint created for session {thread_id}: {len(task_descriptions)} tasks, awaiting confirmation"
                )

        # Use PlanModeGuard for smart path-aware blocking when plan_enter
        # has been called.  Fall back to the simpler BLOCKED_TOOLS set when
        # plan mode was toggled via the UI but plan_enter hasn't run yet.
        blocked = []
        allowed = []
        guard_active = PlanModeGuard.is_active()
        for tc in response.tool_calls:
            tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

            if guard_active:
                # PlanModeGuard handles path-aware blocking
                is_ok, reason = PlanModeGuard.is_tool_allowed(tool_name, tc_args)
            else:
                # Fallback: simple BLOCKED_TOOLS check (UI toggled plan mode
                # but plan_enter hasn't run yet)
                is_ok = tool_name not in self.BLOCKED_TOOLS
                reason = f"Tool '{tool_name}' blocked during planning phase" if not is_ok else ""

            if is_ok:
                allowed.append(tc)
            else:
                blocked.append(f"{tool_name}: {reason}")

        if not blocked:
            return response

        logger.warning(
            f"Planning phase blocked tools for session {thread_id}: {blocked}"
        )

        # Remove blocked tool calls from response
        if hasattr(response, 'override'):
            return response.override(tool_calls=allowed)

        return response
