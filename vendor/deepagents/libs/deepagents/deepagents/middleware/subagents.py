"""Middleware for providing subagents to an agent via a `task` tool."""

import asyncio
import contextvars
import copy
from collections import deque
from pathlib import Path
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from deepagents.middleware._utils import append_to_system_message

logger = __import__("logging").getLogger(__name__)

# Types that are inherently immutable and never need copying
_IMMUTABLE_TYPES = (str, int, float, bool, type(None), tuple, frozenset, bytes)


def _smart_copy(value: Any) -> Any:
    """Return immutable values as-is, deepcopy mutable ones.

    Avoids the overhead of deepcopy for types that can never be mutated
    (str, int, float, bool, None, tuple, frozenset, bytes), which typically
    make up 60-70% of state values.
    """
    if isinstance(value, _IMMUTABLE_TYPES):
        return value
    return copy.deepcopy(value)


# ContextVar for bridging subagent progress events to external systems (e.g. StreamManager).
# The worker sets this to a callback before each turn; _emit_progress calls it
# in addition to the LangGraph stream_writer.
subagent_progress_callback: contextvars.ContextVar[Callable[[dict[str, Any]], None] | None] = contextvars.ContextVar(
    "subagent_progress_callback", default=None
)


def _merge_value(existing: Any, incoming: Any, key: str) -> Any:
    """Merge two values from parallel task results.

    - dict values: shallow merge ({**existing, **incoming})
    - list values: concatenate (existing + incoming)
    - other types: last-write-wins with debug log
    """
    if isinstance(existing, dict) and isinstance(incoming, dict):
        return {**existing, **incoming}
    if isinstance(existing, list) and isinstance(incoming, list):
        return existing + incoming
    logger.debug(
        "parallel_tasks merge: key '%s' overwritten (type %s), last-write-wins",
        key,
        type(incoming).__name__,
    )
    return incoming


class SubagentMailbox:
    """Message queue for parent <-> child agent communication.

    Safe for single-threaded async use (cooperative multitasking).
    """

    def __init__(self) -> None:
        self._queue: deque[dict[str, str]] = deque()

    def send(self, sender: str, content: str) -> None:
        """Enqueue a message."""
        self._queue.append({"sender": sender, "content": content})

    def receive(self) -> dict[str, str] | None:
        """Dequeue and return the next message, or None if empty."""
        if self._queue:
            return self._queue.popleft()
        return None

    def drain(self) -> list[dict[str, str]]:
        """Dequeue and return all pending messages."""
        msgs = list(self._queue)
        self._queue.clear()
        return msgs

    def empty(self) -> bool:
        """Check if the mailbox is empty."""
        return len(self._queue) == 0


def _load_base_prompt() -> str:
    """Load the base agent prompt from base_prompt.md."""
    prompt_path = Path(__file__).parent.parent / "base_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "You are a helpful AI assistant."


GENERAL_PURPOSE_SUBAGENT: dict[str, Any] = {
    "name": "general-purpose",
    "description": "A general-purpose agent that can handle a wide range of tasks autonomously.",
    "system_prompt": _load_base_prompt(),
}


class SubAgent(TypedDict):
    """Specification for an agent.

    When specifying custom agents, the `default_middleware` from `SubAgentMiddleware`
    will be applied first, followed by any `middleware` specified in this spec.
    To use only custom middleware without the defaults, pass `default_middleware=[]`
    to `SubAgentMiddleware`.

    Required fields:
        name: Unique identifier for the subagent.

            The main agent uses this name when calling the `task()` tool.
        description: What this subagent does.

            Be specific and action-oriented. The main agent uses this to decide when to delegate.
        system_prompt: Instructions for the subagent.

            Include tool usage guidance and output format requirements.
        tools: Tools the subagent can use.

            Keep this minimal and include only what's needed.

    Optional fields:
        model: Override the main agent's model.

            Use the format `'provider:model-name'` (e.g., `'openai:gpt-4o'`).
        middleware: Additional middleware for custom behavior, logging, or rate limiting.
        interrupt_on: Configure human-in-the-loop for specific tools.

            Requires a checkpointer.
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does. The main agent uses this to decide when to delegate."""

    system_prompt: str
    """Instructions for the subagent."""

    tools: Sequence[BaseTool | Callable | dict[str, Any]]
    """Tools the subagent can use."""

    model: NotRequired[str | BaseChatModel]
    """Override the main agent's model. Use `'provider:model-name'` format."""

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Configure human-in-the-loop for specific tools."""

    mailbox: NotRequired[SubagentMailbox]
    """Optional mailbox for parent <-> child message passing."""


class CompiledSubAgent(TypedDict):
    """A pre-compiled agent spec.

    !!! note

        The runnable's state schema must include a 'messages' key.

        This is required for the subagent to communicate results back to the main agent.

    When the subagent completes, the final message in the 'messages' list will be
    extracted and returned as a `ToolMessage` to the parent agent.
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does."""

    runnable: Runnable
    """A custom agent implementation.

    Create a custom agent using either:

    1. LangChain's [`create_agent()`](https://docs.langchain.com/oss/python/langchain/quickstart)
    2. A custom graph using [`langgraph`](https://docs.langchain.com/oss/python/langgraph/quickstart)

    If you're creating a custom graph, make sure the state schema includes a 'messages' key.
    This is required for the subagent to communicate results back to the main agent.
    """


DEFAULT_SUBAGENT_PROMPT = "In order to complete the objective that the user asks of you, you have access to a number of standard tools."

# State keys that are excluded when passing state to subagents and when returning
# updates from subagents.
# When returning updates:
# 1. The messages key is handled explicitly to ensure only the final message is included
# 2. The todos and structured_response keys are excluded as they do not have a defined reducer
#    and no clear meaning for returning them from a subagent to the main agent.
_EXCLUDED_STATE_KEYS = {"messages", "todos", "structured_response"}

TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task`tool.
</commentary>
</example>

### Example usage with specialized agents:

IMPORTANT: You must ONLY use agent types that are listed in the "Available agent types" section above.
Do NOT invent, combine, or make up agent type names. If a type is not listed above, it does not exist.

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since code was written, use the reviewer agent to review it for correctness and best practices.
</commentary>
assistant: Now let me use the reviewer agent to check the code
assistant: Uses the Task tool with subagent_type="reviewer" to review the code
</example>

<example>
user: "Can you help me research the environmental impact of renewable energy and write a report?"
<commentary>
This requires both research and writing. Use the researcher agent first to gather information,
then the writer agent to produce the final report.
</commentary>
assistant: I'll research this topic first, then write the report.
assistant: Uses the Task tool with subagent_type="researcher" to gather information on renewable energy environmental impact
assistant: *Receives research results*
assistant: Now let me write the report based on this research.
assistant: Uses the Task tool with subagent_type="writer" to produce a polished report from the research findings
</example>

<example>
user: "Analyze the sales data in sales.csv and find trends"
<commentary>
This is a data analysis task, use the analyst agent.
</commentary>
assistant: Uses the Task tool with subagent_type="analyst" to analyze sales.csv and identify trends
</example>"""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- **CRITICAL: When you have 2 or more independent tasks to delegate, use `parallel_tasks` instead of calling `task` multiple times.** The `parallel_tasks` tool runs all subagents concurrently, which is significantly faster than sequential `task` calls. Only use sequential `task` when a later task depends on the result of an earlier one.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient.

## `parallel_tasks` (concurrent subagent execution)

You also have access to a `parallel_tasks` tool that runs multiple subagents concurrently.

When to use `parallel_tasks`:
- When you have 2+ independent research or analysis tasks
- When you need to analyze multiple files, topics, or areas simultaneously
- When tasks do NOT depend on each other's results

When NOT to use `parallel_tasks`:
- When task B depends on the result of task A (use sequential `task` calls instead)
- When you only have one task (use `task` instead)"""  # noqa: E501


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501


def _get_subagents(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Create subagent instances from specifications.

    Args:
        default_model: Default model for subagents that don't specify one.
        default_tools: Default tools for subagents that don't specify tools.
        default_middleware: Middleware to apply to all subagents. If `None`,
            no default middleware is applied.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent. These
            are also the fallback for any subagents that don't specify their own tool configs.
        subagents: List of agent specifications or pre-compiled agents.
        general_purpose_agent: Whether to include a general-purpose subagent.

    Returns:
        Tuple of (agent_dict, description_list) where agent_dict maps agent names
        to runnable instances and description_list contains formatted descriptions.
    """
    # Use empty list if None (no default middleware)
    default_subagent_middleware = default_middleware or []

    agents: dict[str, Any] = {}
    subagent_descriptions = []

    # Create general-purpose agent if enabled
    if general_purpose_agent:
        general_purpose_middleware = [*default_subagent_middleware]
        if default_interrupt_on:
            general_purpose_middleware.append(HumanInTheLoopMiddleware(interrupt_on=default_interrupt_on))
        general_purpose_subagent = create_agent(
            default_model,
            system_prompt=DEFAULT_SUBAGENT_PROMPT,
            tools=default_tools,
            middleware=general_purpose_middleware,
            name="general-purpose",
        )
        agents["general-purpose"] = general_purpose_subagent
        subagent_descriptions.append(f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")

    # Process custom subagents
    for agent_ in subagents:
        subagent_descriptions.append(f"- {agent_['name']}: {agent_['description']}")
        if "runnable" in agent_:
            custom_agent = cast("CompiledSubAgent", agent_)
            agents[custom_agent["name"]] = custom_agent["runnable"]
            continue
        _tools = agent_.get("tools", list(default_tools))

        subagent_model = agent_.get("model", default_model)

        _middleware = [*default_subagent_middleware, *agent_["middleware"]] if "middleware" in agent_ else [*default_subagent_middleware]

        interrupt_on = agent_.get("interrupt_on", default_interrupt_on)
        if interrupt_on:
            _middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

        agents[agent_["name"]] = create_agent(
            subagent_model,
            system_prompt=agent_["system_prompt"],
            tools=_tools,
            middleware=_middleware,
            name=agent_["name"],
        )
    return agents, subagent_descriptions


def _find_closest_subagent(invalid_type: str, valid_types: dict[str, Any]) -> str | None:
    """Find the closest matching valid subagent type using word overlap.

    When the LLM hallucinates a combined name like "research-analyst", this
    tries to match it to a real type (e.g. "analyst") so the agent can recover
    instead of getting stuck.
    """
    invalid_words = invalid_type.lower().replace("_", "-").split("-")

    best_match: str | None = None
    best_score = 0

    for valid_type in valid_types:
        valid_words = valid_type.lower().replace("_", "-").split("-")
        score = 0

        for iw in invalid_words:
            for vw in valid_words:
                if iw == vw:
                    score += 3
                elif iw.startswith(vw) or vw.startswith(iw):
                    score += 2

        # Substring of the full name
        if invalid_type.lower() in valid_type.lower() or valid_type.lower() in invalid_type.lower():
            score += 1

        if score > best_score:
            best_score = score
            best_match = valid_type

    return best_match if best_score > 0 else None


def _create_task_tool(
    *,
    subagent_graphs: dict[str, Any],
    subagent_description_str: str,
    task_description: str | None = None,
) -> BaseTool:
    """Create a task tool for invoking subagents.

    Args:
        subagent_graphs: Pre-built mapping of subagent names to runnable instances.
        subagent_description_str: Formatted descriptions of available subagents.
        task_description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        # Validate that the result contains a 'messages' key
        if "messages" not in result:
            error_msg = (
                "CompiledSubAgent must return a state containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )
            raise ValueError(error_msg)

        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        # Strip trailing whitespace to prevent API errors with Anthropic
        message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(subagent_type: str, description: str, runtime: ToolRuntime) -> tuple[Runnable, dict]:
        """Prepare state for invocation."""
        subagent = subagent_graphs[subagent_type]
        # Shallow-copy each value to prevent subagent mutations from corrupting parent state.
        # copy.copy creates new top-level containers (lists, dicts) without the cost of deepcopy.
        subagent_state = {k: copy.copy(v) for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state

    # Use custom description if provided, otherwise use default template
    if task_description is None:
        task_description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        # If custom description has placeholder, format with agent descriptions
        task_description = task_description.format(available_agents=subagent_description_str)

    def _start_monitoring(runtime, subagent_type, description):
        """Start monitoring, returns (monitor, exec_id) or (None, None)."""
        try:
            from ag3nt_agent.subagent_monitor import get_subagent_monitor
            monitor = get_subagent_monitor()
            execution = monitor.start_execution(
                parent_id=runtime.tool_call_id or "unknown",
                subagent_type=subagent_type,
                task=description[:500],
            )
            return monitor, execution.id
        except Exception:
            logger.debug("Subagent monitor unavailable, skipping lifecycle tracking", exc_info=True)
            return None, None

    def _end_monitoring(monitor, exec_id, result=None, error=None):
        """End monitoring with result or error."""
        if not exec_id or not monitor:
            return
        try:
            result_text = None
            if error is None and isinstance(result, dict):
                msgs = result.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    result_text = str(getattr(last_msg, "content", ""))[:500]
            if error:
                monitor.end_execution(exec_id, error=str(error)[:500])
            else:
                monitor.end_execution(exec_id, result=result_text)
        except Exception:
            logger.debug("Failed to record subagent execution end", exc_info=True)

    def task(
        description: Annotated[
            str,
            "A detailed description of the task for the subagent to perform autonomously. Include all necessary context and specify the expected output format.",  # noqa: E501
        ],
        subagent_type: Annotated[str, "The type of subagent to use. Must be one of the available agent types listed in the tool description."],
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            closest = _find_closest_subagent(subagent_type, subagent_graphs)
            if closest:
                logger.warning("Subagent type '%s' not found, auto-corrected to '%s'", subagent_type, closest)
                subagent_type = closest
            else:
                allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
                return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)

        monitor, exec_id = _start_monitoring(runtime, subagent_type, description)

        try:
            result = subagent.invoke(subagent_state)
        except Exception as e:
            _end_monitoring(monitor, exec_id, error=e)
            raise

        _end_monitoring(monitor, exec_id, result=result)

        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: Annotated[
            str,
            "A detailed description of the task for the subagent to perform autonomously. Include all necessary context and specify the expected output format.",  # noqa: E501
        ],
        subagent_type: Annotated[str, "The type of subagent to use. Must be one of the available agent types listed in the tool description."],
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            closest = _find_closest_subagent(subagent_type, subagent_graphs)
            if closest:
                logger.warning("Subagent type '%s' not found, auto-corrected to '%s'", subagent_type, closest)
                subagent_type = closest
            else:
                allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
                return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime)

        monitor, exec_id = _start_monitoring(runtime, subagent_type, description)

        try:
            result = await subagent.ainvoke(subagent_state)
        except Exception as e:
            _end_monitoring(monitor, exec_id, error=e)
            raise

        _end_monitoring(monitor, exec_id, result=result)

        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=task_description,
    )


def _create_parallel_tasks_tool(
    *,
    subagent_graphs: dict[str, Any],
    subagent_description_str: str,
) -> BaseTool:
    """Create a parallel_tasks tool for concurrent subagent execution.

    Args:
        subagent_graphs: Pre-built mapping of subagent names to runnable instances.
        subagent_description_str: Formatted descriptions of available subagents.

    Returns:
        A StructuredTool that runs multiple subagents concurrently.
    """

    # Per-subagent timeout (5 minutes)
    _SUBAGENT_TIMEOUT = 300

    def _emit_progress(rt: ToolRuntime, data: dict[str, Any]) -> None:
        """Emit a progress event via stream_writer and the bridge callback."""
        writer = getattr(rt, "stream_writer", None)
        if writer is not None:
            try:
                writer(data)
            except Exception:
                pass  # Don't let stream errors break execution
        # Also forward to the bridge callback so events reach StreamManager
        bridge = subagent_progress_callback.get(None)
        if bridge is not None:
            try:
                bridge(data)
            except Exception:
                pass

    async def aparallel_tasks(
        tasks: Annotated[
            list[dict[str, str]],
            "List of task specs. Each must have 'description' (detailed task) and 'subagent_type' (agent type).",
        ],
        runtime: ToolRuntime,
    ) -> str | Command:
        if not tasks:
            return "No tasks provided."
        if len(tasks) > 10:
            return "Maximum 10 concurrent tasks allowed."

        # Validate all tasks first
        validated: list[tuple[str, str]] = []
        for i, spec in enumerate(tasks):
            desc = spec.get("description", "")
            stype = spec.get("subagent_type", "")
            if not desc or not stype:
                return f"Task {i}: missing 'description' or 'subagent_type'"
            if stype not in subagent_graphs:
                closest = _find_closest_subagent(stype, subagent_graphs)
                if closest:
                    stype = closest
                else:
                    allowed = ", ".join(f"`{k}`" for k in subagent_graphs)
                    return f"Task {i}: invalid subagent_type '{stype}'. Allowed: {allowed}"
            validated.append((desc, stype))

        total = len(validated)
        completed_count = 0
        _lock = asyncio.Lock()

        # Emit initial progress
        _emit_progress(runtime, {
            "type": "parallel_tasks_progress",
            "status": "started",
            "total": total,
            "completed": 0,
            "tasks": [{"index": i, "subagent_type": s, "status": "pending"} for i, (_, s) in enumerate(validated)],
        })

        async def _run_one(desc: str, stype: str, idx: int) -> tuple[int, str, dict[str, Any] | Exception]:
            nonlocal completed_count
            logger.info("parallel_tasks: starting task %d/%d (%s)", idx + 1, total, stype)

            _emit_progress(runtime, {
                "type": "parallel_tasks_progress",
                "status": "running",
                "task_index": idx,
                "subagent_type": stype,
                "message": f"Task {idx + 1}/{total} ({stype}) started",
            })

            subagent = subagent_graphs[stype]
            # Copy state to avoid cross-task interference.
            # _smart_copy returns immutable values as-is and deepcopies mutable ones,
            # skipping expensive deepcopy for ~60-70% of typical state values.
            state = {k: _smart_copy(v) for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
            state["messages"] = [HumanMessage(content=desc)]
            try:
                result = await asyncio.wait_for(
                    subagent.ainvoke(state),
                    timeout=_SUBAGENT_TIMEOUT,
                )
                async with _lock:
                    completed_count += 1
                logger.info("parallel_tasks: task %d/%d (%s) completed (%d/%d done)", idx + 1, total, stype, completed_count, total)
                _emit_progress(runtime, {
                    "type": "parallel_tasks_progress",
                    "status": "task_complete",
                    "task_index": idx,
                    "subagent_type": stype,
                    "completed": completed_count,
                    "total": total,
                    "message": f"Task {idx + 1}/{total} ({stype}) completed ({completed_count}/{total} done)",
                })
                return idx, stype, result
            except asyncio.TimeoutError:
                async with _lock:
                    completed_count += 1
                logger.warning("parallel_tasks: task %d (%s) timed out after %ds", idx + 1, stype, _SUBAGENT_TIMEOUT)
                _emit_progress(runtime, {
                    "type": "parallel_tasks_progress",
                    "status": "task_error",
                    "task_index": idx,
                    "subagent_type": stype,
                    "message": f"Task {idx + 1} ({stype}) timed out after {_SUBAGENT_TIMEOUT}s",
                })
                return idx, stype, TimeoutError(f"Subagent timed out after {_SUBAGENT_TIMEOUT}s")
            except Exception as e:
                async with _lock:
                    completed_count += 1
                logger.warning("parallel_tasks: task %d (%s) failed: %s", idx + 1, stype, e)
                _emit_progress(runtime, {
                    "type": "parallel_tasks_progress",
                    "status": "task_error",
                    "task_index": idx,
                    "subagent_type": stype,
                    "message": f"Task {idx + 1} ({stype}) failed: {e}",
                })
                return idx, stype, e

        # Run all concurrently
        results = await asyncio.gather(
            *[_run_one(d, s, i) for i, (d, s) in enumerate(validated)]
        )

        # Combine results
        parts: list[str] = []
        merged_state: dict[str, Any] = {}
        for idx, stype, result in results:
            if isinstance(result, Exception):
                parts.append(f"## Task {idx + 1} ({stype}): ERROR\n{result}")
            elif "messages" in result and result["messages"]:
                text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
                parts.append(f"## Task {idx + 1} ({stype}): COMPLETE\n{text}")
                for k, v in result.items():
                    if k not in _EXCLUDED_STATE_KEYS:
                        if k in merged_state:
                            merged_state[k] = _merge_value(merged_state[k], v, k)
                        else:
                            merged_state[k] = v
            else:
                parts.append(f"## Task {idx + 1} ({stype}): COMPLETE\n(no output)")

        combined = "\n\n---\n\n".join(parts)
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for parallel_tasks invocation"
            raise ValueError(value_error_msg)

        _emit_progress(runtime, {
            "type": "parallel_tasks_progress",
            "status": "all_complete",
            "total": total,
            "completed": total,
            "message": f"All {total} tasks completed",
        })

        return Command(
            update={
                **merged_state,
                "messages": [ToolMessage(combined, tool_call_id=runtime.tool_call_id)],
            }
        )

    def parallel_tasks_sync(
        tasks: Annotated[
            list[dict[str, str]],
            "List of task specs. Each must have 'description' and 'subagent_type'.",
        ],
        runtime: ToolRuntime,
    ) -> str:
        return "parallel_tasks requires async execution. Use ainvoke."

    description = (
        "Execute multiple subagent tasks concurrently for maximum performance.\n\n"
        "Use this instead of calling `task` multiple times when you have 2 or more independent tasks.\n"
        "All tasks run in parallel and results are returned together, significantly reducing total execution time.\n\n"
        "Available agent types:\n"
        f"{subagent_description_str}\n\n"
        "Each task in the list needs:\n"
        "- description: Detailed task description with all necessary context\n"
        "- subagent_type: One of the available agent types\n\n"
        "Example:\n"
        '  parallel_tasks(tasks=[\n'
        '    {"description": "Research topic A thoroughly and return findings", "subagent_type": "general-purpose"},\n'
        '    {"description": "Research topic B thoroughly and return findings", "subagent_type": "general-purpose"}\n'
        "  ])"
    )

    return StructuredTool.from_function(
        name="parallel_tasks",
        func=parallel_tasks_sync,
        coroutine=aparallel_tasks,
        description=description,
    )


class SubAgentMiddleware(AgentMiddleware):
    """Middleware for providing subagents to an agent via a `task` tool.

    This  middleware adds a `task` tool to the agent that can be used to invoke subagents.
    Subagents are useful for handling complex tasks that require multiple steps, or tasks
    that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks, and then return
    a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require a narrower
    subset of tools and focus.

    This middleware comes with a default general-purpose subagent that can be used to
    handle the same tasks as the main agent, but with isolated context.

    Args:
        default_model: The model to use for subagents.

            Can be a `LanguageModelLike` or a dict for `init_chat_model`.
        default_tools: The tools to use for the default general-purpose subagent.
        default_middleware: Default middleware to apply to all subagents.

            If `None`, no default middleware is applied.

            Pass a list to specify custom middleware.
        default_interrupt_on: The tool configs to use for the default general-purpose subagent.

            These are also the fallback for any subagents that don't specify their own tool configs.
        subagents: A list of additional subagents to provide to the agent.
        system_prompt: Full system prompt override. When provided, completely replaces
            the agent's system prompt.
        general_purpose_agent: Whether to include the general-purpose agent.
        task_description: Custom description for the task tool.

            If `None`, uses the default description template.

    Example:
        ```python
        from langchain.agents.middleware.subagents import SubAgentMiddleware
        from langchain.agents import create_agent

        # Basic usage with defaults (no default middleware)
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    subagents=[],
                )
            ],
        )

        # Add custom middleware to subagents
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                SubAgentMiddleware(
                    default_model="openai:gpt-4o",
                    default_middleware=[TodoListMiddleware()],
                    subagents=[],
                )
            ],
        )
        ```
    """

    def __init__(
        self,
        *,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        default_middleware: list[AgentMiddleware] | None = None,
        default_interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        general_purpose_agent: bool = True,
        task_description: str | None = None,
    ) -> None:
        """Initialize the `SubAgentMiddleware`."""
        super().__init__()
        self.system_prompt = system_prompt
        # Cache computed system messages to avoid redundant allocations.
        # Keyed by id(original_system_message), bounded to 4 entries.
        self._system_message_cache: dict[int, Any] = {}

        # Build subagent graphs once, shared by both task and parallel_tasks tools
        subagent_graphs, subagent_descriptions = _get_subagents(
            default_model=default_model,
            default_tools=default_tools or [],
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
        )
        subagent_description_str = "\n".join(subagent_descriptions)

        task_tool = _create_task_tool(
            subagent_graphs=subagent_graphs,
            subagent_description_str=subagent_description_str,
            task_description=task_description,
        )
        parallel_tool = _create_parallel_tasks_tool(
            subagent_graphs=subagent_graphs,
            subagent_description_str=subagent_description_str,
        )
        self.tools = [task_tool, parallel_tool]

    def _get_cached_system_message(self, system_message: Any) -> Any:
        """Return a cached system message, computing and caching on first access."""
        key = id(system_message)
        cached = self._system_message_cache.get(key)
        if cached is not None:
            return cached
        result = append_to_system_message(system_message, self.system_prompt)
        # Bound cache to 4 entries to prevent unbounded growth
        if len(self._system_message_cache) >= 4:
            # Evict oldest entry
            oldest_key = next(iter(self._system_message_cache))
            del self._system_message_cache[oldest_key]
        self._system_message_cache[key] = result
        return result

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = self._get_cached_system_message(request.system_message)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """(async) Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = self._get_cached_system_message(request.system_message)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
