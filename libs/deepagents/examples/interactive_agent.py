"""Interactive demo of DeepAgents with all middleware.

This script creates a fully-featured agent with all available middleware
and allows you to interact with it in a chat-like interface.

Features:
- Streaming output so you can see the agent thinking in real-time
- Tool call visibility - see which tools are being called and their results
- Full conversation history for multi-turn interactions
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from deepagents.backends import FilesystemBackend
from deepagents.middleware import (
    AdvancedMiddleware,
    FilesystemMiddleware,
    ImageGenerationMiddleware,
    SubAgentMiddleware,
    UtilitiesMiddleware,
    WebMiddleware,
)

# Fix Windows console encoding for emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Loaded environment from: {env_path}")
else:
    print(f"[WARN] No .env file found at: {env_path}")


async def main():
    """Run interactive agent demo with streaming output."""
    # Get OpenRouter configuration from environment
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

    if not openrouter_key:
        print("=" * 70)
        print("[ERROR] OPENROUTER_API_KEY not found")
        print("=" * 70)
        print()
        print("Please set your OpenRouter API key in the .env file:")
        print(f"  Location: {Path(__file__).parent.parent.parent.parent / '.env'}")
        print()
        print("Add this line to .env:")
        print("  OPENROUTER_API_KEY=your-key-here")
        print()
        return

    # Set up OpenRouter using ChatOpenAI with custom base_url
    print("=" * 70)
    print("DeepAgents - Interactive Demo (Streaming Mode)")
    print("=" * 70)
    print()
    print(f"Model: {openrouter_model}")
    print("API: OpenRouter")
    print()
    print("Initializing agent with all middleware...")
    print()

    # Create ChatOpenAI model configured for OpenRouter
    model = ChatOpenAI(
        model=openrouter_model,
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        streaming=True,  # Enable streaming
    )

    # Set up filesystem backend with access to user's home directory
    # virtual_mode=True makes paths relative to root_dir (e.g., "/Downloads" -> "C:\Users\Guerr\Downloads")
    user_home = Path.home()
    print(f"Filesystem root: {user_home}")
    fs_backend = FilesystemBackend(root_dir=str(user_home), virtual_mode=True)

    # Create a custom filesystem system prompt that explains the layout
    fs_system_prompt = f"""You have access to the user's home directory filesystem.
The root "/" represents: {user_home}

Common directories you can access:
- /Downloads - User's downloads folder
- /Documents - User's documents folder
- /Desktop - User's desktop folder
- /Pictures - User's pictures folder

Use these paths directly (e.g., "ls /Downloads" or "glob *.png /Downloads").
"""

    # Create agent with all middleware
    # Pass model to middlewares that support vision/AI capabilities
    try:
        agent = create_agent(
            model=model,
            middleware=[
                FilesystemMiddleware(backend=fs_backend, system_prompt=fs_system_prompt),
                AdvancedMiddleware(backend=fs_backend, model=model),  # Vision analysis
                ImageGenerationMiddleware(backend=fs_backend),  # Image generation & editing
                UtilitiesMiddleware(),
                WebMiddleware(),
                SubAgentMiddleware(default_model=model),
                TodoListMiddleware(),
            ],
        )
        print("[OK] Agent initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Error initializing agent: {e}")
        import traceback

        traceback.print_exc()
        return

    print()
    print("Available capabilities:")
    print("  [FILES] Filesystem: read, write, search files")
    print("  [SEARCH] Advanced: semantic search, media analysis")
    print("  [IMAGE] Image AI: generate & edit images (Gemini 3 Pro)")
    print("  [TOOLS] Utilities: formatting, diagnostics, undo edits, mermaid diagrams")
    print("  [WEB] Web: search, fetch web pages")
    print("  [AGENTS] SubAgents: delegate tasks to specialized agents")
    print("  [TODO] Todos: task planning and tracking")
    print()
    print("=" * 70)
    print()
    print("Example prompts:")
    print("  - 'Search the web for the latest Python best practices'")
    print("  - 'ls Downloads' - list files in Downloads folder")
    print("  - 'Create a plan with 3 steps for building a REST API'")
    print("  - 'Generate an image of a futuristic city at sunset'")
    print()
    print("Type 'quit', 'exit', or press Ctrl+C to stop.")
    print("=" * 70)

    # Interactive loop
    conversation_history = []

    while True:
        try:
            user_input = input("\n[You] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n[Goodbye!]")
                break

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})

            print()

            # Stream agent response with retry on transient errors
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    await stream_agent_response(agent, conversation_history)
                    break  # Success, exit retry loop
                except KeyboardInterrupt:
                    print("\n[Interrupted]")
                    break
                except Exception as e:
                    error_msg = str(e)
                    # Check for transient/retryable errors
                    is_retryable = any(
                        x in error_msg.lower() for x in ["unexpected error", "try your request again", "rate limit", "timeout", "connection"]
                    )

                    if is_retryable and attempt < max_retries:
                        print(f"\n[Retrying... ({attempt + 1}/{max_retries})]")
                        await asyncio.sleep(1)  # Brief pause before retry
                        continue
                    print(f"\n[ERROR] {e}")
                    conversation_history.pop()
                    break

        except KeyboardInterrupt:
            print("\n\n[Goodbye!]")
            break
        except EOFError:
            print("\n\n[Goodbye!]")
            break


async def stream_agent_response(agent, conversation_history: list):
    """Stream agent response with visible tool calls and thinking."""
    accumulated_text = ""
    tool_call_buffers = {}  # Track partial tool calls

    print("[Agent] ", end="", flush=True)

    async for chunk in agent.astream(
        {"messages": conversation_history},
        stream_mode=["messages", "updates"],
    ):
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            continue

        stream_mode, data = chunk

        # Handle message streaming (text and tool calls)
        if stream_mode == "messages":
            if not isinstance(data, tuple) or len(data) != 2:
                continue

            message, metadata = data

            # Handle tool result messages
            if isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", "unknown")
                tool_status = getattr(message, "status", "success")

                # Format tool result
                if tool_status == "success":
                    print(f"\n    ✓ {tool_name} completed", flush=True)
                else:
                    content = str(message.content)[:100] if message.content else "Error"
                    print(f"\n    ✗ {tool_name} failed: {content}", flush=True)
                continue

            # Handle AI message chunks (streaming text and tool calls)
            if hasattr(message, "content_blocks"):
                for block in message.content_blocks:
                    block_type = block.get("type")

                    # Stream text output
                    if block_type == "text":
                        text = block.get("text", "")
                        if text:
                            print(text, end="", flush=True)
                            accumulated_text += text

                    # Show tool calls as they're made
                    elif block_type in ("tool_call_chunk", "tool_call"):
                        chunk_name = block.get("name")
                        chunk_args = block.get("args")
                        chunk_id = block.get("id")
                        chunk_index = block.get("index", 0)

                        # Buffer tool call chunks
                        buffer_key = chunk_index if chunk_index is not None else chunk_id or 0
                        buffer = tool_call_buffers.setdefault(buffer_key, {"name": None, "id": None, "args": "", "shown": False})

                        if chunk_name:
                            buffer["name"] = chunk_name
                        if chunk_id:
                            buffer["id"] = chunk_id
                        if isinstance(chunk_args, str):
                            buffer["args"] += chunk_args
                        elif isinstance(chunk_args, dict):
                            buffer["args"] = json.dumps(chunk_args)

                        # Show tool call once we have the name
                        if buffer["name"] and not buffer["shown"]:
                            buffer["shown"] = True
                            # Parse and format args for display
                            args_preview = ""
                            if buffer["args"]:
                                try:
                                    args_dict = json.loads(buffer["args"]) if isinstance(buffer["args"], str) else buffer["args"]
                                    # Show first arg value as preview
                                    if isinstance(args_dict, dict):
                                        first_val = next(iter(args_dict.values()), "")
                                        if isinstance(first_val, str) and len(first_val) > 40:
                                            first_val = first_val[:40] + "..."
                                        args_preview = f": {first_val}"
                                except:
                                    pass
                            print(f"\n    → {buffer['name']}{args_preview}", flush=True)

        # Handle updates (for completed actions)
        elif stream_mode == "updates":
            if isinstance(data, dict):
                # Check for todo updates
                for node_name, update in data.items():
                    if node_name == "tools" and isinstance(update, dict):
                        if "todos" in update:
                            todos = update["todos"]
                            if todos:
                                print("\n    [TODO List Updated]", flush=True)

    # Add final response to conversation history
    if accumulated_text:
        conversation_history.append({"role": "assistant", "content": accumulated_text})

    print()  # Final newline


if __name__ == "__main__":
    asyncio.run(main())
