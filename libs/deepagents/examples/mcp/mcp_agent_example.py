#!/usr/bin/env python3
"""Example: Using MCP servers with DeepAgents.

This example shows how to connect a DeepAgent to MCP servers
to extend its capabilities with external tools.

Prerequisites:
    pip install deepagents[mcp]

Usage:
    # Start the math server in one terminal:
    python math_server.py

    # Run this example in another terminal:
    python mcp_agent_example.py
"""

import asyncio
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.mcp import MCPConfig, MCPServerConfig, FailBehavior


async def main():
    """Run an agent with MCP tools."""

    # Get the path to the math server
    math_server_path = Path(__file__).parent / "math_server.py"

    # Configure MCP servers
    mcp_config = MCPConfig(
        servers={
            # Stdio transport - runs a local process
            "math": MCPServerConfig(
                transport="stdio",
                command="python",
                args=[str(math_server_path)],
                # Optional: prefix all tools with "math_"
                tool_name_prefix=True,
                # Optional: only allow specific tools
                # allowed_tools=["add", "multiply"],
            ),
            # HTTP transport example (commented out - requires running server)
            # "api": MCPServerConfig(
            #     transport="http",
            #     url="http://localhost:8000/mcp",
            #     headers={"Authorization": "Bearer token"},
            # ),
        },
        # What to do if a server fails to connect
        fail_behavior=FailBehavior.FAIL_OPEN,  # Continue without failed servers
    )

    # Create the agent with MCP tools
    agent = create_deep_agent(
        model="claude-sonnet-4-20250514",
        mcp=mcp_config,
    )

    # Run a conversation
    config = {"configurable": {"thread_id": "mcp-example"}}

    # Ask the agent to use the math tools
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 25 + 17? And what's the square root of 144?"}]},
        config=config,
    )

    print("Agent response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

